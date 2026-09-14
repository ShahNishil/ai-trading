import time
from datetime import date, datetime

from agents.prompts import AUTO_TRADE_AGENT_SYSTEM_PROMPT
from core.ai_engine import AIEngine
import pandas as pd
from core.options import annualized_volatility, bs_price
from data.derivatives import DerivativeUniverse
from data.fetcher import DataFetcher
from strategies.derivatives import (
    OPTION_STRATEGIES,
    create_derivative_strategy,
)
from trading.engine import TradingEngine


class AutoTradeAgent:
    """Feature 2: AI autonomously decides and executes trades (paper/live).

    Supports two independent loops:
    - **Cash equities** (existing): LLM portfolio manager decides ENTRY/EXIT/HOLD.
    - **Derivatives (F&O)** (config-gated by ``derivatives.auto_trade.enabled``):
      a deterministic regime strategy on the underlying spot decides BUY/SELL/HOLD
      which is translated into an option structure (e.g. bull call spread, long
      straddle) or a futures position, sized in whole lots.
    """

    def __init__(
        self,
        ai_engine: AIEngine,
        data_fetcher: DataFetcher,
        trading_engine: TradingEngine,
        config: dict,
    ):
        self.ai = ai_engine
        self.data = data_fetcher
        self.engine = trading_engine
        self.config = config
        at = config.get("auto_trade", {})
        self.min_confidence = float(at.get("min_confidence", 0.70))
        self.scan_interval = int(at.get("scan_interval_seconds", 300))
        self.deriv_cfg = config.get("derivatives", {})
        self.deriv_auto = self.deriv_cfg.get("auto_trade", {}) or {}
        self.deriv_enabled = bool(self.deriv_auto.get("enabled", False))
        self._deriv_universe = None
        self._running = False

    # ------------------------------------------------------------------
    # Cash-equity loop (unchanged behaviour)
    # ------------------------------------------------------------------
    def build_context(self, symbol: str, df, watchlist_item: dict = None) -> str:
        from core.indicators import IndicatorEngine

        engine = IndicatorEngine(df)
        full = engine.compute_all()
        summary = IndicatorEngine.latest_summary(full)
        positions = self.engine.portfolio.get_open_positions()
        capital = self.engine.get_capital()
        realized = self.engine.portfolio.get_realized_pnl()
        return (
            f"SYMBOL: {symbol}\n"
            f"LATEST INDICATORS: {summary}\n\n"
            f"PORTFOLIO STATE:\n"
            f"- Mode: {self.engine.mode}\n"
            f"- Available capital: INR {capital:.0f}\n"
            f"- Realized P&L: INR {realized:.0f}\n"
            f"- Open positions: {len(positions)}\n"
            f"- Positions: {positions}\n"
            f"- Max positions: {self.engine.risk.max_positions}\n"
            f"- Risk per trade: {self.engine.risk.risk_per_trade_pct}%\n"
            f"- Min confidence: {self.min_confidence}\n"
        )

    def evaluate_symbol(self, symbol: str, security_id: str, exchange: str = "NSE_EQ") -> dict:
        try:
            df = self.data.fetch_daily(symbol, security_id, days=200, exchange=exchange)
            if df is None or df.empty:
                return {"symbol": symbol, "decision": "HOLD", "reason": "No data"}
            context = self.build_context(symbol, df)
            result = self.ai.analyze_indicators(AUTO_TRADE_AGENT_SYSTEM_PROMPT, context)
            result["symbol"] = symbol
            decision = str(result.get("decision", "HOLD")).upper()
            if decision not in ("ENTRY", "EXIT", "HOLD"):
                result["decision"] = "HOLD"
            return result
        except Exception as e:
            return {"symbol": symbol, "decision": "HOLD", "reason": str(e)}

    def execute(self, decision: dict, watchlist_item: dict = None) -> dict:
        symbol = decision.get("symbol", "")
        d = decision.get("decision", "HOLD").upper()
        security_id = (watchlist_item or {}).get("security_id", "")
        exchange = (watchlist_item or {}).get("exchange", "NSE_EQ")
        confidence = float(decision.get("confidence", 0) or 0)
        price = float(decision.get("price", 0) or 0)
        qty = int(decision.get("quantity", 0) or 0)
        side = decision.get("side", "BUY").upper()

        if d == "ENTRY":
            if confidence < self.min_confidence:
                return {"success": False, "reason": f"Confidence {confidence:.2f} below min {self.min_confidence}"}
            capital = self.engine.get_capital()
            if qty <= 0:
                qty = self.engine.risk.position_quantity(capital, price or 100)
            if qty <= 0:
                return {"success": False, "reason": "Invalid quantity"}
            return self.engine.enter_position(
                symbol=symbol,
                quantity=qty,
                entry_price=price,
                security_id=security_id,
                side=side,
                order_type=decision.get("order_type", "MARKET"),
                reason=str(decision.get("reason", "")),
            )
        elif d == "EXIT":
            pos = self.engine.portfolio.get_open_position(symbol)
            if pos:
                return self.engine.close_position(
                    trade_id=pos["trade_id"],
                    exit_price=price,
                    security_id=security_id,
                    side=side,
                    reason=str(decision.get("reason", "")),
                    quantity=int(pos["quantity"]),
                    symbol=symbol,
                )
            return {"success": False, "reason": f"No open position for {symbol}"}
        return {"success": True, "skipped": True, "reason": "HOLD decision"}

    # ------------------------------------------------------------------
    # Derivatives (F&O) helpers
    # ------------------------------------------------------------------
    def _get_universe(self) -> DerivativeUniverse:
        if self._deriv_universe is None:
            self._deriv_universe = DerivativeUniverse(config=self.config)
        return self._deriv_universe

    def _spot_series(self, underlying: str, kind: str):
        universe = self._get_universe()
        sym = universe.spot_symbol(underlying.upper(), kind)
        if not sym or self.data is None:
            return None, sym
        try:
            df = self.data.fetch_daily(sym, "0", days=400)
        except Exception:
            df = None
        return (df if df is not None and not df.empty else None), sym

    def _open_deriv_positions(self, underlying: str, kind: str) -> list:
        underlying = underlying.upper()
        out = []
        for pos in self.engine.portfolio.get_open_positions():
            seg = str(pos.get("exchange_segment", "")).upper()
            sym = str(pos.get("symbol", "")).upper()
            if seg == "NSE_FNO" and (sym == underlying or sym.startswith(underlying + " ")):
                out.append(pos)
        return out

    @staticmethod
    def _is_option_structure(pos: dict) -> bool:
        return pos.get("instrument_type", "") in OPTION_STRATEGIES

    def _structure_value(self, underlying: str, kind: str, spot: float, structure: str, expiry, sigma: float, risk_free: float) -> float:
        universe = self._get_universe()
        step = universe.strike_step(underlying, kind)
        t_days = max((pd.Timestamp(expiry).date() - date.today()).days, 1) if expiry else 7
        T = t_days / 365.0
        legs = create_derivative_strategy(structure, {}).build_legs(spot, step, {}).legs
        total = 0.0
        for leg in legs:
            prem = float(bs_price(spot, leg.strike, T, risk_free, sigma, leg.option_type))
            total += (1.0 if leg.side == "BUY" else -1.0) * prem
        return total

    def _close_position(self, pos: dict, exit_price: float, reason: str) -> dict:
        return self.engine.close_position(
            trade_id=pos["trade_id"],
            exit_price=exit_price,
            symbol=str(pos["symbol"]),
            side="SELL",
            reason=reason,
            quantity=int(pos["quantity"]),
            exchange_segment="NSE_FNO",
            instrument_type=str(pos.get("instrument_type", "FUTIDX")),
        )

    def evaluate_derivative(self, underlying: str, kind: str, strategy_name: str, mode: str, params: dict = None) -> dict:
        underlying = underlying.upper()
        df, sym = self._spot_series(underlying, kind)
        out = {
            "symbol": f"{underlying} ({mode})",
            "underlying": underlying,
            "kind": kind,
            "strategy": strategy_name,
            "mode": mode,
            "decision": "HOLD",
            "action": "HOLD",
            "confidence": 0.0,
            "structure": "",
            "reason": "",
            "spot_symbol": sym,
        }
        if df is None:
            out["reason"] = "No spot data"
            return out

        from core.indicators import IndicatorEngine

        enriched = IndicatorEngine(df).compute_all()
        try:
            strat = create_derivative_strategy(strategy_name, params)
        except KeyError:
            strat = create_derivative_strategy("regime_switch", params)
        sig = strat.generate_signal(enriched)
        action = str(sig.get("action", "HOLD")).upper()
        out["action"] = action
        out["confidence"] = float(sig.get("confidence", 0) or 0)
        out["structure"] = str(sig.get("structure") or "")
        out["reason"] = str(sig.get("reason", ""))
        out["decision"] = "ENTRY" if action in ("BUY", "SELL") else "HOLD"
        out["side"] = action
        out["price"] = round(float(enriched["close"].iloc[-1]), 2)
        out["spot_df"] = df
        return out

    def execute_derivative(self, decision: dict) -> dict:
        action = decision.get("action", "HOLD")
        underlying = decision.get("underlying", "")
        kind = decision.get("kind", "IDX")
        mode = decision.get("mode", "options")
        spot_df = decision.get("spot_df")
        strategy_name = decision.get("strategy", "regime_switch")
        confidence = float(decision.get("confidence", 0) or 0)

        if not underlying or spot_df is None:
            return {"success": False, "reason": "No underlying/data"}
        if action not in ("BUY", "SELL"):
            closed = []
            for pos in self._open_deriv_positions(underlying, kind):
                price = 0.0
                if self._is_option_structure(pos):
                    price = self._structure_value(underlying, kind, float(spot_df["close"].iloc[-1]),
                                                  pos["instrument_type"], self._get_universe().nearest_expiry(underlying, kind, "weekly"),
                                                  self._sigma(spot_df), self._risk_free())
                else:
                    price = float(spot_df["close"].iloc[-1])
                closed.append(self._close_position(pos, price, reason=f"regime_{action.lower()}"))
            return {"success": True, "closed": closed, "reason": f"HOLD — squared off existing F&O positions"}

        min_conf = float(self.deriv_auto.get("min_confidence", 0.6) or 0.6)
        if confidence < min_conf:
            return {"success": False, "reason": f"Confidence {confidence:.2f} below min {min_conf}"}

        futures = mode in ("futures", "both")
        options = mode in ("options", "both")

        # Close any contradictory existing positions first.
        for pos in self._open_deriv_positions(underlying, kind):
            if self._is_option_structure(pos) and options:
                price = self._structure_value(underlying, kind, float(spot_df["close"].iloc[-1]),
                                              pos["instrument_type"], self._get_universe().nearest_expiry(underlying, kind, "weekly"),
                                              self._sigma(spot_df), self._risk_free())
                self._close_position(pos, price, reason=f"roll_to_{action.lower()}")
            elif not self._is_option_structure(pos) and futures:
                pos_side = str(pos.get("side", "BUY")).upper()
                if (action == "BUY" and pos_side != "BUY") or (action == "SELL" and pos_side != "SELL"):
                    self._close_position(pos, float(spot_df["close"].iloc[-1]), reason=f"flip_to_{action.lower()}")

        results = []
        if options:
            results.append(self._enter_option_structure(underlying, kind, action, decision.get("structure", ""), spot_df))
        if futures:
            results.append(self._enter_futures(underlying, kind, action, spot_df))
        return {"success": bool(results), "results": results}

    def _sigma(self, spot_df) -> float:
        vol = annualized_volatility(spot_df["close"].values) / 100.0
        if not vol or vol <= 0.02:
            return 0.12
        return float(vol)

    def _risk_free(self) -> float:
        return float(self.deriv_cfg.get("risk_free_rate_pct", 7.0)) / 100.0

    def _enter_option_structure(self, underlying: str, kind: str, action: str, structure_hint: str, spot_df) -> dict:
        universe = self._get_universe()
        structure = structure_hint or self.deriv_cfg.get("default_strategy", "long_straddle")
        if structure not in OPTION_STRATEGIES:
            return {"success": False, "reason": f"Unknown option structure: {structure}"}

        step = universe.strike_step(underlying, kind)
        lot = universe.lot_size(underlying, kind)
        spot = float(spot_df["close"].iloc[-1])
        expiry = universe.nearest_expiry(underlying, kind, mode="weekly")
        T = max((pd.Timestamp(expiry).date() - date.today()).days, 1) / 365.0
        sigma = self._sigma(spot_df)
        r = self._risk_free()
        instrument_type = "OPTIDX" if kind == "IDX" else "OPTSTK"

        legs = []
        premiums = {}
        for i, leg in enumerate(create_derivative_strategy(structure, {}).build_legs(spot, step, {}).legs):
            prem = float(bs_price(spot, leg.strike, T, r, sigma, leg.option_type))
            security_id = ""
            try:
                c = universe.option_contract(underlying, kind, leg.strike, leg.option_type, expiry=expiry)
                security_id = c.security_id
            except Exception:
                pass
            legs.append({
                "option_type": leg.option_type,
                "strike": float(leg.strike),
                "side": leg.side,
                "security_id": security_id,
                "instrument_type": instrument_type,
                "entry_premium": prem,
                "trading_symbol": "",
            })
            premiums[i] = prem

        net_entry = sum((1.0 if l["side"] == "BUY" else -1.0) * premiums[i] for i, l in enumerate(legs))
        capital = self.engine.get_capital()
        num_lots = self.engine.risk.option_lots_quantity(
            capital, net_entry, lot,
            max_lots=int(self.deriv_cfg.get("max_lots_per_trade", 5)),
        )
        return self.engine.enter_option_structure(
            underlying=underlying,
            strategy_name=structure,
            legs=legs,
            num_lots=num_lots,
            lot_size=lot,
            premiums=premiums,
            reason=f"auto_{action}_regime conv={sigma:.2%}",
        )

    def _enter_futures(self, underlying: str, kind: str, action: str, spot_df) -> dict:
        universe = self._get_universe()
        inst = universe.futures_contract(underlying, kind)
        spot = float(spot_df["close"].iloc[-1])
        lot = int(inst.lot_size)
        margin_pct = float(self.deriv_cfg.get("futures_margin_pct", 12.0))
        usage_pct = float(self.deriv_cfg.get("futures_margin_usage_pct", 60.0))

        qty = self.engine.risk.futures_margin_quantity(
            self.engine.get_capital(), spot, lot, margin_pct, usage_pct
        )
        if qty <= 0:
            return {"success": False, "reason": "Futures qty <= 0"}
        return self.engine.enter_futures_position(
            underlying=underlying,
            quantity=qty,
            entry_price=spot,
            security_id=inst.security_id,
            side=action,
            reason=f"auto_{action}_regime",
            instrument_type=inst.instrument_type,
        )

    # ------------------------------------------------------------------
    # Continuous scanning loop
    # ------------------------------------------------------------------
    def scan_once(self, watchlist: list) -> list:
        results = []
        for item in watchlist:
            symbol = item.get("symbol", "")
            security_id = item.get("security_id", "")
            if not symbol or not security_id:
                continue
            decision = self.evaluate_symbol(symbol, security_id, item.get("exchange", "NSE_EQ"))
            execution = self.execute(decision, watchlist_item=item)
            decision["execution"] = execution
            decision["timestamp"] = datetime.now().isoformat()
            results.append(decision)
        if self.deriv_enabled:
            results.extend(self.scan_derivatives_once())
        return results

    def scan_derivatives_once(self) -> list:
        results = []
        strategy_name = self.deriv_auto.get("strategy", "regime_switch")
        mode = str(self.deriv_auto.get("mode", "options")).lower()
        kind = str(self.deriv_auto.get("market", "IDX")).upper()
        underlyings = self.deriv_auto.get("underlyings") or ["NIFTY"]
        params = self.deriv_auto.get("params") or {}
        for u in underlyings:
            decision = self.evaluate_derivative(str(u).upper(), kind, strategy_name, mode, params)
            execution = self.execute_derivative(decision)
            decision["execution"] = execution
            decision["timestamp"] = datetime.now().isoformat()
            decision.pop("spot_df", None)  # don't keep DataFrames in shared UI state
            results.append(decision)
        return results

    def run_forever(self, watchlist: list, stop_event=None):
        """Main auto-trading loop. Runs until stop_event is set."""
        self._running = True
        from ui_state import AutoTradeState  # local import to avoid circular deps

        while self._running:
            if stop_event is not None and stop_event.is_set():
                self._running = False
                break
            try:
                results = self.scan_once(watchlist)
                AutoTradeState.last_scan = results
                AutoTradeState.last_scan_time = datetime.now().isoformat()
            except Exception as e:
                AutoTradeState.last_error = str(e)
            time.sleep(self.scan_interval)

    def stop(self):
        self._running = False