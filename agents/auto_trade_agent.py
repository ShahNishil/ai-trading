import time
from datetime import datetime

from agents.prompts import AUTO_TRADE_AGENT_SYSTEM_PROMPT
from core.ai_engine import AIEngine
from data.fetcher import DataFetcher
from trading.engine import TradingEngine


class AutoTradeAgent:
    """Feature 2: AI autonomously decides and executes trades (paper/live)."""

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
        self._running = False

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

    def current_price(self, symbol: str, security_id: str, exchange: str = "NSE_EQ"):
        """Best available real price: live quote, else last cached close.

        Execution must never rely on a price the LLM wrote. The model is asked
        for one, but it can return 0 or a stale/hallucinated number; an order
        placed at 0 fills at 0 in paper mode and books an entry price of zero.
        """
        try:
            ltp = self.data.fetch_quote(security_id, exchange)
            if ltp:
                return float(ltp)
        except Exception:
            pass
        try:
            df = self.data.fetch_daily(symbol, security_id, days=30, exchange=exchange)
            if df is not None and not df.empty:
                return float(df.iloc[-1]["close"])
        except Exception:
            pass
        return None

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
        llm_price = float(decision.get("price", 0) or 0)
        llm_qty = int(decision.get("quantity", 0) or 0)
        side = decision.get("side", "BUY").upper()

        # Resolve the actual market price; the LLM's number is advisory at best.
        price = self.current_price(symbol, security_id, exchange)
        if price is None or price <= 0:
            if d in ("ENTRY", "EXIT"):
                return {"success": False, "reason": f"No market price available for {symbol}"}
            price = 0.0
        elif llm_price > 0 and abs(llm_price - price) / price > 0.05:
            decision["price_note"] = (
                f"LLM price {llm_price:.2f} deviated >5% from market {price:.2f}; used market"
            )

        if d == "ENTRY":
            if confidence < self.min_confidence:
                return {"success": False, "reason": f"Confidence {confidence:.2f} below min {self.min_confidence}"}
            capital = self.engine.get_capital()
            # Risk-based sizing from the real price. The LLM's quantity can only
            # shrink the position, never exceed the risk budget.
            qty = self.engine.risk.position_quantity(capital, price)
            if llm_qty > 0:
                qty = min(qty, llm_qty)
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
    # Continuous scanning loop
    # ------------------------------------------------------------------
    def manage_open_positions(self, watchlist: list) -> list:
        """Enforce stops/targets on open positions with fresh prices.

        Without this call the stop machinery in TradingEngine never runs in
        production: positions opened by the agent had no exit other than the
        LLM later returning EXIT for that symbol.
        """
        positions = self.engine.portfolio.get_open_positions()
        if not positions:
            return []
        by_symbol = {str(i.get("symbol", "")).upper(): i for i in watchlist}
        prices, sec_ids = {}, {}
        for pos in positions:
            sym = str(pos.get("symbol", "")).upper()
            item = by_symbol.get(sym, {})
            px = self.current_price(sym, item.get("security_id", ""), item.get("exchange", "NSE_EQ"))
            if px:
                prices[sym] = px
                if item.get("security_id"):
                    sec_ids[sym] = item["security_id"]
        if not prices:
            return []
        return self.engine.update_positions_with_prices(prices, security_ids=sec_ids)

    def scan_once(self, watchlist: list) -> list:
        results = []
        # Risk exits come FIRST each cycle so a breached stop is acted on before
        # any new capital is committed.
        for action in self.manage_open_positions(watchlist):
            results.append(
                {
                    "symbol": action.get("symbol", ""),
                    "decision": "RISK_EXIT",
                    "reason": action.get("reason", ""),
                    "execution": action.get("result", {}),
                    "timestamp": datetime.now().isoformat(),
                }
            )
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