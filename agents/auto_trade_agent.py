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