from datetime import datetime

from core.dhan_client import DhanClient
from core.risk_manager import RiskManager
from data.cache import DataCache
from trading.order_manager import OrderManager
from trading.portfolio import Portfolio


class TradingEngine:
    """Unified engine for both paper and live trading."""

    def __init__(
        self,
        config: dict,
        dhan: DhanClient = None,
        cache: DataCache = None,
        mode: str = None,
    ):
        self.config = config
        self.mode = mode or config.get("dhan", {}).get("trading_mode", "paper")
        self.cache = cache or DataCache()
        self.dhan = dhan
        self.orders = OrderManager(self.cache, dhan=dhan, mode=self.mode)
        self.portfolio = Portfolio(self.cache, mode=self.mode)
        self.risk = RiskManager(config, portfolio=self.portfolio)
        # Best price reached since entry, per trade_id. A trailing stop has to
        # ratchet off the peak; recomputing it from the *current* price can never
        # move the stop up. In-memory only, so it reseeds after a restart.
        self._peak_price: dict = {}

    # ------------------------------------------------------------------
    # Capital / account
    # ------------------------------------------------------------------
    def get_capital(self) -> float:
        if self.mode == "live" and self.dhan is not None:
            try:
                fund = self.dhan.get_fund_limit()
                data = fund.get("data", {}) if isinstance(fund.get("data"), dict) else {}
                if isinstance(data, dict):
                    return float(data.get("availableBalance", 0) or 0)
            except Exception:
                pass
        return self.portfolio.equity()

    def get_account_summary(self) -> dict:
        summary = {
            "mode": self.mode,
            "realized_pnl": self.portfolio.get_realized_pnl(),
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(),
            "open_positions": len(self.portfolio.get_open_positions()),
            "capital": self.get_capital(),
        }
        if self.mode == "live":
            summary["dhan_fund"] = self.dhan.get_fund_limit() if self.dhan else {}
        return summary

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------
    def enter_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        security_id: str = "",
        side: str = "BUY",
        strategy: str = "auto_ai",
        order_type: str = "MARKET",
        reason: str = "",
    ) -> dict:
        allowed, block_reason = self.risk.can_open_position(self.get_capital())
        if not allowed:
            return {"success": False, "error": block_reason}

        order = self.orders.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=entry_price,
            order_type=order_type,
            product_type=self.config.get("auto_trade", {}).get("product_type", "INTR"),
            strategy=strategy,
            security_id=security_id,
        )
        if order.get("status") == "ERROR":
            return {"success": False, "error": order.get("error", "Order failed")}

        filled_price = order.get("filled_price") or entry_price
        trade_id = self.portfolio.open_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=filled_price,
            strategy=strategy,
            entry_reason=reason,
        )
        return {
            "success": True,
            "order": order,
            "trade_id": trade_id,
            "filled_price": filled_price,
        }

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        security_id: str = "",
        side: str = "SELL",
        reason: str = "manual_exit",
        quantity: int = 0,
        symbol: str = "",
    ) -> dict:
        order = self.orders.place_order(
            symbol=symbol or "",
            side=side,
            quantity=quantity,
            price=exit_price,
            order_type="MARKET",
            product_type=self.config.get("auto_trade", {}).get("product_type", "INTR"),
            strategy="exit",
            security_id=security_id,
        )
        if order.get("status") == "ERROR":
            # Never mark a position closed in the book when the exit order was
            # rejected - that silently desynchronises the book from the broker
            # and leaves real exposure that the risk manager can no longer see.
            return {"success": False, "error": order.get("error", "Exit order failed"), "order": order}

        result = self.portfolio.close_trade(trade_id, exit_price or order.get("filled_price") or 0, reason)
        if result is None:
            return {"success": False, "error": "Trade not found"}
        self._peak_price.pop(trade_id, None)
        result["success"] = True
        result["order"] = order
        return result

    def update_positions_with_prices(self, current_prices: dict, security_ids: dict = None) -> list:
        """Apply trailing stops and targets based on current prices.

        Three defects previously lived here:
          1. the trailing stop was computed from the CURRENT price, so it never
             ratcheted and behaved as a fixed stop at the initial level;
          2. SHORT positions were skipped entirely - they had no stop and no target;
          3. exits called ``portfolio.close_trade`` directly, updating the book
             without ever sending an order, so in live mode the broker position
             stayed open while the book showed it closed.
        """
        actions = []
        security_ids = security_ids or {}
        for pos in self.portfolio.get_open_positions():
            symbol = str(pos.get("symbol", "")).upper()
            price = current_prices.get(symbol)
            if not price:
                continue
            price = float(price)
            entry = float(pos["entry_price"])
            trade_id = pos["trade_id"]
            qty = int(pos["quantity"])
            is_long = str(pos.get("side", "BUY")).upper() in ("BUY", "LONG")

            # Ratchet the high-water mark in the favourable direction only.
            peak = self._peak_price.get(trade_id)
            if peak is None:
                peak = entry
            peak = max(peak, price) if is_long else min(peak, price)
            self._peak_price[trade_id] = peak

            stop = self.risk.trailing_stop(entry, peak, side="LONG" if is_long else "SHORT")
            target = (
                self.risk.compute_target(entry)
                if is_long
                else round(entry * (1 - self.risk.target_pct / 100.0), 2)
            )

            hit_stop = price <= stop if is_long else price >= stop
            hit_target = price >= target if is_long else price <= target

            # Stop takes precedence: if both are touched between two observed
            # prices we cannot tell which came first, so assume the adverse one.
            if hit_stop:
                reason = "trailing_stop"
            elif hit_target:
                reason = "target_hit"
            else:
                continue

            # The trades table stores no security_id, so the caller must supply
            # it (from the watchlist) or a live exit would silently skip the
            # broker call and fill only on paper.
            result = self.close_position(
                trade_id=trade_id,
                exit_price=price,
                security_id=str(security_ids.get(symbol) or pos.get("security_id", "")),
                side="SELL" if is_long else "BUY",
                reason=reason,
                quantity=qty,
                symbol=symbol,
            )
            actions.append({"symbol": symbol, "trade_id": trade_id, "reason": reason,
                            "price": price, "stop": stop, "target": target, "result": result})
        return actions