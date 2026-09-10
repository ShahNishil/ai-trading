import uuid
from datetime import datetime

from data.cache import DataCache


class Portfolio:
    """Tracks positions, P&L and capital using the local cache."""

    def __init__(self, cache: DataCache, mode: str = "paper", initial_capital: float = 100000):
        self.cache = cache
        self.mode = mode
        self.initial_capital = initial_capital

    def get_open_positions(self) -> list:
        return self.cache.get_open_positions(self.mode)

    def get_open_position(self, symbol: str) -> dict:
        symbol = symbol.upper()
        for pos in self.get_open_positions():
            if pos["symbol"].upper() == symbol:
                return pos
        return None

    def open_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        strategy: str = "auto_ai",
        entry_reason: str = "",
    ) -> str:
        trade_id = str(uuid.uuid4())[:12]
        self.cache.save_trade(
            {
                "trade_id": trade_id,
                "symbol": symbol.upper(),
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": 0,
                "entry_time": datetime.now().isoformat(),
                "exit_time": "",
                "pnl": 0,
                "strategy": strategy,
                "mode": self.mode,
                "entry_reason": entry_reason,
                "exit_reason": "",
            }
        )
        return trade_id

    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str = "") -> dict:
        trades = self.cache.get_trades(limit=10000)
        for t in trades:
            if t["trade_id"] == trade_id:
                qty = int(t["quantity"])
                entry = float(t["entry_price"])
                pnl = (exit_price - entry) * qty
                cur = self.cache._conn.cursor()
                cur.execute(
                    """
                    UPDATE trades SET exit_price=?, exit_time=?, pnl=?, exit_reason=?, mode=?
                    WHERE trade_id=?
                    """,
                    (exit_price, datetime.now().isoformat(), pnl, exit_reason, self.mode, trade_id),
                )
                self.cache._conn.commit()
                return {"trade_id": trade_id, "pnl": pnl, "entry": entry, "exit": exit_price}
        return None

    def get_pnl_between(self, start_date: str, end_date: str) -> float:
        """Sum of realized P&L between two ISO dates."""
        trades = self.cache.get_trades(limit=100000)
        total = 0.0
        for t in trades:
            if t.get("mode", "paper") != self.mode:
                continue
            exit_time = t.get("exit_time", "")
            if not exit_time:
                continue
            day = exit_time[:10]
            if start_date <= day <= end_date:
                total += float(t.get("pnl", 0) or 0)
        return total

    def get_realized_pnl(self) -> float:
        trades = self.cache.get_trades(limit=100000)
        return round(sum(float(t.get("pnl", 0) or 0) for t in trades if t.get("mode") == self.mode and t.get("exit_time")), 2)

    def get_unrealized_pnl(self, current_prices: dict = None) -> float:
        current_prices = current_prices or {}
        total = 0.0
        for pos in self.get_open_positions():
            price = current_prices.get(pos["symbol"].upper()) or float(pos["entry_price"])
            total += (price - float(pos["entry_price"])) * int(pos["quantity"])
        return round(total, 2)

    def equity(self, current_prices: dict = None) -> float:
        return self.initial_capital + self.get_realized_pnl() + self.get_unrealized_pnl(current_prices)