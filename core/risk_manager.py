from datetime import date, datetime


class RiskManager:
    """Position sizing and portfolio risk guardrails for auto-trading."""

    def __init__(self, config: dict, portfolio=None):
        at = config.get("auto_trade", {})
        self.max_positions = int(at.get("max_positions", 5))
        self.risk_per_trade_pct = float(at.get("risk_per_trade_pct", 2.0))
        self.max_daily_loss_pct = float(at.get("max_daily_loss_pct", 5.0))
        self.stop_loss_pct = float(at.get("stop_loss_pct", 2.0))
        self.target_pct = float(at.get("target_pct", 4.0))
        self.trailing_stop_pct = float(at.get("trailing_stop_pct", 1.5))
        self.portfolio = portfolio
        self._daily_limits = {}

    def can_open_position(self, capital: float) -> tuple:
        """Check if a new position can be opened. Returns (allowed: bool, reason: str)."""
        if self.portfolio is None:
            return True, ""

        open_positions = self.portfolio.get_open_positions()
        if len(open_positions) >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Daily loss guard
        today = date.today().isoformat()
        day_pnl = self.portfolio.get_pnl_between(today, today)
        capital_pct = (abs(day_pnl) / capital * 100) if capital > 0 else 0
        if self.max_daily_loss_pct > 0 and day_pnl < 0 and capital_pct >= self.max_daily_loss_pct:
            return False, f"Daily loss limit hit ({self.max_daily_loss_pct}%)"

        return True, ""

    def position_quantity(self, capital: float, entry_price: float, risk_per_trade_pct: float = None) -> int:
        """Calculate quantity based on percentage risk per trade."""
        risk_pct = risk_per_trade_pct or self.risk_per_trade_pct
        if entry_price <= 0:
            return 0
        risk_amount = capital * (risk_pct / 100.0)
        stop_distance = entry_price * (self.stop_loss_pct / 100.0)
        if stop_distance <= 0:
            return 0
        qty = int(risk_amount / stop_distance)
        # Cap so that notional is not absurdly larger than capital
        max_qty = int(capital / entry_price)
        return max(1, min(qty, max_qty))

    def compute_stop_loss(self, entry_price: float, entry_date, volatility_factor: float = 1.0) -> float:
        return round(entry_price * (1 - self.stop_loss_pct * volatility_factor / 100.0), 2)

    def compute_target(self, entry_price: float) -> float:
        return round(entry_price * (1 + self.target_pct / 100.0), 2)

    def trailing_stop(self, entry_price: float, current_price: float, side: str = "LONG") -> float:
        if side.upper() == "LONG":
            base = current_price * (1 - self.trailing_stop_pct / 100.0)
            initial_stop = entry_price * (1 - self.stop_loss_pct / 100.0)
            return max(initial_stop, base)
        base = current_price * (1 + self.trailing_stop_pct / 100.0)
        initial_stop = entry_price * (1 + self.stop_loss_pct / 100.0)
        return min(initial_stop, base)