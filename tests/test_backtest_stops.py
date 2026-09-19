"""The backtest must apply the same stop/target/trailing exits the live engine does."""
import pandas as pd

from backtest.engine import BacktestEngine
from strategies.base import BaseStrategy


class _BuyOnce(BaseStrategy):
    name = "buy_once"

    def generate_signal(self, df):
        if len(df) == 3:
            return {"action": "BUY", "confidence": 0.9, "reason": "test entry"}
        return {"action": "HOLD", "confidence": 0.0, "reason": ""}


def _frame(bars):
    """bars: list of (open, high, low, close). Flat volume, business-day index."""
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="B")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c, "volume": 1000} for o, h, l, c in bars],
        index=idx,
    )


def _run(df, **risk):
    eng = BacktestEngine(commission_pct=0.0, slippage_pct=0.0, **risk)
    res = eng.run(df, _BuyOnce(), "SYN", skip_initial_n=0)
    return res["trades"]


FLAT = (100, 100.5, 99.5, 100)


def test_stop_loss_fills_intrabar_at_the_level():
    bars = [FLAT] * 5 + [(100, 100.0, 97.0, 99.0)] + [FLAT] * 3
    trades = _run(_frame(bars), stop_loss_pct=2.0)
    assert trades, "no trade closed"
    t = trades[0]
    assert t.exit_reason == "stop_loss"
    assert abs(t.exit_price - 98.0) < 1e-9, f"filled at {t.exit_price}, not at the 98.0 stop"


def test_target_fills_at_the_level_not_the_close():
    bars = [FLAT] * 5 + [(100, 105.0, 99.8, 100.2)] + [FLAT] * 3
    trades = _run(_frame(bars), target_pct=4.0)
    t = trades[0]
    assert t.exit_reason == "target_hit"
    assert abs(t.exit_price - 104.0) < 1e-9


def test_stop_first_when_bar_spans_both():
    """One bar touches stop AND target: the adverse fill must be assumed."""
    bars = [FLAT] * 5 + [(100, 106.0, 97.0, 103.0)] + [FLAT] * 3
    trades = _run(_frame(bars), stop_loss_pct=2.0, target_pct=4.0)
    t = trades[0]
    assert t.exit_reason == "stop_loss", f"optimistic fill: {t.exit_reason}"


def test_trailing_stop_ratchets_in_backtest():
    bars = [FLAT, FLAT, FLAT,
            (100, 110, 100, 110),
            (110, 120, 110, 120),
            (120, 120, 117.0, 118.0),  # 120 peak -> trail 117.6; low 117 pierces it
            FLAT, FLAT]
    trades = _run(_frame(bars), trailing_stop_pct=2.0)
    t = trades[0]
    assert t.exit_reason == "trailing_stop"
    assert abs(t.exit_price - 117.6) < 1e-9, f"exit {t.exit_price} != trail level 117.6"


def test_no_risk_exits_when_disabled():
    """All parameters 0 -> old behaviour, position rides to end of data."""
    bars = [FLAT] * 5 + [(100, 100.0, 90.0, 95.0)] + [FLAT] * 3
    trades = _run(_frame(bars))
    assert trades[0].exit_reason == "end of backtest"
