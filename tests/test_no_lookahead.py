"""Guards against lookahead bias in the backtest engine.

The original engine trimmed the frame and then sliced `df[: idx + skip_initial_n + 1]`,
so the strategy's "latest" bar sat `skip_initial_n` bars in the future of the bar
being traded. These tests fail loudly if that ever comes back.
"""
import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine, run_backtest
from core.indicators import IndicatorEngine
from strategies.base import BaseStrategy
from strategies.registry import create_strategy
from tests.synthetic import random_walk_ohlcv


class _WindowSpy(BaseStrategy):
    """Records the last timestamp each window exposed to the strategy."""

    name = "window_spy"

    def __init__(self, params=None):
        super().__init__(params)
        self.seen = []

    def generate_signal(self, df):
        self.seen.append(df.index[-1])
        return {"action": "HOLD", "confidence": 0.0, "reason": "spy"}


def test_window_never_extends_past_traded_bar():
    df = IndicatorEngine(random_walk_ohlcv(400, seed=7)).compute_all()
    spy = _WindowSpy()
    engine = BacktestEngine()
    engine.run(df, spy, "SYN", skip_initial_n=60)

    traded = [e["timestamp"] for e in engine.equity_curve]
    assert len(traded) == len(spy.seen)
    for bar_ts, window_last in zip(traded, spy.seen):
        assert window_last <= bar_ts, (
            f"LOOKAHEAD: trading bar {bar_ts} saw data up to {window_last}"
        )
    # And the window must actually END on the traded bar, not merely precede it.
    assert spy.seen == traded


def test_warmup_is_preserved():
    """The fix must not starve the strategy: the first window needs the warm-up."""
    df = IndicatorEngine(random_walk_ohlcv(400, seed=7)).compute_all()
    spy = _WindowSpy()
    BacktestEngine().run(df, spy, "SYN", skip_initial_n=60)
    assert spy.seen[0] == df.index[60]


def test_pnl_is_net_of_commission():
    df = random_walk_ohlcv(500, seed=3)
    res = run_backtest(
        create_strategy("momentum"), df, "SYN", commission_pct=0.5, slippage_pct=0.0
    )
    closed = [t for t in res["trades"] if t.exit_price is not None]
    if not closed:
        return
    for t in closed:
        assert t.commission > 0
        assert abs(t.pnl - (t.gross_pnl - t.commission)) < 1e-6


def test_intraday_annualization_differs_from_daily():
    from backtest.engine import infer_periods_per_year

    daily = pd.date_range("2024-01-01", periods=100, freq="B")
    five_min = pd.date_range("2024-01-01 09:15", periods=100, freq="5min")
    assert infer_periods_per_year(daily) == 252.0
    assert infer_periods_per_year(five_min) > 252.0 * 50
