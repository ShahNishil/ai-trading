"""Indicator correctness guards for the two silently-dead indicators."""
import numpy as np
import pandas as pd

from core.indicators import IndicatorEngine, supertrend
from tests.synthetic import random_walk_ohlcv


def test_supertrend_is_not_all_nan():
    df = random_walk_ohlcv(400, seed=11)
    line, trend = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    assert line.notna().sum() > len(line) * 0.9, "SuperTrend line is mostly NaN"


def test_supertrend_direction_actually_flips():
    """Pre-fix the trend was pinned to +1 for the entire series."""
    df = random_walk_ohlcv(400, seed=11)
    _, trend = supertrend(df["high"], df["low"], df["close"], 10, 3.0)
    assert set(trend.unique()) == {-1, 1}, f"trend never flips: {set(trend.unique())}"


def test_donchian_breakout_is_reachable():
    """don_upper included the current bar, making close > don_upper impossible."""
    full = IndicatorEngine(random_walk_ohlcv(600, seed=4)).compute_all()
    d = full.dropna(subset=["don_upper", "don_lower"])
    assert (d["close"] > d["don_upper"]).sum() > 0, "upside breakout unreachable"
    assert (d["close"] < d["don_lower"]).sum() > 0, "downside breakdown unreachable"


def test_donchian_excludes_current_bar():
    full = IndicatorEngine(random_walk_ohlcv(200, seed=4)).compute_all()
    expected = full["high"].rolling(20, min_periods=20).max().shift(1)
    pd.testing.assert_series_equal(
        full["don_upper"].dropna(), expected.dropna(), check_names=False
    )


def test_dmn_is_scored_bearish():
    """-DI high must push the momentum score DOWN, not up."""
    from core.signals import SignalGenerator

    base = {c: np.nan for c in ["rsi", "dmp", "macd_hist", "stoch_k", "mfi", "adx",
                                "close", "ema9", "ema21", "ema50", "ema200",
                                "bb_upper", "bb_lower", "bb_middle"]}
    high_dmn = pd.DataFrame([{**base, "dmn": 30.0}])
    low_dmn = pd.DataFrame([{**base, "dmn": 10.0}])
    assert SignalGenerator(high_dmn).momentum_score() < 0
    assert SignalGenerator(low_dmn).momentum_score() > 0


def test_adx_alone_is_not_directional():
    """ADX is trend strength; a high reading must not by itself create a BUY."""
    from core.signals import SignalGenerator

    base = {c: np.nan for c in ["rsi", "dmp", "dmn", "macd_hist", "stoch_k", "mfi",
                                "close", "ema9", "ema21", "ema50", "ema200",
                                "bb_upper", "bb_lower", "bb_middle"]}
    strong = pd.DataFrame([{**base, "adx": 40.0}])
    assert SignalGenerator(strong).momentum_score() == 0.0
