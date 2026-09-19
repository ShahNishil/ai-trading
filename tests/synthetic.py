"""Synthetic OHLCV generator for offline, network-free backtest validation."""
import numpy as np
import pandas as pd


def random_walk_ohlcv(n: int = 600, seed: int = 0, start: float = 100.0,
                      daily_vol: float = 0.012, drift: float = 0.0) -> pd.DataFrame:
    """Geometric random walk with no predictable structure.

    A causal strategy cannot beat buy & hold on this data except by luck;
    a strategy with lookahead can, which is exactly what makes it a useful probe.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, daily_vol, n)
    close = start * np.exp(np.cumsum(rets))
    intrabar = np.abs(rng.normal(0, daily_vol * 0.6, n)) * close
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
