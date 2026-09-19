"""Walk-forward validation: optimize on a training window, evaluate on the
unseen window that follows, roll forward, repeat.

Grid-searching one dataset and reporting the best in-sample Sharpe (what
``strategy_optimize`` does) is overfitting by construction: with enough
combinations, something always looks good on the data it was fitted to. The
only defensible performance estimate is the stitched OUT-OF-SAMPLE record --
each fold traded with parameters chosen before that data was seen.
"""
import itertools
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from strategies.registry import create_strategy

# Bars of history prepended to each test slice so indicators (EMA200 needs the
# most) are warm before the first tradeable bar. Must match run_backtest's
# skip_initial_n so warm-up bars are never traded.
WARMUP_BARS = 60


def _grid(param_grid: dict) -> list:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


def _score(metrics: dict) -> float:
    """Rank in-sample candidates. Sharpe first, drawdown as tie-breaker."""
    if not metrics or "error" in metrics:
        return -np.inf
    sharpe = float(metrics.get("sharpe", 0) or 0)
    max_dd = abs(float(metrics.get("max_drawdown_pct", 0) or 0))
    return sharpe - 0.01 * max_dd


def walk_forward(
    strategy_name: str,
    df: pd.DataFrame,
    param_grid: Optional[dict] = None,
    train_bars: int = 252,
    test_bars: int = 63,
    initial_capital: float = 100000,
    commission_pct: float = 0.03,
    slippage_pct: float = 0.05,
    stop_loss_pct: float = 0.0,
    target_pct: float = 0.0,
    trailing_stop_pct: float = 0.0,
) -> dict:
    """Run rolling walk-forward validation over a raw OHLCV frame.

    Returns {folds, oos_metrics, param_stability, warning?}. ``oos_metrics``
    aggregates ONLY out-of-sample fold results; in-sample scores are reported
    per fold so the in/out gap (the overfitting tax) is visible.
    """
    if df is None or len(df) < train_bars + test_bars + WARMUP_BARS:
        return {
            "error": (
                f"Need at least {train_bars + test_bars + WARMUP_BARS} bars "
                f"({train_bars} train + {test_bars} test + {WARMUP_BARS} warm-up), got {0 if df is None else len(df)}"
            )
        }

    combos = _grid(param_grid or {})
    folds = []
    start = 0
    while start + train_bars + test_bars <= len(df):
        train = df.iloc[start : start + train_bars]
        # Test slice keeps WARMUP_BARS of context; those bars are skipped, not traded.
        test_lo = max(0, start + train_bars - WARMUP_BARS)
        test = df.iloc[test_lo : start + train_bars + test_bars]

        best_params, best_score, best_train_metrics = None, -np.inf, {}
        for params in combos:
            res = run_backtest(
                create_strategy(strategy_name, params or None), train,
                initial_capital=initial_capital,
                commission_pct=commission_pct, slippage_pct=slippage_pct,
                skip_initial_n=WARMUP_BARS,
                stop_loss_pct=stop_loss_pct, target_pct=target_pct,
                trailing_stop_pct=trailing_stop_pct,
            )
            score = _score(res.get("metrics", {}))
            if score > best_score:
                best_score, best_params, best_train_metrics = score, params, res.get("metrics", {})

        oos = run_backtest(
            create_strategy(strategy_name, best_params or None), test,
            initial_capital=initial_capital,
            commission_pct=commission_pct, slippage_pct=slippage_pct,
            skip_initial_n=WARMUP_BARS,
            stop_loss_pct=stop_loss_pct, target_pct=target_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        folds.append({
            "train_start": str(train.index[0]),
            "train_end": str(train.index[-1]),
            "test_start": str(df.index[start + train_bars]),
            "test_end": str(test.index[-1]),
            "chosen_params": best_params or {},
            "train_metrics": best_train_metrics,
            "test_metrics": oos.get("metrics", {}),
        })
        start += test_bars

    if not folds:
        return {"error": "No complete folds"}

    oos_rets = [f["test_metrics"].get("total_return_pct", 0) for f in folds]
    is_rets = [f["train_metrics"].get("total_return_pct", 0) for f in folds]
    oos_sharpes = [f["test_metrics"].get("sharpe", 0) for f in folds]
    trades = int(sum(f["test_metrics"].get("num_trades", 0) for f in folds))

    # Compounded OOS return across folds (each fold restarts at initial_capital,
    # so compounding the per-fold returns approximates one continuous account).
    compounded = float(np.prod([1 + r / 100.0 for r in oos_rets]) - 1) * 100

    # Parameter stability: an edge that flips its parameters every fold was fit
    # to noise even if the stitched OOS return looks fine.
    param_counts = {}
    for f in folds:
        key = str(sorted(f["chosen_params"].items()))
        param_counts[key] = param_counts.get(key, 0) + 1
    stability = max(param_counts.values()) / len(folds)

    result = {
        "folds": folds,
        "n_folds": len(folds),
        "oos_metrics": {
            "compounded_return_pct": round(compounded, 2),
            "mean_fold_return_pct": round(float(np.mean(oos_rets)), 2),
            "mean_sharpe": round(float(np.mean(oos_sharpes)), 2),
            "worst_fold_return_pct": round(float(np.min(oos_rets)), 2),
            "profitable_folds": int(sum(1 for r in oos_rets if r > 0)),
            "num_trades": trades,
        },
        "in_sample_mean_return_pct": round(float(np.mean(is_rets)), 2),
        "overfit_gap_pct": round(float(np.mean(is_rets) - np.mean(oos_rets)), 2),
        "param_stability": round(stability, 2),
    }
    if len(folds) < 4:
        result["warning"] = (
            f"Only {len(folds)} fold(s) -- OOS estimate is noisy; prefer more history"
        )
    return result
