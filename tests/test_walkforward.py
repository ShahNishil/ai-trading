"""Walk-forward validation must be causal and expose the overfitting gap."""
from backtest.walkforward import WARMUP_BARS, walk_forward
from tests.synthetic import random_walk_ohlcv


def test_fold_boundaries_are_causal():
    """Every fold's test window must start after its training window ends."""
    df = random_walk_ohlcv(800, seed=2)
    r = walk_forward("momentum", df, {}, train_bars=252, test_bars=63)
    assert r["n_folds"] >= 4
    for f in r["folds"]:
        assert f["train_end"] < f["test_start"], (
            f"test window {f['test_start']} begins inside training data ending {f['train_end']}"
        )


def test_insufficient_data_is_an_error_not_a_guess():
    df = random_walk_ohlcv(100, seed=2)
    r = walk_forward("momentum", df, {}, train_bars=252, test_bars=63)
    assert "error" in r


def test_no_positive_oos_edge_on_noise():
    """On random walks, OOS mean return across seeds must hover near zero-minus-
    costs. A materially positive OOS mean would indicate leakage in the folds."""
    means = []
    for seed in range(4):
        df = random_walk_ohlcv(700, seed=seed)
        r = walk_forward("momentum", df, {"fast_ema": [9, 12]}, train_bars=252, test_bars=63)
        if "error" not in r:
            means.append(r["oos_metrics"]["mean_fold_return_pct"])
    assert means, "no folds ran"
    assert sum(means) / len(means) < 2.0, f"suspicious OOS edge on pure noise: {means}"


def test_reports_overfit_gap_and_stability():
    df = random_walk_ohlcv(800, seed=3)
    r = walk_forward("momentum", df, {"fast_ema": [5, 9, 12]}, train_bars=252, test_bars=63)
    assert "overfit_gap_pct" in r
    assert 0 < r["param_stability"] <= 1.0
