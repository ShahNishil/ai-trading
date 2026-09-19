"""Signal resolution and calibration reporting."""
import os
import tempfile

import pandas as pd

from core.calibration import calibration_report, resolve_signal
from data.cache import DataCache


def _bars(rows, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(rows), freq="B")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in rows], index=idx
    )


SIG = {"action": "BUY", "entry_price": 100.0, "stop_loss": 97.0, "target": 104.0,
       "created_at": "2024-01-01T16:00:00", "timeframe_hours": 96}


def test_buy_win_when_target_hit_first():
    bars = _bars([(100, 101, 99, 100), (101, 105, 100, 104)])
    r = resolve_signal(SIG, bars)
    assert r["outcome"] == "WIN" and r["exit_price"] == 104.0


def test_buy_loss_when_stop_hit_first():
    bars = _bars([(100, 101, 96, 97), (97, 105, 96, 104)])
    r = resolve_signal(SIG, bars)
    assert r["outcome"] == "LOSS" and r["exit_price"] == 97.0


def test_ambiguous_bar_scores_as_loss():
    """One bar spans stop AND target: adverse-first, same as the backtest."""
    bars = _bars([(100, 105, 96, 101)])
    assert resolve_signal(SIG, bars)["outcome"] == "LOSS"


def test_only_bars_after_creation_count():
    """A signal created after the market closed must not be scored on that day's bar."""
    sig = dict(SIG, created_at="2024-01-02T16:00:00")  # same day as a huge up bar
    # Jan 2 bar would be a WIN (high 106 > target 104); the bar after is neutral.
    bars = _bars([(100, 106, 99, 105), (100, 101, 99, 100.5)])
    r = resolve_signal(sig, bars)
    assert r["outcome"] != "WIN", "scored on a bar that closed before the signal existed"


def test_sell_side_is_mirrored():
    sig = {"action": "SELL", "entry_price": 100.0, "stop_loss": 103.0, "target": 96.0,
           "created_at": "2024-01-01T16:00:00", "timeframe_hours": 96}
    bars = _bars([(100, 101, 95, 96)])
    r = resolve_signal(sig, bars)
    assert r["outcome"] == "WIN"
    assert r["realized_return_pct"] > 0


def test_calibration_report_flags_overconfidence():
    cache = DataCache(os.path.join(tempfile.mkdtemp(), "c.db"))
    # 10 signals at stated 0.9 confidence, but only 2 win -> grossly overconfident
    for i in range(10):
        sid = cache.save_signal({"symbol": "ACME", "action": "BUY", "confidence": 0.9,
                                 "entry_price": 100, "stop_loss": 97, "target": 104})
        cache.save_signal_outcome({"signal_id": sid,
                                   "outcome": "WIN" if i < 2 else "LOSS",
                                   "exit_price": 104 if i < 2 else 97,
                                   "realized_return_pct": 4.0 if i < 2 else -3.0,
                                   "bars_held": 2})
    rep = calibration_report(cache)
    assert rep["n"] == 10
    assert rep["overall_win_rate"] == 0.2
    assert "0.9-1.0" in rep["overconfident_buckets"]
    assert rep["brier_score"] > 0.4  # far worse than a coin flip


def test_unresolved_signals_roundtrip():
    cache = DataCache(os.path.join(tempfile.mkdtemp(), "c.db"))
    sid = cache.save_signal({"symbol": "ACME", "action": "BUY", "confidence": 0.7,
                             "entry_price": 100, "stop_loss": 97, "target": 104})
    cache.save_signal({"symbol": "ACME", "action": "HOLD", "confidence": 0.5,
                       "entry_price": 100, "stop_loss": 0, "target": 0})
    unresolved = cache.get_unresolved_signals()
    assert len(unresolved) == 1 and unresolved[0]["id"] == sid  # HOLD is not scoreable
    cache.save_signal_outcome({"signal_id": sid, "outcome": "WIN", "exit_price": 104,
                               "realized_return_pct": 4.0, "bars_held": 3})
    assert cache.get_unresolved_signals() == []
    assert len(cache.get_resolved_signals()) == 1
