"""Confidence calibration: did signals with confidence 0.7 actually win ~70%?

The prompt asks the LLM for calibrated probabilities and the position gate
keys off them, but nothing closes the loop. This module resolves recorded
signals against subsequent price data and reports stated confidence vs
realized hit rate per bucket, plus a Brier score.

Resolution rule (mirrors the backtest's intrabar convention): walk daily bars
after the signal; a BUY wins if the bar's high reaches the target before any
bar's low reaches the stop -- stop-first when one bar spans both. Signals
whose horizon (timeframe_hours, floor one trading day) passes without either
level being touched EXPIRE and score as wins only if the close moved in the
signal's favour.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def resolve_signal(signal: dict, bars: pd.DataFrame) -> dict:
    """Resolve one signal against OHLC bars strictly AFTER its creation time."""
    created = pd.Timestamp(signal["created_at"])
    horizon_days = max(1, int(round(float(signal.get("timeframe_hours") or 48) / 24.0)))
    entry = float(signal["entry_price"])
    stop = float(signal["stop_loss"])
    target = float(signal["target"])
    is_buy = signal["action"] == "BUY"

    after = bars[bars.index > created]
    if after.empty:
        return {"outcome": "UNRESOLVED"}

    for i, (ts, row) in enumerate(after.iterrows()):
        hi, lo = float(row["high"]), float(row["low"])
        hit_stop = lo <= stop if is_buy else hi >= stop
        hit_target = hi >= target if is_buy else lo <= target
        if hit_stop:  # adverse-first on ambiguous bars
            ret = (stop - entry) / entry * 100 * (1 if is_buy else -1)
            return {"outcome": "LOSS", "exit_price": stop,
                    "realized_return_pct": round(ret, 3), "bars_held": i + 1}
        if hit_target:
            ret = (target - entry) / entry * 100 * (1 if is_buy else -1)
            return {"outcome": "WIN", "exit_price": target,
                    "realized_return_pct": round(ret, 3), "bars_held": i + 1}
        if i + 1 >= horizon_days:
            close = float(row["close"])
            ret = (close - entry) / entry * 100 * (1 if is_buy else -1)
            return {"outcome": "EXPIRED", "exit_price": close,
                    "realized_return_pct": round(ret, 3), "bars_held": i + 1}
    return {"outcome": "UNRESOLVED"}  # horizon still open, keep waiting


def resolve_signals(cache, fetcher, watchlist: list, timeframe: str = "daily") -> dict:
    """Resolve all unresolved signals using fresh price data. Returns counts."""
    by_symbol = {str(i.get("symbol", "")).upper(): i for i in watchlist}
    counts = {"WIN": 0, "LOSS": 0, "EXPIRED": 0, "UNRESOLVED": 0, "NO_DATA": 0}
    bars_cache = {}
    for sig in cache.get_unresolved_signals():
        sym = str(sig["symbol"]).upper()
        if sym not in bars_cache:
            item = by_symbol.get(sym, {})
            try:
                bars_cache[sym] = fetcher.fetch_daily(
                    sym, item.get("security_id", ""), days=120,
                    exchange=item.get("exchange", "NSE_EQ"),
                )
            except Exception:
                bars_cache[sym] = None
        bars = bars_cache[sym]
        if bars is None or bars.empty:
            counts["NO_DATA"] += 1
            continue
        res = resolve_signal(sig, bars)
        counts[res["outcome"]] += 1
        if res["outcome"] in ("WIN", "LOSS", "EXPIRED"):
            cache.save_signal_outcome({"signal_id": sig["id"], **res})
    return counts


def calibration_report(cache, min_bucket: int = 5) -> dict:
    """Stated confidence vs realized win rate, bucketed, plus Brier score."""
    rows = cache.get_resolved_signals()
    if not rows:
        return {"error": "No resolved signals yet", "n": 0}

    outcomes = []
    for r in rows:
        won = r["outcome"] == "WIN" or (
            r["outcome"] == "EXPIRED" and float(r.get("realized_return_pct") or 0) > 0
        )
        outcomes.append((float(r["confidence"]), 1.0 if won else 0.0,
                         float(r.get("realized_return_pct") or 0), r["source"]))

    conf = np.array([o[0] for o in outcomes])
    won = np.array([o[1] for o in outcomes])
    rets = np.array([o[2] for o in outcomes])
    brier = float(np.mean((conf - won) ** 2))

    buckets = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        hi = lo + 0.1
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo)
        n = int(mask.sum())
        if n == 0:
            continue
        buckets.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "stated_confidence": round(float(conf[mask].mean()), 3),
            "realized_win_rate": round(float(won[mask].mean()), 3),
            "gap": round(float(conf[mask].mean() - won[mask].mean()), 3),
            "mean_return_pct": round(float(rets[mask].mean()), 3),
            "reliable": n >= min_bucket,
        })

    overconfident = [b for b in buckets if b["reliable"] and b["gap"] > 0.10]
    return {
        "n": len(outcomes),
        "overall_win_rate": round(float(won.mean()), 3),
        "mean_stated_confidence": round(float(conf.mean()), 3),
        "brier_score": round(brier, 4),  # 0.25 = coin flip at conf 0.5; lower is better
        "buckets": buckets,
        "overconfident_buckets": [b["bucket"] for b in overconfident],
        "verdict": (
            "OVERCONFIDENT: stated confidence exceeds realized win rate by >10pp "
            "in " + ", ".join(b["bucket"] for b in overconfident)
            if overconfident else "No reliable bucket shows >10pp overconfidence"
        ),
    }
