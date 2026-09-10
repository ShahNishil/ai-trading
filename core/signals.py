import numpy as np
import pandas as pd


class SignalGenerator:
    """Rule-based signal generation from technical indicators."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    # Composite signal scores
    # ------------------------------------------------------------------
    def momentum_score(self) -> float:
        """Higher = bullish momentum, lower = bearish. Range roughly -10..+10."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        score = 0.0
        checks = [
            ("rsi", 55, 45),
            ("adx", 25, 20),
            ("dmp", 24, 18),
            ("dmn", 24, 18),
            ("macd_hist", 0.0, 0.0),
            ("stoch_k", 60, 40),
            ("mfi", 60, 40),
        ]
        for col, buy_thresh, sell_thresh in checks:
            val = last.get(col)
            if val is None or pd.isna(val):
                continue
            if val > buy_thresh:
                score += 1
            elif val < sell_thresh:
                score -= 1

        # Trend alignment
        close, ema9, ema21, ema50, ema200 = (
            last.get("close"),
            last.get("ema9"),
            last.get("ema21"),
            last.get("ema50"),
            last.get("ema200"),
        )
        if close and ema9 and ema21 and ema50:
            if close > ema9 > ema21 > ema50:
                score += 2
            elif close < ema9 < ema21 < ema50:
                score -= 2

        # Bollinger band position
        bb_upper, bb_lower, bb_middle = last.get("bb_upper"), last.get("bb_lower"), last.get("bb_middle")
        if bb_upper and bb_lower and bb_middle and close:
            if close > bb_upper:
                score += 1
            elif close < bb_lower:
                score -= 1

        return float(np.clip(score, -10, 10))

    def mean_reversion_score(self) -> float:
        """Higher = oversold (buy candidates), Lower = overbought (sell candidates)."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        score = 0.0

        rsi = last.get("rsi")
        if rsi is not None and not pd.isna(rsi):
            if rsi < 30:
                score += 2
            elif rsi < 40:
                score += 1
            elif rsi > 70:
                score -= 2
            elif rsi > 60:
                score -= 1

        close, bb_lower, bb_upper = last.get("close"), last.get("bb_lower"), last.get("bb_upper")
        if close and bb_lower and close < bb_lower:
            score += 2
        if close and bb_upper and close > bb_upper:
            score -= 2

        stoch_k = last.get("stoch_k")
        if stoch_k is not None and not pd.isna(stoch_k):
            if stoch_k < 20:
                score += 1
            elif stoch_k > 80:
                score -= 1

        return float(np.clip(score, -10, 10))

    def breakout_score(self) -> float:
        """Higher = breakout likelihood (buy), lower = breakdown (sell)."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) > 1 else last
        score = 0.0

        don_upper, don_lower = last.get("don_upper"), last.get("don_lower")
        close = last.get("close")
        if close and don_upper and close >= don_upper:
            score += 3
        elif close and don_lower and close <= don_lower:
            score -= 3

        sup = last.get("supertrend")
        if sup is not None and not pd.isna(sup):
            if last.get("st_direction") == 1:
                score += 2
            else:
                score -= 2

        vol_exp = last.get("volume_expansion")
        if vol_exp is not None and not pd.isna(vol_exp):
            if vol_exp > 1.5:
                score += 1 if pd.notna(last.get("close")) and pd.notna(prev.get("close")) and last["close"] > prev["close"] else -1

        adx = last.get("adx")
        if adx is not None and not pd.isna(adx) and adx < 20:
            score -= 1  # Range-bound, breakout filters

        return float(np.clip(score, -10, 10))

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame, strategy: str = "momentum") -> dict:
        """Return a single latest signal based on the strategy signature."""
        if df is None or df.empty:
            return {"action": "HOLD", "confidence": 0.0, "score": 0.0, "reason": "No data"}
        self.df = df
        score = 0.0
        if strategy == "momentum":
            score = self.momentum_score()
        elif strategy == "mean_reversion":
            score = self.mean_reversion_score()
        elif strategy == "breakout":
            score = self.breakout_score()

        # Calibrated confidence: score 2 -> 0.55, 3 -> 0.65, 5 -> 0.78, 7 -> 0.88
        abs_s = abs(score)
        if abs_s < 1:
            confidence = abs_s * 0.35
        elif abs_s < 2:
            confidence = 0.35 + (abs_s - 1) * 0.15
        elif abs_s < 4:
            confidence = 0.50 + (abs_s - 2) * 0.08
        else:
            confidence = 0.66 + (abs_s - 4) * 0.05
        confidence = float(np.clip(confidence, 0, 0.92))

        if score >= 2:
            action = "BUY"
        elif score <= -2:
            action = "SELL"
        else:
            action = "HOLD"
            # HOLD confidence is low when score near 0, higher when borderline
            confidence = round(0.45 + abs_s * 0.05, 3)

        return {
            "action": action,
            "confidence": round(float(confidence), 3),
            "score": round(float(score), 3),
            "reason": self._build_reason(action),
        }

    def get_all_signals(self, df: pd.DataFrame) -> dict:
        """Return all three strategy signals for ensemble."""
        if df is None or df.empty:
            return {}
        return {
            "momentum": self.get_signal(df, "momentum"),
            "mean_reversion": self.get_signal(df, "mean_reversion"),
            "breakout": self.get_signal(df, "breakout"),
        }

    def ensemble_signal(self, df: pd.DataFrame) -> dict:
        """Best of three signals by confidence."""
        all_s = self.get_all_signals(df)
        if not all_s:
            return {"action": "HOLD", "confidence": 0.0, "score": 0.0, "reason": "No data"}
        # Prefer BUY/SELL over HOLD, then highest confidence
        scored = sorted(
            all_s.items(),
            key=lambda kv: (kv[1]["action"] != "HOLD", kv[1]["confidence"]),
            reverse=True,
        )
        best_name, best = scored[0]
        best = dict(best)
        best["strategy"] = best_name
        best["all"] = all_s
        return best

    def _build_reason(self, action: str) -> str:
        last = self.df.iloc[-1]
        rsi = last.get("rsi")
        close, ema9, ema21 = last.get("close"), last.get("ema9"), last.get("ema21")
        parts = []
        if close and ema9 and ema21:
            if close > ema9 > ema21:
                parts.append("price above rising EMAs")
            elif close < ema9 < ema21:
                parts.append("price below falling EMAs")
        if rsi is not None and not pd.isna(rsi):
            if rsi > 70:
                parts.append(f"RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                parts.append(f"RSI oversold at {rsi:.1f}")
            else:
                parts.append(f"RSI neutral at {rsi:.1f}")
        return "; ".join(parts) if parts else "No strong technical signal"