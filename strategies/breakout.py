import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """Donchian channel + SuperTrend breakout strategy."""

    name = "breakout"
    description = "Breakout: Donchian channel + SuperTrend + volume confirmation"
    default_params = {
        "donchian_length": 20,
        "supertrend_length": 10,
        "supertrend_mult": 3.0,
        "volume_expansion": 1.5,
    }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < self.params["donchian_length"] + 5:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Not enough data"}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close = last.get("close", 0)
        confidence = 0.0
        action, reason = "HOLD", "No signal"

        don_upper = last.get("don_upper")
        don_lower = last.get("don_lower")
        st = last.get("supertrend")
        st_dir = last.get("st_direction", 1)

        if don_upper is None or don_lower is None or pd.isna(don_upper) or pd.isna(don_lower):
            return {"action": "HOLD", "confidence": 0.0, "reason": "Missing Donchian data"}

        vol_exp = last.get("volume_expansion", 0)
        vol_confirm = pd.notna(vol_exp) and vol_exp >= self.params["volume_expansion"]
        breakout_upper = prev["high"] < don_upper and last["high"] >= don_upper
        breakout_lower = prev["low"] > don_lower and last["low"] <= don_lower

        if breakout_upper and close > don_upper:
            action = "BUY"
            confidence = 0.7 if st_dir == 1 else 0.5
            confidence = min(0.85, confidence + 0.1) if vol_confirm else confidence
            reason = f"Price broke above {self.params['donchian_length']}-period high"
        elif breakout_lower and close < don_lower:
            action = "SELL"
            confidence = 0.7 if st_dir == -1 else 0.5
            confidence = min(0.85, confidence + 0.1) if vol_confirm else confidence
            reason = f"Price broke below {self.params['donchian_length']}-period low"

        return {"action": action, "confidence": round(float(confidence), 3), "reason": reason}