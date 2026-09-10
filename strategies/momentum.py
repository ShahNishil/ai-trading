import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """EMA + MACD + RSI momentum crossover strategy."""

    name = "momentum"
    description = "Momentum: EMA golden cross with MACD & RSI confirmation"
    default_params = {
        "fast_ema": 9,
        "slow_ema": 21,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < self.params["slow_ema"] + 10:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Not enough data"}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        confidence = 0.0

        fast, slow = self.params["fast_ema"], self.params["slow_ema"]
        fcol, scol = f"ema{fast}", f"ema{slow}"
        if fcol not in df.columns or scol not in df.columns:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Missing EMA columns"}

        # Golden / death cross
        cross_up = last[fcol] > last[scol] and prev[fcol] <= prev[scol]
        cross_down = last[fcol] < last[scol] and prev[fcol] >= prev[scol]

        rsi = last.get("rsi", 50)
        action, reason = "HOLD", "No crossover"
        if cross_up and rsi < self.params["rsi_overbought"]:
            action, confidence = "BUY", 0.8
            reason = f"EMA{fast} crossed above EMA{slow}, RSI {rsi:.1f}"
        elif cross_down and rsi > self.params["rsi_oversold"]:
            action, confidence = "SELL", 0.8
            reason = f"EMA{fast} crossed below EMA{slow}, RSI {rsi:.1f}"
        else:
            # Trend + momentum confirmation
            macd_hist = last.get("macd_hist", 0)
            if pd.notna(macd_hist):
                rising = macd_hist > 0
                above = last.get("close", 0) > last.get(scol, np.inf)
                if above and rising and rsi > 50 and rsi < 65:
                    action, confidence = "BUY", 0.6
                    reason = "Uptrend with positive MACD momentum"
                elif not above and not rising and rsi < 50:
                    action, confidence = "SELL", 0.6
                    reason = "Downtrend with negative MACD momentum"

        return {"action": action, "confidence": round(float(confidence), 3), "reason": reason}