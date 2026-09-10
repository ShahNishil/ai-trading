import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """RSI + Bollinger Band mean reversion strategy."""

    name = "mean_reversion"
    description = "Mean reversion: Buy oversold at lower band, sell overbought at upper band"
    default_params = {
        "bb_length": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "exit_in_middle": True,
    }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 40:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Not enough data"}
        last = df.iloc[-1]
        rsi = last.get("rsi")
        bb_lower, bb_upper, bb_mid = last.get("bb_lower"), last.get("bb_upper"), last.get("bb_middle")
        confidence = 0.0
        action, reason = "HOLD", "No signal"

        if rsi is None or pd.isna(rsi) or bb_lower is None or pd.isna(bb_lower):
            return {"action": "HOLD", "confidence": 0.0, "reason": "Missing RSI/BB data"}

        close = last.get("close", 0)

        # Entry: oversold near/below lower band
        if rsi < self.params["oversold"] and close <= bb_lower:
            action, confidence = "BUY", 0.75
            reason = f"RSI {rsi:.1f} oversold at lower Bollinger band"
        elif rsi > self.params["overbought"] and close >= bb_upper:
            action, confidence = "SELL", 0.75
            reason = f"RSI {rsi:.1f} overbought at upper Bollinger band"
        # Partial: strong oversold reversion potential
        elif rsi < self.params["oversold"] - 5:
            action, confidence = "BUY", 0.55
            reason = f"Deeply oversold RSI {rsi:.1f}"
        elif rsi > self.params["overbought"] + 5:
            action, confidence = "SELL", 0.55
            reason = f"Deeply overbought RSI {rsi:.1f}"

        return {"action": action, "confidence": round(float(confidence), 3), "reason": reason}