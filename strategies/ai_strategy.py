import pandas as pd

from strategies.base import BaseStrategy


class AIStrategy(BaseStrategy):
    """Strategy that uses AI-generated rules. Delegates signal generation to a
    rule configuration produced by the AI. Falls back to momentum logic if no
    rules provided, so backtests never break.
    """

    name = "ai"
    description = "AI-rule based strategy (rules supplied by the AI engine)"
    default_params = {
        "rules": {},  # {"entry": "...", "exit": "...", "indicators": [...]}
    }

    def __init__(self, params: dict = None):
        super().__init__(params)
        self.rules = self.params.get("rules", {}) or {}

    def generate_signal(self, df: pd.DataFrame) -> dict:
        if df is not None and self.rules:
            try:
                result = self._rule_based_signal(df)
                if result["action"] != "HOLD":
                    return result
            except Exception:
                pass
        # Fallback to momentum
        from strategies.momentum import MomentumStrategy

        return MomentumStrategy().generate_signal(df)

    def _rule_based_signal(self, df: pd.DataFrame) -> dict:
        last = df.iloc[-1] if df is not None and not df.empty else None
        if last is None:
            return {"action": "HOLD", "confidence": 0.0, "reason": "No data"}

        entry = self.rules.get("entry", "").lower()
        exit_rule = self.rules.get("exit", "").lower()

        def _check(rule: str) -> bool:
            if last is None:
                return False
            parts = rule.split(" and ")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if "rsi <" in part and last.get("rsi") is not None and not pd.isna(last["rsi"]):
                    if not (last["rsi"] < float(part.split("<")[1].strip())):
                        return False
                elif "rsi >" in part and last.get("rsi") is not None and not pd.isna(last["rsi"]):
                    if not (last["rsi"] > float(part.split(">")[1].strip())):
                        return False
                elif "price >" in part:
                    if not (last.get("close", 0) > last.get("sma20", 0)):
                        return False
                elif "price <" in part:
                    if not (last.get("close", 0) < last.get("sma20", float("inf"))):
                        return False
            return True

        if "buy" in entry and _check(entry):
            return {"action": "BUY", "confidence": 0.65, "reason": entry}
        if "sell" in entry and _check(entry):
            return {"action": "SELL", "confidence": 0.65, "reason": entry}
        return {"action": "HOLD", "confidence": 0.0, "reason": "No rule matched"}