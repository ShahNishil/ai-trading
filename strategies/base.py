from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name = "base"
    description = "Base strategy"
    default_params = {}

    def __init__(self, params: dict = None):
        self.params = {**self.default_params, **(params or {})}
        self.last_signal = {"action": "HOLD", "confidence": 0.0}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> dict:
        """Given the full indicator dataframe, return latest signal.

        Returns dict: {"action": "BUY"|"SELL"|"HOLD", "confidence": 0..1, "reason": str}
        """
        raise NotImplementedError

    def on_bar(self, df: pd.DataFrame) -> dict:
        self.last_signal = self.generate_signal(df)
        return self.last_signal

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "params": self.params}