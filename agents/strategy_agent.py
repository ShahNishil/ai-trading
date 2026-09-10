import json

from agents.prompts import STRATEGY_GENERATOR_PROMPT
from core.ai_engine import AIEngine


class StrategyAgent:
    """Uses the LLM to generate new trading strategies for backtesting."""

    def __init__(self, ai_engine: AIEngine):
        self.ai = ai_engine

    def generate(
        self,
        symbol: str = "",
        historical_perf: str = "Not provided",
        focus: str = "trend following",
    ) -> dict:
        data_blob = (
            f"Target asset: {symbol or 'unspecified Indian equity'}\n"
            f"Market style: {focus}\n"
            f"Historical notes: {historical_perf}\n"
            f"Available indicators: RSI, MACD, EMA(9,21,50,200), SMA, Bollinger Bands, "
            f"ATR, SuperTrend, ADX, Stoch, MFI, VWAP, OBV, Donchian, CCI, WillR"
        )
        result = self.ai.generate_strategy(STRATEGY_GENERATOR_PROMPT, data_blob)
        strategy = result.get("parsed", {})
        if "name" not in strategy:
            strategy = {"name": "ai_generated", "description": "AI-generated strategy", **strategy}
        strategy["_source"] = result.get("raw", "")
        return strategy

    def to_ai_strategy_params(self, strategy: dict) -> dict:
        """Convert an AI strategy description into params for the AIStrategy class."""
        entry = strategy.get("entry_conditions", "")
        exit_rule = strategy.get("exit_conditions", "")
        return {
            "rules": {
                "entry": entry,
                "exit": exit_rule,
                "indicators": strategy.get("indicators_needed", []),
                "stop_loss": strategy.get("stop_loss_rule", ""),
                "target": strategy.get("take_profit_rule", ""),
            },
            "description": strategy.get("description", "AI generated"),
        }