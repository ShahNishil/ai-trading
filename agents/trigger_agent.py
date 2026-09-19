import json
from typing import Optional

import pandas as pd

from agents.prompts import TRIGGER_AGENT_SYSTEM_PROMPT
from core.ai_engine import AIEngine
from core.indicators import IndicatorEngine
from core.signals import SignalGenerator
from data.fetcher import DataFetcher


class TriggerAgent:
    """Feature 1: AI recommends stocks via technical indicator analysis — hybrid AI + technical fallback."""

    def __init__(self, ai_engine: AIEngine, data_fetcher: DataFetcher, config: dict):
        self.ai = ai_engine
        self.data = data_fetcher
        self.config = config

    def build_data_blob(self, symbol: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame, dict]:
        indicator_engine = IndicatorEngine(df)
        full = indicator_engine.compute_all()
        summary = IndicatorEngine.latest_summary(full)
        sg = SignalGenerator(full)
        ensemble = sg.ensemble_signal(full)
        all_signals = ensemble.get("all", {})
        # Include ensemble + close price for LLM
        blob = {
            "symbol": symbol,
            "close": summary.get("close", 0),
            "indicators": summary,
            "technical_signals": all_signals,
            "ensemble": {k: v for k, v in ensemble.items() if k != "all"},
        }
        return json.dumps(blob, indent=2), full, ensemble

    @staticmethod
    def _levels(close: float, atr, action: str) -> tuple:
        """(stop_loss, target) sized to the stock's own volatility.

        1.5x ATR stop / 2.5x ATR target when ATR is available; the old fixed
        3%/4% otherwise. A fixed percentage is simultaneously too tight for a
        volatile name (stopped out by noise) and too loose for a quiet one
        (gives up more than the setup warrants).
        """
        use_atr = atr is not None and atr == atr and atr > 0 and atr < close * 0.2
        if action == "BUY":
            if use_atr:
                return round(close - 1.5 * atr, 2), round(close + 2.5 * atr, 2)
            return round(close * 0.97, 2), round(close * 1.04, 2)
        if action == "SELL":
            if use_atr:
                return round(close + 1.5 * atr, 2), round(close - 2.5 * atr, 2)
            return round(close * 1.03, 2), round(close * 0.96, 2)
        return 0, 0

    def _fill_prices(self, result: dict, close_price: float, atr=None) -> dict:
        """Ensure entry_price/stop_loss/target are populated from close if LLM omitted."""
        try:
            close = float(close_price) if close_price else 0
        except Exception:
            close = 0
        if close <= 0:
            return result
        action = result.get("action", "HOLD")
        if not result.get("entry_price"):
            result["entry_price"] = round(close, 2)
        stop, target = self._levels(close, atr, action)
        if action in ("BUY", "SELL"):
            if not result.get("stop_loss"):
                result["stop_loss"] = stop
            if not result.get("target"):
                result["target"] = target
        else:  # HOLD
            result.setdefault("stop_loss", 0)
            result.setdefault("target", 0)
        result.setdefault("timeframe_hours", 48)
        return result

    def analyze_symbol(
        self,
        symbol: str,
        security_id: str,
        exchange: str = "NSE_EQ",
        timeframe: str = "daily",
        lookback_days: int = 200,
    ) -> dict:
        """Analyze a single symbol and return an AI recommendation with technical fallback."""
        try:
            # Fetch data — DataFetcher handles Dhan vs yfinance fallback internally
            if timeframe == "daily":
                df = self.data.fetch_daily(symbol, security_id, days=lookback_days, exchange=exchange)
            else:
                interval = int(timeframe.replace("min", ""))
                df = self.data.fetch_intraday(symbol, security_id, interval, days=lookback_days, exchange=exchange)

            if df is None or df.empty:
                return {"symbol": symbol, "action": "ERROR", "confidence": 0.0, "reasoning": "No data available — check symbol/security_id or yfinance connectivity", "source": "error"}

            if len(df) < 30:
                return {"symbol": symbol, "action": "HOLD", "confidence": 0.35, "reasoning": f"Insufficient data ({len(df)} bars) for reliable indicators", "source": "technical"}

            data_blob, full, technical = self.build_data_blob(symbol, df)
            close_price = float(full.iloc[-1]["close"]) if "close" in full.columns else 0
            last_atr = float(full.iloc[-1]["atr"]) if "atr" in full.columns else None

            # Try LLM
            llm_result = self.ai.analyze_indicators(TRIGGER_AGENT_SYSTEM_PROMPT, data_blob)
            has_llm_error = "error" in llm_result

            # Normalize LLM result
            if not has_llm_error:
                llm_result["symbol"] = symbol
                action = str(llm_result.get("action", "HOLD")).upper()
                llm_result["action"] = action if action in ("BUY", "SELL", "HOLD") else "HOLD"
                try:
                    llm_result["confidence"] = float(llm_result.get("confidence", 0) or 0)
                except Exception:
                    llm_result["confidence"] = 0.0
                llm_result = self._fill_prices(llm_result, close_price, atr=last_atr)
                llm_result["source"] = "ai"
            else:
                llm_result = None

            # Technical fallback — always compute
            tech_action = technical.get("action", "HOLD")
            tech_conf = technical.get("confidence", 0)

            # Hybrid decision: prefer LLM when confident, otherwise use technical ensemble
            if llm_result and not has_llm_error:
                llm_conf = llm_result.get("confidence", 0)
                # If LLM is HOLD with low confidence but technical is BUY/SELL with good confidence, use technical
                if llm_result["action"] == "HOLD" and tech_action in ("BUY", "SELL") and tech_conf >= 0.55 and llm_conf < 0.60:
                    # Blend: use technical action but keep LLM reasoning if present
                    t_stop, t_target = self._levels(close_price, last_atr, tech_action)
                    result = {
                        "symbol": symbol,
                        "action": tech_action,
                        "confidence": round(tech_conf, 3),
                        "entry_price": round(close_price, 2),
                        "stop_loss": t_stop,
                        "target": t_target,
                        "timeframe_hours": 48,
                        "reasoning": technical.get("reason", "") + f" (Technical {technical.get('strategy','')})",
                        "source": "technical",
                        "llm_reasoning": llm_result.get("reasoning", ""),
                        "technical": technical,
                    }
                    return result
                # If LLM is decisive, use it but attach technical for reference
                llm_result["technical"] = technical
                return llm_result
            else:
                # LLM failed — pure technical fallback
                f_stop, f_target = self._levels(close_price, last_atr, tech_action)
                result = {
                    "symbol": symbol,
                    "action": tech_action,
                    "confidence": round(tech_conf, 3),
                    "entry_price": round(close_price, 2),
                    "stop_loss": f_stop,
                    "target": f_target,
                    "timeframe_hours": 48,
                    "reasoning": technical.get("reason", "") + f" (Technical fallback — LLM unavailable: {has_llm_error})",
                    "source": "technical",
                    "technical": technical,
                }
                # Ensure HOLD has 0 prices
                if result["action"] == "HOLD":
                    result["stop_loss"] = 0
                    result["target"] = 0
                return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"symbol": symbol, "action": "ERROR", "confidence": 0.0, "reasoning": str(e), "source": "error"}

    def _record_signal(self, result: dict):
        """Persist actionable signals so calibration can score them later.

        Without this record there is nothing to compare stated confidence
        against; the signals table existed but nothing ever wrote to it.
        """
        try:
            if result.get("action") in ("BUY", "SELL") and result.get("entry_price"):
                self.data.cache.save_signal(
                    {
                        "symbol": result.get("symbol", ""),
                        "action": result["action"],
                        "confidence": float(result.get("confidence", 0) or 0),
                        "entry_price": float(result.get("entry_price", 0) or 0),
                        "stop_loss": float(result.get("stop_loss", 0) or 0),
                        "target": float(result.get("target", 0) or 0),
                        "reasoning": str(result.get("reasoning", ""))[:500],
                        "source": result.get("source", "ai"),
                        "timeframe_hours": float(result.get("timeframe_hours", 48) or 48),
                    }
                )
        except Exception:
            pass  # recording must never break a scan

    def analyze_watchlist(self, watchlist: list, timeframe: str = "daily") -> list:
        """Analyze an entire watchlist and return all recommendations ranked by confidence."""
        results = []
        for item in watchlist:
            symbol = item.get("symbol", "")
            security_id = item.get("security_id", "")
            if not symbol:
                continue
            # Allow empty security_id when using yfinance fallback (uses symbol)
            result = self.analyze_symbol(
                symbol, security_id, item.get("exchange", "NSE_EQ"), timeframe
            )
            self._record_signal(result)
            results.append(result)
        # Rank: BUY/SELL first by confidence, then HOLD by confidence
        def rank_key(r):
            is_actionable = 1 if r.get("action") in ("BUY", "SELL") else 0
            return (is_actionable, float(r.get("confidence", 0) or 0))
        results.sort(key=rank_key, reverse=True)
        return results
