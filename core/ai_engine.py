import json
import os
from typing import Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI


class AIEngine:
    """Groq LLM integration using OpenAI-compatible API (free LLaMA models)."""

    def __init__(self, config: dict):
        load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
        self.config = config
        groq_cfg = config.get("groq", {})
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            print(
                "WARNING: GROQ_API_KEY not set. AI features will not work. "
                "Get a free key at https://console.groq.com"
            )
        self.client = OpenAI(
            api_key=api_key or "missing",
            base_url="https://api.groq.com/openai/v1",
            timeout=30,
        )
        self.model = groq_cfg.get("model", "llama-3.3-70b-versatile")
        self.temperature = groq_cfg.get("temperature", 0.1)
        self.max_tokens = groq_cfg.get("max_tokens", 2048)

    def analyze_indicators(self, system_prompt: str, data_blob: str) -> dict:
        """Send indicator summary to the LLM and get a structured recommendation.

        Returns a dict with keys: symbol, action, confidence, entry_price,
        stop_loss, target, reasoning. Retries with fallback model on failure.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_blob},
        ]
        fallback_models = [self.model, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "groq/compound-mini", "groq/compound"]
        # dedupe, keep order
        tried = []
        for m in fallback_models:
            if m not in tried:
                tried.append(m)
        last_err = None
        for model in tried:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                parsed = self._parse_response(content)
                if "error" not in parsed:
                    return parsed
                last_err = parsed.get("error")
                # parse error -> try next model only if it was model-related
                if "model" in str(last_err).lower() or "decommissioned" in str(last_err).lower():
                    continue
                return parsed
            except Exception as e:
                last_err = str(e)
                # retry on model errors
                if "model" in last_err.lower() or "decommissioned" in last_err.lower() or "not found" in last_err.lower():
                    continue
                return {"error": last_err, "action": "HOLD", "confidence": 0.0, "reasoning": last_err}
        return {"error": str(last_err), "action": "HOLD", "confidence": 0.0, "reasoning": str(last_err)}

    def generate_strategy(self, system_prompt: str, data_blob: str) -> dict:
        """Ask the LLM to generate/describe a trading strategy."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_blob},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature + 0.2,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return {"raw": content, "parsed": self._parse_response(content)}
        except Exception as e:
            return {"error": str(e), "raw": "", "parsed": {}}

    @staticmethod
    def _parse_response(content: str) -> dict:
        """Extract a JSON object from the LLM output (handles markdown fences)."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                content = "\n".join(lines[1:])
            if content.endswith("```"):
                content = content[:-3].strip()
        try:
            result = json.loads(content)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        # Fallback: find first {...}
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            try:
                result = json.loads(content[start : end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass
        return {"error": "Could not parse LLM response", "action": "HOLD", "confidence": 0.0}