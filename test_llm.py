import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
import json

rt = bootstrap()
ai = rt['ai_engine']
print("Model:", ai.model)
print("API key present?", bool(ai.client.api_key and ai.client.api_key != "missing"))
# Try simple call
from agents.prompts import TRIGGER_AGENT_SYSTEM_PROMPT
from core.indicators import IndicatorEngine
import pandas as pd
from data.fetcher import DataFetcher

df = rt['data_fetcher'].fetch_daily('RELIANCE','1602',days=20)
eng = IndicatorEngine(df)
full = eng.compute_all()
from core.indicators import IndicatorEngine as IE
summary = IE.latest_summary(full)
from core.signals import SignalGenerator
sg = SignalGenerator(full)
sig = sg.ensemble_signal(full)
blob = json.dumps({"symbol":"RELIANCE","close":summary.get("close"),"indicators":summary,"technical_signals":sig.get("all",{}),"ensemble":sig}, indent=2)
print("Blob preview:", blob[:500])
print("\n--- Calling Groq ---")
res = ai.analyze_indicators(TRIGGER_AGENT_SYSTEM_PROMPT, blob)
print("LLM response:", json.dumps(res, indent=2))

# Also test raw OpenAI call error
try:
    r = ai.client.chat.completions.create(
        model=ai.model,
        messages=[{"role":"system","content":TRIGGER_AGENT_SYSTEM_PROMPT},{"role":"user","content":blob}],
        temperature=0.1,
        max_tokens=500,
    )
    print("Raw success:", r.choices[0].message.content[:300])
except Exception as e:
    print("Raw error:", e)
    import traceback
    traceback.print_exc()
