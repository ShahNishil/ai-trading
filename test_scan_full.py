import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from agents.trigger_agent import TriggerAgent
import time

rt = bootstrap()
print("Model:", rt['config']['groq']['model'])
print("Dhan:", rt['dhan'])
print("Groq key present:", bool(rt['ai_engine'].client.api_key))

# Clear cache for fresh
import pathlib, sqlite3
db = pathlib.Path(r'D:\ai-trading\data\market_cache.db')
# Don't clear all, just test with fresh fetches - fetcher will use cache if valid
# Force use_cache=False for this test via direct yfinance?

agent = TriggerAgent(rt['ai_engine'], rt['data_fetcher'], rt['config'])
from data.universe import StockUniverse
watchlist = StockUniverse(rt['config']).resolve("NIFTY50")
print(f"Testing {len(watchlist)} symbols (first 5 for speed)...")
test_list = watchlist[:6]

results = []
for item in test_list:
    sym = item['symbol']
    print(f"\n--- {sym} ---")
    r = agent.analyze_symbol(sym, item['security_id'])
    print(f"  Action: {r.get('action')} Conf: {r.get('confidence')} Source: {r.get('source')}")
    print(f"  Reasoning: {r.get('reasoning','')[:120]}")
    results.append(r)
    time.sleep(0.5)

print("\n=== SUMMARY ===")
buy = sum(1 for r in results if r['action']=='BUY')
sell = sum(1 for r in results if r['action']=='SELL')
hold = sum(1 for r in results if r['action']=='HOLD')
err = sum(1 for r in results if r['action']=='ERROR')
print(f"BUY: {buy} SELL: {sell} HOLD: {hold} ERROR: {err} / {len(results)}")
for r in results:
    print(f" {r['symbol']:12} {r['action']:5} {r['confidence']:.2f} {r.get('source')}")

# Also test ensemble without LLM to ensure technical gives results
print("\n--- Technical-only check ---")
from core.indicators import IndicatorEngine
from core.signals import SignalGenerator
df = rt['data_fetcher'].fetch_daily('RELIANCE','1602',days=60)
full = IndicatorEngine(df).compute_all()
sg = SignalGenerator(full)
print(sg.ensemble_signal(full))
