import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from agents.trigger_agent import TriggerAgent
rt = bootstrap()
agent = TriggerAgent(rt['ai_engine'], rt['data_fetcher'], rt['config'])
from data.universe import StockUniverse
watchlist = StockUniverse(rt['config']).resolve('NIFTY50')[:6]
for item in watchlist:
    r = agent.analyze_symbol(item['symbol'], item['security_id'])
    reason = r.get('reasoning','').replace('\u2011','-').replace('\u2013','-')[:90]
    print(f"{r['symbol']:12} {r['action']:5} {r['confidence']:.2f} {r.get('source',''):10} : {reason}")
    print(f"  entry {r.get('entry_price')} SL {r.get('stop_loss')} tgt {r.get('target')}")

print("\nSummary check with NIFTY50 full 20 (quick technical-only to avoid LLM rate limit):")
from core.indicators import IndicatorEngine
from core.signals import SignalGenerator
for item in watchlist:
    df = rt['data_fetcher'].fetch_daily(item['symbol'], item['security_id'], days=60)
    if df is None or df.empty:
        print(f"{item['symbol']} no data")
        continue
    full = IndicatorEngine(df).compute_all()
    sg = SignalGenerator(full)
    ens = sg.ensemble_signal(full)
    print(f"{item['symbol']:12} {ens['action']:5} {ens['confidence']:.2f} score {ens['score']} {ens['strategy']}")
