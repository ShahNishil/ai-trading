import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from agents.trigger_agent import TriggerAgent
import pathlib, sqlite3

rt = bootstrap()
db = pathlib.Path(r'D:\ai-trading\data\market_cache.db')
if db.exists():
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM candles WHERE symbol='RELIANCE'")
    conn.commit(); conn.close()
    print('Cache cleared for RELIANCE')

df = rt['data_fetcher'].fetch_daily('RELIANCE', '1602', days=60)
print('RELIANCE df shape:', df.shape if df is not None else None)
if df is not None and not df.empty:
    print(df.tail(2).to_string())
    from core.indicators import IndicatorEngine
    from core.signals import SignalGenerator
    eng = IndicatorEngine(df)
    full = eng.compute_all()
    sg = SignalGenerator(full)
    print('Ensemble:', sg.ensemble_signal(full))
    agent = TriggerAgent(rt['ai_engine'], rt['data_fetcher'], rt['config'])
    res = agent.analyze_symbol('RELIANCE', '1602')
    import json
    print('=== TRIGGER RESULT ===')
    print(json.dumps({k: v for k,v in res.items() if k not in ['technical','all']}, indent=2))
    if 'technical' in res:
        print('Tech:', res['technical'])
else:
    print('Still no data - check yfinance')
