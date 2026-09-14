import sys, json
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from backtest.runner import BacktestRunner
from backtest.visualizer import BacktestVisualizer
from data.derivatives import DerivativeUniverse

rt = bootstrap()
runner = BacktestRunner(rt['data_fetcher'], rt['cache'], config=rt['config'], universe=DerivativeUniverse(config=rt['config']))

print("NIFTY spot fetch (^NSEI)...")
df = rt['data_fetcher'].fetch_daily('^NSEI', '0', days=400)
print("  rows:", len(df), "cols:", list(df.columns))
print(df.tail(2).to_string())

print("\n=== Option: long_straddle on NIFTY ===")
r1 = runner.run_derivative(
    strategy_name='long_straddle',
    underlying='NIFTY',
    kind='IDX',
    params={'num_lots': 1},
    timeframe='daily',
    lookback_days=400,
    initial_capital=500000,
    commission_pct=0.03,
)
print("error:", r1.get('error'))
print("metrics:", r1.get('metrics'))
print("num trades:", len(r1.get('trades', [])))
for t in r1.get('trades', [])[:3]:
    print(" ", t.to_dict())
print("legs:", r1.get('leg_structure'))

print("\n=== Option: iron_condor on NIFTY ===")
r2 = runner.run_derivative(
    strategy_name='iron_condor',
    underlying='NIFTY',
    kind='IDX',
    params={'num_lots': 1},
    timeframe='daily',
    lookback_days=400,
    initial_capital=500000,
    commission_pct=0.03,
)
print("metrics:", r2.get('metrics'))

print("\n=== Futures: futures_trend on NIFTY ===")
r3 = runner.run_derivative(
    strategy_name='futures_trend',
    underlying='NIFTY',
    kind='IDX',
    params={},
    timeframe='daily',
    lookback_days=400,
    initial_capital=500000,
    commission_pct=0.03,
)
print("metrics:", r3.get('metrics'))
print("num trades:", len(r3.get('trades', [])))

print("\n=== Visualiser sanity (r1) ===")
viz = BacktestVisualizer(r1)
print("equity chart traces:", len(viz.equity_chart().data))
print("OK")