import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from backtest.derivatives import DerivativeBacktestEngine
rt = bootstrap()
df = rt['data_fetcher'].fetch_daily('^NSEI', '0', days=400)
eng = DerivativeBacktestEngine(500000, 0.03, 0.0, 7.0)
r = eng.run_options(df, 'iron_condor', {'num_lots': 1}, underlying='NIFTY', lot_size=65, strike_step=50)
print('trades:', len(r['trades']))
ec = r['equity_curve']
print('equity rows:', len(ec), 'cols:', list(ec.columns))
print('equity min/max/first/last:', ec['equity'].min(), ec['equity'].max(), ec['equity'].iloc[0], ec['equity'].iloc[-1])
print('non-const eq count:', (ec['equity'].diff() != 0).sum())
t = r['trades'][0]
print('trade0 pnl:', t.pnl, 'entry price:', t.entry_price, 'exit price:', t.exit_price, 'expiry:', t.expiry)
# sample the equity around trade0
print(ec.loc[t.entry_time:t.exit_time].head(3).to_string())
print(ec.loc[t.entry_time:t.exit_time].tail(3).to_string())