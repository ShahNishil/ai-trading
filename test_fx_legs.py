import sys
sys.path.insert(0, r'D:\ai-trading')
from run import bootstrap
from strategies.derivatives import LongStraddle, IronCondor
rt = bootstrap()
print('--- LongStraddle legs ---')
ls = LongStraddle({}).build_legs(24400, 50)
print(ls.leg_labels(), lstrikes := [l.strike for l in ls.legs])
print('--- IronCondor legs ---')
ic = IronCondor({}).build_legs(24400, 50)
print(ic.leg_labels(), [l.strike for l in ic.legs])
for l in ic.legs:
    print(' ', l)
# simulate premium calc
from core.options import bs_price
for l in ic.legs:
    try:
        v = bs_price(24400, l.strike, 5/365, 0.07, 0.12, l.option_type)
        print('  prem', l, v)
    except Exception as e:
        print('  ERR', l, e)