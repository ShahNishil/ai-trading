"""Side-aware P&L: a SHORT that covers lower must book a PROFIT."""
import os
import tempfile

from data.cache import DataCache
from trading.portfolio import Portfolio


def _portfolio():
    path = os.path.join(tempfile.mkdtemp(), "test_cache.db")
    return Portfolio(DataCache(path), mode="paper")


def test_long_pnl_positive_when_price_rises():
    p = _portfolio()
    tid = p.open_trade("ACME", "BUY", 10, 100.0)
    res = p.close_trade(tid, 110.0, "target")
    assert res["pnl"] == 100.0


def test_short_pnl_positive_when_price_falls():
    """Pre-fix: (exit - entry) * qty regardless of side booked this as -100."""
    p = _portfolio()
    tid = p.open_trade("ACME", "SELL", 10, 100.0)
    res = p.close_trade(tid, 90.0, "target")
    assert res["pnl"] == 100.0, f"short cover at a profit booked pnl={res['pnl']}"


def test_short_pnl_negative_when_price_rises():
    p = _portfolio()
    tid = p.open_trade("ACME", "SELL", 10, 100.0)
    res = p.close_trade(tid, 105.0, "stop")
    assert res["pnl"] == -50.0


def test_unrealized_pnl_is_side_aware():
    p = _portfolio()
    p.open_trade("LONGCO", "BUY", 10, 100.0)
    p.open_trade("SHORTCO", "SELL", 10, 100.0)
    upnl = p.get_unrealized_pnl({"LONGCO": 105.0, "SHORTCO": 95.0})
    assert upnl == 100.0, f"both legs are +50 in profit, got {upnl}"
