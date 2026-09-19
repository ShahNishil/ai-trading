"""Trailing stops must ratchet off the high-water mark and route real orders."""
import sys
import types

# The broker SDK is not needed to exercise position management; stub it so the
# suite runs offline without dhanhq installed.
if "dhanhq" not in sys.modules:
    _stub = types.ModuleType("dhanhq")
    _stub.DhanContext = object
    _stub.dhanhq = object
    sys.modules["dhanhq"] = _stub

from core.risk_manager import RiskManager
from trading.engine import TradingEngine

CFG = {"auto_trade": {"stop_loss_pct": 2.0, "target_pct": 4.0, "trailing_stop_pct": 1.5}}
# Target pushed out of reach so the trailing stop is the only thing that can fire.
CFG_TRAIL = {"auto_trade": {"stop_loss_pct": 2.0, "target_pct": 500.0, "trailing_stop_pct": 1.5}}


class _StubOrders:
    def __init__(self, status="FILLED"):
        self.status, self.placed = status, []

    def place_order(self, **kw):
        self.placed.append(kw)
        return {"status": self.status, "filled_price": kw.get("price"),
                "error": "rejected" if self.status == "ERROR" else None}


class _StubPortfolio:
    def __init__(self, positions):
        self._pos, self.closed = positions, []

    def get_open_positions(self):
        return [p for p in self._pos if p["trade_id"] not in [c[0] for c in self.closed]]

    def close_trade(self, trade_id, exit_price, exit_reason=""):
        self.closed.append((trade_id, exit_price, exit_reason))
        return {"trade_id": trade_id, "exit_price": exit_price, "reason": exit_reason}


def _engine(positions, status="FILLED", cfg=CFG):
    e = object.__new__(TradingEngine)
    e.config, e.mode, e.dhan = cfg, "paper", None
    e.orders, e.portfolio = _StubOrders(status), _StubPortfolio(positions)
    e.risk = RiskManager(cfg, portfolio=e.portfolio)
    e._peak_price = {}
    return e


def _long():
    return [{"trade_id": "t1", "symbol": "ACME", "side": "BUY",
             "quantity": 10, "entry_price": 100.0, "security_id": "1"}]


def test_stop_ratchets_up_with_price():
    e = _engine(_long(), cfg=CFG_TRAIL)
    for p in [100, 105, 110, 120]:            # run up, no exit
        e.update_positions_with_prices({"ACME": p})
    assert not e.portfolio.closed, "should still be open while trending up"
    # 120 peak -> trailing stop 118.2. A drop to 118 must exit near the peak,
    # NOT ride all the way back to the original 98.00 stop.
    e.update_positions_with_prices({"ACME": 118.0})
    assert e.portfolio.closed, "trailing stop failed to fire after a 120->118 reversal"
    assert e.portfolio.closed[0][2] == "trailing_stop"


def test_old_behaviour_would_have_given_back_the_move():
    """Documents the regression: the pre-fix stop sat at the fixed initial level."""
    rm = RiskManager(CFG)
    assert rm.trailing_stop(100.0, 120.0) == 118.2  # noqa: PLR2004   # correct: off the peak
    # the old call site passed the CURRENT price, which for price==98 gives 98.0
    assert rm.trailing_stop(100.0, 98.0) == 98.0


def test_short_positions_get_stops():
    pos = [{"trade_id": "s1", "symbol": "ACME", "side": "SELL",
            "quantity": 10, "entry_price": 100.0, "security_id": "1"}]
    e = _engine(pos, cfg=CFG_TRAIL)
    e.update_positions_with_prices({"ACME": 90.0})   # favourable, trough=90
    assert not e.portfolio.closed
    e.update_positions_with_prices({"ACME": 91.5})   # 90*1.015 = 91.35 -> stop hit
    assert e.portfolio.closed, "SHORT position never got a trailing stop"


def test_short_exit_sends_buy_order():
    pos = [{"trade_id": "s1", "symbol": "ACME", "side": "SELL",
            "quantity": 10, "entry_price": 100.0, "security_id": "1"}]
    e = _engine(pos)
    e.update_positions_with_prices({"ACME": 96.0})   # target = 96.0
    assert e.orders.placed, "exit must place a real order, not just update the book"
    assert e.orders.placed[0]["side"] == "BUY"


def test_rejected_exit_order_does_not_close_the_book():
    e = _engine(_long(), status="ERROR")
    e.update_positions_with_prices({"ACME": 90.0})   # well through the stop
    assert e.orders.placed, "an exit order should have been attempted"
    assert not e.portfolio.closed, "book was closed despite a REJECTED exit order"
