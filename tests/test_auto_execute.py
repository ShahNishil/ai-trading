"""AutoTradeAgent must execute on real market prices, never LLM-supplied ones."""
import sys
import types

import pandas as pd

if "dhanhq" not in sys.modules:
    _stub = types.ModuleType("dhanhq")
    _stub.DhanContext = object
    _stub.dhanhq = object
    sys.modules["dhanhq"] = _stub
if "openai" not in sys.modules:
    _stub = types.ModuleType("openai")
    _stub.OpenAI = object
    sys.modules["openai"] = _stub

from agents.auto_trade_agent import AutoTradeAgent
from core.risk_manager import RiskManager

CFG = {"auto_trade": {"min_confidence": 0.7, "risk_per_trade_pct": 2.0,
                      "stop_loss_pct": 2.0, "target_pct": 4.0, "trailing_stop_pct": 1.5}}


class _Data:
    def __init__(self, quote=500.0):
        self.quote = quote

    def fetch_quote(self, security_id, exchange="NSE_EQ"):
        return self.quote

    def fetch_daily(self, *a, **k):
        return pd.DataFrame()


class _Engine:
    def __init__(self):
        self.entered = []
        self.risk = RiskManager(CFG)
        self.portfolio = types.SimpleNamespace(
            get_open_positions=lambda: [], get_open_position=lambda s: None
        )
        self.stop_calls = []

    def get_capital(self):
        return 100000.0

    def enter_position(self, **kw):
        self.entered.append(kw)
        return {"success": True, **kw}

    def update_positions_with_prices(self, prices, security_ids=None):
        self.stop_calls.append((prices, security_ids))
        return [{"symbol": "ACME", "reason": "trailing_stop", "result": {"success": True}}]


def _agent(data=None, engine=None):
    a = object.__new__(AutoTradeAgent)
    a.ai, a.data, a.engine, a.config = None, data or _Data(), engine or _Engine(), CFG
    a.min_confidence, a.scan_interval, a._running = 0.7, 300, False
    return a


def test_entry_uses_market_price_when_llm_says_zero():
    """Pre-fix: LLM price 0 -> paper order FILLED at 0, sizing assumed a Rs 100 stock."""
    eng = _Engine()
    a = _agent(engine=eng)
    res = a.execute({"symbol": "ACME", "decision": "ENTRY", "confidence": 0.9,
                     "price": 0, "quantity": 0, "side": "BUY"},
                    watchlist_item={"security_id": "123"})
    assert res["success"], res
    assert eng.entered[0]["entry_price"] == 500.0
    # 2% of 100k = 2000 risk; 2% stop on a 500 stock = 10/share -> 200 shares
    assert eng.entered[0]["quantity"] == 200


def test_entry_rejected_when_no_price_exists():
    a = _agent(data=types.SimpleNamespace(
        fetch_quote=lambda *a, **k: None, fetch_daily=lambda *a, **k: pd.DataFrame()))
    res = a.execute({"symbol": "ACME", "decision": "ENTRY", "confidence": 0.9,
                     "price": 999, "quantity": 5, "side": "BUY"},
                    watchlist_item={"security_id": "123"})
    assert not res["success"]
    assert "price" in res["reason"].lower()


def test_llm_quantity_can_only_shrink_the_position():
    eng = _Engine()
    a = _agent(engine=eng)
    a.execute({"symbol": "ACME", "decision": "ENTRY", "confidence": 0.9,
               "price": 500, "quantity": 5, "side": "BUY"},
              watchlist_item={"security_id": "123"})
    assert eng.entered[0]["quantity"] == 5  # LLM asked for less: honoured
    a.execute({"symbol": "ACME", "decision": "ENTRY", "confidence": 0.9,
               "price": 500, "quantity": 99999, "side": "BUY"},
              watchlist_item={"security_id": "123"})
    assert eng.entered[1]["quantity"] == 200  # LLM asked for more: capped by risk


def test_scan_once_enforces_stops_before_new_entries():
    eng = _Engine()
    eng.portfolio = types.SimpleNamespace(
        get_open_positions=lambda: [{"symbol": "ACME", "trade_id": "t1"}],
        get_open_position=lambda s: None,
    )
    a = _agent(engine=eng)
    results = a.scan_once([])
    assert eng.stop_calls, "scan_once never ran stop management"
    assert results and results[0]["decision"] == "RISK_EXIT"
