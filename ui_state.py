"""Shared runtime state between Streamlit pages and background trading threads."""


class _State:
    def __init__(self):
        self.auto_trade_thread = None
        self.auto_trade_stop = None
        self.last_scan = []
        self.last_scan_time = None
        self.last_error = None
        self.ai_engine = None
        self.data_fetcher = None
        self.trading_engine = None
        self.config = None


AutoTradeState = _State()


def init_runtime(config, ai_engine, data_fetcher, trading_engine):
    """Seed shared state with the runtime objects used across pages."""
    AutoTradeState.config = config
    AutoTradeState.ai_engine = ai_engine
    AutoTradeState.data_fetcher = data_fetcher
    AutoTradeState.trading_engine = trading_engine