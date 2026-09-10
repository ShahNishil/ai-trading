from typing import Optional

import yaml

import os


class StockUniverse:
    """Manages the stock watchlists defined in config.yaml."""

    def __init__(self, config: dict):
        self.config = config
        self.watchlists = config.get("watchlists", {})

    def get_watchlist(self, name: str = "NIFTY50") -> list:
        name = name.upper() if name else "NIFTY50"
        return self.watchlists.get(name, [])

    def get_custom_symbols(self, symbols: list) -> list:
        return [{"symbol": s.upper(), "security_id": "", "exchange": "NSE_EQ"} for s in symbols]

    @staticmethod
    def load_config() -> dict:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError("config.yaml not found. See config.yaml.example")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def resolve(self, watchlist: str = "NIFTY50", custom_symbols: list = None) -> list:
        """Resolve a watchlist or custom symbols into a list of instrument dicts."""
        if watchlist and watchlist.upper() != "CUSTOM":
            return self.get_watchlist(watchlist)
        return self.get_custom_symbols(custom_symbols or [])