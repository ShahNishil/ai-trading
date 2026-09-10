import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from core.ai_engine import AIEngine
from core.dhan_client import DhanClient
from data.cache import DataCache
from data.fetcher import DataFetcher
from data.universe import StockUniverse
from trading.engine import TradingEngine


def load_config() -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "config.yaml not found. Create it by copying config.yaml from the repo."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def bootstrap(config: dict = None):
    """Initialize all runtime components. Returns (runtime dict)."""
    config = config or load_config()
    try:
        dhan = DhanClient(config)
    except ValueError as e:
        print(f"[warn] {e}")
        dhan = None

    cache = DataCache()
    # Always create DataFetcher — it will use yfinance fallback if Dhan not configured
    data_fetcher = DataFetcher(dhan, cache)
    ai_engine = AIEngine(config)
    trading_engine = TradingEngine(config, dhan=dhan, cache=cache)
    universe = StockUniverse(config)

    return {
        "config": config,
        "dhan": dhan,
        "cache": cache,
        "data_fetcher": data_fetcher,
        "ai_engine": ai_engine,
        "trading_engine": trading_engine,
        "universe": universe,
    }


def main():
    runtime = bootstrap()
    c = runtime["config"]
    mode = c.get("dhan", {}).get("trading_mode", "paper")
    print("=" * 60)
    print("AI Trading System - Indian Stock Markets")
    print("=" * 60)
    print(f"Trading mode: {mode}")
    if runtime["dhan"] is None:
        print("Dhan: NOT CONFIGURED (set DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN in .env)")
    else:
        print("Dhan: connected")
    if runtime["ai_engine"] is not None:
        print(f"AI: {c.get('groq', {}).get('model', 'llama-3.3-70b-versatile')}")
    print("-" * 60)
    print("Launch the dashboard:")
    print("  streamlit run ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()