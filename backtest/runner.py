import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from backtest.engine import run_backtest
from data.cache import DataCache
from data.fetcher import DataFetcher
from strategies.registry import create_strategy


class BacktestRunner:
    """Ties together data fetching, strategy execution and result persistence."""

    def __init__(self, data_fetcher: DataFetcher, cache: Optional[DataCache] = None):
        self.data = data_fetcher
        self.cache = cache or DataCache()

    def run(
        self,
        strategy_name: str,
        symbol: str,
        security_id: str,
        params: dict = None,
        timeframe: str = "daily",
        lookback_days: int = 365,
        initial_capital: float = 100000,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.05,
    ) -> dict:
        strategy = create_strategy(strategy_name, params)

        if timeframe == "daily":
            df = self.data.fetch_daily(symbol, security_id, days=lookback_days)
        else:
            interval = int(timeframe.replace("min", ""))
            df = self.data.fetch_intraday(symbol, security_id, interval, days=lookback_days)

        if df is None or df.empty:
            return {"error": "No data for symbol"}

        result = run_backtest(
            strategy,
            df,
            symbol=symbol,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
        )
        result["symbol"] = symbol
        result["strategy"] = strategy_name
        result["params"] = params or {}
        result["timeframe"] = timeframe
        result["price_df"] = df

        # Persist stats for history tracking
        entry = {
            "id": str(uuid.uuid4())[:8],
            "strategy": strategy_name,
            "symbol": symbol,
            "params": str(params or {}),
            "start_date": str(df.index[0].date()) if len(df) else "",
            "end_date": str(df.index[-1].date()) if len(df) else "",
            "timeframe": timeframe,
            "metrics": str(result.get("metrics", {})),
            "created_at": datetime.now().isoformat(),
        }
        cur = self.cache._conn.cursor()
        cur.execute(
            """
            INSERT INTO backtests (id, strategy, symbol, params, start_date, end_date, timeframe, metrics, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                entry["id"], entry["strategy"], entry["symbol"], entry["params"],
                entry["start_date"], entry["end_date"], entry["timeframe"],
                entry["metrics"], entry["created_at"],
            ),
        )
        self.cache._conn.commit()

        return result

    def strategy_optimize(self, strategy_name: str, symbol: str, security_id: str, param_grid: dict, **kwargs) -> list:
        """Simple grid-search optimization over param_grid."""
        results = []
        import itertools

        if not param_grid:
            return [self.run(strategy_name, symbol, security_id, {}, **kwargs)]

        keys = list(param_grid.keys())
        combos = list(itertools.product(*param_grid.values()))
        for combo in combos:
            params = dict(zip(keys, combo))
            res = self.run(strategy_name, symbol, security_id, params, **kwargs)
            metrics = res.get("metrics", {})
            results.append(
                {
                    "params": params,
                    "total_return_pct": metrics.get("total_return_pct", 0),
                    "sharpe": metrics.get("sharpe", 0),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                    "win_rate_pct": metrics.get("win_rate_pct", 0),
                }
            )
        results.sort(key=lambda r: r["sharpe"], reverse=True)
        return results