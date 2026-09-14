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

    def __init__(self, data_fetcher: DataFetcher, cache: Optional[DataCache] = None, config: dict = None, universe=None):
        self.data = data_fetcher
        self.cache = cache or DataCache()
        self.config = config or {}
        self.deriv = universe

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

    # ------------------------------------------------------------------
    # Derivatives
    # ------------------------------------------------------------------
    def _ensure_deriv_universe(self):
        if self.deriv is None:
            from data.derivatives import DerivativeUniverse

            self.deriv = DerivativeUniverse(config=self.config)
        return self.deriv

    def run_derivative(
        self,
        strategy_name: str,
        underlying: str,
        kind: str = "IDX",
        params: dict = None,
        timeframe: str = "daily",
        lookback_days: int = 365,
        spot_symbol: str = "",
        expiries: list = None,
        iv_pct: float = 0.0,
        lot_size: int = 0,
        strike_step: float = 0.0,
        initial_capital: float = 100000,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.0,
        risk_free_rate_pct: float = 7.0,
        margin_pct: float = 12.0,
        margin_usage_pct: float = 60.0,
    ) -> dict:
        """Run a derivative strategy backtest on underlying spot (or futures) data.

        Data lookup order:
        1. `spot_symbol` if explicitly given (e.g. "^NSEI", "RELIANCE").
        2. The universe-resolved spot ticker for the underlying.
        3. Real futures/option contract history via Dhan when credentials exist
           (uses `fetch_derivative_history`).

        Returns the same shape as `run` plus derivative metadata.
        """
        underlying = underlying.upper()
        kind = kind.upper()
        universe = self._ensure_deriv_universe()
        params = params or {}

        spot = None
        if spot_symbol:
            spot = self.data.fetch_daily(spot_symbol, "0", days=lookback_days)
        if spot is None or spot.empty:
            sym = universe.spot_symbol(underlying, kind)
            if sym:
                spot = self.data.fetch_daily(sym, "0", days=lookback_days)
        if (spot is None or spot.empty) and self.data.dhan is not None:
            # Real contract data
            try:
                from strategies.derivatives import ALL_DERIVATIVE_STRATEGIES

                if ALL_DERIVATIVE_STRATEGIES.get(strategy_name) and getattr(ALL_DERIVATIVE_STRATEGIES[strategy_name], "market", "option") == "futures":
                    inst = universe.futures_contract(underlying, kind)
                else:
                    inst = universe.option_contract(underlying, kind)
                spot = self.data.fetch_derivative_history(inst, days=lookback_days)
                spot.attrs["instrument"] = inst
            except Exception as e:
                return {"error": f"No data for {underlying}: {e}", "trades": [], "equity_curve": []}
        if spot is None or spot.empty:
            return {"error": f"No data available for {underlying}. Provide a spot symbol or configure Dhan.", "trades": [], "equity_curve": []}

        lot_size = lot_size or universe.lot_size(underlying, kind)
        strike_step = strike_step or universe.strike_step(underlying, kind)
        if not expiries:
            exps = universe.expiries(underlying, kind)
            expiries = [e for e in exps if spot.index.min() <= pd.Timestamp(e).normalize() <= spot.index.max() + pd.Timedelta(days=1)] if exps else None

        from backtest.derivatives import run_derivative_backtest

        result = run_derivative_backtest(
            spot,
            strategy_name,
            params=params,
            underlying=underlying,
            lot_size=lot_size,
            strike_step=strike_step,
            expiries=expiries,
            initial_capital=initial_capital,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            iv_pct=iv_pct,
            margin_pct=margin_pct,
            margin_usage_pct=margin_usage_pct,
        )
        result["symbol"] = underlying
        result["strategy"] = strategy_name
        result["params"] = params
        result["timeframe"] = timeframe
        result["price_df"] = spot
        result["kind"] = kind
        result["lot_size"] = lot_size
        result["strike_step"] = strike_step

        # Persist stats for history tracking
        entry = {
            "id": str(uuid.uuid4())[:8],
            "strategy": strategy_name,
            "symbol": underlying,
            "params": str({"params": params, "kind": kind}),
            "start_date": str(spot.index[0].date()) if len(spot) else "",
            "end_date": str(spot.index[-1].date()) if len(spot) else "",
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