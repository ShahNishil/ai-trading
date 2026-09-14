"""Derivative backtesting engine.

Options are priced with Black-Scholes and marked-to-market bar-by-bar, so an
entire option-strategy backtest can run on nothing but the underlying spot
series (e.g. NIFTY from Yahoo Finance when Dhan is not configured). Futures
mode reuses the core `BacktestEngine` with whole-lot, margin-aware sizing.

The result dict mirrors the equity engine's shape so the existing metrics and
visualiser modules work unchanged: ``trades``, ``equity_curve``, ``metrics``,
``signal_log`` plus derivative metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from core.indicators import IndicatorEngine
from core.options import annualized_volatility, bs_price, settle_intrinsic
from strategies.derivatives import (
    ALL_DERIVATIVE_STRATEGIES,
    create_derivative_strategy,
)


@dataclass
class OptionTrade:
    symbol: str
    side: str
    quantity: int  # units = lot_size * num_lots
    entry_time: pd.Timestamp
    entry_price: float  # net premium paid per unit structure (debit/credit)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    legs: List[dict] = field(default_factory=list)
    entry_date: Optional[date] = None
    expiry: Optional[date] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_time": str(self.entry_time),
            "exit_time": str(self.exit_time) if self.exit_time is not None else "",
            "entry_price": round(float(self.entry_price), 2),
            "exit_price": round(float(self.exit_price or 0.0), 2),
            "entry_date": str(self.entry_date),
            "expiry": str(self.expiry) if self.expiry else "",
            "exit_reason": self.exit_reason,
            "pnl": round(float(self.pnl), 2),
            "legs": [dict(l) for l in self.legs],
        }


def _weekly_expiries(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """Generate Thursday expiry proxies between start and end (approx.)."""
    out = []
    d = start.normalize()
    while d <= end:
        if d.weekday() == 3:
            out.append(d)
        d += timedelta(days=1)
    return out


def _find_entry_bar(dates: Sequence[pd.Timestamp], target: pd.Timestamp) -> Optional[int]:
    """Index of the largest date <= target, or None."""
    arr = np.asarray([pd.Timestamp(x).value for x in dates])
    tgt = pd.Timestamp(target).value
    idx = np.searchsorted(arr, tgt, side="right") - 1
    return int(idx) if idx >= 0 else None


class DerivativeBacktestEngine:
    """Backtests derivative strategies (options via BS MTM, futures via lot engine)."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.0,
        risk_free_rate_pct: float = 7.0,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100.0
        self.slippage_pct = slippage_pct / 100.0
        self.r = risk_free_rate_pct / 100.0

    # ------------------------------------------------------------------
    # Option strategy backtest
    # ------------------------------------------------------------------
    def run_options(
        self,
        spot_df: pd.DataFrame,
        strategy_name: str,
        params: dict = None,
        underlying: str = "NIFTY",
        lot_size: int = 1,
        strike_step: float = 0.0,
        expiries: Optional[Sequence[datetime]] = None,
        iv_pct: float = 0.0,
        skip_initial_n: int = 0,
    ) -> dict:
        """Run a multi-leg option strategy on the underlying spot series.

        Parameters
        ----------
        spot_df : DataFrame (index datetime, needs 'close' column).
        strategy_name : one of ALL_DERIVATIVE_STRATEGIES (option family).
        params : strategy params (entry_days_before_expiry, num_lots, ...).
        expiries : list of expiry dates to trade. When None, weekly Thursdays
            are synthesised across the data range.
        iv_pct : implied vol used for pricing (0 -> realised vol from spot).
        """
        if spot_df is None or spot_df.empty:
            return {"error": "No underlying data", "trades": [], "equity_curve": []}
        if strategy_name not in ALL_DERIVATIVE_STRATEGIES:
            return {"error": f"Unknown option strategy: {strategy_name}", "trades": [], "equity_curve": []}
        cls = ALL_DERIVATIVE_STRATEGIES[strategy_name]
        if getattr(cls, "market", "option") != "option":
            return {"error": f"{strategy_name} is not an option strategy", "trades": [], "equity_curve": []}

        params = params or {}
        strategy = create_derivative_strategy(strategy_name, params)
        entry_days = int(params.get("entry_days_before_expiry", strategy.default_params.get("entry_days_before_expiry", 5)))
        exit_days_before = int(params.get("exit_days_before_expiry", strategy.default_params.get("exit_days_before_expiry", 0)))
        target_mult = float(params.get("target_mult", strategy.default_params.get("target_mult", 0.0)))
        stop_loss_pct = float(params.get("stop_loss_pct", strategy.default_params.get("stop_loss_pct", 50.0)))
        num_lots = int(params.get("num_lots", strategy.default_params.get("num_lots", 1)))
        lot_size = int(lot_size) or 1

        df = spot_df.copy()
        if "close" not in df.columns:
            return {"error": "Underlying data must contain a 'close' column", "trades": [], "equity_curve": []}
        if skip_initial_n:
            df = df.iloc[skip_initial_n:]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        dates = list(df.index)

        start, end = df.index[0], df.index[-1]
        if expiries:
            exps = sorted(pd.Timestamp(e) for e in expiries if start <= pd.Timestamp(e).normalize() <= end + timedelta(days=1))
        else:
            exps = _weekly_expiries(start, end + timedelta(days=7))
        if not exps:
            return {"error": "No expiries in data range", "trades": [], "equity_curve": []}

        cash = self.initial_capital
        trades: List[OptionTrade] = []
        equity_curve: List[dict] = []
        signal_log: List[dict] = []
        open_pos: Optional[OptionTrade] = None
        realized = 0.0

        def sigma_at(i: int) -> float:
            if iv_pct and iv_pct > 0:
                return iv_pct / 100.0
            window = df["close"].iloc[max(0, i - 20): i + 1]
            vol = annualized_volatility(window.values) / 100.0
            if not np.isfinite(vol) or vol <= 0.02:
                return 0.12  # sensible floor for fresh data
            return float(vol)

        def price_leg(opt_type: str, strike: float, spot: float, T: float, sigma: float) -> float:
            if T <= 0:
                return settle_intrinsic(spot, strike, opt_type)
            return float(bs_price(spot, strike, T, self.r, sigma, opt_type))

        def structure_value(i: int, trade: OptionTrade, T: float, sigma: float):
            spot = float(df["close"].iloc[i])
            total = 0.0
            for leg in trade.legs:
                v = price_leg(leg["option_type"], leg["strike"], spot, T, sigma)
                total += (1.0 if leg["side"] == "BUY" else -1.0) * v
            return total * trade.quantity

        def structure_entry_cost(trade: OptionTrade, sigma: float, i: int):
            spot = float(df["close"].iloc[i])
            entry = pd.Timestamp(trade.entry_time)
            T = max((trade.expiry - entry.date()).days, 0) / 365.0
            total = 0.0
            for leg in trade.legs:
                v = price_leg(leg["option_type"], leg["strike"], spot, T, sigma)
                leg["entry_premium"] = v
                total += (1.0 if leg["side"] == "BUY" else -1.0) * v
            return total

        # Prepare schedule of (entry_bar_index, expiry)
        schedule = []
        for e in exps:
            target = e.normalize() - timedelta(days=entry_days)
            if pd.isna(target) or target < start or target > end:
                continue
            ib = _find_entry_bar(dates, target)
            if ib is None:
                continue
            exp_date = e.date()
            eb = _find_entry_bar(dates, pd.Timestamp(exp_date) + timedelta(days=1)) or (len(dates) - 1)
            schedule.append((ib, min(eb, len(dates) - 1), exp_date))

        for entry_i, last_i, exp_date in schedule:
            # Sequential: no overlapping positions
            if open_pos is not None:
                continue
            # Need a warm-up window for realised-vol estimation and stable pricing
            if entry_i < 20 or entry_i >= len(dates) - 1:
                continue
            if df.index[entry_i].date() >= exp_date:
                continue
            spot_entry = float(df["close"].iloc[entry_i])
            sigma = sigma_at(entry_i)
            structure = strategy.build_legs(spot_entry, strike_step, params)
            if not structure.legs:
                continue

            trade = OptionTrade(
                symbol=underlying,
                side=strategy.name.upper(),
                quantity=lot_size * num_lots,
                entry_time=df.index[entry_i],
                entry_price=0.0,
                entry_date=df.index[entry_i].date(),
                expiry=exp_date,
                legs=[
                    {
                        "option_type": l.option_type,
                        "strike": float(l.strike),
                        "side": l.side,
                        "label": f"{l.side} {l.option_type} {l.strike:,.0f}",
                    }
                    for l in structure.legs
                ],
            )
            net_entry = structure_entry_cost(trade, sigma, entry_i)
            trade.entry_price = net_entry  # per unit (times quantity below)
            if net_entry is None or not math.isfinite(float(net_entry)):
                continue
            entry_notional = abs(net_entry) * trade.quantity
            entry_comm = entry_notional * self.commission_pct
            net_debit = net_entry * trade.quantity
            if net_debit + entry_comm > cash:
                # Not enough capital for this structure — skip cycle
                continue

            cash -= net_debit + entry_comm
            open_pos = trade
            signal_log.append(
                {"time": df.index[entry_i], "signal": strategy.name.upper(), "confidence": 1.0, "entry_price": round(net_entry, 2)}
            )

            # Simulate forward from entry
            for i in range(entry_i, last_i + 1):
                bar_ts = df.index[i]
                days_to = (exp_date - bar_ts.date()).days
                if days_to <= 0:
                    T = 0.0
                else:
                    T = days_to / 365.0
                sig = sigma_at(i)
                current_val = structure_value(i, open_pos, T, sig)
                equity = cash + current_val
                equity_curve.append({"timestamp": bar_ts, "equity": equity, "price": float(df["close"].iloc[i])})

                pnl = current_val - open_pos.entry_price * open_pos.quantity
                debit = open_pos.entry_price if open_pos.entry_price > 0 else 0.0
                credit = -open_pos.entry_price if open_pos.entry_price < 0 else 0.0
                exit_now = False
                reason = ""

                if T <= 0:
                    exit_now, reason = True, "expiry_settlement"
                elif exit_days_before > 0 and days_to <= exit_days_before:
                    exit_now, reason = True, f"exit_{exit_days_before}d_before_expiry"
                elif target_mult > 0 and debit > 0 and pnl >= target_mult * debit * open_pos.quantity:
                    exit_now, reason = True, "target_hit"
                elif target_mult > 0 and credit > 0 and pnl >= target_mult * credit * open_pos.quantity:
                    exit_now, reason = True, "target_hit"
                elif stop_loss_pct > 0:
                    if debit > 0 and pnl <= -debit * open_pos.quantity * (stop_loss_pct / 100.0):
                        exit_now, reason = True, "stop_loss"
                    elif credit > 0 and pnl <= -2.0 * credit * open_pos.quantity:
                        # credit structures risk multiples of the credit collected
                        exit_now, reason = True, "stop_loss"

                if i == last_i:
                    exit_now, reason = True, reason or "end_of_data"

                if exit_now:
                    exit_val = current_val
                    exit_comm = abs(exit_val) * self.commission_pct
                    cash += exit_val - exit_comm
                    open_pos.exit_time = bar_ts
                    open_pos.exit_price = exit_val / open_pos.quantity
                    open_pos.pnl = exit_val - open_pos.entry_price * open_pos.quantity - entry_comm - exit_comm
                    open_pos.exit_reason = reason
                    trades.append(open_pos)
                    realized += open_pos.pnl
                    signal_log.append({"time": bar_ts, "signal": "CLOSE", "confidence": 1.0, "reason": reason})
                    open_pos = None
                    break

            if open_pos is not None:
                # safety: should not normally happen (last_i forces close)
                trades.append(open_pos)
                open_pos = None

    # Flat tail: forward-fill equity on dates with no active position
        eq_series = pd.Series({r["timestamp"]: r["equity"] for r in equity_curve})
        full_eq = eq_series.reindex(dates).ffill().bfill()
        if full_eq.isna().all():
            full_eq = pd.Series(self.initial_capital, index=dates)
        full_eq = full_eq.fillna(self.initial_capital)
        equity_df = pd.DataFrame({"equity": full_eq.values, "price": df["close"].values}, index=dates)

        from backtest.engine import compute_metrics

        metrics = compute_metrics(equity_df, trades, self.initial_capital, df)
        metrics["num_trades"] = len(trades)
        return {
            "trades": trades,
            "equity_curve": equity_df,
            "metrics": metrics,
            "signal_log": signal_log,
            "final_cash": round(cash, 2),
            "strategy": strategy_name,
            "instrument_type": "option",
            "underlying": underlying,
            "leg_structure": [
                {
                    "legs": create_derivative_strategy(strategy_name, params).build_legs(
                        float(df["close"].iloc[0]), strike_step, params
                    ).leg_labels(),
                    "label": strategy.description,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Futures backtest
    # ------------------------------------------------------------------
    def run_futures(
        self,
        df: pd.DataFrame,
        strategy_name: str = "futures_trend",
        params: dict = None,
        underlying: str = "NIFTY",
        lot_size: int = 1,
        margin_pct: float = 12.0,
        margin_usage_pct: float = 60.0,
    ) -> dict:
        """Run a futures strategy with whole-lot, margin-aware re-sizing."""
        if df is None or df.empty:
            return {"error": "No data for symbol", "trades": [], "equity_curve": []}
        strategy = create_derivative_strategy(strategy_name, params)
        indicator_engine = IndicatorEngine(df)
        enriched = indicator_engine.compute_all()
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(
            self.initial_capital,
            self.commission_pct * 100,
            self.slippage_pct * 100,
            lot_size=lot_size,
            margin_pct=margin_pct,
            margin_usage_pct=margin_usage_pct,
        )
        result = engine.run(enriched, strategy, symbol=underlying, skip_initial_n=60)
        result["strategy"] = strategy_name
        result["instrument_type"] = "future"
        result["underlying"] = underlying
        return result


def run_derivative_backtest(
    spot_df: pd.DataFrame,
    strategy_name: str,
    params: dict = None,
    underlying: str = "NIFTY",
    lot_size: int = 1,
    strike_step: float = 0.0,
    expiries: Optional[Sequence[datetime]] = None,
    initial_capital: float = 100000,
    commission_pct: float = 0.03,
    slippage_pct: float = 0.0,
    risk_free_rate_pct: float = 7.0,
    iv_pct: float = 0.0,
    margin_pct: float = 12.0,
    margin_usage_pct: float = 60.0,
) -> dict:
    """Dispatch derivative backtest: option strategies vs futures trend."""
    engine = DerivativeBacktestEngine(initial_capital, commission_pct, slippage_pct, risk_free_rate_pct)
    cls = ALL_DERIVATIVE_STRATEGIES.get(strategy_name)
    market = getattr(cls, "market", "option") if cls else "option"
    if market == "futures":
        return engine.run_futures(
            spot_df, strategy_name, params, underlying, lot_size, margin_pct, margin_usage_pct
        )
    return engine.run_options(
        spot_df,
        strategy_name,
        params,
        underlying=underlying,
        lot_size=lot_size,
        strike_step=strike_step,
        expiries=expiries,
        iv_pct=iv_pct,
    )
