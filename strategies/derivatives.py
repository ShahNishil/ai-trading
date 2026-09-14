"""Derivative (F&O) strategies.

Two families of registered strategies:

- **Futures trend** — trades the futures contract with classic momentum rules
  (reuses the `momentum` signal on the futures/underlying series).
- **Option structures** — defined multi-leg positions (straddle, strangle,
  vertical spreads, iron condor) entered N trading days before expiry. Pricing
  and mark-to-market are done with the Black-Scholes model, which makes them
  fully backtestable with nothing but underlying spot data.

Each strategy implements `build_legs(spot, strike_step, params)` returning a
list of legs: `{option_type, strike, side}` where side is "BUY" or "SELL".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from strategies.momentum import MomentumStrategy


@dataclass
class OptionLeg:
    option_type: str  # CE | PE
    strike: float
    side: str  # BUY | SELL


@dataclass
class OptionStructure:
    name: str
    description: str
    legs: List[OptionLeg] = field(default_factory=list)

    def leg_labels(self) -> List[str]:
        return [
            f"{'+ ' if leg.side == 'BUY' else '- '}{leg.option_type} {leg.strike:,.0f}"
            for leg in self.legs
        ]


def _atm_strike(spot: float, step: float, ceiling: bool = True) -> float:
    if step <= 0:
        return spot
    if ceiling:
        return math.ceil(spot / step) * step
    return math.floor(spot / step) * step


def _strike_at(anchor: float, step: float, n: int) -> float:
    return round(anchor + n * step, 2)


class OptionStrategy(BaseStrategy):
    """Base for multi-leg option structures.

    These strategies are executed by the derivatives backtest engine and the
    trading engine, not by the cash `on_bar` loop, so `generate_signal` is a
    no-op placeholder that satisfies the abstract base class.
    """

    name = "option"
    description = "Multi-leg option structure"
    market = "option"
    default_params = {}

    def generate_signal(self, df: pd.DataFrame) -> dict:
        return {"action": "HOLD", "confidence": 0.0, "reason": "Option structure — not a bar signal"}


# ----------------------------------------------------------------------
# Option structures
# ----------------------------------------------------------------------
class LongStraddle(OptionStrategy):
    name = "long_straddle"
    description = "Buy ATM call + ATM put — profit from large moves either way"
    market = "option"
    default_params = {
        "entry_days_before_expiry": 5,
        "target_mult": 1.5,      # exit at x the premium paid
        "stop_loss_pct": 50.0,   # exit if position loses this % of premium
        "num_lots": 1,
        "exit_days_before_expiry": 0,
    }

    def build_legs(self, spot: float, strike_step: float, params: dict = None) -> OptionStructure:
        p = {**self.default_params, **(params or {})}
        atm = _atm_strike(spot, strike_step)
        return OptionStructure(
            name=self.name,
            description=self.description,
            legs=[
                OptionLeg("CE", atm, "BUY"),
                OptionLeg("PE", atm, "BUY"),
            ],
        )


class LongStrangle(OptionStrategy):
    name = "long_strangle"
    description = "Buy OTM call + OTM put — cheaper than straddle, needs bigger move"
    market = "option"
    default_params = {
        "entry_days_before_expiry": 7,
        "otm_strikes": 1,
        "target_mult": 1.5,
        "stop_loss_pct": 50.0,
        "num_lots": 1,
        "exit_days_before_expiry": 0,
    }

    def build_legs(self, spot: float, strike_step: float, params: dict = None) -> OptionStructure:
        p = {**self.default_params, **(params or {})}
        atm_c = _atm_strike(spot, strike_step, ceiling=True)
        atm_p = _atm_strike(spot, strike_step, ceiling=False)
        otm = int(p.get("otm_strikes", 1))
        return OptionStructure(
            name=self.name,
            description=self.description,
            legs=[
                OptionLeg("CE", _strike_at(atm_c, strike_step, otm), "BUY"),
                OptionLeg("PE", _strike_at(atm_p, strike_step, -otm), "BUY"),
            ],
        )


class BullCallSpread(OptionStrategy):
    name = "bull_call_spread"
    description = "Long ATM call + short OTM call — capped-reward bullish view"
    market = "option"
    default_params = {
        "entry_days_before_expiry": 7,
        "width_strikes": 2,
        "target_mult": 0.0,  # hold to expiry by default
        "stop_loss_pct": 50.0,
        "num_lots": 1,
        "exit_days_before_expiry": 0,
    }

    def build_legs(self, spot: float, strike_step: float, params: dict = None) -> OptionStructure:
        p = {**self.default_params, **(params or {})}
        atm = _atm_strike(spot, strike_step)
        width = int(p.get("width_strikes", 2))
        return OptionStructure(
            name=self.name,
            description=self.description,
            legs=[
                OptionLeg("CE", atm, "BUY"),
                OptionLeg("CE", _strike_at(atm, strike_step, width), "SELL"),
            ],
        )


class BearPutSpread(OptionStrategy):
    name = "bear_put_spread"
    description = "Long ATM put + short OTM put — capped-reward bearish view"
    market = "option"
    default_params = {
        "entry_days_before_expiry": 7,
        "width_strikes": 2,
        "target_mult": 0.0,
        "stop_loss_pct": 50.0,
        "num_lots": 1,
        "exit_days_before_expiry": 0,
    }

    def build_legs(self, spot: float, strike_step: float, params: dict = None) -> OptionStructure:
        p = {**self.default_params, **(params or {})}
        atm = _atm_strike(spot, strike_step)
        width = int(p.get("width_strikes", 2))
        return OptionStructure(
            name=self.name,
            description=self.description,
            legs=[
                OptionLeg("PE", atm, "BUY"),
                OptionLeg("PE", _strike_at(atm, strike_step, -width), "SELL"),
            ],
        )


class IronCondor(OptionStrategy):
    name = "iron_condor"
    description = "Sell OTM strangle + far-OUT protections — profits when market stays in range"
    market = "option"
    default_params = {
        "entry_days_before_expiry": 7,
        "short_offset_strikes": 2,
        "width_strikes": 1,
        "target_mult": 0.0,
        "stop_loss_pct": 35.0,
        "num_lots": 1,
        "exit_days_before_expiry": 0,
    }

    def build_legs(self, spot: float, strike_step: float, params: dict = None) -> OptionStructure:
        p = {**self.default_params, **(params or {})}
        atm_c = _atm_strike(spot, strike_step, ceiling=True)
        atm_p = _atm_strike(spot, strike_step, ceiling=False)
        off = int(p.get("short_offset_strikes", 2))
        w = int(p.get("width_strikes", 1))
        return OptionStructure(
            name=self.name,
            description=self.description,
            legs=[
                OptionLeg("PE", _strike_at(atm_p, strike_step, -off), "SELL"),
                OptionLeg("PE", _strike_at(atm_p, strike_step, -(off + w)), "BUY"),
                OptionLeg("CE", _strike_at(atm_c, strike_step, off), "SELL"),
                OptionLeg("CE", _strike_at(atm_c, strike_step, off + w), "BUY"),
            ],
        )


# ----------------------------------------------------------------------
# Futures strategy
# ----------------------------------------------------------------------
class FuturesTrendStrategy(MomentumStrategy):
    name = "futures_trend"
    description = "Trend-following on derivative futures using EMA/MACD/RSI momentum"
    market = "futures"
    default_params = {
        "fast_ema": 9,
        "slow_ema": 21,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    }


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------
OPTION_STRATEGIES: Dict[str, type] = {
    cls.name: cls
    for cls in (
        LongStraddle,
        LongStrangle,
        BullCallSpread,
        BearPutSpread,
        IronCondor,
    )
}

FUTURES_STRATEGIES: Dict[str, type] = {
    FuturesTrendStrategy.name: FuturesTrendStrategy,
}

ALL_DERIVATIVE_STRATEGIES: Dict[str, type] = {**OPTION_STRATEGIES, **FUTURES_STRATEGIES}


def list_derivative_strategies() -> List[dict]:
    return [
        {
            "name": cls.name,
            "description": cls.description,
            "market": cls.market,
            "default_params": cls.default_params,
        }
        for cls in ALL_DERIVATIVE_STRATEGIES.values()
    ]


def create_derivative_strategy(name: str, params: dict = None) -> BaseStrategy:
    if name not in ALL_DERIVATIVE_STRATEGIES:
        raise KeyError(
            f"Unknown derivative strategy: {name}. Available: {list(ALL_DERIVATIVE_STRATEGIES.keys())}"
        )
    cls = ALL_DERIVATIVE_STRATEGIES[name]
    obj = cls(params)
    if name == "futures_trend":
        # reuse the momentum family of signal params
        pass
    return obj


def payoff_table(legs: List[OptionLeg], premium_map: dict) -> dict:
    """Build a payoff preview at expiry for the risk simulator.

    premium_map: {leg_key: entry_premium} keyed by f"{option_type}-{strike}".
    Returns dict with prices, payoffs, max_profit, max_loss, breakevens.
    """
    if not legs or not premium_map:
        return {}
    prices = sorted({l.strike for l in legs})
    lo, hi = prices[0], prices[-1]
    span = (hi - lo) * 2.0
    grid = np.linspace(lo - span, hi + span, 400)
    payoffs = np.zeros_like(grid)
    for leg in legs:
        key = f"{leg.option_type}-{leg.strike}"
        prem = premium_map.get(key, 0.0)
        sign = 1.0 if leg.side == "BUY" else -1.0
        if leg.option_type == "CE":
            intrinsic = np.maximum(grid - leg.strike, 0.0)
        else:
            intrinsic = np.maximum(leg.strike - grid, 0.0)
        payoffs += sign * (intrinsic - prem)
    breakevens = float(grid[np.where(np.diff(np.sign(payoffs)) != 0)].tolist()[0]) if (np.diff(np.sign(payoffs)) != 0).any() else None
    breakevens = []
    idxs = np.where(np.diff(np.sign(payoffs)) != 0)[0]
    for i in idxs:
        a, b = grid[i], grid[i + 1]
        t = abs(payoffs[i]) / (abs(payoffs[i]) + abs(payoffs[i + 1]) + 1e-12)
        breakevens.append(round(float(a + t * (b - a)), 1))
    return {
        "prices": grid.tolist(),
        "payoffs": payoffs.tolist(),
        "max_profit": round(float(payoffs.max()), 2),
        "max_loss": round(float(payoffs.min()), 2),
        "breakevens": sorted(set(breakevens))[:6],
    }