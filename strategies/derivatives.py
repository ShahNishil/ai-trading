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


class RegimeSwitchStrategy(BaseStrategy):
    """Trend + volatility regime strategy for autonomous F&O trading.

    Unlike the pure structures above (which are entered on a fixed schedule),
    this is a bar-based strategy: it watches the underlying spot series and
    emits directional BUY/SELL/HOLD signals the auto-trader can translate
    into an option structure or a futures position:

    - Trending up   -> BUY  (bullish structure / futures long)
    - Trending down -> SELL (bearish structure / futures short)
    - Range-bound   -> HOLD (skip; avoids structure decay)

    In a high-volatility regime the signal prefers a long straddle (direction
    agnostic) because both legs thrive on large moves.
    """

    name = "regime_switch"
    description = "Regime switch: ADX trend + EMA stack + volatility → option structure or futures"
    market = "auto"
    default_params = {
        "adx_threshold": 20.0,        # ADX below this => range-bound (HOLD)
        "high_vol_natr": 1.5,         # NATR % above this => high-volatility regime
        "structure_bull": "bull_call_spread",
        "structure_bear": "bear_put_spread",
        "structure_high_vol": "long_straddle",
        "min_confidence": 0.6,
    }

    def generate_signal(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 60:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Not enough data",
                    "structure": "", "vol_regime": ""}
        last = df.iloc[-1]
        close = float(last.get("close", 0) or 0)
        ema21 = last.get("ema21")
        ema50 = last.get("ema50")
        rsi = float(last.get("rsi", 50) or 50)
        adx = last.get("adx")
        natr = float(last.get("natr", 0) or 0)
        macd_hist = last.get("macd_hist", 0)

        if pd.isna(ema21) or pd.isna(ema50) or close <= 0:
            return {"action": "HOLD", "confidence": 0.0, "reason": "Missing indicator columns",
                    "structure": "", "vol_regime": ""}

        high_vol = natr >= float(self.params.get("high_vol_natr", 1.5))
        vol_regime = "high" if high_vol else "low"

        adx = float(adx) if pd.notna(adx) else 0.0
        threshold = float(self.params.get("adx_threshold", 20.0))
        if adx < threshold:
            return {
                "action": "HOLD", "confidence": 0.45,
                "reason": f"Range-bound (ADX {adx:.1f} < {threshold:.0f}) — {vol_regime} vol",
                "structure": "iron_condor", "vol_regime": vol_regime,
            }

        uptrend = close > ema21 > ema50
        downtrend = close < ema21 < ema50
        macd_ok = pd.notna(macd_hist) and (float(macd_hist) > 0 if uptrend else float(macd_hist) < 0)

        if not uptrend and not downtrend:
            return {"action": "HOLD", "confidence": 0.4,
                    "reason": f"Trending (ADX {adx:.1f}) but EMA stack flat",
                    "structure": "", "vol_regime": vol_regime}

        base = 0.7 if adx >= 30 else 0.6
        confidence = min(0.95, base + (0.1 if macd_ok else 0.0))
        rsi_ok = (rsi < 80) if uptrend else (rsi > 20)
        if not rsi_ok:
            confidence -= 0.05

        action = "BUY" if uptrend else "SELL"
        if high_vol:
            structure = str(self.params.get("structure_high_vol", "long_straddle"))
        else:
            structure = str(self.params.get("structure_bull" if uptrend else "structure_bear", ""))
        reason = (
            f"{action} regime: ADX {adx:.1f}, close {'>' if uptrend else '<'} ema21 "
            f"{'<' if uptrend else '>'} ema50, RSI {rsi:.1f}, {vol_regime} vol "
            f"(NATR {natr:.2f}%)"
        )
        return {
            "action": action,
            "confidence": round(max(0.5, min(confidence, 1.0)), 3),
            "reason": reason,
            "structure": structure,
            "vol_regime": vol_regime,
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

AUTO_STRATEGIES: Dict[str, type] = {
    RegimeSwitchStrategy.name: RegimeSwitchStrategy,
}

ALL_DERIVATIVE_STRATEGIES: Dict[str, type] = {**OPTION_STRATEGIES, **FUTURES_STRATEGIES, **AUTO_STRATEGIES}


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