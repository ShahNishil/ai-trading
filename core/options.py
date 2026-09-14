"""Black-Scholes option pricing, Greeks and implied volatility.

Vectorised with numpy so whole option chains and backtests can be priced
bar-by-bar without Python-level loops.

Market conventions used by this project (Indian F&O):
- Premiums are quoted per underlying unit (INR), contract value = premium * lot size.
- T = calendar days to expiry / 365.
- Risk-free rate default 7% (configurable).
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np

Number = Union[float, int, np.ndarray]


# ----------------------------------------------------------------------
# Normal CDF / PDF
# ----------------------------------------------------------------------
def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(x / math.sqrt(2.0)))


def _phi_density(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ----------------------------------------------------------------------
# Core pricers
# ----------------------------------------------------------------------
def d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        raise ValueError("T must be > 0 and sigma > 0 for Black-Scholes pricing")
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = float(r)
    sigma = float(sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type: str = "CE"):
    """Black-Scholes European option price.

    option_type: "CE" call or "PE" put. Accepts arrays for S, K, T.
    """
    opt = option_type.upper()
    if opt not in ("CE", "PE"):
        raise ValueError("option_type must be 'CE' or 'PE'")
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    d1, d2 = d1_d2(S, K, T, r, sigma)
    if opt == "CE":
        price = S * _phi(d1) - K * np.exp(-r * T) * _phi(d2)
    else:
        price = K * np.exp(-r * T) * _phi(-d2) - S * _phi(-d1)
    return np.clip(price, 0.0, None)


def intrinsic_value(S, K, option_type: str = "CE"):
    opt = option_type.upper()
    if opt == "CE":
        return np.maximum(np.asarray(S, dtype=float) - float(K), 0.0)
    if opt == "PE":
        return np.maximum(float(K) - np.asarray(S, dtype=float), 0.0)
    raise ValueError("option_type must be 'CE' or 'PE'")


def bs_greeks(S, K, T, r, sigma, option_type: str = "CE") -> dict:
    """Return dict of greeks (scalars only)."""
    opt = option_type.upper()
    d1, d2 = d1_d2(S, K, T, r, sigma)
    nd1 = _phi_density(d1)
    if opt == "CE":
        delta = _phi(d1)
        theta_num = -(S * nd1 * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * _phi(d2)
    elif opt == "PE":
        delta = _phi(d1) - 1.0
        theta_num = -(S * nd1 * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * _phi(-d2)
    else:
        raise ValueError("option_type must be 'CE' or 'PE'")
    gamma = nd1 / (S * sigma * math.sqrt(T))
    vega = S * nd1 * math.sqrt(T) / 100.0  # per 1% vol change
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_num / 365.0),  # per calendar day
        "vega": float(vega),
    }


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CE",
    iv_guess: float = 0.15,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> float:
    """Newton-Raphson implied volatility solver."""
    if market_price <= 0 or T <= 0:
        return float(iv_guess)
    iv = float(iv_guess)
    for _ in range(max_iter):
        price = float(bs_price(S, K, T, r, iv, option_type))
        diff = price - market_price
        if abs(diff) < tol:
            return iv
        # Vega per 1.0 (100%) vol move
        d1, _d2 = d1_d2(S, K, T, r, iv)
        vega = float(S * _phi_density(d1) * math.sqrt(T))
        if vega < 1e-12:
            return iv
        step = diff / vega
        iv -= step
        iv = min(max(iv, 0.01), 5.0)
    return iv


def annualized_volatility(closes, window: int = 20) -> float:
    """Realised volatility (annualised, %) from a close-price series.

    Returns 0.0 when there are fewer than 3 return observations so the result
    is never NaN.
    """
    closes = np.asarray(closes, dtype=float)
    if len(closes) < 4:
        return 0.0
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < 3:
        return 0.0
    window = max(3, min(window, len(log_ret)))
    std = np.std(log_ret[-window:], ddof=1)
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(std * math.sqrt(252) * 100.0)


def settle_intrinsic(spot: float, strike: float, option_type: str) -> float:
    return float(intrinsic_value(spot, strike, option_type))


# ----------------------------------------------------------------------
# Spread / strategy helpers
# ----------------------------------------------------------------------
def round_to_strike(price: float, step: float, floor: bool = True) -> float:
    """Round a spot price to the nearest strike step."""
    if step <= 0:
        return price
    if floor:
        return math.floor(price / step) * step
    return math.ceil(price / step) * step