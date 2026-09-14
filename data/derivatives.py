"""Derivative (F&O) universe and instrument model for Indian markets.

Two data paths are supported:

1. **Real contracts** — from Dhan's public scrip master CSV (SEM_* columns).
   Downstream features (live quote, Dhan historical candles, live orders)
   additionally require Dhan credentials in `.env`.

2. **Synthetic chain** — when the master cannot be fetched (offline) we fall
   back to a small built-in table for index underlyings (NIFTY, BANKNIFTY,
   FINNIFTY, MIDCPNIFTY, NIFTYNXT50) with weekly/monthly expiries and ATM
   strike ladders. Synthetic instruments trade with `security_id=""`,
   which is enough for Black-Scholes based backtesting and strategy
   prototyping but not for live execution.

Underlying names are parsed from Dhan's `SEM_TRADING_SYMBOL` (format
``UNDERLYING-<Exp>-<Strike>-<CE|PE>`` for options and
``UNDERLYING-<Exp>-FUT`` for futures).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import List, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

_SEM_COLUMNS = {
    "SEM_SMST_SECURITY_ID": "security_id",
    "SEM_INSTRUMENT_NAME": "instrument_type",
    "SEM_TRADING_SYMBOL": "trading_symbol",
    "SEM_CUSTOM_SYMBOL": "custom_symbol",
    "SEM_LOT_UNITS": "lot_size",
    "SEM_EXPIRY_DATE": "expiry",
    "SEM_STRIKE_PRICE": "strike",
    "SEM_OPTION_TYPE": "option_type",
    "SEM_TICK_SIZE": "tick_size",
    "SEM_EXPIRY_FLAG": "expiry_flag",
}

_SEM_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

_FNO_INSTRUMENTS = {"FUTIDX", "OPTIDX", "FUTSTK", "OPTSTK"}

# Built-in fallback metadata for index derivatives (used when master unavailable)
_INDEX_FALLBACK = {
    "NIFTY": {"lot_size": 75, "strike_step": 50.0, "spot_symbol": "^NSEI"},
    "BANKNIFTY": {"lot_size": 15, "strike_step": 100.0, "spot_symbol": "^NSEBANK"},
    "FINNIFTY": {"lot_size": 40, "strike_step": 50.0, "spot_symbol": "^CNXFIN"},
    "MIDCPNIFTY": {"lot_size": 75, "strike_step": 100.0, "spot_symbol": "^NSEMDCP50"},
    "NIFTYNXT50": {"lot_size": 35, "strike_step": 50.0, "spot_symbol": "^NSMIDCP"},
}


@dataclass
class DerivativeInstrument:
    """A single tradable F&O contract."""

    underlying: str
    instrument_type: str  # FUTIDX | FUTSTK | OPTIDX | OPTSTK
    expiry: Optional[datetime] = None
    strike: float = 0.0
    option_type: str = ""  # CE | PE | XX (futures)
    security_id: str = ""
    tick_size: float = 0.0
    lot_size: int = 1
    trading_symbol: str = ""
    exchange_segment: str = "NSE_FNO"
    expiry_code: int = 0
    synthetic: bool = False

    @property
    def kind(self) -> str:
        return "IDX" if self.instrument_type in ("FUTIDX", "OPTIDX") else "STK"

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("OPTIDX", "OPTSTK")

    @property
    def is_future(self) -> bool:
        return not self.is_option

    @property
    def expiry_date(self) -> Optional[date]:
        return self.expiry.date() if self.expiry is not None else None

    @property
    def expiry_code_ref(self) -> int:
        """1 = near month, 2 = next month, 3 = far month (Dhan expiry code)."""
        return self.expiry_code or (1 if self.is_option else 1)

    def key(self) -> str:
        if self.is_option:
            return f"{self.underlying}|{self.instrument_type}|{self.expiry_date}|{self.strike:.0f}|{self.option_type}"
        return f"{self.underlying}|{self.instrument_type}|{self.expiry_date}|FUT"

    def label(self) -> str:
        if self.trading_symbol:
            return self.trading_symbol
        if self.is_option:
            ot = "CE" if self.option_type == "CE" else "PE"
            return f"{self.underlying} {self.expiry_date} {self.strike:.0f} {ot}"
        return f"{self.underlying} FUT {self.expiry_date}"

    def to_dict(self) -> dict:
        return {
            "underlying": self.underlying,
            "instrument_type": self.instrument_type,
            "expiry": str(self.expiry_date),
            "strike": self.strike,
            "option_type": self.option_type,
            "security_id": self.security_id,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "trading_symbol": self.trading_symbol,
            "exchange_segment": self.exchange_segment,
            "synthetic": self.synthetic,
            "key": self.key(),
        }


def _derive_underlying(trading_symbol: str) -> str:
    if not trading_symbol:
        return ""
    parts = str(trading_symbol).split("-")
    return parts[0]


class DerivativeUniverse:
    """Builds and queries the F&O universe from the Dhan scrip master."""

    def __init__(
        self,
        config: dict = None,
        master_path: Optional[str] = None,
        download: bool = True,
        max_age_hours: float = 24.0,
    ):
        self.config = config or {}
        _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.master_path = master_path or os.path.join(_base, "data", "scrip_master_fno.csv")
        self.max_age_hours = max_age_hours
        self._df: Optional[pd.DataFrame] = None
        self._master_error: str = ""
        if download:
            self._ensure_master()

    # ------------------------------------------------------------------
    # Loading / caching
    # ------------------------------------------------------------------
    def _ensure_master(self, force: bool = False) -> None:
        if self._df is not None and not force:
            return
        if force or self._master_stale():
            self._download_master()
        if self._df is None:
            try:
                if os.path.exists(self.master_path):
                    self._df = pd.read_csv(self.master_path)
                    self._normalize_master()
            except Exception as e:
                self._master_error = f"Failed to read scrip master: {e}"

    def _master_stale(self) -> bool:
        if not os.path.exists(self.master_path):
            return True
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.master_path))).total_seconds() / 3600.0
        return age > self.max_age_hours

    def _download_master(self) -> None:
        if requests is None:
            self._master_error = "requests not installed"
            return
        try:
            os.makedirs(os.path.dirname(self.master_path), exist_ok=True)
            resp = requests.get(_SEM_URL, timeout=120)
            resp.raise_for_status()
            tmp = self.master_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(resp.content)
            raw = pd.read_csv(tmp, low_memory=False)
            self._master_error = ""
            fno = raw[raw["SEM_INSTRUMENT_NAME"].isin(_FNO_INSTRUMENTS)]
            fno = fno.copy()
            # Keep only NSE F&O rows
            fno = fno[fno["SEM_EXM_EXCH_ID"] == "NSE"]
            fno.to_csv(self.master_path, index=False)
            os.remove(tmp)
            self._df = fno
            self._normalize_master()
        except Exception as e:
            self._master_error = f"Scrip master download failed: {e}"
            self._df = None

    def _normalize_master(self) -> None:
        df = self._df
        if df is None or df.empty:
            return
        df = df.rename(columns=_SEM_COLUMNS)
        keep = list(_SEM_COLUMNS.values())
        for col in keep:
            if col not in df.columns:
                df[col] = None
        df = df[keep].copy()
        df["security_id"] = df["security_id"].astype(str).str.strip()
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["tick_size"] = pd.to_numeric(df["tick_size"], errors="coerce")
        df["lot_size"] = pd.to_numeric(df["lot_size"], errors="coerce").fillna(1).astype(int)
        df["option_type"] = df["option_type"].fillna("XX").str.upper()
        df["underlying"] = df["trading_symbol"].astype(str).map(_derive_underlying)
        df["expiry_flag"] = df["expiry_flag"].fillna("").str.upper()
        df = df[df["underlying"].str.len() > 0]
        self._df = df.sort_values(["underlying", "expiry"]).reset_index(drop=True)

    @property
    def master_available(self) -> bool:
        return self._df is not None and not self._df.empty

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def list_underlyings(self, kind: str = "IDX") -> List[str]:
        kind = kind.upper()
        insts = ["FUTIDX", "OPTIDX"] if kind == "IDX" else ["FUTSTK", "OPTSTK"]
        if self.master_available:
            names = self._df[self._df["instrument_type"].isin(insts)]["underlying"].unique()
            out = sorted(str(n) for n in names if str(n) != "nan")
            if kind == "IDX":
                # Drop synthetic/proprietary futures variants (e.g. NIFTYFPI)
                out = [n for n in out if not n.upper().endswith("FPI")]
            if out:
                return out
        return sorted(_INDEX_FALLBACK.keys()) if kind == "IDX" else []

    def expiries(self, underlying: str, kind: str = "IDX") -> List[datetime]:
        underlying = underlying.upper()
        kind = kind.upper()
        insts = ["FUTIDX", "OPTIDX"] if kind == "IDX" else ["FUTSTK", "OPTSTK"]
        if self.master_available:
            sub = self._df[
                (self._df["underlying"].str.upper() == underlying) & (self._df["instrument_type"].isin(insts))
            ]
            exps = pd.to_datetime(sub["expiry"], errors="coerce").dropna().dt.normalize().unique()
            if len(exps):
                return sorted(exps.tolist())
        return []

    def _fallback_expiries(self, underlying: str, kind: str, start: datetime, end: datetime, weekly: bool = True):
        """Synthesise weekly (Thursdays) and monthly (last Thursday) expiries."""
        out = []
        d = start
        while d <= end:
            if weekly and d.weekday() == 3:  # Thursday
                out.append(d)
            # last Thursday of month
            nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            while nxt.weekday() != 3:
                nxt -= timedelta(days=1)
            if start <= nxt <= end and nxt not in out:
                out.append(nxt)
            d += timedelta(days=1)
        if not out:
            # ensure at least one expiry in the future
            anchor = end or datetime.now()
            out = [anchor + timedelta(days=7)]
        return sorted(out)

    def nearest_expiry(
        self,
        underlying: str,
        kind: str = "IDX",
        mode: str = "weekly",
        as_of: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """Choose the nearest expiry for a given roll mode."""
        underlying = underlying.upper()
        as_of = as_of or datetime.now()
        exps = self.expiries(underlying, kind)
        if not exps:
            horizon = as_of + timedelta(days=45)
            exps = self._fallback_expiries(underlying, kind, as_of, horizon, weekly=mode == "weekly")
        future = [e for e in exps if e >= as_of]
        return future[0] if future else exps[-1]

    def strike_step(self, underlying: str, kind: str = "IDX", expiry: Optional[datetime] = None) -> float:
        underlying = underlying.upper()
        kind = kind.upper()
        if self.master_available:
            insts = ["OPTIDX", "OPTSTK"] if kind == "STK" else ["OPTIDX"]
            sub = self._df[(self._df["underlying"].str.upper() == underlying) & (self._df["instrument_type"].isin(insts))]
            if expiry is not None:
                sub = sub[pd.to_datetime(sub["expiry"], errors="coerce").dt.normalize() == pd.Timestamp(expiry).normalize()]
            if not sub.empty and sub["strike"].notna().any():
                diffs = sub["strike"].dropna().diff().dropna()
                diffs = diffs[diffs > 0]
                if not diffs.empty:
                    return float(diffs.min())
        fb = _INDEX_FALLBACK.get(underlying)
        if fb:
            return float(fb["strike_step"])
        return 5.0

    def lot_size(self, underlying: str, kind: str = "IDX", expiry: Optional[datetime] = None) -> int:
        underlying = underlying.upper()
        kind = kind.upper()
        if self.master_available:
            insts = ["FUTIDX", "OPTIDX"] if kind == "IDX" else ["FUTSTK", "OPTSTK"]
            sub = self._df[(self._df["underlying"].str.upper() == underlying) & (self._df["instrument_type"].isin(insts))]
            if expiry is not None:
                sub = sub[pd.to_datetime(sub["expiry"], errors="coerce").dt.normalize() == pd.Timestamp(expiry).normalize()]
            if not sub.empty and sub["lot_size"].notna().any():
                return int(sub["lot_size"].iloc[0])
        fb = _INDEX_FALLBACK.get(underlying)
        if fb:
            return int(fb["lot_size"])
        return 1

    def spot_symbol(self, underlying: str, kind: str = "IDX") -> str:
        underlying = underlying.upper()
        if kind.upper() == "STK":
            return f"{underlying}.NS"
        fb = _INDEX_FALLBACK.get(underlying)
        if fb:
            return fb["spot_symbol"]
        return ""

    # ------------------------------------------------------------------
    # Contract builders
    # ------------------------------------------------------------------
    def _base_query(self, underlying: str, kind: str) -> pd.DataFrame:
        insts = ["FUTIDX", "OPTIDX"] if kind == "IDX" else ["FUTSTK", "OPTSTK"]
        return self._df[
            (self._df["underlying"].str.upper() == underlying.upper()) & (self._df["instrument_type"].isin(insts))
        ]

    def option_chain(
        self,
        underlying: str,
        expiry: Optional[datetime] = None,
        kind: str = "IDX",
        spot: Optional[float] = None,
        n_strikes: int = 5,
    ) -> pd.DataFrame:
        """Return a chain DataFrame of option contracts around a strike.

        If `spot` is given, rows are ordered around the ATM strike (nearest
        higher call, nearest lower put). Otherwise all strikes for the expiry
        are returned.
        """
        underlying = underlying.upper()
        kind = kind.upper()
        if not self.master_available:
            return self._synthetic_chain(underlying, expiry, kind, spot, n_strikes)
        sub = self._base_query(underlying, kind)
        if sub.empty:
            return pd.DataFrame()
        if expiry is not None:
            tgt = pd.Timestamp(expiry).normalize()
            sub = sub[pd.to_datetime(sub["expiry"], errors="coerce").dt.normalize() == tgt]
        if sub.empty:
            return pd.DataFrame()
        step = self.strike_step(underlying, kind, expiry)
        if spot is None or not sub["strike"].notna().any():
            return sub.sort_values("strike").reset_index(drop=True)
        atm = round(float(spot) / step) * step if step and step > 0 else float(spot)
        strikes = sorted(sub["strike"].dropna().unique().tolist())
        lo = [s for s in strikes if s <= atm]
        hi = [s for s in strikes if s > atm]
        chosen = lo[-n_strikes:] + hi[:n_strikes]
        chain = sub[sub["strike"].isin(chosen)].copy()
        chain = chain.pivot_table(index="strike", columns="option_type", values=["security_id", "lot_size", "custom_symbol"], aggfunc="first").reset_index()
        chain.columns = ["_".join(map(str, c)).strip("_") if isinstance(c, tuple) else c for c in chain.columns]
        chain = chain.rename(columns={"strike": "strike"})
        if "strike_" in chain.columns:
            chain = chain.rename(columns={"strike_": "strike"})
        return chain.sort_values("strike").reset_index(drop=True)

    def _synthetic_chain(self, underlying, expiry, kind, spot, n_strikes) -> pd.DataFrame:
        kind = kind.upper()
        step = self.strike_step(underlying, kind, expiry)
        lot = self.lot_size(underlying, kind, expiry)
        if spot is None:
            return pd.DataFrame()
        atm = round(float(spot) / step) * step if step else float(spot)
        strikes = [atm + i * step for i in range(-n_strikes, n_strikes + 1)]
        rows = []
        for s in strikes:
            rows.append(
                {
                    "strike": s,
                    "CE_security_id": "",
                    "CE_lot_size": lot,
                    "CE_custom_symbol": f"{underlying} {expiry.strftime('%d%b%y') if expiry else ''} {s:.0f} CALL",
                    "PE_security_id": "",
                    "PE_lot_size": lot,
                    "PE_custom_symbol": f"{underlying} {expiry.strftime('%d%b%y') if expiry else ''} {s:.0f} PUT",
                }
            )
        return pd.DataFrame(rows)

    def futures_contract(
        self,
        underlying: str,
        kind: str = "IDX",
        expiry: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> DerivativeInstrument:
        """Resolve a futures contract (nearest available expiry by default).

        Futures trade on monthly expiries, so when no expiry is requested we
        pick the nearest available futures expiry from the master instead of
        the (weekly) option expiry.
        """
        underlying = underlying.upper()
        kind = kind.upper()
        inst = "FUTIDX" if kind == "IDX" else "FUTSTK"
        as_of = as_of or datetime.now()
        if self.master_available:
            sub = self._base_query(underlying, kind)
            sub = sub[sub["instrument_type"] == inst]
            if not sub.empty:
                exps = pd.to_datetime(sub["expiry"], errors="coerce").dropna().dt.normalize().unique()
                if expiry is None and len(exps):
                    future = [e for e in sorted(exps) if e >= pd.Timestamp(as_of).normalize()]
                    expiry = future[0] if future else sorted(exps)[0]
                if expiry is not None and sub["expiry"].notna().any():
                    tgt = pd.Timestamp(expiry).normalize()
                    sub = sub[pd.to_datetime(sub["expiry"], errors="coerce").dt.normalize() == tgt]
                if not sub.empty:
                    row = sub.iloc[0]
                    return DerivativeInstrument(
                        underlying=underlying,
                        instrument_type=inst,
                        expiry=pd.Timestamp(row["expiry"]).to_pydatetime() if pd.notna(row["expiry"]) else None,
                        strike=0.0,
                        option_type="XX",
                        security_id=str(row["security_id"]),
                        tick_size=float(row.get("tick_size") or 0),
                        lot_size=int(row.get("lot_size") or 1),
                        trading_symbol=str(row.get("trading_symbol") or ""),
                    )
        if expiry is None:
            # No master data: fall back to a synthetic near-month expiry
            horizon = as_of + timedelta(days=45)
            expiry = self._fallback_expiries(underlying, kind, as_of, horizon, weekly=False)[0]
        return DerivativeInstrument(
            underlying=underlying,
            instrument_type=inst,
            expiry=expiry,
            strike=0.0,
            option_type="XX",
            security_id="",
            lot_size=self.lot_size(underlying, kind),
            synthetic=True,
        )

    def option_contract(
        self,
        underlying: str,
        kind: str = "IDX",
        strike: float = 0.0,
        option_type: str = "CE",
        expiry: Optional[datetime] = None,
        as_of: Optional[datetime] = None,
    ) -> DerivativeInstrument:
        """Resolve a single option contract. Strike 0 → closest ATM contract."""
        underlying = underlying.upper()
        kind = kind.upper()
        inst = "OPTIDX" if kind == "IDX" else "OPTSTK"
        ot = option_type.upper()
        expiry = expiry or self.nearest_expiry(underlying, kind, mode="weekly", as_of=as_of)
        if self.master_available:
            sub = self._base_query(underlying, kind)
            sub = sub[sub["instrument_type"] == inst]
            if expiry is not None:
                tgt = pd.Timestamp(expiry).normalize()
                sub = sub[pd.to_datetime(sub["expiry"], errors="coerce").dt.normalize() == tgt]
            if not sub.empty and sub["strike"].notna().any():
                sub = sub[sub["option_type"] == ot]
                if not sub.empty:
                    if strike:
                        cand = sub.iloc[(sub["strike"] - strike).abs().argmin()]
                    else:
                        cand = sub.iloc[0]
                    return DerivativeInstrument(
                        underlying=underlying,
                        instrument_type=inst,
                        expiry=pd.Timestamp(cand["expiry"]).to_pydatetime() if pd.notna(cand["expiry"]) else None,
                        strike=float(cand["strike"]),
                        option_type=ot,
                        security_id=str(cand["security_id"]),
                        tick_size=float(cand.get("tick_size") or 0),
                        lot_size=int(cand.get("lot_size") or self.lot_size(underlying, kind)),
                        trading_symbol=str(cand.get("trading_symbol") or ""),
                    )
        step = self.strike_step(underlying, kind, expiry)
        if not strike:
            strike = self._fallback_atm_strike(underlying, kind, step)
        return DerivativeInstrument(
            underlying=underlying,
            instrument_type=inst,
            expiry=expiry,
            strike=float(strike),
            option_type=ot,
            security_id="",
            tick_size=step,
            lot_size=self.lot_size(underlying, kind),
            synthetic=True,
        )

    @staticmethod
    def _fallback_atm_strike(underlying: str, kind: str, step: float) -> float:
        # Order-of-magnitude guess so option strategies can run without a spot quote.
        base = {"NIFTY": 19500.0, "BANKNIFTY": 49000.0, "FINNIFTY": 21500.0,
                "MIDCPNIFTY": 12000.0, "NIFTYNXT50": 74000.0}.get(underlying.upper())
        if base is None:
            base = 500.0 if kind.upper() == "STK" else 100.0
        return float(round(base / step) * step if step else base)

    # ------------------------------------------------------------------
    # Convenience maps for UI
    # ------------------------------------------------------------------
    def strategy_lot_sizes(self, underlying: str, kind: str) -> dict:
        exps = self.expiries(underlying, kind)
        out = {}
        for e in exps[:6]:
            out[str(e.date())] = self.lot_size(underlying, kind, e)
        return out

    def describe(self) -> dict:
        return {
            "master_available": self.master_available,
            "master_error": self._master_error,
            "rows": len(self._df) if self.master_available else 0,
            "index_underlyings": self.list_underlyings("IDX")[:8],
            "stock_underlyings": self.list_underlyings("STK")[:8],
        }