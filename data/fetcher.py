import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from core.dhan_client import DhanClient
from data.cache import DataCache


class DataFetcher:
    """Fetches historical and live data from Dhan API with caching and yfinance fallback."""

    def __init__(self, dhan: Optional[DhanClient], cache: Optional[DataCache] = None):
        self.dhan = dhan
        self.cache = cache or DataCache()
        # yfinance is optional — only needed when Dhan not configured
        self._yfinance_available = None

    def _is_yfinance_available(self) -> bool:
        if self._yfinance_available is not None:
            return self._yfinance_available
        try:
            import yfinance  # noqa: F401
            self._yfinance_available = True
        except ImportError:
            self._yfinance_available = False
        return self._yfinance_available

    def _to_yahoo_symbol(self, symbol: str) -> str:
        # NSE symbols need .NS suffix for yfinance
        s = symbol.upper().strip()
        if s.endswith(".NS") or s.endswith(".BO"):
            return s
        return f"{s}.NS"

    def _fetch_yahoo(self, symbol: str, days: int, interval: str = "1d") -> pd.DataFrame:
        """Fetch from Yahoo Finance as fallback when Dhan not configured."""
        if not self._is_yfinance_available():
            raise ValueError("yfinance not installed. Run: pip install yfinance")
        import yfinance as yf

        yahoo_symbol = self._to_yahoo_symbol(symbol)
        # yfinance period: handle days -> period string
        if days <= 7:
            period = f"{days}d"
        elif days <= 60:
            period = f"{days}d"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"

        # Map our interval to yfinance interval
        interval_map = {
            "1d": "1d",
            "daily": "1d",
            "60min": "60m",
            "15min": "15m",
            "5min": "5m",
            "1min": "1m",
        }
        yf_interval = interval_map.get(interval, "1d")

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=yf_interval, auto_adjust=False)
            if df is None or df.empty:
                # Try with period max for daily
                df = yf.download(yahoo_symbol, period=period, interval=yf_interval, progress=False, auto_adjust=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            if df is None or df.empty:
                return pd.DataFrame()
            # Normalize columns to lower
            df.columns = [c.lower() for c in df.columns]
            # Ensure required columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    return pd.DataFrame()
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df.index.name = "timestamp"
            # Filter to requested days — handle tz-aware index from yfinance
            if days and len(df) > 0:
                try:
                    cutoff = datetime.now() - timedelta(days=days + 5)
                    # yfinance returns tz-aware (Asia/Kolkata) index; make comparable
                    if df.index.tz is not None:
                        # Convert df index to naive for simple comparison
                        df.index = df.index.tz_localize(None)
                    df = df[df.index >= cutoff]
                except Exception:
                    # If filtering fails, return as-is
                    pass
            # Ensure index is tz-naive for consistency with cache
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except Exception:
                pass
            return df
        except Exception as e:
            raise ValueError(f"Yahoo fetch failed for {yahoo_symbol}: {e}")

    def _to_dataframe(self, response: dict) -> pd.DataFrame:
        if not response or response.get("status") != "success":
            raise ValueError(f"Failed to fetch data: {response}")
        data = response.get("data", {})
        if not data or "timestamp" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df.set_index("timestamp")
        return df

    def _is_cache_valid(self, symbol: str, timeframe: str, min_rows: int = 60) -> bool:
        """Check if cache has enough recent data."""
        try:
            df = self.cache.get_candles(symbol, timeframe)
            if df is None or df.empty or len(df) < min_rows:
                return False
            # Check freshness: last candle within 5 days for daily, 1 day for intraday
            last_ts = df.index.max()
            if pd.isna(last_ts):
                return False
            age_days = (datetime.now() - last_ts.to_pydatetime()).days if hasattr(last_ts, 'to_pydatetime') else 10
            if timeframe == "daily" and age_days > 7:
                return False
            if timeframe != "daily" and age_days > 2:
                return False
            return True
        except Exception:
            return False

    def fetch_daily(
        self,
        symbol: str,
        security_id: str,
        days: int = 365,
        exchange: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        # Use valid cache if available
        if use_cache and self._is_cache_valid(symbol, "daily"):
            cached = self.cache.get_candles(symbol, "daily")
            if not cached.empty:
                return cached

        # Try Dhan if available
        if self.dhan is not None:
            try:
                to_date = datetime.now().strftime("%Y-%m-%d")
                from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                response = self.dhan.get_historical_daily(
                    security_id=security_id,
                    exchange_segment=exchange,
                    instrument_type=instrument_type,
                    from_date=from_date,
                    to_date=to_date,
                )
                df = self._to_dataframe(response)
                if not df.empty:
                    self.cache.save_candles(symbol, "daily", df.reset_index())
                    return df
            except Exception as e:
                # Fall through to yfinance fallback
                print(f"[DataFetcher] Dhan daily fetch failed for {symbol}: {e} — trying yfinance")

        # Fallback to yfinance
        try:
            df = self._fetch_yahoo(symbol, days, interval="daily")
            if not df.empty:
                self.cache.save_candles(symbol, "daily", df.reset_index())
            return df
        except Exception as e:
            print(f"[DataFetcher] Yahoo fallback also failed for {symbol}: {e}")
            # Last resort: return stale cache even if not valid
            if use_cache:
                return self.cache.get_candles(symbol, "daily")
            return pd.DataFrame()

    def fetch_intraday(
        self,
        symbol: str,
        security_id: str,
        interval: int = 15,
        days: int = 30,
        exchange: str = "NSE_EQ",
        instrument_type: str = "EQUITY",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        timeframe = f"{interval}min"
        if use_cache and self._is_cache_valid(symbol, timeframe):
            cached = self.cache.get_candles(symbol, timeframe)
            if not cached.empty:
                return cached

        if self.dhan is not None:
            try:
                to_date = datetime.now().strftime("%Y-%m-%d")
                from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                response = self.dhan.get_historical_intraday(
                    security_id=security_id,
                    exchange_segment=exchange,
                    instrument_type=instrument_type,
                    from_date=from_date,
                    to_date=to_date,
                    interval=interval,
                )
                df = self._to_dataframe(response)
                if not df.empty:
                    self.cache.save_candles(symbol, timeframe, df.reset_index())
                    return df
            except Exception as e:
                print(f"[DataFetcher] Dhan intraday fetch failed for {symbol}: {e} — trying yfinance")

        # yfinance intraday: limited to 7d for 1m, 60d for 15m/60m
        try:
            yf_interval = f"{interval}min" if interval in [1, 5, 15] else "60m"
            # yfinance only allows 60m for 60, 15m for 15, etc.
            if interval == 60:
                yf_interval = "60m"
            elif interval == 15:
                yf_interval = "15m"
            elif interval == 5:
                yf_interval = "5m"
            df = self._fetch_yahoo(symbol, min(days, 60), interval=yf_interval)
            if not df.empty:
                self.cache.save_candles(symbol, timeframe, df.reset_index())
            return df
        except Exception as e:
            print(f"[DataFetcher] Yahoo intraday fallback failed for {symbol}: {e}")
            if use_cache:
                return self.cache.get_candles(symbol, timeframe)
            return pd.DataFrame()

    def fetch_quote(self, security_id: str, exchange: str = "NSE_EQ") -> Optional[float]:
        if self.dhan is not None:
            try:
                val = self.dhan.get_ltp(security_id, exchange)
                if val is not None:
                    return val
            except Exception:
                pass
        return None
