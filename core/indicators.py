import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    signal_line = ema(line, signal)
    hist = line - signal_line
    return line, signal_line, hist


def bollinger(series: pd.Series, length: int = 20, std: float = 2.0):
    mid = sma(series, length)
    sd = series.rolling(length, min_periods=length).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return upper, mid, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest = low.rolling(k, min_periods=1).min()
    highest = high.rolling(k, min_periods=1).max()
    k_line = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    d_line = sma(k_line, d)
    return k_line, d_line


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    trn = tr.ewm(alpha=1 / length, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / trn.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / trn.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / length, adjust=False).mean()
    return adx_line, plus_di, minus_di


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10, multiplier: float = 3.0):
    """Classic SuperTrend.

    The bands must be seeded at the first bar where ATR is defined. Without that
    seed every comparison against a NaN band evaluates False, the carry-forward
    branch propagates NaN indefinitely, and the trend is pinned to +1 for the
    whole series.
    """
    mid = (high + low) / 2
    a = atr(high, low, close, length)
    upper_basic = (mid + multiplier * a).to_numpy(dtype=float)
    lower_basic = (mid - multiplier * a).to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(close)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    line = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)

    valid = ~np.isnan(upper_basic) & ~np.isnan(lower_basic) & ~np.isnan(c)
    if not valid.any():
        return pd.Series(line, index=close.index), pd.Series(trend, index=close.index)

    first = int(np.argmax(valid))
    upper[first] = upper_basic[first]
    lower[first] = lower_basic[first]
    trend[first] = 1 if c[first] >= lower[first] else -1
    line[first] = lower[first] if trend[first] == 1 else upper[first]

    for i in range(first + 1, n):
        if np.isnan(upper_basic[i]) or np.isnan(lower_basic[i]):
            upper[i], lower[i] = upper[i - 1], lower[i - 1]
            trend[i] = trend[i - 1]
            line[i] = line[i - 1]
            continue

        upper[i] = (
            upper_basic[i]
            if (upper_basic[i] < upper[i - 1]) or (c[i - 1] > upper[i - 1])
            else upper[i - 1]
        )
        lower[i] = (
            lower_basic[i]
            if (lower_basic[i] > lower[i - 1]) or (c[i - 1] < lower[i - 1])
            else lower[i - 1]
        )

        if trend[i - 1] == 1:
            trend[i] = -1 if c[i] < lower[i] else 1
        else:
            trend[i] = 1 if c[i] > upper[i] else -1
        line[i] = lower[i] if trend[i] == 1 else upper[i]

    return pd.Series(line, index=close.index), pd.Series(trend, index=close.index)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    raw = typical * volume
    pos_flow = raw.where(typical.diff() > 0, 0.0)
    neg_flow = raw.where(typical.diff() < 0, 0.0)
    pos_sum = pos_flow.rolling(length, min_periods=1).sum()
    neg_sum = neg_flow.rolling(length, min_periods=1).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(length, min_periods=length).mean()
    mad = tp.rolling(length, min_periods=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, np.nan)


def roc(series: pd.Series, length: int = 12) -> pd.Series:
    return series.pct_change(length) * 100


def cdl_patterns(open_, high, low, close) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    body = (close - open_).abs()
    rng = (high - low).replace(0, np.nan)
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low

    out["engulfing"] = np.where(
        (close.shift(1) < open_.shift(1)) & (close > open_) & (close >= open_.shift(1)) & (open_ <= close.shift(1)),
        1, 0,
    )
    out["hammer"] = np.where((body < rng * 0.3) & (lower > body * 2) & (upper < body), 1, 0)
    out["doji"] = np.where(body <= rng * 0.1, 1, 0)
    out["morning_star"] = np.where(
        (close.shift(2) < open_.shift(2))
        & (body.shift(1) < rng.shift(1) * 0.3)
        & (close > open_)
        & (close > (open_.shift(2) + close.shift(2)) / 2),
        1, 0,
    )
    out["shooting_star"] = np.where((body < rng * 0.3) & (upper > body * 2) & (lower < body), 1, 0)
    return out


class IndicatorEngine:
    """Computes the full technical indicator suite. Pure numpy/pandas, zero heavy deps."""

    def __init__(self, df: pd.DataFrame, timeframe: str = "daily"):
        if df is None or df.empty:
            raise ValueError("Cannot compute indicators on empty dataframe")
        self.df = df.copy()
        self.timeframe = timeframe
        self._prepare()

    def _prepare(self):
        for col in ["open", "high", "low", "close"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        if "volume" not in self.df.columns:
            self.df["volume"] = 0

    def compute_all(self) -> pd.DataFrame:
        df = self.df.copy()
        c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

        # --- Trend / Moving Averages ---
        df["ema9"] = ema(c, 9)
        df["ema21"] = ema(c, 21)
        df["ema50"] = ema(c, 50)
        df["ema200"] = ema(c, 200)
        df["sma20"] = sma(c, 20)
        df["sma50"] = sma(c, 50)
        df["vwap"] = self._vwap()

        # --- Momentum ---
        df["macd"], df["macd_signal"], df["macd_hist"] = macd(c)
        df["rsi"] = rsi(c, 14)
        df["rsi_fast"] = rsi(c, 7)
        df["stoch_k"], df["stoch_d"] = stochastic(h, l, c)
        df["roc"] = roc(c, 12)
        df["mom"] = c.diff(10)
        df["trix"] = ema(ema(ema(c, 15), 15), 15).pct_change()

        # --- Volatility ---
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger(c)
        df["atr"] = atr(h, l, c, 14)
        df["natr"] = df["atr"] / c.replace(0, np.nan) * 100
        df["kc_upper"], df["kc_middle"], df["kc_lower"] = self._keltner()
        # Shifted by one bar: the channel is the PRIOR 20-bar extreme. Including the
        # current bar makes "close breaks above the channel" unreachable, since
        # close <= high <= rolling-max-of-high by construction.
        df["don_upper"] = h.rolling(20, min_periods=20).max().shift(1)
        df["don_lower"] = l.rolling(20, min_periods=20).min().shift(1)

        # --- Volume ---
        df["volume_sma20"] = sma(v, 20)
        df["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
        df["mfi"] = mfi(h, l, c, v, 14)

        # --- Trend strength ---
        df["adx"], df["dmp"], df["dmn"] = adx(h, l, c, 14)
        df["cci"] = cci(h, l, c, 20)
        df["willr"] = willr(h, l, c, 14)
        df["apo"] = ema(c, 12) - ema(c, 26)

        # --- SuperTrend ---
        df["supertrend"], df["st_direction"] = supertrend(h, l, c, 10, 3)

        # --- Candlestick patterns ---
        pats = cdl_patterns(o, h, l, c)
        for col in pats.columns:
            df[col] = pats[col]

        # --- Derived features ---
        df["pct_change"] = c.pct_change() * 100
        df["range"] = (h - l) / c.replace(0, np.nan) * 100
        df["close_above_ema21"] = (c > df["ema21"]).astype(int)
        df["close_above_ema50"] = (c > df["ema50"]).astype(int)
        df["close_above_ema200"] = (c > df["ema200"]).astype(int)
        df["volume_expansion"] = v / df["volume_sma20"].replace(0, np.nan)

        # NaN neutralization for boolean columns used by strategies
        df["close_above_ema21"] = df["close_above_ema21"].fillna(0)
        df["close_above_ema50"] = df["close_above_ema50"].fillna(0)
        df["close_above_ema200"] = df["close_above_ema200"].fillna(0)

        return df

    def _vwap(self, length: int = 20) -> pd.Series:
        typical = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        vol = self.df["volume"].replace(0, np.nan)
        cum_price_vol = (typical * vol).cumsum()
        cum_vol = vol.cumsum()
        vwap = cum_price_vol / cum_vol.replace(0, np.nan)
        return vwap.rolling(length).mean()

    def _keltner(self, length: int = 20, mult: float = 2.0):
        e = ema(self.df["close"], length)
        a = atr(self.df["high"], self.df["low"], self.df["close"], length)
        upper = e + mult * a
        lower = e - mult * a
        return upper, e, lower

    @staticmethod
    def latest_summary(df: pd.DataFrame) -> dict:
        """Extract latest indicator values as a compact dict for AI prompts."""
        if df is None or df.empty:
            return {}
        last = df.iloc[-1]
        cols = [
            "close", "ema9", "ema21", "ema50", "ema200", "sma20", "rsi", "macd",
            "macd_signal", "macd_hist", "atr", "bb_upper", "bb_middle", "bb_lower",
            "supertrend", "adx", "stoch_k", "stoch_d", "mfi", "cci", "obv",
            "vwap", "close_above_ema21", "close_above_ema50", "close_above_ema200",
            "natr", "volume_expansion", "engulfing", "doji", "hammer",
            "morning_star", "shooting_star", "willr",
        ]
        summary = {}
        for col in cols:
            if col in df.columns:
                val = last.get(col)
                if pd.notna(val):
                    try:
                        summary[col] = round(float(val), 4)
                    except (TypeError, ValueError):
                        summary[col] = float(val)
        if "pct_change" in df.columns and pd.notna(last.get("pct_change")):
            summary["pct_change"] = round(float(last["pct_change"]), 4)
        return summary