import numpy as np
import pandas as pd


class SignalGenerator:
    """Rule-based signal generation from technical indicators."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # ------------------------------------------------------------------
    # Composite signal scores
    # ------------------------------------------------------------------
    def momentum_score(self) -> float:
        """Higher = bullish momentum, lower = bearish. Range roughly -10..+10."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        score = 0.0
        # Directional oscillators only. ADX and -DI are handled separately below
        # because neither is bullish-when-high.
        checks = [
            ("rsi", 55, 45),
            ("dmp", 24, 18),
            ("macd_hist", 0.0, 0.0),
            ("stoch_k", 60, 40),
            ("mfi", 60, 40),
        ]
        for col, buy_thresh, sell_thresh in checks:
            val = last.get(col)
            if val is None or pd.isna(val):
                continue
            if val > buy_thresh:
                score += 1
            elif val < sell_thresh:
                score -= 1

        # -DI is the BEARISH half of the DMI pair: a high reading argues for
        # downside. Scoring it like +DI made strong downtrends read as bullish.
        dmn = last.get("dmn")
        if dmn is not None and pd.notna(dmn):
            if dmn > 24:
                score -= 1
            elif dmn < 18:
                score += 1

        # ADX is trend STRENGTH and carries no direction - a reading of 40 is just
        # as consistent with a hard downtrend as an uptrend. Previously ADX > 25
        # added a bullish point outright. Use it only to damp conviction in chop.
        adx_val = last.get("adx")
        if adx_val is not None and pd.notna(adx_val) and adx_val < 20:
            score *= 0.6

        # Trend alignment
        close, ema9, ema21, ema50, ema200 = (
            last.get("close"),
            last.get("ema9"),
            last.get("ema21"),
            last.get("ema50"),
            last.get("ema200"),
        )
        if all(v is not None and pd.notna(v) for v in (close, ema9, ema21, ema50)):
            if close > ema9 > ema21 > ema50:
                score += 2
            elif close < ema9 < ema21 < ema50:
                score -= 2

        # Bollinger band position
        bb_upper, bb_lower, bb_middle = last.get("bb_upper"), last.get("bb_lower"), last.get("bb_middle")
        if all(v is not None and pd.notna(v) for v in (bb_upper, bb_lower, bb_middle, close)):
            if close > bb_upper:
                score += 1
            elif close < bb_lower:
                score -= 1

        return float(np.clip(score, -10, 10))

    def mean_reversion_score(self) -> float:
        """Higher = oversold (buy candidates), Lower = overbought (sell candidates)."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        score = 0.0

        rsi = last.get("rsi")
        if rsi is not None and not pd.isna(rsi):
            if rsi < 30:
                score += 2
            elif rsi < 40:
                score += 1
            elif rsi > 70:
                score -= 2
            elif rsi > 60:
                score -= 1

        close, bb_lower, bb_upper = last.get("close"), last.get("bb_lower"), last.get("bb_upper")
        if pd.notna(close) and pd.notna(bb_lower) and close < bb_lower:
            score += 2
        if pd.notna(close) and pd.notna(bb_upper) and close > bb_upper:
            score -= 2

        stoch_k = last.get("stoch_k")
        if stoch_k is not None and not pd.isna(stoch_k):
            if stoch_k < 20:
                score += 1
            elif stoch_k > 80:
                score -= 1

        return float(np.clip(score, -10, 10))

    def breakout_score(self) -> float:
        """Higher = breakout likelihood (buy), lower = breakdown (sell)."""
        if self.df is None or self.df.empty:
            return 0
        last = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) > 1 else last
        score = 0.0

        don_upper, don_lower = last.get("don_upper"), last.get("don_lower")
        close = last.get("close")
        if pd.notna(close) and pd.notna(don_upper) and close >= don_upper:
            score += 3
        elif pd.notna(close) and pd.notna(don_lower) and close <= don_lower:
            score -= 3

        sup = last.get("supertrend")
        if sup is not None and not pd.isna(sup):
            if last.get("st_direction") == 1:
                score += 2
            else:
                score -= 2

        vol_exp = last.get("volume_expansion")
        if vol_exp is not None and not pd.isna(vol_exp):
            if vol_exp > 1.5:
                score += 1 if pd.notna(last.get("close")) and pd.notna(prev.get("close")) and last["close"] > prev["close"] else -1

        adx = last.get("adx")
        if adx is not None and not pd.isna(adx) and adx < 20:
            score -= 1  # Range-bound, breakout filters

        return float(np.clip(score, -10, 10))

    # ------------------------------------------------------------------
    # Signal extraction
    # ------------------------------------------------------------------
    def get_signal(self, df: pd.DataFrame, strategy: str = "momentum") -> dict:
        """Return a single latest signal based on the strategy signature."""
        if df is None or df.empty:
            return {"action": "HOLD", "confidence": 0.0, "score": 0.0, "reason": "No data"}
        self.df = df
        score = 0.0
        if strategy == "momentum":
            score = self.momentum_score()
        elif strategy == "mean_reversion":
            score = self.mean_reversion_score()
        elif strategy == "breakout":
            score = self.breakout_score()

        # Calibrated confidence: score 2 -> 0.55, 3 -> 0.65, 5 -> 0.78, 7 -> 0.88
        abs_s = abs(score)
        if abs_s < 1:
            confidence = abs_s * 0.35
        elif abs_s < 2:
            confidence = 0.35 + (abs_s - 1) * 0.15
        elif abs_s < 4:
            confidence = 0.50 + (abs_s - 2) * 0.08
        else:
            confidence = 0.66 + (abs_s - 4) * 0.05
        confidence = float(np.clip(confidence, 0, 0.92))

        if score >= 2:
            action = "BUY"
        elif score <= -2:
            action = "SELL"
        else:
            action = "HOLD"
            # HOLD confidence is low when score near 0, higher when borderline
            confidence = round(0.45 + abs_s * 0.05, 3)

        return {
            "action": action,
            "confidence": round(float(confidence), 3),
            "score": round(float(score), 3),
            "reason": self._build_reason(action),
        }

    def get_all_signals(self, df: pd.DataFrame) -> dict:
        """Return all three strategy signals for ensemble."""
        if df is None or df.empty:
            return {}
        return {
            "momentum": self.get_signal(df, "momentum"),
            "mean_reversion": self.get_signal(df, "mean_reversion"),
            "breakout": self.get_signal(df, "breakout"),
        }

    def ensemble_signal(self, df: pd.DataFrame) -> dict:
        """Agreement-weighted vote across the three strategies.

        Previously this returned whichever single strategy was most confident.
        That is a max over three correlated views, so it reports the most extreme
        draw rather than the consensus - it systematically overstates conviction,
        and it would happily return BUY at 0.75 while another strategy was
        simultaneously signalling SELL, with nothing in the output showing the
        contradiction.
        """
        all_s = self.get_all_signals(df)
        if not all_s:
            return {"action": "HOLD", "confidence": 0.0, "score": 0.0, "reason": "No data"}

        net = 0.0
        for sig in all_s.values():
            if sig["action"] == "BUY":
                net += sig["confidence"]
            elif sig["action"] == "SELL":
                net -= sig["confidence"]

        if abs(net) < 1e-9:
            action = "HOLD"
        else:
            action = "BUY" if net > 0 else "SELL"

        agreeing = [k for k, v in all_s.items() if v["action"] == action]
        opposing = [k for k, v in all_s.items() if v["action"] not in (action, "HOLD")]

        if action == "HOLD" or not agreeing:
            conf = min((v["confidence"] for v in all_s.values()), default=0.0)
            conf = round(float(np.clip(conf, 0.0, 0.5)), 3)
            return {
                "action": "HOLD",
                "confidence": conf,
                "score": 0.0,
                "reason": "No directional consensus across strategies",
                "strategy": "ensemble",
                "agreement": f"0/{len(all_s)}",
                "all": all_s,
            }

        # Mean confidence of the agreeing side, then penalised for contradiction.
        conf = sum(all_s[k]["confidence"] for k in agreeing) / len(agreeing)
        conf *= max(0.0, 1.0 - 0.3 * len(opposing))
        conf = float(np.clip(conf, 0.0, 0.92))

        lead = max(agreeing, key=lambda k: all_s[k]["confidence"])
        reason = all_s[lead]["reason"]
        if opposing:
            reason += f" (contested by {', '.join(opposing)})"

        return {
            "action": action,
            "confidence": round(conf, 3),
            "score": round(float(net), 3),
            "reason": reason,
            "strategy": lead,
            "agreement": f"{len(agreeing)}/{len(all_s)}",
            "all": all_s,
        }

    def _build_reason(self, action: str) -> str:
        last = self.df.iloc[-1]
        rsi = last.get("rsi")
        close, ema9, ema21 = last.get("close"), last.get("ema9"), last.get("ema21")
        parts = []
        if all(v is not None and pd.notna(v) for v in (close, ema9, ema21)):
            if close > ema9 > ema21:
                parts.append("price above rising EMAs")
            elif close < ema9 < ema21:
                parts.append("price below falling EMAs")
        if rsi is not None and not pd.isna(rsi):
            if rsi > 70:
                parts.append(f"RSI overbought at {rsi:.1f}")
            elif rsi < 30:
                parts.append(f"RSI oversold at {rsi:.1f}")
            else:
                parts.append(f"RSI neutral at {rsi:.1f}")
        return "; ".join(parts) if parts else "No strong technical signal"