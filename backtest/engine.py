import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from core.indicators import IndicatorEngine
from strategies.base import BaseStrategy


@dataclass
class Position:
    symbol: str
    side: str  # LONG | SHORT
    quantity: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_reason: str = ""
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: str = ""
    pnl: float = 0.0            # net of entry + exit commission
    gross_pnl: float = 0.0      # before commission
    commission: float = 0.0


class BacktestEngine:
    """Bar-by-bar event-driven backtest simulator with slippage + commission."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.03,
        slippage_pct: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_pct = commission_pct / 100.0
        self.slippage_pct = slippage_pct / 100.0
        self.trades: List[Position] = []
        self.equity_curve: List[dict] = []
        self.signal_log: List[dict] = []

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "BUY":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _commission(self, qty: int, price: float) -> float:
        return qty * price * self.commission_pct

    def run(
        self,
        df: pd.DataFrame,
        strategy: BaseStrategy,
        symbol: str = "SYMBOL",
        skip_initial_n: int = 60,
    ) -> dict:
        """Run the backtest. df must already contain all indicator columns."""
        if df is None or df.empty:
            return {"error": "No data provided", "trades": [], "equity_curve": []}
        self.trades = []
        self.equity_curve = []
        self.signal_log = []
        self.cash = self.initial_capital

        open_position: Optional[Position] = None
        # Keep the FULL frame so the strategy retains its indicator warm-up, and
        # advance the start index instead of trimming rows. Trimming and then
        # slicing `df[: idx + skip_initial_n + 1]` fed the strategy a window whose
        # last bar sat `skip_initial_n` bars in the FUTURE of the bar being traded.
        start_idx = skip_initial_n if len(df) > skip_initial_n + 5 else 0

        for idx in range(start_idx, len(df)):
            ts = df.index[idx]
            row = df.iloc[idx]
            window = df.iloc[: idx + 1]  # causal: ends on the bar being traded
            signal = strategy.on_bar(window)
            price = float(row["close"])
            self.signal_log.append({"time": ts, "signal": signal["action"], "confidence": signal.get("confidence", 0)})

            if open_position is None:
                if signal["action"] == "BUY" and signal.get("confidence", 0) >= 0.6:
                    buy_price = self._apply_slippage(price, "BUY")
                    qty = max(1, int(self.cash * 0.95 / buy_price))
                    if qty > 0:
                        entry_comm = self._commission(qty, buy_price)
                        cost = qty * buy_price + entry_comm
                        if cost <= self.cash:
                            self.cash -= cost
                            open_position = Position(
                                symbol=symbol,
                                side="LONG",
                                quantity=qty,
                                entry_price=buy_price,
                                entry_time=ts,
                                entry_reason=signal.get("reason", ""),
                                commission=entry_comm,
                            )
            else:
                exit_signal = signal["action"] == "SELL"
                if exit_signal or self._exit_rule(open_position, price, row):
                    sell_price = self._apply_slippage(price, "SELL")
                    proceeds = open_position.quantity * sell_price
                    exit_comm = self._commission(open_position.quantity, sell_price)
                    self.cash += proceeds - exit_comm
                    open_position.exit_price = sell_price
                    open_position.exit_time = ts
                    open_position.gross_pnl = proceeds - open_position.quantity * open_position.entry_price
                    open_position.commission += exit_comm
                    # Net of costs. Reporting gross P&L here inflated win-rate,
                    # profit-factor and expectancy for every trade.
                    open_position.pnl = open_position.gross_pnl - open_position.commission
                    open_position.exit_reason = signal.get("reason", "") if exit_signal else (row.get("exit_reason", "technical exit"))
                    self.trades.append(open_position)
                    open_position = None

            equity = self.cash
            if open_position is not None:
                equity += open_position.quantity * price
            self.equity_curve.append({"timestamp": ts, "equity": equity, "price": price})

        # Force close any remaining position at last price
        if open_position is not None:
            last_price = float(df.iloc[-1]["close"])
            sell_price = self._apply_slippage(last_price, "SELL")
            proceeds = open_position.quantity * sell_price
            exit_comm = self._commission(open_position.quantity, sell_price)
            self.cash += proceeds - exit_comm
            open_position.exit_price = sell_price
            open_position.exit_time = df.index[-1]
            open_position.gross_pnl = proceeds - open_position.quantity * open_position.entry_price
            open_position.commission += exit_comm
            open_position.pnl = open_position.gross_pnl - open_position.commission
            open_position.exit_reason = "end of backtest"
            self.trades.append(open_position)

        equity_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        traded = df.iloc[start_idx:]  # benchmark must span the traded period, not the warm-up
        metrics = compute_metrics(equity_df, self.trades, self.initial_capital, traded)
        return {
            "trades": self.trades,
            "equity_curve": equity_df,
            "metrics": metrics,
            "final_cash": self.cash,
            "signal_log": self.signal_log,
        }

    @staticmethod
    def _exit_rule(position: Position, price: float, row: pd.Series) -> bool:
        return False


def infer_periods_per_year(index) -> float:
    """Bars per year implied by the index spacing.

    Hardcoding 252 treats a 5-minute bar as a trading day, which inflates both
    CAGR and the Sharpe annualisation by orders of magnitude on intraday data.
    Falls back to 252 when the spacing cannot be determined.
    """
    try:
        if index is None or len(index) < 3:
            return 252.0
        deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
        median = deltas.median()
        seconds = median.total_seconds()
        if not seconds or seconds <= 0:
            return 252.0
        if seconds >= 82800:                      # >= 23h -> daily or slower
            return 252.0
        return 252.0 * (6.25 * 3600.0 / seconds)  # ~6.25h NSE session
    except Exception:
        return 252.0


def compute_metrics(equity_df: pd.DataFrame, trades: list, initial_capital: float, price_df: pd.DataFrame) -> dict:
    if equity_df is None or equity_df.empty:
        return {"error": "No equity data"}

    equity = equity_df["equity"]
    final_equity = float(equity.iloc[-1])
    ret = (final_equity / initial_capital) - 1
    n_periods = len(equity)
    periods_per_year = infer_periods_per_year(equity.index)
    cagr = (final_equity / initial_capital) ** (periods_per_year / max(n_periods, 1)) - 1 if final_equity > 0 else -1

    daily_ret = equity.pct_change().dropna()
    std = float(daily_ret.std()) if len(daily_ret) > 1 else 0.0
    sharpe = float(daily_ret.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(daily_ret.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 0 else 0.0

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    closed = [t for t in trades if t.exit_price is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    win_rate = len(wins) / len(closed) if closed else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if closed else 0.0

    # Benchmark buy & hold
    if price_df is not None and len(price_df) > 0 and "close" in price_df.columns:
        bench_ret = (float(price_df["close"].iloc[-1]) / float(price_df["close"].iloc[0])) - 1
    else:
        bench_ret = 0.0

    total_commission = sum(getattr(t, "commission", 0.0) for t in closed)
    gross_total = sum(getattr(t, "gross_pnl", t.pnl) for t in closed)

    return {
        "initial_capital": initial_capital,
        "periods_per_year": round(periods_per_year, 1),
        "total_commission": round(total_commission, 2),
        "gross_pnl": round(gross_total, 2),
        "net_pnl": round(sum(t.pnl for t in closed), 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(ret * 100, 2),
        "buy_hold_return_pct": round(bench_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "num_trades": len(closed),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "volatility_pct": round(std * 100, 2),
    }


def run_backtest(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    symbol: str = "SYMBOL",
    initial_capital: float = 100000,
    commission_pct: float = 0.03,
    slippage_pct: float = 0.05,
    skip_initial_n: int = 60,
) -> dict:
    """Convenience wrapper: computes indicators, runs backtest, returns results."""
    indicator_engine = IndicatorEngine(df)
    enriched = indicator_engine.compute_all()
    engine = BacktestEngine(initial_capital, commission_pct, slippage_pct)
    return engine.run(enriched, strategy, symbol, skip_initial_n=skip_initial_n)