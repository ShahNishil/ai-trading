import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestVisualizer:
    """Generates Plotly charts for backtest results."""

    def __init__(self, result: dict):
        self.result = result
        self.equity_curve = result.get("equity_curve", pd.DataFrame())
        self.trades = result.get("trades", [])
        self.price_df = result.get("price_df", pd.DataFrame())

    def equity_chart(self) -> go.Figure:
        df = self.equity_curve
        fig = go.Figure()
        if df is None or df.empty:
            return fig
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["equity"],
                mode="lines",
                name="Equity",
                line=dict(color="#0066cc", width=2),
            )
        )
        initial = self.result.get("metrics", {}).get("initial_capital", 100000)
        fig.add_hline(
            y=initial,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital",
        )
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity (INR)",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    def drawdown_chart(self) -> go.Figure:
        df = self.equity_curve
        fig = go.Figure()
        if df is None or df.empty:
            return fig
        equity = df["equity"]
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max * 100
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=dd,
                mode="lines",
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="red", width=1),
            )
        )
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    def price_chart_with_trades(self) -> go.Figure:
        """OHLC-ish price chart with trade entry/exit markers."""
        df = self.equity_curve
        fig = go.Figure()
        if df is None or df.empty:
            return fig
        p = df["price"] if "price" in df.columns else df["equity"]
        fig.add_trace(
            go.Scatter(x=df.index, y=p, mode="lines", name="Price", line=dict(color="#888", width=1.2))
        )
        buys = [t for t in self.trades if t.side == "LONG" and t.exit_price is not None]
        buy_x = [t.entry_time for t in buys]
        buy_y = [t.entry_price for t in buys]
        sell_x = [t.exit_time for t in buys]
        sell_y = [t.exit_price for t in buys]
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="Entry", marker=dict(color="green", size=9, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="Exit", marker=dict(color="red", size=9, symbol="triangle-down")))
        fig.update_layout(
            title="Price with Trade Markers",
            template="plotly_white",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig

    def monthly_returns_heatmap(self) -> go.Figure:
        df = self.equity_curve
        if df is None or df.empty or len(df) < 2:
            return go.Figure()
        equity = df["equity"]
        monthly = equity.resample("ME").last().pct_change().dropna() * 100
        if len(monthly) < 2:
            return go.Figure()
        table = pd.DataFrame({"month": monthly.index.strftime("%Y-%m"), "ret": monthly.values})
        fig = go.Figure(
            go.Bar(x=table["month"], y=table["ret"], marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in table["ret"]])
        )
        fig.update_layout(
            title="Monthly Returns (%)",
            template="plotly_white",
            height=300,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig