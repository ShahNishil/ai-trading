import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.strategy_agent import StrategyAgent  # noqa: E402
from backtest.runner import BacktestRunner  # noqa: E402
from backtest.visualizer import BacktestVisualizer  # noqa: E402
from data.universe import StockUniverse  # noqa: E402
from strategies.registry import create_strategy, list_strategies  # noqa: E402
from ui.components import ensure_runtime  # noqa: E402

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

st.markdown("# 📊 Strategy Backtesting")
st.caption("Backtest built-in or AI-generated strategies on NSE historical data.")

runtime = ensure_runtime()
config = runtime["config"]
bt_cfg = config.get("backtest", {})

with st.sidebar:
    st.markdown("### Backtest Settings")
    universe = StockUniverse(config)
    watchlist = universe.resolve(config.get("triggers", {}).get("watchlist", "NIFTY50"))
    symbol_map = {s["symbol"]: s for s in watchlist}
    symbol = st.selectbox("Symbol", list(symbol_map.keys()))
    timeframe = st.selectbox(
        "Timeframe",
        ["daily", "60min", "15min", "5min"],
        index=0 if bt_cfg.get("default_timeframe", "daily") == "daily" else 0,
        format_func=lambda t: "Daily" if t == "daily" else t.upper(),
    )
    lookback_days = st.slider("Lookback (days)", 60, 1500, 730, 30)
    initial_capital = st.number_input("Initial capital (₹)", 10000, 10_000_000, int(bt_cfg.get("initial_capital", 100000)), 5000)
    commission = st.slider("Commission %", 0.0, 0.5, float(bt_cfg.get("commission_pct", 0.03)), 0.01)
    slippage = st.slider("Slippage %", 0.0, 1.0, float(bt_cfg.get("slippage_pct", 0.05)), 0.05)

    st.markdown("### Risk exits")
    st.caption("Mirror the live auto-trader's exits so the backtest tests what actually runs. 0 disables.")
    at_cfg = config.get("auto_trade", {})
    bt_stop = st.slider("Stop-loss %", 0.0, 10.0, float(at_cfg.get("stop_loss_pct", 2.0)), 0.5)
    bt_target = st.slider("Target %", 0.0, 20.0, float(at_cfg.get("target_pct", 4.0)), 0.5)
    bt_trail = st.slider("Trailing stop %", 0.0, 10.0, float(at_cfg.get("trailing_stop_pct", 1.5)), 0.5)

    st.markdown("### Strategy")
    strategies = list_strategies()
    strat_names = [s["name"] for s in strategies]
    if "ai" not in strat_names:
        strat_names.append("ai")
    selected_strategy = st.selectbox(
        "Strategy", strat_names,
        index=strat_names.index(bt_cfg.get("default_strategy", "momentum")) if bt_cfg.get("default_strategy", "momentum") in strat_names else 0,
    )

    st.markdown("### Parameters")
    params = {}
    strat_meta = next((s for s in strategies if s["name"] == selected_strategy), None)
    if strat_meta and selected_strategy != "ai":
        for pname, pval in strat_meta["default_params"].items():
            if isinstance(pval, (int,)) and pval <= 100:
                params[pname] = st.number_input(pname, 1, 500, int(pval), 1)
            elif isinstance(pval, float):
                params[pname] = st.slider(pname, 0.5, 20.0, float(pval), 0.5)
            elif isinstance(pval, bool):
                params[pname] = st.checkbox(pname, pval)
            else:
                params[pname] = pval
    run_btn = st.button("🏁 Run Backtest", type="primary", use_container_width=True)

st.markdown("### Run")
if not runtime["data_fetcher"]:
    st.error("Dhan not configured. Set credentials in `.env` to fetch historical data.")
else:
    watch_item = symbol_map.get(symbol, {})
    security_id = watch_item.get("security_id", "0")
    placeholders = st.empty()
    if run_btn:
        runner = BacktestRunner(runtime["data_fetcher"], runtime["cache"])
        with st.spinner(f"Backtesting {selected_strategy} on {symbol} ({timeframe})..."):
            result = runner.run(
                strategy_name=selected_strategy,
                symbol=symbol,
                security_id=security_id,
                params=params or None,
                timeframe=timeframe,
                lookback_days=lookback_days,
                initial_capital=initial_capital,
                commission_pct=commission,
                slippage_pct=slippage,
                stop_loss_pct=bt_stop,
                target_pct=bt_target,
                trailing_stop_pct=bt_trail,
            )
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state["bt_result"] = result

    result = st.session_state.get("bt_result")
    if result and result.get("symbol") == symbol:
        metrics = result.get("metrics", {})
        st.markdown("### Performance Metrics")
        mc = st.columns(6)
        mc[0].metric("Total Return", f"{metrics.get('total_return_pct', 0)}%")
        mc[1].metric("Sharpe", metrics.get("sharpe", 0))
        mc[2].metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0)}%")
        mc[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0)}%")
        mc[4].metric("Profit Factor", metrics.get("profit_factor", 0))
        mc[5].metric("Trades", metrics.get("num_trades", 0))

        viz = BacktestVisualizer(result)
        st.plotly_chart(viz.equity_chart(), use_container_width=True)
        st.plotly_chart(viz.drawdown_chart(), use_container_width=True)
        st.plotly_chart(viz.price_chart_with_trades(), use_container_width=True)

        trades = result.get("trades", [])
        if trades:
            st.markdown("### Trades")
            trows = [
                {
                    "Entry": t.entry_time,
                    "Exit": t.exit_time,
                    "Entry Price": round(float(t.entry_price), 2),
                    "Exit Price": round(float(t.exit_price or 0), 2),
                    "Qty": t.quantity,
                    "P&L": round(float(t.pnl), 2),
                    "Reason": t.exit_reason,
                }
                for t in trades
            ]
            st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🤖 Generate AI Strategy")
ai_col, show_col = st.columns([1, 1])
if ai_col.button("Generate strategy with AI", key="gen_ai_strat"):
    if runtime["ai_engine"] is not None:
        sa = StrategyAgent(runtime["ai_engine"])
        with st.spinner("Asking AI to design a strategy..."):
            ai_strat = sa.generate(symbol=symbol, focus="trend following")
        st.session_state["ai_strategy"] = ai_strat
    else:
        st.error("GROQ_API_KEY not set.")

ai_strat = st.session_state.get("ai_strategy")
if ai_strat:
    st.json(ai_strat)
    if show_col.button("▶ Backtest AI Strategy", key="bt_ai"):
        sa = StrategyAgent(runtime["ai_engine"])
        params = sa.to_ai_strategy_params(ai_strat)
        runner = BacktestRunner(runtime["data_fetcher"], runtime["cache"])
        with st.spinner("Backtesting AI-generated strategy..."):
            result = runner.run(
                strategy_name="ai",
                symbol=symbol,
                security_id=security_id,
                params=params,
                timeframe=timeframe,
                lookback_days=lookback_days,
                initial_capital=initial_capital,
                commission_pct=commission,
                slippage_pct=slippage,
                stop_loss_pct=bt_stop,
                target_pct=bt_target,
                trailing_stop_pct=bt_trail,
            )
        st.session_state["bt_result"] = result
        st.rerun()