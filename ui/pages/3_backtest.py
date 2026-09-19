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
    symbol_choice = st.selectbox("Symbol", list(symbol_map.keys()))
    any_symbol = st.text_input(
        "…or type any NSE symbol",
        value="",
        placeholder="e.g. TATAPOWER, IRCTC",
        help="Overrides the dropdown. Data comes from Yahoo (.NS added automatically) unless the symbol is in a watchlist with a Dhan security_id.",
    ).strip().upper()
    symbol = any_symbol or symbol_choice
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
# ----------------------------------------------------------------------
# Walk-forward validation: the honest version of "optimize"
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🚶 Walk-Forward Validation")
st.caption(
    "Optimizes on a rolling training window and evaluates on the unseen window "
    "that follows. The out-of-sample row is the only number to trust; the gap "
    "between in-sample and out-of-sample is the overfitting tax."
)
wf_c1, wf_c2, wf_c3 = st.columns(3)
wf_train = wf_c1.number_input("Train bars", 120, 756, 252, 21)
wf_test = wf_c2.number_input("Test bars", 21, 252, 63, 21)
wf_grid_on = wf_c3.checkbox("Grid-search params per fold", value=True)

if st.button("🏃 Run walk-forward", use_container_width=True):
    if not runtime["data_fetcher"]:
        st.error("No data source configured.")
    else:
        from backtest.walkforward import walk_forward

        grids = {
            "momentum": {"fast_ema": [9, 12], "slow_ema": [21, 26]},
            "mean_reversion": {"oversold": [25, 30, 35], "overbought": [65, 70, 75]},
            "breakout": {"donchian_length": [15, 20, 30]},
        }
        grid = grids.get(selected_strategy, {}) if wf_grid_on else {}
        wf_df = runtime["data_fetcher"].fetch_daily(symbol, security_id, days=lookback_days)
        if wf_df is None or wf_df.empty:
            st.error("No data for symbol.")
        else:
            with st.spinner(f"Walk-forward on {symbol}: {selected_strategy}..."):
                wf = walk_forward(
                    selected_strategy, wf_df, grid,
                    train_bars=int(wf_train), test_bars=int(wf_test),
                    initial_capital=initial_capital,
                    commission_pct=commission, slippage_pct=slippage,
                    stop_loss_pct=bt_stop, target_pct=bt_target, trailing_stop_pct=bt_trail,
                )
            if wf.get("error"):
                st.error(wf["error"])
            else:
                if wf.get("warning"):
                    st.warning(wf["warning"])
                o = wf["oos_metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("OOS compounded return", f"{o['compounded_return_pct']}%")
                m2.metric("OOS mean Sharpe", o["mean_sharpe"])
                m3.metric("Overfit gap (IS − OOS)", f"{wf['overfit_gap_pct']}pp",
                          help="In-sample mean return minus out-of-sample. Large positive = the grid fit noise.")
                m4.metric("Param stability", f"{wf['param_stability']:.0%}",
                          help="Share of folds choosing the same parameters. Low = the 'edge' keeps moving.")
                st.caption(
                    f"{wf['n_folds']} folds · {o['profitable_folds']} profitable · "
                    f"worst fold {o['worst_fold_return_pct']}% · {o['num_trades']} OOS trades"
                )
                fold_rows = [
                    {
                        "test window": f"{f['test_start'][:10]} → {f['test_end'][:10]}",
                        "params": str(f["chosen_params"]),
                        "IS ret%": f["train_metrics"].get("total_return_pct", 0),
                        "OOS ret%": f["test_metrics"].get("total_return_pct", 0),
                        "OOS sharpe": f["test_metrics"].get("sharpe", 0),
                        "OOS trades": f["test_metrics"].get("num_trades", 0),
                    }
                    for f in wf["folds"]
                ]
                st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
