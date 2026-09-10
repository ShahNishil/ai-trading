import os
import sys
import threading

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.auto_trade_agent import AutoTradeAgent  # noqa: E402
from ui.components import ensure_runtime  # noqa: E402
from ui_state import AutoTradeState  # noqa: E402

st.set_page_config(page_title="Auto AI Trading", page_icon="🤖", layout="wide")

st.markdown("# 🤖 Auto AI Trading")
st.caption("AI autonomously scans, decides, and places trades via the Dhan API (paper or live).")

runtime = ensure_runtime()
config = runtime["config"]
at_cfg = config.get("auto_trade", {})

st.sidebar.markdown("### Auto Trade Settings")
mode = st.sidebar.selectbox(
    "Execution mode", options=["paper", "live"], index=0 if runtime["dhan"] else 0, disabled=runtime["dhan"] is None
)
max_positions = st.sidebar.number_input("Max positions", 1, 20, int(at_cfg.get("max_positions", 5)))
risk_pct = st.sidebar.slider("Risk per trade %", 0.5, 10.0, float(at_cfg.get("risk_per_trade_pct", 2.0)), 0.5)
min_conf = st.sidebar.slider("Min AI confidence", 0.5, 0.95, float(at_cfg.get("min_confidence", 0.70)), 0.05)
scan_interval = st.sidebar.number_input("Scan interval (sec)", 30, 3600, int(at_cfg.get("scan_interval_seconds", 300)))

c1, c2 = st.columns([1, 1])
start_btn = c1.button("▶ Start Auto Trader", type="primary", use_container_width=True)
stop_btn = c2.button("⏹ Stop Auto Trader", use_container_width=True)

if start_btn:
    if runtime["data_fetcher"] is None or runtime["ai_engine"] is None:
        st.error("Configure GROQ_API_KEY and Dhan credentials in `.env` first.")
    else:
        # Apply live overrides to config copy used by this session
        config["auto_trade"]["max_positions"] = int(max_positions)
        config["auto_trade"]["risk_per_trade_pct"] = float(risk_pct)
        config["auto_trade"]["min_confidence"] = float(min_conf)
        config["auto_trade"]["scan_interval_seconds"] = int(scan_interval)
        config["dhan"]["trading_mode"] = mode

        runtime["trading_engine"].mode = mode
        runtime["trading_engine"].orders.mode = mode
        runtime["trading_engine"].portfolio.mode = mode

        agent = AutoTradeAgent(
            runtime["ai_engine"], runtime["data_fetcher"], runtime["trading_engine"], config
        )
        from data.universe import StockUniverse

        watchlist = StockUniverse(config).resolve(
            config.get("triggers", {}).get("watchlist", "NIFTY50")
        )
        if AutoTradeState.auto_trade_thread is None or not AutoTradeState.auto_trade_thread.is_alive():
            AutoTradeState.auto_trade_stop = threading.Event()
            AutoTradeState.auto_trade_thread = threading.Thread(
                target=agent.run_forever,
                args=(watchlist, AutoTradeState.auto_trade_stop),
                daemon=True,
            )
            AutoTradeState.auto_trade_thread.start()
            st.success(f"Auto trader started in **{mode}** mode. Scanning {len(watchlist)} symbols every {scan_interval}s.")
        else:
            st.warning("Auto trader already running.")

if stop_btn:
    if AutoTradeState.auto_trade_stop is not None:
        AutoTradeState.auto_trade_stop.set()
        AutoTradeState.auto_trade_thread = None
        AutoTradeState.last_scan = []
        st.warning("Auto trader stopped.")

# ---------------------------------------------------------------
# Status display
# ---------------------------------------------------------------
running = AutoTradeState.auto_trade_thread is not None and AutoTradeState.auto_trade_thread.is_alive()
st.markdown("### Status")
cols = st.columns(4)
cols[0].metric("Status", "RUNNING 🟢" if running else "STOPPED 🔴")
try:
    summary = runtime["trading_engine"].get_account_summary()
except Exception:
    summary = {}
cols[1].metric("Mode", mode)
cols[2].metric("Open Positions", summary.get("open_positions", 0))
cols[3].metric("Realized P&L (₹)", f"{summary.get('realized_pnl', 0):,.0f}")

if AutoTradeState.last_scan_time:
    st.caption(f"Last scan: {AutoTradeState.last_scan_time}")
if AutoTradeState.last_error:
    st.error(f"Last error: {AutoTradeState.last_error}")

# ---------------------------------------------------------------
# Last scan decisions
# ---------------------------------------------------------------
st.markdown("### Latest Scan Decisions")
if AutoTradeState.last_scan:
    rows = []
    for d in AutoTradeState.last_scan:
        exec_detail = d.get("execution", {})
        rows.append(
            {
                "Symbol": d.get("symbol"),
                "Decision": d.get("decision"),
                "Confidence": d.get("confidence"),
                "Side": d.get("side", ""),
                "Qty": d.get("quantity", ""),
                "Price": d.get("price", ""),
                "Exec Success": exec_detail.get("success", ""),
                "Reason": d.get("reason", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No scans yet. Start the auto trader above.")

# ---------------------------------------------------------------
# Open positions & trade history
# ---------------------------------------------------------------
positions = runtime["trading_engine"].portfolio.get_open_positions()
st.markdown("### Open Positions")
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True, hide_index=True)
else:
    st.caption("No open positions.")

st.markdown("### Trade History")
trades = runtime["trading_engine"].portfolio.get_realized_pnl() if hasattr(runtime["trading_engine"], "portfolio") else 0
trades_df = pd.DataFrame(runtime["trading_engine"].cache.get_trades(limit=50))
if not trades_df.empty:
    st.dataframe(trades_df[["symbol", "side", "quantity", "entry_price", "exit_price", "pnl", "exit_time", "mode"]], use_container_width=True, hide_index=True)
else:
    st.caption("No completed trades.")