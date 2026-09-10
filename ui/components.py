"""Shared Streamlit UI components and runtime bootstrap."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import bootstrap  # noqa: E402


@st.cache_resource(show_spinner="Initializing AI Trading System...")
def get_runtime():
    return bootstrap()


def ensure_runtime():
    runtime = get_runtime()
    if "ai_trading_runtime" not in st.session_state:
        st.session_state.ai_trading_runtime = runtime
    return st.session_state.ai_trading_runtime


def sidebar_status(runtime):
    from data.universe import StockUniverse

    st.sidebar.markdown("### System Status")
    mode = runtime["config"].get("dhan", {}).get("trading_mode", "paper")
    if runtime["dhan"] is None:
        st.sidebar.warning("Dhan: NOT configured — using Yahoo Finance fallback for data")
        st.sidebar.caption("Paper trading & scanning work via yfinance (.NS). Add DHAN creds for live trading.")
    else:
        st.sidebar.success(f"Dhan: ready ({mode})")
    ai_key = runtime["config"].get("groq", {}).get("model", "llama-3.3-70b-versatile")
    # Check if Groq key actually present
    import os
    has_groq = bool(os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") not in ["", "your_groq_api_key_here"])
    if has_groq:
        st.sidebar.info(f"AI: {ai_key} ✓")
    else:
        st.sidebar.warning("Groq: NOT configured — AI fallback to pure technical signals")

    st.sidebar.markdown("### Watchlist")
    wl_cfg = runtime["config"].get("triggers", {})
    watchlist_name = wl_cfg.get("watchlist", "NIFTY50")
    symbols = StockUniverse(runtime["config"]).resolve(watchlist_name)
    st.sidebar.caption(f"{len(symbols)} symbols in watchlist")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Config:** `config.yaml`  \n**Data cache:** `data/market_cache.db`")


def format_action_badge(action: str):
    action = (action or "HOLD").upper()
    colors = {"BUY": "green", "SELL": "red", "HOLD": "gray", "ERROR": "red"}
    color = colors.get(action, "gray")
    return f":{color}[**{action}**]"


def render_signals_table(results: list):
    import pandas as pd

    rows = []
    for r in results:
        conf = r.get("confidence", 0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        rows.append(
            {
                "Symbol": r.get("symbol", ""),
                "Action": r.get("action", "HOLD"),
                "Confidence": round(conf, 2),
                "Entry": r.get("entry_price", ""),
                "Stop Loss": r.get("stop_loss", ""),
                "Target": r.get("target", ""),
                "Reasoning": r.get("reasoning", ""),
            }
        )
    if not rows:
        st.info("No signals yet. Click **Analyze Watchlist**.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_style():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )