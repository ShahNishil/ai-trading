import os
import sys

# Ensure project root (D:\ai-trading) is on sys.path when Streamlit execs this file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from ui.components import ensure_runtime, render_style, sidebar_status

st.set_page_config(
    page_title="AI Trading - Indian Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_style()
runtime = ensure_runtime()
sidebar_status(runtime)

st.markdown("# AI Trading System for Indian Stock Markets")

st.markdown(
    """
    This system combines **AI-powered analysis (Groq LLaMA)**, **Dhan broker API**,
    and a **backtesting engine** into one platform.

    ### Features
    | Page | What it does |
    |------|---------------|
    | **1. AI Triggers** | AI scans your watchlist and recommends BUY/SELL/HOLD with confidence, entry, stop-loss, target |
    | **2. Auto Trading** | AI autonomously places paper or live trades via Dhan API with risk guardrails |
    | **3. Backtesting** | Backtest AI-suggested or built-in strategies on historical NSE data |

    ### Quick Start
    1. Copy `.env.example` → `.env` and add your **Dhan** + **Groq** API keys
    2. Configure watchlist / risk / capital in `config.yaml`
    3. Navigate using the sidebar
    """
)

st.info("⚠️ **Disclaimer:** Educational tool. NOT investment advice. Paper-trade first.")