import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.trigger_agent import TriggerAgent  # noqa: E402
from data.universe import StockUniverse  # noqa: E402
from ui.components import ensure_runtime, format_action_badge, render_signals_table  # noqa: E402

st.set_page_config(page_title="AI Triggers", page_icon="🔔", layout="wide")

st.markdown("# 🔔 AI Triggers")
st.caption("AI scans your watchlist using technical indicators and recommends the best trades. Works with Dhan API or Yahoo Finance fallback (no Dhan credentials needed).")

runtime = ensure_runtime()
config = runtime["config"]
trigger_cfg = config.get("triggers", {})

# Detect data source
is_dhan = runtime.get("dhan") is not None
is_fallback = not is_dhan
if is_fallback:
    st.info("ℹ️ **Dhan not configured** — using **Yahoo Finance (.NS)** free data for scanning. Add `DHAN_CLIENT_ID`/`DHAN_ACCESS_TOKEN` in `.env` for Dhan data & live trading. Groq AI still works for recommendations.")

with st.sidebar:
    st.markdown("### Analysis Settings")
    watchlist_name = st.selectbox(
        "Watchlist",
        options=["NIFTY50", "NIFTYBANK", "CUSTOM"],
        index=0 if trigger_cfg.get("watchlist") in ("NIFTY50",) else 0,
        help="Choose predefined watchlist or custom symbols",
    )
    timeframe = st.selectbox(
        "Timeframe",
        options=["daily", "60min", "15min", "5min"],
        index=0 if trigger_cfg.get("analysis_timeframe", "daily") == "daily" else 0,
        format_func=lambda t: "Daily" if t == "daily" else t.upper(),
    )
    confidence_threshold = st.slider(
        "Min confidence", 0.0, 1.0, float(trigger_cfg.get("confidence_threshold", 0.50)), 0.05,
        help="Lower = more results. 0.50 recommended for decisive signals."
    )
    lookback_days = st.number_input(
        "Lookback days", min_value=30, max_value=1500, value=int(trigger_cfg.get("lookback_days", 200))
    )
    show_only_actionable = st.checkbox("Show only BUY/SELL", value=False, help="Hide HOLDs, show only actionable")
    analyze_btn = st.button("🔍 Analyze Watchlist", type="primary", use_container_width=True)
    if st.button("Clear cache & Re-scan", help="Clears local candle cache for fresh data"):
        try:
            # Clear candles for this watchlist
            import pathlib, sqlite3
            db = pathlib.Path("D:/ai-trading/data/market_cache.db")
            if db.exists():
                conn = sqlite3.connect(str(db))
                conn.execute("DELETE FROM candles")
                conn.commit()
                conn.close()
                st.success("Cache cleared — next scan will fetch fresh data")
        except Exception as e:
            st.error(str(e))

main_col, side_col = st.columns([3, 1])

with side_col:
    st.markdown("### Active Watchlist")
    universe = StockUniverse(config)
    symbols = universe.resolve(watchlist_name, [])
    st.caption(f"{len(symbols)} symbols — {'Dhan' if is_dhan else 'Yahoo .NS'} mode")
    for s in symbols:
        st.markdown(f"- {s['symbol']}")
    st.caption(f"Data: {'Dhan API' if is_dhan else 'Yahoo Finance'} | AI: {'Groq ✓' if runtime.get('ai_engine') else 'No Groq key'}")

if analyze_btn:
    # No longer block on missing Dhan — allow yfinance fallback
    if runtime["data_fetcher"] is None:
        st.error("Data fetcher not initialized. Check logs.")
    else:
        agent = TriggerAgent(runtime["ai_engine"], runtime["data_fetcher"], config)
        symbols = universe.resolve(watchlist_name)
        # Update lookback in config for this run
        config["triggers"]["lookback_days"] = int(lookback_days)
        progress = st.progress(0, text="Starting scan...")
        results = []
        for i, item in enumerate(symbols):
            sym = item.get("symbol", "")
            progress.progress((i) / len(symbols), text=f"Analyzing {sym} ({i+1}/{len(symbols)})...")
            try:
                r = agent.analyze_symbol(sym, item.get("security_id", ""), item.get("exchange", "NSE_EQ"), timeframe, int(lookback_days))
                results.append(r)
            except Exception as e:
                results.append({"symbol": sym, "action": "ERROR", "confidence": 0.0, "reasoning": str(e)})
            time.sleep(0.05)  # small pause to show progress
        progress.progress(1.0, text="Done!")
        # Also run ranked watchlist sort
        def rank_key(r):
            is_actionable = 1 if r.get("action") in ("BUY", "SELL") else 0
            return (is_actionable, float(r.get("confidence", 0) or 0))
        results.sort(key=rank_key, reverse=True)
        st.session_state["trigger_results"] = results
        # Auto-lower threshold if too few results
        actionable = sum(1 for r in results if r.get("action") in ("BUY", "SELL"))
        if actionable == 0 and len(results) > 0:
            st.warning(f"Scan found 0 BUY/SELL — all {len(results)} were HOLD/ERROR. Try lowering confidence threshold to 0.40 or check data. Showing top by confidence anyway.")

results = st.session_state.get("trigger_results", [])

if results:
    # Filter: always keep BUY/SELL, only filter HOLDs by threshold
    filterable = []
    for r in results:
        try:
            conf = float(r.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        action = r.get("action", "HOLD")
        if action in ("BUY", "SELL", "ERROR"):
            filterable.append(r)
        elif action == "HOLD" and conf >= confidence_threshold and not show_only_actionable:
            filterable.append(r)
        elif show_only_actionable:
            continue  # skip HOLDs
        elif not show_only_actionable:
            # include HOLDs above threshold
            pass

    # If show_only_actionable, filterable already only has BUY/SELL/ERROR
    if show_only_actionable:
        display = [r for r in results if r.get("action") in ("BUY", "SELL", "ERROR")]
        # Sort by confidence
        display.sort(key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)
        filterable = display

    st.markdown(f"### Results ({len(filterable)} shown / {len(results)} scanned)")
    cols = st.columns(5)
    buy_count = sum(1 for r in results if r.get("action") == "BUY")
    sell_count = sum(1 for r in results if r.get("action") == "SELL")
    hold_count = sum(1 for r in results if r.get("action") == "HOLD")
    err_count = sum(1 for r in results if r.get("action") == "ERROR")
    cols[0].metric("BUY", buy_count)
    cols[1].metric("SELL", sell_count)
    cols[2].metric("HOLD", hold_count)
    cols[3].metric("ERROR", err_count)
    cols[4].metric("Total", len(results))

    # Show source breakdown
    ai_cnt = sum(1 for r in results if r.get("source") == "ai")
    tech_cnt = sum(1 for r in results if r.get("source") == "technical")
    st.caption(f"Sources: AI {ai_cnt} | Technical fallback {tech_cnt} | Dhan: {'✓' if is_dhan else 'Yahoo fallback'}")

    render_signals_table(filterable if filterable else results[:5])

    st.markdown("### Top Picks (ranked by confidence)")
    # Show top 5 regardless of HOLD, to always give something
    top = sorted(results, key=lambda x: float(x.get("confidence", 0) or 0), reverse=True)[:5]
    for r in top:
        action = r.get("action", "HOLD")
        src = r.get("source", "?")
        st.markdown(
            f"{format_action_badge(action)} **{r['symbol']}** — conf {float(r.get('confidence', 0)):.0%} ({src}) "
            f"· entry ₹{r.get('entry_price', '')} · SL ₹{r.get('stop_loss', '')} · target ₹{r.get('target', '')}"
        )
        st.caption(r.get("reasoning", ""))
        # Debug expander with technical details
        with st.expander(f"🔍 Technical details for {r['symbol']}"):
            tech = r.get("technical", {})
            st.json(tech if tech else {"note": "No technical details"})
            if "llm_reasoning" in r:
                st.markdown(f"**LLM reasoning:** {r['llm_reasoning']}")

    # Also show BUY-specific section
    buys = [r for r in filterable if r.get("action") == "BUY"][:5]
    if buys and len(buys) != len(top):
        st.markdown("### BUY Signals")
        for r in buys:
            st.markdown(
                f"{format_action_badge('BUY')} **{r['symbol']}** — conf {float(r.get('confidence', 0)):.0%} "
                f"· entry ₹{r.get('entry_price', '')} · SL ₹{r.get('stop_loss', '')} · target ₹{r.get('target', '')} ({r.get('source','')})"
            )
            st.caption(r.get("reasoning", ""))

    try:
        import pandas as pd
        hist_df = pd.DataFrame(
            [
                {
                    "symbol": r.get("symbol"),
                    "action": r.get("action"),
                    "confidence": r.get("confidence"),
                    "entry": r.get("entry_price"),
                    "stop_loss": r.get("stop_loss"),
                    "target": r.get("target"),
                    "source": r.get("source"),
                    "reasoning": r.get("reasoning", "")[:200],
                }
                for r in results
            ]
        )
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download results (CSV)", csv, "ai_triggers.csv", "text/csv")
    except Exception:
        pass

    if err_count > 0:
        st.warning(f"{err_count} symbols errored — check security_id or network. Errors are shown in table as ERROR.")
        with st.expander("Error details"):
            for r in results:
                if r.get("action") == "ERROR":
                    st.write(f"{r['symbol']}: {r.get('reasoning')}")

else:
    st.info("Select settings and click **Analyze Watchlist** to get AI triggers. Tip: If you see no results, lower threshold to 0.40 or use Yahoo fallback mode (no Dhan needed).")
