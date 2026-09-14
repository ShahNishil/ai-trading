import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from backtest.runner import BacktestRunner  # noqa: E402
from backtest.visualizer import BacktestVisualizer  # noqa: E402
from core.options import annualized_volatility, bs_price  # noqa: E402
from data.derivatives import DerivativeUniverse  # noqa: E402
from strategies.derivatives import (  # noqa: E402
    create_derivative_strategy,
    list_derivative_strategies,
    payoff_table,
)
from ui.components import ensure_runtime  # noqa: E402

st.set_page_config(page_title="Derivatives (F&O)", page_icon="⚡", layout="wide")

st.markdown("# ⚡ Derivatives (F&O)")
st.caption(
    "Index & stock futures/options — scrip-master universe, Black-Scholes option "
    "backtesting, margin futures backtesting and multi-leg strategy tools. "
    "Real F&O data/execution need Dhan; option backtests run on underlying spot via yfinance."
)

runtime = ensure_runtime()
config = runtime["config"]
deriv_cfg = config.get("derivatives", {})
data_fetcher = runtime["data_fetcher"]


@st.cache_resource(show_spinner="Loading F&O scrip master…")
def _cached_universe():
    return DerivativeUniverse(config=config)


universe = _cached_universe()
desc = universe.describe()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### F&O Universe")
    market = st.radio("Market", ["IDX", "STK"], index=0 if deriv_cfg.get("market", "IDX") == "IDX" else 1,
                      format_func=lambda x: "Index (NIFTY…)" if x == "IDX" else "Stock")
    underlyings = universe.list_underlyings(market)
    default_und = deriv_cfg.get("default_underlying", "NIFTY").upper()
    idx = underlyings.index(default_und) if default_und in underlyings else 0
    underlying = st.selectbox("Underlying", underlyings, index=idx)

    if desc["master_available"]:
        st.success(f"Scrip master loaded: {desc['rows']:,} contracts")
    else:
        st.warning("Scrip master unavailable — using built-in fallback chain (BS backtests still work).")

    st.markdown("---")
    st.caption("**Underlyings (IDX):** " + ", ".join(desc["index_underlyings"]))
    st.caption("**Underlyings (STK):** " + ", ".join(desc["stock_underlyings"]))

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _spot_series(symbol: str, days: int = 730):
    if not data_fetcher or not symbol:
        return None
    try:
        df = data_fetcher.fetch_daily(symbol, "0", days=days)
        if df is not None and not df.empty:
            return df
    except Exception:
        return None
    return None


def _nearest_thursdays(n: int = 6):
    import datetime as _dt

    out = []
    d = _dt.date.today()
    while len(out) < n:
        d += _dt.timedelta(days=1)
        if d.weekday() == 3:
            out.append(d)
    return out


tabs = st.tabs(["Universe", "Option Chain", "Option Backtest", "Futures Backtest", "Payoff Simulator"])

# ======================================================================
# TAB 1 — Universe
# ======================================================================
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Scrip master", "Loaded" if desc["master_available"] else "Fallback")
    c2.metric("Contracts", f"{desc['rows']:,}")
    c3.metric("Spot ticker", universe.spot_symbol(underlying, market) or "—")

    st.markdown("#### Expiries & lot sizes")
    exps = universe.expiries(underlying, market)
    if not exps:
        exps = [pd.Timestamp(d) for d in _nearest_thursdays(6)]
    rows = []
    for e in exps[:10]:
        rows.append({
            "Expiry": str(e.date()),
            "Lot size": universe.lot_size(underlying, market, e),
            "Strike step": universe.strike_step(underlying, market, e),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Nearest contract (Dhan)")
    col1, col2 = st.columns(2)
    with col1:
        fut = universe.futures_contract(underlying, market)
        fut_d = fut.to_dict()
        st.markdown(f"**Futures** — {fut_d['trading_symbol'] or fut_d['key']}")
        st.json(fut_d)
    with col2:
        opt = universe.option_contract(underlying, market)
        opt_d = opt.to_dict()
        st.markdown(f"**ATM option** — {opt_d['trading_symbol'] or opt_d['key']}")
        st.json(opt_d)

# ======================================================================
# TAB 2 — Option Chain
# ======================================================================
with tabs[1]:
    spot_symbol = universe.spot_symbol(underlying, market)
    spot_df = _spot_series(spot_symbol) if spot_symbol else None
    spot_price = float(spot_df["close"].iloc[-1]) if spot_df is not None and len(spot_df) else None

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        c_expiry = st.selectbox(
            "Expiry", exps,
            format_func=lambda e: str(pd.Timestamp(e).date()),
        )
    with cc2:
        st.metric("Latest spot", f"{spot_price:,.2f}" if spot_price else "N/A")

    if spot_price:
        chain = universe.option_chain(underlying, c_expiry, market, spot=spot_price, n_strikes=5)
    else:
        chain = universe.option_chain(underlying, c_expiry, market)

    if chain is None or chain.empty:
        st.info("No chain available for this underlying/expiry.")
    else:
        show_cols = [c for c in chain.columns if not str(c).endswith("custom_symbol")]
        st.dataframe(chain[show_cols], use_container_width=True, hide_index=True)
        if not desc["master_available"]:
            st.caption("⚠️ Synthetic chain — security IDs are empty. Real contracts need the Dhan scrip master.")

# ======================================================================
# TAB 3 — Option Backtest
# ======================================================================
with tabs[2]:
    all_strats = list_derivative_strategies()
    opt_strats = [s for s in all_strats if s["market"] == "option"]
    names = [s["name"] for s in opt_strats]
    default_name = deriv_cfg.get("default_strategy", "long_straddle")
    sel_name = st.selectbox("Strategy", names, index=names.index(default_name) if default_name in names else 0,
                            format_func=lambda n: n.replace("_", " ").title())
    meta = next(s for s in opt_strats if s["name"] == sel_name)
    st.caption(meta["description"])

    with st.expander("Parameters", expanded=False):
        params = {}
        for pname, pval in meta["default_params"].items():
            if isinstance(pval, float):
                params[pname] = st.number_input(pname, 0.0, max(10.0, float(pval) * 3), float(pval), 0.1)
            elif isinstance(pval, int):
                params[pname] = st.number_input(pname, 0, 100, int(pval), 1)
            else:
                params[pname] = pval

    with st.expander("Backtest Settings", expanded=False):
        bt1, bt2 = st.columns(2)
        with bt1:
            days = st.slider("Lookback (days)", 120, 1500, 730, 30)
            initial_capital = st.number_input("Initial capital (₹)", 10_000, 10_000_000,
                                              int(deriv_cfg.get("initial_capital", 500000)), 10_000)
            commission = st.slider("Commission %", 0.0, 0.5, float(config.get("backtest", {}).get("commission_pct", 0.03)), 0.01)
        with bt2:
            iv_pct = st.slider("IV % (0 = realised vol)", 0.0, 60.0, float(deriv_cfg.get("iv_pct", 0.0)), 1.0)
            risk_free = st.slider("Risk-free rate %", 0.0, 15.0, float(deriv_cfg.get("risk_free_rate_pct", 7.0)), 0.5)

    run_btn = st.button("🏁 Run Option Backtest", type="primary")

    if run_btn:
        runner = BacktestRunner(data_fetcher, runtime["cache"], config, universe)
        with st.spinner(f"Backtesting {sel_name} on {underlying} ({market})…"):
            result = runner.run_derivative(
                strategy_name=sel_name,
                underlying=underlying,
                kind=market,
                params=params,
                spot_symbol=spot_symbol or "",
                lookback_days=days,
                iv_pct=iv_pct,
                initial_capital=initial_capital,
                commission_pct=commission,
                risk_free_rate_pct=risk_free,
            )
        if "error" in result and result.get("error"):
            st.error(result["error"])
        else:
            st.session_state["deriv_result"] = result

    result = st.session_state.get("deriv_result")
    if result and result.get("strategy") == sel_name and result.get("instrument_type") == "option":
        metrics = result.get("metrics", {})
        mc = st.columns(6)
        mc[0].metric("Total Return", f"{metrics.get('total_return_pct', 0)}%")
        mc[1].metric("Sharpe", metrics.get("sharpe", 0))
        mc[2].metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0)}%")
        mc[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0)}%")
        mc[4].metric("Profit Factor", metrics.get("profit_factor", 0))
        mc[5].metric("Trades", metrics.get("num_trades", 0))

        viz = BacktestVisualizer(result)
        st.plotly_chart(viz.equity_chart(), use_container_width=True)
        st.plotly_chart(viz.price_chart_with_trades(), use_container_width=True)

        trades = result.get("trades", [])
        if trades:
            st.markdown("#### Trades")
            trows = []
            for t in trades:
                leg_str = "; ".join(l.get("label", "") for l in getattr(t, "legs", []) or [])
                trows.append({
                    "Entry": t.entry_time.date(),
                    "Expiry": t.expiry,
                    "Exit": t.exit_time.date() if t.exit_time else "",
                    "Net premium": round(float(t.entry_price), 2),
                    "Qty (lots·size)": t.quantity,
                    "P&L": round(float(t.pnl), 2),
                    "Exit": t.exit_reason,
                    "Legs": leg_str,
                })
            st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

# ======================================================================
# TAB 4 — Futures Backtest
# ======================================================================
with tabs[3]:
    fut_strats = [s for s in all_strats if s["market"] == "futures"]
    f_names = [s["name"] for s in fut_strats]
    f_name = st.selectbox("Futures strategy", f_names, format_func=lambda n: n.replace("_", " ").title())
    f_meta = next((s for s in fut_strats if s["name"] == f_name), None)

    with st.expander("Parameters", expanded=False):
        f_params = {}
        for pname, pval in (f_meta or {}).get("default_params", {}).items():
            if isinstance(pval, float):
                f_params[pname] = st.number_input(pname, 0.5, 200.0, float(pval), 0.5)
            elif isinstance(pval, int):
                f_params[pname] = st.number_input(pname, 1, 200, int(pval), 1)
            else:
                f_params[pname] = pval

    with st.expander("Backtest Settings", expanded=False):
        c4a, c4b = st.columns(2)
        with c4a:
            f_days = st.slider("Lookback (days)", 120, 1500, 730, 30, key="fdays")
            f_cap = st.number_input("Initial capital (₹)", 10_000, 10_000_000,
                                    int(deriv_cfg.get("initial_capital", 500000)), 10_000, key="fcap")
            f_comm = st.slider("Commission %", 0.0, 0.5, 0.03, 0.01, key="fcomm")
        with c4b:
            f_margin = st.slider("Margin % of notional", 5.0, 50.0, float(deriv_cfg.get("futures_margin_pct", 12.0)), 0.5)
            f_usage = st.slider("Margin usage %", 10.0, 100.0, float(deriv_cfg.get("futures_margin_usage_pct", 60.0)), 5.0)

    f_spot_symbol = universe.spot_symbol(underlying, market)
    lot_size = universe.lot_size(underlying, market)
    st.caption(f"Resolving {underlying} futures — spot: `{f_spot_symbol}` · lot size: {lot_size}")

    run_fut = st.button("🏁 Run Futures Backtest", type="primary")
    if run_fut:
        runner = BacktestRunner(data_fetcher, runtime["cache"], config, universe)
        with st.spinner(f"Backtesting {f_name} on {underlying} futures…"):
            fres = runner.run_derivative(
                strategy_name=f_name,
                underlying=underlying,
                kind=market,
                params=f_params,
                spot_symbol=f_spot_symbol or "",
                lookback_days=f_days,
                lot_size=lot_size,
                initial_capital=f_cap,
                commission_pct=f_comm,
                margin_pct=f_margin,
                margin_usage_pct=f_usage,
            )
        if "error" in fres and fres.get("error"):
            st.error(fres["error"])
        else:
            st.session_state["deriv_futures_result"] = fres

    fres = st.session_state.get("deriv_futures_result")
    if fres and fres.get("strategy") == f_name and fres.get("instrument_type") == "future":
        metrics = fres.get("metrics", {})
        mc = st.columns(6)
        mc[0].metric("Total Return", f"{metrics.get('total_return_pct', 0)}%")
        mc[1].metric("Sharpe", metrics.get("sharpe", 0))
        mc[2].metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0)}%")
        mc[3].metric("Win Rate", f"{metrics.get('win_rate_pct', 0)}%")
        mc[4].metric("Profit Factor", metrics.get("profit_factor", 0))
        mc[5].metric("Trades", metrics.get("num_trades", 0))
        viz = BacktestVisualizer(fres)
        st.plotly_chart(viz.equity_chart(), use_container_width=True)
        st.plotly_chart(viz.price_chart_with_trades(), use_container_width=True)
        trades = fres.get("trades", [])
        if trades:
            trows = [{
                "Entry": t.entry_time.date(),
                "Exit": t.exit_time.date() if t.exit_time else "",
                "Entry Price": round(float(t.entry_price), 2),
                "Exit Price": round(float(t.exit_price or 0), 2),
                "Qty": t.quantity,
                "P&L": round(float(t.pnl), 2),
                "Reason": t.exit_reason,
            } for t in trades]
            st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

# ======================================================================
# TAB 5 — Payoff Simulator
# ======================================================================
with tabs[4]:
    p_name = st.selectbox("Structure", [s["name"] for s in opt_strats], key="payoff_strat",
                          format_func=lambda n: n.replace("_", " ").title())
    p_meta = next(s for s in opt_strats if s["name"] == p_name)
    p_spot = spot_df if spot_symbol else _spot_series(spot_symbol)
    if p_spot is not None and len(p_spot):
        p_spot_price = float(p_spot["close"].iloc[-1])
        vol = annualized_volatility(p_spot["close"].values) / 100.0
        if not vol or vol <= 0.02:
            vol = 0.12
    else:
        p_spot_price = 25000.0
        vol = 0.12

    with st.expander("Parameters", expanded=False):
        p_params = {}
        for pname, pval in p_meta["default_params"].items():
            if pname == "entry_days_before_expiry":
                continue
            if isinstance(pval, float):
                p_params[pname] = st.number_input(pname, 0.0, max(10.0, float(pval) * 3), float(pval), 0.1, key=f"pp_{pname}")
            elif isinstance(pval, int):
                p_params[pname] = st.number_input(pname, 0, 100, int(pval), 1, key=f"pp_{pname}")
            else:
                p_params[pname] = pval

    willing = st.columns(3)
    fld = willing[0].slider("Premium spot", 0.75, 1.25, 1.0, 0.01)
    fds = willing[1].slider("Volatility %", 5.0, 80.0, vol * 100.0, 1.0)
    fdt = willing[2].slider("Days to expiry", 1, 30, 7, 1)

    step = universe.strike_step(underlying, market)
    structure = create_derivative_strategy(p_name, p_params).build_legs(p_spot_price * fld, step, p_params)
    premiums = {}
    rows = []
    for leg in structure.legs:
        try:
            prem = float(bs_price(p_spot_price * fld, leg.strike, fdt / 365.0, config.get("derivatives", {}).get("risk_free_rate_pct", 7.0) / 100.0, fds / 100.0, leg.option_type))
        except Exception:
            prem = 0.0
        premiums[f"{leg.option_type}-{leg.strike}"] = prem
        rows.append({
            "Leg": f"{leg.side} {leg.option_type} {leg.strike:,.0f}",
            "Strike": leg.strike,
            "Premium (₹)": round(prem, 2),
        })
        if leg.side == "SELL":
            rows[-1]["Premium (₹)"] *= -1

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    pt = payoff_table(structure.legs, premiums)
    if pt:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pt["prices"], y=pt["payoffs"], mode="lines", name="Payoff at expiry",
                                 line=dict(color="#0066cc", width=2)))
        fig.add_hline(y=0, line_color="gray", line_dash="dash")
        for b in pt["breakevens"]:
            fig.add_vline(x=b, line_color="orange", line_dash="dot")
        fig.update_layout(title="P&L at Expiry", template="plotly_white", height=420,
                          margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Profit", f"₹{pt['max_profit']:,.0f}")
        c2.metric("Max Loss", f"₹{pt['max_loss']:,.0f}")
        c3.metric("Breakevens", ", ".join(f"{b:,.0f}" for b in pt["breakevens"]) or "—")