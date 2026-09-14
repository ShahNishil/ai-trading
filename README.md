# AI Trading System — Indian Stock Markets

> AI-powered automated trading platform for NSE/BSE built on **DhanHQ Free API** + **Groq Free LLM** (Llama 3.3 70B) — with paper/live trading, AI triggers, and strategy backtesting.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start (Windows)](#quick-start-windows)
- [Configuration](#configuration)
  - [Environment Variables (.env)](#environment-variables-env)
  - [config.yaml](#configyaml)
- [Running the App](#running-the-app)
- [The 3 Features in Detail](#the-3-features-in-detail)
- [Technical Indicators](#technical-indicators)
- [Strategies & Backtesting](#strategies--backtesting)
- [Derivatives (F&O)](#derivatives-fo)
- [Risk Management](#risk-management)
- [Data & Caching](#data--caching)
- [API References](#api-references)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)

---

## Features

### 1. AI-Based Triggers — `AI recommends the best trades`
- Scans your watchlist (NIFTY50 / NIFTYBANK / custom) via **Dhan Historical + Live Data**.
- Computes **54 technical indicators** (pure `numpy`/`pandas`, no TA-Lib/pandas-ta).
- Sends a compact indicator summary to **Groq Llama 3.3 70B** with a strict JSON system prompt (Senior Technical Analyst role).
- Returns per-stock: `BUY / SELL / HOLD` + `confidence (0-1)` + `entry_price` + `stop_loss` + `target` + `reasoning` (2-3 sentences).
- Streamlit dashboard with confidence-threshold filter, top-picks list, and CSV download.

### 2. Auto AI Trading — `AI trades directly via Dhan`
- Autonomous loop that re-scans every `scan_interval_seconds` (default 300s) during market hours.
- LLM acts as a **Portfolio Manager** — receives indicator summary + portfolio state + available capital.
- Decision: `ENTRY / EXIT / HOLD` with quantity, price, order type.
- Execution via `dhanhq` SDK:
  - **Paper mode** — simulated fills at last price, persisted to SQLite (`data/market_cache.db`).
  - **Live mode** — real orders via `DhanClient.place_order()` with `INTR/CNC/MTF` and `LIMIT/MARKET`.
- Guardrails: max positions, risk-per-trade, daily loss halt, trailing stop/target, kill-switch in UI.

### 3. Strategy Backtesting — `Backtest AI or custom strategies`
- Bar-by-bar engine (`backtest/engine.py`) with **slippage + commission** simulation.
- Computes: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor, Expectancy, Volatility + Buy & Hold benchmark.
- Built-in strategies: `momentum` (EMA+MACD+RSI), `mean_reversion` (RSI+Bollinger), `breakout` (Donchian+SuperTrend+Volume), `ai` (LLM-generated rules with momentum fallback).
- Charts: equity curve, drawdown, price with trade markers, monthly returns — via Plotly.
- Grid-search optimizer + AI strategy generator (`agents/strategy_agent.py`).

### 4. Derivatives (F&O) — `Futures & Options, pro-style`
- **Universe**: full Dhan F&O scrip master (79k+ contracts) — index & stock futures/options with real lot sizes, strike steps, expiries and security IDs; built-in fallback table for offline use.
- **Option backtesting**: Black-Scholes priced strategies (straddle, strangle, verticals, iron condor) on the underlying spot series — works even without Dhan via Yahoo Finance.
- **Futures backtesting**: whole-lot, margin-aware (`margin_pct`, `margin_usage_pct`) trend-following on futures.
- **Execution layer**: F&O-friendly orders (`exchange_segment`, `instrument_type`) for paper/live, option-structure multi-leg entries, risk sizing in lots.
- Dedicated UI page (`ui/pages/4_derivatives.py`): universe explorer, option chain viewer, payoff simulator.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (ui/)                       │
│   app.py (home)  │  1_ai_triggers.py  │  2_auto_trade.py  │  3_backtest.py  │  4_derivatives.py  │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│  Agents (agents/)                                               │
│  TriggerAgent  │  AutoTradeAgent  │  StrategyAgent              │
│  ─────────── prompts.py (system prompts) ───────────            │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│  Core (core/)                                                   │
│  DhanClient │ AIEngine (Groq/OpenAI SDK) │ IndicatorEngine │ Signals │ RiskManager │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│  Trading (trading/)  │  Strategies (strategies/)  │  Backtest (backtest/) │
│  TradingEngine │ OrderManager │ Portfolio          │  Momentum / MR / Breakout / AI  │  Engine │ Runner │ Visualizer │
└──────────────┬──────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────────┐
│  Data (data/)                                                   │
│  DataFetcher (Dhan API) │ DataCache (SQLite) │ StockUniverse (watchlists) │
└─────────────────────────────────────────────────────────────────┘
               │
        DhanHQ API (api.dhan.co/v2)  +  Groq API (api.groq.com/openai/v1)
```

---

## Project Structure

```
ai-trading/
├── agents/               # AI agent workflows
│   ├── trigger_agent.py  # Feature 1 — watchlist scan → recommendations
│   ├── auto_trade_agent.py # Feature 2 — autonomous trading loop
│   ├── strategy_agent.py # AI strategy generator for backtesting
│   └── prompts.py        # System prompts (analyst / portfolio manager / designer)
├── backtest/
│   ├── engine.py         # Bar-by-bar simulator + metrics (compute_metrics); futures lot/margin mode
│   ├── derivatives.py    # Option (Black-Scholes) + futures backtest engines
│   ├── runner.py         # Ties data fetch + strategy + persistence (run_derivative for F&O)
│   └── visualizer.py     # Plotly equity / drawdown / trade charts
├── core/
│   ├── dhan_client.py    # Wrapper over dhanhq SDK (orders, quotes, historical)
│   ├── ai_engine.py      # Groq/OpenAI-compatible client + JSON parsing
│   ├── indicators.py     # 54 indicators — pure numpy/pandas, no numba
│   ├── options.py        # Black-Scholes pricing, Greeks, implied vol, realised vol
│   ├── signals.py        # Rule-based SignalGenerator (momentum/MR/breakout scores)
│   └── risk_manager.py   # Position sizing, SL/target, trailing, daily-loss halt
├── data/
│   ├── fetcher.py        # Historical (daily/intraday/derivative) + quote fetch with cache
│   ├── cache.py          # SQLite: candles, orders, trades, signals, backtests
│   ├── universe.py       # Watchlist resolver (NIFTY50, NIFTYBANK, custom)
│   ├── derivatives.py    # F&O instrument model + universe (Dhan scrip-master CSV)
│   ├── scrip_master_fno.csv  # Cached F&O contract master (gitignored)
│   └── market_cache.db   # Created on first run (gitignored)
├── strategies/
│   ├── base.py           # Abstract BaseStrategy
│   ├── momentum.py       # EMA cross + MACD + RSI
│   ├── mean_reversion.py # RSI + Bollinger
│   ├── breakout.py       # Donchian + SuperTrend + volume
│   ├── ai_strategy.py    # LLM rule-based with momentum fallback
│   ├── derivatives.py    # Option structures (straddle/strangle/spreads/condor) + futures_trend
│   └── registry.py       # Strategy registry + factory
├── trading/
│   ├── engine.py         # Unified TradingEngine (paper/live)
│   ├── order_manager.py  # Order lifecycle (paper fill vs Dhan API)
│   └── portfolio.py      # Position/PnL tracking via cache
├── ui/
│   ├── app.py            # Streamlit home + navigation
│   ├── components.py     # Shared runtime bootstrap + UI helpers
│   └── pages/
│       ├── 1_ai_triggers.py
│       ├── 2_auto_trade.py
│       ├── 3_backtest.py
│       └── 4_derivatives.py
├── config.yaml           # Central configuration (watchlists, risk, backtest)
├── requirements.txt      # Python dependencies
├── .env.example          # Template for API keys
├── .env                  # Your real keys (gitignored) — create from .env.example
├── run.py                # CLI bootstrap + health check
├── ui_state.py           # Shared state for auto-trade background thread
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10+** (tested on 3.14) | Download from python.org |
| **Dhan Account** | Free — sign up at https://dhan.co → enable **API Access** → get `client_id` + `access_token` |
| **Groq API Key** | Free tier — https://console.groq.com → Create API Key → 30 req/min on Llama 3.3 70B |
| **Windows / macOS / Linux** | Instructions below are for Windows PowerShell |

---

## Quick Start (Windows)

### 1. Clone / open the project

```powershell
cd D:\ai-trading
```

### 2. Create and activate virtual environment

```powershell
# Create venv
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# If execution policy blocks it:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1

# Alternative (CMD):
# venv\Scripts\activate.bat
```

You should see `(venv)` prefix in your prompt.

### 3. Upgrade pip and install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Expected packages:
```
dhanhq>=2.2.0
openai>=1.0.0
groq>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
matplotlib>=3.7.0
streamlit>=1.30.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

> **Note:** `pandas-ta` is intentionally NOT used — it requires `numba` which has no Python 3.14 wheel. All indicators are implemented in pure `numpy`/`pandas` (`core/indicators.py`) and work on Python 3.10–3.14 without compilation.

### 4. Configure API keys

```powershell
copy .env.example .env
notepad .env
```

Fill in:

```ini
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Verify setup

```powershell
python run.py
```

Expected output (with keys configured):

```
============================================================
AI Trading System - Indian Stock Markets
============================================================
Trading mode: paper
Dhan: connected
AI: llama-3.3-70b-versatile
------------------------------------------------------------
Launch the dashboard:
  streamlit run ui/app.py
============================================================
```

If keys are missing you will see `[warn]` messages — the app still boots but AI/Dhan features will be disabled until you add them.

### 6. Launch the dashboard

```powershell
streamlit run ui/app.py
```

Open http://localhost:8501 in your browser. Use the sidebar to navigate:

- **AI Triggers** → pick watchlist + timeframe → **Analyze Watchlist**
- **Backtesting** → pick symbol + strategy + params → **Run Backtest**
- **Derivatives** → explore the F&O universe, view option chains, run option/futures backtests and a payoff simulator
- **Auto Trading** → set risk → choose `paper`/`live` → **Start Auto Trader**

### 7. Deactivate venv when done

```powershell
deactivate
```

---

## Configuration

### Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `DHAN_CLIENT_ID` | Yes (for trading/data) | From Dhan → API Access |
| `DHAN_ACCESS_TOKEN` | Yes (for trading/data) | JWT token from Dhan |
| `GROQ_API_KEY` | Yes (for AI features) | From console.groq.com |

Get Dhan credentials: Dhan App → Profile → API Access → Generate.  
Get Groq key: https://console.groq.com/keys → Create.

### config.yaml

Central config — edit to suit your style:

```yaml
dhan:
  trading_mode: paper          # paper (simulated) | live (real orders)
  ip_address: ""               # optional IP for Dhan

groq:
  model: "openai/gpt-oss-20b"  # verified free tier (llama-3.3-70b-versatile was decommissioned)
  temperature: 0.1
  max_tokens: 2048

triggers:
  watchlist: "NIFTY50"         # NIFTY50 | NIFTYBANK | custom
  custom_symbols: []           # when watchlist=custom
  confidence_threshold: 0.65
  analysis_timeframe: "daily"  # daily | 60min | 15min
  lookback_days: 200

auto_trade:
  enabled: false
  max_positions: 5
  risk_per_trade_pct: 2.0
  max_daily_loss_pct: 5.0
  min_confidence: 0.70
  scan_interval_seconds: 300
  stop_loss_pct: 2.0
  target_pct: 4.0
  trailing_stop_pct: 1.5
  product_type: "INTR"         # INTR | CNC | MTF
  order_type: "LIMIT"          # LIMIT | MARKET

backtest:
  initial_capital: 100000
  default_timeframe: "daily"
  commission_pct: 0.03
  slippage_pct: 0.05
  default_strategy: "momentum"

derivatives:
  default_underlying: "NIFTY"   # NIFTY | BANKNIFTY | FINNIFTY | MIDCPNIFTY | any stock F&O
  market: "IDX"                 # IDX (index F&O) | STK (stock F&O)
  default_strategy: "long_straddle"
  iv_pct: 0.0                   # 0 = realised vol from underlying data (annualised % otherwise)
  risk_free_rate_pct: 7.0
  entry_days_before_expiry: 5   # enter option structures N days before expiry
  risk_per_trade_pct: 0.75      # % of capital risked per option structure
  max_lots_per_trade: 5
  futures_margin_pct: 12.0      # futures margin as % of notional
  futures_margin_usage_pct: 60.0
  product_type: "INTR"          # INTR | CNC | MTF (F&O order product type)

watchlists:
  NIFTY50: [...]
  NIFTYBANK: [...]
```

Add your own watchlist by appending under `watchlists:`:

```yaml
  MYLIST:
    - { symbol: "RELIANCE", security_id: "1602", exchange: "NSE_EQ" }
    - { symbol: "TCS", security_id: "11536", exchange: "NSE_EQ" }
```

Security IDs are Dhan instrument IDs — fetch via `DhanClient.fetch_security_list("compact")` or from https://images.dhan.co/api-data/api-scrip-master.csv

---

## Running the App

| Command | Purpose |
|---------|---------|
| `python run.py` | Health check — prints mode + connection status |
| `streamlit run ui/app.py` | Launch full dashboard on http://localhost:8501 |
| `streamlit run ui/app.py --server.port 8599` | Custom port |
| `streamlit run ui/app.py --server.headless true` | Headless/server mode |

**PowerShell venv workflow:**

```powershell
.\venv\Scripts\Activate.ps1
python run.py
streamlit run ui/app.py
# ... use app ...
deactivate
```

---

## The 3 Features in Detail

### Feature 1 — AI Triggers (`ui/pages/1_ai_triggers.py`)

1. Select watchlist + timeframe + confidence threshold in sidebar.
2. Click **Analyze Watchlist** → for each symbol:
   - `DataFetcher` pulls OHLCV (Dhan `historical_daily_data` / `intraday_minute_data`, cached in SQLite).
   - `IndicatorEngine.compute_all()` enriches with 54 indicators.
   - `IndicatorEngine.latest_summary()` builds a compact JSON blob.
   - `TriggerAgent` sends it to Groq with `TRIGGER_AGENT_SYSTEM_PROMPT` (Senior Technical Analyst).
   - LLM must return strict JSON: `{symbol, action, confidence, entry_price, stop_loss, target, timeframe_hours, reasoning}`.
3. Results filtered by `confidence_threshold`, ranked, displayed as table + top picks. CSV export available.

**Prompt rules (enforced):** no invented prices, confluence scoring, RSI>70 overbought, ADX>25 trending, MACD histogram, volume expansion, candlestick confluence, confidence ≥0.65 for BUY/SELL else HOLD.

### Feature 2 — Auto AI Trading (`ui/pages/2_auto_trade.py` + `agents/auto_trade_agent.py`)

1. Configure risk in sidebar (mode, max positions, risk %, min confidence, scan interval).
2. Click **Start Auto Trader** → spawns a daemon thread: `AutoTradeAgent.run_forever(watchlist)`.
3. Every `scan_interval_seconds`:
   - For each symbol: fetch data → build context (indicators + portfolio state + capital) → `AIEngine.analyze_indicators(AUTO_TRADE_AGENT_SYSTEM_PROMPT, context)`.
   - LLM returns `{symbol, decision: ENTRY|EXIT|HOLD, side, quantity, price, order_type, confidence, reason}`.
   - `AutoTradeAgent.execute()` checks `min_confidence` + `RiskManager.can_open_position()` → `TradingEngine.enter_position()` or `close_position()`.
   - Paper: immediate fill at price, persisted to cache. Live: `DhanClient.place_order()` → real order.
4. `TradingEngine.update_positions_with_prices()` handles trailing stops/targets.
5. **Stop** button sets `AutoTradeState.auto_trade_stop` event → thread exits. State visible in UI (status, last scan time, last error, open positions, trade history).

### Feature 3 — Backtesting (`ui/pages/3_backtest.py` + `backtest/`)

1. Pick symbol + timeframe + lookback + capital + commission/slippage in sidebar.
2. Pick strategy + tune its params (auto-populated from `default_params`).
3. Click **Run Backtest** → `BacktestRunner.run()`:
   - Fetches OHLCV → `IndicatorEngine.compute_all()` → `BacktestEngine.run(strategy, enriched_df)`.
   - Bar-by-bar loop: `strategy.on_bar(window)` → signal → slippage-adjusted fills → equity curve.
4. Metrics + 4 Plotly charts: equity, drawdown, price with entry/exit markers, monthly returns.
5. Trade table + backtest history persisted to SQLite (`backtests` table).
6. **Generate AI Strategy** → `StrategyAgent.generate()` asks LLM to design a strategy (entry/exit/SL/TP rules + indicators + params) → convert to `AIStrategy` params → **Backtest AI Strategy**.

### Feature 4 — Derivatives (F&O) (`ui/pages/4_derivatives.py` + `backtest/derivatives.py`)

1. **Universe tab** — shows the Dhan F&O scrip master status, underlyings, expiries/lot sizes/strike steps and nearest resolvable futures/option contract (with real security IDs when the master is available).
2. **Option Chain tab** — picks an expiry and shows the ATM-anchored chain (CE/PE security IDs, lot sizes). Falls back to a synthetic chain when the master isn't available.
3. **Option Backtest** — strategies are entered N days before expiry, priced and mark-to-market **bar-by-bar with Black-Scholes** using realised volatility of the underlying spot (or a fixed `iv_pct`). Fills are sequential (no overlapping structures); exits on target/stop/expiry settlement. Underlyings resolve through `DerivativeUniverse.spot_symbol()` (e.g. `^NSEI`, `^NSEBANK`) so backtests run even without Dhan.
4. **Futures Backtest** — `futures_trend` (momentum on the futures/spot series) with whole-lot, margin-aware sizing (`margin_pct` of notional, `margin_usage_pct` cap).
5. **Payoff Simulator** — builds the selected structure's legs, prices them with BS, and plots P&L at expiry with breakevens, max profit/loss.
6. Built-in option structures: `long_straddle`, `long_strangle`, `bull_call_spread`, `bear_put_spread`, `iron_condor`. Futures: `futures_trend`.
7. **Autonomous F&O trading** — the auto-trader (`agents/auto_trade_agent.py`, Feature 2 page) has a derivatives loop controlled by `derivatives.auto_trade` (`enabled`, `strategy`, `mode: options|futures|both`). `scan_derivatives_once()` runs the regime strategy on each configured underlying and enters/rolls/flips structures (option legs) or futures via the paper/live engine; `HOLD` or a reversed regime squares off open F&O positions. No LLM needed — deterministic and runs alongside the cash-equity loop.
8. **Live F&O trading** — `place_order()` takes `exchange_segment` (e.g. `NSE_FNO`) + `instrument_type` (OPTIDX/OPTSTK/FUTIDX/FUTSTK); `enter_option_structure()`/`enter_futures_position()` in `TradingEngine` place multi-leg paper/live entries; risk sizing supports `option_lots_quantity` / `futures_margin_quantity`. Real contract history + live orders need Dhan credentials in `.env`.

---

## Derivatives (F&O) Reference

| Component | File | Purpose |
|-----------|------|---------|
| **F&O universe** | `data/derivatives.py` | Parses the public Dhan scrip-master CSV (`https://images.dhan.co/api-data/api-scrip-master.csv`), caches to `data/scrip_master_fno.csv`, resolves underlyings/expiries/lot sizes/strike steps/contracts. |
| **Option pricing** | `core/options.py` | Black-Scholes `bs_price`, `bs_greeks`, `implied_vol` (Newton), `annualized_volatility`, `intrinsic_value`, `settle_intrinsic`. |
| **Derivative backtests** | `backtest/derivatives.py` | `DerivativeBacktestEngine.run_options` (BS mark-to-market, sequential expiry cycles) and `run_futures` (margin-aware). |
| **Derivative strategies** | `strategies/derivatives.py` | Multi-leg structures + `futures_trend`; `list_derivative_strategies()`, `create_derivative_strategy()`, `payoff_table()`. |
| **F&O data fetch** | `data/fetcher.py` | `fetch_derivative_history()` via Dhan for real contract candles; spot underlyings via Yahoo (`^NSEI`, `^NSEBANK`, `.NS`). |

**Why Black-Scholes?** Option backtests work with nothing but the underlying spot series, so strategies can be validated without a broker — pricing quality is the classic European-model approximation, appropriate for benchmarking multi-leg structures.

---

## Technical Indicators

All in `core/indicators.py` — pure `pandas`/`numpy`:

| Category | Indicators |
|----------|------------|
| **Trend / MA** | EMA(9,21,50,200), SMA(20,50), VWAP |
| **Momentum** | MACD(12,26,9) + signal + hist, RSI(14) + RSI(7), Stochastic %K/%D(14,3), ROC(12), MOM(10), TRIX(15) |
| **Volatility** | Bollinger(20,2σ), ATR(14), NATR, Keltner(20,2×ATR), Donchian(20) |
| **Volume** | Volume SMA(20), OBV, MFI(14) |
| **Trend Strength** | ADX(14) + DMP/DMN, CCI(20), Williams %R(14), APO(12,26) |
| **SuperTrend** | SuperTrend(10,3×ATR) + direction |
| **Candlestick** | Engulfing, Hammer, Doji, Morning Star, Shooting Star (pattern detection) |
| **Derived** | pct_change, range, close_above_ema21/50/200, volume_expansion |

`IndicatorEngine.latest_summary(df)` extracts the last bar's values as a compact dict for the LLM.

`core/signals.py` — `SignalGenerator` provides rule-based scores: `momentum_score()`, `mean_reversion_score()`, `breakout_score()` (−10 to +10) with `get_signal()`.

---

## Strategies & Backtesting

| Strategy | File | Logic |
|----------|------|-------|
| **Momentum** | `strategies/momentum.py` | EMA9/21 golden/death cross + RSI filter + MACD momentum confirmation |
| **Mean Reversion** | `strategies/mean_reversion.py` | RSI oversold/overbought at Bollinger bands |
| **Breakout** | `strategies/breakout.py` | Donchian channel break + SuperTrend + volume expansion |
| **AI** | `strategies/ai_strategy.py` | LLM-generated rules (entry/exit/SL/TP) with momentum fallback |
| **Futures Trend** | `strategies/derivatives.py` | Momentum on futures/underlying, whole-lot margin backtest |
| **Option structures** | `strategies/derivatives.py` | `long_straddle` · `long_strangle` · `bull_call_spread` · `bear_put_spread` · `iron_condor` — Black-Scholes backtested |

Add a new strategy:

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    description = "My custom strategy"
    default_params = {"rsi_thresh": 40}

    def generate_signal(self, df):
        last = df.iloc[-1]
        if last["rsi"] < self.params["rsi_thresh"]:
            return {"action": "BUY", "confidence": 0.7, "reason": "RSI oversold"}
        return {"action": "HOLD", "confidence": 0, "reason": "No signal"}
```

Register in `strategies/registry.py` (the `@register` decorator + explicit fallback loop handles it).

**Backtest metrics** (`backtest/engine.py:compute_metrics`): `total_return_pct`, `buy_hold_return_pct`, `cagr_pct`, `sharpe`, `sortino`, `max_drawdown_pct`, `calmar`, `num_trades`, `win_rate_pct`, `profit_factor`, `avg_win`, `avg_loss`, `expectancy`, `volatility_pct`.

---

## Risk Management

`core/risk_manager.py` — `RiskManager`:

- `can_open_position(capital)` — checks `max_positions` + daily loss halt (`max_daily_loss_pct`).
- `position_quantity(capital, entry_price)` — `% risk per trade / stop distance`, capped by `capital / price`.
- `compute_stop_loss(entry_price)` / `compute_target(entry_price)` — % based.
- `trailing_stop(entry_price, current_price, side)` — max of initial SL and trailing level.

`trading/engine.py` enforces these on every `enter_position()` call.

---

## Data & Caching

- **Live/Historical:** `data/fetcher.py` → `DhanClient.historical_daily_data` / `intraday_minute_data` / `quote_data`.
- **Cache:** `data/cache.py` → `DataCache` (SQLite `data/market_cache.db`):
  - `candles(symbol, timeframe, timestamp, open, high, low, close, volume)`
  - `orders`, `trades`, `signals`, `backtests`
  - `save_candles()`, `get_candles()`, `latest_timestamp()`, `save_order/trade/signal()`, `get_open_positions()`, etc.
- Dhan historical: daily → inception, intraday (1/5/15/25/60m) → 5 years, 90-day chunks recommended.
- Timestamps are epoch integers → converted via `dhan.convert_to_date_time()`.

---

## API References

- **DhanHQ Python SDK:** https://github.com/dhan-oss/DhanHQ-py — `pip install dhanhq` — docs at https://dhanhq.co/docs/v2/
- **Dhan API Docs:** https://api.dhan.co/v2/ — REST + WebSocket (MarketFeed, FullDepth)
- **Groq API:** https://console.groq.com/docs/quickstart — OpenAI-compatible, use `openai` SDK with `base_url="https://api.groq.com/openai/v1"` and `model="llama-3.3-70b-versatile"`
- **Streamlit:** https://docs.streamlit.io/

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: dhanhq` | Ensure venv is activated and `pip install -r requirements.txt` ran |
| `numba` build fails | You are on Python 3.14 — this project avoids `pandas-ta`/`numba` entirely; ensure `requirements.txt` does NOT contain `pandas-ta` |
| `DHAN_CLIENT_ID must be set` | `copy .env.example .env` and fill real values; restart Streamlit |
| `GROQ_API_KEY not set` | Add to `.env`; AI features degrade to HOLD without it |
| `No data available` for a symbol | Check `security_id` in `config.yaml` — verify against Dhan scrip master; market may be closed |
| Streamlit `missing ScriptRunContext` warnings | Harmless when importing pages outside Streamlit run |
| `execution policy` blocks `Activate.ps1` | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Port 8501 in use | `streamlit run ui/app.py --server.port 8502` |
| Dhan `access-token` expired | Regenerate in Dhan app → API Access |

---

## Disclaimer

> **Educational tool only. Not investment advice.** Trading in equities, F&O, and commodities involves substantial risk of loss and is not suitable for every investor. Past backtest performance is not indicative of future results. Always paper-trade first. Authors and contributors are not registered investment advisors. Use at your own risk. Validate all AI recommendations independently before committing capital.

---

## License

MIT — see `LICENSE` if present. DhanHQ SDK is MIT licensed by Dhan.

---

## Quick Command Reference

```powershell
# Setup (once)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then edit .env

# Every session
.\venv\Scripts\Activate.ps1
python run.py
streamlit run ui/app.py
deactivate
```
