import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class DataCache:
    """SQLite-backed cache for historical market data."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            self._data_dir = os.path.join(base_dir, "data")
            os.makedirs(self._data_dir, exist_ok=True)
            db_path = os.path.join(self._data_dir, "market_cache.db")
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                PRIMARY KEY (symbol, timeframe, timestamp)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT, side TEXT, quantity INTEGER, price REAL,
                filled_price REAL, status TEXT, strategy TEXT,
                order_type TEXT, product_type TEXT,
                created_at TEXT, mode TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT, side TEXT, quantity INTEGER, entry_price REAL,
                exit_price REAL, entry_time TEXT, exit_time TEXT,
                pnl REAL, strategy TEXT, mode TEXT,
                entry_reason TEXT, exit_reason TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, action TEXT, confidence REAL,
                entry_price REAL, stop_loss REAL, target REAL,
                reasoning TEXT, created_at TEXT, source TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                signal_id INTEGER PRIMARY KEY,
                outcome TEXT,           -- WIN | LOSS | EXPIRED | UNRESOLVED
                exit_price REAL,
                realized_return_pct REAL,
                bars_held INTEGER,
                resolved_at TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
            """
        )
        # Older databases predate the timeframe_hours column on signals.
        try:
            cur.execute("ALTER TABLE signals ADD COLUMN timeframe_hours REAL DEFAULT 48")
        except Exception:
            pass  # column already exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS backtests (
                id TEXT PRIMARY KEY,
                strategy TEXT, symbol TEXT, params TEXT,
                start_date TEXT, end_date TEXT, timeframe TEXT,
                metrics TEXT, created_at TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Candle caching
    # ------------------------------------------------------------------
    def save_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        symbol = symbol.upper().replace(".NS", "")
        rows = []
        for _, r in df.iterrows():
            # Handle timestamp column (yfinance) vs index (Dhan)
            raw_ts = r.get("timestamp") if "timestamp" in r else r.name
            if pd.isna(raw_ts):
                continue
            try:
                # pd.Timestamp, datetime, int, float
                if isinstance(raw_ts, pd.Timestamp):
                    ts = int(raw_ts.timestamp())
                elif hasattr(raw_ts, "timestamp"):
                    ts = int(raw_ts.timestamp())
                else:
                    ts = int(raw_ts)
            except Exception:
                ts = int(pd.Timestamp(raw_ts).timestamp())
            rows.append(
                (
                    symbol,
                    timeframe,
                    ts,
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                    int(r.get("volume", 0) or 0),
                )
            )
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT OR REPLACE INTO candles
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_candles(
        self, symbol: str, timeframe: str, start_ts: int = 0, end_ts: int = 0
    ) -> pd.DataFrame:
        symbol = symbol.upper().replace(".NS", "")
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?"
        params = [symbol, timeframe]
        if start_ts:
            query += " AND timestamp>=?"
            params.append(start_ts)
        if end_ts:
            query += " AND timestamp<=?"
            params.append(end_ts)
        query += " ORDER BY timestamp"
        df = pd.read_sql_query(query, self._conn, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.set_index("timestamp")
        return df

    def latest_timestamp(self, symbol: str, timeframe: str) -> int:
        symbol = symbol.upper().replace(".NS", "")
        cur = self._conn.cursor()
        cur.execute(
            "SELECT MAX(timestamp) FROM candles WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else 0

    # ------------------------------------------------------------------
    # Order / trade / signal persistence
    # ------------------------------------------------------------------
    def save_order(self, order: dict):
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO orders
            (order_id, symbol, side, quantity, price, filled_price, status,
             strategy, order_type, product_type, created_at, mode)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                order.get("order_id", ""),
                order.get("symbol", ""),
                order.get("side", ""),
                order.get("quantity", 0),
                order.get("price", 0),
                order.get("filled_price", 0),
                order.get("status", ""),
                order.get("strategy", ""),
                order.get("order_type", ""),
                order.get("product_type", ""),
                order.get("created_at", datetime.now().isoformat()),
                order.get("mode", "paper"),
            ),
        )
        self._conn.commit()

    def save_trade(self, trade: dict):
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, side, quantity, entry_price, exit_price,
             entry_time, exit_time, pnl, strategy, mode, entry_reason, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.get("trade_id", ""),
                trade.get("symbol", ""),
                trade.get("side", ""),
                trade.get("quantity", 0),
                trade.get("entry_price", 0),
                trade.get("exit_price", 0),
                trade.get("entry_time", ""),
                trade.get("exit_time", ""),
                trade.get("pnl", 0),
                trade.get("strategy", ""),
                trade.get("mode", "paper"),
                trade.get("entry_reason", ""),
                trade.get("exit_reason", ""),
            ),
        )
        self._conn.commit()

    def save_signal(self, signal: dict) -> int:
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO signals
            (symbol, action, confidence, entry_price, stop_loss, target,
             reasoning, created_at, source, timeframe_hours)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                signal.get("symbol", ""),
                signal.get("action", ""),
                signal.get("confidence", 0),
                signal.get("entry_price", 0),
                signal.get("stop_loss", 0),
                signal.get("target", 0),
                signal.get("reasoning", ""),
                signal.get("created_at") or datetime.now().isoformat(),
                signal.get("source", "ai"),
                signal.get("timeframe_hours", 48),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_unresolved_signals(self, limit: int = 500) -> list:
        """Actionable signals that have no recorded outcome yet."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT s.* FROM signals s
            LEFT JOIN signal_outcomes o ON o.signal_id = s.id
            WHERE o.signal_id IS NULL AND s.action IN ('BUY','SELL')
              AND s.entry_price > 0 AND s.stop_loss > 0 AND s.target > 0
            ORDER BY s.id ASC LIMIT ?
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def save_signal_outcome(self, outcome: dict):
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO signal_outcomes
            (signal_id, outcome, exit_price, realized_return_pct, bars_held, resolved_at)
            VALUES (?,?,?,?,?,?)
            """,
            (
                outcome["signal_id"],
                outcome.get("outcome", "UNRESOLVED"),
                outcome.get("exit_price", 0),
                outcome.get("realized_return_pct", 0),
                outcome.get("bars_held", 0),
                outcome.get("resolved_at") or datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

    def get_resolved_signals(self, limit: int = 5000) -> list:
        """Signals joined with their outcomes, for calibration reporting."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT s.id, s.symbol, s.action, s.confidence, s.entry_price,
                   s.stop_loss, s.target, s.source, s.created_at,
                   o.outcome, o.exit_price, o.realized_return_pct, o.bars_held
            FROM signals s JOIN signal_outcomes o ON o.signal_id = s.id
            WHERE o.outcome IN ('WIN','LOSS','EXPIRED')
            ORDER BY s.id DESC LIMIT ?
            """,
            (limit,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_signals(self, limit: int = 100) -> list:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_trades(self, limit: int = 200) -> list:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_open_orders(self, mode: str = "paper") -> list:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM orders WHERE status IN ('OPEN','TRIGGER_PENDING','PENDING') AND mode=?",
            (mode,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_open_positions(self, mode: str = "paper") -> list:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM trades WHERE exit_time='' AND mode=?",
            (mode,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self._conn.close()

    def get_orders(self, limit: int = 200) -> list:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]