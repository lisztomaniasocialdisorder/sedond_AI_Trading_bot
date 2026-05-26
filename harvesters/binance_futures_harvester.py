#!/usr/bin/env python3
"""
binance_futures_harvester.py  ─  v2 (full rewrite)
====================================================
Binance USD-M Futures microstructure data collector.

Streams collected (per symbol)
───────────────────────────────
  <symbol>@trade          → trades          (個別成交)
  <symbol>@aggTrade       → agg_trades      (聚合成交)
  <symbol>@bookTicker     → orderbook_l1    (最佳一檔 + 衍生指標)
  <symbol>@depth5@100ms   → orderbook_l5    (五檔快照 + metrics)
  <symbol>@depth20@100ms  → orderbook_l20   (二十檔快照 + metrics)
  <symbol>@markPrice@1s   → mark_price      (標記價 / 指數價 / 資金費率)
  !forceOrder@arr         → liquidations    (強平訂單，按 symbol 過濾)

Storage
───────
  SQLite  (WAL mode)        : <COIN>_harvester/raw_db/microstructure_<COIN>.db
  Parquet (buffered flush)  : data/parquet/<COIN>/<table>/date=YYYY-MM-DD/

Stability
─────────
  ✓ 指數退避自動重連 (1 → 60 s)
  ✓ pong-timeout watchdog daemon thread
  ✓ depth stream gap detection (U / u / pu sequence)
  ✓ daily rotating log (保留 30 天)

Optional monitoring
───────────────────
  ✓ Prometheus metrics (pass prometheus_port=9100/9101)

Install
───────
  pip install websocket-client pyarrow prometheus-client
"""

from __future__ import annotations

import json
import logging
import signal
import sqlite3
import sys
import time
import threading
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional

import websocket

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

try:
    from prometheus_client import Counter, Gauge, start_http_server
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

# ── tunable constants ─────────────────────────────────────────────────────────
PARQUET_FLUSH_ROWS  = 2_000     # 累積到這個 row 數就 flush
PARQUET_FLUSH_SECS  = 60.0      # 或每 N 秒 flush（兩者以先到者為準）
HEARTBEAT_TIMEOUT_S = 120.0     # 超過此秒數沒有 pong → 強制重連（Binance 每 20s pong）
BACKOFF_INITIAL_S   = 2.0       # 重連初始等待
BACKOFF_MAX_S       = 60.0      # 重連最大等待
LOG_RETAIN_DAYS     = 30        # log 保留天數
COMMIT_INTERVAL     = 20        # 每 N 筆 message commit 一次 SQLite
CONSOLE_REFRESH_S   = 1.0       # CMD 狀態面板刷新頻率
L1_WRITE_INTERVAL   = 1.0       # orderbook_l1 每秒最多寫一筆（節流）


# ── ANSI helpers ──────────────────────────────────────────────────────────────
class _ConsoleDisplay:
    """
    在 CMD / Terminal 裡就地刷新狀態面板，使用 ANSI escape code。
    只移動游標、不清全螢幕 → 幾乎零閃頻。
    """

    # ANSI colors
    _R  = "\033[0m"       # reset
    _B  = "\033[1m"       # bold
    _DIM= "\033[2m"       # dim
    _CY = "\033[96m"      # bright cyan
    _GR = "\033[92m"      # bright green
    _RD = "\033[91m"      # bright red
    _YL = "\033[93m"      # bright yellow
    _WH = "\033[97m"      # bright white
    _MG = "\033[95m"      # magenta
    _BL = "\033[94m"      # blue
    _OR = "\033[38;5;214m" # orange (256-color)

    _PANEL_LINES = 34     # 面板固定行數（多一行 buffer）

    def __init__(self, symbol: str, coin: str):
        self.symbol    = symbol
        self.coin      = coin
        self._started  = False
        self._lock     = threading.Lock()
        # Enable ANSI on Windows
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass
        # stats dict (updated by handlers)
        self.st: dict  = {
            "last_price":      None,
            "bid":             None,
            "ask":             None,
            "spread":          None,
            "spread_bps":      None,
            "obi":             None,
            "mark_price":      None,
            "index_price":     None,
            "funding_rate":    None,
            "next_funding_ms": None,
            "msg_count":       0,
            "msg_rate":        0.0,
            "reconnects":      0,
            "gaps":            0,
            "db_size_mb":      0.0,
            "ws_alive":        False,
            "start_ts":        time.time(),
            # session write counters (reset on restart)
            "cnt_trades":      0,
            "cnt_agg":         0,
            "cnt_l1":          0,
            "cnt_l5":          0,
            "cnt_l20":         0,
            "cnt_mark":        0,
            "cnt_liq":         0,
            # DB total row counts (queried every 5s)
            "db_trades":       None,
            "db_agg":          None,
            "db_l1":           None,
            "db_l5":           None,
            "db_l20":          None,
            "db_mark":         None,
            "db_liq":          None,
            # rolling 1-min trade bucket: list of (ts, qty, is_buy)
            "_trade_buf":      [],
        }
        self._prev_msg_count = 0
        self._prev_ts        = time.time()

    # ── called by handlers ────────────────────────────────────────────────
    def update(self, **kwargs):
        with self._lock:
            self.st.update(kwargs)

    def inc_count(self, key: str, n: int = 1):
        """Increment a session write counter."""
        with self._lock:
            self.st[key] = self.st.get(key, 0) + n

    def record_trade(self, ts: float, qty: float, is_buy: bool):
        """Add trade to rolling 1-min buffer."""
        with self._lock:
            self.st["_trade_buf"].append((ts, qty, is_buy))
            # prune older than 60s
            cutoff = time.time() - 60
            self.st["_trade_buf"] = [t for t in self.st["_trade_buf"] if t[0] >= cutoff]

    # ── formatting helpers ────────────────────────────────────────────────
    def _fmt_price(self, v, dp=2):
        if v is None: return f"{self._DIM}—{self._R}"
        return f"{self._WH}{v:,.{dp}f}{self._R}"

    def _fmt_pct(self, v, scale=100, dp=4):
        if v is None: return f"{self._DIM}—{self._R}"
        val = v * scale
        col = self._GR if val >= 0 else self._RD
        sign = "+" if val >= 0 else ""
        return f"{col}{sign}{val:.{dp}f}%{self._R}"

    def _fmt_obi(self, v):
        if v is None: return f"{self._DIM}—{self._R}"
        val = v * 100
        col = self._GR if val >= 0 else self._RD
        sign = "+" if val >= 0 else ""
        bar_len = int(abs(val) / 2)  # 0-50 → 0-25 chars
        bar_len = max(0, min(bar_len, 20))
        bar = ("█" * bar_len).ljust(20)
        if val >= 0:
            bar_str = f"{self._GR}{bar}{self._R}"
        else:
            bar_str = f"{self._RD}{bar}{self._R}"
        return f"{col}{sign}{val:.2f}%{self._R}  {bar_str}"

    def _fmt_countdown(self, ms):
        if ms is None: return f"{self._DIM}—{self._R}"
        diff = int(ms) - int(time.time() * 1000)
        if diff <= 0: return f"{self._YL}NOW{self._R}"
        h = diff // 3_600_000
        m = (diff % 3_600_000) // 60_000
        s = (diff % 60_000) // 1_000
        return f"{self._WH}{h:02d}:{m:02d}:{s:02d}{self._R}"

    def _fmt_uptime(self, secs):
        h = int(secs) // 3600
        m = (int(secs) % 3600) // 60
        s = int(secs) % 60
        return f"{h:02d}h {m:02d}m {s:02d}s"

    def _divider(self, width=60):
        return f"{self._DIM}{'─' * width}{self._R}"

    # ── main render ───────────────────────────────────────────────────────
    def render(self):
        """Build and print the status panel (overwrites previous output)."""
        import os
        now = time.time()

        with self._lock:
            st = dict(self.st)
            buf = list(st["_trade_buf"])

        # compute 1-min stats from rolling buffer
        cutoff = now - 60
        buf1m  = [(ts, qty, ib) for ts, qty, ib in buf if ts >= cutoff]
        trades_1m   = len(buf1m)
        buy_vol_1m  = sum(q for _, q, ib in buf1m if ib)
        sell_vol_1m = sum(q for _, q, ib in buf1m if not ib)
        total_vol   = buy_vol_1m + sell_vol_1m
        buy_pct     = buy_vol_1m  / total_vol * 100 if total_vol > 0 else 50
        sell_pct    = sell_vol_1m / total_vol * 100 if total_vol > 0 else 50

        # message rate
        elapsed = now - self._prev_ts
        if elapsed >= 1.0:
            rate = (st["msg_count"] - self._prev_msg_count) / elapsed
            self._prev_msg_count = st["msg_count"]
            self._prev_ts        = now
        else:
            rate = 0.0

        uptime = now - st["start_ts"]
        ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

        ws_col = self._GR if st["ws_alive"] else self._RD
        ws_str = f"{ws_col}{'● LIVE' if st['ws_alive'] else '○ DISCONNECTED'}{self._R}"

        # buy/sell ratio bar (30 chars wide)
        bar_w  = 30
        buy_b  = int(buy_pct / 100 * bar_w)
        sell_b = bar_w - buy_b
        ratio_bar = f"{self._GR}{'█' * buy_b}{self._RD}{'█' * sell_b}{self._R}"

        dp = 5 if "ADA" in self.symbol else 2

        def _cnt(sess_key, db_key):
            """Format: session_count  (DB total)"""
            s = st.get(sess_key, 0)
            d = st.get(db_key)
            db_part = f"{self._DIM}/ {d:,}{self._R}" if d is not None else ""
            return f"{self._WH}{s:>8,}{self._R}  {db_part}"

        lines = [
            "",
            f"  {self._B}{self._OR}{'━'*56}{self._R}",
            f"  {self._B}{self._OR}  Binance Futures Harvester  "
            f"{self._CY}{self.symbol}{self._R}{self._B}{self._OR}{'':>20}{self._R}",
            f"  {self._B}{self._OR}{'━'*56}{self._R}",
            f"  {self._DIM}{ts_str:>54}{self._R}  {ws_str}",
            self._divider(58),

            # ── Price ────────────────────────────────────────────
            f"  {self._B}LAST PRICE   {self._R}"
            f"{self._fmt_price(st['last_price'], dp)}",

            f"  {self._DIM}BID {self._R}"
            f"{self._GR}{self._fmt_price(st['bid'], dp)}{self._R}   "
            f"{self._DIM}ASK {self._R}"
            f"{self._RD}{self._fmt_price(st['ask'], dp)}{self._R}",

            f"  {self._DIM}Spread {self._R}"
            f"{self._fmt_price(st['spread'], dp)}   "
            f"{self._DIM}bps {self._R}"
            f"{self._fmt_price(st['spread_bps'], 3)}",

            self._divider(58),

            # ── OBI ──────────────────────────────────────────────
            f"  {self._B}OBI (L1)     {self._R}"
            f"{self._fmt_obi(st['obi'])}",

            self._divider(58),

            # ── Mark Price ───────────────────────────────────────
            f"  {self._B}MARK PRICE   {self._R}"
            f"{self._fmt_price(st['mark_price'], dp)}",

            f"  {self._DIM}Index  {self._R}"
            f"{self._fmt_price(st['index_price'], dp)}",

            f"  {self._DIM}Funding      {self._R}"
            f"{self._fmt_pct(st['funding_rate'], scale=100, dp=4)}",

            f"  {self._DIM}Next Funding {self._R}"
            f"{self._fmt_countdown(st['next_funding_ms'])}",

            self._divider(58),

            # ── 1-Min Stats ───────────────────────────────────────
            f"  {self._B}1-MIN STATS{self._R}",
            f"  Trades {self._WH}{trades_1m:>6}{self._R}   "
            f"Volume {self._WH}{total_vol:>10.4f}{self._R} {self.coin}",

            f"  Buy  {self._GR}{buy_pct:5.1f}%{self._R}  "
            f"Sell {self._RD}{sell_pct:5.1f}%{self._R}",

            f"  {ratio_bar}",

            self._divider(58),

            # ── Writes ───────────────────────────────────────────
            f"  {self._B}WRITES  {self._DIM}session / DB total{self._R}",
            f"  {'trades':<12}{_cnt('cnt_trades','db_trades')}   "
            f"{'agg_trades':<12}{_cnt('cnt_agg','db_agg')}",
            f"  {'book_l1':<12}{_cnt('cnt_l1','db_l1')}   "
            f"{'book_l5':<12}{_cnt('cnt_l5','db_l5')}",
            f"  {'book_l20':<12}{_cnt('cnt_l20','db_l20')}   "
            f"{'mark_price':<12}{_cnt('cnt_mark','db_mark')}",
            f"  {'liquidations':<12}{_cnt('cnt_liq','db_liq')}",

            self._divider(58),

            # ── System ────────────────────────────────────────────
            f"  {self._DIM}Msg/s {self._R}{self._WH}{rate:6.1f}{self._R}   "
            f"{self._DIM}Total {self._R}{self._WH}{st['msg_count']:>9,}{self._R}   "
            f"{self._DIM}DB {self._R}{self._WH}{st['db_size_mb']:.1f}MB{self._R}",

            f"  {self._DIM}Uptime  {self._R}{self._WH}{self._fmt_uptime(uptime)}{self._R}   "
            f"{self._DIM}Reconnects {self._R}{self._WH}{st['reconnects']}{self._R}   "
            f"{self._DIM}Gaps {self._R}"
            f"{(self._YL if st['gaps'] > 0 else self._WH)}{st['gaps']}{self._R}",

            f"  {self._B}{self._OR}{'━'*56}{self._R}",
            "",
        ]

        # Move cursor up to overwrite previous panel
        if self._started:
            sys.stdout.write(f"\033[{self._PANEL_LINES}A")
        else:
            self._started = True

        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

_SQLITE_PRAGMAS = [
    ("journal_mode", "WAL"),
    ("synchronous",  "NORMAL"),
    ("cache_size",   "-65536"),      # 64 MB page cache
    ("temp_store",   "MEMORY"),
    ("mmap_size",    "268435456"),   # 256 MB mmap
]

# ── SQLite DDL ────────────────────────────────────────────────────────────────
_DDL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts        REAL    NOT NULL,
    event_ts        INTEGER,
    trade_ts        INTEGER,
    symbol          TEXT,
    trade_id        INTEGER,
    price           REAL,
    qty             REAL,
    quote_qty       REAL,
    is_buyer_maker  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_trades_trade_ts ON trades (trade_ts);
CREATE INDEX IF NOT EXISTS idx_trades_event_ts ON trades (event_ts);

CREATE TABLE IF NOT EXISTS agg_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts        REAL    NOT NULL,
    event_ts        INTEGER,
    trade_ts        INTEGER,
    symbol          TEXT,
    agg_trade_id    INTEGER,
    price           REAL,
    qty             REAL,
    quote_qty       REAL,
    first_trade_id  INTEGER,
    last_trade_id   INTEGER,
    is_buyer_maker  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_agg_trades_ts      ON agg_trades (trade_ts);
CREATE INDEX IF NOT EXISTS idx_agg_trades_agg_id  ON agg_trades (agg_trade_id);

CREATE TABLE IF NOT EXISTS orderbook_l1 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts        REAL    NOT NULL,
    second_ts       INTEGER NOT NULL,
    symbol          TEXT,
    mid_open        REAL,
    mid_high        REAL,
    mid_low         REAL,
    mid_close       REAL,
    bid_price       REAL,
    ask_price       REAL,
    bid_qty_mean    REAL,
    ask_qty_mean    REAL,
    spread_bps_mean REAL,
    spread_bps_std  REAL,
    obi_mean        REAL,
    obi_std         REAL,
    obi_open        REAL,
    obi_close       REAL,
    tick_count      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_l1_second_ts ON orderbook_l1 (second_ts);

CREATE TABLE IF NOT EXISTS orderbook_l5 (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts    REAL    NOT NULL,
    event_ts    INTEGER,
    update_id   INTEGER,
    symbol      TEXT,
    level       INTEGER,
    bid_price   REAL,
    bid_qty     REAL,
    ask_price   REAL,
    ask_qty     REAL
);
CREATE INDEX IF NOT EXISTS idx_l5_event_ts  ON orderbook_l5 (event_ts);
CREATE INDEX IF NOT EXISTS idx_l5_update_id ON orderbook_l5 (update_id);

CREATE TABLE IF NOT EXISTS orderbook_l20 (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts    REAL    NOT NULL,
    event_ts    INTEGER,
    update_id   INTEGER,
    symbol      TEXT,
    level       INTEGER,
    bid_price   REAL,
    bid_qty     REAL,
    ask_price   REAL,
    ask_qty     REAL
);
CREATE INDEX IF NOT EXISTS idx_l20_event_ts  ON orderbook_l20 (event_ts);
CREATE INDEX IF NOT EXISTS idx_l20_update_id ON orderbook_l20 (update_id);

CREATE TABLE IF NOT EXISTS orderbook_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts        REAL    NOT NULL,
    event_ts        INTEGER,
    update_id       INTEGER,
    symbol          TEXT,
    depth_type      TEXT,
    total_bid_qty   REAL,
    total_ask_qty   REAL,
    total_bid_value REAL,
    total_ask_value REAL,
    depth_imbalance REAL,
    bid_vwap        REAL,
    ask_vwap        REAL,
    weighted_mid    REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_event_ts ON orderbook_metrics (event_ts);
CREATE INDEX IF NOT EXISTS idx_metrics_type     ON orderbook_metrics (depth_type);

CREATE TABLE IF NOT EXISTS mark_price (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts            REAL    NOT NULL,
    event_ts            INTEGER,
    symbol              TEXT,
    mark_price          REAL,
    index_price         REAL,
    est_settle_price    REAL,
    last_funding_rate   REAL,
    next_funding_time   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_mp_event_ts ON mark_price (event_ts);

CREATE TABLE IF NOT EXISTS liquidations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts            REAL    NOT NULL,
    event_ts            INTEGER,
    symbol              TEXT,
    side                TEXT,
    order_type          TEXT,
    time_in_force       TEXT,
    orig_qty            REAL,
    price               REAL,
    avg_price           REAL,
    order_status        TEXT,
    last_filled_qty     REAL,
    filled_accum_qty    REAL,
    trade_time          INTEGER
);
CREATE INDEX IF NOT EXISTS idx_liq_event_ts ON liquidations (event_ts);
CREATE INDEX IF NOT EXISTS idx_liq_symbol   ON liquidations (symbol);

CREATE TABLE IF NOT EXISTS harvester_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    local_ts    REAL    NOT NULL,
    event_type  TEXT,
    detail      TEXT
);
"""


def _build_parquet_schemas() -> Dict[str, "pa.Schema"]:
    """Build Arrow schemas for Parquet output. Called only if pyarrow is available."""
    f = pa.field
    return {
        "trades": pa.schema([
            f("local_ts",       pa.float64()),
            f("event_ts",       pa.int64()),
            f("trade_ts",       pa.int64()),
            f("symbol",         pa.string()),
            f("trade_id",       pa.int64()),
            f("price",          pa.float64()),
            f("qty",            pa.float64()),
            f("quote_qty",      pa.float64()),
            f("is_buyer_maker", pa.bool_()),
        ]),
        "agg_trades": pa.schema([
            f("local_ts",       pa.float64()),
            f("event_ts",       pa.int64()),
            f("trade_ts",       pa.int64()),
            f("symbol",         pa.string()),
            f("agg_trade_id",   pa.int64()),
            f("price",          pa.float64()),
            f("qty",            pa.float64()),
            f("quote_qty",      pa.float64()),
            f("first_trade_id", pa.int64()),
            f("last_trade_id",  pa.int64()),
            f("is_buyer_maker", pa.bool_()),
        ]),
        "orderbook_l1": pa.schema([
            f("local_ts",   pa.float64()),
            f("event_ts",   pa.int64()),
            f("update_id",  pa.int64()),
            f("symbol",     pa.string()),
            f("bid_price",  pa.float64()),
            f("bid_qty",    pa.float64()),
            f("ask_price",  pa.float64()),
            f("ask_qty",    pa.float64()),
            f("spread",     pa.float64()),
            f("spread_bps", pa.float64()),
            f("mid_price",  pa.float64()),
            f("obi",        pa.float64()),
        ]),
        "orderbook_l5": pa.schema([
            f("local_ts",  pa.float64()),
            f("event_ts",  pa.int64()),
            f("update_id", pa.int64()),
            f("symbol",    pa.string()),
            f("level",     pa.int8()),
            f("bid_price", pa.float64()),
            f("bid_qty",   pa.float64()),
            f("ask_price", pa.float64()),
            f("ask_qty",   pa.float64()),
        ]),
        "orderbook_l20": pa.schema([
            f("local_ts",  pa.float64()),
            f("event_ts",  pa.int64()),
            f("update_id", pa.int64()),
            f("symbol",    pa.string()),
            f("level",     pa.int8()),
            f("bid_price", pa.float64()),
            f("bid_qty",   pa.float64()),
            f("ask_price", pa.float64()),
            f("ask_qty",   pa.float64()),
        ]),
        "orderbook_metrics": pa.schema([
            f("local_ts",        pa.float64()),
            f("event_ts",        pa.int64()),
            f("update_id",       pa.int64()),
            f("symbol",          pa.string()),
            f("depth_type",      pa.string()),
            f("total_bid_qty",   pa.float64()),
            f("total_ask_qty",   pa.float64()),
            f("total_bid_value", pa.float64()),
            f("total_ask_value", pa.float64()),
            f("depth_imbalance", pa.float64()),
            f("bid_vwap",        pa.float64()),
            f("ask_vwap",        pa.float64()),
            f("weighted_mid",    pa.float64()),
        ]),
        "mark_price": pa.schema([
            f("local_ts",          pa.float64()),
            f("event_ts",          pa.int64()),
            f("symbol",            pa.string()),
            f("mark_price",        pa.float64()),
            f("index_price",       pa.float64()),
            f("est_settle_price",  pa.float64()),
            f("last_funding_rate", pa.float64()),
            f("next_funding_time", pa.int64()),
        ]),
        "liquidations": pa.schema([
            f("local_ts",         pa.float64()),
            f("event_ts",         pa.int64()),
            f("symbol",           pa.string()),
            f("side",             pa.string()),
            f("order_type",       pa.string()),
            f("time_in_force",    pa.string()),
            f("orig_qty",         pa.float64()),
            f("price",            pa.float64()),
            f("avg_price",        pa.float64()),
            f("order_status",     pa.string()),
            f("last_filled_qty",  pa.float64()),
            f("filled_accum_qty", pa.float64()),
            f("trade_time",       pa.int64()),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
class BinanceFuturesHarvester:
    """
    Binance USD-M Futures microstructure data collector (v2).

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. "BTCUSDT".
    project_root : str | Path
        Root of the trading project (contains harvesters/, data/ …).
    prometheus_port : int | None
        If given, expose Prometheus metrics on this TCP port.
    """

    def __init__(
        self,
        symbol: str,
        project_root,
        prometheus_port: Optional[int] = None,
    ):
        self.symbol       = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.coin         = self.symbol.replace("USDT", "")
        self.project_root = Path(project_root)
        self._prom_port   = prometheus_port

        # ── directory layout ─────────────────────────────────────────────────
        harvester_dir     = self.project_root / "harvesters" / f"{self.coin}_harvester"
        self.db_path      = harvester_dir / "raw_db" / f"microstructure_{self.coin}.db"
        self.log_dir      = harvester_dir / "logs"
        self.parquet_root = self.project_root / "data" / "parquet" / self.coin

        for p in (self.db_path.parent, self.log_dir, self.parquet_root):
            p.mkdir(parents=True, exist_ok=True)

        # ── logging ──────────────────────────────────────────────────────────
        self.log = self._setup_logging()
        self.log.info("=" * 70)
        self.log.info(f"BinanceFuturesHarvester v2  symbol={self.symbol}")
        self.log.info(f"DB      : {self.db_path}")
        self.log.info(f"Parquet : {self.parquet_root}")
        self.log.info(f"Parquet available: {_HAS_PARQUET}")
        self.log.info(f"Prometheus available: {_HAS_PROMETHEUS}")
        self.log.info("=" * 70)

        # ── sqlite ───────────────────────────────────────────────────────────
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        for pragma, val in _SQLITE_PRAGMAS:
            self.conn.execute(f"PRAGMA {pragma} = {val}")
        self._init_db()

        # ── parquet ──────────────────────────────────────────────────────────
        self._pq_schemas: Dict = _build_parquet_schemas() if _HAS_PARQUET else {}
        self._pq_buf: Dict[str, List[dict]] = {t: [] for t in self._pq_schemas}
        self._pq_last_flush = time.time()

        # ── internal state ───────────────────────────────────────────────────
        self._ws: Optional[websocket.WebSocketApp] = None
        self._should_stop = False
        self._shutdown_started = False
        self._msg_count   = 0
        self._l1_cur_sec: Optional[int] = None
        self._l1_buf: List[tuple] = []

        # depth gap detection: track final update_id per stream
        self._l5_prev_u:  Optional[int] = None
        self._l20_prev_u: Optional[int] = None

        # heartbeat watchdog
        self._last_pong_ts = time.time()
        self._ws_alive     = False

        # ── prometheus ───────────────────────────────────────────────────────
        self._prom: Dict = {}
        if _HAS_PROMETHEUS and self._prom_port:
            self._setup_prometheus()

        # ── console display ───────────────────────────────────────────────────
        self._console = _ConsoleDisplay(self.symbol, self.coin)

        # ── websocket URL ─────────────────────────────────────────────────────
        streams = "/".join([
            f"{self.symbol_lower}@trade",
            f"{self.symbol_lower}@aggTrade",
            f"{self.symbol_lower}@bookTicker",
            f"{self.symbol_lower}@depth5@100ms",
            f"{self.symbol_lower}@depth20@100ms",
            f"{self.symbol_lower}@markPrice@1s",
            "!forceOrder@arr",
        ])
        self.ws_url = f"wss://fstream.binance.com/stream?streams={streams}"
        self.log.info(f"WS URL  : {self.ws_url}")

        # ── L1 throttle state（每秒最多寫一筆）──────────────────────────────
        self._l1_last_write: float = 0.0
        self._l1_pending:    Optional[tuple] = None

    # ══════════════════════════════════════════════════════════════════════════
    # SETUP
    # ══════════════════════════════════════════════════════════════════════════

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"harvester.{self.coin}")
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # console (INFO+)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # daily rotating file (DEBUG+)
        log_file = self.log_dir / f"{self.coin.lower()}.log"
        fh = TimedRotatingFileHandler(
            str(log_file),
            when="midnight",
            backupCount=LOG_RETAIN_DAYS,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        return logger

    def _setup_prometheus(self):
        lbl = ["symbol"]
        p   = self._prom
        p["trades_total"]         = Counter("hv_trades_total",         "Total trades",         lbl)
        p["agg_trades_total"]     = Counter("hv_agg_trades_total",     "Total agg trades",     lbl)
        p["l1_updates_total"]     = Counter("hv_l1_updates_total",     "L1 updates",           lbl)
        p["l5_updates_total"]     = Counter("hv_l5_updates_total",     "L5 depth updates",     lbl)
        p["l20_updates_total"]    = Counter("hv_l20_updates_total",    "L20 depth updates",    lbl)
        p["mp_updates_total"]     = Counter("hv_mp_updates_total",     "Mark price updates",   lbl)
        p["liquidations_total"]   = Counter("hv_liquidations_total",   "Liquidation events",   lbl)
        p["reconnects_total"]     = Counter("hv_reconnects_total",     "WS reconnects",        lbl)
        p["gaps_total"]           = Counter("hv_gaps_total",           "Depth gaps detected",  lbl)
        p["last_price"]           = Gauge("hv_last_price",             "Last trade price",     lbl)
        p["last_spread"]          = Gauge("hv_last_spread",            "Last L1 spread",       lbl)
        p["last_spread_bps"]      = Gauge("hv_last_spread_bps",        "Last L1 spread bps",   lbl)
        p["last_obi"]             = Gauge("hv_last_obi",               "Last L1 OBI",          lbl)
        p["last_mark_price"]      = Gauge("hv_last_mark_price",        "Last mark price",      lbl)
        p["last_funding_rate"]    = Gauge("hv_last_funding_rate",      "Last funding rate",    lbl)
        p["last_depth_imbalance"] = Gauge("hv_last_depth_imbalance",   "Last depth imbalance", lbl)
        start_http_server(self._prom_port)
        self.log.info(f"Prometheus listening on :{self._prom_port}")

    def _prom_inc(self, key: str):
        m = self._prom.get(key)
        if m is not None:
            m.labels(symbol=self.symbol).inc()

    def _prom_set(self, key: str, val):
        if val is None:
            return
        m = self._prom.get(key)
        if m is not None:
            m.labels(symbol=self.symbol).set(val)

    def _init_db(self):
        self._migrate_db()
        for stmt in _DDL_STATEMENTS.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)
        self.conn.commit()
        self.log.info("SQLite schema initialised")

    def _migrate_db(self):
        cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(orderbook_l1)").fetchall()
        }
        if not cols or "second_ts" in cols:
            return

        self.log.info("Migrating legacy orderbook_l1 schema for 1-second aggregates")
        migrations = {
            "second_ts": "INTEGER",
            "mid_open": "REAL",
            "mid_high": "REAL",
            "mid_low": "REAL",
            "mid_close": "REAL",
            "bid_qty_mean": "REAL",
            "ask_qty_mean": "REAL",
            "spread_bps_mean": "REAL",
            "spread_bps_std": "REAL",
            "obi_mean": "REAL",
            "obi_std": "REAL",
            "obi_open": "REAL",
            "obi_close": "REAL",
            "tick_count": "INTEGER",
        }
        for name, sql_type in migrations.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE orderbook_l1 ADD COLUMN {name} {sql_type}")
        self.conn.execute(
            """
            UPDATE orderbook_l1
            SET second_ts = COALESCE(second_ts, event_ts),
                mid_open = COALESCE(mid_open, mid_price),
                mid_high = COALESCE(mid_high, mid_price),
                mid_low = COALESCE(mid_low, mid_price),
                mid_close = COALESCE(mid_close, mid_price),
                bid_qty_mean = COALESCE(bid_qty_mean, bid_qty),
                ask_qty_mean = COALESCE(ask_qty_mean, ask_qty),
                spread_bps_mean = COALESCE(spread_bps_mean, spread_bps),
                obi_mean = COALESCE(obi_mean, obi),
                obi_open = COALESCE(obi_open, obi),
                obi_close = COALESCE(obi_close, obi),
                tick_count = COALESCE(tick_count, 1)
            WHERE second_ts IS NULL
            """
        )
        self.conn.commit()

    # ══════════════════════════════════════════════════════════════════════════
    # TYPE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _f(v) -> Optional[float]:
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _i(v) -> Optional[int]:
        try:
            return int(v)
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # DEPTH METRIC COMPUTATION
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_depth_metrics(
        self,
        bids: List,
        asks: List,
    ) -> dict:
        """
        Compute aggregate market microstructure metrics from a list of
        [price_str, qty_str] pairs.

        Returns
        -------
        dict with keys:
          total_bid_qty, total_ask_qty,
          total_bid_value, total_ask_value,
          depth_imbalance, bid_vwap, ask_vwap, weighted_mid
        """
        total_bid_qty   = 0.0
        total_ask_qty   = 0.0
        total_bid_value = 0.0
        total_ask_value = 0.0

        for bp, bq in bids:
            bp, bq          = float(bp), float(bq)
            total_bid_qty   += bq
            total_bid_value += bp * bq

        for ap, aq in asks:
            ap, aq          = float(ap), float(aq)
            total_ask_qty   += aq
            total_ask_value += ap * aq

        depth_imbalance = None
        total_qty = total_bid_qty + total_ask_qty
        if total_qty > 0:
            depth_imbalance = (total_bid_qty - total_ask_qty) / total_qty

        bid_vwap = total_bid_value / total_bid_qty if total_bid_qty > 0 else None
        ask_vwap = total_ask_value / total_ask_qty if total_ask_qty > 0 else None

        # Weighted mid: price weighted by opposite-side liquidity
        weighted_mid = None
        if bid_vwap is not None and ask_vwap is not None and total_qty > 0:
            weighted_mid = (
                bid_vwap * total_ask_qty + ask_vwap * total_bid_qty
            ) / total_qty

        return {
            "total_bid_qty":   total_bid_qty,
            "total_ask_qty":   total_ask_qty,
            "total_bid_value": total_bid_value,
            "total_ask_value": total_ask_value,
            "depth_imbalance": depth_imbalance,
            "bid_vwap":        bid_vwap,
            "ask_vwap":        ask_vwap,
            "weighted_mid":    weighted_mid,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_trade(self, local_ts: float, data: dict):
        price     = self._f(data.get("p"))
        qty       = self._f(data.get("q"))
        quote_qty = (price * qty) if (price is not None and qty is not None) else None

        row = (
            local_ts,
            self._i(data.get("E")),
            self._i(data.get("T")),
            data.get("s", self.symbol),
            self._i(data.get("t")),
            price, qty, quote_qty,
            1 if data.get("m") else 0,
        )
        self.conn.execute(
            "INSERT INTO trades "
            "(local_ts,event_ts,trade_ts,symbol,trade_id,price,qty,quote_qty,is_buyer_maker) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            row,
        )
        if _HAS_PARQUET:
            self._pq_buf["trades"].append({
                "local_ts": local_ts, "event_ts": row[1], "trade_ts": row[2],
                "symbol": row[3], "trade_id": row[4],
                "price": price, "qty": qty, "quote_qty": quote_qty,
                "is_buyer_maker": bool(data.get("m")),
            })
        self._prom_inc("trades_total")
        self._prom_set("last_price", price)
        # ── console update
        if price is not None:
            self._console.update(last_price=price)
        if qty is not None:
            self._console.record_trade(local_ts, qty, not bool(data.get("m")))
        self._console.inc_count("cnt_trades")

    def _handle_agg_trade(self, local_ts: float, data: dict):
        price     = self._f(data.get("p"))
        qty       = self._f(data.get("q"))
        quote_qty = (price * qty) if (price is not None and qty is not None) else None

        row = (
            local_ts,
            self._i(data.get("E")),
            self._i(data.get("T")),
            data.get("s", self.symbol),
            self._i(data.get("a")),   # agg trade id
            price, qty, quote_qty,
            self._i(data.get("f")),   # first trade id
            self._i(data.get("l")),   # last trade id
            1 if data.get("m") else 0,
        )
        self.conn.execute(
            "INSERT INTO agg_trades "
            "(local_ts,event_ts,trade_ts,symbol,agg_trade_id,price,qty,quote_qty,"
            "first_trade_id,last_trade_id,is_buyer_maker) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        if _HAS_PARQUET:
            self._pq_buf["agg_trades"].append({
                "local_ts": local_ts, "event_ts": row[1], "trade_ts": row[2],
                "symbol": row[3], "agg_trade_id": row[4],
                "price": price, "qty": qty, "quote_qty": quote_qty,
                "first_trade_id": row[8], "last_trade_id": row[9],
                "is_buyer_maker": bool(data.get("m")),
            })
        self._prom_inc("agg_trades_total")
        self._console.inc_count("cnt_agg")

    # ── L1 aggregation helpers ─────────────────────────────────────────

    def _flush_l1_buf(self, local_ts: float) -> None:
        """Aggregate the current second's L1 buffer and write one row to DB."""
        if not self._l1_buf:
            return

        mids  = [t[0] for t in self._l1_buf]
        obis  = [t[1] for t in self._l1_buf if t[1] is not None]
        bps   = [t[2] for t in self._l1_buf if t[2] is not None]
        last  = self._l1_buf[-1]   # (mid, obi, spread_bps, bid_p, ask_p, bq, aq)

        import math
        def _mean(xs): return sum(xs) / len(xs) if xs else None
        def _std(xs):
            if len(xs) < 2: return 0.0
            m = sum(xs) / len(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        row = (
            local_ts,
            self._l1_cur_sec,
            self.symbol,
            mids[0],           # mid_open
            max(mids),         # mid_high
            min(mids),         # mid_low
            mids[-1],          # mid_close
            last[3],           # bid_price (close)
            last[4],           # ask_price (close)
            _mean([t[5] for t in self._l1_buf if t[5] is not None]),  # bid_qty_mean
            _mean([t[6] for t in self._l1_buf if t[6] is not None]),  # ask_qty_mean
            _mean(bps),        # spread_bps_mean
            _std(bps),         # spread_bps_std
            _mean(obis),       # obi_mean
            _std(obis),        # obi_std
            obis[0] if obis else None,   # obi_open
            obis[-1] if obis else None,  # obi_close
            len(self._l1_buf), # tick_count
        )

        self.conn.execute(
            "INSERT INTO orderbook_l1 "
            "(local_ts,second_ts,symbol,"
            "mid_open,mid_high,mid_low,mid_close,"
            "bid_price,ask_price,bid_qty_mean,ask_qty_mean,"
            "spread_bps_mean,spread_bps_std,"
            "obi_mean,obi_std,obi_open,obi_close,tick_count) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        self._console.inc_count("cnt_l1")
        self._l1_buf.clear()

    def _handle_book_ticker(self, local_ts: float, data: dict):
        bid_price = self._f(data.get("b"))
        bid_qty   = self._f(data.get("B"))
        ask_price = self._f(data.get("a"))
        ask_qty   = self._f(data.get("A"))
        event_ts  = self._i(data.get("E"))

        spread_bps = mid_price = obi = None
        if bid_price is not None and ask_price is not None:
            mid_price = (bid_price + ask_price) / 2.0
            if mid_price > 0:
                spread_bps = (ask_price - bid_price) / mid_price * 10_000.0
        if bid_qty is not None and ask_qty is not None:
            total = bid_qty + ask_qty
            if total > 0:
                obi = (bid_qty - ask_qty) / total

        # ── accumulate into buffer ────────────────────────────────────
        if mid_price is None:
            return

        # current second bucket (floor to 1000 ms)
        cur_sec = int(event_ts / 1000) * 1000 if event_ts else int(local_ts) * 1000

        if cur_sec != self._l1_cur_sec:
            # second boundary crossed → flush previous bucket
            self._flush_l1_buf(local_ts)
            self._l1_cur_sec = cur_sec

        self._l1_buf.append((mid_price, obi, spread_bps,
                              bid_price, ask_price, bid_qty, ask_qty))

        # ── Prometheus + console (every tick) ─────────────────────────
        self._prom_inc("l1_updates_total")
        self._prom_set("last_spread_bps", spread_bps)
        self._prom_set("last_obi",        obi)
        self._console.update(
            bid=bid_price, ask=ask_price,
            spread=ask_price - bid_price if bid_price and ask_price else None,
            spread_bps=spread_bps, obi=obi,
        )

    def _handle_depth(
        self,
        local_ts: float,
        data: dict,
        table: str,        # "orderbook_l5" or "orderbook_l20"
        depth_type: str,   # "l5" or "l20"
        max_levels: int,   # 5 or 20
    ):
        bids      = data.get("b", [])
        asks      = data.get("a", [])
        event_ts  = self._i(data.get("E"))
        update_id = self._i(data.get("u"))   # final update ID in this event
        prev_u    = self._i(data.get("pu"))  # final update ID of previous event
        symbol    = data.get("s", self.symbol)

        # ── gap detection ────────────────────────────────────────────────────
        stored_prev = getattr(self, f"_{depth_type}_prev_u")
        if stored_prev is not None and prev_u is not None and stored_prev != prev_u:
            msg = (
                f"[GAP] {depth_type} pu={prev_u} expected={stored_prev} "
                f"(update_id={update_id})"
            )
            self.log.warning(msg)
            self.conn.execute(
                "INSERT INTO harvester_events (local_ts,event_type,detail) VALUES (?,?,?)",
                (local_ts, "depth_gap", msg),
            )
            self._prom_inc("gaps_total")
            self._console.update(gaps=self._console.st.get("gaps", 0) + 1)
        setattr(self, f"_{depth_type}_prev_u", update_id)

        # ── level rows ───────────────────────────────────────────────────────
        n    = min(max_levels, len(bids), len(asks))
        rows = []
        for i in range(n):
            rows.append((
                local_ts, event_ts, update_id, symbol, i + 1,
                self._f(bids[i][0]), self._f(bids[i][1]),
                self._f(asks[i][0]), self._f(asks[i][1]),
            ))

        if rows:
            self.conn.executemany(
                f"INSERT INTO {table} "
                f"(local_ts,event_ts,update_id,symbol,level,"
                f"bid_price,bid_qty,ask_price,ask_qty) "
                f"VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
            if _HAS_PARQUET:
                keys = ("local_ts","event_ts","update_id","symbol","level",
                        "bid_price","bid_qty","ask_price","ask_qty")
                self._pq_buf[table].extend(dict(zip(keys, r)) for r in rows)

        # ── depth metrics ─────────────────────────────────────────────────────
        if bids and asks:
            m = self._compute_depth_metrics(bids[:n], asks[:n])
            metric_row = (
                local_ts, event_ts, update_id, symbol, depth_type,
                m["total_bid_qty"],   m["total_ask_qty"],
                m["total_bid_value"], m["total_ask_value"],
                m["depth_imbalance"], m["bid_vwap"],
                m["ask_vwap"],        m["weighted_mid"],
            )
            self.conn.execute(
                "INSERT INTO orderbook_metrics "
                "(local_ts,event_ts,update_id,symbol,depth_type,"
                "total_bid_qty,total_ask_qty,total_bid_value,total_ask_value,"
                "depth_imbalance,bid_vwap,ask_vwap,weighted_mid) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                metric_row,
            )
            if _HAS_PARQUET:
                keys = ("local_ts","event_ts","update_id","symbol","depth_type",
                        "total_bid_qty","total_ask_qty","total_bid_value","total_ask_value",
                        "depth_imbalance","bid_vwap","ask_vwap","weighted_mid")
                self._pq_buf["orderbook_metrics"].append(dict(zip(keys, metric_row)))

            self._prom_set("last_depth_imbalance", m["depth_imbalance"])

        prom_key = "l5_updates_total" if depth_type == "l5" else "l20_updates_total"
        self._prom_inc(prom_key)
        cnt_key = "cnt_l5" if depth_type == "l5" else "cnt_l20"
        self._console.inc_count(cnt_key)

    def _handle_mark_price(self, local_ts: float, data: dict):
        row = (
            local_ts,
            self._i(data.get("E")),
            data.get("s", self.symbol),
            self._f(data.get("p")),   # mark price
            self._f(data.get("i")),   # index price
            self._f(data.get("P")),   # estimated settle price
            self._f(data.get("r")),   # last funding rate
            self._i(data.get("T")),   # next funding time
        )
        self.conn.execute(
            "INSERT INTO mark_price "
            "(local_ts,event_ts,symbol,mark_price,index_price,"
            "est_settle_price,last_funding_rate,next_funding_time) "
            "VALUES (?,?,?,?,?,?,?,?)",
            row,
        )
        if _HAS_PARQUET:
            keys = ("local_ts","event_ts","symbol","mark_price","index_price",
                    "est_settle_price","last_funding_rate","next_funding_time")
            self._pq_buf["mark_price"].append(dict(zip(keys, row)))

        self._prom_inc("mp_updates_total")
        self._prom_set("last_mark_price",   row[3])
        self._prom_set("last_funding_rate", row[6])
        # ── console update
        self._console.update(
            mark_price=row[3], index_price=row[4],
            funding_rate=row[6], next_funding_ms=row[7],
        )
        self._console.inc_count("cnt_mark")

    def _handle_liquidation(self, local_ts: float, data: dict):
        o      = data.get("o", {})
        symbol = o.get("s", "")
        # Filter: only store our symbol's liquidations
        if symbol.upper() != self.symbol:
            return

        row = (
            local_ts,
            self._i(data.get("E")),
            symbol,
            o.get("S"),              # side: BUY or SELL
            o.get("o"),              # order type
            o.get("f"),              # time in force
            self._f(o.get("q")),    # orig qty
            self._f(o.get("p")),    # price
            self._f(o.get("ap")),   # avg price
            o.get("X"),              # order status
            self._f(o.get("l")),    # last filled qty
            self._f(o.get("z")),    # filled accum qty
            self._i(o.get("T")),    # trade time
        )
        self.conn.execute(
            "INSERT INTO liquidations "
            "(local_ts,event_ts,symbol,side,order_type,time_in_force,"
            "orig_qty,price,avg_price,order_status,"
            "last_filled_qty,filled_accum_qty,trade_time) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        if _HAS_PARQUET:
            keys = ("local_ts","event_ts","symbol","side","order_type","time_in_force",
                    "orig_qty","price","avg_price","order_status",
                    "last_filled_qty","filled_accum_qty","trade_time")
            self._pq_buf["liquidations"].append(dict(zip(keys, row)))

        self._prom_inc("liquidations_total")
        self._console.inc_count("cnt_liq")
        self.log.info(
            f"[LIQUIDATION] {symbol} side={o.get('S')} "
            f"qty={o.get('q')} avg_price={o.get('ap')} status={o.get('X')}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # WEBSOCKET CALLBACKS
    # ══════════════════════════════════════════════════════════════════════════

    def on_message(self, ws, message: str):
        if self._should_stop:
            return
        local_ts = time.time()
        try:
            payload    = json.loads(message)
            data       = payload.get("data", {})
            event_type = data.get("e", "")
            stream     = payload.get("stream", "")

            if   event_type == "trade":
                self._handle_trade(local_ts, data)
            elif event_type == "aggTrade":
                self._handle_agg_trade(local_ts, data)
            elif event_type == "bookTicker":
                self._handle_book_ticker(local_ts, data)
            elif event_type == "depthUpdate":
                if "@depth5" in stream:
                    self._handle_depth(local_ts, data, "orderbook_l5",  "l5",  5)
                else:
                    self._handle_depth(local_ts, data, "orderbook_l20", "l20", 20)
            elif event_type == "markPriceUpdate":
                self._handle_mark_price(local_ts, data)
            elif event_type == "forceOrder":
                self._handle_liquidation(local_ts, data)

            # Batch commit: commit every N messages for efficiency
            self._msg_count += 1
            self._console.update(msg_count=self._msg_count)
            if self._msg_count >= COMMIT_INTERVAL:
                self.conn.commit()
                self._msg_count = 0

            # Parquet flush check
            self._maybe_flush_parquet(local_ts)

        except Exception as exc:
            self.log.error(f"on_message error: {exc}", exc_info=True)
            self.log.debug(f"raw: {message[:500]}")

    def on_open(self, ws):
        self._ws_alive     = True
        self._last_pong_ts = time.time()
        self.log.info("[WS OPENED] streaming data…")
        self._console.update(ws_alive=True)
        self.conn.execute(
            "INSERT INTO harvester_events (local_ts,event_type,detail) VALUES (?,?,?)",
            (time.time(), "ws_open", "WebSocket opened"),
        )
        self.conn.commit()

    def on_error(self, ws, error):
        self.log.error(f"[WS ERROR] {error}")

    def on_close(self, ws, code, msg):
        self._ws_alive = False
        self.log.warning(f"[WS CLOSED] code={code}  msg={msg}")
        self._console.update(ws_alive=False)
        try:
            self.conn.execute(
                "INSERT INTO harvester_events (local_ts,event_type,detail) VALUES (?,?,?)",
                (time.time(), "ws_close", f"code={code} msg={msg}"),
            )
            self.conn.commit()
        except Exception:
            pass

    def on_ping(self, ws, data):
        self.log.debug(f"[PING]  data={data!r}")

    def on_pong(self, ws, data):
        self._last_pong_ts = time.time()
        self.log.debug(f"[PONG]  data={data!r}")

    # ══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT WATCHDOG (daemon thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _console_loop(self):
        """Daemon thread: refresh console display every second."""
        _db_check_interval = 5.0
        _last_db_check     = 0.0
        _tables = [
            ("trades",           "db_trades"),
            ("agg_trades",       "db_agg"),
            ("orderbook_l1",     "db_l1"),
            ("orderbook_l5",     "db_l5"),
            ("orderbook_l20",    "db_l20"),
            ("mark_price",       "db_mark"),
            ("liquidations",     "db_liq"),
        ]
        while not self._should_stop:
            try:
                now = time.time()
                # update DB file size
                if self.db_path.exists():
                    self._console.update(db_size_mb=round(self.db_path.stat().st_size / 1_048_576, 1))
                # query DB row counts every 5 s
                if now - _last_db_check >= _db_check_interval:
                    _last_db_check = now
                    try:
                        cur = self.conn.cursor()
                        counts = {}
                        for tbl, key in _tables:
                            try:
                                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                                counts[key] = cur.fetchone()[0]
                            except Exception:
                                counts[key] = None
                        self._console.update(**counts)
                    except Exception:
                        pass
                self._console.render()
            except Exception:
                pass
            time.sleep(CONSOLE_REFRESH_S)

    def _watchdog_loop(self):
        self.log.info("[WATCHDOG] started")
        while not self._should_stop:
            time.sleep(5)
            if self._ws_alive and self._ws is not None:
                age = time.time() - self._last_pong_ts
                if age > HEARTBEAT_TIMEOUT_S:
                    self.log.warning(
                        f"[WATCHDOG] no pong for {age:.1f}s — forcing reconnect"
                    )
                    try:
                        self._ws.close()
                    except Exception:
                        pass
        self.log.info("[WATCHDOG] stopped")

    # ══════════════════════════════════════════════════════════════════════════
    # PARQUET FLUSH
    # ══════════════════════════════════════════════════════════════════════════

    def _maybe_flush_parquet(self, now: float):
        if not _HAS_PARQUET:
            return
        total_rows = sum(len(v) for v in self._pq_buf.values())
        elapsed    = now - self._pq_last_flush
        if total_rows >= PARQUET_FLUSH_ROWS or elapsed >= PARQUET_FLUSH_SECS:
            self._flush_parquet()

    def _flush_parquet(self):
        if not _HAS_PARQUET:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts_part  = int(time.time())
        for table, rows in self._pq_buf.items():
            if not rows:
                continue
            try:
                schema  = self._pq_schemas[table]
                out_dir = self.parquet_root / table / f"date={date_str}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"part-{ts_part}.parquet"
                # Build column-oriented dict in schema field order
                col_names = [field.name for field in schema]
                columns   = {col: [r.get(col) for r in rows] for col in col_names}
                arrow_tbl = pa.table(columns, schema=schema)
                pq.write_table(arrow_tbl, str(out_file), compression="snappy")
                self.log.debug(f"[PARQUET] {table}  {len(rows):,} rows → {out_file.name}")
            except Exception as exc:
                self.log.error(f"[PARQUET] flush error ({table}): {exc}", exc_info=True)
            finally:
                self._pq_buf[table].clear()
        self._pq_last_flush = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def request_stop(self, reason: str = "stop requested"):
        """Stop receiving new websocket messages and let shutdown drain buffers."""
        if self._should_stop:
            return
        self._should_stop = True
        self.log.info(f"[STOP] {reason} received; pausing data capture and draining buffers")
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception as exc:
            self.log.debug(f"[STOP] websocket close ignored: {exc}")

    def run_forever(self):
        """Start the harvester. Blocks until KeyboardInterrupt."""
        old_sigint = signal.getsignal(signal.SIGINT)
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_stop(signum, frame):
            self.request_stop(signal.Signals(signum).name)

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        # Start watchdog daemon thread
        wd = threading.Thread(
            target=self._watchdog_loop, daemon=True, name=f"watchdog-{self.coin}"
        )
        wd.start()

        # Start console display daemon thread
        cd = threading.Thread(
            target=self._console_loop, daemon=True, name=f"console-{self.coin}"
        )
        cd.start()

        backoff = BACKOFF_INITIAL_S

        while not self._should_stop:
            self.log.info(f"[CONNECT] attempting connection…")
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_ping=self.on_ping,
                    on_pong=self.on_pong,
                )
                self._ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                )
                backoff = BACKOFF_INITIAL_S  # reset on clean disconnect

            except KeyboardInterrupt:
                self.log.info("[STOP] KeyboardInterrupt — shutting down")
                self.request_stop("KeyboardInterrupt")
                break
            except Exception as exc:
                self.log.error(f"[CONNECT ERROR] {exc}", exc_info=True)

            if self._should_stop:
                break

            self.log.info(f"[RECONNECT] waiting {backoff:.1f}s…")
            self._prom_inc("reconnects_total")
            self._console.update(reconnects=self._console.st.get("reconnects", 0) + 1)
            self.conn.execute(
                "INSERT INTO harvester_events (local_ts,event_type,detail) VALUES (?,?,?)",
                (time.time(), "reconnect", f"backoff={backoff:.1f}s"),
            )
            self.conn.commit()
            time.sleep(backoff)
            backoff = min(backoff * 2.0, BACKOFF_MAX_S)

        # ── shutdown cleanup ──────────────────────────────────────────────────
        self.log.info("[SHUTDOWN] flushing remaining Parquet buffers…")
        self._flush_parquet()
        self.log.info("[SHUTDOWN] closing SQLite connection…")
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
        self.log.info("[SHUTDOWN] done.")
