#!/usr/bin/env python3
"""
feature_engineering.py
=======================
從 SQLite 讀取微結構資料，resample 到 1 秒 bucket，計算所有特徵。
相容 v1（舊）和 v2（新）DB schema。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── DB helpers ────────────────────────────────────────────────────────────────

def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                           check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-65536")
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())


# ── Stream loaders (each returns a 1-s resampled DataFrame) ─────────────────

def _load_trades_1s(conn: sqlite3.Connection,
                    start_ms: int, end_ms: int) -> pd.DataFrame:
    """Trades → 1-second OHLCV + trade flow."""
    df = pd.read_sql_query(
        """
        SELECT trade_ts, price, qty,
               CASE WHEN is_buyer_maker=0 THEN qty ELSE 0 END AS buy_qty,
               CASE WHEN is_buyer_maker=1 THEN qty ELSE 0 END AS sell_qty
        FROM trades
        WHERE trade_ts >= ? AND trade_ts < ?
        ORDER BY trade_ts
        """,
        conn, params=(start_ms, end_ms),
    )
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["trade_ts"], unit="ms", utc=True)
    df = df.set_index("ts")

    dollar = df["price"] * df["qty"]
    r = pd.DataFrame({
        "open":     df["price"].resample("1s").first(),
        "high":     df["price"].resample("1s").max(),
        "low":      df["price"].resample("1s").min(),
        "close":    df["price"].resample("1s").last(),
        "volume":   df["qty"].resample("1s").sum(),
        "buy_vol":  df["buy_qty"].resample("1s").sum(),
        "sell_vol": df["sell_qty"].resample("1s").sum(),
        "count":    df["price"].resample("1s").count(),
        "dollar":   dollar.resample("1s").sum(),
    })
    r["vwap"] = r["dollar"] / r["volume"].clip(lower=1e-12)
    return r


def _load_l1_1s(conn: sqlite3.Connection,
                start_ms: int, end_ms: int) -> pd.DataFrame:
    """1-second aggregated L1 book ticker — direct read (already 1s)."""
    # Detect schema version: new schema has second_ts, old has event_ts
    if _has_column(conn, "orderbook_l1", "second_ts"):
        # New aggregated schema
        df = pd.read_sql_query(
            """
            SELECT second_ts,
                   mid_open, mid_high, mid_low, mid_close,
                   bid_price, ask_price, bid_qty_mean, ask_qty_mean,
                   spread_bps_mean, spread_bps_std,
                   obi_mean, obi_std, obi_open, obi_close,
                   tick_count
            FROM orderbook_l1
            WHERE second_ts >= ? AND second_ts < ?
            ORDER BY second_ts
            """,
            conn, params=(start_ms, end_ms),
        )
        if df.empty:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["second_ts"], unit="ms", utc=True)
        df = df.set_index("ts").drop(columns=["second_ts"])
        # Rename to standard names used downstream
        df = df.rename(columns={
            "mid_close":       "mid_price",
            "obi_mean":        "obi",
            "spread_bps_mean": "spread_bps",
        })
        return df

    # Legacy schema (event_ts, mid_price, obi, spread_bps)
    ts_col = "event_ts" if _has_column(conn, "orderbook_l1", "event_ts") else None
    if ts_col is None:
        return pd.DataFrame()

    df = pd.read_sql_query(
        f"""
        SELECT {ts_col} AS ts_ms,
               bid_price, ask_price, bid_qty, ask_qty,
               spread_bps, mid_price, obi
        FROM orderbook_l1
        WHERE {ts_col} >= ? AND {ts_col} < ?
        ORDER BY {ts_col}
        """,
        conn, params=(start_ms, end_ms),
    )
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("ts")

    return df[["bid_price", "ask_price", "bid_qty", "ask_qty",
               "spread_bps", "mid_price", "obi"]].resample("1s").agg({
        "bid_price":  "last",
        "ask_price":  "last",
        "bid_qty":    "last",
        "ask_qty":    "last",
        "spread_bps": "mean",
        "mid_price":  "last",
        "obi":        "mean",
    })


def _load_depth_1s(conn: sqlite3.Connection,
                   start_ms: int, end_ms: int) -> pd.DataFrame:
    """Depth metrics (L20 or L5) → 1-second last. Optional."""
    if not _table_exists(conn, "orderbook_metrics"):
        return pd.DataFrame()

    # prefer l20, fall back to l5
    for dtype in ("l20", "l5"):
        df = pd.read_sql_query(
            """
            SELECT event_ts,
                   depth_imbalance, bid_vwap, ask_vwap, weighted_mid,
                   total_bid_qty, total_ask_qty,
                   total_bid_value, total_ask_value
            FROM orderbook_metrics
            WHERE event_ts >= ? AND event_ts < ? AND depth_type = ?
            ORDER BY event_ts
            """,
            conn, params=(start_ms, end_ms, dtype),
        )
        if not df.empty:
            break

    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["event_ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    cols = ["depth_imbalance", "bid_vwap", "ask_vwap", "weighted_mid",
            "total_bid_qty", "total_ask_qty", "total_bid_value", "total_ask_value"]
    return df[cols].resample("1s").last()


def _load_mark_1s(conn: sqlite3.Connection,
                  start_ms: int, end_ms: int) -> pd.DataFrame:
    """Mark price / funding rate → 1-second last. Optional."""
    if not _table_exists(conn, "mark_price"):
        return pd.DataFrame()

    df = pd.read_sql_query(
        """
        SELECT event_ts, mark_price, index_price, last_funding_rate
        FROM mark_price
        WHERE event_ts >= ? AND event_ts < ?
        ORDER BY event_ts
        """,
        conn, params=(start_ms, end_ms),
    )
    if df.empty:
        return pd.DataFrame()

    df["ts"] = pd.to_datetime(df["event_ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    return df[["mark_price", "index_price", "last_funding_rate"]].resample("1s").last()


# ── Feature builder ───────────────────────────────────────────────────────────

def build_features(db_path: Path,
                   start_ms: int,
                   end_ms: int) -> pd.DataFrame:
    """
    Build 1-second feature matrix for [start_ms, end_ms).

    Parameters
    ----------
    db_path  : path to SQLite DB
    start_ms : start Unix milliseconds (inclusive)
    end_ms   : end Unix milliseconds (exclusive)

    Returns
    -------
    pd.DataFrame  index = UTC 1-second timestamps, columns = features
                  Empty DataFrame if insufficient data.
    """
    conn = _open_db(db_path)
    try:
        trades = _load_trades_1s(conn, start_ms, end_ms)
        l1     = _load_l1_1s(conn, start_ms, end_ms)
        depth  = _load_depth_1s(conn, start_ms, end_ms)
        mark   = _load_mark_1s(conn, start_ms, end_ms)
    finally:
        conn.close()

    if trades.empty or l1.empty:
        return pd.DataFrame()

    # ── Merge ────────────────────────────────────────────────────────────────
    df = trades.join(l1, how="inner")
    if not depth.empty:
        df = df.join(depth, how="left")
    if not mark.empty:
        df = df.join(mark, how="left")

    # forward-fill sparse columns
    for col in ("mark_price", "index_price", "last_funding_rate",
                "depth_imbalance", "bid_vwap", "ask_vwap", "weighted_mid"):
        if col in df.columns:
            df[col] = df[col].ffill()

    mid = df["mid_price"]

    # ── Price momentum (bps) ──────────────────────────────────────────────────
    df["ret_1s"]  = mid.pct_change(1)  * 10_000
    df["ret_5s"]  = mid.pct_change(5)  * 10_000
    df["ret_30s"] = mid.pct_change(30) * 10_000
    df["ret_60s"] = mid.pct_change(60) * 10_000

    # ── Realised volatility ───────────────────────────────────────────────────
    log_ret = np.log(mid / mid.shift(1))
    df["vol_5s"]   = log_ret.rolling(5,   min_periods=3).std() * 10_000
    df["vol_30s"]  = log_ret.rolling(30,  min_periods=15).std() * 10_000
    df["vol_300s"] = log_ret.rolling(300, min_periods=60).std() * 10_000

    # Price acceleration
    df["price_accel"] = df["ret_1s"].diff()

    # ── Trade flow ────────────────────────────────────────────────────────────
    vol = df["volume"].clip(lower=1e-12)
    bv5  = df["buy_vol"].rolling(5,  min_periods=1).sum()
    bv30 = df["buy_vol"].rolling(30, min_periods=1).sum()
    tv5  = df["volume"].rolling(5,  min_periods=1).sum().clip(1e-12)
    tv30 = df["volume"].rolling(30, min_periods=1).sum().clip(1e-12)

    df["buy_ratio_1s"]  = df["buy_vol"] / vol
    df["buy_ratio_5s"]  = bv5 / tv5
    df["buy_ratio_30s"] = bv30 / tv30
    df["trade_cnt_5s"]  = df["count"].rolling(5,  min_periods=1).sum()
    df["trade_cnt_30s"] = df["count"].rolling(30, min_periods=1).sum()
    df["vwap_dev_bps"]  = (df["vwap"] - mid) / mid * 10_000

    # ── OBI ───────────────────────────────────────────────────────────────────
    df["obi_ma_10s"]   = df["obi"].rolling(10, min_periods=3).mean()
    df["obi_ma_30s"]   = df["obi"].rolling(30, min_periods=5).mean()
    df["obi_ma_60s"]   = df["obi"].rolling(60, min_periods=10).mean()
    df["obi_std_30s"]  = df["obi"].rolling(30, min_periods=5).std()
    df["obi_momentum"] = df["obi"] - df["obi_ma_60s"]   # OBI vs slow MA

    # ── Spread ────────────────────────────────────────────────────────────────
    df["spread_ma_30s"] = df["spread_bps"].rolling(30, min_periods=5).mean()
    df["spread_dev"]    = df["spread_bps"] - df["spread_ma_30s"]

    # ── Depth (optional) ─────────────────────────────────────────────────────
    if "depth_imbalance" in df.columns:
        df["depth_imb_l20"] = df["depth_imbalance"]
        df["depth_imb_ma30"] = df["depth_imbalance"].rolling(30, min_periods=5).mean()
        if "weighted_mid" in df.columns:
            df["wmid_dev_bps"] = (df["weighted_mid"] - mid) / mid * 10_000
        if "bid_vwap" in df.columns:
            df["bid_vwap_dev"] = (df["bid_vwap"] - mid) / mid * 10_000
        if "ask_vwap" in df.columns:
            df["ask_vwap_dev"] = (df["ask_vwap"] - mid) / mid * 10_000
        if "total_bid_value" in df.columns and "total_ask_value" in df.columns:
            tv = (df["total_bid_value"] + df["total_ask_value"]).clip(1e-12)
            df["liq_imbalance"] = (df["total_bid_value"] - df["total_ask_value"]) / tv

    # ── Mark price (optional) ─────────────────────────────────────────────────
    if "mark_price" in df.columns:
        df["mark_dev_bps"] = (df["mark_price"] - mid) / mid * 10_000
    if "last_funding_rate" in df.columns:
        df["funding_rate"] = df["last_funding_rate"]

    # ── Tick rule ─────────────────────────────────────────────────────────────
    tick = np.sign(df["ret_1s"].fillna(0))
    df["tick_5s"]  = tick.rolling(5,  min_periods=1).sum()
    df["tick_30s"] = tick.rolling(30, min_periods=5).sum()

    # ── Drop raw / intermediate columns ───────────────────────────────────────
    _drop = [
        "open", "high", "low", "close", "dollar",
        "buy_vol", "sell_vol",
        "bid_price", "ask_price", "bid_qty", "ask_qty",
        "mark_price", "index_price", "last_funding_rate",
        "bid_vwap", "ask_vwap", "weighted_mid",
        "total_bid_qty", "total_ask_qty", "total_bid_value", "total_ask_value",
        "depth_imbalance",
    ]
    df = df.drop(columns=[c for c in _drop if c in df.columns])

    return df


# ── Convenience: get DB time range ───────────────────────────────────────────

def db_time_range(db_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Return (min_trade_ts_ms, max_trade_ts_ms) from the DB."""
    conn = _open_db(db_path)
    try:
        row = conn.execute(
            "SELECT MIN(trade_ts), MAX(trade_ts) FROM trades"
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)
    finally:
        conn.close()
