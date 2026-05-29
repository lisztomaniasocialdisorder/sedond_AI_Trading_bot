#!/usr/bin/env python3
"""Export 1-hour signal averages from SQLite DB to Parquet.

Output:
  data/parquet_signal/<COIN>/signal_averages_1h.parquet
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path(project_root: Path, coin: str) -> Path:
    return project_root / "harvesters" / f"{coin}_harvester" / "raw_db" / f"microstructure_{coin}.db"


def open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def q1(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> dict:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else {}


def build_symbol_averages(conn: sqlite3.Connection, coin: str, lookback_sec: int) -> dict:
    cutoff_ms = int((time.time() - lookback_sec) * 1000)

    l1_cols = table_columns(conn, "orderbook_l1")
    if "second_ts" in l1_cols and "obi_mean" in l1_cols:
        l1_avg = q1(
            conn,
            """
            SELECT AVG(obi_mean) AS obi_avg_1h, COUNT(*) AS l1_rows_1h
            FROM orderbook_l1
            WHERE second_ts >= ? AND obi_mean IS NOT NULL
            """,
            (cutoff_ms,),
        )
    else:
        l1_avg = q1(
            conn,
            """
            SELECT AVG(obi) AS obi_avg_1h, COUNT(*) AS l1_rows_1h
            FROM orderbook_l1
            WHERE event_ts >= ? AND obi IS NOT NULL
            """,
            (cutoff_ms,),
        )

    depth_avg = q1(
        conn,
        """
        SELECT AVG(depth_imbalance) AS depth_imbalance_avg_1h, COUNT(*) AS depth_rows_1h
        FROM orderbook_metrics
        WHERE depth_type='l20' AND event_ts >= ? AND depth_imbalance IS NOT NULL
        """,
        (cutoff_ms,),
    )

    flow = q1(
        conn,
        """
        SELECT COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN qty ELSE 0 END), 0) AS buy_volume_1h,
               COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN qty ELSE 0 END), 0) AS sell_volume_1h,
               COUNT(*) AS trade_rows_1h
        FROM trades
        WHERE trade_ts >= ?
        """,
        (cutoff_ms,),
    )

    buy_v = float(flow.get("buy_volume_1h") or 0.0)
    sell_v = float(flow.get("sell_volume_1h") or 0.0)
    total_v = buy_v + sell_v
    buy_pressure = (buy_v / total_v) if total_v > 0 else None

    return {
        "coin": coin,
        "ts": time.time(),
        "lookback_sec": int(lookback_sec),
        "obi_avg_1h": l1_avg.get("obi_avg_1h"),
        "depth_imbalance_avg_1h": depth_avg.get("depth_imbalance_avg_1h"),
        "buy_pressure_1h": buy_pressure,
        "buy_volume_1h": buy_v,
        "sell_volume_1h": sell_v,
        "l1_rows_1h": int(l1_avg.get("l1_rows_1h") or 0),
        "depth_rows_1h": int(depth_avg.get("depth_rows_1h") or 0),
        "trade_rows_1h": int(flow.get("trade_rows_1h") or 0),
    }


def export_symbol(project_root: Path, coin: str, lookback_sec: int, output_root: Path) -> int:
    db_path = default_db_path(project_root, coin)
    if not db_path.exists():
        print(f"[SKIP] DB not found: {db_path}")
        return 0

    conn = open_readonly(db_path)
    try:
        payload = build_symbol_averages(conn, coin, lookback_sec)
    finally:
        conn.close()

    out_dir = output_root / coin
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "signal_averages_1h.parquet"
    pd.DataFrame([payload]).to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
    print(f"[WRITE] {out_file}")
    print(
        f"        obi={payload['obi_avg_1h']} depth={payload['depth_imbalance_avg_1h']} "
        f"buy_pressure={payload['buy_pressure_1h']} rows(l1/depth/trades)="
        f"{payload['l1_rows_1h']}/{payload['depth_rows_1h']}/{payload['trade_rows_1h']}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export 1-hour signal averages from DB to Parquet.")
    parser.add_argument("--symbols", default="BTC,ADA", help="Comma-separated symbols. Default: BTC,ADA")
    parser.add_argument("--lookback-sec", type=int, default=3600, help="Lookback window in seconds. Default: 3600")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--output-root", default=None, help="Default: data/parquet_signal")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "data" / "parquet_signal"
    output_root.mkdir(parents=True, exist_ok=True)

    count = 0
    for raw in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        coin = raw.upper().replace("USDT", "")
        count += export_symbol(project_root, coin, int(args.lookback_sec), output_root)
    print(f"[DONE] exported {count} symbol parquet file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

