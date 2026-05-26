#!/usr/bin/env python3
"""Export harvester SQLite DB tables to partitioned Parquet.

Reads DB in read-only mode, so it can run while harvesters are active.

Output layout:
    data/parquet/<COIN>/<table>/date=YYYY-MM-DD/part-db-*.parquet

Examples:
    python harvesters/db_to_parquet.py --symbol BTC
    python harvesters/db_to_parquet.py --symbol ADA --table trades
    python harvesters/db_to_parquet.py --symbol BTC --table orderbook_l1 --start-id 1000000
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install it before running db_to_parquet.py") from exc

try:
    from csv_to_parquet import TABLE_SCHEMAS, coerce_schema, normalize_symbol, timestamp_to_date
except Exception as exc:  # pragma: no cover
    raise SystemExit("db_to_parquet.py must live next to csv_to_parquet.py") from exc


DEFAULT_TABLES = [
    "trades",
    "agg_trades",
    "orderbook_l1",
    "orderbook_l5",
    "orderbook_l20",
    "orderbook_metrics",
    "mark_price",
    "liquidations",
]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path(project_root: Path, coin: str) -> Path:
    return project_root / "harvesters" / f"{coin}_harvester" / "raw_db" / f"microstructure_{coin}.db"


def open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def row_count(conn: sqlite3.Connection, table: str, where_sql: str, params: list[object]) -> int:
    sql = f"SELECT COUNT(*) FROM {table} {where_sql}"
    return int(conn.execute(sql, params).fetchone()[0])


def build_where(args: argparse.Namespace) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if args.start_id is not None:
        clauses.append("id >= ?")
        params.append(int(args.start_id))
    if args.end_id is not None:
        clauses.append("id <= ?")
        params.append(int(args.end_id))
    if args.start_ts is not None:
        clauses.append(f"{args.date_col} >= ?")
        params.append(int(args.start_ts))
    if args.end_ts is not None:
        clauses.append(f"{args.date_col} <= ?")
        params.append(int(args.end_ts))
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


def write_partitioned(
    df: pd.DataFrame,
    *,
    out_root: Path,
    table: str,
    date_col: str,
    timestamp_unit: str,
    chunk_index: int,
    compression: str,
    prefix: str,
) -> int:
    if df.empty:
        return 0
    if date_col not in df.columns:
        raise ValueError(f"timestamp/date column not found: {date_col}")

    dates = timestamp_to_date(df[date_col], timestamp_unit)
    ts_part = int(time.time())
    written = 0
    for part_index, (date_str, part) in enumerate(df.groupby(dates, dropna=False)):
        if not isinstance(date_str, str) or date_str == "NaT":
            date_str = "unknown"
        out_dir = out_root / table / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{prefix}-{ts_part}-{chunk_index:05d}-{part_index:03d}.parquet"
        part.to_parquet(out_file, index=False, compression=compression, engine="pyarrow")
        written += len(part)
        print(f"[WRITE] {len(part):,} rows -> {out_file}")
    return written


def export_table(
    conn: sqlite3.Connection,
    *,
    table: str,
    symbol: str,
    out_root: Path,
    date_col: str,
    timestamp_unit: str,
    chunk_size: int,
    compression: str,
    strict: bool,
    include_id: bool,
    where_sql: str,
    params: list[object],
    dry_run: bool,
    overwrite: bool,
) -> int:
    if not table_exists(conn, table):
        print(f"[SKIP] table not found: {table}")
        return 0

    columns = table_columns(conn, table)
    select_cols = list(columns)
    if not include_id and "id" in select_cols:
        select_cols.remove("id")
    if date_col not in columns:
        print(f"[SKIP] {table}: date column not found: {date_col}")
        return 0

    total_rows = row_count(conn, table, where_sql, params)
    print(f"[TABLE] {table}: {total_rows:,} rows")
    if total_rows == 0 or dry_run:
        return 0

    if overwrite:
        target = out_root / table
        if target.exists():
            print(f"[OVERWRITE] removing existing parquet table: {target}")
            import shutil

            shutil.rmtree(target)

    col_sql = ", ".join(select_cols)
    sql = f"SELECT {col_sql} FROM {table} {where_sql} ORDER BY id"
    written = 0
    for chunk_index, chunk in enumerate(pd.read_sql_query(sql, conn, params=params, chunksize=chunk_size)):
        chunk = coerce_schema(chunk, table, symbol, strict)
        written += write_partitioned(
            chunk,
            out_root=out_root,
            table=table,
            date_col=date_col,
            timestamp_unit=timestamp_unit,
            chunk_index=chunk_index,
            compression=compression,
            prefix="part-db",
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export harvester SQLite DB tables to partitioned Parquet.")
    parser.add_argument("--symbol", "-s", required=True, help="BTC, ADA, BTCUSDT, ADAUSDT, etc.")
    parser.add_argument("--table", "-t", choices=DEFAULT_TABLES + ["all"], default="all", help="Table to export.")
    parser.add_argument("--db", default=None, help="Override SQLite DB path.")
    parser.add_argument("--project-root", default=str(project_root_from_script()), help="Trading project root.")
    parser.add_argument("--output-root", default=None, help="Override output root. Default: data/parquet/<COIN>.")
    parser.add_argument("--date-col", default="event_ts", help="Timestamp column used for date partitioning.")
    parser.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"], help="Timestamp unit.")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="Rows per DB read chunk.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression.")
    parser.add_argument("--keep-extra-columns", action="store_true", help="Keep DB columns outside known schema.")
    parser.add_argument("--include-id", action="store_true", help="Include SQLite autoincrement id column.")
    parser.add_argument("--start-id", type=int, default=None, help="Only export rows with id >= start-id.")
    parser.add_argument("--end-id", type=int, default=None, help="Only export rows with id <= end-id.")
    parser.add_argument("--start-ts", type=int, default=None, help="Only export rows with date-col >= start-ts.")
    parser.add_argument("--end-ts", type=int, default=None, help="Only export rows with date-col <= end-ts.")
    parser.add_argument("--dry-run", action="store_true", help="Show row counts without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing parquet table before writing.")
    args = parser.parse_args(argv)

    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise SystemExit("Missing dependency: pyarrow. Install it before running db_to_parquet.py") from exc

    coin, symbol = normalize_symbol(args.symbol)
    project_root = Path(args.project_root).resolve()
    db_path = Path(args.db).resolve() if args.db else default_db_path(project_root, coin)
    if not db_path.exists():
        print(f"[ERROR] DB not found: {db_path}", file=sys.stderr)
        return 2

    out_root = Path(args.output_root).resolve() if args.output_root else project_root / "data" / "parquet" / coin
    tables = DEFAULT_TABLES if args.table == "all" else [args.table]
    where_sql, params = build_where(args)

    print(f"[DB]   {db_path}")
    print(f"[OUT]  {out_root}")
    print(f"[MODE] read-only, table={args.table}, chunk_size={args.chunk_size:,}")
    if where_sql:
        print(f"[WHERE] {where_sql}  params={params}")

    grand_total = 0
    conn = open_readonly(db_path)
    try:
        for table in tables:
            grand_total += export_table(
                conn,
                table=table,
                symbol=symbol,
                out_root=out_root,
                date_col=args.date_col,
                timestamp_unit=args.timestamp_unit,
                chunk_size=args.chunk_size,
                compression=args.compression,
                strict=not args.keep_extra_columns,
                include_id=args.include_id,
                where_sql=where_sql,
                params=params,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
    finally:
        conn.close()

    print(f"[DONE] exported {grand_total:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
