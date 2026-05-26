#!/usr/bin/env python3
"""Export harvester SQLite DB tables to Parquet.

Reads DB in read-only mode, so it can run while harvesters are active.

Default partitioned layout:
    data/parquet/<COIN>/<table>/date=YYYY-MM-DD/part-db-*.parquet

Rollup layout:
    data/parquet/<COIN>/daily/<table>/date=YYYY-MM-DD/<table>_YYYY-MM-DD.parquet
    data/parquet/<COIN>/monthly/<table>/month=YYYY-MM/<table>_YYYY-MM.parquet
    data/parquet/<COIN>/quarterly/<table>/quarter=YYYY-QN/<table>_YYYY-QN.parquet
    data/parquet/<COIN>/yearly/<table>/year=YYYY/<table>_YYYY.parquet

Examples:
    python harvesters/db_to_parquet.py --symbol BTC
    python harvesters/db_to_parquet.py --symbol ADA --table trades
    python harvesters/db_to_parquet.py --symbol BTC --table orderbook_l1 --start-id 1000000
    python harvesters/db_to_parquet.py --symbol BTC --layout rollup --periods day,month,quarter,year
"""
from __future__ import annotations

import argparse
import re
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

PERIOD_DIRS = {
    "day": ("daily", "date"),
    "month": ("monthly", "month"),
    "quarter": ("quarterly", "quarter"),
    "year": ("yearly", "year"),
}


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


def parse_periods(value: str) -> list[str]:
    periods = [p.strip().lower() for p in value.split(",") if p.strip()]
    invalid = [p for p in periods if p not in PERIOD_DIRS]
    if invalid:
        valid = ", ".join(PERIOD_DIRS)
        raise argparse.ArgumentTypeError(f"invalid period(s): {', '.join(invalid)}. Valid: {valid}")
    return periods or ["day"]


def timestamp_to_period(series: pd.Series, unit: str, period: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if unit == "auto":
        finite = numeric.dropna()
        if finite.empty:
            unit = "ms"
        else:
            median = float(finite.abs().median())
            unit = "ms" if median > 10_000_000_000 else "s"

    ts = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    if period == "day":
        return ts.dt.strftime("%Y-%m-%d")
    if period == "month":
        return ts.dt.strftime("%Y-%m")
    if period == "quarter":
        quarter = ((ts.dt.month - 1) // 3) + 1
        return ts.dt.year.astype("Int64").astype("string") + "-Q" + quarter.astype("Int64").astype("string")
    if period == "year":
        return ts.dt.strftime("%Y")
    raise ValueError(f"unsupported period: {period}")


def safe_period_key(value: object) -> str:
    if not isinstance(value, str) or value in {"NaT", "<NA>", "nan", "None", ""}:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._=-]+", "_", value)


class RollupParquetWriters:
    def __init__(self, *, out_root: Path, table: str, compression: str) -> None:
        self.out_root = out_root
        self.table = table
        self.compression = compression
        self._writers: dict[tuple[str, str], object] = {}
        self._counts: dict[tuple[str, str], int] = {}
        self._paths: dict[tuple[str, str], Path] = {}

    def _path_for(self, period: str, key: str) -> Path:
        period_dir, partition_name = PERIOD_DIRS[period]
        out_dir = self.out_root / period_dir / self.table / f"{partition_name}={key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{self.table}_{key}.parquet"

    def write(self, df: pd.DataFrame, *, period: str, key: str) -> int:
        import pyarrow as pa
        import pyarrow.parquet as pq

        safe_key = safe_period_key(key)
        table_key = (period, safe_key)
        out_file = self._path_for(period, safe_key)
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)

        writer = self._writers.get(table_key)
        if writer is None:
            writer = pq.ParquetWriter(out_file, arrow_table.schema, compression=self.compression)
            self._writers[table_key] = writer
            self._paths[table_key] = out_file

        writer.write_table(arrow_table)
        self._counts[table_key] = self._counts.get(table_key, 0) + len(df)
        return len(df)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        for key, rows in sorted(self._counts.items()):
            out_file = self._paths[key]
            print(f"[WRITE] {rows:,} rows -> {out_file}")


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


def write_rollups(
    df: pd.DataFrame,
    *,
    writers: RollupParquetWriters,
    date_col: str,
    timestamp_unit: str,
    periods: list[str],
) -> int:
    if df.empty:
        return 0
    if date_col not in df.columns:
        raise ValueError(f"timestamp/date column not found: {date_col}")

    written = 0
    for period in periods:
        keys = timestamp_to_period(df[date_col], timestamp_unit, period)
        for key, part in df.groupby(keys, dropna=False):
            written += writers.write(part, period=period, key=safe_period_key(key))
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
    layout: str,
    periods: list[str],
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
        import shutil

        if layout == "rollup":
            for period in periods:
                period_dir, _partition_name = PERIOD_DIRS[period]
                target = out_root / period_dir / table
                if target.exists():
                    print(f"[OVERWRITE] removing existing rollup table: {target}")
                    shutil.rmtree(target)
        else:
            target = out_root / table
            if target.exists():
                print(f"[OVERWRITE] removing existing parquet table: {target}")
                shutil.rmtree(target)

    col_sql = ", ".join(select_cols)
    sql = f"SELECT {col_sql} FROM {table} {where_sql} ORDER BY id"
    written = 0
    if layout == "rollup":
        writers = RollupParquetWriters(out_root=out_root, table=table, compression=compression)
        try:
            for _chunk_index, chunk in enumerate(pd.read_sql_query(sql, conn, params=params, chunksize=chunk_size)):
                chunk = coerce_schema(chunk, table, symbol, strict)
                written += write_rollups(
                    chunk,
                    writers=writers,
                    date_col=date_col,
                    timestamp_unit=timestamp_unit,
                    periods=periods,
                )
        finally:
            writers.close()
    else:
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
    parser = argparse.ArgumentParser(description="Export harvester SQLite DB tables to Parquet.")
    parser.add_argument("--symbol", "-s", required=True, help="BTC, ADA, BTCUSDT, ADAUSDT, etc.")
    parser.add_argument("--table", "-t", choices=DEFAULT_TABLES + ["all"], default="all", help="Table to export.")
    parser.add_argument("--db", default=None, help="Override SQLite DB path.")
    parser.add_argument("--project-root", default=str(project_root_from_script()), help="Trading project root.")
    parser.add_argument("--output-root", default=None, help="Override output root. Default: data/parquet/<COIN>.")
    parser.add_argument("--date-col", default="event_ts", help="Timestamp column used for date partitioning.")
    parser.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"], help="Timestamp unit.")
    parser.add_argument("--layout", default="partitioned", choices=["partitioned", "rollup"], help="Output layout.")
    parser.add_argument(
        "--periods",
        type=parse_periods,
        default=parse_periods("day"),
        help="Rollup periods for --layout rollup. Example: day,month,quarter,year",
    )
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
    print(f"[MODE] read-only, table={args.table}, layout={args.layout}, chunk_size={args.chunk_size:,}")
    if args.layout == "rollup":
        print(f"[ROLLUP] periods={','.join(args.periods)}")
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
                layout=args.layout,
                periods=args.periods,
            )
    finally:
        conn.close()

    print(f"[DONE] exported {grand_total:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
