#!/usr/bin/env python3
"""Build clean daily/monthly/quarterly/yearly Parquet rollups from harvester DBs.

Output layout:
    data/parquet_rollup/BTC/daily/trades_2026-05-27.parquet
    data/parquet_rollup/BTC/monthly/trades_2026-05.parquet
    data/parquet_rollup/BTC/quarterly/trades_2026-Q2.parquet
    data/parquet_rollup/BTC/yearly/trades_2026.parquet

Each SQLite table becomes its own Parquet file because tables have different schemas.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyarrow") from exc

try:
    from csv_to_parquet import coerce_schema, normalize_symbol
    from db_to_parquet import DEFAULT_TABLES, table_columns, table_exists
except Exception as exc:  # pragma: no cover
    raise SystemExit("auto_parquet_rollup.py must live next to db_to_parquet.py") from exc


PERIODS = ("daily", "monthly", "quarterly", "yearly")


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path(project_root: Path, coin: str) -> Path:
    return project_root / "harvesters" / f"{coin}_harvester" / "raw_db" / f"microstructure_{coin}.db"


def open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def infer_timestamp_unit(values: pd.Series, timestamp_unit: str) -> str:
    if timestamp_unit != "auto":
        return timestamp_unit
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric.dropna()
    if finite.empty:
        return "ms"
    return "ms" if float(finite.abs().median()) > 10_000_000_000 else "s"


def period_key(values: pd.Series, *, timestamp_unit: str, period: str) -> pd.Series:
    unit = infer_timestamp_unit(values, timestamp_unit)
    ts = pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit=unit, utc=True, errors="coerce")
    if period == "daily":
        return ts.dt.strftime("%Y-%m-%d")
    if period == "monthly":
        return ts.dt.strftime("%Y-%m")
    if period == "quarterly":
        quarter = ((ts.dt.month - 1) // 3) + 1
        return ts.dt.year.astype("Int64").astype("string") + "-Q" + quarter.astype("Int64").astype("string")
    if period == "yearly":
        return ts.dt.strftime("%Y")
    raise ValueError(f"unknown period: {period}")


def safe_key(value: object) -> str:
    if not isinstance(value, str) or value in {"", "NaT", "<NA>", "nan", "None"}:
        return "unknown"
    return value.replace("/", "-").replace("\\", "-").replace(":", "-")


class ParquetRollupWriters:
    def __init__(self, *, out_root: Path, table: str, compression: str) -> None:
        self.out_root = out_root
        self.table = table
        self.compression = compression
        self.writers: dict[tuple[str, str], pq.ParquetWriter] = {}
        self.paths: dict[tuple[str, str], Path] = {}
        self.counts: dict[tuple[str, str], int] = {}

    def write(self, df: pd.DataFrame, *, period: str, key: str) -> None:
        clean_key = safe_key(key)
        writer_key = (period, clean_key)
        out_dir = self.out_root / period
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{self.table}_{clean_key}.parquet"

        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        writer = self.writers.get(writer_key)
        if writer is None:
            writer = pq.ParquetWriter(out_file, arrow_table.schema, compression=self.compression)
            self.writers[writer_key] = writer
            self.paths[writer_key] = out_file

        writer.write_table(arrow_table)
        self.counts[writer_key] = self.counts.get(writer_key, 0) + len(df)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        for key, rows in sorted(self.counts.items()):
            print(f"[WRITE] {rows:,} rows -> {self.paths[key]}")


def build_where(args: argparse.Namespace, date_col: str) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if args.start_ts is not None:
        clauses.append(f"{date_col} >= ?")
        params.append(int(args.start_ts))
    if args.end_ts is not None:
        clauses.append(f"{date_col} <= ?")
        params.append(int(args.end_ts))
    if args.start_id is not None:
        clauses.append("id >= ?")
        params.append(int(args.start_id))
    if args.end_id is not None:
        clauses.append("id <= ?")
        params.append(int(args.end_id))
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


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
    keep_extra_columns: bool,
    overwrite: bool,
    periods: list[str],
    where_sql: str,
    params: list[object],
) -> int:
    if not table_exists(conn, table):
        print(f"[SKIP] {table}: table not found")
        return 0

    columns = table_columns(conn, table)
    if date_col not in columns:
        print(f"[SKIP] {table}: timestamp column not found: {date_col}")
        return 0

    if overwrite:
        for period in periods:
            for old_file in (out_root / period).glob(f"{table}_*.parquet"):
                old_file.unlink()

    total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {table} {where_sql}", params).fetchone()[0])
    print(f"[TABLE] {table}: {total_rows:,} rows")
    if total_rows == 0:
        return 0

    writers = ParquetRollupWriters(out_root=out_root, table=table, compression=compression)
    written = 0
    col_sql = ", ".join(columns)
    sql = f"SELECT {col_sql} FROM {table} {where_sql} ORDER BY id"
    try:
        for chunk in pd.read_sql_query(sql, conn, params=params, chunksize=chunk_size):
            chunk = coerce_schema(chunk, table, symbol, strict=not keep_extra_columns)
            for period in periods:
                keys = period_key(chunk[date_col], timestamp_unit=timestamp_unit, period=period)
                for key, part in chunk.groupby(keys, dropna=False):
                    writers.write(part, period=period, key=safe_key(key))
                    written += len(part)
    finally:
        writers.close()
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build daily/monthly/quarterly/yearly Parquet rollups from harvester DBs.")
    parser.add_argument("--symbols", default="BTC,ADA", help="Comma-separated symbols. Default: BTC,ADA")
    parser.add_argument("--table", choices=DEFAULT_TABLES + ["all"], default="all")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--output-root", default=None, help="Default: data/parquet_rollup")
    parser.add_argument("--date-col", default="event_ts")
    parser.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"])
    parser.add_argument("--periods", default="daily,monthly,quarterly,yearly", help="daily,monthly,quarterly,yearly")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--keep-extra-columns", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing rollup files for exported tables.")
    parser.add_argument("--start-id", type=int, default=None)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--start-ts", type=int, default=None)
    parser.add_argument("--end-ts", type=int, default=None)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "data" / "parquet_rollup"
    periods = [p.strip().lower() for p in args.periods.split(",") if p.strip()]
    invalid = [p for p in periods if p not in PERIODS]
    if invalid:
        print(f"[ERROR] invalid periods: {', '.join(invalid)}", file=sys.stderr)
        return 2

    tables = DEFAULT_TABLES if args.table == "all" else [args.table]
    grand_total = 0

    for raw_symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        coin, symbol = normalize_symbol(raw_symbol)
        db_path = default_db_path(project_root, coin)
        if not db_path.exists():
            print(f"[SKIP] DB not found: {db_path}")
            continue

        symbol_out = output_root / coin
        where_sql, params = build_where(args, args.date_col)
        print(f"[DB]  {db_path}")
        print(f"[OUT] {symbol_out}")
        if where_sql:
            print(f"[WHERE] {where_sql} params={params}")

        conn = open_readonly(db_path)
        try:
            for table in tables:
                grand_total += export_table(
                    conn,
                    table=table,
                    symbol=symbol,
                    out_root=symbol_out,
                    date_col=args.date_col,
                    timestamp_unit=args.timestamp_unit,
                    chunk_size=args.chunk_size,
                    compression=args.compression,
                    keep_extra_columns=args.keep_extra_columns,
                    overwrite=args.overwrite,
                    periods=periods,
                    where_sql=where_sql,
                    params=params,
                )
        finally:
            conn.close()

    print(f"[DONE] rollup exported {grand_total:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
