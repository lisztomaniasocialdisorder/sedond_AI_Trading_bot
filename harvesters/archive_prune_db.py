#!/usr/bin/env python3
"""Archive sealed harvester DB days to Parquet, then optionally prune SQLite rows.

This is the safe path for shrinking live harvester DB files:
1. Export only fully closed UTC days to daily Parquet files.
2. Read the Parquet files back and compare row counts.
3. Delete only verified rows from SQLite when --prune is passed.
4. Rebuild monthly/quarterly/yearly files from the daily Parquet archive.

Output layout:
    data/parquet_rollup/BTC/daily/trades_2026-05-27.parquet
    data/parquet_rollup/BTC/monthly/trades_2026-05.parquet
    data/parquet_rollup/BTC/quarterly/trades_2026-Q2.parquet
    data/parquet_rollup/BTC/yearly/trades_2026.parquet
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    raise SystemExit("archive_prune_db.py must live next to db_to_parquet.py") from exc


DAY_MS = 86_400_000
PERIODS = ("monthly", "quarterly", "yearly")


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_db_path(project_root: Path, coin: str) -> Path:
    return project_root / "harvesters" / f"{coin}_harvester" / "raw_db" / f"microstructure_{coin}.db"


def utc_midnight_ms(value: datetime | None = None) -> int:
    now = value or datetime.now(timezone.utc)
    midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return int(midnight.timestamp() * 1000)


def day_key_from_ms(day_start_ms: int) -> str:
    return datetime.fromtimestamp(day_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def monthly_key(day_key: str) -> str:
    return day_key[:7]


def quarterly_key(day_key: str) -> str:
    year = int(day_key[:4])
    month = int(day_key[5:7])
    quarter = ((month - 1) // 3) + 1
    return f"{year}-Q{quarter}"


def yearly_key(day_key: str) -> str:
    return day_key[:4]


def period_key(day_key: str, period: str) -> str:
    if period == "monthly":
        return monthly_key(day_key)
    if period == "quarterly":
        return quarterly_key(day_key)
    if period == "yearly":
        return yearly_key(day_key)
    raise ValueError(f"unknown period: {period}")


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def min_max_ts(conn: sqlite3.Connection, table: str, date_col: str) -> tuple[int | None, int | None]:
    row = conn.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table}").fetchone()
    if row is None or row[0] is None or row[1] is None:
        return None, None
    return int(row[0]), int(row[1])


def count_rows(conn: sqlite3.Connection, table: str, date_col: str, start_ms: int, end_ms: int) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {date_col} >= ? AND {date_col} < ?",
        (start_ms, end_ms),
    ).fetchone()
    return int(row[0] or 0)


def parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def export_day(
    conn: sqlite3.Connection,
    *,
    table: str,
    symbol: str,
    out_file: Path,
    date_col: str,
    day_start_ms: int,
    day_end_ms: int,
    chunk_size: int,
    compression: str,
    keep_extra_columns: bool,
) -> int:
    columns = table_columns(conn, table)
    col_sql = ", ".join(columns)
    sql = (
        f"SELECT {col_sql} FROM {table} "
        f"WHERE {date_col} >= ? AND {date_col} < ? ORDER BY id"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        for chunk in pd.read_sql_query(sql, conn, params=(day_start_ms, day_end_ms), chunksize=chunk_size):
            chunk = coerce_schema(chunk, table, symbol, strict=not keep_extra_columns)
            arrow_table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_file, arrow_table.schema, compression=compression)
            writer.write_table(arrow_table)
            written += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    if written <= 0:
        if tmp_file.exists():
            tmp_file.unlink()
        return 0

    tmp_rows = parquet_rows(tmp_file)
    if tmp_rows != written:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet verification failed: {tmp_file} rows={tmp_rows}, expected={written}")

    tmp_file.replace(out_file)
    return written


def delete_day(conn: sqlite3.Connection, table: str, date_col: str, day_start_ms: int, day_end_ms: int) -> int:
    before = conn.total_changes
    conn.execute(f"DELETE FROM {table} WHERE {date_col} >= ? AND {date_col} < ?", (day_start_ms, day_end_ms))
    conn.commit()
    return int(conn.total_changes - before)


def iter_day_starts(min_ts: int, cutoff_ms: int) -> list[int]:
    first = (int(min_ts) // DAY_MS) * DAY_MS
    return list(range(first, int(cutoff_ms), DAY_MS))


def rebuild_period_rollups(symbol_root: Path, table: str, *, compression: str) -> None:
    daily_files = sorted((symbol_root / "daily").glob(f"{table}_????-??-??.parquet"))
    if not daily_files:
        return

    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in daily_files:
        day_key = path.stem.removeprefix(f"{table}_")
        for period in PERIODS:
            grouped.setdefault((period, period_key(day_key, period)), []).append(path)

    for (period, key), files in sorted(grouped.items()):
        out_dir = symbol_root / period
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{table}_{key}.parquet"
        tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
        if tmp_file.exists():
            tmp_file.unlink()

        writer: pq.ParquetWriter | None = None
        total_rows = 0
        try:
            for daily in files:
                pf = pq.ParquetFile(daily)
                for batch in pf.iter_batches(batch_size=250_000):
                    arrow_table = pa.Table.from_batches([batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_file, arrow_table.schema, compression=compression)
                    writer.write_table(arrow_table)
                    total_rows += arrow_table.num_rows
        finally:
            if writer is not None:
                writer.close()

        if total_rows > 0:
            tmp_file.replace(out_file)
            print(f"[ROLLUP] {total_rows:,} rows -> {out_file}")


def archive_symbol(args: argparse.Namespace, raw_symbol: str) -> int:
    coin, symbol = normalize_symbol(raw_symbol)
    project_root = Path(args.project_root).resolve()
    db_path = default_db_path(project_root, coin)
    if not db_path.exists():
        print(f"[SKIP] DB not found: {db_path}")
        return 0

    symbol_root = Path(args.output_root).resolve() / coin if args.output_root else project_root / "data" / "parquet_rollup" / coin
    cutoff_ms = int(args.cutoff_ts) if args.cutoff_ts is not None else utc_midnight_ms()
    tables = DEFAULT_TABLES if args.table == "all" else [args.table]
    print(f"[DB]  {db_path}")
    print(f"[OUT] {symbol_root}")
    print(f"[CUT] archive rows with {args.date_col} < {cutoff_ms} ({day_key_from_ms(cutoff_ms)})")

    total_archived = 0
    total_deleted = 0
    conn = connect_db(db_path)
    try:
        for table in tables:
            if not table_exists(conn, table):
                print(f"[SKIP] {coin} {table}: table not found")
                continue
            columns = table_columns(conn, table)
            if args.date_col not in columns:
                print(f"[SKIP] {coin} {table}: missing {args.date_col}")
                continue

            min_ts, _max_ts = min_max_ts(conn, table, args.date_col)
            if min_ts is None:
                print(f"[EMPTY] {coin} {table}")
                continue

            archived_table = 0
            deleted_table = 0
            for day_start in iter_day_starts(min_ts, cutoff_ms):
                day_end = day_start + DAY_MS
                day_key = day_key_from_ms(day_start)
                db_rows = count_rows(conn, table, args.date_col, day_start, day_end)
                if db_rows <= 0:
                    continue

                out_file = symbol_root / "daily" / f"{table}_{day_key}.parquet"
                existing_rows = parquet_rows(out_file)
                if existing_rows > db_rows:
                    raise RuntimeError(
                        f"Refusing to overwrite larger archive {out_file}: "
                        f"parquet={existing_rows:,}, db={db_rows:,}"
                    )
                if existing_rows != db_rows:
                    written = export_day(
                        conn,
                        table=table,
                        symbol=symbol,
                        out_file=out_file,
                        date_col=args.date_col,
                        day_start_ms=day_start,
                        day_end_ms=day_end,
                        chunk_size=args.chunk_size,
                        compression=args.compression,
                        keep_extra_columns=args.keep_extra_columns,
                    )
                    existing_rows = parquet_rows(out_file)
                    print(f"[DAY] {coin} {table} {day_key}: wrote {written:,} rows -> {out_file}")

                if existing_rows != db_rows:
                    raise RuntimeError(f"Refusing to prune {coin} {table} {day_key}: parquet={existing_rows:,}, db={db_rows:,}")

                archived_table += db_rows
                if args.prune:
                    deleted = delete_day(conn, table, args.date_col, day_start, day_end)
                    deleted_table += deleted
                    print(f"[PRUNE] {coin} {table} {day_key}: deleted {deleted:,} verified rows")

            if archived_table:
                rebuild_period_rollups(symbol_root, table, compression=args.compression)
            print(f"[TABLE] {coin} {table}: archived {archived_table:,}, deleted {deleted_table:,}")
            total_archived += archived_table
            total_deleted += deleted_table

        if args.prune and args.vacuum:
            print(f"[VACUUM] compacting {db_path}")
            conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"[DONE] {coin}: archived {total_archived:,}, deleted {total_deleted:,}")
    return total_archived


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Archive sealed DB days to Parquet and optionally prune SQLite rows.")
    parser.add_argument("--symbols", default="BTC,ADA", help="Comma-separated symbols. Default: BTC,ADA")
    parser.add_argument("--table", choices=DEFAULT_TABLES + ["all"], default="all")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--output-root", default=None, help="Default: data/parquet_rollup")
    parser.add_argument("--date-col", default="event_ts")
    parser.add_argument("--cutoff-ts", type=int, default=None, help="Archive rows with date-col < cutoff-ts in ms. Default: current UTC midnight.")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--keep-extra-columns", action="store_true")
    parser.add_argument("--prune", action="store_true", help="Delete verified archived rows from SQLite.")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after pruning to shrink DB file size.")
    args = parser.parse_args(argv)

    try:
        for raw_symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
            archive_symbol(args, raw_symbol)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
