#!/usr/bin/env python3
"""Compact live harvester Parquet spool files into rollup Parquet archives.

Input layout:
    data/parquet_spool/BTC/trades/date=2026-05-27/part-*.parquet

Output layout:
    data/parquet_rollup/BTC/daily/trades_2026-05-27.parquet
    data/parquet_rollup/BTC/monthly/trades_2026-05.parquet
    data/parquet_rollup/BTC/quarterly/trades_2026-Q2.parquet
    data/parquet_rollup/BTC/yearly/trades_2026.parquet

By default this compacts only fully closed UTC days and deletes processed spool
folders after a verified write. Use --include-open-day for manual same-day tests.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyarrow") from exc

try:
    from archive_prune_db import rebuild_period_rollups
    from csv_to_parquet import normalize_symbol
    from db_to_parquet import DEFAULT_TABLES
except Exception as exc:  # pragma: no cover
    raise SystemExit("compact_parquet_spool.py must live next to archive_prune_db.py") from exc


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def today_utc_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def date_from_dir(path: Path) -> str | None:
    name = path.name
    if not name.startswith("date="):
        return None
    value = name.split("=", 1)[1]
    if len(value) != 10:
        return None
    return value


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def compact_day(date_dir: Path, out_file: Path, *, compression: str, batch_size: int) -> int:
    files = sorted(date_dir.glob("*.parquet"))
    if not files:
        return 0

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp_file.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    writer_schema = None
    total_rows = 0
    try:
        for src in files:
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer_schema = table.schema
                    writer = pq.ParquetWriter(tmp_file, writer_schema, compression=compression)
                else:
                    # Normalize minor schema drifts across spool parts
                    # (e.g., string vs large_string in the same column).
                    if table.schema != writer_schema:
                        table = table.cast(writer_schema, safe=False)
                writer.write_table(table)
                total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows <= 0:
        tmp_file.unlink(missing_ok=True)
        return 0

    actual_rows = parquet_rows(tmp_file)
    if actual_rows != total_rows:
        tmp_file.unlink(missing_ok=True)
        raise RuntimeError(f"verification failed for {tmp_file}: {actual_rows:,} != {total_rows:,}")

    tmp_file.replace(out_file)
    return total_rows


def compact_symbol(args: argparse.Namespace, raw_symbol: str) -> int:
    coin, _symbol = normalize_symbol(raw_symbol)
    project_root = Path(args.project_root).resolve()
    spool_symbol = (Path(args.spool_root).resolve() if args.spool_root else project_root / "data" / "parquet_spool") / coin
    rollup_symbol = (Path(args.output_root).resolve() if args.output_root else project_root / "data" / "parquet_rollup") / coin
    tables = DEFAULT_TABLES if args.table == "all" else [args.table]
    current_day = today_utc_key()

    if not spool_symbol.exists():
        print(f"[SKIP] spool not found: {spool_symbol}")
        return 0

    print(f"[SPOOL] {spool_symbol}")
    print(f"[OUT]   {rollup_symbol}")
    total = 0

    for table in tables:
        table_dir = spool_symbol / table
        if not table_dir.exists():
            print(f"[SKIP] {coin} {table}: no spool folder")
            continue

        table_rows = 0
        touched = False
        for date_dir in sorted(p for p in table_dir.iterdir() if p.is_dir()):
            day_key = date_from_dir(date_dir)
            if day_key is None:
                print(f"[SKIP] {coin} {table}: unknown partition {date_dir.name}")
                continue
            if not args.include_open_day and day_key >= current_day:
                print(f"[KEEP] {coin} {table} {day_key}: open UTC day")
                continue

            out_file = rollup_symbol / "daily" / f"{table}_{day_key}.parquet"
            if out_file.exists() and day_key < current_day and not args.rebuild_existing:
                print(f"[KEEP] {coin} {table} {day_key}: existing closed-day rollup ({out_file})")
                continue

            rows = compact_day(date_dir, out_file, compression=args.compression, batch_size=args.batch_size)
            if rows <= 0:
                continue

            actual = parquet_rows(out_file)
            if actual != rows:
                raise RuntimeError(f"daily output verification failed: {out_file}")

            table_rows += rows
            total += rows
            touched = True
            print(f"[DAY] {coin} {table} {day_key}: {rows:,} rows -> {out_file}")

            if args.delete_spool:
                shutil.rmtree(date_dir)
                print(f"[CLEAN] removed {date_dir}")

        if touched:
            rebuild_period_rollups(rollup_symbol, table, compression=args.compression)
        print(f"[TABLE] {coin} {table}: compacted {table_rows:,} rows")

    print(f"[DONE] {coin}: compacted {total:,} rows")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compact harvester Parquet spool files into rollup archives.")
    parser.add_argument("--symbols", default="BTC,ADA")
    parser.add_argument("--table", choices=DEFAULT_TABLES + ["all"], default="all")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--spool-root", default=None, help="Default: data/parquet_spool")
    parser.add_argument("--output-root", default=None, help="Default: data/parquet_rollup")
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--include-open-day", action="store_true", help="Also compact the current UTC day.")
    parser.add_argument("--delete-spool", action="store_true", help="Delete verified spool date folders after writing.")
    parser.add_argument("--rebuild-existing", action="store_true", help="Allow overwriting existing closed-day daily rollups.")
    args = parser.parse_args(argv)

    try:
        for raw_symbol in [s.strip() for s in args.symbols.split(",") if s.strip()]:
            compact_symbol(args, raw_symbol)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
