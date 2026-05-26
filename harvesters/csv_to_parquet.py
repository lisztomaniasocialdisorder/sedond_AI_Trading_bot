#!/usr/bin/env python3
"""Convert harvester CSV exports into the project Parquet layout.

Output layout:
    data/parquet/<COIN>/<table>/date=YYYY-MM-DD/part-<timestamp>-<n>.parquet

Examples:
    python harvesters/csv_to_parquet.py --symbol BTC --table trades --input exports/trades.csv
    python harvesters/csv_to_parquet.py --symbol ADA --table orderbook_l1 --input exports/ada_l1_folder
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pandas. Install it before running csv_to_parquet.py") from exc


TABLE_SCHEMAS: dict[str, dict[str, str]] = {
    "trades": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "trade_ts": "Int64",
        "symbol": "string",
        "trade_id": "Int64",
        "price": "float64",
        "qty": "float64",
        "quote_qty": "float64",
        "is_buyer_maker": "boolean",
    },
    "agg_trades": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "trade_ts": "Int64",
        "symbol": "string",
        "agg_trade_id": "Int64",
        "price": "float64",
        "qty": "float64",
        "quote_qty": "float64",
        "first_trade_id": "Int64",
        "last_trade_id": "Int64",
        "is_buyer_maker": "boolean",
    },
    "orderbook_l1": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "update_id": "Int64",
        "symbol": "string",
        "bid_price": "float64",
        "bid_qty": "float64",
        "ask_price": "float64",
        "ask_qty": "float64",
        "spread": "float64",
        "spread_bps": "float64",
        "mid_price": "float64",
        "obi": "float64",
    },
    "orderbook_l5": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "update_id": "Int64",
        "symbol": "string",
        "level": "Int64",
        "bid_price": "float64",
        "bid_qty": "float64",
        "ask_price": "float64",
        "ask_qty": "float64",
    },
    "orderbook_l20": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "update_id": "Int64",
        "symbol": "string",
        "level": "Int64",
        "bid_price": "float64",
        "bid_qty": "float64",
        "ask_price": "float64",
        "ask_qty": "float64",
    },
    "orderbook_metrics": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "update_id": "Int64",
        "symbol": "string",
        "depth_type": "string",
        "total_bid_qty": "float64",
        "total_ask_qty": "float64",
        "total_bid_value": "float64",
        "total_ask_value": "float64",
        "depth_imbalance": "float64",
        "bid_vwap": "float64",
        "ask_vwap": "float64",
        "weighted_mid": "float64",
    },
    "mark_price": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "symbol": "string",
        "mark_price": "float64",
        "index_price": "float64",
        "est_settle_price": "float64",
        "last_funding_rate": "float64",
        "next_funding_time": "Int64",
    },
    "liquidations": {
        "local_ts": "float64",
        "event_ts": "Int64",
        "symbol": "string",
        "side": "string",
        "order_type": "string",
        "time_in_force": "string",
        "orig_qty": "float64",
        "price": "float64",
        "avg_price": "float64",
        "order_status": "string",
        "last_filled_qty": "float64",
        "filled_accum_qty": "float64",
        "trade_time": "Int64",
    },
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def iter_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.csv") if p.is_file())
    matches = sorted(Path().glob(str(path)))
    return [p for p in matches if p.is_file()]


def normalize_symbol(value: str) -> tuple[str, str]:
    raw = value.upper().replace("-", "").replace("_", "")
    coin = raw.replace("USDT", "")
    return coin, f"{coin}USDT"


def timestamp_to_date(series: pd.Series, unit: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if unit == "auto":
        finite = numeric.dropna()
        if finite.empty:
            unit = "ms"
        else:
            median = float(finite.abs().median())
            unit = "ms" if median > 10_000_000_000 else "s"
    return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce").dt.strftime("%Y-%m-%d")


def coerce_schema(df: pd.DataFrame, table: str, symbol: str, strict: bool) -> pd.DataFrame:
    schema = TABLE_SCHEMAS.get(table)
    if not schema:
        return df

    out = df.copy()
    if "symbol" in schema and "symbol" not in out.columns:
        out["symbol"] = symbol

    for col, dtype in schema.items():
        if col not in out.columns:
            out[col] = pd.NA
        if dtype in {"float64", "Int64"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if dtype == "Int64":
                out[col] = out[col].astype("Int64")
        elif dtype == "boolean":
            if out[col].dtype == object:
                out[col] = out[col].map({
                    True: True,
                    False: False,
                    "true": True,
                    "True": True,
                    "1": True,
                    1: True,
                    "false": False,
                    "False": False,
                    "0": False,
                    0: False,
                })
            out[col] = out[col].astype("boolean")
        elif dtype == "string":
            out[col] = out[col].astype("string")

    if strict:
        out = out[list(schema)]
    return out


def write_partitioned(
    df: pd.DataFrame,
    *,
    out_root: Path,
    table: str,
    date_col: str,
    timestamp_unit: str,
    file_index: int,
    chunk_index: int,
    compression: str,
) -> int:
    if df.empty:
        return 0

    if date_col not in df.columns:
        raise ValueError(f"timestamp/date column not found: {date_col}")

    dates = timestamp_to_date(df[date_col], timestamp_unit)
    written = 0
    ts_part = int(time.time())
    for date_str, part in df.groupby(dates, dropna=False):
        if not isinstance(date_str, str) or date_str == "NaT":
            date_str = "unknown"
        out_dir = out_root / table / f"date={date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"part-csv-{ts_part}-{file_index:04d}-{chunk_index:04d}-{written:03d}.parquet"
        part.to_parquet(out_file, index=False, compression=compression, engine="pyarrow")
        written += len(part)
        print(f"[WRITE] {len(part):,} rows -> {out_file}")
    return written


def convert_file(
    csv_file: Path,
    *,
    out_root: Path,
    table: str,
    symbol: str,
    date_col: str,
    timestamp_unit: str,
    chunk_size: int,
    compression: str,
    strict: bool,
    file_index: int,
) -> int:
    total = 0
    print(f"[READ] {csv_file}")
    reader = pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False)
    for chunk_index, chunk in enumerate(reader):
        chunk = coerce_schema(chunk, table, symbol, strict)
        total += write_partitioned(
            chunk,
            out_root=out_root,
            table=table,
            date_col=date_col,
            timestamp_unit=timestamp_unit,
            file_index=file_index,
            chunk_index=chunk_index,
            compression=compression,
        )
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert harvester CSV files to partitioned Parquet.")
    parser.add_argument("--input", "-i", required=True, help="CSV file, folder, or glob pattern.")
    parser.add_argument("--symbol", "-s", required=True, help="BTC, ADA, BTCUSDT, ADAUSDT, etc.")
    parser.add_argument("--table", "-t", required=True, choices=sorted(TABLE_SCHEMAS), help="Harvester table name.")
    parser.add_argument("--project-root", default=str(project_root_from_script()), help="Trading project root.")
    parser.add_argument("--output-root", default=None, help="Override output root. Default: data/parquet/<COIN>.")
    parser.add_argument("--date-col", default="event_ts", help="Timestamp column used for date partitioning.")
    parser.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"], help="Timestamp unit.")
    parser.add_argument("--chunk-size", type=int, default=250_000, help="CSV rows per chunk.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression.")
    parser.add_argument("--keep-extra-columns", action="store_true", help="Keep CSV columns outside known schema.")
    args = parser.parse_args(argv)

    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise SystemExit("Missing dependency: pyarrow. Install it before running csv_to_parquet.py") from exc

    coin, symbol = normalize_symbol(args.symbol)
    project_root = Path(args.project_root).resolve()
    out_root = Path(args.output_root).resolve() if args.output_root else project_root / "data" / "parquet" / coin
    csv_files = iter_csv_files(Path(args.input))
    if not csv_files:
        print(f"[ERROR] no CSV files found: {args.input}", file=sys.stderr)
        return 2

    grand_total = 0
    for idx, csv_file in enumerate(csv_files):
        grand_total += convert_file(
            csv_file,
            out_root=out_root,
            table=args.table,
            symbol=symbol,
            date_col=args.date_col,
            timestamp_unit=args.timestamp_unit,
            chunk_size=args.chunk_size,
            compression=args.compression,
            strict=not args.keep_extra_columns,
            file_index=idx,
        )

    print(f"[DONE] converted {len(csv_files)} file(s), {grand_total:,} rows")
    print(f"[OUT]  {out_root / args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
