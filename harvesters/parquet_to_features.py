#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.errors import PerformanceWarning


warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="Sorting by default when concatenating all DatetimeIndex.*")


ROLL_WINDOWS_SEC = (5, 15, 60)
LABEL_HORIZONS_SEC = (5, 15, 30, 60)


@dataclass(frozen=True)
class DailySources:
    date_key: str
    l1: Path | None = None
    l5: Path | None = None
    l20: Path | None = None
    metrics: Path | None = None
    trades: Path | None = None

    @property
    def paths(self) -> list[Path]:
        return [p for p in (self.l1, self.l5, self.l20, self.metrics, self.trades) if p and p.exists()]


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_coin(raw: str) -> str:
    return raw.strip().upper().replace("USDT", "")


def _date_from_daily_name(path: Path, table: str) -> str | None:
    m = re.match(rf"{re.escape(table)}_(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$", path.name)
    return m.group(1) if m else None


def _daily_map(symbol_root: Path, table: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    daily = symbol_root / "daily"
    for p in sorted(daily.glob(f"{table}_*.parquet")):
        key = _date_from_daily_name(p, table)
        if key:
            out[key] = p
    return out


def discover_daily_sources(project_root: Path, coin: str, limit_days: int | None = None) -> list[DailySources]:
    symbol_root = project_root / "data" / "parquet_rollup" / coin
    maps = {
        "l1": _daily_map(symbol_root, "orderbook_l1"),
        "l5": _daily_map(symbol_root, "orderbook_l5"),
        "l20": _daily_map(symbol_root, "orderbook_l20"),
        "metrics": _daily_map(symbol_root, "orderbook_metrics"),
        "trades": _daily_map(symbol_root, "trades"),
    }
    dates = sorted(set().union(*(set(v) for v in maps.values())))
    if limit_days is not None and limit_days > 0:
        dates = dates[-int(limit_days):]
    return [
        DailySources(
            date_key=d,
            l1=maps["l1"].get(d),
            l5=maps["l5"].get(d),
            l20=maps["l20"].get(d),
            metrics=maps["metrics"].get(d),
            trades=maps["trades"].get(d),
        )
        for d in dates
    ]


def _columns(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return []


def _read(path: Path | None, wanted: Iterable[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    cols = [c for c in wanted if c in set(_columns(path))]
    if not cols:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=cols)


def _ts_ms(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col not in df.columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        if col == "local_ts":
            return (raw * 1000.0).round()
        return raw
    return pd.Series(np.nan, index=df.index)


def _to_second_index(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    ts = _ts_ms(df, candidates)
    sec = np.floor(ts / 1000.0)
    return pd.to_datetime(sec, unit="s", utc=True, errors="coerce")


def _flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    df.columns = [f"{prefix}_{a}_{b}" if b else f"{prefix}_{a}" for a, b in df.columns.to_flat_index()]
    return df


def _safe_imbalance(bid: pd.Series, ask: pd.Series) -> pd.Series:
    den = bid + ask
    return (bid - ask) / den.replace(0, np.nan)


def build_l1_features(path: Path | None) -> pd.DataFrame:
    raw_cols = [
        "event_ts",
        "local_ts",
        "second_ts",
        "bid_price",
        "bid_qty",
        "ask_price",
        "ask_qty",
        "spread",
        "spread_bps",
        "mid_price",
        "obi",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
        "bid_qty_mean",
        "ask_qty_mean",
        "spread_bps_mean",
        "spread_bps_std",
        "obi_mean",
        "obi_std",
        "obi_open",
        "obi_close",
        "tick_count",
    ]
    df = _read(path, raw_cols)
    if df.empty:
        return pd.DataFrame()

    # Newer DB-derived L1 files are already one row per second.
    if "second_ts" in df.columns and "obi_mean" in df.columns:
        idx = _to_second_index(df, ("second_ts", "event_ts", "local_ts"))
        out = pd.DataFrame(index=idx)
        rename = {
            "mid_open": "l1_mid_open",
            "mid_high": "l1_mid_high",
            "mid_low": "l1_mid_low",
            "mid_close": "l1_mid_close",
            "bid_price": "l1_bid_price",
            "ask_price": "l1_ask_price",
            "bid_qty_mean": "l1_bid_qty_mean",
            "ask_qty_mean": "l1_ask_qty_mean",
            "spread_bps_mean": "l1_spread_bps_mean",
            "spread_bps_std": "l1_spread_bps_std",
            "obi_mean": "l1_obi_mean",
            "obi_std": "l1_obi_std",
            "obi_open": "l1_obi_open",
            "obi_close": "l1_obi_close",
            "tick_count": "l1_tick_count",
        }
        for src, dst in rename.items():
            if src in df.columns:
                out[dst] = pd.to_numeric(df[src], errors="coerce").to_numpy()
        return out[~out.index.isna()].sort_index()

    idx = _to_second_index(df, ("event_ts", "local_ts"))
    df = df.assign(_ts=idx).dropna(subset=["_ts"])
    for col in ("bid_qty", "ask_qty", "spread_bps", "mid_price", "obi", "bid_price", "ask_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "obi" not in df.columns and {"bid_qty", "ask_qty"} <= set(df.columns):
        df["obi"] = _safe_imbalance(df["bid_qty"], df["ask_qty"])
    if "mid_price" not in df.columns and {"bid_price", "ask_price"} <= set(df.columns):
        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0

    agg_spec = {}
    if "mid_price" in df.columns:
        agg_spec["mid"] = ("mid_price", ["first", "max", "min", "last", "mean"])
    if "spread_bps" in df.columns:
        agg_spec["spread_bps"] = ("spread_bps", ["mean", "std"])
    if "obi" in df.columns:
        agg_spec["obi"] = ("obi", ["mean", "std", "first", "last"])
    if "bid_qty" in df.columns:
        agg_spec["bid_qty"] = ("bid_qty", ["mean"])
    if "ask_qty" in df.columns:
        agg_spec["ask_qty"] = ("ask_qty", ["mean"])
    if not agg_spec:
        return pd.DataFrame()

    grouped = df.groupby("_ts", sort=True)
    pieces = []
    for name, (col, funcs) in agg_spec.items():
        part = grouped[col].agg(funcs)
        part.columns = [f"l1_{name}_{fn}" for fn in funcs]
        pieces.append(part)
    out = pd.concat(pieces, axis=1)
    out["l1_tick_count"] = grouped.size()
    out = out.rename(
        columns={
            "l1_mid_first": "l1_mid_open",
            "l1_mid_max": "l1_mid_high",
            "l1_mid_min": "l1_mid_low",
            "l1_mid_last": "l1_mid_close",
            "l1_obi_first": "l1_obi_open",
            "l1_obi_last": "l1_obi_close",
        }
    )
    return out


def build_metrics_features(path: Path | None) -> pd.DataFrame:
    cols = [
        "event_ts",
        "local_ts",
        "depth_type",
        "total_bid_qty",
        "total_ask_qty",
        "total_bid_value",
        "total_ask_value",
        "depth_imbalance",
        "bid_vwap",
        "ask_vwap",
        "weighted_mid",
    ]
    df = _read(path, cols)
    if df.empty:
        return pd.DataFrame()
    df["_ts"] = _to_second_index(df, ("event_ts", "local_ts"))
    df = df.dropna(subset=["_ts"])
    if "depth_type" not in df.columns:
        return pd.DataFrame()
    for col in (
        "total_bid_qty",
        "total_ask_qty",
        "total_bid_value",
        "total_ask_value",
        "depth_imbalance",
        "bid_vwap",
        "ask_vwap",
        "weighted_mid",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    parts = []
    for depth_type, prefix in (("l5", "l5"), ("l20", "l20")):
        d = df[df["depth_type"].astype(str).str.lower().eq(depth_type)].copy()
        if d.empty:
            continue
        grouped = d.groupby("_ts", sort=True)
        spec = {
            "depth_imbalance": ["mean", "std", "last"],
            "total_bid_qty": ["mean"],
            "total_ask_qty": ["mean"],
            "total_bid_value": ["mean"],
            "total_ask_value": ["mean"],
            "bid_vwap": ["mean"],
            "ask_vwap": ["mean"],
            "weighted_mid": ["mean", "last"],
        }
        present = {c: funcs for c, funcs in spec.items() if c in d.columns}
        if not present:
            continue
        out = grouped.agg(present)
        out = _flatten_columns(out, prefix)
        out[f"{prefix}_updates"] = grouped.size()
        parts.append(out)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def build_depth_obi_features(path: Path | None, source_depth: str, levels: tuple[int, ...]) -> pd.DataFrame:
    cols = ["event_ts", "local_ts", "update_id", "level", "bid_qty", "ask_qty"]
    df = _read(path, cols)
    if df.empty or not {"level", "bid_qty", "ask_qty"} <= set(df.columns):
        return pd.DataFrame()
    df["_ts"] = _to_second_index(df, ("event_ts", "local_ts"))
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["bid_qty"] = pd.to_numeric(df["bid_qty"], errors="coerce")
    df["ask_qty"] = pd.to_numeric(df["ask_qty"], errors="coerce")
    if "update_id" not in df.columns:
        df["update_id"] = np.arange(len(df), dtype=np.int64)
    df = df.dropna(subset=["_ts", "level", "bid_qty", "ask_qty", "update_id"])
    if df.empty:
        return pd.DataFrame()

    parts = []
    for n in levels:
        d = df[df["level"] <= n]
        if d.empty:
            continue
        g = d.groupby(["_ts", "update_id"], sort=False).agg(
            bid_qty=("bid_qty", "sum"),
            ask_qty=("ask_qty", "sum"),
        )
        g["obi"] = _safe_imbalance(g["bid_qty"], g["ask_qty"])
        by_sec = g.reset_index().groupby("_ts", sort=True)["obi"].agg(["mean", "std", "last", "count"])
        by_sec.columns = [
            f"obi_{source_depth}_l{n}_mean",
            f"obi_{source_depth}_l{n}_std",
            f"obi_{source_depth}_l{n}_last",
            f"obi_{source_depth}_l{n}_updates",
        ]
        parts.append(by_sec)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def build_trade_features(path: Path | None) -> pd.DataFrame:
    cols = ["trade_ts", "event_ts", "local_ts", "price", "qty", "quote_qty", "is_buyer_maker"]
    df = _read(path, cols)
    if df.empty:
        return pd.DataFrame()
    df["_ts"] = _to_second_index(df, ("trade_ts", "event_ts", "local_ts"))
    df = df.dropna(subset=["_ts"])
    for col in ("price", "qty", "quote_qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "qty" not in df.columns:
        return pd.DataFrame()
    if "quote_qty" not in df.columns and "price" in df.columns:
        df["quote_qty"] = df["qty"] * df["price"]
    maker = df.get("is_buyer_maker", pd.Series(False, index=df.index)).astype(bool)
    df["buy_qty"] = np.where(~maker, df["qty"], 0.0)
    df["sell_qty"] = np.where(maker, df["qty"], 0.0)
    df["buy_quote_qty"] = np.where(~maker, df["quote_qty"], 0.0)
    df["sell_quote_qty"] = np.where(maker, df["quote_qty"], 0.0)
    df["signed_qty"] = df["buy_qty"] - df["sell_qty"]
    df["signed_quote_qty"] = df["buy_quote_qty"] - df["sell_quote_qty"]

    grouped = df.groupby("_ts", sort=True)
    out = grouped.agg(
        trade_count=("qty", "size"),
        trade_qty=("qty", "sum"),
        trade_quote_qty=("quote_qty", "sum"),
        trade_buy_qty=("buy_qty", "sum"),
        trade_sell_qty=("sell_qty", "sum"),
        trade_buy_quote_qty=("buy_quote_qty", "sum"),
        trade_sell_quote_qty=("sell_quote_qty", "sum"),
        trade_signed_qty=("signed_qty", "sum"),
        trade_signed_quote_qty=("signed_quote_qty", "sum"),
        trade_avg_price=("price", "mean") if "price" in df.columns else ("qty", "mean"),
        trade_last_price=("price", "last") if "price" in df.columns else ("qty", "last"),
    )
    den = out["trade_buy_qty"] + out["trade_sell_qty"]
    out["trade_buy_pressure"] = out["trade_buy_qty"] / den.replace(0, np.nan)
    return out


def _source_newer_than_output(src: DailySources, out_file: Path) -> bool:
    if not out_file.exists():
        return True
    if not src.paths:
        return False
    newest_src = max(p.stat().st_mtime for p in src.paths)
    return newest_src > out_file.stat().st_mtime


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    state_cols = [
        c
        for c in out.columns
        if any(
            token in c
            for token in (
                "obi",
                "depth_imbalance",
                "spread_bps",
                "weighted_mid",
                "mid_close",
                "bid_vwap",
                "ask_vwap",
            )
        )
    ]
    trade_zero_cols = [c for c in out.columns if c.startswith("trade_") and c not in {"trade_avg_price", "trade_last_price", "trade_buy_pressure"}]

    for c in state_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").ffill(limit=10)
    for c in trade_zero_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    roll_base = [
        c
        for c in (
            "l1_obi_mean",
            "l1_obi_close",
            "obi_l5_l5_mean",
            "obi_l20_l5_mean",
            "obi_l20_l10_mean",
            "obi_l20_l20_mean",
            "l5_depth_imbalance_mean",
            "l20_depth_imbalance_mean",
            "l1_spread_bps_mean",
            "trade_buy_pressure",
        )
        if c in out.columns
    ]
    for col in roll_base:
        s = pd.to_numeric(out[col], errors="coerce")
        for w in ROLL_WINDOWS_SEC:
            r = s.rolling(f"{w}s", min_periods=max(1, min(3, w // 2)))
            out[f"{col}_{w}s_mean"] = r.mean()
            out[f"{col}_{w}s_std"] = r.std()
            out[f"{col}_{w}s_chg"] = s - s.shift(w)

    for col in [
        c
        for c in (
            "trade_signed_qty",
            "trade_signed_quote_qty",
            "trade_qty",
            "trade_quote_qty",
            "trade_buy_qty",
            "trade_sell_qty",
            "trade_buy_quote_qty",
            "trade_sell_quote_qty",
        )
        if c in out.columns
    ]:
        s = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        for w in ROLL_WINDOWS_SEC:
            out[f"{col}_{w}s_sum"] = s.rolling(f"{w}s", min_periods=1).sum()

    return out


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    mid_candidates = [
        "l20_weighted_mid_last",
        "l20_weighted_mid_mean",
        "l1_mid_close",
        "l1_mid_mean",
        "trade_last_price",
    ]
    mid_col = next((c for c in mid_candidates if c in out.columns), None)
    if mid_col is None:
        out["mid_price"] = np.nan
        return out
    out["mid_price"] = pd.to_numeric(out[mid_col], errors="coerce").ffill(limit=30)
    for h in LABEL_HORIZONS_SEC:
        future = out["mid_price"].shift(-h)
        out[f"future_mid_{h}s"] = future
        out[f"future_return_{h}s"] = future / out["mid_price"].replace(0, np.nan) - 1.0
        out[f"target_up_{h}s"] = (out[f"future_return_{h}s"] > 0).astype("Int8")
    return out


def build_one_day(src: DailySources) -> pd.DataFrame:
    frames = [
        build_l1_features(src.l1),
        build_metrics_features(src.metrics),
        build_depth_obi_features(src.l5, "l5", (5,)),
        build_depth_obi_features(src.l20, "l20", (5, 10, 20)),
        build_trade_features(src.trades),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    start = min(f.index.min() for f in frames)
    end = max(f.index.max() for f in frames)
    idx = pd.date_range(start.floor("s"), end.ceil("s"), freq="1s", tz="UTC")
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")].reindex(idx)
    out.index.name = "timestamp_dt"
    out = add_rolling_features(out)
    out = add_labels(out)
    out = out.reset_index()
    out["date"] = src.date_key
    return out.replace([np.inf, -np.inf], np.nan)


def write_daily_feature(
    project_root: Path,
    coin: str,
    src: DailySources,
    output_root: Path,
    *,
    force: bool,
) -> tuple[Path | None, dict]:
    out_dir = output_root / coin / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"microstructure_features_{src.date_key}.parquet"
    meta = {
        "coin": coin,
        "date": src.date_key,
        "output": str(out_file),
        "sources": [str(p) for p in src.paths],
        "rows": 0,
        "skipped": False,
    }
    if not force and not _source_newer_than_output(src, out_file):
        try:
            meta["rows"] = int(pq.ParquetFile(out_file).metadata.num_rows)
        except Exception:
            meta["rows"] = 0
        meta["skipped"] = True
        print(f"[SKIP] {coin} {src.date_key}: up to date")
        return out_file, meta

    df = build_one_day(src)
    if df.empty:
        meta["error"] = "no usable source rows"
        print(f"[WARN] {coin} {src.date_key}: no usable source rows")
        return None, meta
    df.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
    meta["rows"] = int(len(df))
    meta["columns"] = int(len(df.columns))
    print(f"[WRITE] {coin} {src.date_key}: {len(df):,} rows -> {out_file}")
    return out_file, meta


def combine_all_features(coin: str, output_root: Path) -> tuple[Path | None, int]:
    daily_dir = output_root / coin / "daily"
    files = sorted(daily_dir.glob("microstructure_features_*.parquet"))
    if not files:
        return None, 0
    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    if "timestamp_dt" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
        df = df.sort_values("timestamp_dt").reset_index(drop=True)
    out_file = output_root / coin / "microstructure_features_all.parquet"
    df.to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
    print(f"[WRITE] {coin}: {len(df):,} rows -> {out_file}")
    return out_file, int(len(df))


def write_signal_1h(coin: str, feature_root: Path, signal_root: Path, lookback_sec: int) -> Path | None:
    all_file = feature_root / coin / "microstructure_features_all.parquet"
    if not all_file.exists():
        return None
    df = pd.read_parquet(all_file)
    if df.empty or "timestamp_dt" not in df.columns:
        return None
    df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt")
    cutoff = df["timestamp_dt"].max() - pd.Timedelta(seconds=int(lookback_sec))
    win = df[df["timestamp_dt"] >= cutoff].copy()
    if win.empty:
        return None

    def avg(*cols: str) -> float | None:
        for c in cols:
            if c in win.columns:
                s = pd.to_numeric(win[c], errors="coerce").dropna()
                if not s.empty:
                    return float(s.mean())
        return None

    payload = {
        "coin": coin,
        "ts": time.time(),
        "lookback_sec": int(lookback_sec),
        "feature_rows": int(len(win)),
        "feature_start": str(win["timestamp_dt"].min()),
        "feature_end": str(win["timestamp_dt"].max()),
        "obi_avg_1h": avg("l1_obi_mean", "obi_l20_l20_mean", "l20_depth_imbalance_mean"),
        "obi_l10_avg_1h": avg("obi_l20_l10_mean"),
        "obi_l20_avg_1h": avg("obi_l20_l20_mean", "l20_depth_imbalance_mean"),
        "obi_l30_avg_1h": avg("obi_l20_l20_mean", "l20_depth_imbalance_mean"),
        "depth_imbalance_avg_1h": avg("l20_depth_imbalance_mean"),
        "buy_pressure_1h": avg("trade_buy_pressure"),
        "buy_volume_1h": avg("trade_buy_qty_60s_sum"),
        "sell_volume_1h": avg("trade_sell_qty_60s_sum") if "trade_sell_qty_60s_sum" in win.columns else None,
        "l1_rows_1h": int(pd.to_numeric(win.get("l1_tick_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "depth_rows_1h": int(pd.to_numeric(win.get("l20_updates", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "trade_rows_1h": int(pd.to_numeric(win.get("trade_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
    }
    out_dir = signal_root / coin
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "signal_averages_1h.parquet"
    pd.DataFrame([payload]).to_parquet(out_file, index=False, engine="pyarrow", compression="snappy")
    print(f"[WRITE] {coin}: 1h signal -> {out_file}")
    return out_file


def build_symbol(
    project_root: Path,
    coin: str,
    feature_root: Path,
    signal_root: Path,
    *,
    lookback_sec: int,
    limit_days: int | None,
    force: bool,
    no_all_file: bool,
) -> dict:
    sources = discover_daily_sources(project_root, coin, limit_days=limit_days)
    result = {
        "coin": coin,
        "daily_sources": len(sources),
        "daily": [],
        "all_file": None,
        "all_rows": 0,
        "signal_file": None,
    }
    if not sources:
        print(f"[WARN] {coin}: no daily parquet rollups found")
        return result

    for src in sources:
        _path, meta = write_daily_feature(project_root, coin, src, feature_root, force=force)
        result["daily"].append(meta)

    if not no_all_file:
        all_file, all_rows = combine_all_features(coin, feature_root)
        result["all_file"] = str(all_file) if all_file else None
        result["all_rows"] = all_rows
    signal_file = write_signal_1h(coin, feature_root, signal_root, lookback_sec)
    result["signal_file"] = str(signal_file) if signal_file else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build microstructure training features from parquet rollups.")
    parser.add_argument("--symbols", default="BTC,ADA", help="Comma-separated symbols. Default: BTC,ADA")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--output-root", default=None, help="Default: data/features")
    parser.add_argument("--signal-root", default=None, help="Default: data/parquet_signal")
    parser.add_argument("--lookback-sec", type=int, default=3600, help="Signal lookback window. Default: 3600")
    parser.add_argument("--limit-days", type=int, default=None, help="Only process the latest N daily rollup days.")
    parser.add_argument("--force", action="store_true", help="Recompute existing daily feature files.")
    parser.add_argument("--no-all-file", action="store_true", help="Do not rebuild microstructure_features_all.parquet.")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    feature_root = Path(args.output_root).resolve() if args.output_root else project_root / "data" / "features"
    signal_root = Path(args.signal_root).resolve() if args.signal_root else project_root / "data" / "parquet_signal"
    feature_root.mkdir(parents=True, exist_ok=True)
    signal_root.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.time(),
        "project_root": str(project_root),
        "feature_root": str(feature_root),
        "signal_root": str(signal_root),
        "symbols": [],
    }
    for raw in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        coin = normalize_coin(raw)
        report["symbols"].append(
            build_symbol(
                project_root,
                coin,
                feature_root,
                signal_root,
                lookback_sec=int(args.lookback_sec),
                limit_days=args.limit_days,
                force=bool(args.force),
                no_all_file=bool(args.no_all_file),
            )
        )

    summary_path = feature_root / "microstructure_features_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
