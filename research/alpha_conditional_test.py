#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _latest_daily_file(project_root: Path, coin: str, table: str) -> Path:
    coin = coin.upper()
    layout_a = project_root / "data" / "parquet" / coin / "daily" / table
    files = sorted(layout_a.glob(f"date=*/{table}_*.parquet"))
    if files:
        return files[-1]
    layout_b = project_root / "data" / "parquet_rollup" / coin / "daily"
    files = sorted(layout_b.glob(f"{table}_*.parquet"))
    if files:
        return files[-1]
    raise FileNotFoundError(f"no parquet files found for {coin}/{table} under {layout_a} or {layout_b}")


def _pick_ts_col(df: pd.DataFrame) -> str:
    for col in ("second_ts", "event_ts", "trade_ts", "local_ts", "timestamp"):
        if col in df.columns:
            return col
    raise KeyError("no timestamp column found")


def _pick_mid_col(df_l1: pd.DataFrame, df_metrics: pd.DataFrame) -> str:
    for col in ("mid_close", "mid_price", "weighted_mid"):
        if col in df_l1.columns:
            return col
    for col in ("weighted_mid", "mid_price"):
        if col in df_metrics.columns:
            return col
    raise KeyError("no mid-price-like column found")


def _load_joined_frame(l1_path: Path, metrics_path: Path) -> pd.DataFrame:
    l1 = pd.read_parquet(l1_path).copy()
    mx = pd.read_parquet(metrics_path).copy()

    ts_l1 = _pick_ts_col(l1)
    ts_mx = _pick_ts_col(mx)
    l1["ts"] = pd.to_numeric(l1[ts_l1], errors="coerce")
    mx["ts"] = pd.to_numeric(mx[ts_mx], errors="coerce")
    l1 = l1[l1["ts"].notna()].sort_values("ts").reset_index(drop=True)
    mx = mx[mx["ts"].notna()].sort_values("ts").reset_index(drop=True)

    mid_col = _pick_mid_col(l1, mx)
    l1["mid"] = pd.to_numeric(l1.get(mid_col), errors="coerce")

    obi_col = "obi_mean" if "obi_mean" in l1.columns else ("obi" if "obi" in l1.columns else None)
    if obi_col is None:
        raise KeyError("no obi/obi_mean in l1 parquet")
    l1["obi"] = pd.to_numeric(l1[obi_col], errors="coerce")

    if "depth_type" in mx.columns:
        mx = mx[mx["depth_type"].astype(str).str.lower() == "l20"].copy()
    keep = ["ts"]
    for c in ("depth_imbalance", "weighted_mid", "bid_vwap", "ask_vwap", "total_bid_qty", "total_ask_qty"):
        if c in mx.columns:
            keep.append(c)
    mx = mx[keep].copy().drop_duplicates(subset=["ts"], keep="last")

    out = pd.merge_asof(
        l1[["ts", "mid", "obi"]].sort_values("ts"),
        mx.sort_values("ts"),
        on="ts",
        direction="backward",
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["ts", "mid", "obi"]).reset_index(drop=True)
    return out


def _simple_t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 0.0
    std = float(np.std(x, ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(x) / (std / np.sqrt(n)))


def run_test(df: pd.DataFrame, horizon_sec: int, n_bins: int) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["mid_fwd"] = out["mid"].shift(-horizon_sec)
    out["fwd_ret"] = (out["mid_fwd"] / out["mid"]) - 1.0
    out = out.dropna(subset=["fwd_ret", "obi"]).reset_index(drop=True)

    out["obi_bin"] = pd.qcut(out["obi"], q=n_bins, labels=False, duplicates="drop")
    g = out.groupby("obi_bin", dropna=True)
    bucket = g["fwd_ret"].agg(["count", "mean", "std"]).reset_index()
    bucket["t_stat"] = bucket.apply(
        lambda r: 0.0 if (r["count"] < 2 or pd.isna(r["std"]) or float(r["std"]) == 0.0) else float(r["mean"] / (r["std"] / np.sqrt(r["count"]))),
        axis=1,
    )
    bucket["mean_bps"] = bucket["mean"] * 10_000.0

    corr = float(np.corrcoef(out["obi"].to_numpy(), out["fwd_ret"].to_numpy())[0, 1]) if len(out) > 2 else 0.0
    slope = float(np.polyfit(out["obi"].to_numpy(), out["fwd_ret"].to_numpy(), 1)[0]) if len(out) > 2 else 0.0
    spread = float(bucket["mean_bps"].iloc[-1] - bucket["mean_bps"].iloc[0]) if len(bucket) >= 2 else 0.0
    hit = float((out["fwd_ret"] * np.sign(out["obi"]) > 0).mean()) if len(out) else 0.0
    t_all = _simple_t_stat(out["fwd_ret"].to_numpy() * np.sign(out["obi"]).to_numpy())

    summary = {
        "rows": int(len(out)),
        "horizon_sec": int(horizon_sec),
        "bins": int(n_bins),
        "corr_obi_fwd_ret": corr,
        "slope_obi_to_fwd_ret": slope,
        "top_minus_bottom_mean_bps": spread,
        "sign_hit_rate": hit,
        "signal_t_stat": t_all,
    }
    return bucket, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Conditional expectation test for OBI alpha.")
    p.add_argument("--coin", default="BTC", help="BTC/ADA...")
    p.add_argument("--horizon-sec", type=int, default=10)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--project-root", default=r"C:\Users\brian\trading")
    p.add_argument("--l1-path", default=None, help="Optional explicit orderbook_l1 parquet path")
    p.add_argument("--metrics-path", default=None, help="Optional explicit orderbook_metrics parquet path")
    p.add_argument("--out-dir", default=r"C:\Users\brian\trading\research\results")
    args = p.parse_args()

    root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_path = Path(args.l1_path) if args.l1_path else _latest_daily_file(root, args.coin, "orderbook_l1")
    mx_path = Path(args.metrics_path) if args.metrics_path else _latest_daily_file(root, args.coin, "orderbook_metrics")
    df = _load_joined_frame(l1_path, mx_path)

    bucket, summary = run_test(df, horizon_sec=max(1, int(args.horizon_sec)), n_bins=max(3, int(args.bins)))
    tag = f"{args.coin.upper()}_h{int(args.horizon_sec)}s"
    bucket_path = out_dir / f"alpha_conditional_{tag}.csv"
    summary_path = out_dir / f"alpha_conditional_{tag}.json"
    bucket.to_csv(bucket_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] bucket => {bucket_path}")
    print(f"[OK] summary => {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
