from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_l20_snapshots(daily_dir: Path, days: int) -> pd.DataFrame:
    files = sorted(daily_dir.glob("orderbook_l20_*.parquet"))[-days:]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    required = {"event_ts", "update_id", "level", "bid_price", "ask_price", "bid_qty", "ask_qty"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    for c in ["event_ts", "update_id", "level", "bid_price", "ask_price", "bid_qty", "ask_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["event_ts", "update_id", "level"]).copy()
    return df


def _build_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for uid, g in df.groupby("update_id", sort=False):
        g = g.sort_values("level")
        ev = float(g["event_ts"].iloc[-1])
        lv1 = g[g["level"] == 1]
        if lv1.empty:
            continue
        bid1 = float(lv1["bid_price"].iloc[0])
        ask1 = float(lv1["ask_price"].iloc[0])
        mid = (bid1 + ask1) / 2.0 if np.isfinite(bid1) and np.isfinite(ask1) else np.nan
        if not np.isfinite(mid) or mid <= 0:
            continue

        def obi_n(n: int) -> float:
            d = g[g["level"] <= n]
            b = pd.to_numeric(d["bid_qty"], errors="coerce").fillna(0.0).sum()
            a = pd.to_numeric(d["ask_qty"], errors="coerce").fillna(0.0).sum()
            den = b + a
            return float((b - a) / den) if den > 0 else np.nan

        o10 = obi_n(10)
        o20 = obi_n(20)
        rows.append({"event_ts": ev, "update_id": uid, "mid": mid, "obi10": o10, "obi20": o20, "obi30": o20})
    out = pd.DataFrame(rows).sort_values("event_ts").reset_index(drop=True)
    return out


def _attach_future_return(snap: pd.DataFrame, horizon_sec: int) -> pd.DataFrame:
    if snap.empty:
        return snap
    left = snap[["event_ts", "mid"]].copy()
    left["target_ts"] = left["event_ts"] + horizon_sec * 1000
    right = snap[["event_ts", "mid"]].copy().rename(columns={"event_ts": "future_event_ts", "mid": "future_mid"})
    m = pd.merge_asof(
        left.sort_values("target_ts"),
        right.sort_values("future_event_ts"),
        left_on="target_ts",
        right_on="future_event_ts",
        direction="forward",
    )
    snap = snap.copy()
    snap["future_mid"] = m["future_mid"].to_numpy()
    snap["future_ret"] = (snap["future_mid"] / snap["mid"]) - 1.0
    return snap


def _metric_table(snap: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = snap[[metric, "future_ret"]].dropna().copy()
    if len(d) < 30:
        return pd.DataFrame()
    d["bucket"] = pd.qcut(d[metric], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    g = d.groupby("bucket", observed=True)
    out = g["future_ret"].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "mean_ret"})
    out["win_rate"] = g["future_ret"].apply(lambda x: float((x > 0).mean()))
    out = out.reset_index()
    out["metric"] = metric
    return out[["metric", "bucket", "n", "mean_ret", "win_rate"]]


def build_report(root: Path, coin: str, days: int, horizon_sec: int) -> tuple[dict, pd.DataFrame]:
    daily_dir = root / "data" / "parquet_rollup" / coin / "daily"
    raw = _load_l20_snapshots(daily_dir, days=days)
    snap = _build_snapshot(raw)
    snap = _attach_future_return(snap, horizon_sec=horizon_sec)
    tables = []
    for m in ["obi10", "obi20", "obi30"]:
        t = _metric_table(snap, m)
        if not t.empty:
            tables.append(t)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["metric", "bucket", "n", "mean_ret", "win_rate"])
    summary = {
        "coin": coin,
        "rows_l20": int(len(raw)),
        "snapshots": int(len(snap)),
        "usable_rows": int(snap["future_ret"].notna().sum()) if "future_ret" in snap.columns else 0,
        "horizon_sec": int(horizon_sec),
        "days": int(days),
    }
    return summary, table


def main() -> int:
    p = argparse.ArgumentParser(description="Generate daily OBI conditional expectation report.")
    p.add_argument("--symbols", default="BTC,ADA")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--horizon-sec", type=int, default=5)
    p.add_argument("--project-root", default=str(_project_root()))
    args = p.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    all_rows = []
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_sec": int(args.horizon_sec),
        "days": int(args.days),
        "symbols": [],
    }
    for raw in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        coin = raw.upper().replace("USDT", "")
        summary, table = build_report(root, coin, days=int(args.days), horizon_sec=int(args.horizon_sec))
        payload["symbols"].append(summary)
        if not table.empty:
            table.insert(0, "coin", coin)
            all_rows.append(table)

    out_csv = out_dir / f"obi_edge_report_{run_date}.csv"
    out_json = out_dir / f"obi_edge_report_{run_date}.json"
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=["coin", "metric", "bucket", "n", "mean_ret", "win_rate"]).to_csv(out_csv, index=False, encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WRITE] {out_csv}")
    print(f"[WRITE] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

