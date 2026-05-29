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


def _load_frame(project_root: Path, coin: str, l1_path: Path | None, mx_path: Path | None) -> pd.DataFrame:
    lp = l1_path or _latest_daily_file(project_root, coin, "orderbook_l1")
    mp = mx_path or _latest_daily_file(project_root, coin, "orderbook_metrics")
    l1 = pd.read_parquet(lp).copy()
    mx = pd.read_parquet(mp).copy()

    ts_l1 = _pick_ts_col(l1)
    l1["ts"] = pd.to_numeric(l1[ts_l1], errors="coerce")
    l1 = l1[l1["ts"].notna()].sort_values("ts").reset_index(drop=True)

    if "mid_close" in l1.columns:
        l1["mid"] = pd.to_numeric(l1["mid_close"], errors="coerce")
    elif "mid_price" in l1.columns:
        l1["mid"] = pd.to_numeric(l1["mid_price"], errors="coerce")
    else:
        raise KeyError("orderbook_l1 missing mid_close/mid_price")
    obi_col = "obi_mean" if "obi_mean" in l1.columns else ("obi" if "obi" in l1.columns else None)
    if obi_col is None:
        raise KeyError("orderbook_l1 missing obi")
    l1["obi"] = pd.to_numeric(l1[obi_col], errors="coerce")
    l1["spread_bps"] = pd.to_numeric(l1.get("spread_bps_mean", l1.get("spread_bps")), errors="coerce")
    l1["bid_qty"] = pd.to_numeric(l1.get("bid_qty_mean", l1.get("bid_qty")), errors="coerce")
    l1["ask_qty"] = pd.to_numeric(l1.get("ask_qty_mean", l1.get("ask_qty")), errors="coerce")

    ts_mx = _pick_ts_col(mx)
    mx["ts"] = pd.to_numeric(mx[ts_mx], errors="coerce")
    if "depth_type" in mx.columns:
        mx = mx[mx["depth_type"].astype(str).str.lower() == "l20"].copy()
    keep = ["ts"]
    for c in ("depth_imbalance", "total_bid_qty", "total_ask_qty"):
        if c in mx.columns:
            keep.append(c)
    mx = mx[keep].drop_duplicates(subset=["ts"], keep="last").sort_values("ts")

    out = pd.merge_asof(
        l1[["ts", "mid", "obi", "spread_bps", "bid_qty", "ask_qty"]].sort_values("ts"),
        mx,
        on="ts",
        direction="backward",
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["mid", "obi"]).reset_index(drop=True)
    return out


def _fill_probability(spread_bps: float, queue_imb: float, base_fill: float) -> float:
    spread_penalty = max(0.0, min(0.9, (spread_bps - 1.0) * 0.06))
    qi_bonus = max(-0.2, min(0.2, queue_imb * 0.25))
    p = base_fill - spread_penalty + qi_bonus
    return max(0.0, min(1.0, p))


def run_backtest(
    df: pd.DataFrame,
    horizon_sec: int,
    latency_sec: int,
    fee_bps: float,
    slippage_bps: float,
    long_th: float,
    short_th: float,
    base_fill: float,
) -> tuple[pd.DataFrame, dict]:
    x = df.copy()
    x["queue_imb"] = np.where(
        (x["bid_qty"].fillna(0) + x["ask_qty"].fillna(0)) > 0,
        (x["bid_qty"].fillna(0) - x["ask_qty"].fillna(0)) / (x["bid_qty"].fillna(0) + x["ask_qty"].fillna(0)),
        0.0,
    )
    x["signal_raw"] = np.where(x["obi"] >= long_th, 1, np.where(x["obi"] <= short_th, -1, 0))
    x["signal"] = x["signal_raw"].shift(latency_sec).fillna(0).astype(int)
    x["fwd_mid"] = x["mid"].shift(-horizon_sec)
    x["fwd_ret"] = (x["fwd_mid"] / x["mid"]) - 1.0
    x = x.dropna(subset=["fwd_ret"]).reset_index(drop=True)

    # stochastic fill model, deterministic via fixed seed
    rng = np.random.default_rng(42)
    fill_prob = [
        _fill_probability(float(sb if np.isfinite(sb) else 2.0), float(qi if np.isfinite(qi) else 0.0), base_fill)
        for sb, qi in zip(x["spread_bps"].fillna(2.0), x["queue_imb"].fillna(0.0))
    ]
    x["fill_prob"] = fill_prob
    x["filled"] = (rng.random(len(x)) < x["fill_prob"]).astype(int)
    x["exec_signal"] = x["signal"] * x["filled"]

    turnover = (x["exec_signal"] - x["exec_signal"].shift(1).fillna(0)).abs()
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    x["gross_ret"] = x["exec_signal"] * x["fwd_ret"]
    x["cost_ret"] = turnover * cost_rate
    x["net_ret"] = x["gross_ret"] - x["cost_ret"]
    x["equity"] = (1.0 + x["net_ret"].fillna(0.0)).cumprod() * 10_000.0

    trades = int((turnover > 0).sum())
    filled_ratio = float(x["filled"].mean()) if len(x) else 0.0
    net = x["net_ret"].to_numpy(dtype=float)
    pnl = float(x["equity"].iloc[-1] - 10_000.0) if len(x) else 0.0
    max_dd = float((x["equity"] / x["equity"].cummax() - 1.0).min()) if len(x) else 0.0
    hit = float((x.loc[x["exec_signal"] != 0, "gross_ret"] > 0).mean()) if (x["exec_signal"] != 0).any() else 0.0
    sharpe = float(np.mean(net) / (np.std(net, ddof=1) + 1e-12) * np.sqrt(365 * 24 * 3600 / max(1, horizon_sec))) if len(x) > 2 else 0.0

    summary = {
        "rows": int(len(x)),
        "horizon_sec": int(horizon_sec),
        "latency_sec": int(latency_sec),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "long_th": float(long_th),
        "short_th": float(short_th),
        "base_fill": float(base_fill),
        "trades": trades,
        "filled_ratio": filled_ratio,
        "final_capital": float(x["equity"].iloc[-1]) if len(x) else 10_000.0,
        "net_profit": pnl,
        "max_drawdown": max_dd,
        "hit_rate": hit,
        "annualized_sharpe_proxy": sharpe,
    }
    return x, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Execution-aware backtest for OBI signal.")
    p.add_argument("--coin", default="BTC")
    p.add_argument("--project-root", default=r"C:\Users\brian\trading")
    p.add_argument("--l1-path", default=None)
    p.add_argument("--metrics-path", default=None)
    p.add_argument("--horizon-sec", type=int, default=10)
    p.add_argument("--latency-sec", type=int, default=1)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=8.0)
    p.add_argument("--long-th", type=float, default=0.15)
    p.add_argument("--short-th", type=float, default=-0.15)
    p.add_argument("--base-fill", type=float, default=0.85)
    p.add_argument("--max-rows", type=int, default=2_000_000, help="Use tail N rows for faster research iteration.")
    p.add_argument("--out-dir", default=r"C:\Users\brian\trading\research\results")
    args = p.parse_args()

    root = Path(args.project_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    l1_path = Path(args.l1_path) if args.l1_path else None
    mx_path = Path(args.metrics_path) if args.metrics_path else None
    df = _load_frame(root, args.coin.upper(), l1_path, mx_path)
    if int(args.max_rows) > 0 and len(df) > int(args.max_rows):
        df = df.tail(int(args.max_rows)).reset_index(drop=True)

    curve, summary = run_backtest(
        df=df,
        horizon_sec=max(1, int(args.horizon_sec)),
        latency_sec=max(0, int(args.latency_sec)),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        long_th=float(args.long_th),
        short_th=float(args.short_th),
        base_fill=float(args.base_fill),
    )

    tag = f"{args.coin.upper()}_h{int(args.horizon_sec)}s_lag{int(args.latency_sec)}s"
    curve_path = out_dir / f"execution_aware_curve_{tag}.csv"
    summary_path = out_dir / f"execution_aware_summary_{tag}.json"
    curve.to_csv(curve_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] curve => {curve_path}")
    print(f"[OK] summary => {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
