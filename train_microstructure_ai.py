#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.backtest import extract_trades, run_backtest
from src.config import Settings
from src.features import add_technical_features, build_labels
from src.modeling import infer_signals, save_models, train_models
from src.pipeline import _atomic_write_csv, _run_cost_stress_suite


ROOT = Path(__file__).resolve().parent


def _coin_to_symbol(coin: str) -> str:
    c = coin.upper().replace("USDT", "")
    return f"{c}USDT"


def _pandas_freq(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1D",
    }
    return mapping.get(interval, interval)


def _read_parquets(paths: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p, columns=columns))
            print(f"[READ] {p}")
        except Exception as exc:
            print(f"[WARN] skip {p}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _daily_files(coin: str, table: str, limit_days: int) -> list[Path]:
    daily = ROOT / "data" / "parquet_rollup" / coin.upper() / "daily"
    files = sorted(daily.glob(f"{table}_*.parquet"))
    return files[-limit_days:] if limit_days > 0 else files


def _timestamp_index_ms(df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = df.copy()
    x["_ts"] = pd.to_datetime(pd.to_numeric(x[col], errors="coerce"), unit="ms", utc=True, errors="coerce")
    x = x[x["_ts"].notna()].sort_values("_ts")
    return x.set_index("_ts")


def _aggregate_trades(coin: str, interval: str, limit_days: int) -> pd.DataFrame:
    freq = _pandas_freq(interval)
    paths = _daily_files(coin, "trades", limit_days)
    trades = _read_parquets(
        paths,
        columns=["trade_ts", "price", "qty", "quote_qty", "is_buyer_maker"],
    )
    if trades.empty:
        raise RuntimeError(f"No trade parquet files found for {coin}.")

    trades = _timestamp_index_ms(trades, "trade_ts")
    for col in ("price", "qty", "quote_qty"):
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)
    trades = trades.dropna(subset=["price", "qty", "quote_qty"])
    trades = trades[(trades["price"] > 0) & (trades["qty"] > 0) & (trades["quote_qty"] >= 0)]

    buy = trades["is_buyer_maker"] == False  # noqa: E712 - exchange convention: taker buy
    trades["buy_qty"] = np.where(buy, trades["qty"], 0.0)
    trades["sell_qty"] = np.where(buy, 0.0, trades["qty"])
    trades["buy_quote"] = np.where(buy, trades["quote_qty"], 0.0)

    bars = trades.resample(freq).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        quote_asset_volume=("quote_qty", "sum"),
        taker_buy_base=("buy_qty", "sum"),
        taker_buy_quote=("buy_quote", "sum"),
        sell_base=("sell_qty", "sum"),
        trade_count=("price", "size"),
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars[(bars["open"] > 0) & (bars["high"] > 0) & (bars["low"] > 0) & (bars["close"] > 0)]
    bars["micro_buy_pressure"] = bars["taker_buy_base"] / bars["volume"].replace(0, np.nan)
    bars["micro_sell_pressure"] = bars["sell_base"] / bars["volume"].replace(0, np.nan)
    bars["micro_avg_trade_quote"] = bars["quote_asset_volume"] / bars["trade_count"].replace(0, np.nan)
    bars["micro_vwap"] = bars["quote_asset_volume"] / bars["volume"].replace(0, np.nan)
    bars["symbol"] = _coin_to_symbol(coin)
    bars = bars.reset_index().rename(columns={"_ts": "timestamp"})
    return bars


def _aggregate_orderbook_metrics(coin: str, interval: str, limit_days: int) -> pd.DataFrame:
    freq = _pandas_freq(interval)
    paths = _daily_files(coin, "orderbook_metrics", limit_days)
    depth = _read_parquets(
        paths,
        columns=[
            "event_ts",
            "depth_type",
            "total_bid_qty",
            "total_ask_qty",
            "total_bid_value",
            "total_ask_value",
            "depth_imbalance",
            "bid_vwap",
            "ask_vwap",
            "weighted_mid",
        ],
    )
    if depth.empty:
        return pd.DataFrame()
    if "depth_type" in depth.columns:
        l20 = depth[depth["depth_type"].astype(str).str.lower().eq("l20")].copy()
        if not l20.empty:
            depth = l20

    depth = _timestamp_index_ms(depth, "event_ts")
    numeric_cols = [
        "total_bid_qty",
        "total_ask_qty",
        "total_bid_value",
        "total_ask_value",
        "depth_imbalance",
        "bid_vwap",
        "ask_vwap",
        "weighted_mid",
    ]
    for col in numeric_cols:
        depth[col] = pd.to_numeric(depth[col], errors="coerce")

    out = depth.resample(freq).agg(
        ob_total_bid_qty=("total_bid_qty", "mean"),
        ob_total_ask_qty=("total_ask_qty", "mean"),
        ob_total_bid_value=("total_bid_value", "mean"),
        ob_total_ask_value=("total_ask_value", "mean"),
        ob_depth_imbalance=("depth_imbalance", "mean"),
        ob_bid_vwap=("bid_vwap", "mean"),
        ob_ask_vwap=("ask_vwap", "mean"),
        ob_weighted_mid=("weighted_mid", "mean"),
        ob_updates=("depth_imbalance", "size"),
    )
    out["ob_qty_imbalance"] = (
        (out["ob_total_bid_qty"] - out["ob_total_ask_qty"])
        / (out["ob_total_bid_qty"] + out["ob_total_ask_qty"]).replace(0, np.nan)
    )
    out["ob_value_imbalance"] = (
        (out["ob_total_bid_value"] - out["ob_total_ask_value"])
        / (out["ob_total_bid_value"] + out["ob_total_ask_value"]).replace(0, np.nan)
    )
    return out.reset_index().rename(columns={"_ts": "timestamp"})


def _aggregate_l1(coin: str, interval: str, limit_days: int) -> pd.DataFrame:
    freq = _pandas_freq(interval)
    paths = _daily_files(coin, "orderbook_l1", limit_days)
    l1 = _read_parquets(paths, columns=["event_ts", "spread_bps", "mid_price", "obi"])
    if l1.empty:
        return pd.DataFrame()
    l1 = _timestamp_index_ms(l1, "event_ts")
    for col in ("spread_bps", "mid_price", "obi"):
        l1[col] = pd.to_numeric(l1[col], errors="coerce")
    out = l1.resample(freq).agg(
        l1_spread_bps=("spread_bps", "mean"),
        l1_mid_price=("mid_price", "mean"),
        l1_obi=("obi", "mean"),
        l1_updates=("obi", "size"),
    )
    return out.reset_index().rename(columns={"_ts": "timestamp"})


def _merge_latest_signal_parquet(df: pd.DataFrame, coin: str) -> pd.DataFrame:
    p = ROOT / "data" / "parquet_signal" / coin.upper() / "signal_averages_1h.parquet"
    if not p.exists() or df.empty:
        return df
    try:
        sig = pd.read_parquet(p)
    except Exception:
        return df
    if sig.empty:
        return df
    row = sig.iloc[-1].to_dict()
    out = df.copy()
    for src, dst in {
        "obi_avg_1h": "live_obi_avg_1h",
        "depth_imbalance_avg_1h": "live_depth_imbalance_avg_1h",
        "buy_pressure_1h": "live_buy_pressure_1h",
        "buy_volume_1h": "live_buy_volume_1h",
        "sell_volume_1h": "live_sell_volume_1h",
        "l1_rows_1h": "live_l1_rows_1h",
        "depth_rows_1h": "live_depth_rows_1h",
        "trade_rows_1h": "live_trade_rows_1h",
    }.items():
        out[dst] = pd.to_numeric(row.get(src), errors="coerce")
    return out


def build_microstructure_bars(coin: str, interval: str, limit_days: int, include_l1: bool) -> pd.DataFrame:
    bars = _aggregate_trades(coin, interval, limit_days)
    for extra in (
        _aggregate_orderbook_metrics(coin, interval, limit_days),
        _aggregate_l1(coin, interval, limit_days) if include_l1 else pd.DataFrame(),
    ):
        if not extra.empty:
            bars = bars.merge(extra, on="timestamp", how="left")

    bars = bars.sort_values("timestamp").reset_index(drop=True)
    numeric = bars.select_dtypes(include=[np.number]).columns
    bars[numeric] = bars[numeric].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    bars = _merge_latest_signal_parquet(bars, coin)
    return bars


def _write_outputs(
    inferred: pd.DataFrame,
    bt_curve: pd.DataFrame,
    train_metrics: dict,
    bt_report: dict,
    settings: Settings,
    source_meta: dict,
) -> dict:
    latest = bt_curve.iloc[-1]
    decision = {
        "timestamp": str(latest["timestamp"]),
        "price": float(latest["close"]),
        "signal": int(latest["signal"]),
        "suggested_leverage": float(latest["suggested_leverage"]),
        "max_safe_leverage": float(latest["max_safe_leverage"]),
        "p_long": float(latest["p_long"]),
        "p_short": float(latest["p_short"]),
        "trade_allowed": int(latest.get("trade_allowed", 0)),
        "trade_block_reason": str(latest.get("trade_block_reason", "")),
    }
    tag = f"{settings.symbol}_{settings.interval}"
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(bt_curve, settings.output_dir / f"backtest_curve_{tag}.csv", index=False)
    _atomic_write_csv(inferred, settings.output_dir / f"signals_with_features_{tag}.csv", index=False)
    _atomic_write_csv(extract_trades(bt_curve), settings.output_dir / f"trades_{tag}.csv", index=False, encoding="utf-8")

    payload = {
        "train_metrics": train_metrics,
        "backtest_report": bt_report,
        "latest_decision": decision,
        "source_meta": source_meta,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (settings.output_dir / f"report_{tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (settings.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _atomic_write_csv(bt_curve, settings.output_dir / "backtest_curve.csv", index=False)
    _atomic_write_csv(inferred, settings.output_dir / "signals_with_features.csv", index=False)
    _atomic_write_csv(extract_trades(bt_curve), settings.output_dir / "trades.csv", index=False, encoding="utf-8")
    return payload


def _calibrate_strategy_params(results: list[dict], max_leverage_cap: int) -> dict:
    reports = [r["backtest_report"] for r in results if isinstance(r.get("backtest_report"), dict)]
    if not reports:
        raise RuntimeError("No reports to calibrate strategy params.")
    avg_win = float(np.mean([float(r.get("win_rate", 0.0) or 0.0) for r in reports]))
    worst_dd = float(max(abs(float(r.get("max_drawdown", 0.0) or 0.0)) for r in reports))
    pfs = [float(r.get("profit_factor", 0.0) or 0.0) for r in reports if r.get("profit_factor") is not None]
    avg_pf = float(np.mean(pfs)) if pfs else 0.0

    signal_5m = 0.60
    if avg_win >= 0.54 and avg_pf >= 1.10 and worst_dd <= 0.25:
        signal_5m = 0.57
        max_lev = min(max_leverage_cap, 10)
        drawdown_stop = 0.32
    elif avg_win <= 0.49 or avg_pf < 1.0 or worst_dd >= 0.35:
        signal_5m = 0.64
        max_lev = min(max_leverage_cap, 5)
        drawdown_stop = 0.24
    else:
        max_lev = min(max_leverage_cap, 8)
        drawdown_stop = 0.28

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": "microstructure rollup training",
        "long_threshold": 0.003,
        "short_threshold": -0.003,
        "drawdown_stop": float(np.clip(drawdown_stop, 0.20, 0.40)),
        "max_leverage": int(max(2, min(max_leverage_cap, max_lev))),
        "interval_signal_thresholds": {
            "5m": float(np.clip(signal_5m, 0.52, 0.70)),
            "15m": 0.57,
            "30m": 0.54,
            "1h": 0.50,
            "1d": 0.44,
        },
        "stats": {
            "avg_win_rate": avg_win,
            "worst_max_drawdown_abs": worst_dd,
            "avg_profit_factor": avg_pf,
            "trained_reports": len(reports),
        },
    }
    out = ROOT / "outputs" / "strategy_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def train_one(
    coin: str,
    interval: str,
    limit_days: int,
    horizon_hours: int,
    max_leverage: int,
    include_l1: bool,
    min_train_rows: int,
) -> dict:
    coin = coin.upper().replace("USDT", "")
    symbol = _coin_to_symbol(coin)
    print(f"\n=== Training {symbol} {interval} from harvester parquet ===")
    bars = build_microstructure_bars(coin, interval, limit_days, include_l1)
    if len(bars) < min_train_rows:
        raise RuntimeError(f"{symbol} {interval}: not enough bars ({len(bars)} < {min_train_rows}).")

    settings = Settings(
        symbol=symbol,
        interval=interval,
        market_type="futures",
        data_dir=ROOT / "data",
        output_dir=ROOT / "outputs",
        model_dir=ROOT / "models",
        min_train_rows=min_train_rows,
        future_horizon_hours=horizon_hours,
        long_threshold=0.003,
        short_threshold=-0.003,
        max_leverage=max_leverage,
        train_device=os.getenv("TRAIN_DEVICE", "cpu"),
        max_train_rows=0,
        promote_min_trades=1,
    )

    feat = add_technical_features(bars)
    from src.data_sources import interval_to_seconds

    horizon_bars = max(1, int(round((horizon_hours * 3600) / interval_to_seconds(interval))))
    labeled_full = build_labels(
        feat,
        horizon_bars=horizon_bars,
        long_th=settings.long_threshold,
        short_th=settings.short_threshold,
    )
    labeled_train = labeled_full[labeled_full["future_ret"].notna()].reset_index(drop=True)
    if len(labeled_train) < min_train_rows:
        raise RuntimeError(f"{symbol} {interval}: not enough labeled rows ({len(labeled_train)} < {min_train_rows}).")

    models, train_metrics = train_models(labeled_train, settings)
    tag = f"{settings.symbol}_{settings.interval}"
    staged = settings.model_dir / "_staged_microstructure" / tag / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    production = settings.model_dir / tag
    save_models(models, staged)
    if production.exists():
        shutil.rmtree(production)
    shutil.copytree(staged, production)

    inferred = infer_signals(labeled_full, models, settings)
    bt_curve, bt_report = run_backtest(inferred, settings, interval=interval)
    bt_report["cost_stress_tests"] = _run_cost_stress_suite(inferred, settings, interval)
    source_meta = {
        "source": "harvester parquet rollup",
        "coin": coin,
        "symbol": symbol,
        "interval": interval,
        "bars": int(len(bars)),
        "labeled_rows": int(len(labeled_train)),
        "start_utc": str(bars["timestamp"].iloc[0]),
        "end_utc": str(bars["timestamp"].iloc[-1]),
        "model_dir": str(production),
        "staged_model_dir": str(staged),
        "include_l1": bool(include_l1),
        "limit_days": int(limit_days),
    }
    out = _write_outputs(inferred, bt_curve, train_metrics, bt_report, settings, source_meta)
    print(
        "[DONE]",
        tag,
        "signal=",
        out["latest_decision"]["signal"],
        "lev=",
        round(out["latest_decision"]["suggested_leverage"], 2),
        "trades=",
        bt_report.get("trades"),
        "return=",
        round(float(bt_report.get("total_return", 0.0) or 0.0), 4),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train tradable long/short/flat leverage AI from harvester parquet.")
    parser.add_argument("--coins", default="BTC,ADA", help="Comma-separated coins, e.g. BTC,ADA")
    parser.add_argument("--interval", default="5m", help="Trading interval. Recommended with current data: 5m or 15m")
    parser.add_argument("--limit-days", type=int, default=3, help="Use the latest N daily rollup files per table.")
    parser.add_argument("--horizon-hours", type=int, default=4, help="Future return horizon used for labels.")
    parser.add_argument("--max-leverage", type=int, default=8, help="Training and inference leverage cap.")
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--include-l1", action="store_true", help="Also read large orderbook_l1 parquet files for OBI/spread features.")
    args = parser.parse_args(argv)

    results: list[dict] = []
    for raw in [c.strip() for c in args.coins.split(",") if c.strip()]:
        results.append(
            train_one(
                coin=raw,
                interval=args.interval,
                limit_days=args.limit_days,
                horizon_hours=args.horizon_hours,
                max_leverage=args.max_leverage,
                include_l1=bool(args.include_l1),
                min_train_rows=int(args.min_train_rows),
            )
        )

    params = _calibrate_strategy_params(results, max_leverage_cap=int(args.max_leverage))
    print("\n=== Strategy Params ===")
    print(json.dumps(params, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
