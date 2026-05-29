#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report


BUNDLE_PATH = Path(r"C:\Users\brian\klinetraning\data_kline\training_arena_multiscale_bundle.parquet")
DL_EXPORT_PATH = Path(r"G:\我的雲端硬碟\klinetraning\exports\dl_predictions.parquet")
RF_MODEL_PATH = Path(r"C:\Users\brian\trading\experiments\rf_from_dl\rf_model.pkl")
FEAT_COLS_PATH = Path(r"C:\Users\brian\trading\experiments\rf_from_dl\feature_cols.json")
OUT_DIR = Path(r"C:\Users\brian\trading\experiments\rf_from_dl")
OUT_PATH = OUT_DIR / "oos_30d_report.json"


def _load_frame(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p).copy()
    ts_col = "timestamp_dt" if "timestamp_dt" in df.columns else "timestamp"
    df["timestamp_dt"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    return df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)


def _simulate_excess(y_pred: np.ndarray, returns: np.ndarray) -> dict[str, float]:
    # binary y_pred: 1=long, 0=short
    sig = np.where(y_pred.astype(int) == 1, 1.0, -1.0)
    cap = 10000.0
    flat = 10000.0
    fee_rate = (2.0 + 1.0) / 10000.0
    prev = 0.0
    trades = 0
    for r, s in zip(returns, sig):
        if s != prev:
            trades += 1
            cap -= cap * fee_rate * abs(s - prev)
        cap += cap * float(r) * float(s)
        prev = s
    return {
        "test_capital": float(cap),
        "test_excess_vs_flat": float(cap - flat),
        "test_trades": int(trades),
    }


def main() -> int:
    bundle = _load_frame(BUNDLE_PATH)
    dl = _load_frame(DL_EXPORT_PATH)

    merged = pd.merge_asof(
        bundle.sort_values("timestamp_dt"),
        dl.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged["future_return"] = pd.to_numeric(merged["future_return"], errors="coerce")
    merged = merged.dropna(subset=["future_return"]).reset_index(drop=True)
    merged["target"] = (merged["future_return"] > 0).astype(int)

    with RF_MODEL_PATH.open("rb") as f:
        clf = pickle.load(f)
    feat_cols = json.loads(FEAT_COLS_PATH.read_text(encoding="utf-8"))

    # strict out-of-time window: last 30 calendar days
    end_ts = merged["timestamp_dt"].max()
    start_ts = end_ts - pd.Timedelta(days=30)
    oos = merged[merged["timestamp_dt"] >= start_ts].copy()
    if oos.empty:
        raise RuntimeError("OOS window is empty")

    x = oos[feat_cols].fillna(0.0).to_numpy(dtype=float)
    y = oos["target"].to_numpy(dtype=int)
    ret = oos["future_return"].to_numpy(dtype=float)

    y_pred = clf.predict(x).astype(int)
    bacc = float(balanced_accuracy_score(y, y_pred))
    cls = classification_report(y, y_pred, output_dict=True, zero_division=0)
    sim = _simulate_excess(y_pred, ret)

    payload = {
        "window_start_utc": str(start_ts),
        "window_end_utc": str(end_ts),
        "rows_oos": int(len(oos)),
        "balanced_accuracy": bacc,
        "classification_report": cls,
        **sim,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

