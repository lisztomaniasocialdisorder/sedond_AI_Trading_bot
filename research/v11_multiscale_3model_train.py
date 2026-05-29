#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score


DATA_ROOT = Path(r"C:\Users\brian\klinetraning\data_kline")
OUT_DIR = Path(r"C:\Users\brian\trading\experiments\v11_multiscale")
MODELS_DIR = OUT_DIR / "models"

P_DAILY = DATA_ROOT / "BTCUSDT_1d_3000_features.parquet"
P_HOURLY = DATA_ROOT / "BTCUSDT_1h_72000_features.parquet"
P_30M = DATA_ROOT / "BTCUSDT_30m_144000_features.parquet"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

FEE_BPS = 2.0
SLIP_BPS = 1.0
INITIAL_CAPITAL = 10_000.0


@dataclass
class Split:
    train_end_ts: pd.Timestamp
    valid_end_ts: pd.Timestamp


def _to_ts(df: pd.DataFrame) -> pd.Series:
    if "timestamp_dt" in df.columns:
        ts = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "open_time" in df.columns:
        ts = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True, errors="coerce")
    else:
        ts = pd.Series(pd.NaT, index=df.index)
    return ts


def _load_scale(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_parquet(path).copy()
    df["timestamp_dt"] = _to_ts(df)
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    if "target_up" in df.columns:
        y = pd.to_numeric(df["target_up"], errors="coerce")
        y = (y > 0).astype(int)
    elif "future_return" in df.columns:
        y = (pd.to_numeric(df["future_return"], errors="coerce") > 0).astype(int)
    elif "next_return" in df.columns:
        y = (pd.to_numeric(df["next_return"], errors="coerce") > 0).astype(int)
    else:
        raise ValueError(f"{name} has no target_up/future_return/next_return")
    df["target"] = y
    return df


def _time_split(df: pd.DataFrame) -> Split:
    n = len(df)
    if n < 1000:
        raise ValueError(f"not enough rows for split: {n}")
    train_end_idx = int(n * TRAIN_RATIO)
    valid_end_idx = int(n * (TRAIN_RATIO + VALID_RATIO))
    train_end_idx = max(1, min(train_end_idx, n - 2))
    valid_end_idx = max(train_end_idx + 1, min(valid_end_idx, n - 1))
    return Split(
        train_end_ts=pd.Timestamp(df.iloc[train_end_idx]["timestamp_dt"]),
        valid_end_ts=pd.Timestamp(df.iloc[valid_end_idx]["timestamp_dt"]),
    )


def _partition(df: pd.DataFrame, split: Split) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["timestamp_dt"] <= split.train_end_ts].copy()
    va = df[(df["timestamp_dt"] > split.train_end_ts) & (df["timestamp_dt"] <= split.valid_end_ts)].copy()
    te = df[df["timestamp_dt"] > split.valid_end_ts].copy()
    if tr.empty or va.empty or te.empty:
        raise ValueError("empty train/valid/test partition")
    return tr, va, te


def _feature_cols(df: pd.DataFrame) -> list[str]:
    leakage_tokens = ("target", "future_return", "next_return", "label", "_ret")
    drop = {"timestamp", "timestamp_dt", "open_time", "close_time"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if any(tok in c.lower() for tok in leakage_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("no numeric feature columns")
    return cols


def _simulate_excess(returns: np.ndarray, p_up: np.ndarray) -> dict:
    cap = float(INITIAL_CAPITAL)
    flat = float(INITIAL_CAPITAL)
    fee_rate = (FEE_BPS + SLIP_BPS) / 10000.0
    prev_sig = 0.0
    trades = 0

    sig = np.zeros(len(p_up), dtype=np.float32)
    sig[p_up >= 0.55] = 1.0
    sig[p_up <= 0.45] = -1.0

    for r, s in zip(returns, sig):
        if s != prev_sig:
            trades += 1
            cap -= cap * fee_rate * abs(float(s - prev_sig))
        cap += cap * float(r) * float(s)
        prev_sig = s

    return {
        "capital": float(cap),
        "flat_capital": float(flat),
        "excess_vs_flat": float(cap - flat),
        "trades": int(trades),
    }


def _fit_model(x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_depth=6,
        min_samples_leaf=60,
        l2_regularization=0.05,
        random_state=42,
    )
    clf.fit(x, y)
    return clf


def _eval_block(name: str, clf, df: pd.DataFrame, feat_cols: list[str]) -> dict:
    x = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=int)
    p = clf.predict_proba(x)[:, 1]
    pred = (p >= 0.5).astype(int)
    ret_col = "future_return" if "future_return" in df.columns else "next_return"
    returns = pd.to_numeric(df.get(ret_col), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    sim = _simulate_excess(returns, p)
    return {
        "name": name,
        "rows": int(len(df)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "excess_vs_flat": sim["excess_vs_flat"],
        "capital": sim["capital"],
        "trades": sim["trades"],
        "p_up_mean": float(np.mean(p)),
    }


def _merge_probs(base_30m: pd.DataFrame, daily_df: pd.DataFrame, hourly_df: pd.DataFrame, p_day: np.ndarray, p_hour: np.ndarray) -> pd.DataFrame:
    day = daily_df[["timestamp_dt"]].copy()
    day["p_day_up"] = p_day
    hr = hourly_df[["timestamp_dt"]].copy()
    hr["p_hour_up"] = p_hour

    out = pd.merge_asof(
        base_30m.sort_values("timestamp_dt"),
        day.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    out = pd.merge_asof(
        out.sort_values("timestamp_dt"),
        hr.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    out["trend_agree"] = ((out["p_day_up"] > 0.55) & (out["p_hour_up"] > 0.55)).astype(int) - ((out["p_day_up"] < 0.45) & (out["p_hour_up"] < 0.45)).astype(int)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    d1 = _load_scale(P_DAILY, "daily")
    h1 = _load_scale(P_HOURLY, "hourly")
    m30 = _load_scale(P_30M, "30m")

    split = _time_split(m30)
    d1_tr, d1_va, d1_te = _partition(d1, split)
    h1_tr, h1_va, h1_te = _partition(h1, split)
    m30_tr, m30_va, m30_te = _partition(m30, split)

    day_cols = _feature_cols(d1_tr)
    hr_cols = _feature_cols(h1_tr)
    m30_cols = _feature_cols(m30_tr)

    day_clf = _fit_model(
        d1_tr[day_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32),
        d1_tr["target"].to_numpy(dtype=int),
    )
    hr_clf = _fit_model(
        h1_tr[hr_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32),
        h1_tr["target"].to_numpy(dtype=int),
    )

    d1_p_all = day_clf.predict_proba(d1[day_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32))[:, 1]
    h1_p_all = hr_clf.predict_proba(h1[hr_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32))[:, 1]
    m30_aug = _merge_probs(m30, d1, h1, d1_p_all, h1_p_all)

    m30_tr = m30_aug[m30_aug["timestamp_dt"] <= split.train_end_ts].copy()
    m30_va = m30_aug[(m30_aug["timestamp_dt"] > split.train_end_ts) & (m30_aug["timestamp_dt"] <= split.valid_end_ts)].copy()
    m30_te = m30_aug[m30_aug["timestamp_dt"] > split.valid_end_ts].copy()

    entry_cols = m30_cols + ["p_day_up", "p_hour_up", "trend_agree"]
    m30_clf = _fit_model(
        m30_tr[entry_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32),
        m30_tr["target"].to_numpy(dtype=int),
    )

    report = {
        "split": {
            "train_end_ts": str(split.train_end_ts),
            "valid_end_ts": str(split.valid_end_ts),
        },
        "daily": {
            "valid": _eval_block("daily_valid", day_clf, d1_va, day_cols),
            "test": _eval_block("daily_test", day_clf, d1_te, day_cols),
        },
        "hourly": {
            "valid": _eval_block("hourly_valid", hr_clf, h1_va, hr_cols),
            "test": _eval_block("hourly_test", hr_clf, h1_te, hr_cols),
        },
        "entry_30m": {
            "valid": _eval_block("entry_valid", m30_clf, m30_va, entry_cols),
            "test": _eval_block("entry_test", m30_clf, m30_te, entry_cols),
        },
        "config": {
            "fee_bps": FEE_BPS,
            "slippage_bps": SLIP_BPS,
            "initial_capital": INITIAL_CAPITAL,
        },
        "paths": {
            "daily": str(P_DAILY),
            "hourly": str(P_HOURLY),
            "entry_30m": str(P_30M),
        },
    }

    bundle = {
        "version": "v11_multiscale_3model",
        "daily": {"model": day_clf, "feature_cols": day_cols},
        "hourly": {"model": hr_clf, "feature_cols": hr_cols},
        "entry": {"model": m30_clf, "feature_cols": entry_cols},
        "split": {"train_end_ts": str(split.train_end_ts), "valid_end_ts": str(split.valid_end_ts)},
    }

    with (MODELS_DIR / "v11_multiscale_bundle.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] v11 multiscale training finished")
    print(json.dumps(report["entry_30m"]["test"], ensure_ascii=False, indent=2))
    print(f"[OK] model: {MODELS_DIR / 'v11_multiscale_bundle.pkl'}")
    print(f"[OK] report: {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

