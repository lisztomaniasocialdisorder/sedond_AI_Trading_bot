#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report


BUNDLE_PATH = Path(r"C:\Users\brian\klinetraning\data_kline\training_arena_multiscale_bundle.parquet")
DL_EXPORT_PATH = Path(r"G:\我的雲端硬碟\klinetraning\exports\dl_predictions.parquet")
OUT_DIR = Path(r"C:\Users\brian\trading\experiments\rf_from_dl")


def _load_frame(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_parquet(p).copy()
    ts_col = "timestamp_dt" if "timestamp_dt" in df.columns else "timestamp"
    df["timestamp_dt"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    return df


def _prep_dl(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    # keep only model outputs / embeddings
    candidates = [c for c in numeric_cols if c.startswith(("p_", "dl_", "emb_", "latent_"))]
    if not candidates:
        # fallback: take numeric columns except common market columns
        drop = {"open", "high", "low", "close", "volume", "price", "future_return", "label"}
        candidates = [c for c in numeric_cols if c not in drop]
    keep = ["timestamp_dt"] + candidates
    out = out[keep].copy()
    return out


def train() -> dict:
    bundle = _load_frame(BUNDLE_PATH)
    dl = _prep_dl(_load_frame(DL_EXPORT_PATH))

    merged = pd.merge_asof(
        bundle.sort_values("timestamp_dt"),
        dl.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=["future_return"]).reset_index(drop=True)
    merged["target"] = (pd.to_numeric(merged["future_return"], errors="coerce") > 0).astype(int)

    feat_cols = [
        c
        for c in merged.columns
        if c not in {"timestamp_dt", "future_return", "label", "target"}
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    x = merged[feat_cols].fillna(0.0).to_numpy(dtype=float)
    y = merged["target"].to_numpy(dtype=int)

    n = len(merged)
    split = int(n * 0.80)
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    clf = RandomForestClassifier(
        n_estimators=700,
        max_depth=14,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)

    pred = clf.predict(x_test)
    bacc = float(balanced_accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "rf_model.pkl"
    report_path = OUT_DIR / "report.json"
    feats_path = OUT_DIR / "feature_cols.json"

    with model_path.open("wb") as f:
        pickle.dump(clf, f)
    report_payload = {
        "rows_total": int(n),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "balanced_accuracy": bacc,
        "classification_report": report,
        "bundle_path": str(BUNDLE_PATH),
        "dl_export_path": str(DL_EXPORT_PATH),
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    feats_path.write_text(json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "bacc": bacc,
        "n_features": len(feat_cols),
        "rows": n,
        "model_path": str(model_path),
        "report_path": str(report_path),
    }


def main() -> int:
    out = train()
    print("[OK] RF from DL done")
    print(f"[OK] rows={out['rows']:,}, features={out['n_features']}")
    print(f"[OK] balanced_accuracy={out['bacc']:.4f}")
    print(f"[OK] model={out['model_path']}")
    print(f"[OK] report={out['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

