import json
from pathlib import Path


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code_cell(code: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip("\n").split("\n")],
    }


nb = {
    "cells": [
        md_cell(
            """
# v14 Research Pipeline
DL -> RF -> Regime-Routed Specialist GA Ensemble -> Nested Walk-forward
"""
        ),
        code_cell(
            """
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Colab mount, harmless outside Colab.
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

DRIVE_ROOT = Path("/content/drive/MyDrive")
TARGET_ROOT = DRIVE_ROOT / "klinetrading"
LEGACY_ROOT = DRIVE_ROOT / "klinetraning"

def find_project_root():
    candidates = [TARGET_ROOT, LEGACY_ROOT]
    bundle_names = [
        "training_arena_multiscale_bundle.parquet",
        "training_arena_features_v7_1h.parquet",
        "training_arena_features_v7_30m.parquet",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for name in bundle_names:
            if list(root.rglob(name)):
                return root
    return TARGET_ROOT

ROOT = find_project_root()
OUTPUT_ROOT = TARGET_ROOT

FEATURE_OUT = OUTPUT_ROOT / "features"
NOTEBOOK_OUT = OUTPUT_ROOT / "notebook"
FEATURE_OUT.mkdir(parents=True, exist_ok=True)
NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)

bundle_candidates = []
for root in [ROOT, TARGET_ROOT, LEGACY_ROOT]:
    bundle_candidates += [
        root / "features" / "training_arena_multiscale_bundle.parquet",
        root / "data_kline" / "training_arena_multiscale_bundle.parquet",
    ]
    if root.exists():
        bundle_candidates += list(root.rglob("*multiscale*bundle*.parquet"))

P_BUNDLE = next((p for p in bundle_candidates if p.exists()), None)
if P_BUNDLE is None:
    raise FileNotFoundError(f"bundle not found, tried: {bundle_candidates}")

P_DL_EXPORT = ROOT / "exports" / "dl_predictions.parquet"
print("ROOT:", ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("bundle:", P_BUNDLE)
print("dl_export:", P_DL_EXPORT)
"""
        ),
        code_cell(
            """
def load_base_frame():
    df = pd.read_parquet(P_BUNDLE).copy()
    if "timestamp_dt" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        raise ValueError("missing timestamp column")
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    if "target_up" in df.columns:
        df["target"] = (pd.to_numeric(df["target_up"], errors="coerce") > 0).astype(int)
    elif "future_return" in df.columns:
        df["target"] = (pd.to_numeric(df["future_return"], errors="coerce") > 0).astype(int)
    else:
        raise ValueError("missing target")
    return df

base = load_base_frame()
base.shape
"""
        ),
        code_cell(
            """
# Merge DL outputs (teacher / logits / probs)
if P_DL_EXPORT.exists():
    dl = pd.read_parquet(P_DL_EXPORT).copy()
    if "timestamp_dt" in dl.columns:
        dl["timestamp_dt"] = pd.to_datetime(dl["timestamp_dt"], utc=True, errors="coerce")
    merged = pd.merge_asof(
        base.sort_values("timestamp_dt"),
        dl.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
else:
    merged = base.copy()

merged.shape
"""
        ),
        code_cell(
            """
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

drop_cols = {"timestamp_dt", "timestamp", "target", "target_up", "future_return", "next_return"}
feat_cols = [c for c in merged.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(merged[c])]
X = merged[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
y = merged["target"].to_numpy(int)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)
merged["rf_p_up"] = rf.predict_proba(X)[:, 1]
print("rf trained:", len(feat_cols), "features")
"""
        ),
        code_cell(
            """
# Regime routing + specialist score (light GA-like weighted ensemble search)
def build_regime(df):
    out = df.copy()
    ret = pd.to_numeric(out.get("future_return", 0.0), errors="coerce").fillna(0.0)
    out["vol_96"] = ret.rolling(96, min_periods=24).std().fillna(0.0)
    q = out["vol_96"].quantile([0.33, 0.66]).values
    out["regime"] = np.where(out["vol_96"] <= q[0], "calm", np.where(out["vol_96"] <= q[1], "normal", "volatile"))
    return out

rdf = build_regime(merged)

cols_for_ens = [c for c in ["rf_p_up", "p_up", "dl_p_up", "teacher_prob_up"] if c in rdf.columns]
if not cols_for_ens:
    cols_for_ens = ["rf_p_up"]

best = {}
rng = np.random.default_rng(42)
for regime in ["calm", "normal", "volatile"]:
    part = rdf[rdf["regime"] == regime]
    if len(part) < 200:
        best[regime] = np.ones(len(cols_for_ens)) / len(cols_for_ens)
        continue
    Y = part["target"].to_numpy(int)
    P = part[cols_for_ens].to_numpy(np.float32)
    best_score, best_w = -1, None
    for _ in range(400):
        w = rng.random(len(cols_for_ens))
        w = w / w.sum()
        p = (P * w).sum(axis=1)
        pred = (p >= 0.5).astype(int)
        sc = balanced_accuracy_score(Y, pred)
        if sc > best_score:
            best_score, best_w = sc, w.copy()
    best[regime] = best_w
    print(regime, "best_bacc=", round(best_score, 4), "w=", dict(zip(cols_for_ens, best_w.round(4))))

def routed_prob(row):
    w = best.get(row["regime"])
    v = np.array([row[c] for c in cols_for_ens], dtype=np.float32)
    return float(np.dot(w, v))

rdf["routed_p_up"] = rdf.apply(routed_prob, axis=1)
"""
        ),
        code_cell(
            """
# Nested walk-forward (outer test, inner threshold tune)
def wf_eval(df, p_col="routed_p_up", n_outer=6):
    tscv = TimeSeriesSplit(n_splits=n_outer)
    rows = []
    y = df["target"].to_numpy(int)
    p = df[p_col].to_numpy(np.float32)
    for i, (tr, te) in enumerate(tscv.split(df), 1):
        tr_y, tr_p = y[tr], p[tr]
        te_y, te_p = y[te], p[te]
        thr_grid = np.linspace(0.45, 0.55, 11)
        best_thr, best_sc = 0.5, -1
        for thr in thr_grid:
            pred = (tr_p >= thr).astype(int)
            sc = balanced_accuracy_score(tr_y, pred)
            if sc > best_sc:
                best_sc, best_thr = sc, float(thr)
        te_pred = (te_p >= best_thr).astype(int)
        rows.append({
            "fold": i,
            "thr": best_thr,
            "bacc": balanced_accuracy_score(te_y, te_pred),
            "f1": f1_score(te_y, te_pred, zero_division=0),
            "rows_test": int(len(te)),
        })
    return pd.DataFrame(rows)

wf = wf_eval(rdf)
wf
"""
        ),
        code_cell(
            """
# Save feature artifact + report to GDrive
out_feat = FEATURE_OUT / "v14_routed_features.parquet"
out_wf = FEATURE_OUT / "v14_nested_wf_report.csv"
out_cfg = FEATURE_OUT / "v14_regime_weights.json"

keep_cols = ["timestamp_dt", "target", "regime", "rf_p_up", "routed_p_up"] + [c for c in cols_for_ens if c not in {"rf_p_up"}]
keep_cols = [c for c in keep_cols if c in rdf.columns]
rdf[keep_cols].to_parquet(out_feat, index=False)
wf.to_csv(out_wf, index=False)
out_cfg.write_text(json.dumps({k: list(map(float, v)) for k, v in best.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

print("saved:", out_feat)
print("saved:", out_wf)
print("saved:", out_cfg)
print("wf mean bacc:", round(float(wf["bacc"].mean()), 4))
"""
        ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(r"G:\我的雲端硬碟\klinetrading\notebook\v14_dl_rf_regime_ga_nested_wf.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
payload = json.dumps(nb, ensure_ascii=False, indent=2)
out_path.write_text(payload, encoding="utf-8")
(out_path.parent / "v14.ipynb").write_text(payload, encoding="utf-8")
print(out_path)
