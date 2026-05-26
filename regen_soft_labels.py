"""
自訂 Teacher 推理腳本（相容 timeframe_aware_high_confidence_teacher 格式）
用現有 teacher_best_clf.joblib + teacher_reg.joblib 對全量歷史資料做推理，
生成覆蓋完整時間軸的 teacher_soft_labels_{interval}.csv。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

INTERVALS = ["5m", "15m", "30m", "1h"]
model_dir = Path("models")
output_dir = Path("outputs")


def _temperature_scale(proba: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """與 distillation.py 一致的溫度縮放。"""
    temperature = max(temperature, 0.01)
    log_p = np.log(np.clip(proba, 1e-9, 1.0)) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def _clean_xy(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    x = df.reindex(columns=feature_cols, fill_value=0)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.ffill().bfill().fillna(0)
    return x.to_numpy(dtype=np.float64)


for interval in INTERVALS:
    print(f"\n=== {interval} ===")
    tag = f"BTCUSDT_{interval}"
    t_dir = model_dir / "teacher" / tag

    # ── 載入 meta ─────────────────────────────────────────────────────────────
    meta_path = t_dir / "teacher_meta.json"
    if not meta_path.exists():
        print(f"  找不到 teacher_meta.json，跳過")
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    temperature = float(meta.get("temperature", 2.0))
    teacher_quality_weight = float(meta.get("teacher_quality_weight", 0.5))
    print(f"  best_model={meta.get('best_model')}  quality_weight={teacher_quality_weight:.3f}  verdict={meta.get('final_verdict')}")

    # ── 載入特徵欄位 ───────────────────────────────────────────────────────────
    feat_path = t_dir / "teacher_feature_cols.joblib"
    if not feat_path.exists():
        print(f"  找不到 teacher_feature_cols.joblib，跳過")
        continue
    feature_cols: list = joblib.load(feat_path)

    # ── 載入分類器（優先用 best_clf，回退到 rf_clf）───────────────────────────
    clf = None
    for clf_name in ["teacher_best_clf.joblib", "teacher_rf_clf.joblib", "teacher_gb_clf.joblib"]:
        p = t_dir / clf_name
        if p.exists():
            clf = joblib.load(p)
            print(f"  載入分類器: {clf_name} ({p.stat().st_size//1024:,} KB)")
            break
    if clf is None:
        print(f"  找不到任何分類器，跳過")
        continue

    # class_values: 從 classifier 的 classes_ 屬性取得（保證正確）
    class_values = np.array(sorted(clf.classes_.tolist()), dtype=int)
    print(f"  class_values: {class_values.tolist()}")

    # ── 載入回歸器（優先用 teacher_reg，再用 rf_reg）──────────────────────────
    reg = None
    for reg_name in ["teacher_reg.joblib", "teacher_rf_reg.joblib", "teacher_gb_reg.joblib"]:
        p = t_dir / reg_name
        if p.exists():
            reg = joblib.load(p)
            print(f"  載入回歸器: {reg_name} ({p.stat().st_size//1024:,} KB)")
            break

    # ── 載入全量資料 ───────────────────────────────────────────────────────────
    csv_path = output_dir / f"signals_with_features_{tag}.csv"
    if not csv_path.exists():
        print(f"  找不到 signals_with_features，跳過")
        continue

    print(f"  載入 {csv_path.name} ...", end="", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    print(f" {len(df):,} 筆")

    # 特徵對齊（只用 teacher 訓練時見過的欄位）
    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  警告: 缺少 {len(missing)} 個特徵，補 0 → {missing[:5]}...")

    print(f"  推理中 ({len(df):,} 筆)...", end="", flush=True)
    X = _clean_xy(df, feature_cols)

    # 分類器推理
    raw_proba = clf.predict_proba(X)  # shape: (N, n_classes)

    # 對齊 class 順序（classes_ 可能不是 [-1, 0, 1] 排序）
    clf_order = list(clf.classes_)
    aligned_proba = np.zeros((len(X), len(class_values)), dtype=np.float64)
    for i, c in enumerate(class_values):
        if c in clf_order:
            aligned_proba[:, i] = raw_proba[:, clf_order.index(c)]

    soft_proba = _temperature_scale(aligned_proba, temperature)
    print(f" OK")

    # 回歸器推理（槓桿）
    soft_leverage = None
    if reg is not None:
        try:
            soft_leverage = np.clip(reg.predict(X), 1.0, 100.0)
        except Exception as e:
            print(f"  回歸器推理失敗: {e}")

    # ── 組裝 soft labels DataFrame ─────────────────────────────────────────────
    out_df = pd.DataFrame()
    if "timestamp" in df.columns:
        out_df["timestamp"] = df["timestamp"].values

    # class_values 對應: 找 long/flat/short 的 index
    idx_long  = np.where(class_values == 1)[0]
    idx_flat  = np.where(class_values == 0)[0]
    idx_short = np.where(class_values == -1)[0]

    out_df["soft_p_long"]  = (soft_proba[:, idx_long[0]]  if len(idx_long)  else 1/3).round(6)
    out_df["soft_p_flat"]  = (soft_proba[:, idx_flat[0]]  if len(idx_flat)  else 1/3).round(6)
    out_df["soft_p_short"] = (soft_proba[:, idx_short[0]] if len(idx_short) else 1/3).round(6)

    out_df["raw_p_long"]   = (aligned_proba[:, idx_long[0]]  if len(idx_long)  else 1/3).round(6)
    out_df["raw_p_flat"]   = (aligned_proba[:, idx_flat[0]]  if len(idx_flat)  else 1/3).round(6)
    out_df["raw_p_short"]  = (aligned_proba[:, idx_short[0]] if len(idx_short) else 1/3).round(6)

    out_df["teacher_signal"]     = class_values[np.argmax(aligned_proba, axis=1)]
    out_df["teacher_confidence"] = np.max(aligned_proba, axis=1).round(6)

    if soft_leverage is not None:
        out_df["teacher_leverage"] = soft_leverage.round(4)

    # 保留真實標籤
    if "label" in df.columns:
        out_df["true_label"] = df["label"].values
    if "target_leverage" in df.columns:
        out_df["true_leverage"] = df["target_leverage"].values

    # teacher_quality_weight 整批一致（來自 meta）
    out_df["teacher_quality_weight"] = round(teacher_quality_weight, 6)
    out_df["student_sample_weight"]  = round(teacher_quality_weight, 6)

    # 儲存
    out_path = output_dir / f"teacher_soft_labels_{tag}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    sz_kb = out_path.stat().st_size // 1024

    print(f"  => {out_path.name} ({len(out_df):,} 筆, {sz_kb:,} KB)")
    if "timestamp" in out_df.columns:
        print(f"     時間: {out_df['timestamp'].iloc[0]}  →  {out_df['timestamp'].iloc[-1]}")
    print(f"     avg soft_p_long={out_df['soft_p_long'].mean():.3f}  "
          f"soft_p_flat={out_df['soft_p_flat'].mean():.3f}  "
          f"soft_p_short={out_df['soft_p_short'].mean():.3f}")
    print(f"     avg teacher_conf={out_df['teacher_confidence'].mean():.3f}")

print("\n完成！重新生成的 soft labels 已覆蓋全量歷史資料。")
print("請執行: python distill_retrain_all.py")
