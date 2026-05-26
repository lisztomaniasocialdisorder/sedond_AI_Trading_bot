from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INTERVALS = ("5m", "15m", "30m", "1h", "1d")


@dataclass
class QualityResult:
    interval: str
    rows: int
    matched_rows: int
    prob_sum_mae: float
    confidence_mean: float
    confidence_p90: float
    entropy_mean: float
    long_ratio: float
    flat_ratio: float
    short_ratio: float
    top1_accuracy: float | None
    brier_multiclass: float | None
    leverage_mae: float | None
    quality_score: float
    quality_status: str



def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")



def _normalize_probs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("soft_p_long", "soft_p_flat", "soft_p_short"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(0.0).clip(lower=0.0)
    s = out[["soft_p_long", "soft_p_flat", "soft_p_short"]].sum(axis=1)
    bad = s <= 0
    if bad.any():
        out.loc[bad, ["soft_p_long", "soft_p_flat", "soft_p_short"]] = [1 / 3, 1 / 3, 1 / 3]
        s = out[["soft_p_long", "soft_p_flat", "soft_p_short"]].sum(axis=1)
    out[["soft_p_long", "soft_p_flat", "soft_p_short"]] = out[["soft_p_long", "soft_p_flat", "soft_p_short"]].div(s, axis=0)
    return out



def _soft_quality_status(score: float) -> str:
    if score >= 0.75:
        return "good"
    if score >= 0.55:
        return "usable"
    if score >= 0.40:
        return "usable_weak"
    return "poor"



def evaluate_interval(
    interval: str,
    soft_path: Path,
    train_path: Path | None,
) -> QualityResult:
    soft = _safe_read_csv(soft_path)
    if soft.empty:
        raise ValueError(f"soft label file is empty: {soft_path}")

    if "timestamp" not in soft.columns:
        raise ValueError(f"missing timestamp in {soft_path}")

    soft["timestamp"] = pd.to_datetime(soft["timestamp"], utc=True, errors="coerce")
    soft = soft[soft["timestamp"].notna()].copy()
    soft = _normalize_probs(soft)
    soft = soft.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    p = soft[["soft_p_long", "soft_p_flat", "soft_p_short"]].to_numpy(dtype=np.float64)
    prob_sum = p.sum(axis=1)
    prob_sum_mae = float(np.mean(np.abs(prob_sum - 1.0)))

    conf = np.max(p, axis=1)
    confidence_mean = float(np.mean(conf))
    confidence_p90 = float(np.quantile(conf, 0.90))

    entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    entropy_mean = float(np.mean(entropy))

    pred_idx = np.argmax(p, axis=1)
    long_ratio = float(np.mean(pred_idx == 0))
    flat_ratio = float(np.mean(pred_idx == 1))
    short_ratio = float(np.mean(pred_idx == 2))

    top1_accuracy: float | None = None
    brier_mc: float | None = None
    leverage_mae: float | None = None
    matched_rows = 0

    if train_path is not None and train_path.exists():
        tr = _safe_read_csv(train_path)
        if "timestamp" in tr.columns and "label" in tr.columns:
            tr["timestamp"] = pd.to_datetime(tr["timestamp"], utc=True, errors="coerce")
            tr = tr[tr["timestamp"].notna()].copy()
            tr = tr.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            tr["label"] = pd.to_numeric(tr["label"], errors="coerce").fillna(0).round().clip(-1, 1).astype(int)
            m = soft.merge(tr[["timestamp", "label"]], on="timestamp", how="inner")
            matched_rows = int(len(m))
            if matched_rows > 0:
                pred_map = {0: 1, 1: 0, 2: -1}
                pred_label = np.array([pred_map[i] for i in np.argmax(m[["soft_p_long", "soft_p_flat", "soft_p_short"]].to_numpy(), axis=1)])
                y_true = m["label"].to_numpy(dtype=int)
                top1_accuracy = float(np.mean(pred_label == y_true))

                y_onehot = np.zeros((matched_rows, 3), dtype=np.float64)
                idx_map = {-1: 2, 0: 1, 1: 0}
                for i, y in enumerate(y_true):
                    y_onehot[i, idx_map[int(y)]] = 1.0
                pred_prob = m[["soft_p_long", "soft_p_flat", "soft_p_short"]].to_numpy(dtype=np.float64)
                brier_mc = float(np.mean(np.sum((pred_prob - y_onehot) ** 2, axis=1)))

        if "teacher_leverage" in soft.columns and "target_leverage" in tr.columns and "timestamp" in tr.columns:
            tr2 = tr[["timestamp", "target_leverage"]].copy()
            tr2["target_leverage"] = pd.to_numeric(tr2["target_leverage"], errors="coerce")
            m2 = soft.merge(tr2, on="timestamp", how="inner")
            if len(m2) > 0:
                lev_pred = pd.to_numeric(m2["teacher_leverage"], errors="coerce")
                lev_true = pd.to_numeric(m2["target_leverage"], errors="coerce")
                ok = lev_pred.notna() & lev_true.notna()
                if ok.any():
                    leverage_mae = float(np.mean(np.abs(lev_pred[ok] - lev_true[ok])))

    # Quality score: confidence + low entropy + optional supervised metrics
    entropy_norm = float(np.clip(entropy_mean / np.log(3.0), 0.0, 1.0))
    score = 0.30 * confidence_mean + 0.25 * (1.0 - entropy_norm) + 0.20 * (1.0 - min(prob_sum_mae * 50.0, 1.0))
    if top1_accuracy is not None:
        score += 0.20 * top1_accuracy
    if brier_mc is not None:
        score += 0.05 * (1.0 - min(max(brier_mc, 0.0), 1.0))
    score = float(np.clip(score, 0.0, 1.0))

    status = _soft_quality_status(score)

    return QualityResult(
        interval=interval,
        rows=int(len(soft)),
        matched_rows=matched_rows,
        prob_sum_mae=prob_sum_mae,
        confidence_mean=confidence_mean,
        confidence_p90=confidence_p90,
        entropy_mean=entropy_mean,
        long_ratio=long_ratio,
        flat_ratio=flat_ratio,
        short_ratio=short_ratio,
        top1_accuracy=top1_accuracy,
        brier_multiclass=brier_mc,
        leverage_mae=leverage_mae,
        quality_score=score,
        quality_status=status,
    )



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate teacher soft-label quality across intervals.")
    parser.add_argument("--base", default="outputs", help="Base folder containing teacher_soft_labels_*.csv")
    parser.add_argument("--train-base", default="outputs/teacher_training", help="Folder containing teacher_train_*.csv")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--intervals", default=",".join(INTERVALS), help="Comma-separated intervals")
    parser.add_argument("--save-json", default="outputs/teacher_quality_report.json")
    parser.add_argument("--save-csv", default="outputs/teacher_quality_report.csv")
    args = parser.parse_args()

    base = Path(args.base)
    train_base = Path(args.train_base)
    intervals = [x.strip() for x in str(args.intervals).split(",") if x.strip()]

    rows: list[dict] = []
    for tf in intervals:
        soft_path = base / f"teacher_soft_labels_{args.symbol}_{tf}.csv"
        train_path = train_base / f"teacher_train_{args.symbol}_{tf}.csv"
        if not soft_path.exists():
            print(f"[skip] missing soft labels: {soft_path}")
            continue
        try:
            res = evaluate_interval(tf, soft_path, train_path if train_path.exists() else None)
            rows.append(res.__dict__)
            print(f"[ok] {tf}: score={res.quality_score:.4f} status={res.quality_status} rows={res.rows}")
        except Exception as e:
            print(f"[fail] {tf}: {e}")

    if not rows:
        raise SystemExit("No interval was evaluated. Please check file paths.")

    df = pd.DataFrame(rows).sort_values("interval").reset_index(drop=True)

    out_json = Path(args.save_json)
    out_csv = Path(args.save_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "symbol": args.symbol,
        "intervals": intervals,
        "generated_rows": len(df),
        "quality_mean": float(df["quality_score"].mean()),
        "quality_min": float(df["quality_score"].min()),
        "quality_max": float(df["quality_score"].max()),
        "results": rows,
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== Teacher Quality Summary ===")
    print(df[["interval", "rows", "matched_rows", "quality_score", "quality_status", "top1_accuracy", "brier_multiclass", "leverage_mae"]].to_string(index=False))
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
