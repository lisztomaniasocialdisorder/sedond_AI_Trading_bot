#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import lightgbm as lgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v9c regression edge model.")
    p.add_argument("--base-dir", required=True, help="Trading base dir")
    p.add_argument("--experiment", default="v9c_reg", help="Experiment folder under experiments/")
    return p.parse_args()


def make_splits(n_rows: int, folds: int = 3) -> list[dict[str, tuple[int, int]]]:
    min_train = max(9000, int(n_rows * 0.38))
    valid_size = max(3500, int(n_rows * 0.12))
    test_size = max(3500, int(n_rows * 0.15))
    room = n_rows - (min_train + valid_size + test_size)
    if room < 0:
        valid_size = max(2500, int(n_rows * 0.1))
        test_size = max(2500, int(n_rows * 0.1))
        min_train = n_rows - valid_size - test_size
        room = 0
    step = 0 if folds <= 1 else max(1, room // (folds - 1))
    out = []
    for k in range(folds):
        train_end = min_train + k * step
        valid_start = train_end
        valid_end = valid_start + valid_size
        test_start = valid_end
        test_end = test_start + test_size
        if test_end > n_rows:
            shift = test_end - n_rows
            train_end -= shift
            valid_start -= shift
            valid_end -= shift
            test_start -= shift
            test_end -= shift
        out.append(
            {
                "fold": k + 1,
                "train": (0, int(train_end)),
                "valid": (int(valid_start), int(valid_end)),
                "test": (int(test_start), int(test_end)),
            }
        )
    return out


def fit_reg(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        min_child_samples=80,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def decide_from_pred(pred: np.ndarray, abs_thr: float) -> np.ndarray:
    actions = np.zeros(len(pred), dtype=np.int8)
    actions[pred >= abs_thr] = 1
    actions[pred <= -abs_thr] = -1
    return actions


def simulate(actions: np.ndarray, future_ret: np.ndarray, horizon: int, cost_bps: float) -> dict[str, float]:
    picks = []
    last = -10**9
    for i, a in enumerate(actions):
        if a == 0:
            continue
        if i - last < horizon:
            continue
        r = float(future_ret[i])
        if not np.isfinite(r):
            continue
        picks.append((int(a), r))
        last = i

    if not picks:
        return {
            "capital": 10000.0,
            "excess": 0.0,
            "trades": 0,
            "long_count": 0,
            "short_count": 0,
            "direction_share": 0.0,
            "win_rate": 0.0,
            "avg_net_bps": 0.0,
        }

    cap = 10000.0
    wins = 0
    ln = 0
    sn = 0
    nets = []
    for a, r in picks:
        if a > 0:
            ln += 1
        else:
            sn += 1
        net = (a * r) - (cost_bps / 10000.0)
        net = max(net, -0.99)
        cap *= (1.0 + net)
        nets.append(net)
        if net > 0:
            wins += 1

    arr = np.array(nets, dtype=float)
    return {
        "capital": float(cap),
        "excess": float(cap - 10000.0),
        "trades": int(len(arr)),
        "long_count": int(ln),
        "short_count": int(sn),
        "direction_share": float(min(ln, sn) / max(1, ln + sn)),
        "win_rate": float(wins / max(1, len(arr))),
        "avg_net_bps": float(arr.mean() * 10000.0),
    }


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).resolve()
    feat_path = base / "features" / "training_arena_features_v8.parquet"
    exp = base / "experiments" / args.experiment
    model_dir = exp / "models"
    exp.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    horizons = [300, 600, 900]
    cost_bps = 6.0
    pred_abs_thrs = [0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0008, 0.0010, 0.0012]

    df = pd.read_parquet(feat_path).sort_values("timestamp_dt").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    price = pd.to_numeric(df["price"], errors="coerce").astype(float)

    for h in horizons:
        df[f"future_return_{h}s"] = (price.shift(-h) / price) - 1.0

    max_h = max(horizons)
    df = df.iloc[:-max_h].copy().reset_index(drop=True)

    feat_cols = []
    for c in df.columns:
        if c in {"timestamp_dt", "price", "future_return", "label"}:
            continue
        if c.startswith("future_return_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)

    splits = make_splits(len(df), folds=3)
    fold_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    best_model = None
    best_score = -1e18

    for sp in splits:
        fold = int(sp["fold"])
        a, b = sp["train"]
        c, d = sp["valid"]
        e, f = sp["test"]

        train_x_raw = df.iloc[a:b][feat_cols].replace([np.inf, -np.inf], np.nan)
        med = train_x_raw.median(numeric_only=True).replace([np.inf, -np.inf], 0).fillna(0)
        X_train = train_x_raw.fillna(med).fillna(0).astype(np.float32)
        X_valid = df.iloc[c:d][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0).astype(np.float32)
        X_test = df.iloc[e:f][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0).astype(np.float32)

        for h in horizons:
            y_train = df[f"future_return_{h}s"].iloc[a:b].to_numpy(dtype=float)
            y_valid = df[f"future_return_{h}s"].iloc[c:d].to_numpy(dtype=float)
            y_test = df[f"future_return_{h}s"].iloc[e:f].to_numpy(dtype=float)
            y_train = np.nan_to_num(y_train, nan=0.0)
            y_valid = np.nan_to_num(y_valid, nan=0.0)
            y_test = np.nan_to_num(y_test, nan=0.0)

            model = fit_reg(X_train, y_train, seed=7000 + fold * 10 + h)
            pred_valid = model.predict(X_valid)

            best_valid = None
            for thr in pred_abs_thrs:
                acts = decide_from_pred(pred_valid, thr)
                met = simulate(acts, y_valid, h, cost_bps)
                row = {"fold": fold, "horizon": h, "split": "valid", "pred_abs_thr": thr, **met}
                grid_rows.append(row)
                pass_gate = (
                    met["trades"] >= 8
                    and met["avg_net_bps"] > 0
                    and met["excess"] > 0
                    and met["direction_share"] >= 0.05
                )
                score = (1_000_000 if pass_gate else 0) + met["excess"] + 0.4 * met["trades"] + 20.0 * met["direction_share"]
                if best_valid is None or score > best_valid["score"]:
                    best_valid = {"score": score, "pass_gate": pass_gate, "pred_abs_thr": thr, **met}

            assert best_valid is not None
            pred_test = model.predict(X_test)
            acts_test = decide_from_pred(pred_test, float(best_valid["pred_abs_thr"]))
            test_met = simulate(acts_test, y_test, h, cost_bps)

            fold_rows.append(
                {
                    "fold": fold,
                    "horizon": h,
                    "pred_abs_thr": float(best_valid["pred_abs_thr"]),
                    "valid_pass_gate": bool(best_valid["pass_gate"]),
                    "valid_excess": float(best_valid["excess"]),
                    "valid_trades": int(best_valid["trades"]),
                    "valid_avg_net_bps": float(best_valid["avg_net_bps"]),
                    "valid_direction_share": float(best_valid["direction_share"]),
                    "test_excess": float(test_met["excess"]),
                    "test_trades": int(test_met["trades"]),
                    "test_avg_net_bps": float(test_met["avg_net_bps"]),
                    "test_direction_share": float(test_met["direction_share"]),
                    "test_win_rate": float(test_met["win_rate"]),
                    "test_long_count": int(test_met["long_count"]),
                    "test_short_count": int(test_met["short_count"]),
                }
            )

            local = test_met["excess"] + 0.4 * test_met["trades"] + 20.0 * test_met["direction_share"]
            if local > best_score:
                best_score = local
                best_model = {
                    "horizon": h,
                    "pred_abs_thr": float(best_valid["pred_abs_thr"]),
                    "cost_bps": cost_bps,
                    "feature_cols": feat_cols,
                    "medians": med,
                    "model": model,
                    "version": "v9c_regression",
                }

            print(
                f"[v9c] fold={fold} h={h} | valid ex={best_valid['excess']:.2f} tr={best_valid['trades']} "
                f"| test ex={test_met['excess']:.2f} tr={test_met['trades']} dir={test_met['direction_share']:.3f} "
                f"avg={test_met['avg_net_bps']:.3f}bps"
            )

    folds_df = pd.DataFrame(fold_rows)
    grid_df = pd.DataFrame(grid_rows)

    if folds_df.empty:
        summary = {"status": "no_valid_folds", "reason": "empty folds", "model_saved": False}
    else:
        agg = (
            folds_df.groupby("horizon", as_index=False)
            .agg(
                folds=("fold", "count"),
                positive_test_folds=("test_excess", lambda s: int((s > 0).sum())),
                mean_test_excess=("test_excess", "mean"),
                median_test_excess=("test_excess", "median"),
                total_test_trades=("test_trades", "sum"),
                mean_test_avg_net_bps=("test_avg_net_bps", "mean"),
                mean_test_direction_share=("test_direction_share", "mean"),
            )
        )
        agg["positive_fold_ratio"] = agg["positive_test_folds"] / agg["folds"].clip(lower=1)
        agg["gate_pass"] = (
            (agg["positive_fold_ratio"] >= 0.67)
            & (agg["mean_test_excess"] > 0)
            & (agg["total_test_trades"] >= 12)
            & (agg["mean_test_direction_share"] >= 0.08)
        )
        agg = agg.sort_values(["gate_pass", "mean_test_excess", "positive_fold_ratio"], ascending=[False, False, False])
        top = agg.iloc[0].to_dict()
        model_saved = bool(top["gate_pass"]) and best_model is not None
        if model_saved:
            with (model_dir / "best_edge_model.pkl").open("wb") as f:
                pickle.dump(best_model, f)
        summary = {
            "status": "edge_model_ready" if model_saved else "no_valid_edge_model",
            "reason": "cross-fold gate passed" if model_saved else "cross-fold gate failed",
            "model_saved": bool(model_saved),
            "best_horizon": int(top["horizon"]),
            "best_positive_fold_ratio": float(top["positive_fold_ratio"]),
            "best_mean_test_excess": float(top["mean_test_excess"]),
            "best_total_test_trades": int(top["total_test_trades"]),
            "best_mean_test_avg_net_bps": float(top["mean_test_avg_net_bps"]),
            "best_mean_test_direction_share": float(top["mean_test_direction_share"]),
        }
        agg.to_parquet(exp / "horizon_summary.parquet", index=False)

    folds_df.to_parquet(exp / "fold_log.parquet", index=False)
    grid_df.to_parquet(exp / "threshold_search.parquet", index=False)
    pd.DataFrame([summary]).to_parquet(exp / "summary.parquet", index=False)
    (exp / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[v9c] summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
