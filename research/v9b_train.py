#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"LightGBM is required for v9b_train.py: {exc}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v9b edge models (long/short binary).")
    p.add_argument("--base-dir", required=True, help="Trading base dir.")
    p.add_argument("--experiment", default="v9b", help="Experiment folder name under experiments/.")
    return p.parse_args()


def make_splits(n_rows: int, folds: int = 3) -> list[dict[str, tuple[int, int]]]:
    min_train = max(8000, int(n_rows * 0.35))
    valid_size = max(3500, int(n_rows * 0.12))
    test_size = max(3500, int(n_rows * 0.15))
    total = min_train + valid_size + test_size
    if total >= n_rows:
        valid_size = max(2000, int(n_rows * 0.12))
        test_size = max(2000, int(n_rows * 0.12))
        min_train = n_rows - valid_size - test_size
    room = n_rows - (min_train + valid_size + test_size)
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


def fit_binary(X: pd.DataFrame, y: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=360,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_samples=60,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def decide_actions(
    p_long: np.ndarray,
    p_short: np.ndarray,
    p_edge: float,
    delta: float,
) -> np.ndarray:
    actions = np.zeros(len(p_long), dtype=np.int8)
    long_mask = (p_long >= p_edge) & ((p_long - p_short) >= delta)
    short_mask = (p_short >= p_edge) & ((p_short - p_long) >= delta)
    actions[long_mask] = 1
    actions[short_mask] = -1
    return actions


def simulate(actions: np.ndarray, future_ret: np.ndarray, horizon: int, cost_bps: float) -> dict[str, float]:
    idxs = []
    last = -10**9
    for i, a in enumerate(actions):
        if a == 0:
            continue
        if i - last < horizon:
            continue
        fr = float(future_ret[i])
        if not np.isfinite(fr):
            continue
        idxs.append((i, int(a), fr))
        last = i

    if not idxs:
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
    long_count = 0
    short_count = 0
    nets = []
    for _, a, fr in idxs:
        net = (a * fr) - (cost_bps / 10000.0)
        net = max(net, -0.99)
        cap *= (1.0 + net)
        nets.append(net)
        if a > 0:
            long_count += 1
        else:
            short_count += 1
        if net > 0:
            wins += 1

    trades = len(idxs)
    return {
        "capital": float(cap),
        "excess": float(cap - 10000.0),
        "trades": int(trades),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "direction_share": float(min(long_count, short_count) / max(1, long_count + short_count)),
        "win_rate": float(wins / max(1, trades)),
        "avg_net_bps": float(np.mean(np.array(nets, dtype=float)) * 10000.0),
    }


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir).resolve()
    feat_path = base / "features" / "training_arena_features_v8.parquet"
    exp_dir = base / "experiments" / args.experiment
    model_dir = exp_dir / "models"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature parquet: {feat_path}")

    horizons = [60, 180, 300]
    edge_threshold_bps = {60: 5.0, 180: 7.0, 300: 9.0}
    cost_bps = 6.0
    p_edges = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    deltas = [0.00, 0.03, 0.06, 0.10, 0.14]

    print("[v9b] loading:", feat_path)
    df = pd.read_parquet(feat_path).sort_values("timestamp_dt").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    price = pd.to_numeric(df["price"], errors="coerce").astype("float64")

    for h in horizons:
        fr_col = f"future_return_{h}s"
        th = edge_threshold_bps[h] / 10000.0
        df[fr_col] = (price.shift(-h) / price) - 1.0
        df[f"long_y_{h}s"] = (df[fr_col] > th).astype(np.int8)
        df[f"short_y_{h}s"] = (df[fr_col] < -th).astype(np.int8)

    max_h = max(horizons)
    df = df.iloc[:-max_h].copy().reset_index(drop=True)

    feature_cols = []
    for c in df.columns:
        if c in {"timestamp_dt", "price", "future_return", "label"}:
            continue
        if c.startswith("future_return_") or c.startswith("long_y_") or c.startswith("short_y_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    splits = make_splits(len(df), folds=3)
    fold_rows: list[dict[str, object]] = []
    grid_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    best_bundle: dict[str, object] | None = None
    best_score = -1e18

    for h in horizons:
        ly = df[f"long_y_{h}s"]
        sy = df[f"short_y_{h}s"]
        label_rows.append({"horizon": h, "target": "long", "positive_ratio": float(ly.mean())})
        label_rows.append({"horizon": h, "target": "short", "positive_ratio": float(sy.mean())})

    for split in splits:
        fold = int(split["fold"])
        a, b = split["train"]
        c, d = split["valid"]
        e, f = split["test"]

        train_x_raw = df.iloc[a:b][feature_cols].replace([np.inf, -np.inf], np.nan)
        medians = train_x_raw.median(numeric_only=True).replace([np.inf, -np.inf], 0).fillna(0)
        X_train = train_x_raw.fillna(medians).fillna(0).astype(np.float32)
        X_valid = df.iloc[c:d][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0).astype(np.float32)
        X_test = df.iloc[e:f][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0).astype(np.float32)

        for h in horizons:
            y_long = df[f"long_y_{h}s"].to_numpy(dtype=np.int8)
            y_short = df[f"short_y_{h}s"].to_numpy(dtype=np.int8)
            fr = df[f"future_return_{h}s"].to_numpy(dtype=float)

            y_long_train = y_long[a:b]
            y_short_train = y_short[a:b]
            if int(y_long_train.sum()) < 50 or int(y_short_train.sum()) < 50:
                print(f"[v9b] fold {fold} h{h}: skipped due to too few positives")
                continue

            long_model = fit_binary(X_train, y_long_train, seed=1000 + fold * 10 + h)
            short_model = fit_binary(X_train, y_short_train, seed=2000 + fold * 10 + h)

            p_long_valid = long_model.predict_proba(X_valid)[:, 1]
            p_short_valid = short_model.predict_proba(X_valid)[:, 1]
            fr_valid = fr[c:d]

            best_valid = None
            for p_edge in p_edges:
                for delta in deltas:
                    acts = decide_actions(p_long_valid, p_short_valid, p_edge, delta)
                    met = simulate(acts, fr_valid, h, cost_bps)
                    row = {"fold": fold, "horizon": h, "split": "valid", "p_edge": p_edge, "delta": delta, **met}
                    grid_rows.append(row)
                    pass_gate = (
                        met["trades"] >= 10
                        and met["avg_net_bps"] > 0
                        and met["excess"] > 0
                        and met["direction_share"] >= 0.02
                    )
                    score = (1000000 if pass_gate else 0) + met["excess"] + 0.2 * met["trades"] + 5.0 * met["direction_share"]
                    if best_valid is None or score > best_valid["score"]:
                        best_valid = {"score": score, "pass_gate": pass_gate, "p_edge": p_edge, "delta": delta, **met}

            assert best_valid is not None

            p_long_test = long_model.predict_proba(X_test)[:, 1]
            p_short_test = short_model.predict_proba(X_test)[:, 1]
            fr_test = fr[e:f]
            acts_test = decide_actions(p_long_test, p_short_test, float(best_valid["p_edge"]), float(best_valid["delta"]))
            met_test = simulate(acts_test, fr_test, h, cost_bps)

            fold_rows.append(
                {
                    "fold": fold,
                    "horizon": h,
                    "p_edge": float(best_valid["p_edge"]),
                    "delta": float(best_valid["delta"]),
                    "valid_pass_gate": bool(best_valid["pass_gate"]),
                    "valid_excess": float(best_valid["excess"]),
                    "valid_trades": int(best_valid["trades"]),
                    "valid_avg_net_bps": float(best_valid["avg_net_bps"]),
                    "valid_direction_share": float(best_valid["direction_share"]),
                    "test_excess": float(met_test["excess"]),
                    "test_trades": int(met_test["trades"]),
                    "test_avg_net_bps": float(met_test["avg_net_bps"]),
                    "test_direction_share": float(met_test["direction_share"]),
                    "test_win_rate": float(met_test["win_rate"]),
                    "test_long_count": int(met_test["long_count"]),
                    "test_short_count": int(met_test["short_count"]),
                }
            )

            local_score = met_test["excess"] + 0.3 * met_test["trades"] + 12.0 * met_test["direction_share"]
            if local_score > best_score:
                best_score = local_score
                best_bundle = {
                    "horizon": h,
                    "p_edge": float(best_valid["p_edge"]),
                    "delta": float(best_valid["delta"]),
                    "feature_cols": feature_cols,
                    "medians": medians,
                    "long_model": long_model,
                    "short_model": short_model,
                    "cost_bps": cost_bps,
                    "edge_threshold_bps": edge_threshold_bps[h],
                    "version": "v9b_dual_binary",
                }

            print(
                f"[v9b] fold={fold} h={h} | valid ex={best_valid['excess']:.2f} tr={best_valid['trades']} "
                f"| test ex={met_test['excess']:.2f} tr={met_test['trades']} "
                f"dir={met_test['direction_share']:.3f} avg={met_test['avg_net_bps']:.3f}bps"
            )

    df_folds = pd.DataFrame(fold_rows)
    df_grid = pd.DataFrame(grid_rows)
    df_labels = pd.DataFrame(label_rows)

    if df_folds.empty:
        summary = {
            "status": "no_valid_folds",
            "reason": "No fold/horizon produced trainable labels.",
            "model_saved": False,
        }
    else:
        agg = (
            df_folds.groupby("horizon", as_index=False)
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
            & (agg["total_test_trades"] >= 15)
        )
        agg = agg.sort_values(["gate_pass", "mean_test_excess", "positive_fold_ratio"], ascending=[False, False, False])
        top = agg.iloc[0].to_dict()
        model_saved = bool(top["gate_pass"]) and best_bundle is not None
        if model_saved:
            with (model_dir / "best_edge_model.pkl").open("wb") as f:
                pickle.dump(best_bundle, f)
        summary = {
            "status": "edge_model_ready" if model_saved else "no_valid_edge_model",
            "reason": "cross-fold gate passed" if model_saved else "cross-fold gate failed",
            "model_saved": model_saved,
            "best_horizon": int(top["horizon"]),
            "best_positive_fold_ratio": float(top["positive_fold_ratio"]),
            "best_mean_test_excess": float(top["mean_test_excess"]),
            "best_total_test_trades": int(top["total_test_trades"]),
            "best_mean_test_avg_net_bps": float(top["mean_test_avg_net_bps"]),
        }
        agg.to_parquet(exp_dir / "horizon_summary.parquet", index=False)

    df_labels.to_parquet(exp_dir / "label_report.parquet", index=False)
    df_grid.to_parquet(exp_dir / "threshold_search.parquet", index=False)
    df_folds.to_parquet(exp_dir / "fold_log.parquet", index=False)
    pd.DataFrame([summary]).to_parquet(exp_dir / "summary.parquet", index=False)
    (exp_dir / "status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[v9b] summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
