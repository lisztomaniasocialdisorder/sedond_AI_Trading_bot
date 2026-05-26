#!/usr/bin/env python3
"""
evaluate.py
===========
Walk-forward 評估指標 + 圖表輸出。

指標說明
--------
IC      : Spearman rank correlation(prediction, actual) per fold
ICIR    : mean(IC) / std(IC)  — 信號穩定性（越高越好）
Hit     : sign(pred) == sign(actual)  方向準確率
Sharpe  : long top-quintile / short bottom-quintile 模擬年化 Sharpe
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── Per-fold metrics ──────────────────────────────────────────────────────────

def spearman_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation between prediction and actual return."""
    if len(y_pred) < 5:
        return float("nan")
    r, _ = stats.spearmanr(y_pred, y_true, nan_policy="omit")
    return float(r)


def directional_accuracy(y_pred: np.ndarray, y_true: np.ndarray,
                         min_abs: float = 0.0) -> float:
    """
    Fraction of samples where sign(pred) == sign(actual).
    Optionally filter samples where |actual| < min_abs (near-zero).
    """
    mask = np.abs(y_true) >= min_abs
    if mask.sum() < 5:
        return float("nan")
    return float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())


def quintile_sharpe(y_pred: np.ndarray, y_true: np.ndarray,
                    ann_factor: float = np.sqrt(60 * 24 * 365)) -> float:
    """
    Simulate: long top-20% predictions, short bottom-20%.
    Return annualised Sharpe of the daily PnL stream.

    ann_factor default = sqrt(seconds per year) for 1-second data.
    """
    if len(y_pred) < 20:
        return float("nan")

    q20 = np.percentile(y_pred, 20)
    q80 = np.percentile(y_pred, 80)
    long_pnl  = y_true[y_pred >= q80]
    short_pnl = -y_true[y_pred <= q20]
    pnl = np.concatenate([long_pnl, short_pnl])

    if pnl.std() < 1e-12:
        return float("nan")
    return float(pnl.mean() / pnl.std() * ann_factor)


# ── Aggregate over folds ──────────────────────────────────────────────────────

def summarise(results: pd.DataFrame) -> pd.Series:
    """
    Summarise walk-forward results DataFrame.

    Expected columns: ic, hit, sharpe, n_test, fold_id
    """
    ic   = results["ic"].dropna()
    hit  = results["hit"].dropna()
    sh   = results["sharpe"].dropna()

    return pd.Series({
        "n_folds":         len(results),
        "ic_mean":         ic.mean(),
        "ic_std":          ic.std(),
        "icir":            ic.mean() / ic.std() if ic.std() > 1e-12 else float("nan"),
        "ic_positive_pct": (ic > 0).mean() * 100,
        "hit_mean":        hit.mean() * 100,
        "sharpe_mean":     sh.mean(),
        "total_test_rows": results["n_test"].sum(),
    })


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: pd.DataFrame,
                 feature_names: Optional[list[str]] = None,
                 importances: Optional[np.ndarray] = None,
                 output_dir: Path = Path("results")) -> None:
    """
    Save diagnostic plots to output_dir.

    Plots
    -----
    1. Cumulative IC over folds
    2. IC distribution (histogram)
    3. Feature importances (if provided)
    4. Hit rate over folds
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless (no GUI needed)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[evaluate] matplotlib/seaborn not installed — skipping plots.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="darkgrid", palette="muted")
    fig_size = (10, 4)

    # 1. Cumulative IC
    fig, ax = plt.subplots(figsize=fig_size)
    cum_ic = results["ic"].fillna(0).cumsum()
    ax.plot(cum_ic.values, color="#00e891", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative IC (Spearman) over Walk-Forward Folds")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Cumulative IC")
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_ic.png", dpi=150)
    plt.close(fig)

    # 2. IC distribution
    fig, ax = plt.subplots(figsize=fig_size)
    ic_vals = results["ic"].dropna()
    ax.hist(ic_vals, bins=max(10, len(ic_vals) // 5),
            color="#0cc9fa", edgecolor="black", alpha=0.8)
    ax.axvline(ic_vals.mean(), color="#f7931a", linestyle="--",
               label=f"Mean={ic_vals.mean():.4f}")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("IC Distribution")
    ax.set_xlabel("IC (Spearman)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ic_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Feature importances
    if feature_names and importances is not None and len(feature_names) == len(importances):
        n = min(30, len(feature_names))
        idx = np.argsort(importances)[-n:]
        fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
        ax.barh(np.array(feature_names)[idx], importances[idx],
                color="#f7931a", alpha=0.85)
        ax.set_title(f"Top {n} Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    # 4. Hit rate over folds
    fig, ax = plt.subplots(figsize=fig_size)
    hit_vals = results["hit"].fillna(0.5) * 100
    ax.plot(hit_vals.values, color="#f7931a", linewidth=1.5)
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50% (random)")
    ax.set_ylim(30, 70)
    ax.set_title("Directional Hit Rate over Folds (%)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Hit Rate (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hit_rate.png", dpi=150)
    plt.close(fig)

    print(f"[evaluate] Saved plots → {output_dir.resolve()}")
