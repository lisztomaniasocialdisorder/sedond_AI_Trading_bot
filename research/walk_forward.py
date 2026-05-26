#!/usr/bin/env python3
"""
walk_forward.py
===============
Walk-forward 引擎：依時間順序切 train/test fold，
在每個 fold 重新訓練模型並記錄評估指標。

設計原則
--------
- 嚴格無 look-ahead：test 永遠在 train 之後，中間保留 gap
- expanding window（預設）或 rolling window 可選
- 每 fold 返回 predictions，方便後續分析
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_engineering import build_features, db_time_range
from targets import add_targets
from models import BaseModel, make_model
from evaluate import spearman_ic, directional_accuracy, quintile_sharpe


# ── Fold definition ────────────────────────────────────────────────────────────

@dataclass
class Fold:
    fold_id:      int
    train_start:  pd.Timestamp
    train_end:    pd.Timestamp
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp


@dataclass
class FoldResult:
    fold_id:      int
    test_start:   pd.Timestamp
    test_end:     pd.Timestamp
    n_train:      int
    n_test:       int
    ic:           float
    hit:          float
    sharpe:       float
    fit_time_s:   float
    # Detailed predictions (optional, kept in memory)
    predictions:  Optional[pd.Series] = field(default=None, repr=False)
    actuals:      Optional[pd.Series] = field(default=None, repr=False)


# ── Walk-Forward Engine ────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Parameters
    ----------
    db_path          : path to SQLite DB
    model_name       : "ridge" | "lgbm"
    fwd_seconds      : prediction horizon (default 30)
    train_hours      : initial training window in hours (default 4)
    test_minutes     : test window per fold in minutes (default 30)
    gap_seconds      : gap between train end and test start (default 60)
    expanding        : True = expanding window; False = rolling window
    feature_cols     : explicit feature list; None = use all non-meta cols
    save_predictions : keep raw prediction series in FoldResult
    model_kwargs     : extra keyword arguments forwarded to make_model()
    """

    def __init__(
        self,
        db_path:           Path,
        model_name:        str   = "lgbm",
        fwd_seconds:       int   = 30,
        train_hours:       float = 4.0,
        test_minutes:      float = 30.0,
        gap_seconds:       int   = 60,
        expanding:         bool  = True,
        feature_cols:      Optional[list[str]] = None,
        save_predictions:  bool  = True,
        **model_kwargs,
    ):
        self.db_path          = Path(db_path)
        self.model_name       = model_name
        self.fwd_seconds      = fwd_seconds
        self.train_hours      = train_hours
        self.test_minutes     = test_minutes
        self.gap_seconds      = gap_seconds
        self.expanding        = expanding
        self.feature_cols     = feature_cols
        self.save_predictions = save_predictions
        self.model_kwargs     = model_kwargs

        # Loaded lazily
        self._df: Optional[pd.DataFrame] = None
        self._feature_names: list[str] = []

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_all(self) -> pd.DataFrame:
        """Load and cache the complete feature + target DataFrame."""
        if self._df is not None:
            return self._df

        min_ms, max_ms = db_time_range(self.db_path)
        if min_ms is None:
            raise RuntimeError("DB is empty — no trades found.")

        print(f"[WF] Loading data  {pd.Timestamp(min_ms, unit='ms', tz='UTC')} "
              f"→ {pd.Timestamp(max_ms, unit='ms', tz='UTC')}")

        t0 = time.perf_counter()
        df = build_features(self.db_path, min_ms, max_ms + 1)
        if df.empty:
            raise RuntimeError("Feature matrix is empty — check DB tables.")

        df = add_targets(df, fwd_seconds=self.fwd_seconds)

        # Winsorise targets (clip extreme outliers ±50 bps)
        df["target"] = df["target"].clip(-50, 50)

        # Drop rows with too many NaN features
        df = df.dropna(thresh=int(len(df.columns) * 0.7))

        elapsed = time.perf_counter() - t0
        print(f"[WF] Loaded {len(df):,} rows, {len(df.columns)} cols in {elapsed:.1f}s")

        self._df = df
        return df

    # ── Fold generation ───────────────────────────────────────────────────────

    def generate_folds(self) -> list[Fold]:
        df = self._load_all()
        t_min = df.index.min()
        t_max = df.index.max()

        train_td = pd.Timedelta(hours=self.train_hours)
        test_td  = pd.Timedelta(minutes=self.test_minutes)
        gap_td   = pd.Timedelta(seconds=self.gap_seconds)

        folds = []
        fold_id = 0
        test_start = t_min + train_td + gap_td

        while test_start + test_td <= t_max:
            train_end   = test_start - gap_td
            train_start = t_min if self.expanding else (train_end - train_td)
            test_end    = test_start + test_td

            folds.append(Fold(
                fold_id     = fold_id,
                train_start = train_start,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
            ))
            fold_id    += 1
            test_start += test_td   # roll forward by one test window

        print(f"[WF] Generated {len(folds)} folds")
        return folds

    # ── Single fold ───────────────────────────────────────────────────────────

    def _run_fold(self, fold: Fold, model: BaseModel,
                  feat_cols: list[str]) -> FoldResult:
        df = self._df

        train = df[(df.index >= fold.train_start) & (df.index < fold.train_end)]
        test  = df[(df.index >= fold.test_start)  & (df.index < fold.test_end)]

        # Drop NaN in both X and y
        train = train.dropna(subset=feat_cols + ["target"])
        test  = test.dropna(subset=feat_cols + ["target"])

        if len(train) < 60 or len(test) < 10:
            return FoldResult(
                fold_id=fold.fold_id, test_start=fold.test_start,
                test_end=fold.test_end, n_train=len(train), n_test=len(test),
                ic=float("nan"), hit=float("nan"), sharpe=float("nan"),
                fit_time_s=0.0,
            )

        X_train = train[feat_cols].copy()
        y_train = train["target"].values
        X_test  = test[feat_cols].copy()
        y_test  = test["target"].values

        t0 = time.perf_counter()
        model.fit(X_train.values, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = model.predict(X_test.values)

        ic     = spearman_ic(y_pred, y_test)
        hit    = directional_accuracy(y_pred, y_test)
        sharpe = quintile_sharpe(y_pred, y_test)

        return FoldResult(
            fold_id    = fold.fold_id,
            test_start = fold.test_start,
            test_end   = fold.test_end,
            n_train    = len(train),
            n_test     = len(test),
            ic         = ic,
            hit        = hit,
            sharpe     = sharpe,
            fit_time_s = fit_time,
            predictions = pd.Series(y_pred, index=test.index) if self.save_predictions else None,
            actuals     = pd.Series(y_test, index=test.index) if self.save_predictions else None,
        )

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        """
        Run the full walk-forward loop.

        Returns
        -------
        results_df     : DataFrame with one row per fold (ic, hit, sharpe …)
        feature_names  : list of feature column names used
        importances    : mean feature importances across all folds (LightGBM)
                         or absolute coefficients (Ridge); shape (n_features,)
        """
        df     = self._load_all()
        folds  = self.generate_folds()

        # Determine feature columns
        _exclude = {"target", "direction", "mid_price", "vwap", "volume",
                    "count", "close", "open", "high", "low"}
        if self.feature_cols:
            feat_cols = [c for c in self.feature_cols if c in df.columns]
        else:
            feat_cols = [c for c in df.columns if c not in _exclude]

        self._feature_names = feat_cols
        print(f"[WF] Features ({len(feat_cols)}): {feat_cols}")

        results    = []
        imp_accum  = np.zeros(len(feat_cols))
        imp_count  = 0

        pbar = tqdm(folds, desc="Walk-Forward", unit="fold")
        for fold in pbar:
            model  = make_model(self.model_name, **self.model_kwargs)
            result = self._run_fold(fold, model, feat_cols)
            results.append(result)

            # Accumulate feature importances
            fi = model.feature_importances
            if fi is not None and len(fi) == len(feat_cols):
                imp_accum += fi
                imp_count += 1

            pbar.set_postfix({
                "IC":  f"{result.ic:.4f}" if not np.isnan(result.ic) else "nan",
                "hit": f"{result.hit*100:.1f}%" if not np.isnan(result.hit) else "nan",
            })

        results_df = pd.DataFrame([
            {
                "fold_id":    r.fold_id,
                "test_start": r.test_start,
                "test_end":   r.test_end,
                "n_train":    r.n_train,
                "n_test":     r.n_test,
                "ic":         r.ic,
                "hit":        r.hit,
                "sharpe":     r.sharpe,
                "fit_time_s": r.fit_time_s,
            }
            for r in results
        ])

        mean_imp = imp_accum / max(imp_count, 1)
        return results_df, feat_cols, mean_imp
