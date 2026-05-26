#!/usr/bin/env python3
"""
models.py
=========
Ridge regression（baseline）和 LightGBM 的統一介面。
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


# ── Base interface ─────────────────────────────────────────────────────────────

class BaseModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        return None


# ── Ridge (linear baseline) ───────────────────────────────────────────────────

class RidgeModel(BaseModel):
    """
    RobustScaler + Ridge regression。
    alpha 建議值：1.0（可調）。
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("ridge",  Ridge(alpha=alpha, fit_intercept=True)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeModel":
        self._pipe.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipe.predict(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        # Return absolute ridge coefficients (after scaling, for relative comparison)
        coef = self._pipe.named_steps["ridge"].coef_
        return np.abs(coef)


# ── LightGBM ──────────────────────────────────────────────────────────────────

class LGBMModel(BaseModel):
    """
    LightGBM Regressor，包含基本防 overfitting 設定。
    使用 early stopping（需要 validation set）或固定 n_estimators。
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        learning_rate: float = 0.05,
        max_depth:     int   = 5,
        num_leaves:    int   = 31,
        min_child_samples: int = 20,
        subsample:     float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha:     float = 0.1,
        reg_lambda:    float = 1.0,
        n_jobs:        int   = -1,
        verbose:       int   = -1,
    ):
        import lightgbm as lgb
        self._params = dict(
            n_estimators      = n_estimators,
            learning_rate     = learning_rate,
            max_depth         = max_depth,
            num_leaves        = num_leaves,
            min_child_samples = min_child_samples,
            subsample         = subsample,
            colsample_bytree  = colsample_bytree,
            reg_alpha         = reg_alpha,
            reg_lambda        = reg_lambda,
            n_jobs            = n_jobs,
            verbose           = verbose,
        )
        self._model = lgb.LGBMRegressor(**self._params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> "LGBMModel":
        if X_val is not None and y_val is not None:
            self._model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                callbacks=[__import__("lightgbm").early_stopping(20, verbose=False),
                           __import__("lightgbm").log_evaluation(period=-1)],
            )
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(name: str, **kwargs) -> BaseModel:
    """
    Factory.

    Parameters
    ----------
    name : "ridge" | "lgbm"
    """
    name = name.lower()
    if name == "ridge":
        return RidgeModel(**kwargs)
    elif name in ("lgbm", "lightgbm"):
        return LGBMModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose 'ridge' or 'lgbm'.")
