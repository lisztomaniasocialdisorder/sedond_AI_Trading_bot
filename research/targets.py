#!/usr/bin/env python3
"""
targets.py
==========
計算 forward return targets，供 walk-forward 模型使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_targets(df: pd.DataFrame,
                    fwd_seconds: int = 30,
                    mid_col: str = "mid_price") -> pd.Series:
    """
    30-second forward return in basis points.

    target[t] = (mid[t + fwd_seconds] - mid[t]) / mid[t] * 10_000

    Parameters
    ----------
    df          : feature DataFrame with 1-second index
    fwd_seconds : prediction horizon in seconds
    mid_col     : column name for mid price

    Returns
    -------
    pd.Series  named 'target', aligned with df.index.
                NaN for the last `fwd_seconds` rows (no future data).
    """
    if mid_col not in df.columns:
        raise ValueError(f"Column '{mid_col}' not found in DataFrame.")

    mid = df[mid_col]
    fwd = mid.shift(-fwd_seconds)                        # future price
    target = (fwd - mid) / mid * 10_000                 # bps
    target.name = "target"
    return target


def compute_direction(target: pd.Series,
                      threshold_bps: float = 0.5) -> pd.Series:
    """
    Discretise forward return into direction classes.

    Returns
    -------
    pd.Series of int8:  +1 (Up), 0 (Flat), -1 (Down)
    """
    direction = pd.Series(0, index=target.index, dtype="int8", name="direction")
    direction[target >  threshold_bps] =  1
    direction[target < -threshold_bps] = -1
    return direction


def add_targets(df: pd.DataFrame,
                fwd_seconds: int = 30,
                threshold_bps: float = 0.5) -> pd.DataFrame:
    """
    Convenience: attach target + direction columns to the feature DataFrame.
    Rows with NaN target (last fwd_seconds rows) are dropped.
    """
    df = df.copy()
    df["target"]    = compute_targets(df, fwd_seconds)
    df["direction"] = compute_direction(df["target"], threshold_bps)
    return df.dropna(subset=["target"])
