#!/usr/bin/env python3
from __future__ import annotations

"""
DL ==> RF ==> A/B 雙族群混戰 GA ==> 負和市場 Nested Walk-forward Arena

重點：
1. 不再讓 GA 直接在 final test 上演化。
2. 每個 fold 切成四段：
   train     : 訓練 DL
   val       : RF 訓練 + DL early stopping
   ga_arena  : A/B GA 混戰演化
   final_test: 完全沒看過，用冠軍策略最後驗證
3. 負和市場：
   - 初始資金 INITIAL_CAPITAL
   - 資金低於 DEATH_CAPITAL 死亡
   - 每根 bar 收生活費
   - 每筆交易收固定費 + 比例手續費
   - 持倉太短 / 太長罰金
   - 統計有效交易 / 無效交易
4. 停止方式：
   - 在 export 目錄建立 STOP 檔案即可安全停止
   - 每個 fold 後會自動保存 .json / .pkl / parquet
5. 輸出：
   - walkforward_summary.parquet / .json
   - walkforward_trades.parquet
   - generation_battle_log.parquet
   - equity_curves.parquet
   - walkforward_state.pkl
   - meta.json

Colab 執行：
    from google.colab import drive
    drive.mount("/content/drive")
    %run /content/drive/MyDrive/klinetraning/notebooks/dl_rf_ga_nested_negative_sum_wf.py
"""

import gc
import json
import pickle
import random
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Config
# ============================================================

BASE = Path("/content/drive/MyDrive/klinetraning")
FEATURE_PATH = BASE / "features" / "BTCUSDT_30m_144000_features.parquet"

EXPORT_DIR = BASE / "exports" / "dl_rf_ga_nested_negative_sum_wf"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_SUMMARY = EXPORT_DIR / "walkforward_summary.parquet"
EXPORT_SUMMARY_JSON = EXPORT_DIR / "walkforward_summary.json"
EXPORT_TRADES = EXPORT_DIR / "walkforward_trades.parquet"
EXPORT_BATTLE_LOG = EXPORT_DIR / "generation_battle_log.parquet"
EXPORT_EQUITY = EXPORT_DIR / "equity_curves.parquet"
EXPORT_STATE = EXPORT_DIR / "walkforward_state.pkl"
EXPORT_META = EXPORT_DIR / "meta.json"
STOP_FILE = EXPORT_DIR / "STOP"

SEED = 79

# Walk-forward
N_FOLDS = 6
TRAIN_RATIO_START = 0.42
TRAIN_RATIO_STEP = 0.07
VAL_RATIO = 0.10
GA_RATIO = 0.10
TEST_RATIO = 0.10

# DL
DL_EPOCHS = 60
DL_BATCH_SIZE = 512
DL_EMBED_DIM = 32

# RF
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 30

# GA
POP_SIZE_PER_COLONY = 160
GENERATIONS = 12
TOP_WINNERS = 10
ELITE_PER_COLONY = 4
MUTATION_RATE = 0.30

# Negative-sum market
INITIAL_CAPITAL = 10_000.0
DEATH_CAPITAL = 5_000.0
POSITION_FRACTION = 0.95

FEE_RATE = 0.0005
FIXED_FEE_PER_TRADE = 0.20
LIVING_COST_PER_BAR = 0.02

MIN_VALID_TRADES = 30
MIN_ABS_GROSS_RETURN_VALID = 0.0003
LOW_TRADE_PENALTY = 4000.0
DEATH_PENALTY = 1_000_000.0

MIN_HOLD_BARS = 2
MAX_HOLD_BARS = 16
SHORT_HOLD_PENALTY = 2.0
LONG_HOLD_PENALTY = 1.0

TIME_COL = "timestamp"
LABEL_COL = "target_up"
RETURN_COL = "next_return"

BASE_DROP_COLS = {
    TIME_COL,
    LABEL_COL,
    RETURN_COL,
    "timestamp_dt",
    "future_return",
    "label",
    "target",
    "y",
}


# ============================================================
# Data classes
# ============================================================

@dataclass
class FoldInfo:
    fold: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    ga_start: int
    ga_end: int
    test_start: int
    test_end: int


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def reset_tf() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def log(msg: str) -> None:
    print(msg, flush=True)


def should_stop() -> bool:
    return STOP_FILE.exists()


def safe_numeric_series(df: pd.DataFrame, col: str | None, default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def calc_max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_partial_state(
    all_summary: list[dict],
    all_trades: list[pd.DataFrame],
    all_equity: list[pd.DataFrame],
    all_battle_logs: list[pd.DataFrame],
    meta_extra: dict | None = None,
) -> None:
    summary_df = pd.DataFrame(all_summary)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    battle_df = pd.concat(all_battle_logs, ignore_index=True) if all_battle_logs else pd.DataFrame()

    summary_df.to_parquet(EXPORT_SUMMARY, index=False, compression="snappy")
    trades_df.to_parquet(EXPORT_TRADES, index=False, compression="snappy")
    equity_df.to_parquet(EXPORT_EQUITY, index=False, compression="snappy")
    battle_df.to_parquet(EXPORT_BATTLE_LOG, index=False, compression="snappy")

    EXPORT_SUMMARY_JSON.write_text(
        json.dumps(to_jsonable(all_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state = {
        "summary": all_summary,
        "trades_rows": int(len(trades_df)),
        "equity_rows": int(len(equity_df)),
        "battle_log_rows": int(len(battle_df)),
        "meta_extra": meta_extra or {},
    }
    with open(EXPORT_STATE, "wb") as f:
        pickle.dump(state, f)

    log(f"[保存] 已寫出 summary/json/pkl/parquet 到：{EXPORT_DIR}")


# ============================================================
# Load data
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"找不到特徵檔案: {FEATURE_PATH}")

    df = pd.read_parquet(FEATURE_PATH).copy()

    for col in [TIME_COL, LABEL_COL, RETURN_COL]:
        if col not in df.columns:
            raise ValueError(f"缺少欄位 {col}，目前欄位：{df.columns.tolist()}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df[RETURN_COL] = pd.to_numeric(df[RETURN_COL], errors="coerce")

    df = df.dropna(subset=[TIME_COL, LABEL_COL, RETURN_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    y = df[LABEL_COL].astype(int).to_numpy()

    leak_keywords = [
        "future",
        "next_return",
        "target",
        "label",
        "y_true",
        "pred",
    ]

    feature_cols = []
    for c in df.columns:
        if c in BASE_DROP_COLS:
            continue
        lc = c.lower()
        if any(k in lc for k in leak_keywords):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if not feature_cols:
        raise ValueError("沒有找到可用 numeric feature 欄位。")

    x = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    return df, x, y, feature_cols


def make_walkforward_splits(n: int) -> list[FoldInfo]:
    folds: list[FoldInfo] = []

    for fold in range(N_FOLDS):
        train_end = int(n * (TRAIN_RATIO_START + fold * TRAIN_RATIO_STEP))
        val_end = int(train_end + n * VAL_RATIO)
        ga_end = int(val_end + n * GA_RATIO)
        test_end = int(ga_end + n * TEST_RATIO)

        if test_end > n:
            break

        folds.append(
            FoldInfo(
                fold=fold,
                train_start=0,
                train_end=train_end,
                val_start=train_end,
                val_end=val_end,
                ga_start=val_end,
                ga_end=ga_end,
                test_start=ga_end,
                test_end=test_end,
            )
        )

    return folds


# ============================================================
# DL
# ============================================================

def build_dl_model(n_features: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(n_features,), name="features")

    x = tf.keras.layers.Dense(256, activation="gelu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Dense(64, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)

    emb = tf.keras.layers.Dense(DL_EMBED_DIM, activation="linear", name="embedding")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="p_long")(emb)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_dl(x: np.ndarray, y: np.ndarray, fold: FoldInfo) -> dict:
    reset_tf()

    x_train = x[fold.train_start:fold.train_end]
    y_train = y[fold.train_start:fold.train_end]

    x_val = x[fold.val_start:fold.val_end]
    y_val = y[fold.val_start:fold.val_end]

    x_ga = x[fold.ga_start:fold.ga_end]
    y_ga = y[fold.ga_start:fold.ga_end]

    x_test = x[fold.test_start:fold.test_end]
    y_test = y[fold.test_start:fold.test_end]

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_ga_s = scaler.transform(x_ga)
    x_test_s = scaler.transform(x_test)

    classes = np.unique(y_train)
    class_weight = None
    if len(classes) == 2:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight = {int(cls): float(w) for cls, w in zip(classes, weights)}

    model = build_dl_model(x_train_s.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
    ]

    log("  [DL] 訓練 deep learning（深度學習）模型。")
    model.fit(
        x_train_s,
        y_train,
        validation_data=(x_val_s, y_val),
        epochs=DL_EPOCHS,
        batch_size=DL_BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )

    p_val = model.predict(x_val_s, verbose=0).reshape(-1)
    p_ga = model.predict(x_ga_s, verbose=0).reshape(-1)
    p_test = model.predict(x_test_s, verbose=0).reshape(-1)

    pred_val = (p_val >= 0.5).astype(int)
    pred_ga = (p_ga >= 0.5).astype(int)
    pred_test = (p_test >= 0.5).astype(int)

    encoder = tf.keras.Model(inputs=model.input, outputs=model.get_layer("embedding").output)

    emb_val = encoder.predict(x_val_s, verbose=0)
    emb_ga = encoder.predict(x_ga_s, verbose=0)
    emb_test = encoder.predict(x_test_s, verbose=0)

    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "p_val": p_val,
        "p_ga": p_ga,
        "p_test": p_test,
        "pred_val": pred_val,
        "pred_ga": pred_ga,
        "pred_test": pred_test,
        "emb_val": emb_val,
        "emb_ga": emb_ga,
        "emb_test": emb_test,
        "metrics": {
            "dl_val_accuracy": float(accuracy_score(y_val, pred_val)),
            "dl_val_balanced_accuracy": float(balanced_accuracy_score(y_val, pred_val)),
            "dl_ga_accuracy": float(accuracy_score(y_ga, pred_ga)),
            "dl_ga_balanced_accuracy": float(balanced_accuracy_score(y_ga, pred_ga)),
            "dl_test_accuracy": float(accuracy_score(y_test, pred_test)),
            "dl_test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        },
    }


# ============================================================
# RF
# ============================================================

def build_rf_features(raw_x: np.ndarray, dl_p: np.ndarray, emb: np.ndarray) -> np.ndarray:
    dl_p = dl_p.reshape(-1, 1)
    p_short = (1.0 - dl_p).reshape(-1, 1)
    out = np.concatenate([raw_x, dl_p, p_short, emb], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def train_rf(x: np.ndarray, y: np.ndarray, fold: FoldInfo, dl_result: dict) -> dict:
    x_val_raw = x[fold.val_start:fold.val_end]
    y_val = y[fold.val_start:fold.val_end]

    x_ga_raw = x[fold.ga_start:fold.ga_end]
    y_ga = y[fold.ga_start:fold.ga_end]

    x_test_raw = x[fold.test_start:fold.test_end]
    y_test = y[fold.test_start:fold.test_end]

    rf_x_train = build_rf_features(x_val_raw, dl_result["p_val"], dl_result["emb_val"])
    rf_x_ga = build_rf_features(x_ga_raw, dl_result["p_ga"], dl_result["emb_ga"])
    rf_x_test = build_rf_features(x_test_raw, dl_result["p_test"], dl_result["emb_test"])

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )

    log("  [RF] 訓練 Random Forest（隨機森林），吃原始特徵 + DL 輸出。")
    rf.fit(rf_x_train, y_val)

    p_ga = rf.predict_proba(rf_x_ga)[:, 1]
    pred_ga = (p_ga >= 0.5).astype(int)

    p_test = rf.predict_proba(rf_x_test)[:, 1]
    pred_test = (p_test >= 0.5).astype(int)

    return {
        "model": rf,
        "p_ga": p_ga,
        "pred_ga": pred_ga,
        "p_test": p_test,
        "pred_test": pred_test,
        "metrics": {
            "rf_ga_accuracy": float(accuracy_score(y_ga, pred_ga)),
            "rf_ga_balanced_accuracy": float(balanced_accuracy_score(y_ga, pred_ga)),
            "rf_test_accuracy": float(accuracy_score(y_test, pred_test)),
            "rf_test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
        },
    }


# ============================================================
# Arena
# ============================================================

def build_arena(
    df: pd.DataFrame,
    start: int,
    end: int,
    dl_p: np.ndarray,
    dl_pred: np.ndarray,
    rf_p: np.ndarray,
    rf_pred: np.ndarray,
) -> pd.DataFrame:
    part = df.iloc[start:end].reset_index(drop=True).copy()

    obi_col = find_col(part, ["obi", "OBI", "orderbook_imbalance", "imbalance", "book_imbalance", "obi_1"])
    vol_col = find_col(part, ["volatility", "realized_volatility", "rv", "ret_std", "rolling_vol", "vol_30m"])
    momentum_col = find_col(part, ["momentum", "ret_1", "return_1", "logret_1", "close_return", "price_change"])

    arena = pd.DataFrame({
        "timestamp": part[TIME_COL],
        "future_return": safe_numeric_series(part, RETURN_COL),
        "y_true": part[LABEL_COL].astype(int).to_numpy(),
        "dl_p_long": dl_p.astype(float),
        "dl_pred": dl_pred.astype(int),
        "rf_p_long": rf_p.astype(float),
        "rf_pred": rf_pred.astype(int),
        "obi": safe_numeric_series(part, obi_col),
        "volatility": safe_numeric_series(part, vol_col),
        "momentum": safe_numeric_series(part, momentum_col),
    })

    return arena.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ============================================================
# GA
# ============================================================

GENE_KEYS = [
    "dl_th",
    "rf_th",
    "obi_th",
    "vol_max",
    "mom_th",
    "use_obi",
    "use_vol",
    "use_mom",
    "hold_bars",
]


def random_gene(colony: str) -> dict:
    if colony == "A":
        return {
            "colony": "A",
            "dl_th": random.uniform(0.52, 0.86),
            "rf_th": random.uniform(0.52, 0.86),
            "obi_th": random.uniform(-0.25, 0.25),
            "vol_max": random.uniform(0.0, 0.08),
            "mom_th": random.uniform(-0.01, 0.01),
            "use_obi": random.choice([0, 1]),
            "use_vol": random.choice([0, 1]),
            "use_mom": random.choice([0, 1]),
            "hold_bars": random.randint(1, 24),
        }

    return {
        "colony": "B",
        "dl_th": random.uniform(0.60, 0.92),
        "rf_th": random.uniform(0.60, 0.92),
        "obi_th": random.uniform(-0.15, 0.30),
        "vol_max": random.uniform(0.0, 0.04),
        "mom_th": random.uniform(-0.005, 0.015),
        "use_obi": random.choice([0, 1]),
        "use_vol": random.choice([0, 1]),
        "use_mom": random.choice([0, 1]),
        "hold_bars": random.randint(3, 36),
    }


def clamp_gene(g: dict) -> dict:
    g = dict(g)
    g["dl_th"] = float(np.clip(g["dl_th"], 0.50, 0.95))
    g["rf_th"] = float(np.clip(g["rf_th"], 0.50, 0.95))
    g["obi_th"] = float(np.clip(g["obi_th"], -0.40, 0.40))
    g["vol_max"] = float(np.clip(g["vol_max"], 0.0, 0.10))
    g["mom_th"] = float(np.clip(g["mom_th"], -0.05, 0.05))
    g["hold_bars"] = int(np.clip(g["hold_bars"], 1, 48))
    g["use_obi"] = int(g["use_obi"] >= 0.5)
    g["use_vol"] = int(g["use_vol"] >= 0.5)
    g["use_mom"] = int(g["use_mom"] >= 0.5)
    if g.get("colony") not in ["A", "B", "MIX"]:
        g["colony"] = "MIX"
    return g


def evaluate_gene(
    arena: pd.DataFrame,
    g: dict,
    style: str,
    return_details: bool = False,
) -> dict | tuple[dict, pd.DataFrame, pd.DataFrame]:
    cond = (
        (arena["dl_p_long"] >= g["dl_th"])
        &
        (arena["rf_p_long"] >= g["rf_th"])
    )

    if g["use_obi"]:
        cond &= arena["obi"] >= g["obi_th"]
    if g["use_vol"]:
        cond &= arena["volatility"] <= g["vol_max"]
    if g["use_mom"]:
        cond &= arena["momentum"] >= g["mom_th"]

    signal_set = set(np.where(cond.to_numpy())[0].tolist())

    capital = INITIAL_CAPITAL
    equity_records = []
    trade_records = []

    valid_trade_count = 0
    invalid_trade_count = 0
    short_hold_count = 0
    long_hold_count = 0
    dead = False

    hold_bars = int(g["hold_bars"])
    next_available_idx = 0

    returns = arena["future_return"].to_numpy(dtype=float)
    timestamps = arena["timestamp"].to_numpy()

    for i in range(len(arena)):
        capital -= LIVING_COST_PER_BAR

        if capital < DEATH_CAPITAL:
            dead = True
            equity_records.append({"bar_index": i, "timestamp": timestamps[i], "capital": capital, "dead": 1})
            break

        if i < next_available_idx or i not in signal_set:
            equity_records.append({"bar_index": i, "timestamp": timestamps[i], "capital": capital, "dead": 0})
            continue

        exit_i = min(i + hold_bars, len(arena) - 1)
        actual_hold_bars = exit_i - i + 1
        gross_return = float(np.sum(returns[i:exit_i + 1]))

        position_size = capital * POSITION_FRACTION
        fee = position_size * FEE_RATE + FIXED_FEE_PER_TRADE
        pnl = position_size * gross_return - fee

        hold_penalty = 0.0
        if actual_hold_bars < MIN_HOLD_BARS:
            hold_penalty += SHORT_HOLD_PENALTY
            short_hold_count += 1
        if actual_hold_bars > MAX_HOLD_BARS:
            hold_penalty += LONG_HOLD_PENALTY * (actual_hold_bars - MAX_HOLD_BARS)
            long_hold_count += 1

        pnl_after_penalty = pnl - hold_penalty
        capital += pnl_after_penalty
        net_return = pnl_after_penalty / max(position_size, 1e-9)

        is_valid = int(abs(gross_return) >= MIN_ABS_GROSS_RETURN_VALID and pnl_after_penalty > 0)
        if is_valid:
            valid_trade_count += 1
        else:
            invalid_trade_count += 1

        trade_records.append({
            "entry_index": i,
            "exit_index": exit_i,
            "entry_timestamp": timestamps[i],
            "exit_timestamp": timestamps[exit_i],
            "hold_bars": actual_hold_bars,
            "gross_return": gross_return,
            "position_size": position_size,
            "fee": fee,
            "hold_penalty": hold_penalty,
            "pnl": pnl_after_penalty,
            "net_return": net_return,
            "capital_after": capital,
            "valid_trade": is_valid,
            "dl_p_long": float(arena["dl_p_long"].iloc[i]),
            "rf_p_long": float(arena["rf_p_long"].iloc[i]),
            "obi": float(arena["obi"].iloc[i]),
            "volatility": float(arena["volatility"].iloc[i]),
            "momentum": float(arena["momentum"].iloc[i]),
        })

        next_available_idx = exit_i + 1
        equity_records.append({"bar_index": i, "timestamp": timestamps[i], "capital": capital, "dead": 0})

        if capital < DEATH_CAPITAL:
            dead = True
            break

    if not equity_records:
        equity_records.append({
            "bar_index": 0,
            "timestamp": arena["timestamp"].iloc[0] if len(arena) else None,
            "capital": INITIAL_CAPITAL,
            "dead": 0,
        })

    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trade_records)

    equity = equity_df["capital"].to_numpy(dtype=float)
    final_capital = float(equity[-1])
    total_profit = final_capital - INITIAL_CAPITAL
    max_drawdown = calc_max_drawdown(equity)

    trade_count = valid_trade_count + invalid_trade_count

    if trade_count > 0:
        winrate = float((trades_df["pnl"] > 0).mean())
        avg_trade_return = float(trades_df["net_return"].mean())
    else:
        winrate = 0.0
        avg_trade_return = 0.0

    valid_ratio = valid_trade_count / max(trade_count, 1)

    low_trade_penalty = 0.0
    if valid_trade_count < MIN_VALID_TRADES:
        low_trade_penalty = LOW_TRADE_PENALTY * (1.0 - valid_trade_count / MIN_VALID_TRADES)

    death_penalty = DEATH_PENALTY if dead else 0.0

    if style == "A":
        fitness = (
            total_profit
            + valid_trade_count * 12.0
            + winrate * 600.0
            + valid_ratio * 400.0
            - abs(max_drawdown) * 0.8
            - invalid_trade_count * 25.0
            - short_hold_count * 40.0
            - long_hold_count * 25.0
            - low_trade_penalty
            - death_penalty
        )
    elif style == "B":
        fitness = (
            total_profit
            + valid_trade_count * 8.0
            + winrate * 900.0
            + valid_ratio * 600.0
            - abs(max_drawdown) * 1.5
            - invalid_trade_count * 35.0
            - short_hold_count * 60.0
            - long_hold_count * 40.0
            - low_trade_penalty
            - death_penalty
        )
    else:
        fitness = (
            total_profit
            + valid_trade_count * 10.0
            + winrate * 750.0
            + valid_ratio * 500.0
            - abs(max_drawdown) * 1.2
            - invalid_trade_count * 30.0
            - short_hold_count * 50.0
            - long_hold_count * 35.0
            - low_trade_penalty
            - death_penalty
        )

    result = {
        **g,
        "style": style,
        "fitness": float(fitness),
        "final_capital": final_capital,
        "total_profit": float(total_profit),
        "dead": int(dead),
        "trade_count": int(trade_count),
        "valid_trade_count": int(valid_trade_count),
        "invalid_trade_count": int(invalid_trade_count),
        "valid_ratio": float(valid_ratio),
        "winrate": float(winrate),
        "avg_trade_return": float(avg_trade_return),
        "max_drawdown": float(max_drawdown),
        "short_hold_count": int(short_hold_count),
        "long_hold_count": int(long_hold_count),
        "low_trade_penalty": float(low_trade_penalty),
        "death_penalty": float(death_penalty),
    }

    if return_details:
        return result, trades_df, equity_df
    return result


def crossover(a: dict, b: dict, child_colony: str) -> dict:
    child = {"colony": child_colony}
    for k in GENE_KEYS:
        child[k] = a[k] if random.random() < 0.5 else b[k]
    return clamp_gene(child)


def mutate(g: dict) -> dict:
    g = dict(g)

    if random.random() < MUTATION_RATE:
        g["dl_th"] += random.gauss(0, 0.035)
    if random.random() < MUTATION_RATE:
        g["rf_th"] += random.gauss(0, 0.035)
    if random.random() < MUTATION_RATE:
        g["obi_th"] += random.gauss(0, 0.045)
    if random.random() < MUTATION_RATE:
        g["vol_max"] += random.gauss(0, 0.008)
    if random.random() < MUTATION_RATE:
        g["mom_th"] += random.gauss(0, 0.006)
    if random.random() < MUTATION_RATE:
        g["hold_bars"] += int(round(random.gauss(0, 4)))
    if random.random() < MUTATION_RATE:
        g["use_obi"] = 1 - int(g["use_obi"])
    if random.random() < MUTATION_RATE:
        g["use_vol"] = 1 - int(g["use_vol"])
    if random.random() < MUTATION_RATE:
        g["use_mom"] = 1 - int(g["use_mom"])

    return clamp_gene(g)


def pick_parent(winners: list[dict]) -> dict:
    if not winners:
        raise ValueError("沒有 winners 可以交配。這族群已經絕種了。")
    ranks = np.arange(len(winners), 0, -1, dtype=float)
    probs = ranks / ranks.sum()
    idx = np.random.choice(len(winners), p=probs)
    return winners[int(idx)]


def summarize_best(row: dict) -> str:
    return (
        f"{row['colony']}族 "
        f"fitness={row['fitness']:.2f} "
        f"資金={row.get('final_capital', 0):.2f} "
        f"損益={row.get('total_profit', 0):.2f} "
        f"死亡={int(row.get('dead', 0))} "
        f"交易={int(row.get('trade_count', 0))} "
        f"有效={int(row.get('valid_trade_count', 0))} "
        f"無效={int(row.get('invalid_trade_count', 0))} "
        f"勝率={row.get('winrate', 0):.2%} "
        f"DD={row.get('max_drawdown', 0):.2f} "
        f"hold={int(row.get('hold_bars', 0))} "
        f"DL>{row['dl_th']:.3f} "
        f"RF>{row['rf_th']:.3f}"
    )


def run_dual_colony_battle_ga(arena: pd.DataFrame, fold_id: int) -> tuple[dict, pd.DataFrame]:
    pop_a = [random_gene("A") for _ in range(POP_SIZE_PER_COLONY)]
    pop_b = [random_gene("B") for _ in range(POP_SIZE_PER_COLONY)]

    logs = []

    for gen in range(GENERATIONS):
        if should_stop():
            log("[停止] 偵測到 STOP 檔案，GA 將提前停止並保存目前最佳結果。")
            break

        scored_a = [evaluate_gene(arena, g, "A") for g in pop_a]
        scored_b = [evaluate_gene(arena, g, "B") for g in pop_b]

        scored_a = sorted(scored_a, key=lambda x: x["fitness"], reverse=True)
        scored_b = sorted(scored_b, key=lambda x: x["fitness"], reverse=True)

        battle_pool_raw = scored_a + scored_b
        battle_scored = [evaluate_gene(arena, g, "BATTLE") for g in battle_pool_raw]
        battle_scored = sorted(battle_scored, key=lambda x: x["fitness"], reverse=True)

        winners = battle_scored[:TOP_WINNERS]

        best_a = scored_a[0]
        best_b = scored_b[0]
        battle_champ = winners[0]

        a_in_winners = sum(1 for w in winners if w["colony"] == "A")
        b_in_winners = sum(1 for w in winners if w["colony"] == "B")

        log(
            f"  [第 {gen:02d} 代戰況] "
            f"混戰冠軍：{summarize_best(battle_champ)} | "
            f"前{TOP_WINNERS}名：A族 {a_in_winners} 個、B族 {b_in_winners} 個"
        )
        log(f"      A族最佳：{summarize_best(best_a)}")
        log(f"      B族最佳：{summarize_best(best_b)}")

        for rank, row in enumerate(battle_scored):
            rec = dict(row)
            rec["fold"] = fold_id
            rec["generation"] = gen
            rec["battle_rank"] = rank + 1
            rec["is_winner"] = int(rank < TOP_WINNERS)
            logs.append(rec)

        next_a = [{k: elite[k] for k in ["colony"] + GENE_KEYS} for elite in scored_a[:ELITE_PER_COLONY]]
        next_b = [{k: elite[k] for k in ["colony"] + GENE_KEYS} for elite in scored_b[:ELITE_PER_COLONY]]

        while len(next_a) < POP_SIZE_PER_COLONY:
            p1 = pick_parent(winners)
            p2 = pick_parent(winners)
            next_a.append(mutate(crossover(p1, p2, "A")))

        while len(next_b) < POP_SIZE_PER_COLONY:
            p1 = pick_parent(winners)
            p2 = pick_parent(winners)
            next_b.append(mutate(crossover(p1, p2, "B")))

        pop_a = next_a
        pop_b = next_b

    final_pool = pop_a + pop_b
    final_battle = [evaluate_gene(arena, g, "BATTLE") for g in final_pool]
    final_battle = sorted(final_battle, key=lambda x: x["fitness"], reverse=True)

    champion = final_battle[0]
    log(f"  [FOLD {fold_id}] GA Arena 最終總冠軍：{summarize_best(champion)}")

    return champion, pd.DataFrame(logs)


def apply_strategy(arena: pd.DataFrame, gene: dict, fold_id: int, arena_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    result, trades, equity = evaluate_gene(arena, gene, "BATTLE", return_details=True)

    trades = trades.copy()
    equity = equity.copy()

    if len(trades) > 0:
        trades["fold"] = fold_id
        trades["arena"] = arena_name
        trades["signal"] = 1
        for k in GENE_KEYS:
            trades[k] = gene[k]
        trades["colony"] = gene["colony"]

    if len(equity) > 0:
        equity["fold"] = fold_id
        equity["arena"] = arena_name
        equity["colony"] = gene["colony"]

    return trades, equity, result


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(SEED)

    if STOP_FILE.exists():
        log(f"[提醒] STOP 檔案已存在：{STOP_FILE}")
        log("[提醒] 如果你要正常開始，請先刪掉 STOP 檔案。")

    log("=" * 78)
    log("DL ==> RF ==> A/B 雙族群混戰 GA ==> 負和市場 Nested Walk-forward Arena")
    log("=" * 78)

    df, x, y, feature_cols = load_dataset()
    folds = make_walkforward_splits(len(df))

    log(f"[資料] rows={len(df)}, features={len(feature_cols)}")
    log(f"[資料] time_col={TIME_COL}, label_col={LABEL_COL}, return_col={RETURN_COL}")
    log(f"[Walk-forward] folds={len(folds)}")
    log(f"[Split] train -> val/RF -> GA arena -> final test")
    log(f"[GA] 每族族群={POP_SIZE_PER_COLONY}, 代數={GENERATIONS}, 每代取前{TOP_WINNERS}名交配")
    log(f"[負和市場] 初始資金={INITIAL_CAPITAL}, 死亡資金={DEATH_CAPITAL}, 每bar生活費={LIVING_COST_PER_BAR}")
    log(f"[停止] 建立此檔可安全停止：{STOP_FILE}")
    log("=" * 78)

    all_summary = []
    all_trades = []
    all_equity = []
    all_battle_logs = []

    try:
        for fold in folds:
            if should_stop():
                log("[停止] 偵測到 STOP 檔案，停止進入下一個 fold。")
                break

            log("\n" + "=" * 78)
            log(f"[FOLD {fold.fold}] 區間：{asdict(fold)}")
            log("=" * 78)

            dl_result = train_dl(x, y, fold)
            rf_result = train_rf(x, y, fold, dl_result)

            ga_arena = build_arena(
                df,
                fold.ga_start,
                fold.ga_end,
                dl_result["p_ga"],
                dl_result["pred_ga"],
                rf_result["p_ga"],
                rf_result["pred_ga"],
            )

            test_arena = build_arena(
                df,
                fold.test_start,
                fold.test_end,
                dl_result["p_test"],
                dl_result["pred_test"],
                rf_result["p_test"],
                rf_result["pred_test"],
            )

            log(f"  [GA Arena] rows={len(ga_arena)}")
            log(f"  [Final Test] rows={len(test_arena)}")
            log(
                f"  [模型分數] "
                f"DL ga_bacc={dl_result['metrics']['dl_ga_balanced_accuracy']:.4f}, "
                f"RF ga_bacc={rf_result['metrics']['rf_ga_balanced_accuracy']:.4f}, "
                f"DL test_bacc={dl_result['metrics']['dl_test_balanced_accuracy']:.4f}, "
                f"RF test_bacc={rf_result['metrics']['rf_test_balanced_accuracy']:.4f}"
            )

            champion, battle_log = run_dual_colony_battle_ga(ga_arena, fold.fold)

            ga_trades, ga_equity, ga_realized = apply_strategy(ga_arena, champion, fold.fold, "ga_arena")
            test_trades, test_equity, test_realized = apply_strategy(test_arena, champion, fold.fold, "final_test")

            summary = {
                "fold": fold.fold,
                **asdict(fold),
                **dl_result["metrics"],
                **rf_result["metrics"],

                "champion_colony": champion["colony"],
                "champion_fitness_on_ga": champion["fitness"],
                "champion_final_capital_on_ga": champion["final_capital"],
                "champion_total_profit_on_ga": champion["total_profit"],
                "champion_dead_on_ga": champion["dead"],
                "champion_trade_count_on_ga": champion["trade_count"],
                "champion_valid_trade_count_on_ga": champion["valid_trade_count"],
                "champion_invalid_trade_count_on_ga": champion["invalid_trade_count"],
                "champion_winrate_on_ga": champion["winrate"],
                "champion_max_drawdown_on_ga": champion["max_drawdown"],

                "test_final_capital": test_realized["final_capital"],
                "test_total_profit": test_realized["total_profit"],
                "test_dead": test_realized["dead"],
                "test_trade_count": test_realized["trade_count"],
                "test_valid_trade_count": test_realized["valid_trade_count"],
                "test_invalid_trade_count": test_realized["invalid_trade_count"],
                "test_valid_ratio": test_realized["valid_ratio"],
                "test_winrate": test_realized["winrate"],
                "test_max_drawdown": test_realized["max_drawdown"],

                "best_dl_th": champion["dl_th"],
                "best_rf_th": champion["rf_th"],
                "best_obi_th": champion["obi_th"],
                "best_vol_max": champion["vol_max"],
                "best_mom_th": champion["mom_th"],
                "best_use_obi": champion["use_obi"],
                "best_use_vol": champion["use_vol"],
                "best_use_mom": champion["use_mom"],
                "best_hold_bars": champion["hold_bars"],
            }

            log(f"\n  [FOLD {fold.fold} 結果：GA 訓練區 vs 完全未見 final_test]")
            log(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))

            all_summary.append(summary)

            if len(ga_trades) > 0:
                all_trades.append(ga_trades)
            if len(test_trades) > 0:
                all_trades.append(test_trades)

            if len(ga_equity) > 0:
                all_equity.append(ga_equity)
            if len(test_equity) > 0:
                all_equity.append(test_equity)

            all_battle_logs.append(battle_log)

            fold_dir = EXPORT_DIR / f"fold_{fold.fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            dl_result["model"].save(fold_dir / "dl_model.keras")
            dl_result["encoder"].save(fold_dir / "dl_encoder.keras")

            joblib.dump(
                {"scaler": dl_result["scaler"], "feature_cols": feature_cols},
                fold_dir / "dl_scaler.joblib",
            )

            joblib.dump(
                {
                    "model": rf_result["model"],
                    "feature_cols": feature_cols,
                    "dl_embedding_dim": DL_EMBED_DIM,
                },
                fold_dir / "rf_model.joblib",
            )

            with open(fold_dir / "champion_gene.json", "w", encoding="utf-8") as f:
                json.dump(to_jsonable(champion), f, ensure_ascii=False, indent=2)

            with open(fold_dir / "champion_gene.pkl", "wb") as f:
                pickle.dump(champion, f)

            save_partial_state(
                all_summary,
                all_trades,
                all_equity,
                all_battle_logs,
                meta_extra={"last_completed_fold": fold.fold},
            )

            reset_tf()

    except KeyboardInterrupt:
        log("\n[中斷] 收到 KeyboardInterrupt，正在保存目前成果。")
    finally:
        save_partial_state(
            all_summary,
            all_trades,
            all_equity,
            all_battle_logs,
            meta_extra={"stopped": should_stop()},
        )

        meta = {
            "feature_path": str(FEATURE_PATH),
            "export_dir": str(EXPORT_DIR),
            "rows": int(len(df)) if "df" in locals() else None,
            "features": int(len(feature_cols)) if "feature_cols" in locals() else None,
            "folds_planned": int(len(folds)) if "folds" in locals() else None,
            "folds_completed": int(len(all_summary)),
            "time_col": TIME_COL,
            "label_col": LABEL_COL,
            "return_col": RETURN_COL,
            "split": {
                "train_ratio_start": TRAIN_RATIO_START,
                "train_ratio_step": TRAIN_RATIO_STEP,
                "val_ratio": VAL_RATIO,
                "ga_ratio": GA_RATIO,
                "test_ratio": TEST_RATIO,
            },
            "negative_sum_market": {
                "initial_capital": INITIAL_CAPITAL,
                "death_capital": DEATH_CAPITAL,
                "position_fraction": POSITION_FRACTION,
                "fee_rate": FEE_RATE,
                "fixed_fee_per_trade": FIXED_FEE_PER_TRADE,
                "living_cost_per_bar": LIVING_COST_PER_BAR,
                "min_valid_trades": MIN_VALID_TRADES,
                "min_abs_gross_return_valid": MIN_ABS_GROSS_RETURN_VALID,
                "min_hold_bars": MIN_HOLD_BARS,
                "max_hold_bars": MAX_HOLD_BARS,
            },
            "ga": {
                "pop_size_per_colony": POP_SIZE_PER_COLONY,
                "generations": GENERATIONS,
                "top_winners": TOP_WINNERS,
                "elite_per_colony": ELITE_PER_COLONY,
                "mutation_rate": MUTATION_RATE,
            },
            "outputs": {
                "summary": str(EXPORT_SUMMARY),
                "summary_json": str(EXPORT_SUMMARY_JSON),
                "trades": str(EXPORT_TRADES),
                "equity": str(EXPORT_EQUITY),
                "battle_log": str(EXPORT_BATTLE_LOG),
                "state_pkl": str(EXPORT_STATE),
            },
        }

        EXPORT_META.write_text(
            json.dumps(to_jsonable(meta), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        log("\n" + "=" * 78)
        log("[結束 / 已保存]")
        log("=" * 78)

        if all_summary:
            summary_df = pd.DataFrame(all_summary)
            display_cols = [
                "fold",
                "dl_test_balanced_accuracy",
                "rf_test_balanced_accuracy",
                "champion_colony",
                "champion_total_profit_on_ga",
                "test_final_capital",
                "test_total_profit",
                "test_dead",
                "test_trade_count",
                "test_valid_trade_count",
                "test_invalid_trade_count",
                "test_winrate",
                "test_max_drawdown",
                "best_hold_bars",
                "best_dl_th",
                "best_rf_th",
            ]
            cols = [c for c in display_cols if c in summary_df.columns]
            log(summary_df[cols].to_string(index=False))

            aggregate = {
                "folds_completed": int(len(summary_df)),
                "dead_test_folds": int(summary_df["test_dead"].sum()) if "test_dead" in summary_df else None,
                "avg_test_final_capital": float(summary_df["test_final_capital"].mean()) if "test_final_capital" in summary_df else None,
                "sum_test_total_profit": float(summary_df["test_total_profit"].sum()) if "test_total_profit" in summary_df else None,
                "avg_test_winrate": float(summary_df["test_winrate"].mean()) if "test_winrate" in summary_df else None,
                "avg_dl_test_bacc": float(summary_df["dl_test_balanced_accuracy"].mean()) if "dl_test_balanced_accuracy" in summary_df else None,
                "avg_rf_test_bacc": float(summary_df["rf_test_balanced_accuracy"].mean()) if "rf_test_balanced_accuracy" in summary_df else None,
            }

            log("\n[總結]")
            log(json.dumps(to_jsonable(aggregate), ensure_ascii=False, indent=2))

        log(f"\n輸出位置：{EXPORT_DIR}")
        log(f"停止檔：{STOP_FILE}")
        log("要安全停止：在 export 目錄建立 STOP 檔，或直接中斷 cell，程式會保存目前成果。")


if __name__ == "__main__":
    main()
