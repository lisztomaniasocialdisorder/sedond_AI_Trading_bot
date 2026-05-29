#!/usr/bin/env python3
"""
dashboards/server.py
====================
Dashboard API server for Binance Futures Harvester v2.

Run:
    cd dashboards
    pip install flask
    python server.py

Then open: http://localhost:5000
"""

import json
import pickle
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, request, send_from_directory

import okx_client

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rule_brain import evaluate_micro_signal

CACHE_TTL    = 0.4   # seconds (400 ms)  — shared read cache
CHART_LIMIT  = 400   # trade points for chart initialisation
FNG_CACHE_PATH = PROJECT_ROOT / "data" / "cache_fng.json"
FNG_CACHE_TTL = 12 * 60 * 60
NEWS_CACHE_TTL = 5 * 60
TRADING_STATE_PATH = PROJECT_ROOT / "data" / "trading_state.json"
EQUITY_SNAPSHOT_LIMIT = 24 * 365
V7_MODEL_DIR = PROJECT_ROOT / "experiments" / "v7" / "models"
V10_MODEL_DIR_CANDIDATES = [
    Path(r"G:\我的雲端硬碟\klinetraning\experiments\v10_dl_wf\models"),
    PROJECT_ROOT / "experiments" / "v10_dl_wf" / "models",
]
V10_STATUS_CANDIDATES = [
    Path(r"G:\我的雲端硬碟\klinetraning\experiments\v10_dl_wf\status.json"),
    PROJECT_ROOT / "experiments" / "v10_dl_wf" / "status.json",
]
V10_BUNDLE_CANDIDATES = [
    Path(r"C:\Users\brian\klinetraning\data_kline\training_arena_multiscale_bundle.parquet"),
    Path(r"G:\我的雲端硬碟\klinetraning\features\training_arena_multiscale_bundle.parquet"),
]
PARQUET_SIGNAL_ROOT = PROJECT_ROOT / "data" / "parquet_signal"
PARQUET_ROLLUP_ROOT = PROJECT_ROOT / "data" / "parquet_rollup"
STRATEGY_PARAMS_PATH = PROJECT_ROOT / "outputs" / "strategy_params.json"
NIGHTLY_LOG_DIR = PROJECT_ROOT / "logs" / "parquet_rollup"
NIGHTLY_LOCK_DIR = PROJECT_ROOT / "logs" / "locks" / "parquet_pipeline.lock"

# Override candidate roots with stable local-first paths.
KLINE_ROOT = Path.home() / "klinetraning"
GDRIVE_ROOT = None
try:
    g_root = Path("G:/")
    if g_root.exists():
        g_dirs = [p for p in g_root.iterdir() if p.is_dir()]
        if g_dirs:
            GDRIVE_ROOT = g_dirs[0]
except Exception:
    GDRIVE_ROOT = None

V10_MODEL_DIR_CANDIDATES = [KLINE_ROOT / "experiments" / "v10_dl_wf" / "models"]
V10_STATUS_CANDIDATES = [KLINE_ROOT / "experiments" / "v10_dl_wf" / "status.json"]
V10_BUNDLE_CANDIDATES = [
    KLINE_ROOT / "data_kline" / "training_arena_multiscale_bundle.parquet",
    KLINE_ROOT / "features" / "training_arena_multiscale_bundle.parquet",
]

if GDRIVE_ROOT is not None:
    V10_MODEL_DIR_CANDIDATES.append(GDRIVE_ROOT / "klinetraning" / "experiments" / "v10_dl_wf" / "models")
    V10_STATUS_CANDIDATES.append(GDRIVE_ROOT / "klinetraning" / "experiments" / "v10_dl_wf" / "status.json")
    V10_BUNDLE_CANDIDATES.append(GDRIVE_ROOT / "klinetraning" / "features" / "training_arena_multiscale_bundle.parquet")

V10_MODEL_DIR_CANDIDATES.append(PROJECT_ROOT / "experiments" / "v10_dl_wf" / "models")
V10_STATUS_CANDIDATES.append(PROJECT_ROOT / "experiments" / "v10_dl_wf" / "status.json")
V10_BUNDLE_CANDIDATES.append(PROJECT_ROOT / "features" / "training_arena_multiscale_bundle.parquet")

app    = Flask(__name__, static_folder=".")
_cache = {}
_lock  = Lock()
_v7_model_cache: dict | None = None
_v10_model_cache: dict | None = None
_v10_bundle_cache: dict | None = None
_news_cache: dict[str, dict] = {}
_parquet_rows_cache: dict[str, dict] = {}
_kline_cache: dict[str, dict] = {}

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_SEC_TO_INTERVAL = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
    14400: "4h",
    86400: "1d",
}


def _read_parquet_signal_1h(coin: str) -> dict | None:
    p = PARQUET_SIGNAL_ROOT / coin.upper() / "signal_averages_1h.parquet"
    if not p.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(p)
        if df.empty:
            return None
        row = df.iloc[-1].to_dict()
        return {
            "obi": row.get("obi_avg_1h"),
            "depth_imbalance": row.get("depth_imbalance_avg_1h"),
            "buy_pressure": row.get("buy_pressure_1h"),
            "ts": row.get("ts"),
            "source": "parquet",
            "path": str(p),
        }
    except Exception:
        return None


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        return total
    return total


def _parquet_rows_for_coin(coin: str) -> int:
    now = time.time()
    key = coin.upper()
    cached = _parquet_rows_cache.get(key)
    if cached and (now - float(cached.get("ts", 0))) < 60:
        return int(cached.get("rows", 0))

    root = PARQUET_ROLLUP_ROOT / key
    rows = 0
    if root.exists():
        files = list(root.rglob("*.parquet"))
        for p in files:
            try:
                import pyarrow.parquet as pq  # type: ignore

                rows += int(pq.ParquetFile(p).metadata.num_rows)
                continue
            except Exception:
                pass
            try:
                import pandas as pd

                rows += int(len(pd.read_parquet(p, columns=[])))
            except Exception:
                continue
    _parquet_rows_cache[key] = {"rows": rows, "ts": now}
    return rows


def _fetch_binance_futures_klines(symbol: str, interval: str, limit: int = 500) -> list[dict]:
    symbol = symbol.upper()
    limit = max(20, min(int(limit), 1500))
    cache_key = f"{symbol}:{interval}:{limit}"
    now = time.time()
    cached = _kline_cache.get(cache_key)
    if cached and (now - float(cached.get("ts", 0))) < 20:
        return list(cached.get("rows") or [])

    params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    req = Request(f"{BINANCE_FUTURES_KLINES_URL}?{params}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected kline response: {payload}")

    rows = []
    for r in payload:
        if not isinstance(r, list) or len(r) < 11:
            continue
        open_time = int(r[0])
        close_time = int(r[6])
        rows.append(
            {
                "time": int(open_time // 1000),
                "open_time": open_time,
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "close_time": close_time,
                "quote_asset_volume": float(r[7]),
                "number_of_trades": float(r[8]),
                "taker_buy_base_asset_volume": float(r[9]),
                "taker_buy_quote_asset_volume": float(r[10]),
                "ignore": 0.0,
            }
        )
    _kline_cache[cache_key] = {"ts": now, "rows": rows}
    return rows


def _feature_frame_from_kline_candles(candles: list[dict]):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(candles).copy()
    if df.empty:
        return df
    for col in (
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    qv = pd.to_numeric(df["quote_asset_volume"], errors="coerce")
    notr = pd.to_numeric(df["number_of_trades"], errors="coerce")

    df["price"] = close
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)
    df["log_ret_1"] = np.log(close / close.shift(1))

    for w in (5, 10, 20, 50, 100):
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        df[f"sma_{w}"] = sma
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"vol_{w}"] = df["log_ret_1"].rolling(w).std()
        df[f"zscore_{w}"] = (close - sma) / std.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr_pct"] = df["atr_14"] / close.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2 * bb_std
    df["bb_dn"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (close - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"]).replace(0, np.nan)

    df["hl_spread"] = (high - low) / close.replace(0, np.nan)
    df["co_spread"] = (close - open_) / open_.replace(0, np.nan)
    df["volume_chg_1"] = volume.pct_change(1)
    df["quote_volume_chg_1"] = qv.pct_change(1)
    df["trades_chg_1"] = notr.pct_change(1)
    df["hour"] = df["timestamp_dt"].dt.hour
    df["day_of_week"] = df["timestamp_dt"].dt.dayofweek
    df["month"] = df["timestamp_dt"].dt.month
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _prepare_live_scale(df, suffix: str, keep_base: bool):
    import pandas as pd

    base_cols = {"timestamp_dt", "price"}
    keep = ["timestamp_dt"] + [c for c in df.columns if c != "timestamp_dt" and pd.api.types.is_numeric_dtype(df[c])]
    out = df[keep].copy()
    rename_map = {}
    for c in out.columns:
        if c == "timestamp_dt":
            continue
        if keep_base and c in base_cols:
            continue
        rename_map[c] = f"{c}_{suffix}"
    return out.rename(columns=rename_map)


def _build_live_v10_bundle(symbol: str):
    import numpy as np
    import pandas as pd

    m30 = _feature_frame_from_kline_candles(_fetch_binance_futures_klines(symbol, "30m", 1200))
    h1 = _feature_frame_from_kline_candles(_fetch_binance_futures_klines(symbol, "1h", 1200))
    d1 = _feature_frame_from_kline_candles(_fetch_binance_futures_klines(symbol, "1d", 1200))
    if m30.empty or h1.empty or d1.empty:
        return None

    base = _prepare_live_scale(m30, "30m", keep_base=True)
    one_h = _prepare_live_scale(h1, "1h", keep_base=False)
    one_d = _prepare_live_scale(d1, "1d", keep_base=False)

    merged = pd.merge_asof(
        base.sort_values("timestamp_dt"),
        one_h.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp_dt"),
        one_d.sort_values("timestamp_dt"),
        on="timestamp_dt",
        direction="backward",
    )
    merged = merged.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in merged.columns if c != "timestamp_dt"]
    merged[feature_cols] = merged[feature_cols].ffill().bfill()
    return merged.dropna(subset=["timestamp_dt"]).reset_index(drop=True)


def _empty_trading_state() -> dict:
    return {
        "prefs": {},
        "equitySnapshots": [],
        "tradeRecords": [],
        "updated_at": 0,
    }


def _read_trading_state() -> dict:
    if not TRADING_STATE_PATH.exists():
        return _empty_trading_state()
    try:
        payload = json.loads(TRADING_STATE_PATH.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict):
            return _empty_trading_state()
        state = _empty_trading_state()
        state.update(payload)
        if not isinstance(state.get("prefs"), dict):
            state["prefs"] = {}
        if not isinstance(state.get("equitySnapshots"), list):
            state["equitySnapshots"] = []
        if not isinstance(state.get("tradeRecords"), list):
            state["tradeRecords"] = []
        return state
    except Exception:
        return _empty_trading_state()


def _write_trading_state(state: dict) -> None:
    state = dict(state)
    state["equitySnapshots"] = list(state.get("equitySnapshots") or [])[-EQUITY_SNAPSHOT_LIMIT:]
    state["tradeRecords"] = list(state.get("tradeRecords") or [])[:500]
    state["updated_at"] = time.time()
    TRADING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADING_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_v7_model() -> dict | None:
    global _v7_model_cache
    if _v7_model_cache is not None:
        return _v7_model_cache

    if not V7_MODEL_DIR.exists():
        _v7_model_cache = {}
        return None

    model_files = sorted(V7_MODEL_DIR.glob("fold_*_best.pkl"), key=lambda p: p.stat().st_mtime)
    if not model_files:
        _v7_model_cache = {}
        return None

    chosen = model_files[-1]
    try:
        payload = pickle.loads(chosen.read_bytes())
    except Exception:
        _v7_model_cache = {}
        return None

    if not isinstance(payload, dict):
        _v7_model_cache = {}
        return None
    required = ("weights", "feature_cols", "feature_mean", "feature_std", "window_size")
    if not all(k in payload for k in required):
        _v7_model_cache = {}
        return None

    _v7_model_cache = {
        "path": str(chosen),
        "weights": payload["weights"],
        "feature_cols": list(payload["feature_cols"]),
        "feature_mean": payload["feature_mean"],
        "feature_std": payload["feature_std"],
        "window_size": int(payload.get("window_size", 10)),
        "fold": int(payload.get("fold", 0) or 0),
        "generation": int(payload.get("best_generation", 0) or 0),
    }
    return _v7_model_cache


def _pick_existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None


def _load_v10_bundle_df():
    global _v10_bundle_cache
    bundle_path = _pick_existing_path(V10_BUNDLE_CANDIDATES)
    if bundle_path is None:
        return None

    try:
        mtime = bundle_path.stat().st_mtime
    except Exception:
        mtime = None

    if (
        _v10_bundle_cache
        and _v10_bundle_cache.get("path") == str(bundle_path)
        and _v10_bundle_cache.get("mtime") == mtime
    ):
        return _v10_bundle_cache.get("df")

    try:
        import pandas as pd

        df = pd.read_parquet(bundle_path)
    except Exception:
        return None

    if "timestamp_dt" in df.columns:
        ts = pd.to_datetime(df["timestamp_dt"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    else:
        return None

    df = df.copy()
    df["timestamp_dt"] = ts
    df = df[df["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    _v10_bundle_cache = {"path": str(bundle_path), "mtime": mtime, "df": df}
    return df


def _load_v10_model() -> dict | None:
    global _v10_model_cache
    if _v10_model_cache is not None:
        return _v10_model_cache

    model_dir = _pick_existing_path(V10_MODEL_DIR_CANDIDATES)
    status_path = _pick_existing_path(V10_STATUS_CANDIDATES)
    if model_dir is None or not model_dir.exists():
        _v10_model_cache = {}
        return None

    best_fold = None
    if status_path is not None:
        try:
            st = json.loads(status_path.read_text(encoding="utf-8-sig"))
            best_fold = int(st.get("best_fold")) if st.get("best_fold") is not None else None
        except Exception:
            best_fold = None

    chosen_model = None
    chosen_scaler = None
    if best_fold is not None:
        m = model_dir / f"fold_{best_fold}_dl.keras"
        s = model_dir / f"fold_{best_fold}_scaler.json"
        if m.exists() and s.exists():
            chosen_model, chosen_scaler = m, s
    if chosen_model is None or chosen_scaler is None:
        candidates = sorted(model_dir.glob("fold_*_dl.keras"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            _v10_model_cache = {}
            return None
        chosen_model = candidates[-1]
        stem = chosen_model.name.replace("_dl.keras", "")
        s = model_dir / f"{stem}_scaler.json"
        if not s.exists():
            _v10_model_cache = {}
            return None
        chosen_scaler = s

    try:
        scaler_payload = json.loads(chosen_scaler.read_text(encoding="utf-8-sig"))
        mean = scaler_payload.get("mean") or []
        scale = scaler_payload.get("scale") or []
        import numpy as np

        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    except Exception:
        _v10_model_cache = {}
        return None

    df = _load_v10_bundle_df()
    if df is None:
        try:
            df = _build_live_v10_bundle("BTCUSDT")
        except Exception:
            df = None
    if df is None or df.empty:
        _v10_model_cache = {}
        return None

    drop_cols = {"timestamp_dt", "future_return", "label", "_ret", "target", "y", "price"}
    leak_tokens = ("future_return", "next_return", "target_up", "label")
    feat_cols = [
        c
        for c in df.columns
        if c not in drop_cols
        and str(df[c].dtype) != "object"
        and not any(tok in c.lower() for tok in leak_tokens)
    ]

    if len(feat_cols) != len(mean) or len(feat_cols) != len(scale):
        _v10_model_cache = {}
        return None

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(chosen_model)
    except Exception:
        _v10_model_cache = {}
        return None

    _v10_model_cache = {
        "path": str(chosen_model),
        "fold": int(str(chosen_model.stem).split("_")[1]),
        "model": model,
        "feature_cols": feat_cols,
        "mean": mean,
        "scale": scale,
        "model_name": "v10_dl",
    }
    return _v10_model_cache


def _tail_lines(path: Path, limit: int = 30) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return lines[-max(1, int(limit)) :]


def _nightly_pipeline_status() -> dict:
    lock_exists = NIGHTLY_LOCK_DIR.exists()
    logs = []
    if NIGHTLY_LOG_DIR.exists():
        try:
            logs = sorted(
                NIGHTLY_LOG_DIR.glob("midnight_aggregate_*.log"),
                key=lambda p: p.stat().st_mtime,
            )
        except Exception:
            logs = []
    latest = logs[-1] if logs else None
    if latest is None:
        return {
            "ok": True,
            "exists": False,
            "running": bool(lock_exists),
            "status": "未找到執行紀錄",
            "progress": "0/4",
            "last_run": None,
            "last_log": None,
            "tail": [],
            "ts": time.time(),
        }

    tail = _tail_lines(latest, limit=40)
    step_current = 0
    step_total = 4
    step_title = "等待中"
    finished = False
    success = False
    exit_code = None
    status = "等待中"
    step_re = re.compile(r"step(\d+)/(\d+)\s+(.*)$", re.IGNORECASE)
    done_re = re.compile(r"finished exit=(\d+)", re.IGNORECASE)

    for line in tail:
        m = step_re.search(line)
        if m:
            step_current = int(m.group(1))
            step_total = int(m.group(2))
            step_title = m.group(3).strip()
        d = done_re.search(line)
        if d:
            finished = True
            exit_code = int(d.group(1))
            success = exit_code == 0

    running_process_detected = False
    if finished:
        status = "成功" if success else f"失敗(exit={exit_code})"
    elif lock_exists:
        status = f"執行中 step{step_current}/{step_total}"
    else:
        status = "等待下次 00:30"

    # If log has not emitted step lines yet, infer step from running commands.
    if lock_exists and not finished and step_current == 0:
        try:
            ps = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|py.exe' } | Select-Object -ExpandProperty CommandLine",
                ],
                capture_output=True,
                text=True,
                timeout=2.5,
            )
            text = (ps.stdout or "").lower()
            if "compact_parquet_spool.py" in text:
                running_process_detected = True
                step_current = 1
                step_title = "compact spool -> rollup"
            elif "parquet_to_features.py" in text:
                running_process_detected = True
                step_current = 2
                step_title = "parquet -> features parquet"
            elif "obi_edge_report.py" in text:
                running_process_detected = True
                step_current = 3
                step_title = "build OBI edge report"
            elif re.search(r"(^|[\\/ ])run\.py([ ']|$)", text):
                running_process_detected = True
                step_current = 4
                step_title = "train model"
            if running_process_detected:
                status = f"執行中 step{step_current}/{step_total}"
            else:
                status = "鎖檔殘留（未執行）"
        except Exception:
            pass

    try:
        mtime = latest.stat().st_mtime
    except Exception:
        mtime = None

    return {
        "ok": True,
        "exists": True,
        "running": bool((lock_exists and not finished) and running_process_detected),
        "status": status,
        "progress": f"{step_current}/{step_total}",
        "step_current": step_current,
        "step_total": step_total,
        "step_title": step_title,
        "finished": finished,
        "success": success,
        "exit_code": exit_code,
        "last_run": mtime,
        "last_log": str(latest),
        "tail": tail[-12:],
        "ts": time.time(),
    }


def _detect_pipeline_process_v2() -> tuple[bool, int | None, str | None]:
    try:
        ps = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|py.exe|cmd.exe|powershell.exe' } | Select-Object -ExpandProperty CommandLine",
            ],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
    except Exception:
        return False, None, None

    text = (ps.stdout or "").lower()
    if "compact_parquet_spool.py" in text or "hourly_db_to_parquet.py" in text:
        return True, 1, "db/parquet spool -> rollup"
    if "parquet_to_features.py" in text:
        return True, 2, "parquet -> features parquet"
    if "obi_edge_report.py" in text or "alpha_conditional_test.py" in text:
        return True, 3, "validate OBI edge"
    if re.search(r"(^|[\\/ ])run\.py([ ']|$)", text) or "train_microstructure_ai.py" in text:
        return True, 4, "train model"
    return False, None, None


def _nightly_pipeline_status() -> dict:
    lock_exists = NIGHTLY_LOCK_DIR.exists()
    logs = []
    if NIGHTLY_LOG_DIR.exists():
        try:
            logs = sorted(
                NIGHTLY_LOG_DIR.glob("midnight_aggregate_*.log"),
                key=lambda p: p.stat().st_mtime,
            )
        except Exception:
            logs = []

    latest = logs[-1] if logs else None
    process_running, detected_step, detected_title = _detect_pipeline_process_v2()
    step_total = 4

    if latest is None:
        return {
            "ok": True,
            "exists": False,
            "running": bool(process_running),
            "status": "執行中" if process_running else "未找到執行紀錄",
            "progress": f"{int(detected_step or 0)}/{step_total}",
            "step_current": int(detected_step or 0),
            "step_total": step_total,
            "step_title": detected_title or "等待中",
            "finished": False,
            "success": False,
            "exit_code": None,
            "last_run": None,
            "last_log": None,
            "tail": [],
            "ts": time.time(),
        }

    tail = _tail_lines(latest, limit=40)
    step_current = 0
    step_title = "等待中"
    finished = False
    success = False
    exit_code = None
    step_re = re.compile(r"step(\d+)/(\d+)\s+(.*)$", re.IGNORECASE)
    done_re = re.compile(r"finished exit=(\d+)", re.IGNORECASE)

    for line in tail:
        m = step_re.search(line)
        if m:
            step_current = int(m.group(1))
            step_total = int(m.group(2))
            step_title = m.group(3).strip()
        d = done_re.search(line)
        if d:
            finished = True
            exit_code = int(d.group(1))
            success = exit_code == 0

    if process_running and detected_step is not None:
        step_current = max(step_current, int(detected_step))
        if detected_title:
            step_title = detected_title

    if finished:
        status = "成功" if success else f"失敗(exit={exit_code})"
        running = False
        if success:
            step_current = step_total
    elif process_running:
        status = f"執行中 step{step_current}/{step_total}"
        running = True
    elif lock_exists:
        status = "鎖檔殘留（未執行）"
        running = False
    else:
        status = "等待下次 00:30"
        running = False

    try:
        mtime = latest.stat().st_mtime
    except Exception:
        mtime = None

    return {
        "ok": True,
        "exists": True,
        "running": bool(running),
        "status": status,
        "progress": f"{step_current}/{step_total}",
        "step_current": step_current,
        "step_total": step_total,
        "step_title": step_title,
        "finished": finished,
        "success": success,
        "exit_code": exit_code,
        "last_run": mtime,
        "last_log": str(latest),
        "tail": tail[-12:],
        "ts": time.time(),
    }


# ── path helpers ──────────────────────────────────────────────────────────────
def _db_path(symbol: str) -> Path:
    coin = symbol.upper().replace("USDT", "")
    return (
        PROJECT_ROOT
        / "harvesters"
        / f"{coin}_harvester"
        / "raw_db"
        / f"microstructure_{coin}.db"
    )


def _open_ro(symbol: str):
    """Open SQLite in read-only WAL mode. Returns None if DB doesn't exist."""
    p = _db_path(symbol)
    if not p.exists():
        return None
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _q(conn, sql, params=()):
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def _q1(conn, sql, params=()):
    rows = _q(conn, sql, params)
    return rows[0] if rows else {}


# ── snapshot builder ──────────────────────────────────────────────────────────
def _build_snapshot(symbol: str) -> dict:
    symbol = symbol.upper()
    conn   = _open_ro(symbol)

    if conn is None:
        return {
            "symbol": symbol,
            "ts":     time.time(),
            "error":  "DB not found — is the harvester running?",
        }

    try:
        # Latest L1 quote — support both new aggregated schema and legacy schema
        _l1_cols = [r[1] for r in conn.execute("PRAGMA table_info(orderbook_l1)").fetchall()]
        if "second_ts" in _l1_cols:
            # New 1-second aggregated schema
            l1 = _q1(conn, """
                SELECT second_ts AS event_ts,
                       bid_price, ask_price,
                       bid_qty_mean AS bid_qty, ask_qty_mean AS ask_qty,
                       spread_bps_mean AS spread_bps,
                       (spread_bps_mean * mid_close / 10000.0) AS spread,
                       mid_close AS mid_price,
                       obi_mean AS obi,
                       obi_std, obi_open, obi_close,
                       spread_bps_std,
                       tick_count
                FROM orderbook_l1 ORDER BY id DESC LIMIT 1
            """)
        else:
            l1 = _q1(conn, "SELECT * FROM orderbook_l1 ORDER BY id DESC LIMIT 1")

        # Latest mark price
        mp = _q1(conn, "SELECT * FROM mark_price ORDER BY id DESC LIMIT 1")

        # Last 25 individual trades (trade feed + chart update)
        trades = _q(conn, "SELECT * FROM trades ORDER BY id DESC LIMIT 25")

        # 1-minute aggregate stats
        cutoff_ms = int((time.time() - 60) * 1_000)
        stats = _q1(conn, """
            SELECT
                COUNT(*)                                                         AS count,
                COALESCE(SUM(qty),        0)                                     AS volume,
                COALESCE(SUM(quote_qty),  0)                                     AS quote_volume,
                COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN qty   ELSE 0 END), 0) AS buy_volume,
                COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN qty   ELSE 0 END), 0) AS sell_volume,
                MAX(price)                                                       AS high_1m,
                MIN(price)                                                       AS low_1m
            FROM trades
            WHERE trade_ts >= ?
        """, (cutoff_ms,))

        # L5 order book — latest snapshot
        uid5 = _q1(conn, "SELECT MAX(update_id) AS u FROM orderbook_l5").get("u")
        l5 = _q(conn,
            "SELECT level,bid_price,bid_qty,ask_price,ask_qty "
            "FROM orderbook_l5 WHERE update_id=? ORDER BY level",
            (uid5,)) if uid5 else []

        # L20 order book — latest snapshot
        uid20 = _q1(conn, "SELECT MAX(update_id) AS u FROM orderbook_l20").get("u")
        l20 = _q(conn,
            "SELECT level,bid_price,bid_qty,ask_price,ask_qty "
            "FROM orderbook_l20 WHERE update_id=? ORDER BY level",
            (uid20,)) if uid20 else []

        # Depth metrics
        mx5  = _q1(conn, "SELECT * FROM orderbook_metrics WHERE depth_type='l5'  ORDER BY id DESC LIMIT 1")
        mx20 = _q1(conn, "SELECT * FROM orderbook_metrics WHERE depth_type='l20' ORDER BY id DESC LIMIT 1")

        # Recent liquidations
        liq = _q(conn, "SELECT * FROM liquidations ORDER BY id DESC LIMIT 12")

        # DB file size
        db_bytes = _db_path(symbol).stat().st_size

        # Row counts per table
        _tables = [
            "trades", "agg_trades", "orderbook_l1",
            "orderbook_l5", "orderbook_l20", "orderbook_metrics",
            "mark_price", "liquidations",
        ]
        row_counts = {}
        for tbl in _tables:
            try:
                row_counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except Exception:
                row_counts[tbl] = None

        return {
            "symbol":        symbol,
            "ts":            time.time(),
            "l1":            l1,
            "mark_price":    mp,
            "recent_trades": trades,
            "stats_1m":      stats,
            "l5":            l5,
            "l20":           l20,
            "metrics_l5":    mx5,
            "metrics_l20":   mx20,
            "liquidations":  liq,
            "db_size_mb":    round(db_bytes / 1_048_576, 2),
            "row_counts":    row_counts,
        }

    except Exception as exc:
        return {"symbol": symbol, "ts": time.time(), "error": str(exc)}

    finally:
        conn.close()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/api/snapshot/<symbol>")
def route_snapshot(symbol):
    symbol = symbol.upper()
    now    = time.time()
    with _lock:
        cached = _cache.get(symbol)
        if cached and (now - cached["ts"]) < CACHE_TTL:
            return jsonify(cached["data"])
        data = _build_snapshot(symbol)
        _cache[symbol] = {"data": data, "ts": now}
    return jsonify(data)


@app.route("/api/chart/<symbol>")
def route_chart(symbol):
    """Return last N trade prices for chart initialisation (oldest → newest)."""
    symbol = symbol.upper()
    limit  = min(int(request.args.get("limit", CHART_LIMIT)), 2_000)
    conn   = _open_ro(symbol)
    if conn is None:
        return jsonify([])
    try:
        rows = _q(conn,
            "SELECT trade_ts, price FROM trades ORDER BY id DESC LIMIT ?",
            (limit,))
        return jsonify([
            {"time": int(r["trade_ts"] // 1_000), "value": r["price"]}
            for r in reversed(rows)
            if r.get("trade_ts") and r.get("price")
        ])
    finally:
        conn.close()


def _build_signal_dashboard(symbol: str) -> dict:
    symbol = symbol.upper()
    conn = _open_ro(symbol)
    if conn is None:
        return {"ok": False, "symbol": symbol, "error": "DB not found", "ts": time.time()}

    try:
        l1_cols = [r[1] for r in conn.execute("PRAGMA table_info(orderbook_l1)").fetchall()]
        if "second_ts" in l1_cols:
            latest_l1 = _q1(conn, """
                SELECT second_ts AS ts,
                       bid_price, ask_price,
                       bid_qty_mean AS bid_qty,
                       ask_qty_mean AS ask_qty,
                       mid_close AS mid_price,
                       spread_bps_mean AS spread_bps,
                       obi_mean AS obi,
                       tick_count
                FROM orderbook_l1
                ORDER BY id DESC
                LIMIT 1
            """)
            l1_series = _q(conn, """
                SELECT second_ts AS ts,
                       mid_close AS mid_price,
                       spread_bps_mean AS spread_bps,
                       obi_mean AS obi
                FROM orderbook_l1
                ORDER BY id DESC
                LIMIT 240
            """)
        else:
            latest_l1 = _q1(conn, "SELECT event_ts AS ts, * FROM orderbook_l1 ORDER BY id DESC LIMIT 1")
            l1_series = _q(conn, """
                SELECT event_ts AS ts, mid_price, spread_bps, obi
                FROM orderbook_l1
                ORDER BY id DESC
                LIMIT 240
            """)

        trades = _q(conn, """
            SELECT trade_ts AS ts, price, qty, is_buyer_maker
            FROM trades
            ORDER BY id DESC
            LIMIT 240
        """)

        metrics = _q(conn, """
            SELECT event_ts AS ts,
                   total_bid_qty,
                   total_ask_qty,
                   depth_imbalance,
                   bid_vwap,
                   ask_vwap,
                   weighted_mid
            FROM orderbook_metrics
            WHERE depth_type='l20'
            ORDER BY id DESC
            LIMIT 240
        """)

        uid5 = _q1(conn, "SELECT MAX(update_id) AS u FROM orderbook_l5").get("u")
        book = _q(conn, """
            SELECT level, bid_price, bid_qty, ask_price, ask_qty
            FROM orderbook_l5
            WHERE update_id=?
            ORDER BY level
        """, (uid5,)) if uid5 else []

        cutoff_ms = int((time.time() - 60) * 1_000)
        cutoff_1h_ms = int((time.time() - 3600) * 1_000)
        stats = _q1(conn, """
            SELECT COUNT(*) AS trades_1m,
                   COALESCE(SUM(qty), 0) AS volume_1m,
                   COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN qty ELSE 0 END), 0) AS buy_volume_1m,
                   COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN qty ELSE 0 END), 0) AS sell_volume_1m,
                   MAX(price) AS high_1m,
                   MIN(price) AS low_1m
            FROM trades
            WHERE trade_ts >= ?
        """, (cutoff_ms,))
        # 1-hour averages for reason panel
        l1_1h = _q1(conn, """
            SELECT AVG(obi) AS obi_avg_1h
            FROM (
                SELECT obi_mean AS obi
                FROM orderbook_l1
                WHERE second_ts >= ? AND obi_mean IS NOT NULL
            )
        """, (cutoff_1h_ms,))
        metrics_1h = _q1(conn, """
            SELECT AVG(depth_imbalance) AS depth_imbalance_avg_1h
            FROM orderbook_metrics
            WHERE depth_type='l20' AND event_ts >= ? AND depth_imbalance IS NOT NULL
        """, (cutoff_1h_ms,))
        flow_1h = _q1(conn, """
            SELECT COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN qty ELSE 0 END), 0) AS buy_volume_1h,
                   COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN qty ELSE 0 END), 0) AS sell_volume_1h
            FROM trades
            WHERE trade_ts >= ?
        """, (cutoff_1h_ms,))

        row_counts = {}
        for tbl in ("trades", "agg_trades", "orderbook_l1", "orderbook_l5", "orderbook_l20", "orderbook_metrics"):
            row_counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

        latest_metric = metrics[0] if metrics else {}
        buy_pressure = None
        if stats:
            total_volume = (stats.get("buy_volume_1m") or 0) + (stats.get("sell_volume_1m") or 0)
            if total_volume > 0:
                buy_pressure = (stats.get("buy_volume_1m") or 0) / total_volume
        buy_pressure_1h = None
        if flow_1h:
            total_volume_1h = (flow_1h.get("buy_volume_1h") or 0) + (flow_1h.get("sell_volume_1h") or 0)
            if total_volume_1h > 0:
                buy_pressure_1h = (flow_1h.get("buy_volume_1h") or 0) / total_volume_1h

        averages_1h_db = {
            "obi": l1_1h.get("obi_avg_1h"),
            "depth_imbalance": metrics_1h.get("depth_imbalance_avg_1h"),
            "buy_pressure": buy_pressure_1h,
            "source": "db",
        }
        coin = symbol.replace("USDT", "")
        averages_1h = _read_parquet_signal_1h(coin) or averages_1h_db
        db_total_rows = sum(v for v in row_counts.values() if isinstance(v, int))
        parquet_rows_1h = 0
        for k in ("l1_rows_1h", "depth_rows_1h", "trade_rows_1h"):
            try:
                v = averages_1h.get(k) if isinstance(averages_1h, dict) else None
                if v is not None:
                    parquet_rows_1h += int(v)
            except Exception:
                pass
        coin = symbol.replace("USDT", "")
        parquet_rows_all = _parquet_rows_for_coin(coin)
        total_rows_combined = db_total_rows + parquet_rows_all
        db_bytes = _db_path(symbol).stat().st_size
        parquet_bytes = _dir_size_bytes(PARQUET_SIGNAL_ROOT / coin) + _dir_size_bytes(PARQUET_ROLLUP_ROOT / coin)
        storage_mb = round((db_bytes + parquet_bytes) / 1_048_576, 2)

        return {
            "ok": True,
            "symbol": symbol,
            "ts": time.time(),
            "latest": {
                "price": latest_l1.get("mid_price"),
                "best_bid": latest_l1.get("bid_price"),
                "best_ask": latest_l1.get("ask_price"),
                "spread_bps": latest_l1.get("spread_bps"),
                "obi": latest_l1.get("obi"),
                "tick_count": latest_l1.get("tick_count"),
                "depth_imbalance": latest_metric.get("depth_imbalance"),
                "weighted_mid": latest_metric.get("weighted_mid"),
                "buy_pressure": buy_pressure,
            },
            "averages_1h": averages_1h,
            "stats": stats,
            "counts": row_counts,
            "db_rows": db_total_rows,
            "parquet_rows_1h": parquet_rows_1h,
            "parquet_rows_all": parquet_rows_all,
            "total_rows": total_rows_combined,
            "db_size_mb": storage_mb,
            "series": {
                "trades": list(reversed(trades)),
                "l1": list(reversed(l1_series)),
                "metrics": list(reversed(metrics)),
            },
            "book": book,
        }
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": str(exc), "ts": time.time()}
    finally:
        conn.close()


@app.route("/api/signal/<symbol>")
def route_signal_dashboard(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return jsonify(_build_signal_dashboard(symbol))


@app.route("/api/rule-brain/<symbol>")
def route_rule_brain(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    snapshot = _build_signal_dashboard(symbol)
    if not snapshot.get("ok"):
        return jsonify({
            "ok": False,
            "symbol": symbol,
            "error": snapshot.get("error", "snapshot unavailable"),
            "ts": snapshot.get("ts", time.time()),
        })
    return jsonify({
        "ok": True,
        "symbol": symbol,
        "ts": time.time(),
        "decision": evaluate_micro_signal(snapshot),
    })


@app.route("/api/model/v7-decision/<symbol>")
def route_model_v7_decision(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    try:
        pos = int(float(request.args.get("pos", 0) or 0))
    except Exception:
        pos = 0
    try:
        capital_ratio = float(request.args.get("capital_ratio", 1.0) or 1.0)
    except Exception:
        capital_ratio = 1.0
    pos = -1 if pos < 0 else (1 if pos > 0 else 0)
    capital_ratio = max(0.01, min(10.0, capital_ratio))
    return jsonify(_predict_ai_decision(symbol, pos=pos, capital_ratio=capital_ratio))


@app.route("/api/model/ai-decision/<symbol>")
def route_model_ai_decision(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    try:
        pos = int(float(request.args.get("pos", 0) or 0))
    except Exception:
        pos = 0
    try:
        capital_ratio = float(request.args.get("capital_ratio", 1.0) or 1.0)
    except Exception:
        capital_ratio = 1.0
    try:
        interval_sec = int(float(request.args.get("interval_sec", 3600) or 3600))
    except Exception:
        interval_sec = 3600
    model_mode = str(request.args.get("model_mode", "hybrid_v10_v13") or "hybrid_v10_v13")
    pos = -1 if pos < 0 else (1 if pos > 0 else 0)
    capital_ratio = max(0.01, min(10.0, capital_ratio))
    interval_sec = max(60, min(86400, interval_sec))
    return jsonify(_predict_ai_decision(symbol, pos=pos, capital_ratio=capital_ratio, interval_sec=interval_sec, model_mode=model_mode))


def _ema(values: list[float], period: int) -> list[float | None]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out: list[float | None] = []
    cur: float | None = None
    for value in values:
        cur = value if cur is None else (value * alpha) + (cur * (1.0 - alpha))
        out.append(cur)
    return out


def _read_fng_cache() -> dict:
    if not FNG_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(FNG_CACHE_PATH.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _fetch_fear_greed_history() -> list[dict]:
    cached = _read_fng_cache()
    cached_at = float(cached.get("cached_at", 0) or 0)
    cached_rows = cached.get("rows")
    if isinstance(cached_rows, list) and cached_rows and (time.time() - cached_at) < FNG_CACHE_TTL:
        return cached_rows

    try:
        req = Request(
            "https://api.alternative.me/fng/?limit=0&format=json",
            headers={
                "Accept": "application/json",
                "User-Agent": "trading-dashboard/1.0",
            },
        )
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = []
        for item in payload.get("data", []):
            ts = int(item.get("timestamp") or 0)
            value = float(item.get("value"))
            if ts > 0:
                rows.append({
                    "time": ts,
                    "value": max(0.0, min(100.0, value)),
                    "label": str(item.get("value_classification") or ""),
                })
        rows = sorted(rows, key=lambda row: row["time"])
        if rows:
            FNG_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            FNG_CACHE_PATH.write_text(
                json.dumps({"cached_at": time.time(), "rows": rows}, ensure_ascii=False),
                encoding="utf-8",
            )
            return rows
    except Exception:
        pass

    return cached_rows if isinstance(cached_rows, list) else []


def _fetch_keyword_news_items(keyword: str, limit: int = 6) -> list[dict]:
    key = str(keyword or "").strip()
    if not key:
        return []
    now_ts = time.time()
    cached = _news_cache.get(key)
    if cached and (now_ts - float(cached.get("ts", 0) or 0)) < NEWS_CACHE_TTL:
        rows = cached.get("rows")
        return rows if isinstance(rows, list) else []

    q = quote(key)
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    rows: list[dict] = []
    try:
        req = Request(
            url,
            headers={
                "Accept": "application/rss+xml, application/xml",
                "User-Agent": "trading-dashboard/1.0",
            },
        )
        with urlopen(req, timeout=10) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[: max(1, int(limit))]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            if not title:
                continue
            rows.append(
                {
                    "keyword": key,
                    "title": title,
                    "link": link,
                    "published_at": pub_date,
                    "source": source,
                }
            )
    except Exception:
        rows = []

    _news_cache[key] = {"ts": now_ts, "rows": rows}
    return rows


def _build_keyword_news_payload() -> dict:
    keywords = ["川普", "美聯儲", "CPI", "PPI"]
    grouped = []
    all_rows = []
    for kw in keywords:
        rows = _fetch_keyword_news_items(kw, limit=6)
        grouped.append({"keyword": kw, "items": rows[:6]})
        all_rows.extend(rows[:3])
    all_rows = sorted(all_rows, key=lambda r: str(r.get("published_at") or ""), reverse=True)[:20]
    return {
        "ok": True,
        "ts": time.time(),
        "keywords": keywords,
        "grouped": grouped,
        "items": all_rows,
    }


def _indicator_payload_from_candles(symbol: str, interval_sec: int, candles: list[dict], source: str) -> dict:
    candles = sorted(candles, key=lambda r: int(r.get("time") or 0))[-500:]
    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [
        (a - b) if a is not None and b is not None else None
        for a, b in zip(ema12, ema26)
    ]
    signal_vals = _ema([v if v is not None else 0.0 for v in macd], 9)
    hist = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd, signal_vals)
    ]

    gains: list[float] = []
    losses: list[float] = []
    rsi: list[float | None] = []
    for idx, close in enumerate(closes):
        if idx == 0:
            gains.append(0.0)
            losses.append(0.0)
            rsi.append(None)
            continue
        diff = close - closes[idx - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
        if idx < 14:
            rsi.append(None)
        else:
            avg_gain = sum(gains[idx - 13:idx + 1]) / 14.0
            avg_loss = sum(losses[idx - 13:idx + 1]) / 14.0
            rsi.append(100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))))

    tr_values: list[float] = []
    atr: list[float | None] = []
    for idx, candle in enumerate(candles):
        prev_close = closes[idx - 1] if idx else candle["close"]
        tr = max(highs[idx] - lows[idx], abs(highs[idx] - prev_close), abs(lows[idx] - prev_close))
        tr_values.append(float(tr))
        atr.append(None if idx < 14 else sum(tr_values[idx - 13:idx + 1]) / 14.0)

    fg = _fetch_fear_greed_history()
    return {
        "ok": True,
        "symbol": symbol,
        "interval_sec": interval_sec,
        "source": source,
        "candles": candles,
        "macd": [
            {"time": candles[i]["time"], "macd": macd[i], "signal": signal_vals[i], "hist": hist[i]}
            for i in range(len(candles))
        ],
        "rsi": [{"time": candles[i]["time"], "value": rsi[i]} for i in range(len(candles))],
        "atr": [{"time": candles[i]["time"], "value": atr[i]} for i in range(len(candles))],
        "fear_greed": fg,
    }


def _build_indicator_chart(symbol: str) -> dict:
    symbol = symbol.upper()
    try:
        interval_sec = max(60, min(int(request.args.get("interval", 3600)), 86400))
        interval = KLINE_SEC_TO_INTERVAL.get(interval_sec, "1h")
        limit = max(200, min(int(request.args.get("limit", 600)), 1500))
        candles = _fetch_binance_futures_klines(symbol, interval, limit=limit)
        if candles:
            return _indicator_payload_from_candles(
                symbol=symbol,
                interval_sec=interval_sec,
                candles=candles,
                source="binance_futures_kline",
            )
    except Exception:
        pass

    conn = _open_ro(symbol)
    if conn is None:
        return {"ok": False, "symbol": symbol, "error": "DB not found"}

    try:
        interval_sec = max(30, min(int(request.args.get("interval", 60)), 3600))
        limit = max(600, min(int(request.args.get("limit", 21600)), 50000))
        rows = _q(conn, """
            SELECT second_ts AS ts,
                   mid_close AS price,
                   COALESCE(tick_count, 1) AS qty
            FROM orderbook_l1
            WHERE second_ts IS NOT NULL AND mid_close IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = list(reversed(rows))
        buckets: dict[int, dict] = {}
        for row in rows:
            ts = int(row.get("ts") or 0)
            price = row.get("price")
            if not ts or price is None:
                continue
            bucket = (ts // 1000 // interval_sec) * interval_sec
            qty = float(row.get("qty") or 0)
            item = buckets.setdefault(bucket, {
                "time": bucket,
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "volume": 0.0,
            })
            item["high"] = max(item["high"], float(price))
            item["low"] = min(item["low"], float(price))
            item["close"] = float(price)
            item["volume"] += qty
        candles = [buckets[k] for k in sorted(buckets)]
        candles = candles[-360:]

        closes = [float(c["close"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd = [
            (a - b) if a is not None and b is not None else None
            for a, b in zip(ema12, ema26)
        ]
        signal_vals = _ema([v if v is not None else 0.0 for v in macd], 9)
        hist = [
            (m - s) if m is not None and s is not None else None
            for m, s in zip(macd, signal_vals)
        ]

        gains: list[float] = []
        losses: list[float] = []
        rsi: list[float | None] = []
        for idx, close in enumerate(closes):
            if idx == 0:
                gains.append(0.0)
                losses.append(0.0)
                rsi.append(None)
                continue
            diff = close - closes[idx - 1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))
            if idx < 14:
                rsi.append(None)
            else:
                avg_gain = sum(gains[idx - 13:idx + 1]) / 14.0
                avg_loss = sum(losses[idx - 13:idx + 1]) / 14.0
                if avg_loss == 0:
                    rsi.append(100.0)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100.0 - (100.0 / (1.0 + rs)))

        tr_values: list[float] = []
        atr: list[float | None] = []
        for idx, candle in enumerate(candles):
            prev_close = closes[idx - 1] if idx else candle["close"]
            tr = max(
                highs[idx] - lows[idx],
                abs(highs[idx] - prev_close),
                abs(lows[idx] - prev_close),
            )
            tr_values.append(float(tr))
            if idx < 14:
                atr.append(None)
            else:
                atr.append(sum(tr_values[idx - 13:idx + 1]) / 14.0)

        fg = _fetch_fear_greed_history()

        return {
            "ok": True,
            "symbol": symbol,
            "interval_sec": interval_sec,
            "candles": candles,
            "macd": [
                {"time": candles[i]["time"], "macd": macd[i], "signal": signal_vals[i], "hist": hist[i]}
                for i in range(len(candles))
            ],
            "rsi": [
                {"time": candles[i]["time"], "value": rsi[i]}
                for i in range(len(candles))
            ],
            "atr": [
                {"time": candles[i]["time"], "value": atr[i]}
                for i in range(len(candles))
            ],
            "fear_greed": fg,
        }
    except Exception as exc:
        return {"ok": False, "symbol": symbol, "error": str(exc)}
    finally:
        conn.close()


def _v7_features_from_candles(candles: list[dict], interval_sec: int, feature_cols: list[str]):
    import numpy as np
    import pandas as pd

    if not candles:
        return None
    df = pd.DataFrame(candles).copy()
    if df.empty:
        return None

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    if df.empty:
        return None

    # Recreate v7-like base columns.
    df["open_time"] = (pd.to_numeric(df["time"], errors="coerce") * 1000).astype("int64")
    df["close_time"] = (pd.to_numeric(df["time"], errors="coerce") + int(interval_sec) - 1).astype("int64") * 1000
    df["quote_asset_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) * pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df["number_of_trades"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df["taker_buy_base_asset_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) * 0.5
    df["taker_buy_quote_asset_volume"] = pd.to_numeric(df["quote_asset_volume"], errors="coerce").fillna(0.0) * 0.5
    df["ignore"] = 0.0

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    qv = pd.to_numeric(df["quote_asset_volume"], errors="coerce")
    notr = pd.to_numeric(df["number_of_trades"], errors="coerce")

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_12"] = close.pct_change(12)
    df["log_ret_1"] = np.log(close / close.shift(1))

    for w in (5, 10, 20, 50, 100):
        sma = close.rolling(w).mean()
        std = close.rolling(w).std()
        df[f"sma_{w}"] = sma
        df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
        df[f"vol_{w}"] = df["log_ret_1"].rolling(w).std()
        df[f"zscore_{w}"] = (close - sma) / std.replace(0, np.nan)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    df["atr_pct"] = df["atr_14"] / close.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_up"] = bb_mid + 2 * bb_std
    df["bb_dn"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (close - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"]).replace(0, np.nan)

    df["hl_spread"] = (high - low) / close.replace(0, np.nan)
    df["co_spread"] = (close - open_) / open_.replace(0, np.nan)
    df["volume_chg_1"] = volume.pct_change(1)
    df["quote_volume_chg_1"] = qv.pct_change(1)
    df["trades_chg_1"] = notr.pct_change(1)

    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    # Guarantee required columns.
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    out = df[feature_cols].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _predict_v7_decision(symbol: str, pos: int = 0, capital_ratio: float = 1.0) -> dict:
    import numpy as np

    model = _load_v7_model()
    if not model:
        return {"ok": False, "error": f"v7 model not found under {V7_MODEL_DIR}"}

    chart = _build_indicator_chart(symbol)
    if not chart.get("ok"):
        return {"ok": False, "error": chart.get("error", "indicator unavailable")}

    candles = chart.get("candles") or []
    interval_sec = int(chart.get("interval_sec") or 60)
    feature_cols = model["feature_cols"]
    window_size = int(model["window_size"])

    feature_df = _v7_features_from_candles(candles, interval_sec, feature_cols)
    if feature_df is None or len(feature_df) < max(120, window_size):
        return {"ok": False, "error": "not enough candle rows for v7 model"}

    feature_matrix = feature_df.values.astype(np.float32)
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    std = np.asarray(model["feature_std"], dtype=np.float32) + 1e-9
    feature_matrix = ((feature_matrix - mean) / std).astype(np.float32)

    window = feature_matrix[-window_size:].reshape(-1)
    extra = np.array([float(capital_ratio), float(pos)], dtype=np.float32)
    obs = np.concatenate([window, extra]).astype(np.float32)

    W1, W2, W3 = model["weights"]
    h1 = np.maximum(0.0, obs @ np.asarray(W1, dtype=np.float32))
    h2 = np.maximum(0.0, h1 @ np.asarray(W2, dtype=np.float32))
    logits = h2 @ np.asarray(W3, dtype=np.float32)
    logits = np.asarray(logits, dtype=np.float32)

    cls_logits = logits[:3]
    m = np.max(cls_logits)
    probs = np.exp(cls_logits - m)
    probs = probs / np.sum(probs)
    action = int(np.argmax(cls_logits))
    confidence = float(np.max(probs))
    lev_raw = float(1.0 / (1.0 + np.exp(-float(logits[3]))))
    leverage = int(lev_raw * 2) + 1

    if action == 1:
        direction = "long"
        side = "buy"
    elif action == 2:
        direction = "short"
        side = "sell"
    else:
        direction = "flat"
        side = "hold"

    allowed = direction in {"long", "short"} and confidence >= 0.38
    return {
        "ok": True,
        "symbol": symbol,
        "source": "v7_model",
        "model_name": "v7",
        "fold": model.get("fold"),
        "generation": model.get("generation"),
        "model_path": model.get("path"),
        "direction": direction,
        "side": side,
        "action": "HOLD" if not allowed else ("OPEN_LONG" if direction == "long" else "OPEN_SHORT"),
        "allowed": bool(allowed),
        "confidence": confidence,
        "prob_flat": float(probs[0]),
        "prob_long": float(probs[1]),
        "prob_short": float(probs[2]),
        "suggested_leverage": int(leverage),
        "window_size": window_size,
        "feature_count": int(len(feature_cols)),
        "rows_used": int(len(feature_df)),
        "ts": time.time(),
    }


def _predict_v10_decision(symbol: str, interval_sec: int = 3600) -> dict:
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    if symbol != "BTCUSDT":
        return {"ok": False, "error": "v10_dl currently supports BTCUSDT only"}

    df = _load_v10_bundle_df()
    data_source_note = "cached bundle"
    if df is None or df.empty:
        try:
            df = _build_live_v10_bundle(symbol)
            data_source_note = "live kline"
        except Exception as exc:
            return {"ok": False, "error": f"v10 live kline unavailable: {exc}"}
    if df is None or df.empty:
        return {"ok": False, "error": "v10 bundle/live kline not available"}

    model_pack = _load_v10_model()
    if not model_pack:
        # TensorFlow-free fallback: use recent 1h trend bias as v10 proxy brain.
        # This keeps hybrid gate online when DL runtime is unavailable.
        try:
            import numpy as np
            import pandas as pd

            proxy = df.copy()
            if "open_time_1h" in proxy.columns:
                key = pd.to_numeric(proxy["open_time_1h"], errors="coerce")
                proxy["_bar_key"] = key
                proxy = proxy[proxy["_bar_key"].notna()].drop_duplicates(subset=["_bar_key"], keep="last")
                proxy = proxy.sort_values("_bar_key")
            close_col = "close_1h" if "close_1h" in proxy.columns else ("close_30m" if "close_30m" in proxy.columns else None)
            if close_col is None:
                return {"ok": False, "error": "v10 proxy missing close column"}
            s = pd.to_numeric(proxy[close_col], errors="coerce").dropna()
            if len(s) < 80:
                return {"ok": False, "error": "v10 proxy not enough rows"}
            ema_fast = s.ewm(span=12, adjust=False).mean()
            ema_slow = s.ewm(span=48, adjust=False).mean()
            mom = s.pct_change(6).fillna(0.0)
            trend = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(abs(ema_slow.iloc[-1]), 1e-9))
            momentum = float(mom.iloc[-1])
            score = float(np.clip((trend * 120.0) + (momentum * 6.0), -1.0, 1.0))
            confidence = float(np.clip(abs(score), 0.0, 1.0))
            if score >= 0.10:
                direction = "long"
                side = "buy"
            elif score <= -0.10:
                direction = "short"
                side = "sell"
            else:
                direction = "flat"
                side = "hold"
            allowed = direction in {"long", "short"} and confidence >= 0.20
            p_long = float(np.clip(0.5 + score * 0.5, 0.0, 1.0))
            p_short = float(np.clip(1.0 - p_long, 0.0, 1.0))
            p_flat = float(np.clip(1.0 - abs(score), 0.0, 1.0))
            leverage = max(1, min(5, int(1 + confidence * 3)))
            return {
                "ok": True,
                "symbol": symbol,
                "source": "v10_trend_proxy",
                "model_name": "v10_trend_proxy",
                "fold": None,
                "model_path": None,
                "direction": direction,
                "side": side,
                "action": "HOLD" if not allowed else ("OPEN_LONG" if direction == "long" else "OPEN_SHORT"),
                "allowed": bool(allowed),
                "confidence": confidence,
                "prob_flat": p_flat,
                "prob_long": p_long,
                "prob_short": p_short,
                "suggested_leverage": int(leverage),
                "window_size": 80,
                "feature_count": 0,
                "rows_used": int(len(s)),
                "rows_used_total": int(len(df)),
                "interval_sec": int(interval_sec),
                "last_feature_ts": str(proxy["timestamp_dt"].iloc[-1]) if "timestamp_dt" in proxy.columns else "",
                "note": f"tensorflow unavailable; using v10 trend proxy from {data_source_note}",
                "ts": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": f"v10 proxy failed: {exc}"}

    feat_cols = model_pack["feature_cols"]
    data_source = "v10_dl_live_kline"
    live_df = None
    live_error = ""
    try:
        live_df = _build_live_v10_bundle(symbol)
    except Exception as exc:
        live_error = str(exc)
        live_df = None
    if live_df is not None and len(live_df) >= 120:
        df = live_df
    else:
        data_source = "v10_dl_cached_bundle"

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    window_size = 50
    window_df = df.copy()
    interval_sec = int(max(60, min(86400, interval_sec)))
    # For 1h mode, use the latest 50 unique 1h candles.
    if interval_sec >= 3600 and "open_time_1h" in df.columns:
        try:
            key = pd.to_numeric(df["open_time_1h"], errors="coerce")
            window_df = df.copy()
            window_df["_bar_key"] = key
            window_df = window_df[window_df["_bar_key"].notna()].drop_duplicates(subset=["_bar_key"], keep="last")
            window_df = window_df.sort_values("_bar_key")
        except Exception:
            window_df = df.copy()

    window_df = window_df.tail(window_size).copy()
    if len(window_df) < window_size:
        return {
            "ok": False,
            "error": f"v10 requires at least {window_size} rows, got {len(window_df)}",
            "rows_used": int(len(window_df)),
        }

    x = window_df[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = x.to_numpy(dtype=np.float32)
    x = (x - model_pack["mean"]) / model_pack["scale"]
    probs = model_pack["model"].predict(x, verbose=0).reshape(-1)
    p_long = float(np.mean(probs))
    p_short = float(1.0 - p_long)
    confidence = float(max(p_long, p_short))
    row = window_df.iloc[-1]

    if p_long >= 0.55:
        direction = "long"
        side = "buy"
    elif p_long <= 0.45:
        direction = "short"
        side = "sell"
    else:
        direction = "flat"
        side = "hold"

    allowed = direction in {"long", "short"} and confidence >= 0.55
    leverage = max(1, min(10, int(1 + confidence * 4)))

    return {
        "ok": True,
        "symbol": symbol,
        "source": data_source,
        "model_name": "v10_dl",
        "fold": model_pack.get("fold"),
        "model_path": model_pack.get("path"),
        "direction": direction,
        "side": side,
        "action": "HOLD" if not allowed else ("OPEN_LONG" if direction == "long" else "OPEN_SHORT"),
        "allowed": bool(allowed),
        "confidence": confidence,
        "prob_flat": float(max(0.0, 1.0 - abs(p_long - 0.5) * 2.0)),
        "prob_long": p_long,
        "prob_short": p_short,
        "prob_long_last": float(probs[-1]),
        "suggested_leverage": int(leverage),
        "window_size": window_size,
        "feature_count": int(len(feat_cols)),
        "rows_used": int(len(window_df)),
        "rows_used_total": int(len(df)),
        "interval_sec": int(interval_sec),
        "last_feature_ts": str(row.get("timestamp_dt")),
        "kline_source": "binance_futures_klines_1h_clock",
        "fallback_reason": live_error if data_source == "v10_dl_cached_bundle" else "",
        "ts": time.time(),
    }


def _predict_v13_decision(symbol: str) -> dict:
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"

    snap = _build_signal_dashboard(symbol)
    if not snap.get("ok"):
        return {"ok": False, "error": snap.get("error", "snapshot unavailable")}

    rb = evaluate_micro_signal(snap)
    latest = snap.get("latest") or {}
    avg1h = snap.get("averages_1h") or {}
    obi = avg1h.get("obi")
    if obi is None:
        obi = latest.get("obi", 0.0)
    depth = avg1h.get("depth_imbalance")
    if depth is None:
        depth = latest.get("depth_imbalance", 0.0)
    buy_pressure = avg1h.get("buy_pressure")
    if buy_pressure is None:
        buy_pressure = latest.get("buy_pressure", 0.5)

    try:
        obi_f = float(obi or 0.0)
        depth_f = float(depth or 0.0)
        bp_f = float(buy_pressure if buy_pressure is not None else 0.5)
    except Exception:
        obi_f, depth_f, bp_f = 0.0, 0.0, 0.5

    score = max(-1.0, min(1.0, (obi_f * 0.40) + (depth_f * 0.35) + ((bp_f - 0.5) * 2.0 * 0.25)))
    confidence = max(0.0, min(1.0, abs(score) * 1.8))
    if score > 0.08:
        direction = "long"
        side = "buy"
    elif score < -0.08:
        direction = "short"
        side = "sell"
    else:
        direction = "flat"
        side = "hold"

    rb_allowed = bool((rb or {}).get("allowed", False))
    if not rb_allowed:
        direction = "flat"
        side = "hold"

    allowed = rb_allowed and direction in {"long", "short"} and confidence >= 0.28
    leverage = max(1, min(6, int(1 + confidence * 4)))
    p_long = max(0.0, min(1.0, 0.5 + score * 0.5))
    p_short = max(0.0, min(1.0, 1.0 - p_long))
    p_flat = max(0.0, min(1.0, 1.0 - abs(score)))
    action = "HOLD" if not allowed else ("OPEN_LONG" if direction == "long" else "OPEN_SHORT")

    return {
        "ok": True,
        "symbol": symbol,
        "source": "v13_micro_obi",
        "model_name": "v13_micro_obi",
        "direction": direction,
        "side": side,
        "action": action,
        "allowed": bool(allowed),
        "confidence": float(confidence),
        "prob_flat": float(p_flat),
        "prob_long": float(p_long),
        "prob_short": float(p_short),
        "suggested_leverage": int(leverage),
        "gate": {
            "rule_brain_allowed": bool(rb_allowed),
            "rule_blocks": list((rb or {}).get("blocks") or []),
        },
        "ts": time.time(),
    }


def _predict_hybrid_v10_v13(symbol: str, interval_sec: int = 3600) -> dict:
    v10 = _predict_v10_decision(symbol, interval_sec=interval_sec)
    if not v10.get("ok"):
        return {"ok": False, "error": f"v10 unavailable: {v10.get('error', 'unknown')}"}

    v13 = _predict_v13_decision(symbol)
    if not v13.get("ok"):
        return {"ok": False, "error": f"v13 unavailable: {v13.get('error', 'unknown')}"}

    v10_dir = str(v10.get("direction") or "flat").lower()
    v13_dir = str(v13.get("direction") or "flat").lower()
    v10_allowed = bool(v10.get("allowed", False))
    v13_allowed = bool(v13.get("allowed", False))

    gate_reason = ""
    if not v10_allowed or v10_dir == "flat":
        final_dir = "flat"
        gate_reason = "v10 trend gate hold"
    elif not v13_allowed or v13_dir == "flat":
        final_dir = "flat"
        gate_reason = "v13 micro gate hold"
    elif v10_dir != v13_dir:
        final_dir = "flat"
        gate_reason = f"direction conflict: v10={v10_dir}, v13={v13_dir}"
    else:
        final_dir = v10_dir
        gate_reason = "v10 trend + v13 micro agree"

    if final_dir == "long":
        side = "buy"
    elif final_dir == "short":
        side = "sell"
    else:
        side = "hold"

    allowed = final_dir in {"long", "short"}
    confidence = float(min(1.0, max(0.0, (float(v10.get("confidence", 0.0)) * 0.55) + (float(v13.get("confidence", 0.0)) * 0.45))))
    leverage = int(max(1, min(10, min(int(v10.get("suggested_leverage", 1) or 1), int(v13.get("suggested_leverage", 1) or 1)))))
    p_long = float((float(v10.get("prob_long", 0.5)) + float(v13.get("prob_long", 0.5))) / 2.0)
    p_short = float((float(v10.get("prob_short", 0.5)) + float(v13.get("prob_short", 0.5))) / 2.0)
    p_flat = float(max(0.0, min(1.0, 1.0 - abs(p_long - p_short))))
    action = "HOLD" if not allowed else ("OPEN_LONG" if final_dir == "long" else "OPEN_SHORT")

    return {
        "ok": True,
        "symbol": symbol.upper(),
        "source": "hybrid_v10_v13",
        "model_name": "hybrid_v10_v13",
        "direction": final_dir,
        "side": side,
        "action": action,
        "allowed": bool(allowed),
        "confidence": confidence,
        "prob_flat": p_flat,
        "prob_long": p_long,
        "prob_short": p_short,
        "suggested_leverage": leverage,
        "gate_reason": gate_reason,
        "components": {
            "v10": {
                "direction": v10_dir,
                "allowed": v10_allowed,
                "confidence": float(v10.get("confidence", 0.0) or 0.0),
            },
            "v13": {
                "direction": v13_dir,
                "allowed": v13_allowed,
                "confidence": float(v13.get("confidence", 0.0) or 0.0),
            },
        },
        "ts": time.time(),
    }


def _predict_ai_decision(
    symbol: str,
    pos: int = 0,
    capital_ratio: float = 1.0,
    interval_sec: int = 3600,
    model_mode: str = "hybrid_v10_v13",
) -> dict:
    mode = str(model_mode or "hybrid_v10_v13").strip().lower()
    if mode == "v10":
        return _predict_v10_decision(symbol, interval_sec=interval_sec)
    if mode == "v13":
        return _predict_v13_decision(symbol)
    return _predict_hybrid_v10_v13(symbol, interval_sec=interval_sec)


def _read_strategy_params() -> dict:
    if not STRATEGY_PARAMS_PATH.exists():
        return {}
    try:
        payload = json.loads(STRATEGY_PARAMS_PATH.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _interval_key_from_sec(interval_sec: int) -> str:
    if interval_sec >= 86400:
        return "1d"
    if interval_sec >= 14400:
        return "4h"
    if interval_sec >= 3600:
        return "1h"
    if interval_sec >= 900:
        return "15m"
    if interval_sec >= 300:
        return "5m"
    return "1m"


def _strategy_precheck(
    symbol: str,
    *,
    side: str,
    leverage: int,
    interval_sec: int,
    model_mode: str,
    ai_order: bool,
    locked_ai: dict | None = None,
) -> dict:
    params = _read_strategy_params()
    decision = _predict_ai_decision(symbol, interval_sec=interval_sec, model_mode=model_mode)
    reasons: list[str] = []
    checks: list[dict] = []

    max_leverage = int(params.get("max_leverage", 10) or 10)
    if int(leverage) > max_leverage:
        reasons.append(f"leverage {leverage} > strategy max {max_leverage}")
    checks.append({"name": "leverage_cap", "ok": int(leverage) <= max_leverage})

    interval_key = _interval_key_from_sec(interval_sec)
    thresholds = params.get("interval_signal_thresholds") if isinstance(params.get("interval_signal_thresholds"), dict) else {}
    conf_threshold = float(thresholds.get(interval_key, thresholds.get("1h", 0.48)) or 0.48)

    ai_ok = bool(decision.get("ok", False))
    ai_allowed = bool(decision.get("allowed", False))
    ai_dir = str(decision.get("direction") or "flat").lower()
    ai_conf = float(decision.get("confidence", 0.0) or 0.0)
    if ai_order and isinstance(locked_ai, dict):
        ai_ok = True
        ai_dir = str(locked_ai.get("direction") or ai_dir).lower()
        ai_conf = float(locked_ai.get("confidence", ai_conf) or ai_conf)
        ai_allowed = bool(locked_ai.get("allowed", ai_allowed))

    if ai_order:
        if not ai_ok:
            reasons.append(f"ai decision unavailable: {decision.get('error', 'unknown')}")
        elif not ai_allowed or ai_dir not in {"long", "short"}:
            reasons.append("ai gate hold")
        expected_side = "buy" if ai_dir == "long" else ("sell" if ai_dir == "short" else "")
        if expected_side and side.lower() != expected_side:
            reasons.append(f"side mismatch: order={side.lower()} ai={expected_side}")
        if ai_conf < conf_threshold:
            reasons.append(f"ai confidence {ai_conf:.3f} < threshold {conf_threshold:.3f}")

    checks.append({"name": "ai_available", "ok": (ai_ok if ai_order else True)})
    checks.append({"name": "ai_direction_allowed", "ok": ((ai_allowed and ai_dir in {'long', 'short'}) if ai_order else True)})
    checks.append({"name": "ai_confidence", "ok": ((ai_conf >= conf_threshold) if ai_order else True)})

    return {
        "ok": True,
        "allowed": len(reasons) == 0,
        "reasons": reasons,
        "checks": checks,
        "params_path": str(STRATEGY_PARAMS_PATH),
        "params_exists": STRATEGY_PARAMS_PATH.exists(),
        "params": {
            "generated_at_utc": params.get("generated_at_utc"),
            "max_leverage": max_leverage,
            "drawdown_stop": params.get("drawdown_stop"),
            "interval_signal_thresholds": thresholds,
            "active_interval_key": interval_key,
            "active_conf_threshold": conf_threshold,
        },
        "ai": {
            "ok": ai_ok,
            "model_mode": model_mode,
            "direction": ai_dir,
            "allowed": ai_allowed,
            "confidence": ai_conf,
            "gate_reason": decision.get("gate_reason"),
        },
        "ts": time.time(),
    }


@app.route("/api/trading/indicators/<symbol>")
def route_trading_indicators(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return jsonify(_build_indicator_chart(symbol))


@app.route("/api/news/keywords")
def route_news_keywords():
    return jsonify(_build_keyword_news_payload())


@app.route("/api/pipeline/nightly-status")
def route_pipeline_nightly_status():
    return jsonify(_nightly_pipeline_status())

@app.route("/api/nightly-status")
def route_nightly_status_alias():
    return jsonify(_nightly_pipeline_status())

@app.route("/api/pipeline/status")
def route_pipeline_status_alias():
    return jsonify(_nightly_pipeline_status())


@app.route("/api/trading/state", methods=["GET", "POST"])
def route_trading_state():
    if request.method == "GET":
        state = _read_trading_state()
        state["ok"] = True
        return jsonify(state)

    body = request.get_json(force=True, silent=True) or {}
    state = _read_trading_state()
    if isinstance(body.get("prefs"), dict):
        state["prefs"] = body["prefs"]
    if isinstance(body.get("equitySnapshots"), list):
        state["equitySnapshots"] = body["equitySnapshots"]
    if isinstance(body.get("tradeRecords"), list):
        state["tradeRecords"] = body["tradeRecords"]
    _write_trading_state(state)
    state["ok"] = True
    return jsonify(state)


@app.route("/api/strategy/params/<symbol>")
def route_strategy_params(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    try:
        interval_sec = int(request.args.get("interval_sec", 3600) or 3600)
    except Exception:
        interval_sec = 3600
    model_mode = str(request.args.get("model_mode", "hybrid_v10_v13") or "hybrid_v10_v13")
    dec = _predict_ai_decision(symbol, interval_sec=interval_sec, model_mode=model_mode)
    ai_dir = str(dec.get("direction") or "flat").lower()
    side = "buy" if ai_dir == "long" else ("sell" if ai_dir == "short" else "buy")
    locked_ai = {
        "direction": ai_dir,
        "confidence": float(dec.get("confidence", 0.0) or 0.0),
        "allowed": bool(dec.get("allowed", False)),
    }
    pre = _strategy_precheck(
        symbol,
        side=side,
        leverage=1,
        interval_sec=interval_sec,
        model_mode=model_mode,
        ai_order=False,
        locked_ai=locked_ai,
    )
    return jsonify(pre)


@app.route("/api/trading/precheck", methods=["POST"])
def route_trading_precheck():
    body = request.get_json(force=True, silent=True) or {}
    symbol = str(body.get("symbol") or "BTCUSDT").upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    side = str(body.get("side") or "buy").lower()
    try:
        leverage = int(body.get("leverage", 1) or 1)
    except Exception:
        leverage = 1
    try:
        interval_sec = int(body.get("interval_sec", 3600) or 3600)
    except Exception:
        interval_sec = 3600
    model_mode = str(body.get("model_mode", "hybrid_v10_v13") or "hybrid_v10_v13")
    ai_order = bool(body.get("ai_order", False))
    locked_ai = body.get("locked_ai") if isinstance(body.get("locked_ai"), dict) else None
    return jsonify(
        _strategy_precheck(
            symbol,
            side=side,
            leverage=max(1, leverage),
            interval_sec=max(60, interval_sec),
            model_mode=model_mode,
            ai_order=ai_order,
            locked_ai=locked_ai,
        )
    )


@app.route("/api/okx/status")
def route_okx_status():
    return jsonify(okx_client.get_status())


@app.route("/api/okx/ticker/<inst_id>")
def route_okx_ticker(inst_id):
    try:
        return jsonify(okx_client.get_ticker(inst_id.upper()))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/okx/account")
def route_okx_account():
    try:
        return jsonify(okx_client.get_account())
    except Exception as exc:
        return jsonify({"error": str(exc), "status": okx_client.get_status()}), 400


@app.route("/api/okx/orders")
def route_okx_orders():
    try:
        inst_id = request.args.get("instId") or None
        return jsonify(okx_client.get_open_orders(inst_id))
    except Exception as exc:
        return jsonify({"error": str(exc), "status": okx_client.get_status()}), 400


@app.route("/api/okx/order", methods=["POST"])
def route_okx_order():
    try:
        body = request.get_json(force=True, silent=False) or {}
        confirm_live = body.get("confirm") == "OKX LIVE"
        order = body.get("order") or {}
        try:
            order_leverage = int(body.get("leverage") or order.get("leverage") or 1)
        except Exception:
            order_leverage = 1
        order_leverage = max(1, order_leverage)
        if bool(body.get("enforce_strategy_gate", False)):
            symbol = str(order.get("instId") or "").upper()
            side = str(order.get("side") or "buy").lower()
            gate = _strategy_precheck(
                "BTCUSDT" if symbol.startswith("BTC-") else ("ADAUSDT" if symbol.startswith("ADA-") else "BTCUSDT"),
                side=side,
                leverage=order_leverage,
                interval_sec=int(body.get("interval_sec", 3600) or 3600),
                model_mode=str(body.get("model_mode", "hybrid_v10_v13") or "hybrid_v10_v13"),
                ai_order=bool(body.get("ai_order", False)),
            )
            if not gate.get("allowed", False):
                return jsonify({"blocked": True, "message": "strategy gate blocked", "precheck": gate})
        leverage_result = None
        if confirm_live and not bool(order.get("reduceOnly")):
            leverage_result = okx_client.set_leverage(
                str(order.get("instId") or ""),
                lever=order_leverage,
                mgn_mode=str(order.get("tdMode") or "cross"),
                pos_side=str(order.get("posSide") or ""),
                confirm_live=confirm_live,
            )
            if leverage_result.get("blocked"):
                return jsonify({"blocked": True, "message": "set leverage blocked", "set_leverage": leverage_result})
        result = okx_client.place_order(order, confirm_live=confirm_live)
        result["leverage"] = order_leverage
        if leverage_result is not None:
            result["set_leverage"] = leverage_result
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "status": okx_client.get_status()}), 400


@app.route("/api/okx/cancel-order", methods=["POST"])
def route_okx_cancel_order():
    try:
        body = request.get_json(force=True, silent=False) or {}
        confirm_live = body.get("confirm") == "OKX LIVE"
        order = body.get("order") or {}
        return jsonify(okx_client.cancel_order(order, confirm_live=confirm_live))
    except Exception as exc:
        return jsonify({"error": str(exc), "status": okx_client.get_status()}), 400


@app.route("/api/okx/close-position", methods=["POST"])
def route_okx_close_position():
    try:
        body = request.get_json(force=True, silent=False) or {}
        confirm_live = body.get("confirm") == "OKX LIVE"
        position = body.get("position") or {}
        return jsonify(okx_client.close_position(position, confirm_live=confirm_live))
    except Exception as exc:
        return jsonify({"error": str(exc), "status": okx_client.get_status()}), 400


@app.route("/")
def route_index():
    return send_from_directory(".", "index.html")


@app.route("/trading.html")
def route_trading():
    return send_from_directory("trading", "trading.html")


@app.route("/signal_dashboard.html")
def route_signal_dashboard_page():
    return send_from_directory("control", "signal_dashboard.html")


@app.route("/<path:filename>")
def route_static(filename):
    return send_from_directory(".", filename)


@app.after_request
def add_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Binance Futures Dashboard Server")
    print("  Open: http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
