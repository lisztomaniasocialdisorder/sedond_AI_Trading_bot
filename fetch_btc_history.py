"""
BTC 歷史 K 線 + 技術特徵自動下載腳本
來源：Binance 公開 API（不需要 API Key）
每週日凌晨 3:00 由 Windows 工作排程器自動執行

輸出：btc_data/BTCUSDT_{interval}_features.csv
  ├─ 原始 OHLCV + Binance 擴充欄位
  ├─ ATR_14, atr_pct
  ├─ RSI, MACD, Bollinger Bands, EMA/SMA
  ├─ ADX, DI+, DI-, Regime
  ├─ SNR 支撐/阻力特徵
  └─ 其他技術特徵（與 Student 訓練完全一致，無未來資訊）

注意：此腳本直接使用 src/features.py 的計算邏輯，
      確保 Teacher 和 Student 的特徵矩陣 100% 相同。
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np

# ── 路徑設定：把專案根目錄加入 sys.path，才能 import src.features ───────────
PROJECT_ROOT = Path(r"C:\Users\brian\OneDrive\桌面\btc-1-k-ai-100-ma")
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "btc_data"
LOG_PATH   = OUTPUT_DIR / "download_log.txt"
BASE_URL   = "https://api.binance.com/api/v3/klines"
SYMBOL     = "BTCUSDT"

# 各週期目標筆數
TARGETS: dict[str, int] = {
    "5m":  300_000,
    "15m": 200_000,
    "30m": 150_000,
    "1h":  100_000,
    "1d":  5_000,
}

BATCH_SIZE = 1000   # Binance 每次最多 1000 根

# Binance klines 的 12 個欄位
_BINANCE_COLS = [
    "open_time",                   # 0  開盤時間 (ms)
    "open", "high", "low",         # 1-3
    "close",                       # 4
    "volume",                      # 5  base asset volume（BTC）
    "close_time",                  # 6  收盤時間 (ms)
    "quote_asset_volume",          # 7  報價資產成交量（USDT）
    "number_of_trades",            # 8  成交筆數
    "taker_buy_base",              # 9  主動買方 base 量（BTC）
    "taker_buy_quote",             # 10 主動買方 quote 量（USDT）
    "ignore",                      # 11
]

# ── 日誌設定 ─────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Binance API 下載 ──────────────────────────────────────────────────────────
def _get_batch(interval: str, end_ms: int | None, limit: int) -> list:
    """單次 API 呼叫，最多 limit 根 K 線，5 次重試。"""
    params: dict = {"symbol": SYMBOL, "interval": interval, "limit": limit}
    if end_ms is not None:
        params["endTime"] = end_ms

    for attempt in range(5):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"  Retry {attempt+1}/5 ({e})，等待 {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"無法取得 {interval} 資料（重試 5 次失敗）")


def fetch_raw(interval: str, total: int) -> pd.DataFrame:
    """
    從現在往前抓 total 根 K 線（含 Binance 全部 12 欄）。
    回傳排序好的 DataFrame，欄位已命名。
    """
    all_rows: list = []
    end_ms: int | None = None
    remaining = total

    log.info(f"[{interval}] 開始下載 {total:,} 根 K 線...")

    while remaining > 0:
        limit = min(BATCH_SIZE, remaining)
        batch = _get_batch(interval, end_ms, limit)
        if not batch:
            log.warning(f"[{interval}] Binance 回傳空資料，停止")
            break

        all_rows = batch + all_rows      # 前置（往舊的方向累積）
        end_ms   = int(batch[0][0]) - 1  # 下一次從更早的時間點開始
        remaining -= len(batch)

        fetched = total - remaining
        log.info(f"  [{interval}] {fetched:,}/{total:,} ({fetched/total*100:.0f}%)")
        time.sleep(0.12)   # Binance rate limit

    # 只取最後 total 根（以防多抓）
    all_rows = all_rows[-total:]

    df = pd.DataFrame(all_rows, columns=_BINANCE_COLS)

    # 型別轉換
    df["timestamp"] = (
        pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    )
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")

    df = (
        df.drop(columns=["open_time", "close_time", "ignore"], errors="ignore")
          .drop_duplicates("timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    log.info(f"[{interval}] 下載完成：{len(df):,} 根  "
             f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
    return df


# ── 特徵計算（直接呼叫 src/features.py，與 Student 完全一致）────────────────
def compute_features(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    呼叫專案的 add_technical_features，
    計算所有技術特徵（無未來資訊 / 無 lookahead bias）。
    """
    try:
        from src.features import add_technical_features
    except ImportError as e:
        log.error(f"無法 import src.features: {e}")
        log.error("請確認在專案根目錄執行此腳本")
        raise

    # timestamp 欄位轉為 ISO 字串，才能讓 features.py 的 _infer_bar_seconds 正確解析
    df_in = df.copy()
    df_in["timestamp"] = df_in["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    log.info(f"[{interval}] 計算技術特徵（src.features.add_technical_features）...")
    df_feat = add_technical_features(df_in)

    # 把 timestamp 轉回 ISO UTC 字串格式（與 pipeline 一致）
    if pd.api.types.is_datetime64_any_dtype(df_feat["timestamp"]):
        df_feat["timestamp"] = df_feat["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # 移除包含未來資訊的欄位（防止 Teacher 洩漏）
    future_cols = {"future_ret", "label", "target_leverage"}
    df_feat = df_feat.drop(columns=[c for c in future_cols if c in df_feat.columns])

    n_feat = len(df_feat.columns) - 1   # 扣掉 timestamp
    log.info(f"[{interval}] 特徵計算完成：{n_feat} 個特徵  "
             f"（含 ATR, RSI, MACD, BB, ADX, SNR, Regime...）")
    return df_feat


# ── 主程式 ────────────────────────────────────────────────────────────────────
def main() -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(f"{'='*60}")
    log.info(f"  BTC 歷史資料下載 + 特徵計算  {now_str}")
    log.info(f"{'='*60}")

    results = []

    for interval, total in TARGETS.items():
        try:
            # ① 下載原始 K 線（OHLCV + Binance 擴充欄位）
            df_raw = fetch_raw(interval, total)

            # ② 儲存原始 OHLCV（備用）
            raw_path = OUTPUT_DIR / f"BTCUSDT_{interval}_raw.csv"
            df_raw.to_csv(raw_path, index=False, encoding="utf-8")

            # ③ 計算技術特徵（與 Student 完全一致的特徵矩陣）
            df_feat = compute_features(df_raw, interval)

            # ④ 儲存特徵矩陣（Teacher 訓練用）
            feat_path = OUTPUT_DIR / f"BTCUSDT_{interval}_features.csv"
            df_feat.to_csv(feat_path, index=False, encoding="utf-8")

            kb_raw  = raw_path.stat().st_size // 1024
            kb_feat = feat_path.stat().st_size // 1024
            n_feat  = len(df_feat.columns) - 1
            log.info(f"[{interval}] 儲存完成")
            log.info(f"  raw:      {raw_path.name}  ({kb_raw:,} KB)")
            log.info(f"  features: {feat_path.name}  ({kb_feat:,} KB)  {n_feat} 個特徵")
            results.append((interval, len(df_feat), n_feat, "OK"))

        except Exception as e:
            log.error(f"[{interval}] 失敗: {e}", exc_info=True)
            results.append((interval, 0, 0, f"FAIL: {e}"))

    log.info(f"{'='*60}")
    log.info("  全部完成")
    log.info(f"{'='*60}")
    for iv, rows, feats, status in results:
        log.info(f"  {iv:4s}  {rows:>7,} 筆  {feats:>3} 特徵  {status}")

    log.info(f"  輸出目錄: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
