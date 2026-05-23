# 另存成：build_teacher_training_from_data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.config import Settings
from src.features import add_technical_features, build_labels
from src.data_sources import merge_event_features, interval_to_seconds

BASE = Path(r"C:\Users\brian\OneDrive\桌面\btc-1-k-ai-100-ma")
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs" / "teacher_training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

symbol = "BTCUSDT"
intervals = ["5m", "15m", "30m", "1h", "1d"]

for tf in intervals:
    p = DATA_DIR / f"{symbol}_{tf}_ohlcv.csv"
    if not p.exists():
        print(f"[MISS] {p.name}")
        continue

    cfg = Settings(symbol=symbol, interval=tf)
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)

    with_events = merge_event_features(df, cfg, fast_mode=True)
    feat = add_technical_features(with_events)

    bars = int(round((cfg.future_horizon_hours * 3600) / interval_to_seconds(tf)))
    labeled = build_labels(feat, horizon_bars=max(1, bars), long_th=cfg.long_threshold, short_th=cfg.short_threshold)

    # teacher 訓練必要欄位補齊
    if "label" not in labeled.columns:
        labeled["label"] = 0
    if "target_leverage" not in labeled.columns:
        labeled["target_leverage"] = 1.0
    if "future_ret" not in labeled.columns:
        labeled["future_ret"] = labeled["close"].pct_change().shift(-1)

    labeled["label"] = pd.to_numeric(labeled["label"], errors="coerce").fillna(0).round().clip(-1, 1).astype(int)
    labeled["target_leverage"] = pd.to_numeric(labeled["target_leverage"], errors="coerce").fillna(1.0).clip(lower=1.0)
    labeled["future_ret"] = pd.to_numeric(labeled["future_ret"], errors="coerce").fillna(0.0)

    out = OUT_DIR / f"teacher_train_{symbol}_{tf}.csv"
    labeled.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] {tf}: {len(labeled):,} rows -> {out.name}")

print("Done:", OUT_DIR)