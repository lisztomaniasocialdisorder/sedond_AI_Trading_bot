"""
全週期蒸餾再訓練腳本
- 5 個週期依序執行 run_quick_update（改用 run_pipeline 重新訓練模型）
- Teacher soft labels 已存在 outputs/teacher_soft_labels_BTCUSDT_{interval}.csv
- 使用 MAX_TRAIN_ROWS=0（全量），TORCH_EPOCHS=20（更充分訓練）
"""
import os, sys, time
from pathlib import Path

# ── 設定環境變數，覆蓋 config 預設值 ────────────────────────────────────────
os.environ["MAX_TRAIN_ROWS"] = "0"        # 全量資料
os.environ["TORCH_EPOCHS"] = "20"         # 預設 5 → 20，更充分學習
os.environ["TORCH_BATCH_SIZE"] = "2048"   # 較大 batch，穩定梯度

INTERVALS = ["5m", "15m", "30m", "1h", "1d"]

def progress(pct: int, msg: str) -> None:
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:3d}%  {msg}", end="", flush=True)
    if pct >= 100:
        print()

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import run_pipeline

total_start = time.time()
results_summary = []

for interval in INTERVALS:
    print(f"\n{'='*60}")
    print(f"  訓練 Student [{interval}]  (Teacher 蒸餾 + 全量資料 + 20 epochs)")
    print(f"{'='*60}")
    t0 = time.time()

    # Verify soft labels exist
    sl_path = Path(f"outputs/teacher_soft_labels_BTCUSDT_{interval}.csv")
    if not sl_path.exists():
        print(f"  ⚠️  找不到 soft labels: {sl_path.name}，跳過")
        continue

    try:
        result = run_pipeline(
            force_full_refresh=False,   # 使用現有 OHLCV cache，不重新下載
            progress_cb=progress,
            symbol="BTCUSDT",
            interval=interval,
        )
        elapsed = time.time() - t0
        bt = result.get("backtest_report", {})
        train = result.get("train_metrics", {})
        promoted = result.get("model_promotion", {}).get("promoted", "N/A")
        distilled = train.get("distillation_applied", False)

        print(f"\n  ✅ [{interval}] 完成 ({elapsed:.0f}s)")
        print(f"     蒸餾: {'✓' if distilled else '✗'}  |  "
              f"晉升: {'✓' if promoted else '✗'}  |  "
              f"勝率: {bt.get('win_rate', 0):.2%}  |  "
              f"報酬: {bt.get('total_return', 0):.2%}  |  "
              f"DD: {bt.get('max_drawdown', 0):.2%}")
        results_summary.append({
            "interval": interval,
            "ok": True,
            "distilled": distilled,
            "promoted": promoted,
            "win_rate": bt.get("win_rate", 0),
            "total_return": bt.get("total_return", 0),
            "max_drawdown": bt.get("max_drawdown", 0),
            "elapsed_s": int(elapsed),
        })

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ [{interval}] 失敗 ({elapsed:.0f}s): {e}")
        results_summary.append({"interval": interval, "ok": False, "error": str(e)})

# ── 最終摘要 ─────────────────────────────────────────────────────────────────
total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"  全週期蒸餾再訓練完成  (總耗時 {total_elapsed/60:.1f} 分鐘)")
print(f"{'='*60}")
for r in results_summary:
    if r["ok"]:
        tag = ("✅" if r.get("promoted") else "⏸")
        print(f"  {tag} {r['interval']:4s}  distilled={r['distilled']}  "
              f"promoted={r['promoted']}  win={r['win_rate']:.1%}  "
              f"ret={r['total_return']:.1%}  dd={r['max_drawdown']:.1%}  "
              f"({r['elapsed_s']}s)")
    else:
        print(f"  ❌ {r['interval']:4s}  ERROR: {r['error'][:80]}")
