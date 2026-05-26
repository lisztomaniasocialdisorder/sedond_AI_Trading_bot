#!/usr/bin/env python3
"""
pipeline.py
===========
一鍵執行 Walk-Forward 30秒預測 Pipeline。

用法
----
  python pipeline.py --symbol BTCUSDT
  python pipeline.py --symbol ADAUSDT --model lgbm --fwd-sec 30 --train-hours 6
  python pipeline.py --symbol BTCUSDT --model ridge --list-info
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root discovery ────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
PROJ_ROOT   = HERE.parent
DB_TEMPLATE = PROJ_ROOT / "harvesters" / "{coin}_harvester" / "raw_db" / "microstructure_{coin}.db"
RESULTS_DIR = HERE / "results"

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Walk-Forward 30s price prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol",      default="BTCUSDT",
                   help="Trading symbol, e.g. BTCUSDT or ADAUSDT")
    p.add_argument("--model",       default="lgbm",  choices=["ridge", "lgbm"],
                   help="Model to use")
    p.add_argument("--fwd-sec",     type=int,   default=30,
                   help="Forward return horizon in seconds")
    p.add_argument("--train-hours", type=float, default=4.0,
                   help="Training window length in hours")
    p.add_argument("--test-min",    type=float, default=30.0,
                   help="Test window length in minutes per fold")
    p.add_argument("--gap-sec",     type=int,   default=60,
                   help="Gap between train end and test start (seconds)")
    p.add_argument("--rolling",     action="store_true",
                   help="Use rolling window instead of expanding")
    p.add_argument("--list-info",   action="store_true",
                   help="Print DB info (time range, row counts) and exit")
    p.add_argument("--db-path",     default=None,
                   help="Override DB path (auto-detected if omitted)")
    return p.parse_args()


def _resolve_db(symbol: str, override: str | None) -> Path:
    if override:
        return Path(override)
    coin = symbol.upper().replace("USDT", "")
    return Path(str(DB_TEMPLATE).format(coin=coin))


# ── List info ─────────────────────────────────────────────────────────────────

def print_db_info(db_path: Path):
    import sqlite3
    if not db_path.exists():
        print(f"[pipeline] DB not found: {db_path}")
        return

    size_mb = db_path.stat().st_size / 1_048_576
    print(f"\n{'='*55}")
    print(f"  DB: {db_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"{'='*55}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        for tbl in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                print(f"  {tbl:<25} {n:>12,} rows")
            except Exception:
                pass

        # Time range
        try:
            row = conn.execute("SELECT MIN(trade_ts), MAX(trade_ts) FROM trades").fetchone()
            if row and row[0]:
                t0 = pd.Timestamp(row[0], unit="ms", tz="UTC")
                t1 = pd.Timestamp(row[1], unit="ms", tz="UTC")
                dur = t1 - t0
                print(f"\n  Time range : {t0.strftime('%Y-%m-%d %H:%M')} → {t1.strftime('%Y-%m-%d %H:%M')}")
                print(f"  Duration   : {dur}")
                hrs = dur.total_seconds() / 3600
                print(f"  Available folds (4h train / 30m test): ~{max(0, int((hrs - 4) / 0.5))}")
        except Exception:
            pass
    finally:
        conn.close()
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    db_path = _resolve_db(args.symbol, args.db_path)

    if args.list_info:
        print_db_info(db_path)
        return

    if not db_path.exists():
        print(f"[pipeline] ERROR: DB not found at {db_path}")
        print(f"           Set --db-path to specify the path manually.")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  Walk-Forward Pipeline")
    print(f"  Symbol      : {args.symbol}")
    print(f"  Model       : {args.model.upper()}")
    print(f"  Forward sec : {args.fwd_sec}s")
    print(f"  Train window: {args.train_hours}h")
    print(f"  Test window : {args.test_min}min per fold")
    print(f"  Window type : {'rolling' if args.rolling else 'expanding'}")
    print(f"  DB          : {db_path}")
    print(f"{'='*55}\n")

    # Import here so sys.path adjustments (if any) take effect
    from walk_forward import WalkForwardEngine
    from evaluate import summarise, plot_results

    engine = WalkForwardEngine(
        db_path         = db_path,
        model_name      = args.model,
        fwd_seconds     = args.fwd_sec,
        train_hours     = args.train_hours,
        test_minutes    = args.test_min,
        gap_seconds     = args.gap_sec,
        expanding       = not args.rolling,
        save_predictions= True,
    )

    results_df, feat_names, importances = engine.run()

    # ── Print summary ─────────────────────────────────────────────────────────
    summary = summarise(results_df)
    print(f"\n{'='*55}")
    print("  WALK-FORWARD SUMMARY")
    print(f"{'='*55}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<25} {v:>10.4f}")
        else:
            print(f"  {k:<25} {v:>10}")
    print(f"{'='*55}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.symbol}_{args.model}_fwd{args.fwd_sec}s"

    results_path = RESULTS_DIR / f"wf_results_{tag}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[pipeline] Results saved → {results_path}")

    summary_path = RESULTS_DIR / f"wf_summary_{tag}.csv"
    summary.to_csv(summary_path, header=["value"])
    print(f"[pipeline] Summary saved → {summary_path}")

    # Feature importance CSV
    if feat_names and importances is not None:
        fi_df = pd.DataFrame({
            "feature":    feat_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi_path = RESULTS_DIR / f"feature_importance_{tag}.csv"
        fi_df.to_csv(fi_path, index=False)
        print(f"[pipeline] Feature importances → {fi_path}")
        print("\n  Top 10 Features:")
        for _, row in fi_df.head(10).iterrows():
            print(f"    {row['feature']:<30} {row['importance']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(
        results_df,
        feature_names = feat_names,
        importances   = importances,
        output_dir    = RESULTS_DIR,
    )

    print("\n[pipeline] Done.")


if __name__ == "__main__":
    main()
