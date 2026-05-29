from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "N/A"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |"]
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(_fmt(value, 4))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def analyze_experiment(experiment_dir: Path, feature_meta: Path | None = None) -> str:
    summary = _read_parquet(experiment_dir / "summary.parquet")
    validation = _read_parquet(experiment_dir / "validation_log.parquet")
    generation = _read_parquet(experiment_dir / "generation_log.parquet")
    baseline = _read_parquet(experiment_dir / "baseline.parquet")

    boss_path = experiment_dir / "boss.pkl"
    boss: dict[str, Any] = {}
    if boss_path.exists():
        with boss_path.open("rb") as fh:
            payload = pickle.load(fh)
        boss = payload if isinstance(payload, dict) else {}

    meta: dict[str, Any] = {}
    if feature_meta and feature_meta.exists():
        meta = json.loads(feature_meta.read_text(encoding="utf-8-sig"))

    lines: list[str] = []
    lines.append("# AI Brain Analysis")
    lines.append("")
    lines.append(f"- Experiment: `{experiment_dir}`")
    lines.append(f"- Boss file: {'present' if boss_path.exists() else 'missing'}")
    if meta:
        lines.append(f"- Feature rows: `{meta.get('rows')}`")
        lines.append(f"- Feature columns: `{meta.get('columns')}`")
        lines.append(f"- Label horizon sec: `{meta.get('label_horizon_sec')}`")
    lines.append("")

    status = "UNKNOWN"
    decision = "Do not connect to live auto trading yet."
    if summary.empty:
        status = "NO_SUMMARY"
    elif "status" in summary.columns and str(summary.iloc[-1].get("status", "")) == "no_valid_boss":
        status = "NO_VALID_BOSS"
        decision = "Training correctly rejected all candidates."
    else:
        row = summary.iloc[-1]
        valid_excess = float(row.get("valid_excess_vs_flat", 0.0) or 0.0)
        test_excess = float(row.get("test_excess_vs_flat", 0.0) or 0.0)
        test_death = float(row.get("test_death_rate", 1.0) or 1.0)
        if valid_excess > 0 and test_excess > 0 and test_death == 0:
            status = "PROMISING"
            decision = "Candidate may be used for limited simulation only, still behind a rule guard."
        elif test_death == 0:
            status = "ALIVE_BUT_NOT_PROFITABLE"
        else:
            status = "UNSAFE"

        lines.append("## Summary")
        for key in [
            "best_generation",
            "best_colony",
            "best_valid_robust",
            "valid_capital",
            "valid_excess_vs_flat",
            "valid_entropy",
            "valid_effective_trades",
            "valid_direction_share",
            "test_capital",
            "test_excess_vs_flat",
            "test_entropy",
            "test_effective_trades",
            "test_direction_share",
            "test_death_rate",
        ]:
            if key in row:
                value = row.get(key)
                lines.append(f"- {key}: `{_fmt(value, 4) if isinstance(value, float) else value}`")
        lines.append("")

    lines.append("## Verdict")
    lines.append(f"- Status: `{status}`")
    lines.append(f"- Decision: {decision}")
    lines.append("")

    if boss:
        lines.append("## Boss Metadata")
        lines.append(f"- Version: `{boss.get('version', 'unknown')}`")
        lines.append(f"- Window size: `{boss.get('window_size')}`")
        lines.append(f"- Input dim: `{boss.get('input_dim')}`")
        lines.append(f"- Feature count: `{len(boss.get('feature_cols', []))}`")
        best_valid = boss.get("best_valid_summary")
        if isinstance(best_valid, dict):
            lines.append(f"- Best valid excess vs flat: `{_fmt(best_valid.get('excess_vs_flat'), 4)}`")
            lines.append(f"- Best valid long/short: `{_fmt(best_valid.get('long_count'), 2)} / {_fmt(best_valid.get('short_count'), 2)}`")
        lines.append("")

    if not validation.empty:
        lines.append("## Validation Log")
        cols = [c for c in [
            "generation", "colony", "valid_capital", "valid_excess_vs_flat",
            "valid_entropy", "valid_effective_trades", "valid_direction_share",
            "valid_death_rate",
        ] if c in validation.columns]
        lines.append(_markdown_table(validation[cols]))
        lines.append("")

    if not baseline.empty:
        lines.append("## Baseline")
        lines.append(_markdown_table(baseline))
        lines.append("")

    if not generation.empty:
        best_train_excess = generation.get("best_train_excess_vs_flat")
        if best_train_excess is not None:
            lines.append("## Training Notes")
            lines.append(f"- Best train excess max: `{_fmt(best_train_excess.max(), 4)}`")
            lines.append(f"- Best train excess last: `{_fmt(best_train_excess.iloc[-1], 4)}`")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v8/v8b AI brain outputs.")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--feature-meta", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = analyze_experiment(args.experiment_dir, args.feature_meta)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
