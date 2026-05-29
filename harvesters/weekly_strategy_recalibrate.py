#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = _read_json(root / "outputs" / "report_BTCUSDT_1h.json")
    if not report:
        report = _read_json(root / "outputs" / "report.json")
    bt = report.get("backtest_report", report if isinstance(report, dict) else {})

    win_rate = float(bt.get("win_rate", 0.0) or 0.0)
    max_dd = abs(float(bt.get("max_drawdown", 0.0) or 0.0))
    pf = float(bt.get("profit_factor", 1.0) or 1.0)

    base_signal_1h = 0.48
    if win_rate >= 0.56 and pf >= 1.15 and max_dd <= 0.20:
        signal_1h = base_signal_1h - 0.01
        max_leverage = 12
        drawdown_stop = 0.32
    elif win_rate <= 0.50 or pf < 1.0 or max_dd >= 0.30:
        signal_1h = base_signal_1h + 0.03
        max_leverage = 6
        drawdown_stop = 0.25
    else:
        signal_1h = base_signal_1h
        max_leverage = 10
        drawdown_stop = 0.30

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str((root / "outputs" / "report_BTCUSDT_1h.json").resolve()),
        "long_threshold": 0.005,
        "short_threshold": -0.005,
        "drawdown_stop": float(_clamp(drawdown_stop, 0.20, 0.40)),
        "max_leverage": int(max(3, min(20, max_leverage))),
        "interval_signal_thresholds": {
            "5m": 0.60,
            "15m": 0.55,
            "30m": 0.52,
            "1h": float(_clamp(signal_1h, 0.42, 0.58)),
            "1d": 0.42,
        },
        "stats": {
            "win_rate": win_rate,
            "max_drawdown_abs": max_dd,
            "profit_factor": pf,
        },
    }

    out = root / "outputs" / "strategy_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote strategy params => {out}")
    print(json.dumps(payload["interval_signal_thresholds"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
