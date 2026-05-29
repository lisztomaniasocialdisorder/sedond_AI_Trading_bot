from __future__ import annotations

from typing import Any


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:
        return default
    return out


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def evaluate_micro_signal(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Conservative rule brain for live microstructure gating."""
    latest = snapshot.get("latest") if isinstance(snapshot, dict) else {}
    latest = latest if isinstance(latest, dict) else {}
    stats = snapshot.get("stats") if isinstance(snapshot, dict) else {}
    stats = stats if isinstance(stats, dict) else {}

    obi = _num(latest.get("obi"))
    depth = _num(latest.get("depth_imbalance"))
    buy_pressure = _num(latest.get("buy_pressure"), 0.5)
    spread_bps = abs(_num(latest.get("spread_bps")))
    trades_1m = _num(stats.get("trades_1m"))
    volume_1m = _num(stats.get("volume_1m"))

    flow = (buy_pressure - 0.5) * 2.0
    raw_score = (obi * 0.38) + (depth * 0.34) + (flow * 0.28)
    score = _clip(raw_score, -1.0, 1.0)

    reasons: list[str] = []
    blocks: list[str] = []

    if spread_bps > 8.0:
        blocks.append("spread_too_wide")
    elif spread_bps > 2.5:
        score *= 0.72
        reasons.append("spread widened, score discounted")

    if trades_1m < 5:
        blocks.append("too_few_recent_trades")
    if volume_1m <= 0:
        blocks.append("no_recent_volume")

    confidence = _clip(abs(score) * 1.65, 0.0, 1.0)
    if confidence < 0.12:
        blocks.append("low_confidence")

    if score > 0.08 and confidence >= 0.28:
        direction = "long"
        action = "OPEN_LONG"
        reasons.append("obi/depth/flow favor long")
    elif score < -0.08 and confidence >= 0.28:
        direction = "short"
        action = "OPEN_SHORT"
        reasons.append("obi/depth/flow favor short")
    else:
        direction = "flat"
        action = "HOLD"
        reasons.append("microstructure edge is not strong enough")

    if blocks:
        action = "HOLD"

    if confidence >= 0.65:
        capital_fraction = 0.35
    elif confidence >= 0.45:
        capital_fraction = 0.25
    elif confidence >= 0.28:
        capital_fraction = 0.15
    else:
        capital_fraction = 0.0

    return {
        "version": "rule_micro_brain_v1",
        "direction": direction,
        "action": action,
        "score": score,
        "confidence": confidence,
        "capital_fraction": capital_fraction,
        "allowed": action != "HOLD" and not blocks,
        "blocks": blocks,
        "reasons": reasons,
        "inputs": {
            "obi": obi,
            "depth_imbalance": depth,
            "buy_pressure": buy_pressure,
            "spread_bps": spread_bps,
            "trades_1m": trades_1m,
            "volume_1m": volume_1m,
        },
    }
