#!/usr/bin/env python3
"""Small OKX v5 REST adapter used by the local trading cockpit.

Secrets must stay on the Flask side. The browser only calls local endpoints.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


OKX_BASE_URL = os.getenv("OKX_BASE_URL", "https://www.okx.com").rstrip("/")
OKX_ENABLE_LIVE_TRADING = os.getenv("OKX_ENABLE_LIVE_TRADING", "0") == "1"
OKX_ALLOW_REAL_ENV_TRADING = os.getenv("OKX_ALLOW_REAL_ENV_TRADING", "0") == "1"


class OkxConfig:
    def __init__(self):
        self.api_key = os.getenv("OKX_API_KEY", "")
        self.api_secret = os.getenv("OKX_API_SECRET", "")
        self.passphrase = os.getenv("OKX_API_PASSPHRASE", "")
        self.demo = os.getenv("OKX_DEMO", "1") != "0"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret and self.passphrase)

    def public_status(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "demo": self.demo,
            "live_enabled": OKX_ENABLE_LIVE_TRADING,
            "real_env_trading_allowed": OKX_ALLOW_REAL_ENV_TRADING,
            "base_url": OKX_BASE_URL,
        }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_body(payload: Any | None) -> str:
    if payload is None:
        return ""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _sign(secret: str, timestamp: str, method: str, request_path: str, body: str) -> str:
    prehash = f"{timestamp}{method.upper()}{request_path}{body}"
    digest = hmac.new(secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _request(method: str, path: str, *, params: dict[str, Any] | None = None,
             payload: Any | None = None, private: bool = False) -> dict[str, Any]:
    cfg = OkxConfig()
    query = f"?{urlencode(params)}" if params else ""
    request_path = f"{path}{query}"
    body = _json_body(payload)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    if private:
        if not cfg.configured:
            raise RuntimeError("OKX credentials are not configured")
        ts = _timestamp()
        headers.update({
            "OK-ACCESS-KEY": cfg.api_key,
            "OK-ACCESS-SIGN": _sign(cfg.api_secret, ts, method, request_path, body),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": cfg.passphrase,
        })
        if cfg.demo:
            headers["x-simulated-trading"] = "1"

    req = Request(
        f"{OKX_BASE_URL}{request_path}",
        data=body.encode("utf-8") if body else None,
        headers=headers,
        method=method.upper(),
    )
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OKX HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"OKX network error: {exc.reason}") from exc


def get_status() -> dict[str, Any]:
    return OkxConfig().public_status()


def get_ticker(inst_id: str) -> dict[str, Any]:
    return _request("GET", "/api/v5/market/ticker", params={"instId": inst_id})


def get_account() -> dict[str, Any]:
    balances = _request("GET", "/api/v5/account/balance", private=True)
    positions = _request("GET", "/api/v5/account/positions", private=True)
    return {"balances": balances, "positions": positions}


def get_open_orders(inst_id: str | None = None) -> dict[str, Any]:
    params = {}
    if inst_id:
        params["instId"] = inst_id
    return _request("GET", "/api/v5/trade/orders-pending", params=params, private=True)


def _can_send_trade(confirm_live: bool = True) -> tuple[bool, dict[str, Any], str | None]:
    status = get_status()
    if not (status["configured"] and status["live_enabled"] and confirm_live):
        return False, status, "Trading is not enabled or request is not confirmed."
    if not status["demo"] and not status["real_env_trading_allowed"]:
        return False, status, "Live order blocked because OKX_DEMO=0 and real-environment trading is not allowed."
    return True, status, None


def cancel_order(order: dict[str, Any], *, confirm_live: bool = False) -> dict[str, Any]:
    inst_id = str(order.get("instId") or "").strip().upper()
    ord_id = str(order.get("ordId") or "").strip()
    cl_ord_id = str(order.get("clOrdId") or "").strip()
    if not inst_id:
        raise ValueError("instId is required")
    if not (ord_id or cl_ord_id):
        raise ValueError("ordId or clOrdId is required")
    payload = {"instId": inst_id}
    if ord_id:
        payload["ordId"] = ord_id
    else:
        payload["clOrdId"] = cl_ord_id
    allowed, status, reason = _can_send_trade(confirm_live)
    if not allowed:
        return {"dry_run": True, "blocked": True, "okx_status": status, "payload": payload, "message": reason}
    result = _request("POST", "/api/v5/trade/cancel-order", payload=payload, private=True)
    return {"dry_run": False, "payload": payload, "result": result}


def close_position(position: dict[str, Any], *, confirm_live: bool = False) -> dict[str, Any]:
    inst_id = str(position.get("instId") or "").strip().upper()
    mgn_mode = str(position.get("mgnMode") or position.get("tdMode") or "cross").strip().lower()
    pos_side = str(position.get("posSide") or "").strip().lower()
    if not inst_id:
        raise ValueError("instId is required")
    payload = {"instId": inst_id, "mgnMode": mgn_mode}
    if pos_side:
        payload["posSide"] = pos_side
    if position.get("autoCxl") is not None:
        payload["autoCxl"] = bool(position.get("autoCxl"))
    allowed, status, reason = _can_send_trade(confirm_live)
    if not allowed:
        return {"dry_run": True, "blocked": True, "okx_status": status, "payload": payload, "message": reason}
    result = _request("POST", "/api/v5/trade/close-position", payload=payload, private=True)
    return {"dry_run": False, "payload": payload, "result": result}


def build_order_payload(order: dict[str, Any]) -> dict[str, str]:
    inst_id = str(order.get("instId") or "").strip().upper()
    side = str(order.get("side") or "").strip().lower()
    ord_type = str(order.get("ordType") or "").strip().lower()
    size = str(order.get("sz") or "").strip()
    td_mode = str(order.get("tdMode") or "cross").strip().lower()

    if not inst_id:
        raise ValueError("instId is required")
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    if ord_type not in {"market", "limit", "post_only", "ioc", "fok"}:
        raise ValueError("unsupported ordType")
    if not size:
        raise ValueError("sz is required")

    payload = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": side,
        "ordType": ord_type,
        "sz": size,
    }
    px = str(order.get("px") or "").strip()
    if ord_type != "market":
        if not px:
            raise ValueError("px is required for non-market orders")
        payload["px"] = px

    pos_side = str(order.get("posSide") or "").strip().lower()
    if pos_side:
        payload["posSide"] = pos_side

    reduce_only = order.get("reduceOnly")
    if reduce_only in {True, "true", "True", "1", 1}:
        payload["reduceOnly"] = "true"

    cl_ord_id = str(order.get("clOrdId") or "").strip()
    if cl_ord_id:
        payload["clOrdId"] = cl_ord_id[:32]

    return payload


def place_order(order: dict[str, Any], *, confirm_live: bool = False) -> dict[str, Any]:
    payload = build_order_payload(order)
    allowed, status, reason = _can_send_trade(confirm_live)
    if not allowed:
        return {
            "dry_run": True,
            "blocked": True,
            "okx_status": status,
            "payload": payload,
            "message": reason,
        }
    result = _request("POST", "/api/v5/trade/order", payload=payload, private=True)
    return {"dry_run": False, "payload": payload, "result": result}
