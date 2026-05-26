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
import sqlite3
import time
from pathlib import Path
from threading import Lock
from urllib.request import Request, urlopen

from flask import Flask, jsonify, request, send_from_directory

import okx_client

# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_TTL    = 0.4   # seconds (400 ms)  — shared read cache
CHART_LIMIT  = 400   # trade points for chart initialisation
FNG_CACHE_PATH = PROJECT_ROOT / "data" / "cache_fng.json"
FNG_CACHE_TTL = 12 * 60 * 60
TRADING_STATE_PATH = PROJECT_ROOT / "data" / "trading_state.json"

app    = Flask(__name__, static_folder=".")
_cache = {}
_lock  = Lock()


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
    state["equitySnapshots"] = list(state.get("equitySnapshots") or [])[-730:]
    state["tradeRecords"] = list(state.get("tradeRecords") or [])[:500]
    state["updated_at"] = time.time()
    TRADING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRADING_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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

        row_counts = {}
        for tbl in ("trades", "agg_trades", "orderbook_l1", "orderbook_l5", "orderbook_l20", "orderbook_metrics"):
            row_counts[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]

        latest_metric = metrics[0] if metrics else {}
        buy_pressure = None
        if stats:
            total_volume = (stats.get("buy_volume_1m") or 0) + (stats.get("sell_volume_1m") or 0)
            if total_volume > 0:
                buy_pressure = (stats.get("buy_volume_1m") or 0) / total_volume

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
            "stats": stats,
            "counts": row_counts,
            "total_rows": sum(v for v in row_counts.values() if isinstance(v, int)),
            "db_size_mb": round(_db_path(symbol).stat().st_size / 1_048_576, 2),
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


def _build_indicator_chart(symbol: str) -> dict:
    symbol = symbol.upper()
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


@app.route("/api/trading/indicators/<symbol>")
def route_trading_indicators(symbol):
    symbol = symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"
    return jsonify(_build_indicator_chart(symbol))


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
        return jsonify(okx_client.place_order(order, confirm_live=confirm_live))
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
