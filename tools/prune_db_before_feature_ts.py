import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TARGETS = [
    {
        "symbol": "BTC",
        "db": Path("harvesters/BTC_harvester/raw_db/microstructure_BTC.db"),
        "cutoff_ms": 1779993464220,
    },
    {
        "symbol": "ADA",
        "db": Path("harvesters/ADA_harvester/raw_db/microstructure_ADA.db"),
        "cutoff_ms": 1779993466828,
    },
]

EVENT_TS_TABLES = [
    "trades",
    "agg_trades",
    "orderbook_l1",
    "orderbook_l5",
    "orderbook_l20",
    "orderbook_metrics",
    "mark_price",
    "liquidations",
]

for t in TARGETS:
    db_path = t["db"]
    cutoff_ms = int(t["cutoff_ms"])
    cutoff_s = cutoff_ms / 1000.0
    print(f"\n=== {t['symbol']} ===")
    print(f"db: {db_path}")
    print(
        "cutoff:",
        datetime.fromtimestamp(cutoff_s, tz=timezone.utc).isoformat(),
        f"({cutoff_ms} ms)",
    )
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    total_deleted = 0
    for table in EVENT_TS_TABLES:
        exists = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            continue
        to_del = cur.execute(
            f"SELECT COUNT(1) FROM {table} WHERE event_ts IS NOT NULL AND event_ts <= ?",
            (cutoff_ms,),
        ).fetchone()[0]
        if to_del > 0:
            cur.execute(f"DELETE FROM {table} WHERE event_ts IS NOT NULL AND event_ts <= ?", (cutoff_ms,))
        total_deleted += int(to_del)
        print(f"  {table}: deleted={to_del}")

    # harvester_events uses seconds
    exists = cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='harvester_events'"
    ).fetchone()
    if exists:
        to_del = cur.execute(
            "SELECT COUNT(1) FROM harvester_events WHERE local_ts IS NOT NULL AND local_ts <= ?",
            (cutoff_s,),
        ).fetchone()[0]
        if to_del > 0:
            cur.execute(
                "DELETE FROM harvester_events WHERE local_ts IS NOT NULL AND local_ts <= ?",
                (cutoff_s,),
            )
        total_deleted += int(to_del)
        print(f"  harvester_events: deleted={to_del}")

    con.commit()
    print(f"total_deleted={total_deleted}")
    print("running VACUUM...")
    cur.execute("VACUUM")
    con.close()
    print("done")
