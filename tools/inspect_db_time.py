import sqlite3
from pathlib import Path

DBS = [
    Path("harvesters/BTC_harvester/raw_db/microstructure_BTC.db"),
    Path("harvesters/ADA_harvester/raw_db/microstructure_ADA.db"),
]

TS_COL_CANDIDATES = ["ts", "timestamp", "event_ts", "local_ts", "server_ts", "time"]

for db_path in DBS:
    print(f"\nDB: {db_path}")
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    tables = [
        r[0]
        for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    for table in tables:
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols]
        ts_col = next((c for c in TS_COL_CANDIDATES if c in col_names), None)
        row_count = cur.execute(f"SELECT COUNT(1) FROM {table}").fetchone()[0]
        if ts_col is None:
            print(f"  - {table}: rows={row_count}, ts_col=None")
            continue
        min_ts, max_ts = cur.execute(
            f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {table}"
        ).fetchone()
        print(
            f"  - {table}: rows={row_count}, ts_col={ts_col}, min_ts={min_ts}, max_ts={max_ts}"
        )
    con.close()
