# Trading Folder Map

Last cleanup: 2026-05-29

## Keep

`dashboards/`
- Flask API and trading dashboard UI.
- Do not delete while dashboard is in use.

`harvesters/`
- Live BTC/ADA harvester, DB, parquet conversion, schedules, and maintenance scripts.
- `BTC_harvester/raw_db/` and `ADA_harvester/raw_db/` are live SQLite stores.
- Do not delete raw DB files unless parquet/rollup recovery is confirmed.

`data/parquet_rollup/`
- Main historical microstructure parquet store.
- Used by research, feature generation, dashboard row/size display, and OBI reports.
- Keep.

`data/parquet_signal/`
- Latest compact feature snapshots for dashboard and signal display.
- Keep.

`experiments/`
- Local model experiments and reports.
- Keep unless a specific version is clearly retired.

`models/`
- Local production/staged model artifacts.
- Keep.

`outputs/`
- Strategy params, OBI edge reports, generated reports.
- Keep recent files; archive manually if needed.

`research/`
- Training/evaluation/research scripts.
- Keep.

`src/`
- Core trading, modeling, backtest, pipeline code.
- Keep.

`tools/`
- Maintenance and notebook-generation scripts.
- Keep.

## Disposable

`__pycache__/`, `*.pyc`
- Python runtime cache.
- Safe to delete anytime.

`data/parquet/`
- Legacy partitioned parquet layout from older `db_to_parquet.py` runs.
- Current pipeline uses `data/parquet_rollup/`.
- Safe to delete after rollup exists.

`data/parquet_spool/<COIN>/<table>/date=<old-date>/`
- Live small-batch parquet spool.
- Safe to delete only after it has been compacted into rollup.
- Keep today's folders while harvester is running.

`build_tools/downloads/*.exe`
- Installer cache.
- Safe to delete after tooling is installed.

`logs/`
- Operational logs.
- Safe to prune old logs; keep latest logs when debugging.

## Current Pipeline

Harvester writes:
`raw_db + data/parquet_spool`

Daily 00:30 task runs:
`compact spool -> parquet_rollup -> parquet_signal -> OBI report -> v10 kline bundle -> train`

Trading dashboard currently uses:
`v10_dl` as the active AI model.
