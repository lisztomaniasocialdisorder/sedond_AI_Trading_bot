# Trading Pipeline Overview

## 1) One-page flow

```text
Binance Streams
  └─> Harvester
      (binance_futures_harvester.py)
      ├─> SQLite raw DB
      │    ├─ BTC_harvester/raw_db/microstructure_BTC.db
      │    └─ ADA_harvester/raw_db/microstructure_ADA.db
      └─> Live Parquet Spool
           └─ data/parquet_spool/{BTC,ADA}/<table>/date=YYYY-MM-DD/part-*.parquet

Hourly job (every hour)
  hourly_db_to_parquet.bat
    └─ compact_parquet_spool.py
         └─> data/parquet_rollup/{BTC,ADA}/daily/*.parquet

Daily job (00:30)
  midnight_parquet_aggregate.bat
    ├─ compact_parquet_spool.py
    ├─ parquet_to_features.py
    │    └─> data/parquet_signal/{BTC,ADA}/signal_averages_1h.parquet
    │         (contains OBI10/OBI20/OBI30, depth imbalance, buy pressure...)
    └─ run.py
         └─ train + promotion gate
              ├─ models/...
              └─ outputs/report*.json

Weekly job (Sun 02:30)
  weekly_strategy_recalibrate.bat
    └─ weekly_strategy_recalibrate.py
         └─> outputs/strategy_params.json

Trading Dashboard
  dashboards/server.py + dashboards/trading/trading.html
    ├─ reads AI decision (v10/v13 hybrid)
    ├─ reads strategy params
    └─ applies precheck/gates before OKX demo order
```

## 2) Scheduler map

- Hourly data rollup:
  - Task: `Trading DB To Parquet Hourly`
  - Script: `harvesters/hourly_db_to_parquet.bat`

- Daily feature + train:
  - Task: `Trading Parquet Rollup`
  - Script: `harvesters/midnight_parquet_aggregate.bat`

- Weekly strategy recalibration:
  - Task: `Trading Weekly Strategy Recalibrate`
  - Script: `harvesters/weekly_strategy_recalibrate.bat`

## 3) Current design decisions

- Keep old DBs (do not delete immediately).
- Hourly: only `spool -> rollup` for stability.
- Daily: `features + train` to avoid intraday training noise.
- Feature output includes:
  - `obi_l10_avg_1h`
  - `obi_l20_avg_1h`
  - `obi_l30_avg_1h` (currently L20 proxy if only depth20 is available)

## 4) If something looks wrong in dashboard

1. Restart dashboard server from:
   - `C:\Users\brian\trading\dashboards`
   - `py -3.11 server.py`
2. Hard refresh browser:
   - `Ctrl + F5`
3. Check APIs:
   - `/api/model/ai-decision/BTCUSDT?interval_sec=3600&model_mode=hybrid_v10_v13`
   - `/api/strategy/params/BTCUSDT?interval_sec=3600&model_mode=hybrid_v10_v13`

