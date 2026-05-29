# Trading Research And Execution Console

這個 repo 是 Brian 的 BTC/ADA 交易研究與交易面板。核心目標不是一直重構，而是把資料管線、特徵、模型、風控、交易面板接成可以持續驗證 alpha 的系統。

目前主線：

```text
harvester
-> parquet spool
-> daily/hourly parquet rollup
-> microstructure features
-> OBI edge validation / model training
-> v10 direction brain + microstructure timing gate
-> OKX demo/live guarded order panel
```

## Current Model Setup

- `v10`: 中長尺度方向腦，使用 1h kline / TradingView-style feature 預測大方向。
- `v13`: 小尺度 orderbook/OBI timing 研究腦，目前不直接取代 v10。
- `v14`: 下一步研究方向，目標是用 microstructure features 做 timing/filter。
- Trading panel 目前以 `v10` 為主，microstructure 資料先用來驗證 edge，再接成 timing gate。

## Folder Map

```text
C:\Users\brian\trading
├─ dashboards
│  ├─ server.py                    Flask API server, dashboard backend
│  ├─ okx_client.py                OKX v5 REST client
│  ├─ start_dashboard.bat          啟動 dashboard
│  └─ trading\trading.html         交易面板前端
├─ harvesters
│  ├─ binance_futures_harvester.py Binance futures market data harvester
│  ├─ start_harvesters.bat         啟動 BTC/ADA harvester
│  ├─ stop_harvesters.bat          停止 harvester
│  ├─ compact_parquet_spool.py     parquet spool -> daily/monthly rollup
│  ├─ parquet_to_features.py       rollup parquet -> training features
│  ├─ midnight_parquet_aggregate.bat 00:30 daily pipeline
│  └─ weekly_strategy_recalibrate.py 每週策略參數校準
├─ research
│  ├─ alpha_conditional_test.py    conditional expectation / OBI edge test
│  ├─ obi_edge_report.py           OBI edge report
│  └─ execution_aware_backtest.py  execution-aware backtest
├─ src
│  ├─ config.py                    shared config helpers
│  └─ rule_brain.py                rule-based signal helpers
├─ tools
│  ├─ build_v14_notebook.py        產生 v14 notebook
│  ├─ inspect_db_time.py           檢查 DB 時間覆蓋
│  └─ prune_db_before_feature_ts.py 刪除已轉 feature 前的 DB rows
├─ data                           本機資料輸出，不進 GitHub
├─ experiments                    模型/回測輸出，不進 GitHub
├─ outputs                        runtime strategy params，不進 GitHub
└─ logs                           runtime logs，不進 GitHub
```

## Data Pipeline

### 1. Harvester

抓 Binance futures microstructure 資料，寫入 SQLite 與 parquet spool。

```powershell
cd C:\Users\brian\trading
harvesters\start_harvesters.bat
```

停止：

```powershell
harvesters\stop_harvesters.bat
```

主要 raw DB：

```text
harvesters\BTC_harvester\raw_db\microstructure_BTC.db
harvesters\ADA_harvester\raw_db\microstructure_ADA.db
```

### 2. Parquet Rollup

把 harvester 產生的 parquet spool 聚合成 daily/monthly/quarterly/yearly rollup。

```powershell
python harvesters\compact_parquet_spool.py --symbols BTC,ADA --table all --include-open-day
```

輸出：

```text
data\parquet_rollup\<COIN>\daily\<table>_YYYY-MM-DD.parquet
```

### 3. Feature Engineering

把 daily rollup 做成 microstructure training features。

```powershell
python harvesters\parquet_to_features.py --symbols BTC,ADA --force
```

輸出：

```text
data\features\BTC\microstructure_features_all.parquet
data\features\ADA\microstructure_features_all.parquet
data\features\microstructure_features_summary.json
```

目前 feature 重點：

- L1 mid / spread / OBI
- OBI L20 rolling 5s / 15s / 60s mean/std/change
- depth imbalance
- signed trade flow
- future_return_5s/15s/30s/60s
- target_up_5s/15s/30s/60s

### 4. Nightly Pipeline

每天 00:30 執行聚合與 feature 更新。Dashboard banner 會讀：

```text
logs\parquet_rollup\midnight_aggregate_*.log
logs\locks\parquet_pipeline.lock
```

手動跑：

```powershell
harvesters\midnight_parquet_aggregate.bat
```

### 5. Weekly Strategy Parameters

每週重新校準策略參數，例如最大槓桿與 active confidence threshold。

```powershell
python harvesters\weekly_strategy_recalibrate.py
```

輸出：

```text
outputs\strategy_params.json
```

## Dashboard

啟動：

```powershell
cd C:\Users\brian\trading\dashboards
python server.py
```

交易面板：

```text
http://localhost:5000/trading.html
```

常用 API：

```text
/api/pipeline/nightly-status
/api/signal/BTCUSDT
/api/model/ai-decision/BTCUSDT?model_mode=v10
/api/strategy/params/BTCUSDT?model_mode=v10
/api/okx/status
```

## OKX Environment

`.env` 不進 GitHub。範例：

```env
OKX_API_KEY=your_api_key
OKX_API_SECRET=your_secret
OKX_API_PASSPHRASE=your_passphrase
OKX_DEMO=1
OKX_ENABLE_LIVE_TRADING=1
OKX_ALLOW_REAL_ENV_TRADING=0
OKX_BASE_URL=https://www.okx.com
```

安全邏輯：

- `OKX_DEMO=1`: 使用 OKX demo trading header。
- `OKX_ENABLE_LIVE_TRADING=0`: 所有下單都會被 dry-run 擋住。
- `OKX_ALLOW_REAL_ENV_TRADING=0`: 防止誤接真實環境。

## Git Hygiene

這些不進 GitHub：

- SQLite DB / WAL / SHM
- parquet spool / rollup / features
- models / experiments
- logs / outputs / runtime state
- API keys / `.env`

GitHub 只放程式、設定範例、README、研究腳本。資料與模型留本機或 Google Drive。

## Recommended Operating Rhythm

```text
hourly: update raw/parquet data
daily 00:30: rollup -> features -> reports
weekly: recalibrate strategy parameters
when enough data: train or evaluate v14 microstructure timing model
```

目前最重要的研究問題：

```text
OBI / depth imbalance 出現後，5s/15s/60s future midprice return 是否穩定偏移？
```

如果這個 edge 穩，才值得把 v14 接到交易面板當 timing gate。
