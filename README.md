# Sedond AI Trading Bot

這個專案目前分成兩個核心區塊：

1. **既有 AI 交易框架**：保留原本 repo 的 `src/`、`dashboard.py`、`run.py`、OKX paper trading、回測、walk-forward、SNR、特徵工程與模型訓練工具。
2. **新增 Microstructure + Trading Console**：新增 BTC / ADA harvester、SQLite/Parquet 資料管線、Flask dashboard、BTC/ADA split signal dashboard，以及 OKX 模擬盤交易介面。

目前最重要的未完成項目是：**AI 大腦還沒完成接上**。現在已有資料蒐集、觀測、交易面板、OKX 模擬盤送單與資料轉檔工具，下一步會是把長期收集到的 microstructure 資料訓練成真正的交易決策模型。

## 快速入口

```text
C:\Users\brian\trading
```

常用頁面：

```text
http://localhost:5000/                       原始 microstructure dashboard
http://localhost:5000/signal_dashboard.html   Signal dashboard
http://localhost:5000/split.html              BTC / ADA split dashboard
http://localhost:5000/trading.html            OKX 模擬盤交易介面
```

桌面捷徑：

```text
C:\Users\brian\OneDrive\桌面\open_dashboard.bat
C:\Users\brian\OneDrive\桌面\open_trading.bat
```

## 資料夾結構

```text
C:\Users\brian\trading
├─ .env                         # OKX API key 與本機設定，不會進 git
├─ README.md
├─ .gitignore
├─ dashboard.py                  # 原 repo 的 Streamlit AI dashboard
├─ run.py                        # 原 repo 的訓練 / 更新流程入口
├─ okx_paper_trade.py            # 原 repo 的 OKX paper trading 入口
├─ src                           # AI、回測、特徵、SNR、OKX、pipeline 核心模組
├─ dashboards
│  ├─ server.py                  # Flask API server，port 5000
│  ├─ okx_client.py              # OKX v5 REST client
│  ├─ index.html                 # 原始 microstructure dashboard
│  ├─ split.html                 # BTC / ADA split dashboard
│  ├─ control
│  │  └─ signal_dashboard.html   # Signal dashboard
│  └─ trading
│     └─ trading.html            # 中文 OKX 模擬盤交易介面
├─ harvesters
│  ├─ binance_futures_harvester.py
│  ├─ btc_harvester.py
│  ├─ ada_harvester.py
│  ├─ start_harvesters.bat
│  ├─ db_to_parquet.py           # SQLite DB 轉 Parquet
│  └─ csv_to_parquet.py          # CSV 轉 Parquet
├─ data
│  ├─ parquet                    # 轉出的 Parquet，不進 git
│  ├─ cache_fng.json             # Fear & Greed cache，不進 git
│  └─ trading_state.json         # 交易介面狀態，不進 git
├─ configs
├─ research
└─ reports
```

## OKX 模擬盤設定

請把 OKX 模擬盤 key 放在：

```text
C:\Users\brian\trading\.env
```

需要的欄位：

```env
OKX_API_KEY=your_api_key
OKX_API_SECRET=your_secret
OKX_API_PASSPHRASE=your_passphrase
OKX_DEMO=1
OKX_ENABLE_LIVE_TRADING=1
OKX_ALLOW_REAL_ENV_TRADING=0
OKX_BASE_URL=https://www.okx.com
```

說明：

- `OKX_DEMO=1`：使用 OKX 模擬盤 header。
- `OKX_ENABLE_LIVE_TRADING=1`：允許交易介面真的送單到模擬盤。
- `OKX_ALLOW_REAL_ENV_TRADING=0`：避免誤打到真實盤。

`.env` 已被 `.gitignore` 排除，不會推到 GitHub。

## Dashboard Server

啟動 dashboard server：

```powershell
cd C:\Users\brian\trading\dashboards
python server.py
```

如果 port 5000 被佔用，可先關掉舊程序再啟動。

## Harvester

BTC / ADA harvester 會把 Binance futures microstructure 資料寫進 SQLite DB。

常用入口：

```text
C:\Users\brian\trading\harvesters\start_harvesters.bat
```

DB 位置：

```text
C:\Users\brian\trading\harvesters\BTC_harvester\raw_db\microstructure_BTC.db
C:\Users\brian\trading\harvesters\ADA_harvester\raw_db\microstructure_ADA.db
```

DB、WAL、log 都已被 `.gitignore` 排除。

## DB / CSV 轉 Parquet

Harvester 目前主要產出是 SQLite DB，所以通常使用 `db_to_parquet.py`。

DB to Parquet：

```powershell
cd C:\Users\brian\trading
python harvesters\db_to_parquet.py --symbol BTC
python harvesters\db_to_parquet.py --symbol ADA
```

只轉單一 table：

```powershell
python harvesters\db_to_parquet.py --symbol BTC --table trades
python harvesters\db_to_parquet.py --symbol ADA --table orderbook_l1
```

只檢查筆數、不輸出：

```powershell
python harvesters\db_to_parquet.py --symbol BTC --table trades --dry-run
```

CSV to Parquet：

```powershell
python harvesters\csv_to_parquet.py --input exports\trades.csv --symbol BTC --table trades
```

輸出位置：

```text
data\parquet\<COIN>\<table>\date=YYYY-MM-DD\part-*.parquet
```

## 已完成

- BTC / ADA harvester 可重開。
- ADA harvester 支援 Ctrl+C 後先 flush queue 再停機。
- 原始 microstructure dashboard。
- Signal dashboard。
- BTC / ADA split dashboard。
- 中文 OKX 模擬盤交易介面。
- OKX 模擬盤餘額、下單、撤單、平倉 API。
- Kline / MACD / RSI / ATR 圖表。
- Fear & Greed 歷史資料。
- 交易紀錄與每日總資產折線圖。
- 交易介面狀態保存，下次打開會沿用。
- SQLite DB to Parquet。
- CSV to Parquet。

## 待完成

- AI 大腦。
- 使用長期 harvester 資料建立 feature pipeline。
- target labeling。
- 模型訓練與 walk-forward 驗證。
- 把 AI 大腦輸出的 signal 接回 trading panel。
- 更完整的風控與自動交易策略。

## 目前狀態

```text
資料收集：完成基礎版
Dashboard：完成基礎版
OKX 模擬盤交易：完成基礎版
Parquet 資料管線：完成基礎版
AI 大腦：尚未完成
```
