# Trading Microstructure Console

這個專案目前分成三條線：

1. **Harvester 資料蒐集**：持續抓 BTC / ADA 的 Binance futures microstructure 資料。
2. **Dashboard / Trading Cockpit**：看資料、看訊號、接 OKX 模擬盤、記錄交易與資產。
3. **AI Brain**：還沒開始訓練，等 harvester 跑出足夠資料後再接上。

目前 AI 和 harvester 本體先不要亂動；交易介面、OKX、紀錄、風控、報表這些可以持續擴充。

---

## 資料夾結構

```text
C:\Users\brian\trading
├─ .env                         # OKX API key 與本機設定，不要公開
├─ .env.example                 # env 範本
├─ README.md                    # 本文件
│
├─ configs                      # 之後放策略 / 風控 / 模型設定
│
├─ dashboards
│  ├─ server.py                 # Flask API server，port 5000
│  ├─ okx_client.py             # OKX v5 REST API 串接
│  ├─ index.html                # 原始 microstructure 觀測 dashboard
│  ├─ split.html                # BTC / ADA signal split 頁
│  ├─ start_dashboard.bat       # dashboard 啟動腳本
│  ├─ control
│  │  └─ signal_dashboard.html  # 單一幣種 signal dashboard
│  └─ trading
│     └─ trading.html           # OKX 模擬盤交易介面
│
├─ harvesters
│  ├─ binance_futures_harvester.py  # BTC / ADA 共用 harvester 核心
│  ├─ btc_harvester.py              # BTC 啟動器
│  ├─ ada_harvester.py              # ADA 啟動器
│  ├─ start_harvesters.bat          # 同時啟動 BTC / ADA harvester
│  ├─ db_to_parquet.py              # SQLite DB 轉 Parquet
│  ├─ csv_to_parquet.py             # CSV 轉 Parquet
│  ├─ BTC_harvester
│  │  ├─ raw_db
│  │  │  └─ microstructure_BTC.db
│  │  └─ logs
│  └─ ADA_harvester
│     ├─ raw_db
│     │  └─ microstructure_ADA.db
│     └─ logs
│
├─ data
│  ├─ cache_fng.json            # 真實歷史 Fear & Greed 快取
│  ├─ trading_state.json        # 交易介面狀態、交易紀錄、每日資產日結
│  ├─ events
│  ├─ features
│  └─ parquet
│
├─ research                     # 未來 AI brain / 回測 / walk-forward 研究用
│
├─ reports                      # 未來報表輸出
│
└─ src                          # 未來正式共用模組
```

---

## 快速啟動

桌面目前有兩個 bat：

```text
C:\Users\brian\OneDrive\桌面\open_dashboard.bat
C:\Users\brian\OneDrive\桌面\open_trading.bat
```

用途：

```text
open_dashboard.bat  →  http://localhost:5000/signal_dashboard.html
open_trading.bat    →  http://localhost:5000/trading.html
```

它們會先檢查 `localhost:5000` 是否已經在跑；如果沒有，會自動啟動：

```text
C:\Users\brian\trading\dashboards\server.py
```

---

## 頁面說明

### 1. 原始觀測 Dashboard

```text
http://localhost:5000/
```

用途：

- 看 BTC / ADA 的 order book
- 看 DB 寫入筆數
- 看 L1 / L20 / trades / depth metrics
- 適合 debug harvester

### 2. Signal Dashboard

```text
http://localhost:5000/signal_dashboard.html
```

用途：

- 看單一幣種 signal 狀態
- 可切換 BTC / ADA
- 看 OBI、spread、depth、buy pressure
- AI Brain slot 目前是 standby

也支援固定幣種：

```text
http://localhost:5000/signal_dashboard.html?symbol=BTCUSDT
http://localhost:5000/signal_dashboard.html?symbol=ADAUSDT
```

### 3. BTC / ADA Split

```text
http://localhost:5000/split.html
```

用途：

- 左邊 BTCUSDT
- 右邊 ADAUSDT
- 兩邊各自刷新
- 適合平常當主控台看

### 4. Trading Panel

```text
http://localhost:5000/trading.html
```

用途：

- OKX 模擬盤真送單
- 顯示帳戶總資產與 USDT 可用資金
- 設定 AI 可用資金上限
- Kline / MACD / RSI / ATR
- 歷史 Fear & Greed
- 總資產每日結算折線圖
- 交易紀錄
- OKX 持倉
- 未成交委託
- 撤單 / 平倉

---

## OKX 模擬盤設定

設定檔：

```text
C:\Users\brian\trading\.env
```

必要欄位：

```env
OKX_API_KEY=你的 API Key
OKX_API_SECRET=你的 Secret Key
OKX_API_PASSPHRASE=你的 Passphrase
OKX_DEMO=1
OKX_ENABLE_LIVE_TRADING=1
OKX_ALLOW_REAL_ENV_TRADING=0
OKX_BASE_URL=https://www.okx.com
```

說明：

- `OKX_DEMO=1`：使用 OKX 模擬盤 header。
- `OKX_ENABLE_LIVE_TRADING=1`：允許真的送到 OKX 模擬盤。
- `OKX_ALLOW_REAL_ENV_TRADING=0`：正式環境保險絲，正常保持 0。

目前交易介面已確認可讀到模擬盤資產，並顯示「模擬盤真送單」。

---

## DB / CSV 轉 Parquet

Harvester 正在跑時會邊寫 SQLite DB，也會邊寫 Parquet。  
如果要補轉舊資料、重建 Parquet，使用 `db_to_parquet.py`。

### DB to Parquet

```bat
cd C:\Users\brian\trading
python harvesters\db_to_parquet.py --symbol BTC
python harvesters\db_to_parquet.py --symbol ADA
```

只轉單一 table：

```bat
python harvesters\db_to_parquet.py --symbol BTC --table trades
python harvesters\db_to_parquet.py --symbol ADA --table orderbook_l1
```

先看 row count，不寫檔：

```bat
python harvesters\db_to_parquet.py --symbol BTC --table trades --dry-run
```

輸出位置：

```text
data\parquet\BTC\<table>\date=YYYY-MM-DD\part-db-*.parquet
data\parquet\ADA\<table>\date=YYYY-MM-DD\part-db-*.parquet
```

### CSV to Parquet

如果手上已經有 CSV 才使用：

```bat
python harvesters\csv_to_parquet.py --input exports\trades.csv --symbol BTC --table trades
python harvesters\csv_to_parquet.py --input exports\ADA --symbol ADA --table orderbook_l1
```

輸出位置：

```text
data\parquet\<COIN>\<table>\date=YYYY-MM-DD\part-csv-*.parquet
```

---

## 已完成

- 專案搬到 `C:\Users\brian\trading`
- BTC / ADA harvester 可獨立啟動
- Dashboard server 整合 Flask API
- 原始 microstructure dashboard
- signal dashboard
- BTC / ADA split dashboard
- trading panel
- OKX 模擬盤 API 串接
- OKX 帳戶餘額顯示
- AI 可用資金上限欄位
- 模擬盤送單
- 交易紀錄
- 每日總資產日結折線圖
- trading state server 端保存
- 持倉顯示
- 未成交委託顯示
- 撤單 / 平倉 API 與 UI
- 歷史 Fear & Greed 真實日資料
- Kline / MACD / RSI / ATR 技術圖
- 技術圖同步縮放，Fear & Greed 獨立縮放
- DB to Parquet 工具
- CSV to Parquet 工具

---

## 尚未完成

- AI Brain 訓練
- AI Brain 接上 trading panel
- 正式 feature engineering pipeline
- target labeling
- walk-forward 驗證
- 回測報表頁
- 單筆風控上限
- 每日虧損上限
- 連續虧損暫停
- 黑天鵝暫停交易規則
- 更完整的 OKX 歷史成交同步

---

## 目前系統定位

目前系統已經有：

```text
眼睛：harvester + dashboards
手：OKX 模擬盤送單
記憶：交易紀錄 + 每日資產日結
駕駛艙：signal dashboard / split dashboard / trading panel
```

但真正的 AI Brain 還沒有訓練出來。

等 harvester 跑出足夠長的資料後，下一階段才是：

```text
資料整理
→ 特徵工程
→ 標籤設計
→ 模型訓練
→ walk-forward
→ 模擬盤驗證
→ 接上 trading panel
```
