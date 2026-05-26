# Binance Futures Harvester v2

## 目錄

```
harvesters/
├── binance_futures_harvester.py   ← 核心 (v2 全量重寫)
├── btc_harvester.py               ← BTC 啟動入口
├── ada_harvester.py               ← ADA 啟動入口
├── requirements_harvester.txt
├── BTC_harvester/
│   ├── logs/                      ← 每日 rotating log
│   └── raw_db/microstructure_BTC.db
└── ADA_harvester/
    ├── logs/
    └── raw_db/microstructure_ADA.db
```

## 安裝

```bat
cd C:\Users\brian\OneDrive\桌面\trading\harvesters
pip install -r requirements_harvester.txt
```

## 執行

**開兩個 terminal，各跑一個：**

```bat
python btc_harvester.py
```

```bat
python ada_harvester.py
```

---

## 收集的 WebSocket Streams

| Stream | 說明 | 寫入 Table |
|--------|------|-----------|
| `@trade` | 個別成交 | `trades` |
| `@aggTrade` | 聚合成交 | `agg_trades` |
| `@bookTicker` | 最佳一檔 + spread/OBI | `orderbook_l1` |
| `@depth5@100ms` | 五檔掛單快照 | `orderbook_l5` |
| `@depth20@100ms` | 二十檔掛單快照 | `orderbook_l20` |
| `@markPrice@1s` | 標記價/指數價/資金費率 | `mark_price` |
| `!forceOrder@arr` | 強平訂單 (全市場，按 symbol 過濾) | `liquidations` |

> 另外每次 depth update 都會計算 `orderbook_metrics`（OBI、VWAP、weighted mid、depth imbalance 等）

---

## SQLite Tables

### trades
| 欄位 | 型別 | 說明 |
|------|------|------|
| local_ts | REAL | 本機收到時間 (Unix float) |
| event_ts | INTEGER | Binance 事件時間 (ms) |
| trade_ts | INTEGER | 成交時間 (ms) |
| price / qty / quote_qty | REAL | 價格、數量、計價金額 |
| is_buyer_maker | INTEGER | 1=賣方主動 0=買方主動 |

### agg_trades
| 欄位 | 說明 |
|------|------|
| agg_trade_id | 聚合成交 ID |
| first_trade_id / last_trade_id | 涵蓋的個別成交 ID 範圍 |

### orderbook_l1
| 欄位 | 說明 |
|------|------|
| spread / spread_bps | 點差 / 基點點差 |
| mid_price | 中間價 |
| obi | Order Book Imbalance = (bid_qty − ask_qty) / (bid_qty + ask_qty) |

### orderbook_l5 / orderbook_l20
每次更新寫 5 or 20 rows（每 level 一 row）

### orderbook_metrics
| 欄位 | 說明 |
|------|------|
| depth_type | 'l5' or 'l20' |
| total_bid_qty / total_ask_qty | 總掛單量 |
| total_bid_value / total_ask_value | 總掛單價值 |
| depth_imbalance | 深度不平衡度 |
| bid_vwap / ask_vwap | 量加權平均買/賣價 |
| weighted_mid | 流動性加權中間價 |

### mark_price
| 欄位 | 說明 |
|------|------|
| mark_price | 標記價格 |
| index_price | 指數價格 |
| est_settle_price | 預估結算價 |
| last_funding_rate | 最新資金費率 |
| next_funding_time | 下次結算時間 (ms) |

### liquidations
強平訂單欄位：side, order_type, orig_qty, price, avg_price, order_status, etc.

### harvester_events
記錄 WS open/close/reconnect/depth gap 等系統事件

---

## Parquet 輸出路徑

```
data/parquet/BTC/
├── trades/date=2025-01-01/part-*.parquet
├── agg_trades/...
├── orderbook_l1/...
├── orderbook_l5/...
├── orderbook_l20/...
├── orderbook_metrics/...
├── mark_price/...
└── liquidations/...
```

讀取方式：
```python
import pandas as pd
df = pd.read_parquet("data/parquet/BTC/trades/")
```

---

## 穩定性功能

| 功能 | 說明 |
|------|------|
| 自動重連 | 指數退避 1s → 60s |
| Pong Watchdog | 30 秒無 pong 強制重連 |
| Gap Detection | depth stream 序列號不連續時記錄警告 |
| Daily Log | 每天切割 log，保留 30 天 |
| WAL Mode | SQLite WAL + mmap，寫入更快更安全 |

---

## Prometheus Metrics

| Metric | 說明 |
|--------|------|
| `hv_trades_total` | 成交筆數 |
| `hv_agg_trades_total` | 聚合成交筆數 |
| `hv_l1_updates_total` | L1 更新次數 |
| `hv_l5_updates_total` | L5 更新次數 |
| `hv_l20_updates_total` | L20 更新次數 |
| `hv_liquidations_total` | 強平事件數 |
| `hv_reconnects_total` | 重連次數 |
| `hv_gaps_total` | Gap 偵測數 |
| `hv_last_price` | 最新成交價 |
| `hv_last_spread` | 最新點差 |
| `hv_last_obi` | 最新 L1 OBI |
| `hv_last_mark_price` | 最新標記價 |
| `hv_last_funding_rate` | 最新資金費率 |
| `hv_last_depth_imbalance` | 最新深度不平衡 |

- BTC metrics: http://localhost:9100/metrics
- ADA metrics: http://localhost:9101/metrics
