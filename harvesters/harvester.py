import asyncio
import websockets
import json
import aiosqlite
import logging
import signal
import sys
import os
from pathlib import Path

# --- 1. 路徑與日誌設定 ---
BASE_DIR = str(Path(__file__).resolve().parents[2] / "micro_signal_trading")
os.makedirs(BASE_DIR, exist_ok=True)

DB_NAME = os.path.join(BASE_DIR, "microstructure.db")
LOG_FILE = os.path.join(BASE_DIR, "harvester.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

WS_URL = "wss://ws.bitget.com/v2/ws/public"

# 就是這三行被你無情刪除的全域變數，現在我把它們放回來了
trade_queue = asyncio.Queue()
orderbook_queue = asyncio.Queue()
shutdown_event = asyncio.Event()

# --- 2. 資料庫初始化 ---
async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('PRAGMA journal_mode=WAL;')
        await db.execute('PRAGMA synchronous=NORMAL;')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                price REAL,
                size REAL,
                side TEXT
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS orderbooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                bids_vol REAL,
                asks_vol REAL,
                obi REAL
            )
        ''')
        await db.commit()
    logger.info("💽 本地資料庫初始化完成，WAL 模式已啟動。")

# --- 3. 磁碟寫入消費者 ---
async def db_writer():
    async with aiosqlite.connect(DB_NAME) as db:
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(1)
                
                trades_batch = []
                ob_batch = []
                
                while not trade_queue.empty():
                    trades_batch.append(trade_queue.get_nowait())
                while not orderbook_queue.empty():
                    ob_batch.append(orderbook_queue.get_nowait())
                    
                if trades_batch:
                    await db.executemany(
                        "INSERT INTO trades (timestamp, price, size, side) VALUES (?, ?, ?, ?)", 
                        trades_batch
                    )
                if ob_batch:
                    await db.executemany(
                        "INSERT INTO orderbooks (timestamp, bids_vol, asks_vol, obi) VALUES (?, ?, ?, ?)", 
                        ob_batch
                    )
                    
                if trades_batch or ob_batch:
                    await db.commit()
                    logger.info(f"💾 批次寫入: {len(trades_batch)} 筆成交, {len(ob_batch)} 筆盤口深度")
                    
            except Exception as e:
                logger.error(f"❌ 寫入資料庫失敗: {e}")

# --- 4. 網路接收生產者 ---
async def ws_receiver():
    payload = {
        "op": "subscribe",
        "args": [
            {"instType": "USDT-FUTURES", "channel": "trade", "instId": "BTCUSDT"},
            {"instType": "USDT-FUTURES", "channel": "books15", "instId": "BTCUSDT"}
        ]
    }
    retry_delay = 1

    while not shutdown_event.is_set():
        try:
            async for websocket in websockets.connect(WS_URL, ping_interval=20, ping_timeout=10):
                logger.info("🌐 WebSocket 連線成功！已發送訂閱請求...")
                await websocket.send(json.dumps(payload))
                retry_delay = 1
                
                async for message in websocket:
                    if shutdown_event.is_set():
                        return
                        
                    data = json.loads(message)
                    if data == "pong":
                        continue
                        
                    if "action" in data and "data" in data:
                        channel = data["arg"]["channel"]
                        
                        if channel == "trade":
                            for t in data["data"]:
                                trade_queue.put_nowait((int(t["ts"]), float(t["price"]), float(t["size"]), t["side"]))
                                
                        elif channel == "books15":
                            for ob in data["data"]:
                                bids_vol = sum([float(b[1]) for b in ob["bids"][:5]])
                                asks_vol = sum([float(a[1]) for a in ob["asks"][:5]])
                                obi = (bids_vol - asks_vol) / (bids_vol + asks_vol + 1e-8)
                                orderbook_queue.put_nowait((int(ob["ts"]), bids_vol, asks_vol, obi))

        except websockets.ConnectionClosed as e:
            logger.warning(f"⚠️ 交易所斷開連線 ({e})，準備重連...")
        except Exception as e:
            logger.error(f"❌ 發生網路異常: {e}")
            
        if not shutdown_event.is_set():
            logger.info(f"⏳ 等待 {retry_delay} 秒後嘗試重新連線...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

# --- 5. 優雅關閉處理 ---
def signal_handler(sig, frame):
    logger.info("\n🛑 接收到中斷訊號 (SIGINT/SIGTERM)，正在清空佇列並安全關閉系統...")
    shutdown_event.set()

async def main():
    await init_db()
    
    # 支援 Windows 環境的事件迴圈中斷處理
    if sys.platform == 'win32':
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: shutdown_event.set())
        loop.add_signal_handler(signal.SIGTERM, lambda: shutdown_event.set())
    
    writer_task = asyncio.create_task(db_writer())
    receiver_task = asyncio.create_task(ws_receiver())
    
    await shutdown_event.wait()
    
    logger.info("系統關閉中，等待最後的 I/O 寫入...")
    await asyncio.sleep(3) 
    writer_task.cancel()
    receiver_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 捕捉 Windows 下強制 Ctrl+C 產生未被 signal 攔截的例外
        pass
