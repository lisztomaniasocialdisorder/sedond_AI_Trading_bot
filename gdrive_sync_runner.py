"""
gdrive_sync_runner.py
=====================
Google Drive 模型同步背景服務入口（本機磁碟版，Drive 掛載為 G:）。

用法：
    python gdrive_sync_runner.py              # 持續輪詢（預設 30 分鐘一次）
    python gdrive_sync_runner.py --once       # 執行一次後退出
    python gdrive_sync_runner.py --once --skip-gate   # 跳過 promotion gate（測試）
    python gdrive_sync_runner.py --exp v12_multiscale_safe  # 指定實驗
    python gdrive_sync_runner.py --drive "G:\\我的雲端硬碟\\klinetraning"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=root / ".env")

    parser = argparse.ArgumentParser(description="Google Drive 本機模型同步服務（G: 磁碟機版）")
    parser.add_argument(
        "--drive",
        default=os.getenv("GDRIVE_ROOT", r"G:\我的雲端硬碟\klinetraning"),
        help="Google Drive 本機掛載路徑（預設 G:\\我的雲端硬碟\\klinetraning）",
    )
    parser.add_argument(
        "--exp",
        default=None,
        help="指定實驗名稱（如 v12_multiscale_safe），不指定則自動選最新",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="本機模型標籤（預設由 SYMBOL+INTERVAL 組合，如 BTCUSDT_1h）",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("GDRIVE_SYNC_INTERVAL_SEC", "1800")),
        help="掃描間隔秒數（預設 1800 = 30 分鐘）",
    )
    parser.add_argument("--once", action="store_true", help="只執行一次後退出")
    parser.add_argument("--skip-gate", action="store_true", help="跳過 promotion gate（測試用）")
    args = parser.parse_args()

    from src.gdrive_sync import sync_loop, sync_once, _is_drive_available

    drive_root = Path(args.drive)
    tag = args.tag or f"{os.getenv('SYMBOL', 'BTCUSDT')}_{os.getenv('INTERVAL', '1h')}"

    logger.info("=== Google Drive 本機模型同步服務 ===")
    logger.info("Drive 路徑：%s", drive_root)
    logger.info("監控標籤：%s", tag)
    logger.info("實驗：%s", args.exp or "自動選最新")

    if not _is_drive_available(drive_root):
        logger.error("Drive 路徑不存在：%s", drive_root)
        logger.error("請確認 Google Drive 已掛載，或用 --drive 指定正確路徑")
        sys.exit(1)

    if args.once:
        result = sync_once(
            root_dir=root,
            drive_root=drive_root,
            experiment_name=args.exp,
            skip_promotion_gate=args.skip_gate,
            tag=tag,
        )
        exp = result.get("experiment", "N/A")
        action = result.get("action", "none")
        promoted = result.get("promoted", False)
        error = result.get("error")

        if error:
            logger.error("[%s] 錯誤：%s", tag, error)
        elif promoted:
            logger.info("[%s] 模型已更新：%s", tag, exp)
        elif action == "already_synced":
            logger.info("[%s] 已是最新版本：%s", tag, exp)
        else:
            logger.info("[%s] 狀態：%s  exp=%s  reasons=%s",
                        tag, action, exp, result.get("promotion_reasons"))
    else:
        logger.info("掃描間隔：%ds，按 Ctrl+C 停止", args.interval)
        try:
            sync_loop(
                root_dir=root,
                drive_root=drive_root,
                interval_sec=args.interval,
                tag=tag,
                experiment_name=args.exp,
            )
        except KeyboardInterrupt:
            logger.info("已停止。")


if __name__ == "__main__":
    main()
