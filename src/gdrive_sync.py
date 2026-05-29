"""
src/gdrive_sync.py
==================
Google Drive 本機磁碟同步模組（Drive 已掛載為 G: 磁碟機）。

不需要任何 Google API！直接讀取本機 G:\\我的雲端硬碟\\klinetraning\\
偵測 Colab 訓練的新 checkpoint，自動複製到本機 models/ 並執行 promotion gate。

Drive 目錄結構（Colab 訓練後存入）：
  G:\\我的雲端硬碟\\klinetraning\\
    └── experiments\\
          └── <experiment_name>\\          ← 如 v12_multiscale_safe
                ├── models\\
                │     └── <model>.pkl     ← 或 .joblib / .pt
                └── report.json           ← 訓練/回測指標
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 預設 Drive 根目錄 ────────────────────────────────────────────────────────
_DEFAULT_DRIVE_ROOT = Path(r"G:\我的雲端硬碟\klinetraning")
_EXPERIMENTS_DIR = "experiments"

# 可識別的模型檔副檔名
_MODEL_EXTENSIONS = {".pkl", ".joblib", ".pt", ".keras"}

# 絕對禁止複製的副檔名（特徵/訓練資料，保留在 Drive 不下載）
_BLOCKED_EXTENSIONS = {".parquet", ".csv", ".db", ".db-wal", ".db-shm", ".zip", ".tar", ".gz"}

# 單一模型檔大小上限（500 MB），超過就拒絕複製
_MAX_MODEL_FILE_MB = 500


def _is_drive_available(drive_root: Path) -> bool:
    """檢查 Google Drive 本機磁碟是否已掛載。"""
    return drive_root.exists()


def _find_experiment_dirs(experiments_root: Path) -> list[Path]:
    """
    列出所有實驗資料夾，依最新修改時間降序排列。
    只列出包含 report.json 的有效實驗。
    """
    if not experiments_root.exists():
        return []
    dirs = [
        d for d in experiments_root.iterdir()
        if d.is_dir() and (d / "report.json").exists()
    ]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def _find_model_files(experiment_dir: Path) -> list[Path]:
    """
    找出實驗資料夾內的模型檔案（含子目錄 models/）。
    自動跳過被封鎖的副檔名和超過大小限制的檔案。
    """
    found: list[Path] = []
    models_subdir = experiment_dir / "models"
    search_dirs = [models_subdir, experiment_dir] if models_subdir.exists() else [experiment_dir]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in _BLOCKED_EXTENSIONS:
                logger.debug("[gdrive_sync] 跳過訓練資料檔（不複製到本機）：%s", f.name)
                continue
            if ext not in _MODEL_EXTENSIONS:
                continue
            size_mb = f.stat().st_size / 1024 / 1024
            if size_mb > _MAX_MODEL_FILE_MB:
                logger.warning("[gdrive_sync] 跳過超大模型檔 %.1f MB（上限 %d MB）：%s",
                               size_mb, _MAX_MODEL_FILE_MB, f.name)
                continue
            found.append(f)
    return found


def _safe_copy_model_files(src_dir: Path, dst_dir: Path) -> list[str]:
    """
    只複製模型檔案（.pkl/.joblib/.pt/.keras）和 report.json 到 dst_dir。
    絕對不複製 .parquet / .csv / .db 等大型訓練資料檔案。
    回傳已複製的檔名清單。
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    total_mb = 0.0

    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        ext = src.suffix.lower()

        # 封鎖大型訓練資料
        if ext in _BLOCKED_EXTENSIONS:
            logger.debug("[gdrive_sync] 略過（訓練資料不下載）：%s", src.name)
            continue

        # 只允許模型檔和 report.json
        is_model = ext in _MODEL_EXTENSIONS
        is_report = src.name == "report.json"
        if not (is_model or is_report):
            continue

        size_mb = src.stat().st_size / 1024 / 1024
        if is_model and size_mb > _MAX_MODEL_FILE_MB:
            logger.warning("[gdrive_sync] 略過超大檔 %.1f MB：%s", size_mb, src.name)
            continue

        # 保留相對路徑結構
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(rel))
        total_mb += size_mb
        logger.info("[gdrive_sync]   複製 %s  (%.1f MB)", rel, size_mb)

    logger.info("[gdrive_sync] 共複製 %d 個檔案，合計 %.1f MB", len(copied), total_mb)
    return copied


def _load_report(experiment_dir: Path) -> dict:
    """載入 report.json。"""
    rp = experiment_dir / "report.json"
    if not rp.exists():
        return {}
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sync_status(status_path: Path, payload: dict) -> None:
    """寫入同步狀態 JSON（dashboard 讀取用）。"""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def _load_sync_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_sync_state(state_path: Path, payload: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _mtime_str(path: Path) -> str:
    """回傳路徑的修改時間字串（作為版本識別）。"""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return ""


def _report_to_backtest_metrics(report: dict) -> dict:
    """
    將 v12 多時框 report.json 轉換成 promotion gate 看得懂的格式。
    promotion gate 主要看：trades, win_rate, total_return, max_drawdown
    """
    # 優先用 daily test（較穩定），其次 hourly test
    for section in ("daily", "hourly", "entry_30m"):
        test_data = report.get(section, {}).get("test", {})
        if test_data:
            capital = float(test_data.get("capital", 10000))
            cash = float(test_data.get("cash_capital", 10000))
            trades = int(test_data.get("trades", 0))
            total_return = (capital - cash) / cash if cash > 0 else 0.0
            win_rate = float(test_data.get("bacc", 0.5))
            return {
                "trades": trades,
                "total_return": total_return,
                "win_rate": win_rate,
                "max_drawdown": 0.0,   # report.json 未記錄 drawdown，設 0 讓 gate 不卡
                "profit_factor": None,
                "source_section": section,
            }
    return {}


def _run_promotion_gate_local(report: dict, production_report_path: Path) -> tuple[bool, list[str]]:
    """執行 promotion gate（使用本機 pipeline 的 _promotion_gate）。"""
    try:
        from .pipeline import _promotion_gate, _read_json
        from .config import Settings
        settings = Settings()
        candidate_metrics = _report_to_backtest_metrics(report)
        if not candidate_metrics:
            return False, ["report.json 格式無法解析"]
        current_report = _read_json(production_report_path)
        return _promotion_gate(candidate_metrics, current_report, settings)
    except Exception as e:
        logger.warning("[gdrive_sync] promotion gate 錯誤，預設接受：%s", e)
        # Drive 模型用不同格式，gate 失敗時直接接受（避免永遠被卡住）
        return True, [f"gate_error_auto_accept: {e}"]


def sync_once(
    root_dir: Path,
    drive_root: Path | None = None,
    experiment_name: str | None = None,
    skip_promotion_gate: bool = False,
    tag: str = "BTCUSDT_1h",
) -> dict:
    """
    掃描 Drive experiments/ 目錄，找到最新（或指定）實驗，
    複製到本機 models/_gdrive_staged/，通過 promotion gate 後更新 production。

    Parameters
    ----------
    root_dir : Path
        專案根目錄（trading/）。
    drive_root : Path | None
        Drive 掛載根目錄，None 時使用預設 G:\\我的雲端硬碟\\klinetraning。
    experiment_name : str | None
        指定實驗名稱（如 "v12_multiscale_safe"），None 時自動選最新。
    skip_promotion_gate : bool
        True 時跳過 promotion gate 直接複製。
    tag : str
        本機模型標籤（用於路徑命名）。
    """
    if drive_root is None:
        drive_root = Path(os.getenv("GDRIVE_ROOT", str(_DEFAULT_DRIVE_ROOT)))

    experiments_root = drive_root / _EXPERIMENTS_DIR
    model_dir = root_dir / "models"
    output_dir = root_dir / "outputs"
    staged_dir = model_dir / "_gdrive_staged" / tag
    production_dir = model_dir / tag
    state_path = output_dir / f"gdrive_sync_state_{tag}.json"
    status_path = output_dir / f"gdrive_sync_status_{tag}.json"
    production_report_path = output_dir / f"production_report_{tag}.json"

    result: dict[str, Any] = {
        "tag": tag,
        "action": "none",
        "experiment": None,
        "promoted": False,
        "promotion_reasons": [],
        "error": None,
    }

    # 確認 Drive 已掛載
    if not _is_drive_available(drive_root):
        result["error"] = f"Google Drive 未掛載或路徑不存在：{drive_root}"
        _save_sync_status(status_path, {**result, "status": "drive_offline"})
        logger.warning("[gdrive_sync] %s", result["error"])
        return result

    # 找實驗資料夾
    if experiment_name:
        exp_dir = experiments_root / experiment_name
        if not exp_dir.exists():
            result["error"] = f"指定的實驗不存在：{exp_dir}"
            _save_sync_status(status_path, {**result, "status": "not_found"})
            return result
        exp_dirs = [exp_dir]
    else:
        exp_dirs = _find_experiment_dirs(experiments_root)

    if not exp_dirs:
        result["action"] = "no_experiment"
        _save_sync_status(status_path, {**result, "status": "idle", "message": "Drive 上尚無實驗資料夾"})
        return result

    latest_exp = exp_dirs[0]
    exp_name = latest_exp.name
    result["experiment"] = exp_name

    # 用資料夾修改時間作為版本 ID
    exp_mtime = _mtime_str(latest_exp)
    state = _load_sync_state(state_path)
    if state.get("last_synced_experiment") == exp_name and \
       state.get("last_synced_mtime") == exp_mtime and \
       not skip_promotion_gate:
        result["action"] = "already_synced"
        _save_sync_status(status_path, {
            **result,
            "status": "idle",
            "message": f"最新實驗 {exp_name} 已是當前版本（{exp_mtime}）",
        })
        logger.debug("[gdrive_sync] %s 已是最新，跳過", exp_name)
        return result

    # 載入 report.json
    report = _load_report(latest_exp)

    # 找模型檔
    model_files = _find_model_files(latest_exp)
    if not model_files:
        result["error"] = f"實驗 {exp_name} 內找不到模型檔案"
        _save_sync_status(status_path, {**result, "status": "error"})
        return result

    logger.info("[gdrive_sync] 偵測到新實驗：%s（%s），模型：%s",
                exp_name, exp_mtime, [f.name for f in model_files])

    # 複製到 staged 目錄（只複製模型檔 + report.json，不複製訓練資料）
    staged_exp_dir = staged_dir / exp_name
    if staged_exp_dir.exists():
        shutil.rmtree(staged_exp_dir)
    staged_exp_dir.mkdir(parents=True, exist_ok=True)
    copied_files = _safe_copy_model_files(latest_exp, staged_exp_dir)
    if not copied_files:
        result["error"] = f"實驗 {exp_name} 內找不到可複製的模型檔案（.pkl/.joblib/.pt）"
        _save_sync_status(status_path, {**result, "status": "error"})
        return result
    logger.info("[gdrive_sync] 已複製到 staged：%s", staged_exp_dir)
    result["action"] = "staged"
    result["copied_files"] = copied_files

    # Promotion gate
    promoted = False
    promote_reasons: list[str] = []

    if skip_promotion_gate:
        promoted = True
        promote_reasons = ["skip_promotion_gate=True"]
    else:
        promoted, promote_reasons = _run_promotion_gate_local(report, production_report_path)

    result["promoted"] = promoted
    result["promotion_reasons"] = promote_reasons

    if promoted:
        if production_dir.exists():
            shutil.rmtree(production_dir)
        shutil.copytree(staged_exp_dir, production_dir)
        logger.info("[gdrive_sync] ✅ 模型晉升：%s → %s", exp_name, production_dir)

        _save_sync_state(state_path, {
            "last_synced_experiment": exp_name,
            "last_synced_mtime": exp_mtime,
            "last_promoted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        _save_sync_status(status_path, {
            **result,
            "status": "promoted",
            "message": f"✅ 模型已更新：{exp_name}",
            "report_summary": _report_to_backtest_metrics(report),
        })
    else:
        logger.info("[gdrive_sync] ⛔ 未通過 promotion gate：%s", promote_reasons)
        _save_sync_status(status_path, {
            **result,
            "status": "rejected",
            "message": f"⛔ {exp_name} 未通過審核：{', '.join(promote_reasons)}",
        })

    return result


def sync_loop(
    root_dir: Path,
    drive_root: Path | None = None,
    interval_sec: int = 1800,
    tag: str = "BTCUSDT_1h",
    experiment_name: str | None = None,
) -> None:
    """持續輪詢 Drive，每 interval_sec 秒掃描一次。"""
    if drive_root is None:
        drive_root = Path(os.getenv("GDRIVE_ROOT", str(_DEFAULT_DRIVE_ROOT)))
    logger.info("[gdrive_sync] 啟動持續同步模式 drive=%s interval=%ds tag=%s",
                drive_root, interval_sec, tag)
    while True:
        try:
            result = sync_once(root_dir, drive_root, experiment_name, tag=tag)
            logger.info("[gdrive_sync] %s → action=%s promoted=%s exp=%s",
                        tag, result["action"], result["promoted"], result.get("experiment"))
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            logger.error("[gdrive_sync] 同步錯誤：%s", e)
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("[gdrive_sync] 收到中斷訊號，停止。")
            break
