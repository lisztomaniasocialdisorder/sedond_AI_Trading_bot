from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .config import Settings


def _asymmetric_lev_weights(
    lev_train: np.ndarray,
    penalty_factor: float = 3.0,
) -> np.ndarray:
    """
    非對稱槓桿樣本權重：對高估槓桿施以更高懲罰。
    對高槓桿預測錯誤的代價遠高於低槓桿，因此使用二次方放大權重。
    penalty_factor：高槓桿樣本相對低槓桿的最大懲罰倍數，建議 2.0~5.0。
    """
    lev_min, lev_max = lev_train.min(), lev_train.max()
    if lev_max <= lev_min:
        return np.ones(len(lev_train), dtype=np.float64)
    # 正規化至 [0, 1]，使用二次曲線讓高槓桿樣本獲得更大權重
    norm = ((lev_train - lev_min) / (lev_max - lev_min)) ** 2
    weights = 1.0 + (penalty_factor - 1.0) * norm
    return (weights / weights.mean()).clip(0.3, penalty_factor * 1.5)


def _pinball_sample_weights(
    lev_train: np.ndarray,
    lev_pred_warm: np.ndarray,
    tau: float = 0.35,
) -> np.ndarray:
    """
    Pinball / 分位數樣本權重。
    Pinball loss 定義：
      L(y, y_hat) = (y - y_hat) * tau       若 y >= y_hat（低估）
      L(y, y_hat) = (y_hat - y) * (1 - tau) 若 y < y_hat （高估，即預測過高）

    tau=0.35：對高估槓桿的懲罰倍數為 (1-0.35)/0.35 ≈ 1.86 倍。
    """
    residual = lev_train - lev_pred_warm
    weights = np.where(
        residual >= 0,
        tau,
        1 - tau,
    )
    return (weights / weights.mean()).clip(0.2, 4.0)


@dataclass
class TrainedModels:
    clf: Any
    lev_reg: Any
    feature_cols: list[str]
    backend: str = "sklearn_rf"
    backend_meta: dict | None = None


class EnsembleLevReg:
    """
    Top-level ensemble regressor so joblib/pickle can serialize it.
    Combines a warm-up RF and a quantile GB regressor.
    """

    def __init__(self, rf: Any, gb: Any, rf_w: float = 0.40) -> None:
        self.rf = rf
        self.gb = gb
        self.rf_w = rf_w

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        rf_p = self.rf.predict(X)
        gb_p = self.gb.predict(X)
        blended = self.rf_w * rf_p + (1.0 - self.rf_w) * gb_p
        return np.clip(blended * 0.95, 1.0, 1e6)


class _TorchSignalWrapper:
    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray, class_values: np.ndarray, torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.classes_ = class_values.astype(int)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            logits = self.model(xt)
            p = self._torch.softmax(logits, dim=1).detach().cpu().numpy()
        return p

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class _TorchLevWrapper:
    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray, torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            out = self.model(xt).squeeze(1)
            p = out.detach().cpu().numpy()
        return p



class _StudentNetClfWrapper:
    """predict_proba / predict wrapper for the dual-head StudentNet."""

    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray,
                 class_values: np.ndarray, torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self.classes_ = class_values.astype(int)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            logits, _ = self.model(xt)
            p = self._torch.softmax(logits, dim=1).detach().cpu().numpy()
        return p

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


class _StudentNetLevWrapper:
    """predict wrapper for the StudentNet leverage head."""

    def __init__(self, model: Any, device: Any, mean: np.ndarray, scale: np.ndarray,
                 torch_mod: Any) -> None:
        self.model = model
        self.device = device
        self.mean = mean.astype(np.float32)
        self.scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        self._torch = torch_mod

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        arr = x.to_numpy(dtype=np.float32, copy=False)
        return (arr - self.mean) / self.scale

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with self._torch.no_grad():
            xt = self._torch.from_numpy(self._transform(x)).to(self.device)
            _, lev = self.model(xt)
            return lev.detach().cpu().numpy()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {
        "timestamp",
        "date",
        "future_ret",
        "label",
        "target_leverage",
        "open_time",
        "close_time",
        "equity_curve_proxy",
        "rolling_peak",
    }
    cols = [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _clean_xy(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.ffill().bfill()
    x = x.fillna(0)
    return x


def _build_cls_mlp(torch_mod: Any, in_dim: int, out_dim: int) -> Any:
    """Legacy builder kept for loading old torch_accel checkpoints."""
    return torch_mod.nn.Sequential(
        torch_mod.nn.Linear(in_dim, 256),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Dropout(0.05),
        torch_mod.nn.Linear(256, 128),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(128, out_dim),
    )


def _build_reg_mlp(torch_mod: Any, in_dim: int) -> Any:
    """Legacy builder kept for loading old torch_accel checkpoints."""
    return torch_mod.nn.Sequential(
        torch_mod.nn.Linear(in_dim, 256),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Dropout(0.05),
        torch_mod.nn.Linear(256, 128),
        torch_mod.nn.ReLU(),
        torch_mod.nn.Linear(128, 1),
    )


def _make_student_net(torch_mod: Any, in_dim: int) -> Any:
    """
    Build a StudentNet instance using the provided torch module.
    Using a factory function (instead of a top-level class) keeps torch
    as a lazy import while still allowing both training and loading to
    reconstruct the same architecture deterministically given in_dim.

    Architecture: shared GELU backbone → cls_branch (3 classes) + lev_branch (scalar).
    Returns logits and lev_pred together in one forward pass.
    """
    nn = torch_mod.nn

    class StudentNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Shared backbone: 瘦身設計，追求推論速度
            self.shared = nn.Sequential(
                nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Linear(128, 64),     nn.LayerNorm(64),  nn.GELU(),
            )
            # 左腦：方向預測 (對齊 Teacher 軟標籤)
            self.cls_branch = nn.Linear(64, 3)
            # 右腦：槓桿倍數預測
            self.lev_branch = nn.Sequential(
                nn.Linear(64, 32), nn.GELU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: Any) -> tuple:
            h = self.shared(x)
            logits   = self.cls_branch(h)
            lev_pred = self.lev_branch(h).squeeze(1)
            return logits, lev_pred

    return StudentNet()


def _make_distillation_loss(torch_mod: Any, lev_weight: float = 0.5) -> Any:
    """
    Build a DistillationLoss module.

    forward(student_logits, student_lev, teacher_probs, teacher_lev):
      - teacher_probs: (B, 3) raw probabilities  [short, flat, long] order
      - student_logits: (B, 3) raw logits        same order
      - KLDivLoss(batchmean) — student must be log_softmax, teacher must be raw probs
      - SmoothL1Loss for leverage
    """
    nn  = torch_mod.nn
    F   = torch_mod.nn.functional

    class DistillationLoss(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # batchmean is the standard for KL Divergence in deep learning
            self.kl_loss  = nn.KLDivLoss(reduction='batchmean')
            self.lev_loss = nn.SmoothL1Loss()
            self.lev_w    = lev_weight

        def forward(self, student_logits: Any, student_lev: Any,
                    teacher_probs: Any, teacher_lev: Any) -> Any:
            # ⚠️ 天坑：student 必須是 log_softmax，teacher 必須是 raw probabilities
            log_p_student = F.log_softmax(student_logits, dim=1)
            l_dir = self.kl_loss(log_p_student, teacher_probs)
            l_lev = self.lev_loss(student_lev, teacher_lev)
            return l_dir + self.lev_w * l_lev

    return DistillationLoss()


def _resolve_torch_device(requested: str) -> tuple[Any, str, Any]:
    try:
        import torch  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"PyTorch not available: {e}") from e

    req = str(requested or "auto").lower()

    if req in {"directml", "npu"}:
        try:
            import torch_directml  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"DirectML backend not available: {e}") from e
        return torch_directml.device(), "directml", torch

    if req in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda", torch
        raise RuntimeError("CUDA backend not available on this machine.")

    if req == "cpu":
        return torch.device("cpu"), "cpu", torch

    # auto / cloud: prefer cloud-style accelerators first.
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", torch
    try:
        import torch_directml  # type: ignore
        return torch_directml.device(), "directml", torch
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps", torch
    return torch.device("cpu"), "cpu", torch


def _fit_torch_accelerated(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    requested_device: str,
    progress_cb: Callable[[int, str], None] | None = None,
    soft_labels_df: pd.DataFrame | None = None,
    distill_alpha: float = 0.4,
) -> Tuple[TrainedModels, dict]:
    device, resolved_device_name, torch = _resolve_torch_device(requested_device)

    x_train      = _clean_xy(train_df, feature_cols).to_numpy(dtype=np.float32, copy=False)
    x_test       = _clean_xy(test_df,  feature_cols).to_numpy(dtype=np.float32, copy=False)
    y_train_hard = train_df["label"].astype(int).to_numpy()
    y_test       = test_df["label"].astype(int).to_numpy()
    lev_test     = test_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)

    scaler    = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_s  = scaler.transform(x_test).astype(np.float32, copy=False)
    mean      = scaler.mean_.astype(np.float32)
    scale     = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)

    # class_values sorted: [-1, 0, 1]  =>  index 0=short, 1=flat, 2=long
    class_values = np.array(sorted(set(y_train_hard.tolist()) | set(y_test.tolist())), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(class_values)}
    y_train_idx  = np.array([class_to_idx[v] for v in y_train_hard], dtype=np.int64)

    in_dim     = x_train_s.shape[1]
    n_train    = len(x_train_s)
    batch_size = max(64, int(settings.torch_batch_size))
    epochs     = max(3, int(settings.torch_epochs))

    # ── Align teacher soft labels via vectorized timestamp merge ───────────────
    distill_applied   = False
    soft_proba_tensor = None   # (N, 3) float32, order: [short, flat, long]
    soft_lev_tensor   = None   # (N,) float32

    if soft_labels_df is not None and not soft_labels_df.empty and distill_alpha > 0:
        _sl        = soft_labels_df.copy()
        _prob_cols = ["soft_p_long", "soft_p_flat", "soft_p_short"]
        if (all(c in _sl.columns for c in _prob_cols)
                and "timestamp" in _sl.columns
                and "timestamp" in train_df.columns):

            _sl["_ts"]  = pd.to_datetime(_sl["timestamp"], utc=True, errors="coerce")
            _tr         = train_df[["timestamp"]].copy()
            _tr["_ts"]  = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
            _tr["_row"] = np.arange(len(_tr))
            _extra      = ["teacher_leverage"] if "teacher_leverage" in _sl.columns else []
            merged = _tr[["_ts", "_row"]].merge(
                _sl[["_ts"] + _prob_cols + _extra], on="_ts", how="left"
            )
            match_mask = merged["soft_p_long"].notna()
            n_match    = int(match_mask.sum())

            if n_match > 0:
                rows = merged["_row"][match_mask].values

                # Build (N, 3) soft prob matrix — column order matches class_values: [short, flat, long]
                sp        = np.full((n_train, 3), 1.0 / 3, dtype=np.float32)
                sp[rows, 0] = merged["soft_p_short"][match_mask].values.astype(np.float32)  # index 0 = -1 = short
                sp[rows, 1] = merged["soft_p_flat"][match_mask].values.astype(np.float32)   # index 1 =  0 = flat
                sp[rows, 2] = merged["soft_p_long"][match_mask].values.astype(np.float32)   # index 2 =  1 = long
                soft_proba_tensor = torch.from_numpy(sp).to(device)

                if "teacher_leverage" in merged.columns:
                    lev_hard = train_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)
                    soft_lev = merged["teacher_leverage"].fillna(pd.Series(lev_hard)).to_numpy(dtype=np.float32)
                    blended  = np.clip(
                        (1.0 - distill_alpha) * lev_hard + distill_alpha * soft_lev,
                        1.0, float(settings.max_leverage),
                    ).astype(np.float32)
                    soft_lev_tensor = torch.from_numpy(blended).to(device)

                distill_applied = True
                agree_pct = float((np.argmax(sp, axis=1) == y_train_idx).mean() * 100)
                if progress_cb:
                    progress_cb(63, f"Torch 蒸餾對齊：{n_match:,}/{n_train:,} 筆，Teacher/Student 一致率 {agree_pct:.1f}%")

    # ── Build StudentNet + optimiser ───────────────────────────────────────────
    student   = _make_student_net(torch, in_dim).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)

    # Hard-label fallback targets
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    hard_lev   = train_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)
    hard_lev_t = torch.from_numpy(hard_lev)

    # Distillation criterion (KLDiv + SmoothL1)
    distill_criterion = _make_distillation_loss(torch, lev_weight=0.5).to(device) if distill_applied else None

    # ── Training loop ──────────────────────────────────────────────────────────
    for ep in range(epochs):
        if progress_cb:
            tag = "，Teacher蒸餾" if distill_applied else ""
            progress_cb(
                66 + int(ep / epochs * 14),
                f"訓練 StudentNet（Epoch {ep+1}/{epochs}{tag}）",
            )
        student.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            xb  = torch.from_numpy(x_train_s[idx]).to(device)
            optimizer.zero_grad()
            student_logits, student_lev_pred = student(xb)

            if distill_applied and distill_criterion is not None:
                # yb_teacher_probs: (B, 3)  [soft_p_short, soft_p_flat, soft_p_long]
                # yb_teacher_lev:   (B,)
                t_probs = soft_proba_tensor[idx]  # type: ignore[index]
                t_lev   = (
                    soft_lev_tensor[idx]          # type: ignore[index]
                    if soft_lev_tensor is not None
                    else hard_lev_t[idx].to(device)
                )
                loss = distill_criterion(student_logits, student_lev_pred, t_probs, t_lev)
            else:
                # Pure hard-label fallback when no soft labels available
                yb  = torch.from_numpy(y_train_idx[idx]).to(device)
                ybl = hard_lev_t[idx].to(device)
                loss = (
                    ce_loss_fn(student_logits, yb)
                    + 0.5 * torch.nn.functional.smooth_l1_loss(student_lev_pred, ybl)
                )

            loss.backward()
            optimizer.step()

    # ── Evaluation ─────────────────────────────────────────────────────────────
    student.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x_test_s).to(device)
        test_logits, test_lev_out = student(xt)
        proba    = torch.softmax(test_logits, dim=1).detach().cpu().numpy()
        lev_pred = test_lev_out.detach().cpu().numpy()

    y_pred     = class_values[np.argmax(proba, axis=1)]
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    lev_mae    = mean_absolute_error(lev_test, lev_pred)

    models = TrainedModels(
        clf     = _StudentNetClfWrapper(student, device, mean, scale, class_values, torch),
        lev_reg = _StudentNetLevWrapper(student, device, mean, scale, torch),
        feature_cols = feature_cols,
        backend      = "student_net",
        backend_meta = {
            "device":               str(device),
            "in_dim":               int(in_dim),
            "class_values":         class_values.tolist(),
            "mean":                 mean.tolist(),
            "scale":                scale.tolist(),
            "distillation_applied": distill_applied,
            "distill_alpha":        distill_alpha if distill_applied else 0.0,
        },
    )
    metrics = {
        "classification_report": cls_report,
        "leverage_mae":           float(lev_mae),
        "train_rows":             int(len(train_df)),
        "test_rows":              int(len(test_df)),
        "training_backend":       "student_net",
        "training_device":        str(device),
        "training_device_kind":   resolved_device_name,
        "distillation_applied":   distill_applied,
        "distill_alpha":          distill_alpha if distill_applied else 0.0,
    }
    return models, metrics



    # ── 準備蒸餾軟標籤（Teacher KL-Divergence loss）───────────────────────────
    # Align soft labels to training rows via timestamp merge (vectorized).
    distill_applied = False
    soft_proba_train: np.ndarray | None = None   # (N, 3) float32, class order: long/flat/short
    soft_lev_train: np.ndarray | None = None     # (N,) float32

    if soft_labels_df is not None and not soft_labels_df.empty and distill_alpha > 0:
        _sl = soft_labels_df.copy()
        _label_cols = ["soft_p_long", "soft_p_flat", "soft_p_short"]
        if all(c in _sl.columns for c in _label_cols):
            # Vectorized timestamp merge
            if "timestamp" in _sl.columns and "timestamp" in train_df.columns:
                _sl["_ts"] = pd.to_datetime(_sl["timestamp"], utc=True, errors="coerce")
                _tr = train_df[["timestamp"]].copy()
                _tr["_ts"] = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
                _tr["_row"] = np.arange(len(_tr))
                merged = _tr[["_ts", "_row"]].merge(
                    _sl[["_ts"] + _label_cols + (["teacher_leverage"] if "teacher_leverage" in _sl.columns else [])],
                    on="_ts", how="left",
                )
                match_mask = merged[_label_cols[0]].notna()
                if match_mask.sum() > 0:
                    n = len(train_df)
                    soft_proba_train = np.full((n, 3), 1.0/3, dtype=np.float32)
                    soft_proba_train[merged["_row"][match_mask].values, 0] = merged["soft_p_long"][match_mask].values.astype(np.float32)
                    soft_proba_train[merged["_row"][match_mask].values, 1] = merged["soft_p_flat"][match_mask].values.astype(np.float32)
                    soft_proba_train[merged["_row"][match_mask].values, 2] = merged["soft_p_short"][match_mask].values.astype(np.float32)
                    if "teacher_leverage" in merged.columns:
                        lev_hard = train_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)
                        soft_lev = merged["teacher_leverage"].fillna(pd.Series(lev_hard)).to_numpy(dtype=np.float32)
                        soft_lev_train = (1 - distill_alpha) * lev_hard + distill_alpha * np.clip(soft_lev, 1.0, settings.max_leverage)
                    distill_applied = True
                    agree_pct = float((np.argmax(soft_proba_train, axis=1) == y_train_idx).mean() * 100)
                    if progress_cb:
                        progress_cb(63, f"Torch 蒸餾對齊：{match_mask.sum():,}/{n:,} 筆，Teacher/Student 一致率 {agree_pct:.1f}%")

    # ── 分類模型訓練（CrossEntropy + 可選 KL-Div 蒸餾）──────────────────────
    cls_model = _build_cls_mlp(torch, in_dim, n_classes).to(device)
    cls_opt = torch.optim.AdamW(cls_model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    n_train = len(x_train_s)
    # Convert soft labels to tensor if available
    soft_proba_tensor: Any = None
    if distill_applied and soft_proba_train is not None:
        soft_proba_tensor = torch.from_numpy(soft_proba_train).to(device)  # (N, 3)

    for ep in range(epochs):
        if progress_cb:
            progress_cb(66 + int(ep / epochs * 7), f"訓練分類模型（Epoch {ep+1}/{epochs}{'，Teacher蒸餾' if distill_applied else ''}）")
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            xb = torch.from_numpy(x_train_s[idx]).to(device)
            yb = torch.from_numpy(y_train_idx[idx]).to(device)
            cls_opt.zero_grad()
            logits = cls_model(xb)
            # Hard label CE loss
            loss = ce_loss_fn(logits, yb)
            # Soft label KL-Div distillation
            if soft_proba_tensor is not None:
                sb = soft_proba_tensor[idx]  # (B, 3) teacher soft probs
                log_pred = torch.log_softmax(logits, dim=1)
                kl = kl_loss_fn(log_pred, sb)
                loss = (1.0 - distill_alpha) * loss + distill_alpha * kl
            loss.backward()
            cls_opt.step()

    with torch.no_grad():
        cls_model.eval()
        xt = torch.from_numpy(x_test_s).to(device)
        proba = torch.softmax(cls_model(xt), dim=1).detach().cpu().numpy()
    y_pred = class_values[np.argmax(proba, axis=1)]
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # ── 槓桿模型訓練（MSE，可選混入 teacher_leverage 軟目標）────────────────
    lev_train_raw = train_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)
    lev_train = soft_lev_train if soft_lev_train is not None else lev_train_raw
    lev_test = test_df["target_leverage"].clip(1, settings.max_leverage).to_numpy(dtype=np.float32)

    reg_model = _build_reg_mlp(torch, in_dim).to(device)
    reg_opt = torch.optim.AdamW(reg_model.parameters(), lr=1e-3, weight_decay=1e-4)
    reg_loss_fn = torch.nn.MSELoss()

    lev_train_t = torch.from_numpy(lev_train)
    for ep in range(max(3, epochs // 2)):
        if progress_cb:
            progress_cb(73 + int(ep / max(3, epochs // 2) * 7), f"訓練槓桿模型（Epoch {ep+1}/{max(3, epochs // 2)}）")
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i: i + batch_size]
            xb = torch.from_numpy(x_train_s[idx]).to(device)
            yb = lev_train_t[idx].to(device)
            reg_opt.zero_grad()
            pred = reg_model(xb).squeeze(1)
            loss = reg_loss_fn(pred, yb)
            loss.backward()
            reg_opt.step()

    with torch.no_grad():
        reg_model.eval()
        xt = torch.from_numpy(x_test_s).to(device)
        lev_pred = reg_model(xt).squeeze(1).detach().cpu().numpy()
    lev_mae = mean_absolute_error(lev_test, lev_pred)

    models = TrainedModels(
        clf=_TorchSignalWrapper(cls_model, device, mean, scale, class_values, torch),
        lev_reg=_TorchLevWrapper(reg_model, device, mean, scale, torch),
        feature_cols=feature_cols,
        backend="torch_accel",
        backend_meta={
            "device": str(device),
            "in_dim": int(in_dim),
            "class_values": class_values.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "distillation_applied": distill_applied,
            "distill_alpha": distill_alpha if distill_applied else 0.0,
        },
    )
    metrics = {
        "classification_report": cls_report,
        "leverage_mae": float(lev_mae),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_backend": "torch_accel",
        "training_device": str(device),
        "training_device_kind": resolved_device_name,
        "distillation_applied": distill_applied,
        "distill_alpha": distill_alpha if distill_applied else 0.0,
    }
    return models, metrics


def _fit_sklearn_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    progress_cb: Callable[[int, str], None] | None = None,
    soft_labels_df: pd.DataFrame | None = None,
    distill_alpha: float = 0.4,
) -> Tuple[TrainedModels, dict]:
    """
    distill_alpha: Teacher 頠?蝐斗毽?交?靘€?.0=蝝′璅惜, 1.0=蝝?璅惜
    撖阡??剖??孵?: y_blend = (1-alpha)*hard + alpha*soft
    """
    x_train = _clean_xy(train_df, feature_cols)
    y_train_hard = train_df["label"].to_numpy(dtype=int)
    x_test = _clean_xy(test_df, feature_cols)
    y_test = test_df["label"]

    distill_applied = False
    sample_weight: np.ndarray | None = None

    if soft_labels_df is not None and not soft_labels_df.empty and distill_alpha > 0:
        _sl = soft_labels_df.copy()
        _label_cols = ["soft_p_long", "soft_p_flat", "soft_p_short"]
        _has_ts = "timestamp" in _sl.columns and "timestamp" in train_df.columns

        # Vectorized timestamp merge (replaces O(n) Python loop)
        soft_p_long  = np.full(len(train_df), 1/3, dtype=np.float64)
        soft_p_flat  = np.full(len(train_df), 1/3, dtype=np.float64)
        soft_p_short = np.full(len(train_df), 1/3, dtype=np.float64)

        if _has_ts and all(c in _sl.columns for c in _label_cols):
            _sl["_ts"] = pd.to_datetime(_sl["timestamp"], utc=True, errors="coerce")
            _tr = train_df[["timestamp"]].copy()
            _tr["_ts"] = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
            _tr["_row"] = np.arange(len(_tr))
            _merged = _tr[["_ts", "_row"]].merge(
                _sl[["_ts"] + _label_cols], on="_ts", how="left"
            )
            _mask = _merged["soft_p_long"].notna()
            soft_p_long[_merged["_row"][_mask].values]  = _merged["soft_p_long"][_mask].values
            soft_p_flat[_merged["_row"][_mask].values]  = _merged["soft_p_flat"][_mask].values
            soft_p_short[_merged["_row"][_mask].values] = _merged["soft_p_short"][_mask].values
        elif all(c in _sl.columns for c in _label_cols):
            n_match = min(len(train_df), len(_sl))
            soft_p_long[:n_match]  = _sl["soft_p_long"].values[:n_match]
            soft_p_flat[:n_match]  = _sl["soft_p_flat"].values[:n_match]
            soft_p_short[:n_match] = _sl["soft_p_short"].values[:n_match]

        # Teacher soft probabilities -> class index.
        soft_class_arr = np.array([soft_p_long, soft_p_flat, soft_p_short], dtype=np.float64)  # (3, N)
        soft_pred_idx = np.argmax(soft_class_arr, axis=0)  # 0=long, 1=flat, 2=short
        class_map = {0: 1, 1: 0, 2: -1}  # 頧? label 撠望頠?蝐斤?銝餃撐
        y_soft = np.array([class_map[i] for i in soft_pred_idx], dtype=int)
        teacher_conf = np.max(soft_class_arr, axis=0)  # 靽∪?摨?
        # sample_weight: 蝖祆?蝐文?頠?蝐支?????甈? 1.0嚗?銝??????alpha*teacher_conf 瘙箏?
        agree = (y_train_hard == y_soft)
        sample_weight = np.where(
            agree,
            1.0,
            distill_alpha * teacher_conf + (1.0 - distill_alpha) * (1.0 - teacher_conf)
        )
        sample_weight = (sample_weight / sample_weight.mean()).clip(0.2, 3.0)  # 璅???
        # y_train ?寧頠?蝐文
        y_train_mixed = np.where(agree, y_train_hard, y_soft)
        y_train = pd.Series(y_train_mixed)
        distill_applied = True
        if progress_cb:
            agree_pct = agree.mean() * 100
            progress_cb(63, f"蒸餾對齊完成：Teacher/Student 一致率 {agree_pct:.1f}%")
    else:
        y_train = pd.Series(y_train_hard)

    if progress_cb:
        progress_cb(68, "訓練分類模型（RandomForest" + (" + Teacher 蒸餾" if distill_applied else "") + "）")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train, sample_weight=sample_weight)

    y_pred = clf.predict(x_test)
    cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    lev_train = train_df["target_leverage"].clip(1, settings.max_leverage)
    lev_test  = test_df["target_leverage"].clip(1, settings.max_leverage)

    # ?? ?賊冗瑽▼瘛瑕? ?????????????????????????????????????????
    if distill_applied and soft_labels_df is not None and "teacher_leverage" in soft_labels_df.columns:
        # Vectorized timestamp merge for leverage soft labels
        _has_ts2 = "timestamp" in soft_labels_df.columns and "timestamp" in train_df.columns
        lev_hard = lev_train.to_numpy(dtype=np.float64)
        soft_lev = lev_hard.copy()
        if _has_ts2:
            _sl2 = soft_labels_df.copy()
            _sl2["_ts"] = pd.to_datetime(_sl2["timestamp"], utc=True, errors="coerce")
            _tr2 = train_df[["timestamp"]].copy()
            _tr2["_ts"] = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
            _tr2["_row"] = np.arange(len(_tr2))
            _merged2 = _tr2[["_ts", "_row"]].merge(
                _sl2[["_ts", "teacher_leverage"]], on="_ts", how="left"
            )
            _mask2 = _merged2["teacher_leverage"].notna()
            soft_lev[_merged2["_row"][_mask2].values] = _merged2["teacher_leverage"][_mask2].values
        lev_train_arr = ((1 - distill_alpha) * lev_hard + distill_alpha * soft_lev)
    else:
        lev_train_arr = lev_train.to_numpy(dtype=np.float64)
    lev_train_arr = np.clip(lev_train_arr, 1.0, float(settings.max_leverage))

    asym_w = _asymmetric_lev_weights(lev_train_arr, penalty_factor=3.0)
    if sample_weight is not None:
        lev_sample_w = (sample_weight * asym_w)
        lev_sample_w = (lev_sample_w / lev_sample_w.mean()).clip(0.2, 5.0)
    else:
        lev_sample_w = asym_w

    # ?? 撅?2嚗??典翰??RF ??warm-up ?葫嚗???Pinball 甈?蝎曄毀 ????
    if progress_cb:
        progress_cb(74, "訓練槓桿模型（RF warm-up）")
    rf_warm = RandomForestRegressor(
        n_estimators=50,   # 敹恍?warm-up
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_warm.fit(x_train, lev_train_arr, sample_weight=lev_sample_w)
    lev_pred_warm = rf_warm.predict(x_train)

    # Pinball-style weighting: penalize over-estimated leverage more.
    pinball_w = _pinball_sample_weights(lev_train_arr, lev_pred_warm, tau=0.35)
    final_lev_w = (lev_sample_w * pinball_w)
    final_lev_w = (final_lev_w / final_lev_w.mean()).clip(0.15, 6.0)

    # ?? 撅?3嚗 GradientBoosting Quantile ?飛嚗撱粹?撠迂 loss嚗???
    if progress_cb:
        progress_cb(78, "訓練槓桿模型（GB Quantile loss, tau=0.35）")
    gb_lev = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        loss="quantile",      # ?批遣 Pinball loss
        alpha=0.35,           # ?葫蝚?35 ?曉?雿???憭拍?靽???        random_state=42,
    )
    gb_lev.fit(x_train, lev_train_arr, sample_weight=final_lev_w)

    # RF leverage warm model
    lev_reg_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )
    lev_reg_rf.fit(x_train, lev_train_arr, sample_weight=final_lev_w)
    lev_reg = EnsembleLevReg(lev_reg_rf, gb_lev, rf_w=0.40)

    # Evaluate leverage regression on the test split.
    lev_pred = lev_reg.predict(x_test)
    lev_mae = mean_absolute_error(lev_test.to_numpy(), lev_pred)
    overestimate_rate = float((lev_pred > lev_test.to_numpy()).mean())

    models = TrainedModels(
        clf=clf,
        lev_reg=lev_reg,
        feature_cols=feature_cols,
        backend="sklearn_rf",
        backend_meta={
            "device": "cpu",
            "distilled": distill_applied,
            "lev_loss": "quantile_tau035+asymmetric_weight",
            "overestimate_rate": overestimate_rate,
        },
    )
    metrics = {
        "classification_report": cls_report,
        "leverage_mae": float(lev_mae),
        "leverage_overestimate_rate": overestimate_rate,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_backend": "sklearn_rf",
        "training_device": "cpu",
        "distillation_applied": distill_applied,
        "distill_alpha": distill_alpha if distill_applied else 0.0,
        "lev_loss": "quantile_tau035+asymmetric_weight",
    }
    return models, metrics


def train_models(
    df: pd.DataFrame,
    settings: Settings,
    progress_cb: Callable[[int, str], None] | None = None,
    soft_labels_df: pd.DataFrame | None = None,
    distill_alpha: float = 0.4,
) -> Tuple[TrainedModels, dict]:
    # soft_labels_df: Teacher soft-label DataFrame with timestamp and soft probabilities.
    # distill_alpha: 0.4 means 60% hard labels + 40% teacher guidance.
    max_rows = int(getattr(settings, "max_train_rows", 0) or 0)
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)

    if len(df) < settings.min_train_rows:
        raise RuntimeError(f"Not enough rows for training. Need >= {settings.min_train_rows}, got {len(df)}")

    feature_cols = _feature_columns(df)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    requested = str(settings.train_device or "auto").lower()

    # ?? CPU 璅∪?嚗?渲擗???????????????????????????????????
    if requested == "cpu":
        return _fit_sklearn_rf(
            train_df, test_df, feature_cols, settings, progress_cb,
            soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
        )

    # ?? ?芋撘??岫 torch嚗仃?? fallback ??sklearn嚗?賊冗嚗????????
    if requested in {"auto", "cloud", "npu", "directml", "cuda", "gpu", "mps"}:
        try:
            return _fit_torch_accelerated(
                train_df, test_df, feature_cols, settings, requested, progress_cb,
                soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
            )
        except Exception as e:  # noqa: BLE001
            if requested in {"npu", "directml", "cuda", "gpu", "mps"} and settings.npu_strict:
                raise RuntimeError(f"Accelerated mode enabled, but accelerator training failed: {e}") from e
            models, metrics = _fit_sklearn_rf(
                train_df, test_df, feature_cols, settings, progress_cb,
                soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
            )
            metrics["training_note"] = f"Accelerator unavailable, fallback to CPU (+distill): {e}"
            return models, metrics

    return _fit_sklearn_rf(
        train_df, test_df, feature_cols, settings, progress_cb,
        soft_labels_df=soft_labels_df, distill_alpha=distill_alpha,
    )


def save_models(models: TrainedModels, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    backend = str(models.backend)
    if backend.startswith("torch") or backend == "student_net":
        try:
            import torch  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"torch is required to save torch models: {e}") from e

        if backend == "student_net":
            # clf and lev_reg share the same StudentNet instance — save once
            bundle: dict = {
                "backend":         "student_net",
                "feature_cols":    models.feature_cols,
                "backend_meta":    models.backend_meta or {},
                "student_state_dict": models.clf.model.state_dict(),
            }
        else:
            # Legacy torch_accel format: separate clf and reg models
            bundle = {
                "backend":         models.backend,
                "feature_cols":    models.feature_cols,
                "backend_meta":    models.backend_meta or {},
                "clf_state_dict":  models.clf.model.state_dict(),
                "lev_state_dict":  models.lev_reg.model.state_dict(),
            }
        torch.save(bundle, model_dir / "torch_models.pt")
        return

    joblib.dump(models.clf, model_dir / "signal_clf.joblib")
    joblib.dump(models.lev_reg, model_dir / "leverage_reg.joblib")
    joblib.dump(models.feature_cols, model_dir / "feature_cols.joblib")


def load_models(model_dir: Path) -> TrainedModels:
    torch_bundle_path = model_dir / "torch_models.pt"
    clf_path = model_dir / "signal_clf.joblib"
    lev_path = model_dir / "leverage_reg.joblib"
    feat_path = model_dir / "feature_cols.joblib"
    _has_sklearn = clf_path.exists() and lev_path.exists() and feat_path.exists()

    if torch_bundle_path.exists():
        try:
            import torch  # type: ignore
        except Exception as torch_err:  # noqa: BLE001
            if _has_sklearn:
                import warnings
                warnings.warn(
                    f"[load_models] torch unavailable ({torch_err}); using sklearn fallback.",
                    stacklevel=2,
                )
                clf = joblib.load(clf_path)
                lev_reg = joblib.load(lev_path)
                feature_cols = joblib.load(feat_path)
                return TrainedModels(
                    clf=clf, lev_reg=lev_reg, feature_cols=feature_cols,
                    backend="sklearn_rf",
                    backend_meta={"device": "cpu", "note": "torch_unavailable_sklearn_fallback"},
                )
            raise RuntimeError(
                f"torch is required but unavailable: {torch_err}\n"
                "Use the local .venv311 to run the dashboard or install a torch-compatible environment.\n"
                "If you only need inference, keep the sklearn .joblib models available.",
            ) from torch_err

        # ?? torch ?舐嚗迤撣貉???torch bundle ???????????????????????????
        bundle = torch.load(torch_bundle_path, map_location="cpu")
        backend_name = str(bundle.get("backend", "torch_accel"))
        meta = bundle.get("backend_meta") or {}
        feature_cols = list(bundle.get("feature_cols") or [])
        in_dim = int(meta.get("in_dim", 0))
        class_values = np.array(meta.get("class_values", [-1, 0, 1]), dtype=int)
        mean = np.array(meta.get("mean", []), dtype=np.float32)
        scale = np.array(meta.get("scale", []), dtype=np.float32)

        device_str = str(meta.get("device", "")).lower()
        device = torch.device("cpu")
        if "cuda" in device_str and torch.cuda.is_available():
            device = torch.device("cuda")
        elif "mps" in device_str and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif "privateuseone" in device_str or "directml" in device_str:
            try:
                import torch_directml  # type: ignore
                device = torch_directml.device()
            except Exception:
                device = torch.device("cpu")

        if in_dim <= 0:
            raise RuntimeError("Invalid torch model bundle: missing input dimension.")

        if backend_name == "student_net":
            # Reconstruct dual-head StudentNet
            student = _make_student_net(torch, in_dim)
            student.load_state_dict(bundle["student_state_dict"])
            student = student.to(device)
            student.eval()
            return TrainedModels(
                clf     = _StudentNetClfWrapper(student, device, mean, scale, class_values, torch),
                lev_reg = _StudentNetLevWrapper(student, device, mean, scale, torch),
                feature_cols = feature_cols,
                backend      = backend_name,
                backend_meta = meta,
            )

        # Legacy torch_accel format: separate clf + reg models
        cls_model = _build_cls_mlp(torch, in_dim, len(class_values))
        reg_model = _build_reg_mlp(torch, in_dim)
        cls_model.load_state_dict(bundle["clf_state_dict"])
        reg_model.load_state_dict(bundle["lev_state_dict"])
        cls_model = cls_model.to(device)
        reg_model = reg_model.to(device)

        return TrainedModels(
            clf     = _TorchSignalWrapper(cls_model, device, mean, scale, class_values, torch),
            lev_reg = _TorchLevWrapper(reg_model, device, mean, scale, torch),
            feature_cols = feature_cols,
            backend      = backend_name,
            backend_meta = meta,
        )

    if not _has_sklearn:
        raise FileNotFoundError(
            "No torch bundle was found, and the fallback sklearn .joblib files are missing.",
        )

    clf = joblib.load(clf_path)
    lev_reg = joblib.load(lev_path)
    feature_cols = joblib.load(feat_path)
    return TrainedModels(
        clf=clf, lev_reg=lev_reg, feature_cols=feature_cols,
        backend="sklearn_rf", backend_meta={"device": "cpu"},
    )


def infer_signals(df: pd.DataFrame, models: TrainedModels, settings: Settings) -> pd.DataFrame:
    x = _clean_xy(df, models.feature_cols)
    proba = models.clf.predict_proba(x)
    classes = list(models.clf.classes_)

    idx_map = {c: i for i, c in enumerate(classes)}
    p_long = proba[:, idx_map.get(1, 0)] if 1 in idx_map else np.zeros(len(df))
    p_short = proba[:, idx_map.get(-1, 0)] if -1 in idx_map else np.zeros(len(df))
    p_flat = proba[:, idx_map.get(0, 0)] if 0 in idx_map else np.zeros(len(df))

    signal_threshold = settings.get_signal_threshold()
    signal = np.where(
        (p_long > signal_threshold) & (p_long > p_short),
        1,
        np.where((p_short > signal_threshold) & (p_short > p_long), -1, 0),
    )

    # SNR-aware override: when multi-layer supports/resistances are clearly broken,
    # allow breaking out of flat state even if base probability is slightly below threshold.
    snr_break_s = pd.to_numeric(df.get("snr_break_support_count", 0), errors="coerce").fillna(0).to_numpy()
    snr_break_r = pd.to_numeric(df.get("snr_break_resistance_count", 0), errors="coerce").fillna(0).to_numpy()
    snr_overlap_s = pd.to_numeric(df.get("snr_overlap_support_count", 0), errors="coerce").fillna(0).to_numpy()
    snr_overlap_r = pd.to_numeric(df.get("snr_overlap_resistance_count", 0), errors="coerce").fillna(0).to_numpy()

    snr_strong_bear = (snr_break_s >= 3) & (snr_break_s >= (snr_break_r + 1)) & (snr_overlap_s >= 2)
    snr_strong_bull = (snr_break_r >= 3) & (snr_break_r >= (snr_break_s + 1)) & (snr_overlap_r >= 2)
    soft_th = max(0.33, float(signal_threshold) - 0.08)
    promote_short = (signal == 0) & snr_strong_bear & (p_short >= soft_th) & (p_short > p_long)
    promote_long = (signal == 0) & snr_strong_bull & (p_long >= soft_th) & (p_long > p_short)
    signal = np.where(promote_short, -1, np.where(promote_long, 1, signal))

    raw_lev = models.lev_reg.predict(x)
    confidence = np.maximum(p_long, p_short) - p_flat
    conf_scale = np.clip(confidence * 2.0, 0.2, 1.0)

    max_safe_lev = compute_max_safe_leverage(df, settings.max_leverage)

    confidence_index = np.maximum(p_long, p_short)
    out = df.copy()
    out["p_long"] = p_long
    out["p_short"] = p_short
    out["p_flat"] = p_flat
    out["signal"] = signal
    out["confidence_index"] = confidence_index.round(4)

    atr_pct = pd.to_numeric(out["atr_pct"], errors="coerce") if "atr_pct" in out.columns else pd.Series(np.nan, index=out.index)
    realized_vol = pd.to_numeric(out["realized_vol_24"], errors="coerce") if "realized_vol_24" in out.columns else pd.Series(np.nan, index=out.index)
    atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).fillna(0.015)
    realized_vol = realized_vol.replace([np.inf, -np.inf], np.nan).fillna(0.03)

    regime = out["regime"].astype(str).str.lower() if "regime" in out.columns else pd.Series("ranging", index=out.index, dtype=str)
    plus_di = pd.to_numeric(out["plus_di"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if "plus_di" in out.columns else pd.Series(0.0, index=out.index)
    minus_di = pd.to_numeric(out["minus_di"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if "minus_di" in out.columns else pd.Series(0.0, index=out.index)

    regime_bias = np.where(plus_di.to_numpy() >= minus_di.to_numpy(), 1, -1)
    regime_strength = np.clip(np.abs(plus_di.to_numpy() - minus_di.to_numpy()) / 100.0, 0.0, 1.0)
    expected_move_pct = np.clip(np.maximum(atr_pct.to_numpy(), realized_vol.to_numpy()) * (0.45 + 1.1 * confidence_index), 0.0, 0.25)
    round_trip_cost_pct = (
        ((float(settings.fee_bps) + float(settings.slippage_bps)) * 2.0) / 10_000.0
        + (float(getattr(settings, "funding_rate_8h_bps", 2.5) or 2.5) / 10_000.0) * (float(settings.future_horizon_hours) / 8.0)
    )
    buffer_pct = np.where(
        regime.to_numpy() == "trend",
        0.0010,
        np.where(regime.to_numpy() == "volatile", 0.0018, 0.0015),
    )
    expected_cost_pct = round_trip_cost_pct + buffer_pct
    net_edge_pct = expected_move_pct - expected_cost_pct

    regime_leverage_cap = np.where(
        regime.to_numpy() == "trend",
        np.minimum(float(settings.max_leverage), np.maximum(1.0, max_safe_lev)),
        np.where(
            regime.to_numpy() == "volatile",
            np.minimum(2.0, np.maximum(1.0, max_safe_lev)),
            np.minimum(1.5, np.maximum(1.0, max_safe_lev)),
        ),
    )
    leverage = np.clip(raw_lev * conf_scale, 1, settings.max_leverage)
    leverage = np.minimum(leverage, max_safe_lev)
    leverage = np.minimum(leverage, regime_leverage_cap)

    out["suggested_leverage"] = leverage.round(2)
    out["max_safe_leverage"] = max_safe_lev.round(2)
    out["regime_bias"] = regime_bias
    out["regime_strength"] = np.round(regime_strength, 4)
    out["expected_move_pct"] = np.round(expected_move_pct, 4)
    out["expected_cost_pct"] = np.round(expected_cost_pct, 4)
    out["net_edge_pct"] = np.round(net_edge_pct, 4)
    out["regime_alignment"] = np.where(signal == regime_bias, 1, 0)
    out["snr_strong_bear_break"] = snr_strong_bear.astype(int)
    out["snr_strong_bull_break"] = snr_strong_bull.astype(int)

    trade_allowed = signal != 0
    block_reason = np.full(len(out), "", dtype=object)
    flat_mask = signal == 0
    edge_mask = (~flat_mask) & (net_edge_pct <= 0)
    trend_mask = (regime.to_numpy() == "trend") & (~flat_mask) & (signal != regime_bias)
    volatile_mask = (regime.to_numpy() == "volatile") & (~flat_mask) & (confidence_index < (signal_threshold + 0.05))
    ranging_mask = (regime.to_numpy() == "ranging") & (~flat_mask) & (confidence_index < (signal_threshold + 0.02))
    snr_break_override_mask = (signal != 0) & (snr_strong_bear | snr_strong_bull)
    volatile_mask = volatile_mask & (~snr_break_override_mask)
    ranging_mask = ranging_mask & (~snr_break_override_mask)
    trade_allowed = trade_allowed & (~edge_mask) & (~trend_mask) & (~volatile_mask) & (~ranging_mask)
    block_reason = np.where(flat_mask, "flat signal", block_reason)
    block_reason = np.where(edge_mask, "expected edge <= cost", block_reason)
    block_reason = np.where(trend_mask, "trend regime mismatch", block_reason)
    block_reason = np.where(volatile_mask, "volatile regime needs stronger confidence", block_reason)
    block_reason = np.where(ranging_mask, "ranging regime needs stronger confidence", block_reason)
    out["trade_allowed"] = trade_allowed.astype(int)
    out["trade_block_reason"] = block_reason
    out["trade_net_edge_pct"] = np.round(net_edge_pct, 4)
    out["trade_expected_cost_pct"] = np.round(expected_cost_pct, 4)

    # ── Vectorized ai_style classification ────────────────────────────────────
    # Replaces O(n) Python row loop with numpy operations for 10-100x speedup.
    _fg = (
        pd.to_numeric(out["fear_greed_value"], errors="coerce").fillna(50.0).to_numpy()
        if "fear_greed_value" in out.columns
        else np.full(len(out), 50.0)
    )
    _vol24 = (
        pd.to_numeric(out["realized_vol_24"], errors="coerce").fillna(0.03).to_numpy()
        if "realized_vol_24" in out.columns
        else np.full(len(out), 0.03)
    )
    _atr_p = (
        pd.to_numeric(out["atr_pct"], errors="coerce").fillna(0.015).to_numpy()
        if "atr_pct" in out.columns
        else np.full(len(out), 0.015)
    )
    _conf = confidence_index  # already np.ndarray
    _macd_h = (
        pd.to_numeric(out["macd_hist"], errors="coerce").fillna(0.0).to_numpy()
        if "macd_hist" in out.columns
        else np.zeros(len(out))
    )
    _dd = (
        pd.to_numeric(out["drawdown"], errors="coerce").fillna(0.0).to_numpy()
        if "drawdown" in out.columns
        else np.zeros(len(out))
    )

    _s = np.zeros(len(out), dtype=np.float64)
    _s += np.where(_fg >= 75, 1.2, np.where(_fg >= 55, 0.6, np.where(_fg <= 25, -1.5, np.where(_fg <= 40, -0.7, 0.0))))
    _s += np.where(_vol24 < 0.015, 0.8, np.where(_vol24 < 0.025, 0.3, np.where(_vol24 > 0.06, -1.2, np.where(_vol24 > 0.04, -0.6, 0.0))))
    _s += np.where(_atr_p < 0.008, 0.5, np.where(_atr_p > 0.025, -0.8, 0.0))
    _s += np.where(_conf >= 0.35, 0.8, np.where(_conf >= 0.20, 0.3, np.where(_conf < 0.05, -0.5, 0.0)))
    _s += np.where(_macd_h != 0, 0.4 * np.sign(_macd_h), 0.0)
    _s += np.where(_dd < -0.15, -1.0, np.where(_dd < -0.08, -0.5, 0.0))
    _s = np.clip(_s, -3.0, 3.0)
    out["ai_style"] = np.where(_s >= 0.8, "aggressive", np.where(_s <= -0.6, "conservative", "neutral"))
    return out



def compute_max_safe_leverage(df: pd.DataFrame, hard_cap: int) -> np.ndarray:
    atr_pct = df["atr_pct"].replace(0, np.nan).ffill().fillna(0.01)
    vol = df["realized_vol_24"].replace(0, np.nan).ffill().fillna(0.03)
    drawdown = df["drawdown"].abs().fillna(0)

    # Lower volatility and lower drawdown allow higher leverage.
    base = 0.03 / (atr_pct + vol)
    dd_penalty = np.clip(1 - drawdown * 2.5, 0.1, 1.0)
    lev = base * dd_penalty * 12
    lev = np.clip(lev, 1, hard_cap)
    return lev.to_numpy()
