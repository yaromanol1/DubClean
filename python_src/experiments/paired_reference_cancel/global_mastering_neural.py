#!/usr/bin/env python
"""Neural global mastering/channel estimator.

This is intentionally not a waveform model.  It receives robust statistics
from several speech-free original/dubbed M&E windows and predicts one bounded
static profile (gain + smooth mel-EQ) for the whole film.  The predicted JSON is
compatible with ``ReferenceAdapterProfile`` and is applied by the existing
``adapt_reference_file_with_profile`` path.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from experiments.paired_reference_cancel.global_mastering_profile import (
    PROFILE_MODE as ALGORITHMIC_GLOBAL_MODE,
    fit_global_profile_for_files,
)
from experiments.paired_reference_cancel.reference_adapter import (
    ReferenceAdapterProfile,
    _mel_filterbank,
)
from experiments.paired_reference_cancel.storage import atomic_json

FORMAT = "neural_global_mastering_estimator_v1"
PROFILE_MODE = "neural_global_mastering_speechfree_v1"

MODEL_CONFIG_KEYS = {
    "bands",
    "hidden",
    "layers",
    "dropout",
    "max_abs_gain_db",
}


def effective_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = {
        "bands": 40,
        "hidden": 192,
        "layers": 4,
        "dropout": 0.05,
        "max_abs_gain_db": 12.0,
    }
    if config:
        merged.update({key: config[key] for key in MODEL_CONFIG_KEYS if key in config})
    merged["bands"] = int(merged["bands"])
    merged["hidden"] = int(merged["hidden"])
    merged["layers"] = int(merged["layers"])
    merged["dropout"] = float(merged["dropout"])
    merged["max_abs_gain_db"] = float(merged["max_abs_gain_db"])
    return merged


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, str(temporary))
    os.replace(temporary, path)


class GlobalMasteringEstimator(nn.Module):
    """Predicts a smooth bounded band-gain profile from robust observations."""

    def __init__(
        self,
        *,
        bands: int = 40,
        hidden: int = 192,
        layers: int = 4,
        dropout: float = 0.05,
        max_abs_gain_db: float = 12.0,
    ) -> None:
        super().__init__()
        self.bands = int(bands)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.max_abs_gain_db = float(max_abs_gain_db)
        input_dim = self.bands * 4 + 4
        modules: list[nn.Module] = []
        width = input_dim
        for _ in range(max(1, self.layers)):
            modules.append(nn.Linear(width, self.hidden))
            modules.append(nn.GELU())
            modules.append(nn.LayerNorm(self.hidden))
            if self.dropout > 0:
                modules.append(nn.Dropout(self.dropout))
            width = self.hidden
        modules.append(nn.Linear(width, self.bands))
        self.network = nn.Sequential(*modules)
        self._init_identity_bias()

    def _init_identity_bias(self) -> None:
        last = None
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        observed = features[:, : self.bands] * self.max_abs_gain_db
        residual = self.network(features)
        return torch.clamp(
            observed + residual,
            -self.max_abs_gain_db,
            self.max_abs_gain_db,
        )


def build_model_from_config(config: dict[str, Any] | None = None) -> GlobalMasteringEstimator:
    return GlobalMasteringEstimator(**effective_model_config(config))


def model_config_from(model: GlobalMasteringEstimator) -> dict[str, Any]:
    return {
        "bands": int(model.bands),
        "hidden": int(model.hidden),
        "layers": int(model.layers),
        "dropout": float(model.dropout),
        "max_abs_gain_db": float(model.max_abs_gain_db),
    }


def load_estimator_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> tuple[GlobalMasteringEstimator, dict[str, Any]]:
    payload = torch.load(str(path), map_location=map_location)
    config = effective_model_config(payload.get("model_config") or payload.get("config") or {})
    model = build_model_from_config(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def _centers_for_profile(sample_rate: int, n_fft: int, bands: int) -> list[float]:
    _filters, centers = _mel_filterbank(sample_rate, n_fft, bands)
    return [float(value) for value in centers]


def _smooth(values: np.ndarray) -> np.ndarray:
    return gaussian_filter1d(np.asarray(values, dtype=np.float32), sigma=1.0, mode="nearest")


def features_from_profile(profile: dict[str, Any]) -> np.ndarray:
    """Builds model input from an algorithmic global profile report."""
    gains = np.asarray(profile.get("band_gains_db") or [], dtype=np.float32)
    bands = int(profile.get("bands") or len(gains) or 40)
    if len(gains) != bands:
        padded = np.zeros(bands, dtype=np.float32)
        padded[: min(len(gains), bands)] = gains[: min(len(gains), bands)]
        gains = padded
    iqr = np.asarray(profile.get("band_spread_iqr_db") or np.zeros(bands), dtype=np.float32)
    if len(iqr) != bands:
        resized = np.zeros(bands, dtype=np.float32)
        resized[: min(len(iqr), bands)] = iqr[: min(len(iqr), bands)]
        iqr = resized
    confidence = 1.0 / (1.0 + np.maximum(iqr, 0.0) / 6.0)
    smooth_gains = _smooth(gains)
    residual = np.clip(gains - smooth_gains, -12.0, 12.0)
    summary = profile.get("summary") or {}
    selection = profile.get("selection") or {}
    median_iqr_db = summary.get("median_band_iqr_db")
    if median_iqr_db is None:
        median_iqr_db = float(np.median(iqr)) if len(iqr) else 0.0
    scalar = np.asarray(
        [
            float(summary.get("fit_segments") or 0.0) / 20.0,
            float(summary.get("holdout_segments") or 0.0) / 8.0,
            float(median_iqr_db) / 12.0,
            float(selection.get("rejected_coherence") or 0.0) / max(float(selection.get("windows_total") or 1.0), 1.0),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            gains / 12.0,
            iqr / 12.0,
            confidence,
            residual / 12.0,
            scalar,
        ]
    )
    return np.nan_to_num(features, copy=False).astype(np.float32)


def refine_profile_with_model(
    profile: dict[str, Any],
    checkpoint: Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_estimator_checkpoint(checkpoint, map_location=device)
    model.to(device)
    features = torch.from_numpy(features_from_profile(profile))[None, :].to(device)
    with torch.no_grad():
        predicted = model(features)[0].detach().cpu().numpy().astype(np.float32)
    config = effective_model_config(payload.get("model_config") or payload.get("config") or {})
    predicted = _smooth(predicted)
    predicted = np.clip(predicted, -float(config["max_abs_gain_db"]), float(config["max_abs_gain_db"]))
    result = dict(profile)
    result["mode"] = PROFILE_MODE
    result["band_gains_db"] = [float(value) for value in predicted]
    result["neural_global_mastering"] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_format": payload.get("format", ""),
        "epoch": payload.get("epoch"),
        "model_config": config,
        "source_profile_mode": profile.get("mode") or ALGORITHMIC_GLOBAL_MODE,
    }
    summary = dict(result.get("summary") or {})
    summary.update(
        {
            "verdict": "ok" if summary.get("verdict") not in {"insufficient_segments"} else summary.get("verdict"),
            "neural_refined": True,
            "median_gain_db": float(np.median(predicted)) if len(predicted) else 0.0,
            "min_gain_db": float(np.min(predicted)) if len(predicted) else 0.0,
            "max_gain_db": float(np.max(predicted)) if len(predicted) else 0.0,
        }
    )
    result["summary"] = summary
    # Sanity: keep JSON consumable by the existing applier.
    ReferenceAdapterProfile.from_json(result)
    return result


def fit_neural_global_profile_for_files(
    original_mix_path: Path,
    dubbed_mix_path: Path,
    original_speech_path: Path,
    dubbed_speech_path: Path,
    checkpoint: Path,
    *,
    report_path: Path | None = None,
    algorithmic_report_path: Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    algorithmic = fit_global_profile_for_files(
        original_mix_path,
        dubbed_mix_path,
        original_speech_path,
        dubbed_speech_path,
        report_path=algorithmic_report_path,
        **kwargs,
    )
    refined = refine_profile_with_model(algorithmic, checkpoint)
    if report_path is not None:
        atomic_json(Path(report_path), refined)
    return refined


def save_checkpoint(
    path: Path,
    *,
    model: GlobalMasteringEstimator,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    payload = {
        "format": FORMAT,
        "epoch": int(epoch),
        "model_config": model_config_from(model),
        "config": effective_model_config(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "history": history,
    }
    atomic_torch_save(path, payload)


def profile_json_from_gains(
    gains_db: np.ndarray,
    *,
    sample_rate: int = 48000,
    n_fft: int = 1024,
    hop: int = 512,
) -> dict[str, Any]:
    gains = np.asarray(gains_db, dtype=np.float32)
    bands = int(len(gains))
    profile = {
        "schema_version": 1,
        "mode": PROFILE_MODE,
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "hop": int(hop),
        "bands": bands,
        "band_centers_hz": _centers_for_profile(sample_rate, n_fft, bands),
        "band_gains_db": [float(value) for value in gains],
        "delay_sec": 0.0,
        "delay_confidence": 0.0,
        "distance_before": {},
        "distance_after_estimate": {},
        "summary": {
            "median_gain_db": float(np.median(gains)) if bands else 0.0,
            "min_gain_db": float(np.min(gains)) if bands else 0.0,
            "max_gain_db": float(np.max(gains)) if bands else 0.0,
            "distance_reduction_percent": 0.0,
            "neural_refined": True,
        },
    }
    ReferenceAdapterProfile.from_json(profile)
    return profile
