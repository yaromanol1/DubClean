#!/usr/bin/env python
"""Parametric neural adapter for EN reference mastering.

The adapter deliberately cannot synthesize new speech from the EN+RU mixture.
It predicts only bounded mel-band gains and applies them to the supplied
reference magnitude.  This keeps the failure mode safe: the model may make the
reference too weak/strong, but it cannot copy Russian speech from the mixture
into the reference that the separator will later remove.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from experiments.paired_reference_cancel import audio_io


FORMAT = "neural_reference_adapter_v1"
MODEL_CONFIG_KEYS = {
    "sample_rate",
    "n_fft",
    "hop_length",
    "bands",
    "hidden",
    "layers",
    "min_gain_db",
    "max_gain_db",
    "mix_smoothing_frames",
}


def model_config_from(config: dict[str, Any]) -> dict[str, Any]:
    """Keep only the keys the model constructor accepts.

    Training configs also carry batch_size/learning_rate/workers; every model
    key present here MUST reach the constructor, otherwise the checkpoint
    config and the trained weights silently disagree (the v3 smoothmix bug).
    """
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


def build_model_from_config(config: dict[str, Any]) -> MelGainReferenceAdapter:
    return MelGainReferenceAdapter(**model_config_from(config))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if values.shape[1] > 1:
        values = np.mean(values, axis=1)
    else:
        values = values[:, 0]
    return np.asarray(values, dtype=np.float32), int(rate)


def _match_length(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(left), len(right))
    return left[:length], right[:length]


def _mel_filterbank(sample_rate: int, n_fft: int, bands: int) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)

    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(np.array([50.0]))[0], hz_to_mel(np.array([sample_rate / 2.0]))[0], bands + 2)
    hz_points = mel_to_hz(mel_points)
    fb = np.zeros((bands, len(frequencies)), dtype=np.float32)
    for index in range(bands):
        left, center, right = hz_points[index : index + 3]
        up = (frequencies - left) / max(center - left, 1e-6)
        down = (right - frequencies) / max(right - center, 1e-6)
        fb[index] = np.maximum(0.0, np.minimum(up, down))
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-8)
    return fb.astype(np.float32)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = int(dilation)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.PReLU(channels),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.PReLU(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(value + self.net(value))


class MelGainReferenceAdapter(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        n_fft: int = 512,
        hop_length: int = 256,
        bands: int = 40,
        hidden: int = 96,
        layers: int = 5,
        min_gain_db: float = -30.0,
        max_gain_db: float = 12.0,
        mix_smoothing_frames: int = 1,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.bands = int(bands)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.min_gain_db = float(min_gain_db)
        self.max_gain_db = float(max_gain_db)
        self.mix_smoothing_frames = max(1, int(mix_smoothing_frames))
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        mel = _mel_filterbank(self.sample_rate, self.n_fft, self.bands)
        self.register_buffer("mel_filter", torch.from_numpy(mel), persistent=False)
        input_channels = self.bands * 5
        blocks: list[nn.Module] = [
            nn.Conv1d(input_channels, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.PReLU(hidden),
        ]
        for layer in range(layers):
            blocks.append(ResidualTemporalBlock(hidden, dilation=2 ** layer))
        blocks.extend([nn.Conv1d(hidden, hidden, 1), nn.PReLU(hidden), nn.Conv1d(hidden, self.bands, 1)])
        self.net = nn.Sequential(*blocks)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Conv1d):
                final.weight.zero_()
                zero_gain_position = (0.0 - self.min_gain_db) / max(self.max_gain_db - self.min_gain_db, 1e-6)
                zero_gain_position = float(min(max(zero_gain_position, 1e-4), 1.0 - 1e-4))
                final.bias.fill_(math.log(zero_gain_position / (1.0 - zero_gain_position)))

    def stft(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(waveform.device),
            center=True,
            return_complex=True,
        )

    def istft(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(spectrum.device),
            center=True,
            length=length,
        )

    def mel(self, magnitude: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mf,bft->bmt", self.mel_filter.to(magnitude.device), magnitude)

    def _smooth_mixture_context(self, mix_mel: torch.Tensor) -> torch.Tensor:
        """Return a deliberately low-detail mixture envelope.

        The adapter is allowed to use the mixture only as a channel cue.  It must
        not get enough temporal detail to follow RU speech syllables.  A
        ~100 ms moving average keeps EQ/level/ducking information while making
        the context too coarse to copy speech content.
        """
        if self.mix_smoothing_frames <= 1 or mix_mel.shape[-1] < 3:
            return mix_mel
        wanted = self.mix_smoothing_frames if self.mix_smoothing_frames % 2 == 1 else self.mix_smoothing_frames + 1
        kernel = min(wanted, mix_mel.shape[-1] if mix_mel.shape[-1] % 2 == 1 else mix_mel.shape[-1] - 1)
        kernel = max(3, int(kernel))
        padding = kernel // 2
        return F.avg_pool1d(mix_mel, kernel_size=kernel, stride=1, padding=padding)

    def _features(self, reference_mag: torch.Tensor, mixture_mag: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        ref_mel = self.mel(reference_mag)
        mix_mel = self._smooth_mixture_context(self.mel(mixture_mag))
        ref_log = torch.log(ref_mel + eps)
        mix_log = torch.log(mix_mel + eps)
        diff = mix_log - ref_log
        similarity = torch.tanh(diff)
        ref_norm = ref_log - ref_log.mean(dim=2, keepdim=True)
        return torch.cat([ref_log, mix_log, diff, similarity, ref_norm], dim=1)

    def predict_gains_db_from_mag(
        self, reference_mag: torch.Tensor, mixture_mag: torch.Tensor
    ) -> torch.Tensor:
        features = self._features(reference_mag, mixture_mag)
        raw = self.net(features)
        return self.min_gain_db + (self.max_gain_db - self.min_gain_db) * torch.sigmoid(raw)

    def expand_gains_to_frequency(self, gains_db: torch.Tensor, frequency_bins: int) -> torch.Tensor:
        gains = gains_db.unsqueeze(1)
        expanded = F.interpolate(
            gains,
            size=(frequency_bins, gains_db.shape[-1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return torch.pow(10.0, expanded / 20.0)

    def forward(self, mixture: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        reference_spectrum = self.stft(reference)
        mixture_spectrum = self.stft(mixture)
        reference_mag = torch.abs(reference_spectrum)
        mixture_mag = torch.abs(mixture_spectrum)
        gains_db = self.predict_gains_db_from_mag(reference_mag, mixture_mag)
        gains = self.expand_gains_to_frequency(gains_db, reference_mag.shape[1])
        adapted_mag = reference_mag * gains
        adapted_spectrum = torch.polar(adapted_mag, torch.angle(reference_spectrum))
        adapted_wave = self.istft(adapted_spectrum, length=reference.shape[-1])
        return {
            "adapted_wave": adapted_wave,
            "adapted_mag": adapted_mag,
            "reference_mag": reference_mag,
            "mixture_mag": mixture_mag,
            "gains_db": gains_db,
        }


def load_adapter_checkpoint(checkpoint_path: Path, device: torch.device | str) -> tuple[MelGainReferenceAdapter, dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device)
    # "model_config" is written by the trainer from the exact kwargs the model
    # was constructed with; older checkpoints only carry the training config.
    raw_config = dict(payload.get("model_config") or payload.get("config") or {})
    config = model_config_from(raw_config)
    model = MelGainReferenceAdapter(**config).to(device)
    state = payload.get("model_state") or payload.get("state_dict")
    if state is None:
        raise RuntimeError(f"В checkpoint нет model_state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def adapt_reference_file_neural(
    reference_path: Path,
    mixture_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
    block_sec: float = 30.0,
    device: str | None = None,
) -> dict[str, Any]:
    device_value = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device_value)
    model, payload = load_adapter_checkpoint(Path(checkpoint_path), torch_device)
    reference, ref_rate = _read_mono(Path(reference_path))
    mixture, mix_rate = _read_mono(Path(mixture_path))
    if mix_rate != ref_rate:
        mixture = audio_io.resample(mixture[:, None], mix_rate, ref_rate)[:, 0]
    reference, mixture = _match_length(reference, mixture)
    if ref_rate != int(model.sample_rate):
        reference_model = audio_io.resample(reference[:, None], ref_rate, model.sample_rate)[:, 0]
        mixture_model = audio_io.resample(mixture[:, None], ref_rate, model.sample_rate)[:, 0]
    else:
        reference_model = reference
        mixture_model = mixture
    block = int(round(block_sec * model.sample_rate))
    margin = int(round(0.5 * model.sample_rate))
    chunks: list[np.ndarray] = []
    gains: list[float] = []
    with torch.no_grad():
        cursor = 0
        total = len(reference_model)
        while cursor < total:
            stop = min(total, cursor + block)
            read_left = max(0, cursor - margin)
            read_right = min(total, stop + margin)
            ref_block = reference_model[read_left:read_right]
            mix_block = mixture_model[read_left:read_right]
            ref_tensor = torch.from_numpy(ref_block[None, :]).to(torch_device)
            mix_tensor = torch.from_numpy(mix_block[None, :]).to(torch_device)
            output = model(mix_tensor, ref_tensor)
            wave = output["adapted_wave"][0].detach().cpu().numpy().astype(np.float32)
            trim_left = cursor - read_left
            chunk = wave[trim_left : trim_left + (stop - cursor)]
            chunks.append(chunk)
            gains.append(float(output["gains_db"].mean().detach().cpu()))
            cursor = stop
    adapted = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
    if ref_rate != int(model.sample_rate):
        adapted = audio_io.resample(adapted[:, None], model.sample_rate, ref_rate)[:, 0]
        adapted = adapted[: len(reference)]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.clip(adapted, -1.0, 1.0), ref_rate, format="FLAC", subtype="PCM_24")
    report = {
        "schema_version": 1,
        "mode": FORMAT,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_id": payload.get("dataset_id", ""),
        "epoch": payload.get("epoch"),
        "reference": str(Path(reference_path).resolve()),
        "mixture": str(Path(mixture_path).resolve()),
        "output": str(output_path.resolve()),
        "sample_rate": ref_rate,
        "duration_sec": float(len(reference) / max(ref_rate, 1)),
        "mean_predicted_gain_db": float(np.mean(gains)) if gains else 0.0,
    }
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report
