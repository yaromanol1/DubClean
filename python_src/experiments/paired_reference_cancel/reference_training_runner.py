#!/usr/bin/env python
"""PyTorch runner for supervised reference-conditioned audio subtraction."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from experiments.paired_reference_cancel.recipe_audio import (
    load_recipe_example as synthesize_recipe_example,
)


def emit(kind: str, **values) -> None:
    print(json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.mean(data, axis=1).astype(np.float32), int(sample_rate)


def write_audio(path: Path, data: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_16")
    return str(path.resolve())


def read_interval_mono(
    path: str | Path,
    start_sec: float,
    duration_sec: float,
    target_rate: int,
    target_frames: int,
) -> np.ndarray:
    with sf.SoundFile(str(path), "r") as reader:
        source_rate = int(reader.samplerate)
        requested = int(round(duration_sec * source_rate))
        start_frame = int(round(start_sec * source_rate))
        prefix = max(0, -start_frame)
        reader.seek(max(0, start_frame))
        available = max(0, min(requested - prefix, len(reader) - max(0, start_frame)))
        data = reader.read(available, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1, dtype=np.float64).astype(np.float32)
    if prefix:
        mono = np.pad(mono, (prefix, 0))
    if len(mono) < requested:
        mono = np.pad(mono, (0, requested - len(mono)))
    mono = mono[:requested]
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        mono = signal.resample_poly(
            mono, target_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    if len(mono) != target_frames:
        mono = signal.resample(mono, target_frames).astype(np.float32)
    return np.nan_to_num(mono[:target_frames], copy=False).astype(np.float32)


def transform_reference(
    reference: np.ndarray, recipe: dict, sample_rate: int
) -> np.ndarray:
    value = reference.astype(np.float64)
    spectrum = np.fft.rfft(value)
    frequencies = np.fft.rfftfreq(len(value), 1.0 / sample_rate)
    tilt = float(recipe["spectral_tilt_db"])
    normalized = np.log2(np.maximum(frequencies, 80.0) / 1000.0)
    curve_db = np.clip(normalized * tilt / 4.0, -abs(tilt), abs(tilt))
    value = np.fft.irfft(
        spectrum * np.power(10.0, curve_db / 20.0), n=len(value)
    )
    drive = float(recipe["soft_compression_drive"])
    if drive > 1.0001:
        value = np.tanh(value * drive) / np.tanh(drive)
    gain_change = float(recipe["english_gain_change_db"])
    envelope_db = np.linspace(-gain_change / 2.0, gain_change / 2.0, len(value))
    value *= np.power(10.0, envelope_db / 20.0)
    wet = float(recipe["reverb_wet"])
    if wet > 1e-6:
        impulse_length = max(8, int(round(sample_rate * 0.18)))
        impulse = np.exp(-np.arange(impulse_length) / max(1.0, sample_rate * 0.045))
        impulse[0] = 1.0
        reverberated = signal.fftconvolve(value, impulse, mode="full")[: len(value)]
        reverberated /= max(float(np.sum(np.abs(impulse))), 1.0)
        value = (1.0 - wet) * value + wet * reverberated
    delay = int(round(float(recipe["english_delay_ms"]) * sample_rate / 1000.0))
    shifted = np.zeros_like(value)
    if delay > 0:
        shifted[delay:] = value[:-delay]
    elif delay < 0:
        shifted[:delay] = value[-delay:]
    else:
        shifted = value
    return np.nan_to_num(shifted, copy=False).astype(np.float32)


ENGLISH_SPEECH_GATE_DB = -48.0
MAX_ENGLISH_AMPLIFICATION_DB = 24.0


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """O(n) centred moving average via cumulative sums."""
    window = max(1, int(window))
    padded = np.pad(values.astype(np.float64), (window // 2, window - window // 2))
    sums = np.cumsum(padded)
    return (sums[window:] - sums[:-window])[: len(values)] / window


def _ducking_envelope(target: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalised 0..1 activity of the Russian voice with studio-like smoothing."""
    energy = _moving_average(
        target.astype(np.float64) ** 2, int(round(sample_rate * 0.040))
    )
    envelope = np.sqrt(np.maximum(energy, 0.0))
    scale = float(np.percentile(envelope, 95))
    if scale < 1e-7:
        return np.zeros_like(envelope)
    envelope = np.clip(envelope / scale, 0.0, 1.0)
    return _moving_average(envelope, int(round(sample_rate * 0.150)))


def load_recipe_example(row: dict) -> tuple[np.ndarray, ...]:
    recipe = row["recipe"]
    sources = recipe["sources"]
    sample_rate = int(row["sample_rate"])
    duration = float(row["duration_sec"])
    frames = int(round(sample_rate * duration))
    reference = read_interval_mono(
        sources["reference"],
        float(recipe["original_start_sec"]),
        float(recipe["original_duration_sec"]),
        sample_rate,
        frames,
    )
    target_start = float(recipe["dubbed_start_sec"])
    # Positive shift moves the Russian voice later relative to English;
    # negative shift moves it earlier.  Both model inputs are speech-only.
    target = read_interval_mono(
        sources["target_speech"],
        target_start - float(recipe["russian_shift_sec"]),
        duration,
        sample_rate,
        frames,
    )
    embedded_reference = transform_reference(reference, recipe, sample_rate)
    target_rms = max(float(np.sqrt(np.mean(target.astype(np.float64) ** 2))), 1e-5)
    reference_rms = max(
        float(np.sqrt(np.mean(embedded_reference.astype(np.float64) ** 2))), 1e-6
    )
    if float(recipe.get("english_speech_rms_db", 0.0)) < ENGLISH_SPEECH_GATE_DB:
        # Below the gate the extractor output is residual noise, not speech.
        # Blowing it up to voice level teaches the model to subtract noise;
        # keeping it silent teaches the pass-through case instead.
        embedded_reference.fill(0.0)
    else:
        # Keep EN-only examples audible even when the Russian target is silent.
        level_reference = max(target_rms, 10.0 ** (-35.0 / 20.0))
        desired_rms = level_reference * 10.0 ** (
            float(recipe["english_to_target_db"]) / 20.0
        )
        gain = float(desired_rms / reference_rms)
        gain = min(gain, 10.0 ** (MAX_ENGLISH_AMPLIFICATION_DB / 20.0))
        embedded_reference *= gain
    ducking_db = float(recipe.get("ducking_db", 0.0))
    if ducking_db > 0.05 and float(np.max(np.abs(embedded_reference))) > 1e-7:
        envelope = _ducking_envelope(target, sample_rate)
        embedded_reference = (
            embedded_reference.astype(np.float64)
            * np.power(10.0, -ducking_db * envelope / 20.0)
        ).astype(np.float32)
    mixture = target + embedded_reference
    peak = max(float(np.max(np.abs(mixture))), 0.98)
    scale = 0.98 / peak
    mixture *= scale
    embedded_reference *= scale
    target *= scale
    reference_peak = max(float(np.max(np.abs(reference))), 0.98)
    reference *= 0.98 / reference_peak
    return (
        mixture.astype(np.float32),
        reference.astype(np.float32),
        embedded_reference.astype(np.float32),
        target.astype(np.float32),
    )


class PairDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        if row.get("recipe"):
            values = synthesize_recipe_example(row)
            return tuple(torch.from_numpy(value) for value in values), row
        values = []
        sample_rate = None
        for key in ("mixture", "reference", "target_reference", "target_voice"):
            audio, rate = read_mono(row["files"][key])
            sample_rate = sample_rate or rate
            if rate != sample_rate:
                raise RuntimeError("Частоты файлов одного примера не совпадают.")
            values.append(torch.from_numpy(audio))
        length = min(value.numel() for value in values)
        return tuple(value[:length] for value in values), row


def collate_training_batch(batch):
    """Collate tensors without recursively converting large recipe seeds."""
    values = default_collate([item[0] for item in batch])
    rows = {
        "film_name": [item[1].get("film_name", "") for item in batch],
        "id": [item[1].get("id", "") for item in batch],
    }
    return values, rows


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(max(1, min(8, output_channels // 4)), output_channels),
            nn.PReLU(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(max(1, min(8, output_channels // 4)), output_channels),
            nn.PReLU(output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class ReferenceSubtractor(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, base: int = 16):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate_hint = 24000
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base * 2, stride=2)
        self.enc3 = ConvBlock(base * 2, base * 4, stride=2)
        self.bottleneck = ConvBlock(base * 4, base * 6, stride=2)
        self.dec3 = ConvBlock(base * 10, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.output = nn.Conv2d(base, 2, 1)

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    @torch.no_grad()
    def estimate_reference_delay(
        self, mixture: torch.Tensor, reference: torch.Tensor, max_delay_sec: float = 0.25
    ) -> torch.Tensor:
        """GCC-PHAT delay of the English component inside the mixture.

        A frame-wise spectral mask cannot move the reference across STFT
        frames, so misalignments beyond ~20 ms make subtraction impossible.
        The delay is therefore estimated per clip and compensated before the
        network runs.  A prominence guard keeps zero shift when no reliable
        peak exists (silent or heavily buried reference)."""
        length = mixture.shape[-1]
        fft_size = 1
        while fft_size < length * 2:
            fft_size *= 2
        mix_fft = torch.fft.rfft(mixture, n=fft_size)
        ref_fft = torch.fft.rfft(reference, n=fft_size)
        cross = mix_fft * torch.conj(ref_fft)
        cross = cross / torch.clamp(torch.abs(cross), min=1e-12)
        correlation = torch.fft.irfft(cross, n=fft_size)
        sample_rate = getattr(self, "sample_rate_hint", 24000)
        max_shift = max(1, min(int(max_delay_sec * sample_rate), length - 1))
        tail = correlation[..., -max_shift:]
        head = correlation[..., : max_shift + 1]
        windowed = torch.cat([tail, head], dim=-1)
        peak_value, peak_index = torch.max(torch.abs(windowed), dim=-1)
        delay = peak_index - max_shift
        zero_lag = torch.abs(windowed[..., max_shift])
        keep = peak_value > 1.2 * zero_lag
        return torch.where(keep, delay, torch.zeros_like(delay))

    @staticmethod
    def shift_reference(reference: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """Shift each clip by an integer sample count, filling with silence."""
        result = torch.zeros_like(reference)
        for index in range(reference.shape[0]):
            offset = int(delay[index])
            if offset > 0:
                result[index, offset:] = reference[index, : reference.shape[-1] - offset]
            elif offset < 0:
                result[index, :offset] = reference[index, -offset:]
            else:
                result[index] = reference[index]
        return result

    def forward(self, mixture: torch.Tensor, reference: torch.Tensor):
        delay = self.estimate_reference_delay(mixture, reference)
        reference = self.shift_reference(reference, delay)
        mix_spectrum = self.stft(mixture)
        ref_spectrum = self.stft(reference)
        phase_difference = torch.angle(mix_spectrum) - torch.angle(ref_spectrum)
        features = torch.stack(
            [
                torch.log1p(torch.abs(mix_spectrum) * 10.0),
                torch.log1p(torch.abs(ref_spectrum) * 10.0),
                torch.cos(phase_difference),
                torch.sin(phase_difference),
            ],
            dim=1,
        )
        e1 = self.enc1(features)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        value = self.bottleneck(e3)
        value = F.interpolate(value, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec3(torch.cat([value, e3], dim=1))
        value = F.interpolate(value, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat([value, e2], dim=1))
        value = F.interpolate(value, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat([value, e1], dim=1))
        raw_mask = self.output(value)
        # A frequency-wise least-squares estimate provides the physically
        # meaningful starting point.  The network learns the non-stationary
        # correction caused by mastering, compression and local voice overlap.
        base_gain = torch.sum(mix_spectrum * torch.conj(ref_spectrum), dim=-1, keepdim=True) / (
            torch.sum(torch.abs(ref_spectrum) ** 2, dim=-1, keepdim=True) + 1e-8
        )
        base_magnitude = torch.abs(base_gain)
        base_gain = base_gain * torch.clamp(base_magnitude, 0.0, 3.0) / (base_magnitude + 1e-8)
        # The real part may reach zero so the model can fully mute the
        # subtraction in frames where the mixture holds no English speech
        # (studio ducking, removed lines).  The previous floor of 0.25 kept
        # an audible false subtraction in those frames.
        correction = torch.complex(
            1.0 + torch.tanh(raw_mask[:, 0]),
            0.75 * torch.tanh(raw_mask[:, 1]),
        )
        mask = base_gain * correction
        raw_reference_spectrum = ref_spectrum * mask
        raw_reference = torch.istft(
            raw_reference_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=mixture.shape[-1],
            center=True,
        )
        # The neural mask learns EQ, delay and compression.  A closed-form
        # per-clip projection then prevents audible over-subtraction (the main
        # failure observed on real films after the first training pass).
        projection_gain = torch.sum(mixture * raw_reference, dim=-1, keepdim=True) / (
            torch.sum(raw_reference * raw_reference, dim=-1, keepdim=True) + 1e-8
        )
        projection_gain = torch.clamp(projection_gain, 0.0, 2.0)
        estimated_reference = raw_reference * projection_gain
        estimated_reference_spectrum = raw_reference_spectrum * projection_gain[:, :, None]
        estimated_voice_spectrum = mix_spectrum - estimated_reference_spectrum
        estimated_voice = mixture - estimated_reference
        return {
            "reference": estimated_reference,
            "voice": estimated_voice,
            "reference_spectrum": estimated_reference_spectrum,
            "voice_spectrum": estimated_voice_spectrum,
            "reference_delay_samples": delay,
        }


def complex_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.view_as_real(predicted), torch.view_as_real(target))


def calculate_loss(
    model: ReferenceSubtractor,
    output: dict,
    target_reference: torch.Tensor,
    target_voice: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    target_reference_spectrum = model.stft(target_reference)
    target_voice_spectrum = model.stft(target_voice)
    reference_spectral = complex_l1(output["reference_spectrum"], target_reference_spectrum)
    voice_spectral = complex_l1(output["voice_spectrum"], target_voice_spectrum)
    voice_waveform = F.l1_loss(output["voice"], target_voice)
    reference_waveform = F.l1_loss(output["reference"], target_reference)
    total = reference_spectral + voice_spectral + 0.5 * voice_waveform + 0.25 * reference_waveform
    return total, {
        "reference_spectral": float(reference_spectral.detach().cpu()),
        "voice_spectral": float(voice_spectral.detach().cpu()),
        "voice_waveform": float(voice_waveform.detach().cpu()),
        "reference_waveform": float(reference_waveform.detach().cpu()),
    }


def reference_leakage(voice: np.ndarray, reference: np.ndarray, sample_rate: int, max_lag_sec: float = 0.25) -> float:
    """Max |normalised cross-correlation| over ±max_lag_sec.

    Zero-lag correlation is blind to any delay between the tracks and reported
    near-zero even for scenes full of English speech; the lag search makes the
    number reflect actual reference content in the result."""
    a = voice.astype(np.float64) - float(np.mean(voice))
    b = reference.astype(np.float64) - float(np.mean(reference))
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))) + 1e-12
    size = 1
    while size < len(a) * 2:
        size *= 2
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    correlation = np.fft.irfft(spectrum, size)
    max_lag = max(1, min(int(max_lag_sec * sample_rate), len(a) - 1))
    window = np.concatenate([correlation[-max_lag:], correlation[: max_lag + 1]])
    return float(np.max(np.abs(window)) / denominator)


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    estimate = estimate.reshape(-1).double()
    target = target.reshape(-1).double()
    target_energy = torch.sum(target * target) + 1e-9
    projection = torch.sum(estimate * target) / target_energy * target
    noise = estimate - projection
    return float((10.0 * torch.log10((torch.sum(projection * projection) + 1e-9) / (torch.sum(noise * noise) + 1e-9))).cpu())


@torch.no_grad()
def evaluate(model, rows, device, maximum: int = 128) -> dict:
    """Separate answers to the questions that matter:

    - en_suppression_db: how much of the embedded English component energy
      was removed (only counted when English is actually present);
    - ru_si_sdr_db: how intact the Russian voice is after processing;
    - si_sdr_improvement_db: overall gain over doing nothing.
    """
    model.eval()
    dataset = PairDataset(rows)
    losses = []
    improvements = []
    suppressions = []
    ru_quality = []
    for index in range(min(len(rows), maximum)):
        values, _row = dataset[index]
        mixture, reference, target_reference, target_voice = [value[None].to(device) for value in values]
        output = model(mixture, reference)
        loss, _parts = calculate_loss(model, output, target_reference, target_voice)
        losses.append(float(loss.cpu()))
        after = si_sdr(output["voice"], target_voice)
        improvements.append(after - si_sdr(mixture, target_voice))
        ru_quality.append(after)
        english_energy = float(torch.sum(target_reference.double() ** 2).cpu())
        voice_energy = float(torch.sum(target_voice.double() ** 2).cpu())
        if english_energy > 1e-8 and english_energy > 1e-4 * voice_energy:
            residual = output["voice"] - target_voice
            residual_energy = float(torch.sum(residual.double() ** 2).cpu())
            suppressions.append(
                10.0 * math.log10((english_energy + 1e-12) / (residual_energy + 1e-12))
            )
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "si_sdr_improvement_db": float(np.mean(improvements)) if improvements else float("nan"),
        "en_suppression_db": float(np.mean(suppressions)) if suppressions else float("nan"),
        "ru_si_sdr_db": float(np.mean(ru_quality)) if ru_quality else float("nan"),
        "en_examples": len(suppressions),
        "examples": len(losses),
    }


@torch.no_grad()
def save_controls(model, manifest: dict, run_root: Path, epoch: int, device) -> dict:
    destination = run_root / "controls" / f"epoch_{epoch:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    validation = [row for row in manifest["examples"] if row["split"] == "valid"] or manifest["examples"]
    values, row = PairDataset(validation)[0]
    mixture, reference, target_reference, target_voice = [value[None].to(device) for value in values]
    output = model(mixture, reference)
    sample_rate = int(row["sample_rate"])
    synthetic_files = {}
    for name, value in {
        "input_mixture": mixture,
        "input_reference": reference,
        "expected_original_component": target_reference,
        "expected_voice": target_voice,
        "model_original_component": output["reference"],
        "model_voice": output["voice"],
    }.items():
        synthetic_files[name] = write_audio(
            destination / "synthetic" / f"{name}.flac",
            value[0].detach().cpu().numpy(),
            sample_rate,
        )
    real_controls = []
    for real in manifest.get("real_evaluation", []):
        real_mixture, mix_rate = read_mono(real["files"]["mixture"])
        real_reference, ref_rate = read_mono(real["files"]["reference"])
        target_rate = int(manifest.get("sample_rate", mix_rate))
        if mix_rate != target_rate:
            divisor = math.gcd(mix_rate, target_rate)
            real_mixture = signal.resample_poly(
                real_mixture, target_rate // divisor, mix_rate // divisor
            ).astype(np.float32)
        if ref_rate != target_rate:
            divisor = math.gcd(ref_rate, target_rate)
            real_reference = signal.resample_poly(
                real_reference, target_rate // divisor, ref_rate // divisor
            ).astype(np.float32)
        mix_rate = target_rate
        requested = int(round(float(real.get("duration_sec", 8.0)) * target_rate))
        length = min(len(real_mixture), len(real_reference), requested)
        mix_tensor = torch.from_numpy(real_mixture[:length])[None].to(device)
        ref_tensor = torch.from_numpy(real_reference[:length])[None].to(device)
        real_output = model(mix_tensor, ref_tensor)
        control_dir = (
            destination
            / "real"
            / real["pair_id"]
            / str(real.get("id") or "scene")
        )
        files = {
            "input_mixture": write_audio(control_dir / "input_mixture.flac", real_mixture[:length], mix_rate),
            "input_reference": write_audio(control_dir / "input_reference.flac", real_reference[:length], mix_rate),
            "model_original_component": write_audio(control_dir / "model_original_component.flac", real_output["reference"][0].cpu().numpy(), mix_rate),
            "model_voice": write_audio(control_dir / "model_voice.flac", real_output["voice"][0].cpu().numpy(), mix_rate),
        }
        before = reference_leakage(real_mixture[:length], real_reference[:length], mix_rate)
        after = reference_leakage(
            real_output["voice"][0].cpu().numpy(), real_reference[:length], mix_rate
        )
        output_voice = real_output["voice"][0].cpu().numpy()
        removed_voice = real_output["reference"][0].cpu().numpy()

        def rms_db(values: np.ndarray) -> float:
            rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
            return 20.0 * math.log10(max(rms, 1e-12))

        real_controls.append(
            {
                "film_name": real["film_name"],
                "example_id": real.get("id"),
                "dubbed_start_sec": real["dubbed_start_sec"],
                "files": files,
                "reference_correlation_before": float(np.nan_to_num(before)),
                "reference_correlation_after": float(np.nan_to_num(after)),
                "input_rms_db": rms_db(real_mixture[:length]),
                "reference_rms_db": rms_db(real_reference[:length]),
                "output_rms_db": rms_db(output_voice),
                "removed_rms_db": rms_db(removed_voice),
                "control_source_kind": real.get("source_kind"),
                "source_reference_correlation": real.get(
                    "reference_correlation"
                ),
                "source_estimated_lag_sec": real.get("estimated_lag_sec"),
            }
        )
    return {
        "synthetic": {
            "film_name": row["film_name"],
            "example_id": row["id"],
            "files": synthetic_files,
        },
        "real": real_controls,
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, global_step: int, manifest: dict, config: dict, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "format": "speech_reference_subtractor_v4",
            "dataset_id": manifest["dataset_id"],
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "configuration": config,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--mode", choices=["quick", "short", "full"], required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    config = json.loads(args.config_json)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run_root = Path(args.run_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit("log", message=f"Устройство обучения: {device}{' · ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    model = ReferenceSubtractor(
        int(config["n_fft"]), int(config["hop_length"]), int(config["base_channels"])
    ).to(device)
    model.sample_rate_hint = int(manifest.get("sample_rate", 24000))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-5)
    start_epoch = 0
    global_step = 0
    history = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("format") not in {
            "speech_reference_subtractor_v3",
            "speech_reference_subtractor_v4",
        }:
            raise RuntimeError("Выбрано сохранение несовместимой модели.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        history = list(state.get("history", []))
    train_rows = [row for row in manifest["examples"] if row["split"] == "train"]
    valid_rows = [row for row in manifest["examples"] if row["split"] == "valid"]
    test_rows = [row for row in manifest["examples"] if row["split"] == "test"]
    if not train_rows or not valid_rows or not test_rows:
        raise RuntimeError("Датасет должен содержать отдельные обучение, проверку и тест.")
    worker_count = max(0, int(config.get("data_loader_workers", 0)))
    loader = DataLoader(
        PairDataset(train_rows),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=worker_count,
        collate_fn=collate_training_batch,
        persistent_workers=worker_count > 0,
        prefetch_factor=2 if worker_count > 0 else None,
    )
    maximum_batches = int(config["quick_check_batches"]) if args.mode == "quick" else len(loader)
    total_steps = max(1, args.epochs * min(len(loader), maximum_batches))
    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch + 1
        model.train()
        losses = []
        parts_sum = {"reference_spectral": 0.0, "voice_spectral": 0.0, "voice_waveform": 0.0, "reference_waveform": 0.0}
        for batch_index, batch in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            values, rows = batch
            mixture, reference, target_reference, target_voice = [value.to(device) for value in values]
            optimizer.zero_grad(set_to_none=True)
            output = model(mixture, reference)
            loss, parts = calculate_loss(model, output, target_reference, target_voice)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_sum[key] += value
            progress = (local_epoch * min(len(loader), maximum_batches) + batch_index + 1) / total_steps * 100.0
            emit(
                "progress",
                stage="Обучение парному вычитанию",
                substage=f"Проход {epoch}, пакет {batch_index + 1} из {min(len(loader), maximum_batches)}",
                progress=progress,
                current_file=str(rows["film_name"][0] if isinstance(rows, dict) else ""),
                loss=float(loss.detach().cpu()),
            )
        validation = evaluate(model, valid_rows, device)
        test = evaluate(model, test_rows, device)
        controls = save_controls(model, manifest, run_root, epoch, device)
        count = max(1, len(losses))
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "test_loss": test["loss"],
            "validation_si_sdr_improvement_db": validation["si_sdr_improvement_db"],
            "test_si_sdr_improvement_db": test["si_sdr_improvement_db"],
            "validation_en_suppression_db": validation["en_suppression_db"],
            "test_en_suppression_db": test["en_suppression_db"],
            "validation_ru_si_sdr_db": validation["ru_si_sdr_db"],
            "test_ru_si_sdr_db": test["ru_si_sdr_db"],
            "loss_parts": {key: value / count for key, value in parts_sum.items()},
            "control_examples": controls,
            "dataset_id": manifest["dataset_id"],
            "algorithm_version": "speech_reference_subtractor_v4",
        }
        history.append(row)
        if args.mode != "quick":
            checkpoint = checkpoint_root / f"{run_root.name}_epoch{epoch}.pt"
            row["checkpoint"] = str(checkpoint.resolve())
            save_checkpoint(checkpoint, model, optimizer, epoch, global_step, manifest, config, history)
            save_checkpoint(checkpoint_root / "speech_reference_subtractor_last.pt", model, optimizer, epoch, global_step, manifest, config, history)
        best_epoch = min(history, key=lambda item: float(item.get("validation_loss", float("inf"))))
        if args.mode != "quick" and best_epoch is row:
            save_checkpoint(
                checkpoint_root / "speech_reference_subtractor_best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                manifest,
                config,
                history,
            )
        latest_epoch = history[-1]
        if int(latest_epoch["epoch"]) != int(best_epoch["epoch"]):
            recommendation = (
                f"Остановить обучение: лучший проход {best_epoch['epoch']}, "
                f"после него ошибка проверки выросла. Используйте его сохранение модели."
            )
        else:
            recommendation = (
                f"Лучший текущий результат — проход {best_epoch['epoch']}. "
                "Перед продолжением сравните реальные контрольные сцены на слух."
            )
        report = {
            "schema_version": 3,
            "run_id": run_root.name,
            "dataset_id": manifest["dataset_id"],
            "algorithm_version": "speech_reference_subtractor_v4",
            "mode": args.mode,
            "dry_run": args.mode == "quick",
            "device": str(device),
            "history": history,
            "best_epoch": best_epoch,
            "recommended_checkpoint": best_epoch.get("checkpoint"),
            "recommendation": recommendation,
            "note": "Синтетическая проверка имеет точные цели; реальные сцены оцениваются на слух и по утечке оригинального референса.",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(run_root / "history.json", report)
    emit("complete", history=str((run_root / "history.json").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
