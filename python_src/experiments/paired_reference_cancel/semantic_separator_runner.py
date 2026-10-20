#!/usr/bin/env python
"""Train a mastering-invariant EN-reference conditioned RU speech separator.

Unlike the legacy subtractor this model never synthesises an English waveform
from the reference and never subtracts it sample-for-sample.  It predicts a
complex mask for the Russian source directly from the EN+RU mixture.  The
reference contributes magnitude/content hints only, so different phase,
compression and EQ between releases are valid inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
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

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.recipe_audio import (
    ENGLISH_SPEECH_GATE_DB,
    load_recipe_example,
    place_short_utterances,
)


FORMAT = "semantic_ru_separator_v1"
DEFAULT_TRAINING_CLIP_SEC = 4.0


def _training_clip_sec(row: dict) -> float:
    """Allow new manifests to match the 6 s production window without breaking old 4 s sets."""
    value = float(row.get("clip_duration_sec", DEFAULT_TRAINING_CLIP_SEC))
    if value <= 0.0 or value > 60.0:
        raise ValueError(f"{row.get('id')}: invalid clip_duration_sec={value!r}")
    return value


def emit(kind: str, **values) -> None:
    print(json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_audio(path: Path, data: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(data, -1.0, 1.0), sample_rate, format="FLAC", subtype="PCM_24")
    return str(path.resolve())


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))


def _spectral_tilt(value: np.ndarray, sample_rate: int, amount_db: float) -> np.ndarray:
    spectrum = np.fft.rfft(value.astype(np.float64))
    frequencies = np.fft.rfftfreq(len(value), 1.0 / sample_rate)
    octaves = np.log2(np.maximum(frequencies, 80.0) / 1000.0)
    curve = np.clip(octaves * amount_db / 4.0, -abs(amount_db), abs(amount_db))
    return np.fft.irfft(spectrum * np.power(10.0, curve / 20.0), n=len(value)).astype(np.float32)


def _shift(value: np.ndarray, samples: int) -> np.ndarray:
    output = np.zeros_like(value)
    if samples > 0:
        output[samples:] = value[:-samples]
    elif samples < 0:
        output[:samples] = value[-samples:]
    else:
        output[:] = value
    return output


def _independent_mastering(
    value: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    maximum_shift_ms: float,
) -> np.ndarray:
    """Make a release that shares content but not waveform phase."""
    output = _spectral_tilt(value, sample_rate, float(rng.uniform(-12.0, 12.0)))
    # Random short room/mastering impulse.  Multiple taps intentionally make
    # direct phase cancellation impossible while preserving spoken content.
    impulse_length = max(32, int(round(sample_rate * float(rng.uniform(0.025, 0.16)))))
    impulse = np.zeros(impulse_length, dtype=np.float64)
    impulse[0] = 1.0
    for _ in range(int(rng.integers(2, 7))):
        position = int(rng.integers(1, impulse_length))
        impulse[position] += float(rng.uniform(-0.45, 0.45)) * math.exp(-3.0 * position / impulse_length)
    output = signal.fftconvolve(output, impulse, mode="full")[: len(output)].astype(np.float32)
    drive = float(rng.uniform(1.0, 3.5))
    if drive > 1.02:
        output = (np.tanh(output * drive) / np.tanh(drive)).astype(np.float32)
    shift = int(round(float(rng.uniform(-maximum_shift_ms, maximum_shift_ms)) * sample_rate / 1000.0))
    output = _shift(output, shift)
    return np.nan_to_num(output, copy=False).astype(np.float32)


def _quality_eq(value: np.ndarray, sample_rate: int, gains_db: list[float]) -> np.ndarray:
    gains = np.asarray(gains_db, dtype=np.float64)
    if len(gains) < 2 or float(np.max(np.abs(gains))) < 1e-6:
        return value.astype(np.float32, copy=True)
    spectrum = np.fft.rfft(value.astype(np.float64))
    frequencies = np.fft.rfftfreq(len(value), 1.0 / sample_rate)
    # The anchor grid is pinned to the 24 kHz training band (0.48 * 24000) so
    # the same eq_db profile realises the same curve when the extractor cache
    # synthesises at 48 kHz.  At the 24 kHz training rate this is identical to
    # the previous sample_rate-derived grid.
    anchors = np.geomspace(80.0, max(81.0, min(sample_rate * 0.48, 11520.0)), len(gains))
    curve_db = np.interp(
        np.maximum(frequencies, anchors[0]),
        anchors,
        gains,
        left=float(gains[0]),
        right=float(gains[-1]),
    )
    return np.fft.irfft(
        spectrum * np.power(10.0, curve_db / 20.0), n=len(value)
    ).astype(np.float32)


def _dynamic_compression(
    value: np.ndarray, sample_rate: int, settings: dict
) -> np.ndarray:
    """Deterministic broadcast-style RMS compressor.

    Unlike the static tanh drive in ``_independent_mastering`` this changes the
    *dynamics* of the release: loud syllables are pulled towards the threshold
    while pauses keep their level, which is what real loudness processing does.
    """
    threshold_db = float(settings.get("threshold_db", 0.0))
    ratio = max(1.0, float(settings.get("ratio", 1.0)))
    if ratio <= 1.001:
        return value.astype(np.float32, copy=True)
    window = max(8, int(round(sample_rate * float(settings.get("window_ms", 25.0)) / 1000.0)))
    smooth = max(8, int(round(sample_rate * float(settings.get("smooth_ms", 120.0)) / 1000.0)))
    energy = _runner_moving_average(value.astype(np.float64) ** 2, window)
    envelope = np.sqrt(np.maximum(energy, 0.0))
    reference_rms = max(_rms(value), 1e-7)
    threshold = reference_rms * 10.0 ** (threshold_db / 20.0)
    envelope_db = 20.0 * np.log10(np.maximum(envelope, threshold * 1e-3))
    threshold_db_abs = 20.0 * math.log10(threshold)
    over_db = np.maximum(envelope_db - threshold_db_abs, 0.0)
    gain_db = -over_db * (1.0 - 1.0 / ratio)
    gain_db = _runner_moving_average(gain_db, smooth)
    return (value.astype(np.float64) * np.power(10.0, gain_db / 20.0)).astype(np.float32)


def _runner_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    padded = np.pad(values.astype(np.float64), (window // 2, window - window // 2))
    sums = np.cumsum(padded)
    return (sums[window:] - sums[:-window])[: len(values)] / window


def _quality_channel(
    value: np.ndarray,
    sample_rate: int,
    profile: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one deterministic release/channel degradation while preserving RMS.

    Preserving RMS is deliberate: existing recipes already cover loudness, and
    this augmentation must isolate sample-rate/bandwidth/codec/mastering mismatch.
    """
    original_rms = _rms(value)
    if original_rms <= 1e-7:
        return value.astype(np.float32, copy=True)
    output = value.astype(np.float32, copy=True)
    roundtrip_rate = int(profile.get("resample_hz") or sample_rate)
    roundtrip_rate = max(4000, min(sample_rate, roundtrip_rate))
    if roundtrip_rate != sample_rate:
        divisor = math.gcd(sample_rate, roundtrip_rate)
        output = signal.resample_poly(
            output, roundtrip_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
        output = signal.resample_poly(
            output, sample_rate // divisor, roundtrip_rate // divisor
        ).astype(np.float32)
        if len(output) != len(value):
            output = signal.resample(output, len(value)).astype(np.float32)
    highpass = float(profile.get("highpass_hz") or 0.0)
    lowpass = float(profile.get("lowpass_hz") or 0.0)
    nyquist = sample_rate / 2.0
    if 15.0 < highpass < nyquist * 0.9:
        output = signal.sosfiltfilt(
            signal.butter(4, highpass / nyquist, btype="highpass", output="sos"),
            output,
        ).astype(np.float32)
    if 100.0 < lowpass < nyquist * 0.98:
        output = signal.sosfiltfilt(
            signal.butter(6, lowpass / nyquist, btype="lowpass", output="sos"),
            output,
        ).astype(np.float32)
    output = _quality_eq(output, sample_rate, list(profile.get("eq_db") or []))
    drive = float(profile.get("compand_drive") or 1.0)
    if drive > 1.001:
        output = (np.tanh(output * drive) / np.tanh(drive)).astype(np.float32)
    dynamic = profile.get("dynamic_compression") or {}
    if dynamic:
        output = _dynamic_compression(output, sample_rate, dynamic)
    bit_depth = int(profile.get("bit_depth") or 24)
    if bit_depth < 24:
        peak = max(float(np.max(np.abs(output))), 1e-6)
        levels = float((1 << max(2, bit_depth - 1)) - 1)
        output = (np.round(output / peak * levels) / levels * peak).astype(np.float32)
    snr_db = float(profile.get("noise_snr_db") or 120.0)
    if snr_db < 100.0:
        noise_rms = max(_rms(output), 1e-7) / (10.0 ** (snr_db / 20.0))
        output += rng.normal(0.0, noise_rms, size=len(output)).astype(np.float32)
    output *= original_rms / max(_rms(output), 1e-7)
    return np.nan_to_num(output, copy=False).astype(np.float32)


def _row_target_policy(row: dict) -> str | None:
    factors = row.get("factors") or {}
    if factors.get("target_policy"):
        return str(factors["target_policy"])
    recipe = row.get("recipe") or {}
    if recipe.get("target_policy"):
        return str(recipe["target_policy"])
    return None


def enforce_training_manifest_guards(manifest: dict, train_rows: list[dict]) -> None:
    """Refuse gradients on manifests or rows that are not allowed to train.

    Two independent locks make the frozen safety holdout technically unusable
    for training:

    * a manifest with ``training_allowed: false`` (the negative safety holdout)
      is rejected outright, before any DataLoader is built;
    * a manifest with ``training_invariant.target_policy`` (the positive-only
      overlay) rejects every train row whose target policy is not in the
      declared allow-list, so a negative example can never slip into the
      gradient path through manifest editing or concatenation.

    Validation/test rows are exempt: the invariant protects gradients only.
    """
    if manifest.get("training_allowed") is False:
        raise RuntimeError(
            "Манифест помечен training_allowed=false (замороженный safety "
            "holdout) — обучение на нём запрещено."
        )
    allowed = (manifest.get("training_invariant") or {}).get("target_policy")
    if not allowed:
        return
    allowed_set = {str(value) for value in allowed}
    for row in train_rows:
        policy = _row_target_policy(row)
        if policy not in allowed_set:
            raise RuntimeError(
                f"{row.get('id')}: target_policy={policy!r} нарушает "
                f"training_invariant {sorted(allowed_set)} — train loader "
                "получил недопустимый пример."
            )


def load_extractor_domain_example(row: dict) -> tuple[np.ndarray, ...]:
    """Read one pre-built extractor-domain example from the disk cache.

    The extractor is intentionally never invoked here: extractor-domain audio
    is materialised offline by ``extractor_cache_builder.py``.  The removed
    component is recomputed as ``mixture - target_russian`` so the additive
    decomposition ``mixture == target_russian + target_removed`` holds exactly
    by construction — the extractor is nonlinear, and the clean pre-extraction
    English is *not* a valid removal target in this domain.
    """
    cache = row.get("extractor_cache") or {}
    files = cache.get("files") or {}
    frames = int(round(_training_clip_sec(row) * int(row["sample_rate"])))
    values: dict[str, np.ndarray] = {}
    for name in ("mixture", "reference", "target_russian"):
        path = files.get(name)
        if not path:
            raise RuntimeError(
                f"{row.get('id')}: extractor-домен без пути кеша {name}; "
                "манифест собран без finalize-этапа?"
            )
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        if int(rate) != int(row["sample_rate"]):
            raise RuntimeError(f"{row.get('id')}: {name} кеширован на {rate} Гц.")
        mono = np.mean(data, axis=1, dtype=np.float64).astype(np.float32)
        if len(mono) != frames:
            raise RuntimeError(f"{row.get('id')}: {name} длиной {len(mono)} != {frames}.")
        values[name] = np.nan_to_num(mono, copy=False)
    mixture = values["mixture"]
    target_russian = values["target_russian"]
    target_removed = (mixture - target_russian).astype(np.float32)
    return mixture, values["reference"], target_removed, target_russian


def _event_read_mono(path: str | Path, sample_rate: int, frames: int) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1, dtype=np.float64).astype(np.float32)
    if int(rate) != int(sample_rate):
        divisor = math.gcd(int(rate), int(sample_rate))
        mono = signal.resample_poly(
            mono, int(sample_rate) // divisor, int(rate) // divisor
        ).astype(np.float32)
    if len(mono) < frames:
        mono = np.pad(mono, (0, frames - len(mono)), mode="constant")
    elif len(mono) > frames:
        mono = mono[:frames]
    return np.nan_to_num(mono, copy=False).astype(np.float32)


def _event_active_start(
    value: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> int:
    if length >= len(value):
        return 0
    step = max(1, int(round(len(value) / 80)))
    candidates = list(range(0, len(value) - length + 1, step))
    if not candidates:
        return 0
    scored = []
    for start in candidates:
        chunk = value[start : start + length]
        scored.append((float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2) + 1e-12)), start))
    scored.sort(reverse=True)
    # Sample from the best few starts instead of always using the absolute peak:
    # this keeps words/short phrases varied while still avoiding pure silence.
    top = scored[: max(1, min(8, len(scored)))]
    return int(top[int(rng.integers(0, len(top)))][1])


def _event_fade(length: int, sample_rate: int) -> np.ndarray:
    fade_len = min(length // 3, max(8, int(round(sample_rate * 0.018))))
    mask = np.ones(length, dtype=np.float32)
    if fade_len > 1:
        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        mask[:fade_len] *= ramp
        mask[-fade_len:] *= ramp[::-1]
    return mask


def _event_slice(
    source: np.ndarray,
    source_start: int,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    end = source_start + length
    if source_start < 0:
        source = np.pad(source, (-source_start, 0), mode="constant")
        end -= source_start
        source_start = 0
    if end > len(source):
        source = np.pad(source, (0, end - len(source)), mode="constant")
    return (source[source_start:end].astype(np.float32) * _event_fade(length, sample_rate)).astype(
        np.float32
    )


def _event_add(target: np.ndarray, segment: np.ndarray, start: int) -> None:
    left = max(0, int(start))
    right = min(len(target), int(start) + len(segment))
    if right <= left:
        return
    source_left = left - int(start)
    target[left:right] += segment[source_left : source_left + (right - left)]


def load_event_semantic_example(row: dict) -> tuple[np.ndarray, ...]:
    """Build a fixed-window event example from existing extractor-domain stems.

    The important distinction from the rejected anti-mute attempt: the target is
    not derived from "the output got quiet".  Every example has an explicit
    short event structure:
      * matched EN event -> remove it;
      * RU event without a matching reference -> preserve it;
      * EN/RU overlap -> remove only the matched EN component.
    """
    sample_rate = int(row["sample_rate"])
    frames = int(round(_training_clip_sec(row) * sample_rate))
    sources = row.get("event_sources") or {}
    files = sources.get("files") or {}
    if not files:
        raise RuntimeError(f"{row.get('id')}: event_sources.files отсутствует.")
    source_mixture = _event_read_mono(files["mixture"], sample_rate, frames)
    source_reference = _event_read_mono(files["reference"], sample_rate, frames)
    source_russian = _event_read_mono(files["target_russian"], sample_rate, frames)
    source_english = (source_mixture - source_russian).astype(np.float32)
    transform = row.get("event_transform") or {}
    seed = int(transform.get("seed", 0))
    rng = np.random.default_rng(seed)
    en_duration = max(0.10, float(transform.get("english_duration_sec", 0.55)))
    ru_duration = max(0.10, float(transform.get("russian_duration_sec", 0.55)))
    en_len = max(1, min(frames, int(round(en_duration * sample_rate))))
    ru_len = max(1, min(frames, int(round(ru_duration * sample_rate))))
    en_start = int(round(float(transform.get("english_start_sec", 0.7)) * sample_rate))
    ru_start = int(round(float(transform.get("russian_start_sec", 1.7)) * sample_rate))
    en_source_start = _event_active_start(source_english, en_len, rng)
    ru_source_start = _event_active_start(source_russian, ru_len, rng)
    en_segment = _event_slice(source_english, en_source_start, en_len, sample_rate)
    ref_segment = _event_slice(source_reference, en_source_start, en_len, sample_rate)
    ru_segment = _event_slice(source_russian, ru_source_start, ru_len, sample_rate)
    en_segment *= np.float32(10.0 ** (float(transform.get("english_gain_db", 0.0)) / 20.0))
    ru_segment *= np.float32(10.0 ** (float(transform.get("russian_gain_db", 0.0)) / 20.0))
    mixture = np.zeros(frames, dtype=np.float32)
    target_russian = np.zeros(frames, dtype=np.float32)
    target_removed = np.zeros(frames, dtype=np.float32)
    reference = np.zeros(frames, dtype=np.float32)
    include_english = bool(transform.get("include_english", True))
    include_russian = bool(transform.get("include_russian", True))
    target_policy = str(transform.get("target_policy") or "remove_matched_english")
    if include_english:
        _event_add(mixture, en_segment, en_start)
        if target_policy == "preserve_all":
            _event_add(target_russian, en_segment, en_start)
        else:
            _event_add(target_removed, en_segment, en_start)
    if include_russian:
        _event_add(mixture, ru_segment, ru_start)
        _event_add(target_russian, ru_segment, ru_start)
    reference_mode = str(transform.get("reference_mode") or "matched")
    if reference_mode in {"matched", "weak_matched"} and include_english:
        ref_start = en_start + int(round(float(transform.get("reference_shift_sec", 0.0)) * sample_rate))
        ref_gain = 10.0 ** (float(transform.get("reference_gain_db", 0.0)) / 20.0)
        _event_add(reference, ref_segment * ref_gain, ref_start)
    elif reference_mode == "wrong":
        wrong = files.get("wrong_reference") or files.get("reference")
        wrong_reference = _event_read_mono(wrong, sample_rate, frames)
        wrong_len = max(en_len, ru_len)
        wrong_source_start = _event_active_start(wrong_reference, wrong_len, rng)
        wrong_segment = _event_slice(wrong_reference, wrong_source_start, wrong_len, sample_rate)
        wrong_start = int(round(float(transform.get("wrong_reference_start_sec", 1.1)) * sample_rate))
        _event_add(reference, wrong_segment, wrong_start)
    elif reference_mode == "zero":
        pass
    else:
        raise ValueError(f"{row.get('id')}: unknown event reference_mode={reference_mode!r}")
    noise_db = transform.get("speech_stem_noise_db")
    if noise_db is not None:
        noise_rms = max(_rms(mixture), 1e-7) / (10.0 ** (float(noise_db) / 20.0))
        stem_noise = rng.normal(0.0, noise_rms, size=frames).astype(np.float32)
        mixture += stem_noise
        target_russian += stem_noise
    peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(target_russian))), 1e-7)
    rms = max(_rms(mixture), 1e-7)
    scale = min(0.98 / peak, (10.0 ** (-18.0 / 20.0)) / rms)
    mixture = (mixture * scale).astype(np.float32)
    target_russian = (target_russian * scale).astype(np.float32)
    target_removed = (target_removed * scale).astype(np.float32)
    reference /= max(float(np.max(np.abs(reference))) / 0.95, 1.0)
    return mixture, reference.astype(np.float32), target_removed, target_russian


def synthesize_semantic_example(row: dict) -> tuple[np.ndarray, ...]:
    if row.get("event_transform"):
        return load_event_semantic_example(row)
    if str(row.get("input_domain") or "direct") == "extractor":
        return load_extractor_domain_example(row)
    mixture, reference, embedded_english, target_russian = load_recipe_example(row)
    recipe = row.get("recipe", {})
    seed = int(recipe.get("seed", 0)) ^ 0x5E6A17C
    rng = np.random.default_rng(seed)
    sample_rate = int(row["sample_rate"])
    if recipe.get("short_utterance"):
        reference, embedded_english, target_russian = place_short_utterances(
            reference,
            embedded_english,
            target_russian,
            sample_rate,
            recipe["short_utterance"],
        )

    # The embedded English and the external reference receive independent
    # mastering.  Their phonetic content still matches, their phase does not.
    english_level = _rms(embedded_english)
    if english_level > 1e-6:
        embedded_english = _independent_mastering(
            embedded_english, sample_rate, rng, maximum_shift_ms=100.0
        )
        embedded_english *= english_level / max(_rms(embedded_english), 1e-7)

    reference = _independent_mastering(reference, sample_rate, rng, maximum_shift_ms=0.0)
    channel_mismatch = recipe.get("channel_mismatch") or {}
    if channel_mismatch:
        reference = _quality_channel(
            reference,
            sample_rate,
            channel_mismatch.get("reference") or {},
            rng,
        )
        embedded_english = _quality_channel(
            embedded_english,
            sample_rate,
            channel_mismatch.get("embedded_english") or {},
            rng,
        )
    reference_condition = str(recipe.get("reference_condition") or "matched")
    if reference_condition == "zero":
        # Real target films can contain an English underlay that does not match
        # the supplied original closely enough.  These examples force the model
        # to behave as a language separator instead of blindly passing the input
        # through when the reference is useless.
        reference *= 0.0
    elif reference_condition == "weak":
        reference *= 10.0 ** (float(rng.uniform(-24.0, -9.0)) / 20.0)
    elif reference_condition == "bad_timing":
        shift_ms = float(rng.choice([-1.0, 1.0]) * rng.uniform(450.0, 1800.0))
        reference = _shift(reference, int(round(shift_ms * sample_rate / 1000.0)))
    elif reference_condition == "bad_timing_weak":
        shift_ms = float(rng.choice([-1.0, 1.0]) * rng.uniform(450.0, 1800.0))
        reference = _shift(reference, int(round(shift_ms * sample_rate / 1000.0)))
        reference *= 10.0 ** (float(rng.uniform(-24.0, -9.0)) / 20.0)
    elif reference_condition != "matched":
        raise ValueError(f"Неизвестное состояние reference_condition={reference_condition!r}")
    reference /= max(float(np.max(np.abs(reference))) / 0.95, 1.0)

    # Vary the Russian target as a separate studio chain.  This prevents the
    # model from identifying only the narrators present in the three films.
    target_level = _rms(target_russian)
    if target_level > 1e-6:
        target_russian = _spectral_tilt(
            target_russian, sample_rate, float(rng.uniform(-8.0, 8.0))
        )
        drive = float(rng.uniform(1.0, 2.2))
        target_russian = (np.tanh(target_russian * drive) / np.tanh(drive)).astype(np.float32)
        target_russian *= target_level / max(_rms(target_russian), 1e-7)

    # Rare default reference drop-out prevents a hard dependency for legacy
    # manifests.  Robust manifests use explicit reference_condition above.
    # Channel-mismatch examples are exempt: a silent reference would make the
    # per-category quality metrics meaningless, and the zero-reference regime
    # is already covered by clean_guard rows.
    if (
        reference_condition == "matched"
        and not channel_mismatch
        and float(rng.random()) < 0.02
    ):
        reference *= 0.0
    crop_frames = int(round(_training_clip_sec(row) * sample_rate))
    if len(target_russian) > crop_frames:
        maximum_start = len(target_russian) - crop_frames
        starts = np.linspace(0, maximum_start, num=9, dtype=np.int64)
        # Always train on the most informative four seconds of the recipe.
        # Random cropping frequently selected pauses and taught pass-through.
        start = max(
            (int(value) for value in starts),
            key=lambda value: min(
                _rms(target_russian[value : value + crop_frames]),
                _rms(embedded_english[value : value + crop_frames]),
            ),
        )
        stop = start + crop_frames
        reference = reference[start:stop]
        embedded_english = embedded_english[start:stop]
        target_russian = target_russian[start:stop]
    # Very quiet examples are mostly trivial. Enforce the
    # useful -20..+6 dB range on the actual selected training crop.
    english_rms = _rms(embedded_english)
    russian_rms = _rms(target_russian)
    if english_rms > 1e-6 and russian_rms > 1e-6:
        ratio_db = 20.0 * math.log10(english_rms / russian_rms)
        if ratio_db < -20.0:
            embedded_english *= 10.0 ** ((-20.0 - ratio_db) / 20.0)
    mixture = target_russian + embedded_english
    peak = max(float(np.max(np.abs(mixture))), 1e-7)
    scale = min(0.98 / peak, (10.0 ** (-18.0 / 20.0)) / max(_rms(mixture), 1e-7))
    return (
        (mixture * scale).astype(np.float32),
        reference.astype(np.float32),
        (embedded_english * scale).astype(np.float32),
        (target_russian * scale).astype(np.float32),
    )


class SemanticDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        values = synthesize_semantic_example(row)
        return tuple(torch.from_numpy(value) for value in values), {
            "id": row.get("id", ""),
            "film_name": row.get("film_name", ""),
        }


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        groups = max(1, min(8, output_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.PReLU(output_channels),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels),
            nn.PReLU(output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TemporalResidualBlock(nn.Module):
    """Identity-initialised time-dilated residual block for the bottleneck.

    The closing convolution starts at zero, so a freshly added block is an
    exact identity: an expanded checkpoint behaves bit-for-bit like the source
    model until training moves the new weights.  Dilation applies to the time
    axis only — the goal is a larger temporal receptive field (jitter/drift
    tolerance), not more frequency mixing.
    """

    def __init__(self, channels: int, time_dilation: int):
        super().__init__()
        groups = max(1, min(8, channels // 4))
        dilation = max(1, int(time_dilation))
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.PReLU(channels),
            nn.Conv2d(
                channels, channels, 3,
                padding=(1, dilation), dilation=(1, dilation),
            ),
            nn.GroupNorm(groups, channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


def _shift_time(value: torch.Tensor, lag: int) -> torch.Tensor:
    """Shift [..., T] by ``lag`` frames (positive = later), zero padded."""
    if lag == 0:
        return value
    if lag > 0:
        return F.pad(value, (lag, 0))[..., : value.shape[-1]]
    return F.pad(value, (0, -lag))[..., -lag:]


class SemanticRuSeparator(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        base: int = 16,
        *,
        lag_bank_max_frames: int = 0,
        lag_bank_step_frames: int = 2,
        bottleneck_time_dilations: tuple[int, ...] | list[int] = (),
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        # Explicit reference lag bank: per-frame cosine similarity between the
        # mixture and the reference shifted by each lag, plus a soft-aligned
        # reference and the aligned difference.  This gives the network a
        # direct time-offset matching signal instead of relying on strictly
        # frame-aligned difference/similarity channels — the exact mechanism
        # that breaks on independently mastered releases with jitter/drift.
        self.lag_bank_max_frames = max(0, int(lag_bank_max_frames))
        self.lag_bank_step_frames = max(1, int(lag_bank_step_frames))
        if self.lag_bank_max_frames > 0:
            self.lag_offsets = tuple(
                range(
                    -self.lag_bank_max_frames,
                    self.lag_bank_max_frames + 1,
                    self.lag_bank_step_frames,
                )
            )
        else:
            self.lag_offsets = ()
        extra_channels = (len(self.lag_offsets) + 2) if self.lag_offsets else 0
        # Magnitude-only local and long-term reference cues.  Phase relation
        # never enters the network, but the explicit difference/similarity
        # channels make matching the same phonetic content much easier.
        self.enc1 = ConvBlock(6 + extra_channels, base)
        self.enc2 = ConvBlock(base, base * 2, stride=2)
        self.enc3 = ConvBlock(base * 2, base * 4, stride=2)
        self.enc4 = ConvBlock(base * 4, base * 6, stride=2)
        self.bottleneck = ConvBlock(base * 6, base * 8, stride=2)
        self.bottleneck_time_dilations = tuple(
            int(d) for d in (bottleneck_time_dilations or ())
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalResidualBlock(base * 8, dilation)
            for dilation in self.bottleneck_time_dilations
        )
        self.dec4 = ConvBlock(base * 14, base * 6)
        self.dec3 = ConvBlock(base * 10, base * 4)
        self.dec2 = ConvBlock(base * 6, base * 2)
        self.dec1 = ConvBlock(base * 3, base)
        self.output = nn.Conv2d(base, 2, 1)
        with torch.no_grad():
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def _lag_bank_features(
        self, mix_log: torch.Tensor, ref_log: torch.Tensor
    ) -> list[torch.Tensor]:
        """Extra input channels: per-lag similarity + soft-aligned reference."""
        eps = 1e-6
        mix_norm = mix_log / (mix_log.norm(dim=-2, keepdim=True) + eps)
        shifted = torch.stack(
            [_shift_time(ref_log, lag) for lag in self.lag_offsets], dim=1
        )  # [B, L, F, T]
        shifted_norm = shifted / (shifted.norm(dim=-2, keepdim=True) + eps)
        similarity = (mix_norm.unsqueeze(1) * shifted_norm).sum(dim=-2)  # [B, L, T]
        weights = torch.softmax(similarity * 8.0, dim=1)
        aligned = (shifted * weights.unsqueeze(2)).sum(dim=1)  # [B, F, T]
        aligned_difference = mix_log - aligned
        frequency_bins = mix_log.shape[-2]
        similarity_maps = similarity.unsqueeze(2).expand(
            -1, -1, frequency_bins, -1
        )  # [B, L, F, T]
        return [aligned, aligned_difference, *similarity_maps.unbind(dim=1)]

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    def forward(self, mixture: torch.Tensor, reference: torch.Tensor) -> dict:
        mixture_spectrum = self.stft(mixture)
        reference_spectrum = self.stft(reference)
        mix_log = torch.log1p(torch.abs(mixture_spectrum) * 12.0)
        ref_log = torch.log1p(torch.abs(reference_spectrum) * 12.0)
        ref_profile = ref_log.mean(dim=-1, keepdim=True).expand_as(ref_log)
        flux = F.pad(torch.abs(mix_log[..., 1:] - mix_log[..., :-1]), (1, 0))
        difference = mix_log - ref_log
        similarity = (mix_log * ref_log) / (
            torch.sqrt(torch.mean(mix_log * mix_log, dim=-2, keepdim=True) + 1e-6)
            * torch.sqrt(torch.mean(ref_log * ref_log, dim=-2, keepdim=True) + 1e-6)
        )
        channels = [mix_log, ref_log, ref_profile, flux, difference, similarity]
        if self.lag_offsets:
            channels.extend(self._lag_bank_features(mix_log, ref_log))
        features = torch.stack(channels, dim=1)
        e1 = self.enc1(features)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        value = self.bottleneck(e4)
        for block in self.temporal_blocks:
            value = block(value)
        value = F.interpolate(value, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec4(torch.cat([value, e4], dim=1))
        value = F.interpolate(value, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec3(torch.cat([value, e3], dim=1))
        value = F.interpolate(value, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat([value, e2], dim=1))
        value = F.interpolate(value, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat([value, e1], dim=1))
        raw = self.output(value)
        # An ideal complex ratio mask is not bounded to 0..1: destructive
        # interference can require gain above one, negative real values and a
        # substantial phase correction.  The old narrow mask capped achievable
        # suppression at roughly 4--6 dB even when deliberately overfitting.
        mask = torch.complex(
            1.0 + 2.0 * torch.tanh(raw[:, 0]),
            2.0 * torch.tanh(raw[:, 1]),
        )
        russian_spectrum = mixture_spectrum * mask
        russian = torch.istft(
            russian_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            length=mixture.shape[-1],
            center=True,
        )
        english = mixture - russian
        return {
            "voice": russian,
            "reference": english,
            "voice_spectrum": russian_spectrum,
            "reference_spectrum": mixture_spectrum - russian_spectrum,
        }


def semantic_model_configuration(config: dict) -> dict:
    """Model-defining keys of a training config / checkpoint configuration."""
    return {
        "n_fft": int(config.get("n_fft", 512)),
        "hop_length": int(config.get("hop_length", 128)),
        "base_channels": int(config.get("base_channels", 16)),
        "lag_bank_max_frames": int(config.get("lag_bank_max_frames", 0)),
        "lag_bank_step_frames": int(config.get("lag_bank_step_frames", 2)),
        "bottleneck_time_dilations": [
            int(d) for d in (config.get("bottleneck_time_dilations") or ())
        ],
    }


def build_semantic_model(config: dict) -> "SemanticRuSeparator":
    """Single construction point so every loader honours capacity keys.

    Configs and checkpoints written before the capacity extension carry none
    of the new keys, so they build the exact historical architecture.
    """
    resolved = semantic_model_configuration(config)
    return SemanticRuSeparator(
        resolved["n_fft"],
        resolved["hop_length"],
        resolved["base_channels"],
        lag_bank_max_frames=resolved["lag_bank_max_frames"],
        lag_bank_step_frames=resolved["lag_bank_step_frames"],
        bottleneck_time_dilations=resolved["bottleneck_time_dilations"],
    )


def _complex_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.view_as_real(predicted), torch.view_as_real(target))


def _negative_si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = torch.sum(estimate * target, dim=-1, keepdim=True) * target
    projection = projection / (torch.sum(target * target, dim=-1, keepdim=True) + 1e-8)
    noise = estimate - projection
    ratio = (torch.sum(projection * projection, dim=-1) + 1e-8) / (
        torch.sum(noise * noise, dim=-1) + 1e-8
    )
    return -10.0 * torch.log10(ratio).mean()


def calculate_loss(model, output, target_english, target_russian):
    # Loss contract per input domain:
    #  * direct (clean_guard / channel mismatch): target_english is the clean
    #    embedded EN signal, mixture == EN + RU exactly — unchanged behaviour;
    #  * extractor domain: target_english arrives from the dataset already as
    #    extracted_mixture - target_russian, so the very same terms compare the
    #    predicted removed component with the true removed component.  The
    #    clean pre-extraction EN never enters this loss — the extractor is
    #    nonlinear and extractor(EN+RU) != extractor(EN) + extractor(RU).
    target_ru_spectrum = model.stft(target_russian)
    target_en_spectrum = model.stft(target_english)
    ru_spectral = _complex_l1(output["voice_spectrum"], target_ru_spectrum)
    en_spectral = _complex_l1(output["reference_spectrum"], target_en_spectrum)
    ru_wave = F.l1_loss(output["voice"], target_russian)
    en_wave = F.l1_loss(output["reference"], target_english)
    sisdr = _negative_si_sdr(output["voice"], target_russian)
    numerator = torch.abs(torch.sum(output["voice"] * target_english, dim=-1))
    denominator = torch.sqrt(
        torch.sum(output["voice"] ** 2, dim=-1)
        * torch.sum(target_english ** 2, dim=-1)
        + 1e-8
    )
    english_leakage = torch.mean(numerator / denominator)
    residual_energy = torch.mean((output["voice"] - target_russian) ** 2, dim=-1)
    english_energy = torch.mean(target_english ** 2, dim=-1)
    normalized_residual = torch.mean(
        torch.clamp(residual_energy / (english_energy + 1e-6), max=10.0)
    )
    total = (
        ru_spectral
        + 0.6 * en_spectral
        + 0.8 * ru_wave
        + 0.25 * en_wave
        + 0.002 * sisdr
        + 0.10 * english_leakage
        + 1.00 * normalized_residual
    )
    return total, {
        "ru_spectral": float(ru_spectral.detach().cpu()),
        "en_spectral": float(en_spectral.detach().cpu()),
        "ru_wave": float(ru_wave.detach().cpu()),
        "en_wave": float(en_wave.detach().cpu()),
        "ru_si_sdr_loss": float(sisdr.detach().cpu()),
        "english_leakage": float(english_leakage.detach().cpu()),
        "normalized_residual": float(normalized_residual.detach().cpu()),
    }


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float((-_negative_si_sdr(estimate, target)).detach().cpu())


# Canonical EN-vs-RU loudness regimes for per-bucket reporting.  They are keyed
# on the *intended* recipe level so every regime is measurable regardless of the
# small crop bias in synthesis.
REGIMES = (
    "en_below_ru",
    "en_equal_ru",
    "en_louder_ru",
    "en_much_louder_ru",
    "en_only",
    "ru_only",
)

REFERENCE_CONDITIONS = (
    "matched",
    "bad_timing",
    "weak",
    "bad_timing_weak",
    "zero",
)


def _en_regime(row: dict) -> str:
    recipe = row.get("recipe") or {}
    short = recipe.get("short_utterance") or {}
    if short.get("english_only"):
        return "en_only"
    if short.get("russian_only"):
        return "ru_only"
    # Below the training gate the English reference is zeroed, so there is no EN
    # in the mixture at all — treat it as a Russian-only preservation case.
    if float(recipe.get("english_speech_rms_db", 0.0)) < ENGLISH_SPEECH_GATE_DB:
        return "ru_only"
    level = float(recipe.get("english_to_target_db", -30.0))
    if level < -1.0:
        return "en_below_ru"
    if level <= 1.0:
        return "en_equal_ru"
    if level < 6.0:
        return "en_louder_ru"
    return "en_much_louder_ru"


def _reference_condition(row: dict) -> str:
    return str((row.get("recipe") or {}).get("reference_condition") or "matched")


def _input_domain(row: dict) -> str:
    return str(row.get("input_domain") or "direct")


def _channel_condition(row: dict) -> str:
    condition = str(
        row.get("channel_condition")
        or (row.get("recipe") or {}).get("channel_condition")
        or "legacy_random_mastering"
    )
    # Extractor-domain rows are reported as separate buckets so
    # clean_guard/channel metrics keep their historical meaning and the
    # extractor and extractor+channel regressions can never hide behind them.
    if _input_domain(row) == "extractor":
        return f"extractor_{condition}"
    return condition


def _stratified_selection(rows: list[dict], maximum: int) -> list[dict]:
    """Round-robin across loudness regimes so every regime (including the new
    hard EN-loud buckets) is represented in the evaluation sample."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_en_regime(row), []).append(row)
    order = sorted(groups)
    selected: list[dict] = []
    depth = 0
    while len(selected) < maximum and order:
        progressed = False
        for regime in order:
            group = groups[regime]
            if depth < len(group):
                selected.append(group[depth])
                progressed = True
                if len(selected) >= maximum:
                    break
        if not progressed:
            break
        depth += 1
    return selected


_EMPTY_METRICS = {
    "loss": float("nan"),
    "si_sdr_improvement_db": 0.0,
    "en_suppression_db": 0.0,
    "ru_si_sdr_db": 0.0,
    "examples": 0,
}


@torch.no_grad()
def _evaluate_selected(model, selected: list[dict], device) -> dict:
    model.eval()
    if not selected:
        return dict(_EMPTY_METRICS)
    losses: list[float] = []
    improvements: list[float] = []
    english_suppression: list[float] = []
    preservation: list[float] = []
    evaluation_batch = 4
    for left in range(0, len(selected), evaluation_batch):
        batch_rows = selected[left : left + evaluation_batch]
        examples = [synthesize_semantic_example(row) for row in batch_rows]
        values = [
            torch.from_numpy(np.stack([example[index] for example in examples])).to(device)
            for index in range(4)
        ]
        mix_t, ref_t, en_t, ru_t = values
        output = model(mix_t, ref_t)
        loss, _ = calculate_loss(model, output, en_t, ru_t)
        losses.extend([float(loss.cpu())] * len(batch_rows))
        for index in range(len(batch_rows)):
            estimated = output["voice"][index : index + 1]
            target = ru_t[index : index + 1]
            source = mix_t[index : index + 1]
            improvements.append(si_sdr(estimated, target) - si_sdr(source, target))
            before_error = torch.mean((source - target) ** 2)
            after_error = torch.mean((estimated - target) ** 2)
            english_suppression.append(
                float(10.0 * torch.log10((before_error + 1e-9) / (after_error + 1e-9)).cpu())
            )
            preservation.append(si_sdr(estimated, target))
    return {
        "loss": float(np.mean(losses)),
        "si_sdr_improvement_db": float(np.mean(improvements)),
        "en_suppression_db": float(np.mean(english_suppression)),
        "ru_si_sdr_db": float(np.mean(preservation)),
        "examples": len(selected),
    }


@torch.no_grad()
def evaluate(model, rows: list[dict], device, maximum: int = 64) -> dict:
    return _evaluate_selected(model, _stratified_selection(rows, maximum), device)


@torch.no_grad()
def evaluate_by_regime(model, rows: list[dict], device, per_regime: int = 40) -> dict:
    """Per-regime EN suppression / RU preservation, so EN<RU, EN≈RU and EN>RU
    can be tracked separately as stop-criteria."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_en_regime(row), []).append(row)
    result: dict[str, dict] = {}
    for regime in REGIMES:
        group = groups.get(regime)
        if group:
            result[regime] = _evaluate_selected(model, group[:per_regime], device)
    return result


@torch.no_grad()
def evaluate_by_reference_condition(
    model,
    rows: list[dict],
    device,
    per_condition: int = 40,
) -> dict:
    """Track the exact failure mode that Bronson exposed: the EN reference can
    be weak, badly timed or completely useless."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_reference_condition(row), []).append(row)
    result: dict[str, dict] = {}
    for condition in REFERENCE_CONDITIONS:
        group = groups.get(condition)
        if group:
            result[condition] = _evaluate_selected(model, group[:per_condition], device)
    return result


@torch.no_grad()
def evaluate_by_channel_condition(
    model,
    rows: list[dict],
    device,
    per_condition: int = 40,
) -> dict:
    """Report quality-mismatch buckets separately from loudness/timing buckets."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(_channel_condition(row), []).append(row)
    return {
        condition: _evaluate_selected(model, group[:per_condition], device)
        for condition, group in sorted(groups.items())
        if group
    }


# Regimes to render as listenable controls every epoch.  The loud EN buckets
# come first because they are the ones that must be checked before a full run.
CONTROL_REGIMES = (
    "en_much_louder_ru",
    "en_louder_ru",
    "en_equal_ru",
    "en_below_ru",
)

CONTROL_REFERENCE_CONDITIONS = (
    "zero",
    "bad_timing",
    "bad_timing_weak",
    "weak",
)

# Channel-quality buckets rendered as listenable controls: every mismatch
# category plus the clean guard, so a regression in any category is audible
# without re-running evaluation.  The extractor-domain variants exist only in
# manifests with input_domain rows and are skipped silently otherwise.
_DIRECT_CHANNEL_CONDITIONS = (
    "clean_guard",
    "embedded_low_bandwidth",
    "reference_low_bandwidth",
    "asymmetric_sample_rate",
    "asymmetric_eq",
    "codec_like",
    "compression_mismatch",
    "compound_hard",
    "event_en_only_matched",
    "event_en_then_ru",
    "event_ru_then_en",
    "event_overlap_matched",
    "event_weak_matched",
    "event_ru_only_zero_ref",
    "event_ru_only_wrong_ref",
    "event_both_wrong_ref_preserve",
)
# Independent hard-master conditions (positive-only 20260719 fine-tune).  The
# Rows from mastering-stress validation carry
# channel_condition="mastering_stress_<name>" and input_domain="extractor".
MASTERING_STRESS_CONDITIONS = (
    "ac3_5_1_to_aac_stereo",
    "independent_downmix",
    "broadcast_vs_release_master",
    "time_varying_ducking",
    "piecewise_gain_envelope",
    "compound_codec_eq_dynamics",
    "extractor_pair_asymmetry",
    "micro_timing_jitter",
    "drift_and_local_edit_positive",
    "very_low_embedded_en",
)
CONTROL_CHANNEL_CONDITIONS = (
    _DIRECT_CHANNEL_CONDITIONS
    + tuple(f"extractor_{name}" for name in _DIRECT_CHANNEL_CONDITIONS)
    + tuple(f"extractor_mastering_stress_{name}" for name in MASTERING_STRESS_CONDITIONS)
)


@torch.no_grad()
def save_controls(model, manifest: dict, run_root: Path, epoch: int, device) -> dict:
    valid = [row for row in manifest["examples"] if row["split"] == "valid"]
    by_regime: dict[str, list[dict]] = {}
    by_condition: dict[str, list[dict]] = {}
    by_channel: dict[str, list[dict]] = {}
    for row in valid:
        by_regime.setdefault(_en_regime(row), []).append(row)
        by_condition.setdefault(_reference_condition(row), []).append(row)
        by_channel.setdefault(_channel_condition(row), []).append(row)
    picks: list[tuple[str, dict]] = []
    for regime in CONTROL_REGIMES:
        group = by_regime.get(regime)
        if group:
            picks.append((regime, group[(epoch - 1) % len(group)]))
    for condition in CONTROL_REFERENCE_CONDITIONS:
        group = by_condition.get(condition)
        if group:
            picks.append((f"ref_{condition}", group[(epoch - 1) % len(group)]))
    for condition in CONTROL_CHANNEL_CONDITIONS:
        group = by_channel.get(condition)
        if group:
            picks.append((f"channel_{condition}", group[(epoch - 1) % len(group)]))
    if not picks:
        picks.append(("sample", valid[(epoch - 1) % len(valid)]))
    buckets: dict[str, dict] = {}
    for regime, row in picks:
        mixture, reference, english, russian = synthesize_semantic_example(row)
        output = model(
            torch.from_numpy(mixture).unsqueeze(0).to(device),
            torch.from_numpy(reference).unsqueeze(0).to(device),
        )
        rate = int(row["sample_rate"])
        root = run_root / "controls" / f"epoch_{epoch:03d}" / regime
        files = {
            "input_mixture": write_audio(root / "input_mixture.flac", mixture, rate),
            "input_reference": write_audio(root / "input_reference.flac", reference, rate),
            "expected_voice": write_audio(root / "expected_voice.flac", russian, rate),
            "expected_original_component": write_audio(root / "expected_original_component.flac", english, rate),
            "model_voice": write_audio(root / "model_voice.flac", output["voice"][0].cpu().numpy(), rate),
            "model_original_component": write_audio(root / "model_original_component.flac", output["reference"][0].cpu().numpy(), rate),
        }
        buckets[regime] = {
            "example_id": row["id"],
            "film_name": row.get("film_name", ""),
            "hardcase_bucket": row.get("hardcase_bucket"),
            "reference_condition": _reference_condition(row),
            "channel_condition": _channel_condition(row),
            "files": files,
        }
    # The first pick is the loudest available EN regime; expose it as the primary
    # "synthetic" control so the training UI keeps rendering, and keep the full
    # per-bucket set under "buckets" for manual pre-run review.
    primary_regime, primary_row = picks[0]
    primary = buckets[primary_regime]
    return {
        "example_id": primary["example_id"],
        "film_name": primary["film_name"],
        "primary_regime": primary_regime,
        "synthetic": {"film_name": primary["film_name"], "files": primary["files"]},
        "buckets": buckets,
    }


def save_checkpoint(path: Path, model, optimizer, epoch, global_step, manifest, config, history) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "format": FORMAT,
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
    parser.add_argument("--config-json", default="")
    parser.add_argument("--config-file", default="")
    args = parser.parse_args()
    if args.config_file:
        config = json.loads(Path(args.config_file).read_text(encoding="utf-8"))
    elif args.config_json:
        config = json.loads(args.config_json)
    else:
        raise RuntimeError("Укажите --config-json или --config-file.")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run_root = Path(args.run_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    random.seed(71337)
    np.random.seed(71337)
    torch.manual_seed(71337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(config.get("training_device_policy") or "").lower() == "cuda_required" and device.type != "cuda":
        raise RuntimeError("Конфигурация требует CUDA: обучение на CPU запрещено.")
    emit("log", message=f"Устройство: {device}{' · ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    model = build_semantic_model(config).to(device)
    emit(
        "log",
        message=(
            "Параметров модели: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-5)
    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("format") != FORMAT:
            raise RuntimeError("Выбрано сохранение несовместимой модели.")
        model.load_state_dict(state["model_state"])
        # Expanded checkpoints (expand_semantic_checkpoint.py) carry no
        # optimizer moments: parameter shapes changed, so Adam starts fresh.
        if state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        # Resume momentum/Adam moments, but honour the learning rate explicitly
        # selected for this continuation instead of silently restoring the old one.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(config["learning_rate"])
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        # The continuation controller relaunches this script once per epoch.
        # Without epoch-dependent seeding every relaunch would replay the same
        # DataLoader shuffle order, so consecutive epochs would see identical
        # batch sequences.
        random.seed(71337 + start_epoch)
        np.random.seed(71337 + start_epoch)
        torch.manual_seed(71337 + start_epoch)
        resumed_dataset = state.get("dataset_id")
        if resumed_dataset == manifest["dataset_id"]:
            history = list(state.get("history", []))
        else:
            # Continuing on a different dataset (fine-tuning on the hard set):
            # the inherited validation_loss was measured on an easier
            # distribution, so keep the weights/optimizer but start best- and
            # history-tracking fresh.  Otherwise the old, lower loss would keep
            # `semantic_ru_separator_best.pt` frozen at the pre-fine-tune model.
            history = []
            emit(
                "log",
                message=(
                    f"Смена датасета {resumed_dataset} -> {manifest['dataset_id']}: "
                    "история метрик и выбор лучшего чекпойнта сброшены."
                ),
            )

    train_rows = [row for row in manifest["examples"] if row["split"] == "train"]
    valid_rows = [row for row in manifest["examples"] if row["split"] == "valid"]
    test_rows = [row for row in manifest["examples"] if row["split"] == "test"]
    enforce_training_manifest_guards(manifest, train_rows)
    workers = max(0, int(config.get("data_loader_workers", 0)))
    loader = DataLoader(
        SemanticDataset(train_rows),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=max(1, int(config.get("data_loader_prefetch_factor", 2))) if workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    configured_maximum_batches = int(config.get("maximum_batches_per_epoch", 0))
    maximum_batches = (
        min(len(loader), configured_maximum_batches)
        if configured_maximum_batches > 0
        else int(config.get("quick_check_batches", 50))
        if args.mode == "quick"
        else len(loader)
    )
    gradient_accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    total_steps = max(1, args.epochs * min(len(loader), maximum_batches))
    best_loss = min((float(row.get("validation_loss", float("inf"))) for row in history), default=float("inf"))
    early_stopping_patience = max(1, int(config.get("early_stopping_patience", 3)))
    early_stopping_min_delta = max(0.0, float(config.get("early_stopping_min_delta", 0.0)))
    substantial_bad_patience = max(
        1, int(config.get("early_stopping_substantial_patience", 2))
    )
    substantial_bad_relative = max(
        0.0, float(config.get("early_stopping_substantial_relative", 0.10))
    )
    substantial_bad_absolute = max(
        0.0, float(config.get("early_stopping_substantial_absolute", 0.02))
    )
    consecutive_bad_epochs = 0
    consecutive_substantial_bad_epochs = 0
    running_best = float("inf")
    for previous in history:
        previous_loss = float(previous.get("validation_loss", float("inf")))
        if previous_loss < running_best - early_stopping_min_delta:
            running_best = previous_loss
            consecutive_bad_epochs = 0
        else:
            consecutive_bad_epochs += 1
    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch + 1
        model.train()
        losses: list[float] = []
        parts_sum: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (values, rows) in enumerate(loader):
            if batch_index >= maximum_batches:
                break
            mixture, reference, english, russian = [value.to(device, non_blocking=True) for value in values]
            output = model(mixture, reference)
            loss, parts = calculate_loss(model, output, english, russian)
            (loss / gradient_accumulation_steps).backward()
            should_step = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 >= min(len(loader), maximum_batches)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.0) + value
            progress = (local_epoch * min(len(loader), maximum_batches) + batch_index + 1) / total_steps * 100.0
            batch_number = batch_index + 1
            batch_count = min(len(loader), maximum_batches)
            if batch_number == 1 or batch_number % 50 == 0 or batch_number == batch_count:
                emit(
                    "progress",
                    stage="Обучение смысловому разделению RU/EN",
                    substage=f"Проход {epoch}, пакет {batch_number} из {batch_count}",
                    progress=progress,
                    current_file=str(rows.get("film_name", [""])[0]),
                    loss=losses[-1],
                )
        validation = evaluate(model, valid_rows, device)
        test = evaluate(model, test_rows, device)
        validation_short = evaluate(
            model,
            [
                row
                for row in valid_rows
                if row.get("augmentation_kind") == "short_utterance"
            ],
            device,
        )
        validation_long = evaluate(
            model,
            [
                row
                for row in valid_rows
                if row.get("augmentation_kind") != "short_utterance"
            ],
            device,
        )
        test_short = evaluate(
            model,
            [
                row
                for row in test_rows
                if row.get("augmentation_kind") == "short_utterance"
            ],
            device,
        )
        test_long = evaluate(
            model,
            [
                row
                for row in test_rows
                if row.get("augmentation_kind") != "short_utterance"
            ],
            device,
        )
        validation_regimes = evaluate_by_regime(model, valid_rows, device)
        test_regimes = evaluate_by_regime(model, test_rows, device)
        validation_reference_conditions = evaluate_by_reference_condition(
            model, valid_rows, device
        )
        test_reference_conditions = evaluate_by_reference_condition(
            model, test_rows, device
        )
        validation_channel_conditions = evaluate_by_channel_condition(
            model, valid_rows, device
        )
        test_channel_conditions = evaluate_by_channel_condition(
            model, test_rows, device
        )
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
            "validation_short_en_suppression_db": validation_short[
                "en_suppression_db"
            ],
            "validation_long_en_suppression_db": validation_long[
                "en_suppression_db"
            ],
            "test_short_en_suppression_db": test_short["en_suppression_db"],
            "test_long_en_suppression_db": test_long["en_suppression_db"],
            "validation_short_ru_si_sdr_db": validation_short["ru_si_sdr_db"],
            "validation_long_ru_si_sdr_db": validation_long["ru_si_sdr_db"],
            "test_short_ru_si_sdr_db": test_short["ru_si_sdr_db"],
            "test_long_ru_si_sdr_db": test_long["ru_si_sdr_db"],
            "validation_regime_en_suppression_db": {
                regime: metrics["en_suppression_db"]
                for regime, metrics in validation_regimes.items()
            },
            "validation_regime_ru_si_sdr_db": {
                regime: metrics["ru_si_sdr_db"]
                for regime, metrics in validation_regimes.items()
            },
            "test_regime_en_suppression_db": {
                regime: metrics["en_suppression_db"]
                for regime, metrics in test_regimes.items()
            },
            "test_regime_ru_si_sdr_db": {
                regime: metrics["ru_si_sdr_db"]
                for regime, metrics in test_regimes.items()
            },
            "validation_reference_en_suppression_db": {
                condition: metrics["en_suppression_db"]
                for condition, metrics in validation_reference_conditions.items()
            },
            "validation_reference_ru_si_sdr_db": {
                condition: metrics["ru_si_sdr_db"]
                for condition, metrics in validation_reference_conditions.items()
            },
            "test_reference_en_suppression_db": {
                condition: metrics["en_suppression_db"]
                for condition, metrics in test_reference_conditions.items()
            },
            "test_reference_ru_si_sdr_db": {
                condition: metrics["ru_si_sdr_db"]
                for condition, metrics in test_reference_conditions.items()
            },
            "validation_channel_en_suppression_db": {
                condition: metrics["en_suppression_db"]
                for condition, metrics in validation_channel_conditions.items()
            },
            "validation_channel_ru_si_sdr_db": {
                condition: metrics["ru_si_sdr_db"]
                for condition, metrics in validation_channel_conditions.items()
            },
            "test_channel_en_suppression_db": {
                condition: metrics["en_suppression_db"]
                for condition, metrics in test_channel_conditions.items()
            },
            "test_channel_ru_si_sdr_db": {
                condition: metrics["ru_si_sdr_db"]
                for condition, metrics in test_channel_conditions.items()
            },
            "regime_examples": {
                regime: metrics["examples"]
                for regime, metrics in validation_regimes.items()
            },
            "reference_condition_examples": {
                condition: metrics["examples"]
                for condition, metrics in validation_reference_conditions.items()
            },
            "channel_condition_examples": {
                condition: metrics["examples"]
                for condition, metrics in validation_channel_conditions.items()
            },
            "loss_parts": {key: value / count for key, value in parts_sum.items()},
            "control_examples": controls,
            "dataset_id": manifest["dataset_id"],
            "algorithm_version": FORMAT,
        }
        history.append(row)
        if args.mode != "quick":
            checkpoint = checkpoint_root / f"{run_root.name}_epoch{epoch}.pt"
            row["checkpoint"] = str(checkpoint.resolve())
            save_checkpoint(checkpoint, model, optimizer, epoch, global_step, manifest, config, history)
            save_checkpoint(checkpoint_root / "semantic_ru_separator_last.pt", model, optimizer, epoch, global_step, manifest, config, history)
            improved = float(validation["loss"]) < best_loss - early_stopping_min_delta
            if improved:
                best_loss = float(validation["loss"])
                save_checkpoint(checkpoint_root / "semantic_ru_separator_best.pt", model, optimizer, epoch, global_step, manifest, config, history)
                consecutive_bad_epochs = 0
                consecutive_substantial_bad_epochs = 0
            else:
                consecutive_bad_epochs += 1
                substantial_margin = max(
                    substantial_bad_absolute,
                    abs(best_loss) * substantial_bad_relative,
                )
                if float(validation["loss"]) >= best_loss + substantial_margin:
                    consecutive_substantial_bad_epochs += 1
                else:
                    consecutive_substantial_bad_epochs = 0
            row["improved_best"] = improved
            row["consecutive_bad_epochs"] = consecutive_bad_epochs
            row["consecutive_substantial_bad_epochs"] = (
                consecutive_substantial_bad_epochs
            )
        best_epoch = min(history, key=lambda item: float(item["validation_loss"]))
        report = {
            "schema_version": 4,
            "run_id": run_root.name,
            "dataset_id": manifest["dataset_id"],
            "algorithm_version": FORMAT,
            "mode": args.mode,
            "dry_run": args.mode == "quick",
            "device": str(device),
            "history": history,
            "best_epoch": best_epoch,
            "recommended_checkpoint": best_epoch.get("checkpoint"),
            "recommendation": "Продолжать, пока независимая ошибка и реальные речевые пробы улучшаются.",
            "early_stopping": {
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "consecutive_bad_epochs": consecutive_bad_epochs,
                "substantial_patience": substantial_bad_patience,
                "substantial_relative": substantial_bad_relative,
                "substantial_absolute": substantial_bad_absolute,
                "consecutive_substantial_bad_epochs": (
                    consecutive_substantial_bad_epochs
                ),
                "triggered": (
                    consecutive_bad_epochs >= early_stopping_patience
                    or consecutive_substantial_bad_epochs >= substantial_bad_patience
                ),
            },
            "note": "Модель предсказывает RU напрямую и не требует фазового совпадения мастер-версий.",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(run_root / "history.json", report)
        stop_for_plateau = consecutive_bad_epochs >= early_stopping_patience
        stop_for_substantial = (
            consecutive_substantial_bad_epochs >= substantial_bad_patience
        )
        if args.mode != "quick" and (stop_for_plateau or stop_for_substantial):
            emit(
                "log",
                message=(
                    "Ранняя остановка: "
                    + (
                        f"{consecutive_substantial_bad_epochs} эпохи подряд с существенным ухудшением. "
                        if stop_for_substantial
                        else f"{consecutive_bad_epochs} эпох подряд без улучшения. "
                    )
                    + "Рекомендуется semantic_ru_separator_best.pt."
                ),
            )
            break
    emit("complete", history=str((run_root / "history.json").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
