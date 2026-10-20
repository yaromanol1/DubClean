"""Deterministic synthesis shared by training and dataset inspection."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


ENGLISH_SPEECH_GATE_DB = -48.0
MAX_ENGLISH_AMPLIFICATION_DB = 24.0


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
        available = max(
            0, min(requested - prefix, len(reader) - max(0, start_frame))
        )
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
        impulse = np.exp(
            -np.arange(impulse_length) / max(1.0, sample_rate * 0.045)
        )
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


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    padded = np.pad(
        values.astype(np.float64), (window // 2, window - window // 2)
    )
    sums = np.cumsum(padded)
    return (sums[window:] - sums[:-window])[: len(values)] / window


def _ducking_envelope(target: np.ndarray, sample_rate: int) -> np.ndarray:
    energy = _moving_average(
        target.astype(np.float64) ** 2, int(round(sample_rate * 0.040))
    )
    envelope = np.sqrt(np.maximum(energy, 0.0))
    scale = float(np.percentile(envelope, 95))
    if scale < 1e-7:
        return np.zeros_like(envelope)
    envelope = np.clip(envelope / scale, 0.0, 1.0)
    return _moving_average(envelope, int(round(sample_rate * 0.150)))


def _best_active_slice(value: np.ndarray, frames: int) -> tuple[int, int]:
    frames = max(1, min(int(frames), len(value)))
    if frames >= len(value):
        return 0, len(value)
    energy = value.astype(np.float64) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    window_energy = cumulative[frames:] - cumulative[:-frames]
    start = int(np.argmax(window_energy))
    return start, start + frames


def place_short_utterances(
    reference: np.ndarray,
    embedded_english: np.ndarray,
    target_russian: np.ndarray,
    sample_rate: int,
    profile: dict,
    output_duration_sec: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_frames = int(round(output_duration_sec * sample_rate))
    english_frames = int(round(float(profile["english_duration_sec"]) * sample_rate))
    russian_frames = int(round(float(profile["russian_duration_sec"]) * sample_rate))
    en_left, en_right = _best_active_slice(embedded_english, english_frames)
    ru_left, ru_right = _best_active_slice(target_russian, russian_frames)
    en_reference = reference[en_left:en_right].copy()
    en_embedded = embedded_english[en_left:en_right].copy()
    ru_target = target_russian[ru_left:ru_right].copy()
    fade_frames = min(
        int(round(0.025 * sample_rate)),
        max(1, len(en_embedded) // 4),
        max(1, len(ru_target) // 4),
    )
    fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    for value in (en_reference, en_embedded, ru_target):
        value[:fade_frames] *= fade
        value[-fade_frames:] *= fade[::-1]
    reference_output = np.zeros(output_frames, dtype=np.float32)
    english_output = np.zeros(output_frames, dtype=np.float32)
    russian_output = np.zeros(output_frames, dtype=np.float32)

    def place(destination: np.ndarray, source: np.ndarray, offset_sec: float) -> None:
        start = max(0, int(round(offset_sec * sample_rate)))
        stop = min(len(destination), start + len(source))
        if stop > start:
            destination[start:stop] += source[: stop - start]

    if not bool(profile.get("russian_only")):
        place(reference_output, en_reference, float(profile["english_offset_sec"]))
        place(english_output, en_embedded, float(profile["english_offset_sec"]))
    if not bool(profile.get("english_only")):
        place(russian_output, ru_target, float(profile["russian_offset_sec"]))
    return reference_output, english_output, russian_output


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
    target = read_interval_mono(
        sources["target_speech"],
        float(recipe["dubbed_start_sec"]) - float(recipe["russian_shift_sec"]),
        duration,
        sample_rate,
        frames,
    )
    embedded_reference = transform_reference(reference, recipe, sample_rate)
    target_rms = max(
        float(np.sqrt(np.mean(target.astype(np.float64) ** 2))), 1e-5
    )
    reference_rms = max(
        float(np.sqrt(np.mean(embedded_reference.astype(np.float64) ** 2))), 1e-6
    )
    if float(recipe.get("english_speech_rms_db", 0.0)) < ENGLISH_SPEECH_GATE_DB:
        embedded_reference.fill(0.0)
    else:
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
