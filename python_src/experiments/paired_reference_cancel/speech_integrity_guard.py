"""Conservative, stage-wise protection against locally lost Russian speech.

The subtraction and speech-enhancement models are useful candidates, not an
authority on whether a phrase may disappear.  This module keeps a known-safe
signal next to every destructive stage and selects it only on intervals where
several independent DSP measurements agree that speech was lost.

Two guards are intentionally separate:

``guard_primary_pass``
    Compares the dubbed EN+RU speech stem with the first DubClean Voice output.
    The mixture is restored only when the raw, aligned *and* adapted English
    references all agree that English is weak and the processed signal has
    lost sustained speech.  Ambiguous windows remain processed.  The two
    separated stems always vote; the raw programme reference votes only when it
    carries *periodic* speech, because music and effects occupy the same bands
    there and would otherwise veto a repair by loudness alone.

``guard_second_pass``
    Compares the already protected first-pass Russian speech with a destructive
    enhancement candidate (currently MossFormer).  A confirmed or plausible
    local speech loss selects the safe first pass.  This asymmetry is deliberate:
    keeping an already cleaned first pass cannot re-introduce the original
    English mix, while accepting a doubtful second pass can erase a phrase.

Detection is streaming at the block level and deterministic.  VAD-like energy
and periodicity are gates only; decisions additionally require multi-band
loss and spectral/temporal-envelope evidence.  The writer streams the complete
file and joins selected intervals with a raised-cosine crossfade.
"""
from __future__ import annotations

import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel.storage import atomic_json, utc_now


SCHEMA_VERSION = 1
REPORT_KIND = "dubclean_speech_integrity_guard"
ALGORITHM = "stagewise_multisignal_integrity_v5"

ProgressCallback = Callable[..., None]


def _defaults() -> dict[str, Any]:
    return {
        "enabled": True,
        "algorithm": ALGORITHM,
        "analysis_sample_rate": 16000,
        "analysis_chunk_sec": 60.0,
        "window_sec": 0.50,
        "hop_sec": 0.25,
        "minimum_active_rms_db": -48.0,
        # The safe side of the second guard is already the protected Russian
        # first pass.  A narrowly lower voiced-speech floor prevents the
        # optional enhancer from deleting quiet syllables without relaxing the
        # primary EN+RU restoration or the unvoiced/noise route.
        "second_pass_quiet_voiced_minimum_rms_db": -50.0,
        "second_pass_quiet_voiced_minimum_loss_db": 6.0,
        "second_pass_quiet_voiced_minimum_speech_band_share": 0.90,
        "second_pass_quiet_voiced_minimum_temporal_dynamic_db": 12.0,
        "minimum_speech_band_share": 0.52,
        "minimum_periodicity": 0.18,
        # A second pass is optional and destructive.  A modest, coherent loss
        # is therefore enough to retain the known-safe first-pass interval.
        "second_pass_suspect_loss_db": 3.5,
        "second_pass_confirmed_loss_db": 6.0,
        "second_pass_severe_loss_db": 12.0,
        "second_pass_band_loss_db": 3.0,
        "second_pass_confirmed_band_loss_db": 5.5,
        "second_pass_minimum_shape_similarity": 0.76,
        "second_pass_minimum_temporal_similarity": 0.58,
        # The safe first pass may contain unvoiced speech that misses only the
        # ordinary periodicity gate.  Restoring it is allowed solely on a
        # severe, structured and strongly matching multi-band loss.
        "second_pass_unvoiced_minimum_loss_db": 12.0,
        "second_pass_unvoiced_minimum_band_loss_db": 8.0,
        "second_pass_unvoiced_minimum_lost_bands": 3,
        "second_pass_unvoiced_minimum_speech_band_share": 0.90,
        "second_pass_unvoiced_minimum_shape_similarity": 0.88,
        "second_pass_unvoiced_minimum_temporal_similarity": 0.85,
        "second_pass_unvoiced_minimum_temporal_dynamic_db": 10.0,
        "second_pass_unvoiced_minimum_spectral_peak_share": 0.006,
        "candidate_silence_rms_db": -68.0,
        # Restoring an EN+RU stem is riskier.  All three English references and
        # all loss measurements must agree before the primary guard acts.
        "primary_minimum_loss_db": 7.5,
        "primary_minimum_band_loss_db": 6.0,
        "primary_minimum_en_margin_db": 10.0,
        "primary_minimum_en_band_margin_db": 8.0,
        "primary_minimum_removed_shape_similarity": 0.80,
        "primary_minimum_removed_temporal_similarity": 0.64,
        "primary_minimum_confidence": 0.86,
        # A raw programme reference is a useful last-resort EN veto only when
        # it matches the audio that the destructive pass actually removed.
        # This excludes unrelated periodic music while retaining missed EN.
        "primary_raw_reference_minimum_removed_shape_similarity": 0.80,
        "primary_raw_reference_minimum_removed_temporal_similarity": 0.55,
        "primary_raw_reference_sustained_shape_similarity": 0.94,
        "primary_raw_reference_maximum_sustained_peak_share": 0.55,
        "primary_boundary_probe_sec": 0.05,
        # Alternative for genuinely unvoiced/muffled Russian speech.  It does
        # not relax the ordinary periodicity gate: every stronger condition
        # below must agree before that one missing cue may be bypassed.
        "primary_unvoiced_minimum_loss_db": 12.0,
        "primary_unvoiced_minimum_band_loss_db": 8.0,
        "primary_unvoiced_minimum_lost_bands": 3,
        "primary_unvoiced_minimum_speech_band_share": 0.90,
        "primary_unvoiced_minimum_removed_shape_similarity": 0.995,
        "primary_unvoiced_minimum_removed_temporal_similarity": 0.97,
        "primary_unvoiced_minimum_temporal_dynamic_db": 10.0,
        "primary_unvoiced_minimum_spectral_peak_share": 0.006,
        "primary_unvoiced_minimum_confidence": 0.97,
        "minimum_interval_sec": 0.20,
        "maximum_merge_gap_sec": 0.16,
        "boundary_padding_sec": 0.04,
        "crossfade_sec": 0.10,
        "maximum_auto_interval_sec": 8.0,
        "maximum_restored_fraction": 0.02,
        "minimum_restoration_budget_sec": 3.0,
        "maximum_reported_rejections": 250,
        "maximum_duration_delta_sec": 0.10,
        "writer_block_sec": 20.0,
    }


def resolved_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a complete, finite and safely bounded configuration."""
    defaults = _defaults()
    value = {**defaults, **(config or {})}
    value["enabled"] = bool(value.get("enabled", True))
    value["algorithm"] = ALGORITHM
    for key, default in defaults.items():
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError, OverflowError):
            number = float(default)
        value[key] = number if math.isfinite(number) else float(default)

    value["analysis_sample_rate"] = int(
        np.clip(value["analysis_sample_rate"], 8000, 24000)
    )
    value["analysis_chunk_sec"] = float(
        np.clip(value["analysis_chunk_sec"], 2.0, 180.0)
    )
    value["window_sec"] = float(np.clip(value["window_sec"], 0.20, 1.50))
    value["hop_sec"] = float(
        np.clip(value["hop_sec"], 0.05, value["window_sec"])
    )
    value["minimum_speech_band_share"] = float(
        np.clip(value["minimum_speech_band_share"], 0.0, 1.0)
    )
    value["minimum_periodicity"] = float(
        np.clip(value["minimum_periodicity"], 0.0, 1.0)
    )
    value["minimum_active_rms_db"] = float(
        np.clip(value["minimum_active_rms_db"], -120.0, 0.0)
    )
    value["second_pass_quiet_voiced_minimum_rms_db"] = float(
        np.clip(
            value["second_pass_quiet_voiced_minimum_rms_db"],
            value["minimum_active_rms_db"] - 12.0,
            value["minimum_active_rms_db"],
        )
    )
    for key in (
        "second_pass_minimum_shape_similarity",
        "second_pass_minimum_temporal_similarity",
        "second_pass_quiet_voiced_minimum_speech_band_share",
        "second_pass_unvoiced_minimum_speech_band_share",
        "second_pass_unvoiced_minimum_shape_similarity",
        "second_pass_unvoiced_minimum_temporal_similarity",
        "second_pass_unvoiced_minimum_spectral_peak_share",
        "primary_minimum_removed_shape_similarity",
        "primary_minimum_removed_temporal_similarity",
        "primary_minimum_confidence",
        "primary_raw_reference_minimum_removed_shape_similarity",
        "primary_raw_reference_minimum_removed_temporal_similarity",
        "primary_raw_reference_sustained_shape_similarity",
        "primary_raw_reference_maximum_sustained_peak_share",
        "primary_unvoiced_minimum_speech_band_share",
        "primary_unvoiced_minimum_removed_shape_similarity",
        "primary_unvoiced_minimum_removed_temporal_similarity",
        "primary_unvoiced_minimum_spectral_peak_share",
        "primary_unvoiced_minimum_confidence",
        "maximum_restored_fraction",
    ):
        value[key] = float(np.clip(value[key], 0.0, 1.0))
    for key in (
        "minimum_interval_sec",
        "maximum_merge_gap_sec",
        "boundary_padding_sec",
        "crossfade_sec",
        "minimum_restoration_budget_sec",
        "maximum_duration_delta_sec",
        "writer_block_sec",
    ):
        value[key] = max(0.0, float(value[key]))
    value["maximum_auto_interval_sec"] = max(
        value["minimum_interval_sec"], float(value["maximum_auto_interval_sec"])
    )
    value["primary_boundary_probe_sec"] = float(
        np.clip(value["primary_boundary_probe_sec"], 0.02, 0.20)
    )
    value["maximum_reported_rejections"] = max(
        0, int(value["maximum_reported_rejections"])
    )
    value["primary_unvoiced_minimum_lost_bands"] = int(
        np.clip(round(value["primary_unvoiced_minimum_lost_bands"]), 1, 3)
    )
    value["second_pass_unvoiced_minimum_lost_bands"] = int(
        np.clip(round(value["second_pass_unvoiced_minimum_lost_bands"]), 1, 3)
    )
    for key in (
        "second_pass_quiet_voiced_minimum_loss_db",
        "second_pass_quiet_voiced_minimum_temporal_dynamic_db",
        "second_pass_unvoiced_minimum_loss_db",
        "second_pass_unvoiced_minimum_band_loss_db",
        "second_pass_unvoiced_minimum_temporal_dynamic_db",
        "primary_unvoiced_minimum_loss_db",
        "primary_unvoiced_minimum_band_loss_db",
        "primary_unvoiced_minimum_temporal_dynamic_db",
    ):
        value[key] = max(0.0, float(value[key]))
    return value


def _audio_identity(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 6),
    }


def _safe_db(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), 1e-12)))


def _mono(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.shape[1] == 1:
        return data[:, 0]
    if data.shape[1] >= 3:
        # The independent raw-program veto must follow the same dialogue
        # convention as the rest of DubClean: in surround sources the centre
        # channel is the strongest evidence that English is actually present.
        return data[:, 2]
    return np.mean(data, axis=1, dtype=np.float32)


def _match_channels(values: np.ndarray, channels: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == channels:
        return data
    if channels == 1:
        return np.mean(data, axis=1, keepdims=True, dtype=np.float32)
    if data.shape[1] == 1:
        return np.repeat(data, channels, axis=1)
    if data.shape[1] > channels:
        return data[:, :channels]
    repeats = int(math.ceil(channels / data.shape[1]))
    return np.tile(data, (1, repeats))[:, :channels]


def _resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if source_rate == target_rate or data.size == 0:
        return data
    divisor = math.gcd(int(source_rate), int(target_rate))
    result = signal.resample_poly(
        data,
        int(target_rate) // divisor,
        int(source_rate) // divisor,
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def _normalise_scopes(
    intervals: Iterable[tuple[float, float] | dict[str, Any]] | None,
    duration: float,
) -> list[tuple[float, float]]:
    if intervals is None:
        return [(0.0, max(0.0, duration))]
    parsed: list[tuple[float, float]] = []
    for item in intervals:
        if isinstance(item, dict):
            start = item.get("start_sec", item.get("start", 0.0))
            end = item.get("end_sec", item.get("end", 0.0))
        else:
            start, end = item
        try:
            left = float(start)
            right = float(end)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(left) or not math.isfinite(right):
            continue
        left = max(0.0, min(duration, left))
        right = max(left, min(duration, right))
        if right - left > 1e-6:
            parsed.append((left, right))
    merged: list[list[float]] = []
    for left, right in sorted(parsed):
        if merged and left <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(left, right) for left, right in merged]


class _Progress:
    def __init__(
        self,
        callback: ProgressCallback | None,
        ctx: Any | None,
        stage: str,
    ) -> None:
        self.callback = callback
        self.ctx = ctx
        self.stage = stage
        self.last_bucket = -1

    def emit(self, progress: float, detail: str) -> None:
        value = float(np.clip(progress, 0.0, 100.0))
        bucket = int(value)
        if bucket == self.last_bucket and value < 100.0:
            return
        self.last_bucket = bucket
        if self.callback is not None:
            try:
                self.callback(value, detail)
            except TypeError:
                self.callback(value)
        if self.ctx is not None:
            try:
                self.ctx.update(
                    stage=self.stage,
                    substage=detail,
                    local_progress=value,
                )
                if hasattr(self.ctx, "check_stop"):
                    self.ctx.check_stop()
            except TypeError:
                self.ctx.update(local_progress=value)


def _validate_sources(
    paths: Sequence[Path], cfg: dict[str, Any]
) -> tuple[int, int, float]:
    if not paths:
        raise ValueError("No audio sources were supplied.")
    infos = [sf.info(str(path)) for path in paths]
    # Stages intentionally operate at different native rates: the primary
    # DubClean output can be 24 kHz while mixture/references and MossFormer are
    # 48 kHz.  Alignment is therefore checked in seconds and every analysis
    # block is independently resampled to a shared rate.
    rate = int(infos[0].samplerate)
    durations = [float(info.frames) / max(int(info.samplerate), 1) for info in infos]
    delta = max(durations) - min(durations)
    if delta > cfg["maximum_duration_delta_sec"]:
        raise ValueError("Speech integrity sources have different durations.")
    duration = min(durations)
    return rate, int(math.floor(duration * rate)), duration


def _read_mono_block(
    handle: sf.SoundFile, start: int, end: int
) -> np.ndarray:
    handle.seek(max(0, start))
    values = handle.read(
        max(0, end - start), dtype="float32", always_2d=True
    )
    result = _mono(values)
    needed = max(0, end - start)
    if result.size < needed:
        result = np.pad(result, (0, needed - result.size))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _spectral_features(
    values: np.ndarray,
    rate: int,
    *,
    include_envelopes: bool = True,
    include_periodicity: bool = True,
) -> dict[str, Any]:
    data = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    if data.size == 0:
        data = np.zeros(1, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    centred = data - float(np.mean(data))
    taper = np.hanning(centred.size) if centred.size > 2 else np.ones(centred.size)
    spectrum = np.fft.rfft(centred * taper)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    frequencies = np.fft.rfftfreq(centred.size, d=1.0 / rate)
    usable_max = min(7000.0, rate * 0.48)

    def band_power(low: float, high: float) -> float:
        mask = (frequencies >= low) & (frequencies < min(high, rate * 0.5))
        return float(np.sum(power[mask])) if np.any(mask) else 0.0

    speech_bands = np.asarray(
        [
            band_power(120.0, 500.0),
            band_power(500.0, 2000.0),
            band_power(2000.0, 6000.0),
        ],
        dtype=np.float64,
    )
    total_power = band_power(50.0, usable_max)
    speech_share = float(np.sum(speech_bands) / max(total_power, 1e-18))
    speech_spectrum_mask = (frequencies >= 120.0) & (
        frequencies < min(6000.0, rate * 0.5)
    )
    speech_spectrum = power[speech_spectrum_mask]
    spectral_peak_share = (
        float(np.max(speech_spectrum) / max(float(np.sum(speech_spectrum)), 1e-18))
        if speech_spectrum.size
        else 0.0
    )

    if include_envelopes:
        upper = max(850.0, usable_max)
        edges = np.geomspace(90.0, upper, 13)
        envelope = np.asarray(
            [band_power(float(edges[i]), float(edges[i + 1])) for i in range(12)],
            dtype=np.float64,
        )
        envelope += max(float(np.sum(envelope)), 1e-18) * 1e-10
        envelope /= max(float(np.sum(envelope)), 1e-18)

        frame = max(8, int(round(rate * 0.025)))
        hop = max(4, int(round(rate * 0.010)))
        squared = np.square(data, dtype=np.float64)
        cumulative_power = np.concatenate(([0.0], np.cumsum(squared)))
        if data.size >= frame:
            starts = np.arange(0, data.size - frame + 1, hop, dtype=np.int64)
            frame_power = (
                cumulative_power[starts + frame] - cumulative_power[starts]
            ) / frame
            temporal_values = np.sqrt(np.maximum(frame_power, 0.0))
        else:
            temporal_values = np.asarray([rms], dtype=np.float64)
        temporal_envelope = np.log10(temporal_values + 1e-9)
        temporal_dynamic_db = float(
            20.0
            * (
                np.percentile(temporal_envelope, 90)
                - np.percentile(temporal_envelope, 10)
            )
        )
    else:
        envelope = np.empty(0, dtype=np.float64)
        temporal_envelope = np.empty(0, dtype=np.float64)
        temporal_dynamic_db = 0.0

    if include_periodicity:
        # Normalised pitch-range autocorrelation is a deliberately modest
        # voicing gate.  Pre-emphasis prevents slowly varying background/noise
        # from looking periodic merely because adjacent samples are correlated.
        voiced = signal.lfilter([1.0, -0.97], [1.0], centred)
        nfft = 1 << max(1, (max(1, voiced.size * 2 - 1)).bit_length())
        autocorrelation = np.fft.irfft(
            np.square(np.abs(np.fft.rfft(voiced, n=nfft))), n=nfft
        )[: voiced.size]
        lag_low = max(1, int(round(rate / 400.0)))
        lag_high = min(voiced.size, int(round(rate / 70.0)) + 1)
        if lag_high > lag_low and voiced.size:
            lags = np.arange(lag_low, lag_high, dtype=np.int64)
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.square(voiced, dtype=np.float64)))
            )
            left_energy = cumulative[voiced.size - lags]
            right_energy = cumulative[voiced.size] - cumulative[lags]
            denominator = np.sqrt(
                np.maximum(left_energy * right_energy, 1e-24)
            )
            periodicity = float(
                np.max(np.maximum(0.0, autocorrelation[lags] / denominator))
            )
        else:
            periodicity = 0.0
    else:
        periodicity = 0.0
    return {
        "rms_db": _safe_db(rms),
        "band_db": np.asarray([_safe_db(math.sqrt(v)) for v in speech_bands]),
        "speech_band_share": float(np.clip(speech_share, 0.0, 1.0)),
        "spectral_peak_share": float(np.clip(spectral_peak_share, 0.0, 1.0)),
        "spectral_envelope": envelope,
        "temporal_envelope": temporal_envelope,
        "temporal_dynamic_db": max(0.0, temporal_dynamic_db),
        "periodicity": float(np.clip(periodicity, 0.0, 1.0)),
    }


def _shape_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
    left = np.asarray(first["spectral_envelope"], dtype=np.float64)
    right = np.asarray(second["spectral_envelope"], dtype=np.float64)
    return float(np.clip(np.sum(np.sqrt(left * right)), 0.0, 1.0))


def _temporal_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
    left = np.asarray(first["temporal_envelope"], dtype=np.float64)
    right = np.asarray(second["temporal_envelope"], dtype=np.float64)
    length = min(left.size, right.size)
    if length < 3:
        return 0.0
    left = left[:length] - float(np.mean(left[:length]))
    right = right[:length] - float(np.mean(right[:length]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-10:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _speech_gate(features: dict[str, Any], cfg: dict[str, Any]) -> bool:
    # Periodicity is deliberately only a veto/gate.  It never confirms a
    # repair without level, multi-band and envelope evidence.
    return bool(
        features["rms_db"] >= cfg["minimum_active_rms_db"]
        and features["speech_band_share"] >= cfg["minimum_speech_band_share"]
        and features["periodicity"] >= cfg["minimum_periodicity"]
    )


def _raw_reference_matches_removed_speech(
    raw_reference: dict[str, Any],
    removed: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[bool, float, float]:
    """Return whether raw-program speech is present in the restored delta.

    Periodicity alone is not sufficient for a complete programme mix: a
    stationary tone or chord is periodic too.  The raw signal may veto a
    repair only when its robust spectral/temporal envelopes agree with the
    audio that the destructive pass actually removed.  The sustained fallback
    covers a stable voiced vowel whose temporal envelope has too little
    variation for correlation to be meaningful.
    """
    shape = _shape_similarity(raw_reference, removed)
    temporal = _temporal_similarity(raw_reference, removed)
    removed_active = bool(
        removed["rms_db"] >= cfg["minimum_active_rms_db"]
        and removed["speech_band_share"] >= cfg["minimum_speech_band_share"]
    )
    envelope_match = bool(
        (
            shape
            >= cfg["primary_raw_reference_minimum_removed_shape_similarity"]
            and temporal
            >= cfg["primary_raw_reference_minimum_removed_temporal_similarity"]
        )
        or (
            shape >= cfg["primary_raw_reference_sustained_shape_similarity"]
            and raw_reference["spectral_peak_share"]
            <= cfg["primary_raw_reference_maximum_sustained_peak_share"]
        )
    )
    return bool(_speech_gate(raw_reference, cfg) and removed_active and envelope_match), shape, temporal


def _second_pass_decision(
    safe: dict[str, Any], candidate: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    loss_db = float(safe["rms_db"] - candidate["rms_db"])
    band_losses = np.asarray(safe["band_db"] - candidate["band_db"])
    suspect_bands = int(np.sum(band_losses >= cfg["second_pass_band_loss_db"]))
    confirmed_bands = int(
        np.sum(band_losses >= cfg["second_pass_confirmed_band_loss_db"])
    )
    unvoiced_lost_bands = int(
        np.sum(
            band_losses
            >= cfg["second_pass_unvoiced_minimum_band_loss_db"]
        )
    )
    shape = _shape_similarity(safe, candidate)
    temporal = _temporal_similarity(safe, candidate)
    active = _speech_gate(safe, cfg)
    quiet_voiced_active = bool(
        not active
        and safe["rms_db"]
        >= cfg["second_pass_quiet_voiced_minimum_rms_db"]
        and safe["rms_db"] < cfg["minimum_active_rms_db"]
        and safe["speech_band_share"]
        >= cfg["second_pass_quiet_voiced_minimum_speech_band_share"]
        and safe["periodicity"] >= cfg["minimum_periodicity"]
        and safe["temporal_dynamic_db"]
        >= cfg["second_pass_quiet_voiced_minimum_temporal_dynamic_db"]
    )
    near_silence = bool(candidate["rms_db"] <= cfg["candidate_silence_rms_db"])
    envelope_support = bool(
        shape >= cfg["second_pass_minimum_shape_similarity"]
        or temporal >= cfg["second_pass_minimum_temporal_similarity"]
    )
    severe = bool(
        active
        and loss_db >= cfg["second_pass_severe_loss_db"]
        and suspect_bands >= 2
    )
    confirmed = bool(
        active
        and loss_db >= cfg["second_pass_confirmed_loss_db"]
        and confirmed_bands >= 2
        and (envelope_support or near_silence)
    )
    suspect = bool(
        active
        and loss_db >= cfg["second_pass_suspect_loss_db"]
        and suspect_bands >= 2
        and envelope_support
    )
    # This route exists only for a narrow, empirically observed band of quiet
    # voiced Russian speech.  It deliberately excludes the 3.5 dB suspect
    # branch and demands both sustained speech structure and at least 6 dB of
    # coherent multi-band loss.  A destroyed candidate may have no meaningful
    # envelope similarity, so the severe case does not require it.
    quiet_severe = bool(
        quiet_voiced_active
        and loss_db
        >= max(
            cfg["second_pass_severe_loss_db"],
            cfg["second_pass_quiet_voiced_minimum_loss_db"],
        )
        and suspect_bands >= 2
    )
    quiet_confirmed = bool(
        quiet_voiced_active
        and loss_db >= cfg["second_pass_quiet_voiced_minimum_loss_db"]
        and confirmed_bands >= 2
        and (envelope_support or near_silence)
    )
    unvoiced_speech_evidence = bool(
        not active
        and not quiet_voiced_active
        # As in the primary guard, only periodicity may be absent.  Level and
        # a stricter speech-band occupancy are still mandatory.
        and safe["rms_db"] >= cfg["minimum_active_rms_db"]
        and safe["periodicity"] < cfg["minimum_periodicity"]
        and safe["speech_band_share"]
        >= cfg["second_pass_unvoiced_minimum_speech_band_share"]
        and safe["temporal_dynamic_db"]
        >= cfg["second_pass_unvoiced_minimum_temporal_dynamic_db"]
        and safe["spectral_peak_share"]
        >= cfg["second_pass_unvoiced_minimum_spectral_peak_share"]
    )
    unvoiced_severe = bool(
        unvoiced_speech_evidence
        and loss_db >= cfg["second_pass_unvoiced_minimum_loss_db"]
        and unvoiced_lost_bands
        >= cfg["second_pass_unvoiced_minimum_lost_bands"]
        and shape >= cfg["second_pass_unvoiced_minimum_shape_similarity"]
        and temporal
        >= cfg["second_pass_unvoiced_minimum_temporal_similarity"]
    )
    selected = (
        severe
        or confirmed
        or suspect
        or quiet_severe
        or quiet_confirmed
        or unvoiced_severe
    )
    decision = (
        "safe_previous_unvoiced_severe"
        if unvoiced_severe
        else "safe_previous_quiet_voiced"
        if quiet_severe or quiet_confirmed
        else "safe_previous_confirmed"
        if severe or confirmed
        else "safe_previous_conservative"
        if suspect
        else "candidate"
    )
    confidence = float(
        np.clip(
            0.30
            * min(
                1.0,
                max(0.0, loss_db)
                / max(cfg["second_pass_confirmed_loss_db"], 1e-6),
            )
            + 0.25 * confirmed_bands / 3.0
            + 0.20 * shape
            + 0.15 * max(0.0, temporal)
            + 0.10 * min(1.0, safe["periodicity"] / 0.08),
            0.0,
            1.0,
        )
    )
    reasons: list[str] = []
    if unvoiced_severe:
        reasons.append("severe_unvoiced_second_pass_loss_consensus")
    elif quiet_severe or quiet_confirmed:
        reasons.append("quiet_voiced_second_pass_loss_consensus")
    elif not active:
        reasons.append("safe_not_confirmed_as_speech")
    if loss_db >= cfg["second_pass_suspect_loss_db"]:
        reasons.append("local_level_loss")
    if suspect_bands >= 2:
        reasons.append("multi_band_speech_loss")
    if shape >= cfg["second_pass_minimum_shape_similarity"]:
        reasons.append("same_spectral_envelope_at_lower_level")
    if temporal >= cfg["second_pass_minimum_temporal_similarity"]:
        reasons.append("same_temporal_speech_envelope_at_lower_level")
    if near_silence:
        reasons.append("candidate_became_silence")
    return {
        "selected": selected,
        "decision": decision,
        "confidence": round(confidence, 6),
        "reasons": reasons,
        "metrics": {
            "safe_rms_db": round(float(safe["rms_db"]), 4),
            "candidate_rms_db": round(float(candidate["rms_db"]), 4),
            "loss_db": round(loss_db, 4),
            "band_loss_db": [round(float(v), 4) for v in band_losses],
            "suspect_band_count": suspect_bands,
            "confirmed_band_count": confirmed_bands,
            "unvoiced_lost_band_count": unvoiced_lost_bands,
            "spectral_envelope_similarity": round(shape, 6),
            "temporal_envelope_similarity": round(temporal, 6),
            "safe_periodicity": round(float(safe["periodicity"]), 6),
            "quiet_voiced_speech_evidence": quiet_voiced_active,
            "quiet_voiced_severe_loss": quiet_severe,
            "quiet_voiced_confirmed_loss": quiet_confirmed,
            "safe_speech_band_share": round(
                float(safe["speech_band_share"]), 6
            ),
            "safe_temporal_dynamic_db": round(
                float(safe["temporal_dynamic_db"]), 4
            ),
            "safe_spectral_peak_share": round(
                float(safe["spectral_peak_share"]), 6
            ),
            "unvoiced_speech_evidence": unvoiced_speech_evidence,
        },
    }


def _primary_decision(
    mixture: dict[str, Any],
    processed: dict[str, Any],
    references: Sequence[dict[str, Any]],
    removed: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    loss_db = float(mixture["rms_db"] - processed["rms_db"])
    band_losses = np.asarray(mixture["band_db"] - processed["band_db"])
    lost_bands = int(np.sum(band_losses >= cfg["primary_minimum_band_loss_db"]))
    unvoiced_lost_bands = int(
        np.sum(
            band_losses >= cfg["primary_unvoiced_minimum_band_loss_db"]
        )
    )
    margins = [float(mixture["rms_db"] - item["rms_db"]) for item in references]
    band_margins = [
        float(np.median(np.asarray(mixture["band_db"] - item["band_db"])))
        for item in references
    ]
    raw_reference = references[0]
    # Reported for diagnostics only.  The unvoiced route is *not* allowed to
    # admit the raw reference as English speech: unlike the two separated
    # stems, the raw reference is a complete programme mix in which speech,
    # music and effects share every band, so a non-periodic detector cannot
    # tell a whispered line from a broadband impact.  Loud effects are far
    # more common than unvoiced English, and admitting them here vetoes
    # genuine Russian-only repairs (a door slam over a Russian phrase blocked
    # the repair while both speech stems reported English 40 dB down).
    # Real English speech in the programme mix is periodic enough for the
    # ordinary gate below, which stays in force.
    raw_reference_unvoiced_evidence = bool(
        raw_reference["rms_db"] >= cfg["minimum_active_rms_db"]
        and raw_reference["periodicity"] < cfg["minimum_periodicity"]
        and raw_reference["speech_band_share"]
        >= cfg["primary_unvoiced_minimum_speech_band_share"]
        and raw_reference["temporal_dynamic_db"]
        >= cfg["primary_unvoiced_minimum_temporal_dynamic_db"]
        and raw_reference["spectral_peak_share"]
        >= cfg["primary_unvoiced_minimum_spectral_peak_share"]
    )
    (
        raw_reference_speech_evidence,
        raw_reference_removed_shape,
        raw_reference_removed_temporal,
    ) = _raw_reference_matches_removed_speech(raw_reference, removed, cfg)
    # The two separated English speech stems always participate.  The raw
    # centre/programme reference is an independent veto only when it contains
    # speech evidence; a loud effect or music hit is not English speech.
    effective_margins = list(margins[1:])
    effective_band_margins = list(band_margins[1:])
    if raw_reference_speech_evidence:
        effective_margins.append(margins[0])
        effective_band_margins.append(band_margins[0])
    triple_weak = bool(
        all(
            value >= cfg["primary_minimum_en_margin_db"]
            for value in effective_margins
        )
        and all(
            value >= cfg["primary_minimum_en_band_margin_db"]
            for value in effective_band_margins
        )
    )
    shape = _shape_similarity(mixture, removed)
    temporal = _temporal_similarity(mixture, removed)
    active = _speech_gate(mixture, cfg)
    processed_silence = bool(
        processed["rms_db"] <= cfg["candidate_silence_rms_db"]
    )
    confidence = float(
        np.clip(
            0.22
            * min(
                1.0,
                max(0.0, loss_db) / max(cfg["primary_minimum_loss_db"], 1e-6),
            )
            + 0.18 * lost_bands / 3.0
            + 0.22
            * min(
                1.0,
                max(0.0, min(effective_margins))
                / max(cfg["primary_minimum_en_margin_db"], 1e-6),
            )
            + 0.14
            * min(
                1.0,
                max(0.0, min(effective_band_margins))
                / max(cfg["primary_minimum_en_band_margin_db"], 1e-6),
            )
            + 0.12 * shape
            + 0.07 * max(0.0, temporal)
            + 0.05 * min(1.0, mixture["periodicity"] / 0.08),
            0.0,
            1.0,
        )
    )
    normally_confirmed = bool(
        active
        and triple_weak
        and loss_db >= cfg["primary_minimum_loss_db"]
        and lost_bands >= 2
        and shape >= cfg["primary_minimum_removed_shape_similarity"]
        and (
            temporal >= cfg["primary_minimum_removed_temporal_similarity"]
            or processed_silence
        )
        and confidence >= cfg["primary_minimum_confidence"]
    )
    unvoiced_speech_evidence = bool(
        not active
        # This route may bypass only periodicity, never level or spectral
        # occupancy.  Thus a quiet/non-speech window cannot enter it.
        and mixture["rms_db"] >= cfg["minimum_active_rms_db"]
        and mixture["periodicity"] < cfg["minimum_periodicity"]
        and mixture["speech_band_share"]
        >= cfg["primary_unvoiced_minimum_speech_band_share"]
        # Speech has syllabic modulation and structured spectral peaks;
        # stationary broadband noise and steady tonal beds fail these vetoes.
        and mixture["temporal_dynamic_db"]
        >= cfg["primary_unvoiced_minimum_temporal_dynamic_db"]
        and mixture["spectral_peak_share"]
        >= cfg["primary_unvoiced_minimum_spectral_peak_share"]
    )
    unvoiced_confirmed = bool(
        unvoiced_speech_evidence
        and triple_weak
        and loss_db >= cfg["primary_unvoiced_minimum_loss_db"]
        and unvoiced_lost_bands
        >= cfg["primary_unvoiced_minimum_lost_bands"]
        and shape
        >= cfg["primary_unvoiced_minimum_removed_shape_similarity"]
        and temporal
        >= cfg["primary_unvoiced_minimum_removed_temporal_similarity"]
        and confidence >= cfg["primary_unvoiced_minimum_confidence"]
    )
    confirmed = normally_confirmed or unvoiced_confirmed
    reasons: list[str] = []
    if unvoiced_confirmed:
        reasons.append("severe_unvoiced_speech_loss_consensus")
    elif not active:
        reasons.append("mixture_not_confirmed_as_speech")
    if triple_weak:
        reasons.append(
            "three_english_references_agree_english_is_weak"
            if raw_reference_speech_evidence
            else "english_speech_stems_agree_english_is_weak"
        )
    else:
        reasons.append("english_reference_consensus_failed")
    if raw_reference_speech_evidence:
        reasons.append("raw_reference_speech_evidence")
    else:
        reasons.append("raw_reference_non_speech_ignored")
    if loss_db >= cfg["primary_minimum_loss_db"]:
        reasons.append("primary_level_loss")
    if lost_bands >= 2:
        reasons.append("primary_multi_band_speech_loss")
    if shape >= cfg["primary_minimum_removed_shape_similarity"]:
        reasons.append("removed_audio_matches_input_spectral_envelope")
    if temporal >= cfg["primary_minimum_removed_temporal_similarity"]:
        reasons.append("removed_audio_matches_input_temporal_envelope")
    return {
        "selected": confirmed,
        "decision": (
            "restore_ru_only_mixture_unvoiced_severe"
            if unvoiced_confirmed
            else "restore_ru_only_mixture"
            if normally_confirmed
            else "processed"
        ),
        "confidence": round(confidence, 6),
        "reasons": reasons,
        "metrics": {
            "mixture_rms_db": round(float(mixture["rms_db"]), 4),
            "processed_rms_db": round(float(processed["rms_db"]), 4),
            "loss_db": round(loss_db, 4),
            "band_loss_db": [round(float(v), 4) for v in band_losses],
            "lost_band_count": lost_bands,
            "unvoiced_lost_band_count": unvoiced_lost_bands,
            "english_reference_margin_db": [round(v, 4) for v in margins],
            "english_reference_band_margin_db": [
                round(v, 4) for v in band_margins
            ],
            "triple_reference_consensus": triple_weak,
            "raw_reference_speech_evidence": raw_reference_speech_evidence,
            "raw_reference_periodic_gate": bool(
                _speech_gate(raw_reference, cfg)
            ),
            "raw_reference_unvoiced_speech_evidence": (
                raw_reference_unvoiced_evidence
            ),
            "raw_reference_rms_db": round(float(raw_reference["rms_db"]), 4),
            "raw_reference_periodicity": round(
                float(raw_reference["periodicity"]), 6
            ),
            "raw_reference_speech_band_share": round(
                float(raw_reference["speech_band_share"]), 6
            ),
            "raw_reference_temporal_dynamic_db": round(
                float(raw_reference["temporal_dynamic_db"]), 4
            ),
            "raw_reference_spectral_peak_share": round(
                float(raw_reference["spectral_peak_share"]), 6
            ),
            "raw_reference_removed_spectral_similarity": round(
                float(raw_reference_removed_shape), 6
            ),
            "raw_reference_removed_temporal_similarity": round(
                float(raw_reference_removed_temporal), 6
            ),
            "removed_spectral_envelope_similarity": round(shape, 6),
            "removed_temporal_envelope_similarity": round(temporal, 6),
            "mixture_periodicity": round(float(mixture["periodicity"]), 6),
            "mixture_speech_band_share": round(
                float(mixture["speech_band_share"]), 6
            ),
            "mixture_temporal_dynamic_db": round(
                float(mixture["temporal_dynamic_db"]), 4
            ),
            "mixture_spectral_peak_share": round(
                float(mixture["spectral_peak_share"]), 6
            ),
            "unvoiced_speech_evidence": unvoiced_speech_evidence,
        },
    }


def _window_count(scopes: Sequence[tuple[float, float]], hop: float) -> int:
    return sum(max(1, int(math.ceil((right - left) / hop))) for left, right in scopes)


def _scan(
    paths: Sequence[Path],
    cfg: dict[str, Any],
    scopes: Sequence[tuple[float, float]],
    evaluator: Callable[[Sequence[np.ndarray], int], dict[str, Any]],
    progress: _Progress,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rates = [int(sf.info(str(path)).samplerate) for path in paths]
    analysis_rate = min(min(source_rates), int(cfg["analysis_sample_rate"]))
    window_sec = cfg["window_sec"]
    hop_sec = cfg["hop_sec"]
    chunk_sec = cfg["analysis_chunk_sec"]
    total_windows = _window_count(scopes, hop_sec)
    completed = 0
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    handles = [sf.SoundFile(str(path)) for path in paths]
    try:
        for scope_index, (scope_start, scope_end) in enumerate(scopes):
            next_window = scope_start
            core_start = scope_start
            while core_start < scope_end - 1e-9:
                core_end = min(scope_end, core_start + chunk_sec)
                starts: list[float] = []
                while next_window < core_end - 1e-9:
                    starts.append(next_window)
                    next_window += hop_sec
                if not starts:
                    core_start = core_end
                    continue
                read_start = max(scope_start, starts[0] - window_sec)
                read_end = min(scope_end, starts[-1] + window_sec * 2.0)
                arrays = [
                    _resample(
                        _read_mono_block(
                            handle,
                            int(math.floor(read_start * source_rate)),
                            int(math.ceil(read_end * source_rate)),
                        ),
                        source_rate,
                        analysis_rate,
                    )
                    for handle, source_rate in zip(handles, source_rates)
                ]
                for start_sec in starts:
                    end_sec = min(scope_end, start_sec + window_sec)
                    local_left = int(round((start_sec - read_start) * analysis_rate))
                    local_right = int(round((end_sec - read_start) * analysis_rate))
                    if local_right - local_left < int(analysis_rate * 0.15):
                        completed += 1
                        continue
                    windows = [item[local_left:local_right] for item in arrays]
                    length = min((item.size for item in windows), default=0)
                    if length <= 0:
                        completed += 1
                        continue
                    decision = evaluator([item[:length] for item in windows], analysis_rate)
                    row = {
                        "start_sec": round(start_sec, 6),
                        "end_sec": round(end_sec, 6),
                        "analysis_scope_index": scope_index,
                        **decision,
                    }
                    if decision["selected"]:
                        selected.append(row)
                    elif (
                        "local_level_loss" in decision["reasons"]
                        or "primary_level_loss" in decision["reasons"]
                    ):
                        rejected.append(row)
                    completed += 1
                    progress.emit(
                        75.0 * completed / max(1, total_windows),
                        "Анализ сохранности речи",
                    )
                core_start = core_end
    finally:
        for handle in handles:
            handle.close()
    return selected, rejected


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    losses = [float(item["metrics"].get("loss_db", 0.0)) for item in rows]
    result: dict[str, Any] = {
        "window_count": len(rows),
        "maximum_loss_db": round(max(losses, default=0.0), 4),
        "median_loss_db": round(float(np.median(losses)) if losses else 0.0, 4),
    }
    raw_speech_evidence = [
        bool(item["metrics"]["raw_reference_speech_evidence"])
        for item in rows
        if "raw_reference_speech_evidence" in item.get("metrics", {})
    ]
    if raw_speech_evidence:
        result["raw_reference_speech_evidence"] = any(raw_speech_evidence)
        result["raw_reference_speech_evidence_window_count"] = sum(
            raw_speech_evidence
        )
    # Non-periodic raw-programme energy no longer vetoes a repair, so it has to
    # stay visible on the interval that was repaired anyway: it is the one
    # signal an operator needs in order to review this class of decision.
    raw_unvoiced = [
        bool(item["metrics"]["raw_reference_unvoiced_speech_evidence"])
        for item in rows
        if "raw_reference_unvoiced_speech_evidence" in item.get("metrics", {})
    ]
    if raw_unvoiced:
        result["raw_reference_unvoiced_speech_evidence"] = any(raw_unvoiced)
        result["raw_reference_unvoiced_speech_evidence_window_count"] = sum(
            raw_unvoiced
        )
    periodicity = [
        float(item["metrics"]["raw_reference_periodicity"])
        for item in rows
        if "raw_reference_periodicity" in item.get("metrics", {})
    ]
    if periodicity:
        result["maximum_raw_reference_periodicity"] = round(
            max(periodicity), 6
        )
    return result


def _merge_selected(
    rows: Sequence[dict[str, Any]],
    duration: float,
    cfg: dict[str, Any],
    scopes: Sequence[tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["start_sec"], item["end_sec"])):
        if (
            merged
            and row["start_sec"]
            <= merged[-1]["end_sec"] + cfg["maximum_merge_gap_sec"]
            and row["decision"] == merged[-1]["decision"]
            and row["analysis_scope_index"]
            == merged[-1]["analysis_scope_index"]
        ):
            current = merged[-1]
            current["end_sec"] = max(current["end_sec"], row["end_sec"])
            current["confidence"] = max(current["confidence"], row["confidence"])
            current["window_rows"].append(row)
            current["reasons"] = sorted(
                set(current["reasons"]).union(row["reasons"])
            )
        else:
            merged.append(
                {
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "decision": row["decision"],
                    "analysis_scope_index": int(row["analysis_scope_index"]),
                    "confidence": float(row["confidence"]),
                    "reasons": list(row["reasons"]),
                    "window_rows": [row],
                }
            )

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in merged:
        scope_index = int(item["analysis_scope_index"])
        scope_left, scope_right = scopes[scope_index]
        item["start_sec"] = max(
            scope_left, item["start_sec"] - cfg["boundary_padding_sec"]
        )
        item["end_sec"] = min(
            scope_right, item["end_sec"] + cfg["boundary_padding_sec"]
        )
        item["duration_sec"] = item["end_sec"] - item["start_sec"]
        item["metrics"] = _aggregate_metrics(item.pop("window_rows"))
        if item["duration_sec"] < cfg["minimum_interval_sec"]:
            item["veto_reason"] = "interval_too_short"
            rejected.append(item)
        elif item["duration_sec"] > cfg["maximum_auto_interval_sec"]:
            item["veto_reason"] = "interval_too_long_for_automatic_repair"
            rejected.append(item)
        else:
            eligible.append(item)

    budget = max(
        cfg["minimum_restoration_budget_sec"],
        sum(right - left for left, right in scopes)
        * cfg["maximum_restored_fraction"],
    )
    used = 0.0
    accepted_ids: set[int] = set()
    priority = sorted(
        enumerate(eligible),
        key=lambda pair: (-pair[1]["confidence"], pair[1]["start_sec"]),
    )
    for index, item in priority:
        if used + item["duration_sec"] <= budget + 1e-9:
            accepted_ids.add(index)
            used += item["duration_sec"]
        else:
            rejected.append({**item, "veto_reason": "restoration_budget_exceeded"})
    accepted = [item for index, item in enumerate(eligible) if index in accepted_ids]
    return accepted, rejected


def _trim_to_weak_english_edges(
    mixture_path: Path,
    processed_path: Path,
    raw_reference_path: Path,
    separated_reference_paths: Sequence[Path],
    intervals: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    scopes: Sequence[tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shrink repaired spans whose edges no longer keep English weak.

    A decision window is ``window_sec`` long, so an English onset inside its
    last fraction can hide behind the window average: the window still reports
    a large mean margin while the final tens of milliseconds are already
    English.  Restoring the mixture there swaps a clean Russian phrase for an
    English one.  Boundary padding widens the same edge further.

    This pass only ever moves an edge inwards, one crossfade-length slice at a
    time, and drops an interval that cannot keep a boundary clean.  It never
    grows a repair and never creates one.
    """
    slice_sec = max(0.02, min(cfg["crossfade_sec"], cfg["window_sec"] * 0.5))
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    if not intervals:
        return kept, dropped
    paths = [
        mixture_path,
        processed_path,
        raw_reference_path,
        *separated_reference_paths,
    ]
    source_rates = [int(sf.info(str(path)).samplerate) for path in paths]
    analysis_rate = min(min(source_rates), int(cfg["analysis_sample_rate"]))
    handles = [sf.SoundFile(str(path)) for path in paths]
    try:
        def edge_is_weak(left: float, right: float) -> bool:
            if right - left <= 1e-6:
                return False
            values = [
                _resample(
                    _read_mono_block(
                        handle,
                        int(math.floor(left * rate)),
                        int(math.ceil(right * rate)),
                    ),
                    rate,
                    analysis_rate,
                )
                for handle, rate in zip(handles, source_rates)
            ]
            length = min(item.size for item in values)
            if length <= 0:
                return False
            probe_frames = min(
                length,
                max(
                    int(round(analysis_rate * 0.02)),
                    int(round(analysis_rate * cfg["primary_boundary_probe_sec"])),
                ),
            )
            probe_hop = max(1, probe_frames // 2)
            probe_starts = list(range(0, max(1, length - probe_frames + 1), probe_hop))
            final_start = max(0, length - probe_frames)
            if not probe_starts or probe_starts[-1] != final_start:
                probe_starts.append(final_start)
            for probe_start in probe_starts:
                probe_end = min(length, probe_start + probe_frames)
                probes = [item[probe_start:probe_end] for item in values]
                mixture_values = probes[0]
                processed_values = probes[1]
                raw_values = probes[2]
                mixture = _spectral_features(mixture_values, analysis_rate)
                removed = _spectral_features(
                    mixture_values - processed_values,
                    analysis_rate,
                )
                raw_reference = _spectral_features(raw_values, analysis_rate)
                raw_matches_removed, _, _ = _raw_reference_matches_removed_speech(
                    raw_reference, removed, cfg
                )
                if (
                    raw_matches_removed
                    and mixture["rms_db"] - raw_reference["rms_db"]
                    < cfg["primary_minimum_en_margin_db"]
                ):
                    return False
                for reference_values in probes[3:]:
                    reference = _spectral_features(
                        reference_values,
                        analysis_rate,
                        include_envelopes=False,
                        include_periodicity=False,
                    )
                    # Digital silence in both signals is not English evidence.
                    if (
                        reference["rms_db"] > cfg["candidate_silence_rms_db"]
                        and mixture["rms_db"] - reference["rms_db"]
                        < cfg["primary_minimum_en_margin_db"]
                    ):
                        return False
            return True

        # The crossfade reaches one fade length beyond each edge, so the region
        # that has to stay English-free is the interval plus both fade tails.
        fade = cfg["crossfade_sec"]
        for item in intervals:
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            scope_index = int(item.get("analysis_scope_index", 0))
            scope_left, scope_right = scopes[scope_index]
            trimmed_start = 0.0
            trimmed_end = 0.0
            while (
                end - start >= cfg["minimum_interval_sec"]
                and not edge_is_weak(
                    max(scope_left, start - fade), min(end, start + slice_sec)
                )
            ):
                start += slice_sec
                trimmed_start += slice_sec
            while (
                end - start >= cfg["minimum_interval_sec"]
                and not edge_is_weak(
                    max(start, end - slice_sec), min(scope_right, end + fade)
                )
            ):
                end -= slice_sec
                trimmed_end += slice_sec
            if end - start < cfg["minimum_interval_sec"]:
                dropped.append(
                    {**item, "veto_reason": "english_active_at_repair_boundary"}
                )
                continue
            if trimmed_start or trimmed_end:
                item = {
                    **item,
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "duration_sec": round(end - start, 6),
                    "boundary_trim_start_sec": round(trimmed_start, 6),
                    "boundary_trim_end_sec": round(trimmed_end, 6),
                }
            kept.append(item)
    finally:
        for handle in handles:
            handle.close()
    return kept, dropped


def _base_report(
    mode: str,
    cfg: dict[str, Any],
    sources: dict[str, Path],
    duration: float,
    scopes: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "algorithm": ALGORITHM,
        "mode": mode,
        "created_at": utc_now(),
        "enabled": bool(cfg["enabled"]),
        "config": cfg,
        "sources": {key: _audio_identity(path) for key, path in sources.items()},
        "duration_sec": round(duration, 6),
        "analysis_scopes": [
            {"start_sec": round(left, 6), "end_sec": round(right, 6)}
            for left, right in scopes
        ],
    }


def _copy_if_needed(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if source != output:
        shutil.copy2(source, output)


def _selection_weight(
    positions: np.ndarray,
    intervals: Sequence[dict[str, Any]],
    fade_frames: int,
    rate: int,
    analysis_scopes: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    weight = np.zeros(positions.size, dtype=np.float32)
    for item in intervals:
        start = int(round(float(item["start_sec"]) * rate))
        end = int(round(float(item["end_sec"]) * rate))
        inside = (positions >= start) & (positions < end)
        weight[inside] = 1.0
        if fade_frames > 0:
            left = (positions >= start - fade_frames) & (positions < start)
            if np.any(left):
                phase = (positions[left] - (start - fade_frames)) / fade_frames
                weight[left] = np.maximum(
                    weight[left], 0.5 - 0.5 * np.cos(np.pi * phase)
                )
            right = (positions >= end) & (positions < end + fade_frames)
            if np.any(right):
                phase = (positions[right] - end) / fade_frames
                weight[right] = np.maximum(
                    weight[right], 0.5 + 0.5 * np.cos(np.pi * phase)
                )
    if analysis_scopes is not None:
        # Preview protection is allowed to modify only its selected scenes.
        # Crossfade tails must not leak into neighbouring, unanalysed audio.
        allowed = np.zeros(positions.size, dtype=bool)
        for scope_start, scope_end in analysis_scopes:
            left = int(round(float(scope_start) * rate))
            right = int(round(float(scope_end) * rate))
            allowed |= (positions >= left) & (positions < right)
        weight[~allowed] = 0.0
    return weight


def _read_at_target_rate(
    reader: sf.SoundFile,
    target_start_frame: int,
    target_frame_count: int,
    target_rate: int,
) -> np.ndarray:
    """Read a time-aligned block and resample it without loading the file.

    A small native-rate margin is decoded around every block so resample_poly's
    FIR edge does not create a periodic seam in long-form output.  The margin
    is cropped after conversion using the exact time origin.
    """
    source_rate = int(reader.samplerate)
    source_total = int(len(reader))
    if target_frame_count <= 0:
        return np.empty((0, int(reader.channels)), dtype=np.float32)
    if source_rate == target_rate:
        if target_start_frame >= source_total:
            return np.empty((0, int(reader.channels)), dtype=np.float32)
        reader.seek(target_start_frame)
        return np.nan_to_num(
            reader.read(
                min(target_frame_count, source_total - target_start_frame),
                dtype="float32",
                always_2d=True,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    start_sec = target_start_frame / target_rate
    end_sec = (target_start_frame + target_frame_count) / target_rate
    padding_sec = 0.10
    source_left = max(0, int(math.floor((start_sec - padding_sec) * source_rate)))
    source_right = min(
        source_total, int(math.ceil((end_sec + padding_sec) * source_rate))
    )
    if source_right <= source_left:
        return np.empty((0, int(reader.channels)), dtype=np.float32)
    reader.seek(source_left)
    native = np.nan_to_num(
        reader.read(
            source_right - source_left, dtype="float32", always_2d=True
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    divisor = math.gcd(source_rate, target_rate)
    converted = signal.resample_poly(
        native,
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    ).astype(np.float32, copy=False)
    converted_origin_sec = source_left / source_rate
    offset = max(0, int(round((start_sec - converted_origin_sec) * target_rate)))
    return np.nan_to_num(
        converted[offset : offset + target_frame_count],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _apply_selection(
    base_path: Path,
    replacement_path: Path,
    output_path: Path,
    intervals: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    progress: _Progress,
    analysis_scopes: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    if not intervals or not cfg["enabled"]:
        _copy_if_needed(base_path, output_path)
        progress.emit(100.0, "Замена не требуется")
        return {
            "modified": False,
            "selected_interval_count": 0,
            "selected_duration_sec": 0.0,
            "output": _audio_identity(output_path),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
    try:
        with sf.SoundFile(str(base_path)) as base_reader, sf.SoundFile(
            str(replacement_path)
        ) as replacement_reader:
            rate = int(base_reader.samplerate)
            channels = int(base_reader.channels)
            # The deliverable follows the base/candidate duration exactly.
            # A replacement that is a few frames shorter (within the validated
            # tolerance) must never truncate a feature film.
            total_frames = int(len(base_reader))
            block_frames = max(rate, int(round(rate * cfg["writer_block_sec"])))
            fade_frames = int(round(rate * cfg["crossfade_sec"]))
            with sf.SoundFile(
                str(temporary),
                mode="w",
                samplerate=rate,
                channels=channels,
                format=base_reader.format,
                subtype=base_reader.subtype,
            ) as writer:
                cursor = 0
                while cursor < total_frames:
                    count = min(block_frames, total_frames - cursor)
                    base_reader.seek(cursor)
                    base = np.nan_to_num(
                        base_reader.read(count, dtype="float32", always_2d=True),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    replacement_read = _read_at_target_rate(
                        replacement_reader, cursor, count, rate
                    )
                    replacement = _match_channels(replacement_read, channels)
                    length = base.shape[0]
                    if length == 0:
                        break
                    if replacement.shape[0] < length:
                        # Outside available replacement audio the no-op base is
                        # the only safe value, including within a fade tail.
                        replacement = np.concatenate(
                            [replacement, base[replacement.shape[0] : length]],
                            axis=0,
                        )
                    positions = np.arange(cursor, cursor + length, dtype=np.int64)
                    block_start_sec = (cursor - fade_frames) / rate
                    block_end_sec = (cursor + length + fade_frames) / rate
                    relevant_intervals = [
                        item
                        for item in intervals
                        if float(item["end_sec"]) >= block_start_sec
                        and float(item["start_sec"]) <= block_end_sec
                    ]
                    weight = _selection_weight(
                        positions,
                        relevant_intervals,
                        fade_frames,
                        rate,
                        analysis_scopes,
                    )[:, None]
                    combined = base[:length] * (1.0 - weight) + replacement[:length] * weight
                    writer.write(np.clip(combined, -1.0, 1.0))
                    cursor += length
                    progress.emit(
                        75.0 + 25.0 * cursor / max(1, total_frames),
                        "Безопасная склейка речи",
                    )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "modified": True,
        "selected_interval_count": len(intervals),
        "selected_duration_sec": round(
            sum(float(item["end_sec"]) - float(item["start_sec"]) for item in intervals),
            6,
        ),
        "crossfade_sec": cfg["crossfade_sec"],
        "output": _audio_identity(output_path),
    }


def guard_second_pass(
    safe_previous_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    analysis_intervals: Iterable[tuple[float, float] | dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Protect an already-safe first pass from a destructive second pass."""
    cfg = resolved_config(config)
    safe = Path(safe_previous_path).resolve()
    candidate = Path(candidate_path).resolve()
    output = Path(output_path).resolve()
    progress = _Progress(progress_callback, ctx, "Проверка сохранности речи")
    try:
        _, _, duration = _validate_sources([safe, candidate], cfg)
    except ValueError as exc:
        # For an optional destructive stage, an invalid comparison must fall
        # back to the known-safe previous pass rather than accepting candidate.
        _copy_if_needed(safe, output)
        report = _base_report(
            "post_second_pass",
            cfg,
            {"safe_previous": safe, "candidate": candidate},
            float(sf.info(str(safe)).duration),
            [],
        )
        report["summary"] = {
            "modified": True,
            "fallback": "safe_previous",
            "reason": str(exc),
            "confirmed_interval_count": 0,
        }
        report["application"] = {"output": _audio_identity(output)}
        if report_path is not None:
            atomic_json(Path(report_path), report)
        progress.emit(100.0, "Сохранён безопасный первый проход")
        return report
    scopes = _normalise_scopes(analysis_intervals, duration)
    report = _base_report(
        "post_second_pass",
        cfg,
        {"safe_previous": safe, "candidate": candidate},
        duration,
        scopes,
    )
    if not cfg["enabled"] or not scopes:
        _copy_if_needed(candidate, output)
        report["confirmed_intervals"] = []
        report["rejected_candidates"] = []
        report["summary"] = {
            "modified": False,
            "reason": "disabled" if not cfg["enabled"] else "empty_analysis_scope",
            "confirmed_interval_count": 0,
        }
        report["application"] = {"output": _audio_identity(output)}
        if report_path is not None:
            atomic_json(Path(report_path), report)
        return report

    def evaluator(arrays: Sequence[np.ndarray], rate: int) -> dict[str, Any]:
        return _second_pass_decision(
            _spectral_features(arrays[0], rate),
            _spectral_features(arrays[1], rate, include_periodicity=False),
            cfg,
        )

    selected_rows, rejected_rows = _scan(
        [safe, candidate], cfg, scopes, evaluator, progress
    )
    intervals, vetoed = _merge_selected(selected_rows, duration, cfg, scopes)
    application = _apply_selection(
        candidate,
        safe,
        output,
        intervals,
        cfg,
        progress,
        scopes,
    )
    reason_counts = Counter(
        reason for item in intervals for reason in item.get("reasons", [])
    )
    report["confirmed_intervals"] = intervals
    report["rejected_candidates"] = (
        rejected_rows + vetoed
    )[: cfg["maximum_reported_rejections"]]
    report["summary"] = {
        "modified": bool(application["modified"]),
        "confirmed_interval_count": len(intervals),
        "confirmed_duration_sec": application.get("selected_duration_sec", 0.0),
        "confirmed_window_count": len(selected_rows),
        "rejected_candidate_count": len(rejected_rows) + len(vetoed),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
    }
    report["application"] = application
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


def guard_primary_pass(
    mixture_path: str | Path,
    processed_path: str | Path,
    raw_english_reference_path: str | Path,
    aligned_english_reference_path: str | Path,
    adapted_english_reference_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    analysis_intervals: Iterable[tuple[float, float] | dict[str, Any]] | None = None,
    report_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Restore only high-confidence RU-only losses after the primary pass."""
    cfg = resolved_config(config)
    mixture = Path(mixture_path).resolve()
    processed = Path(processed_path).resolve()
    raw_reference = Path(raw_english_reference_path).resolve()
    aligned_reference = Path(aligned_english_reference_path).resolve()
    adapted_reference = Path(adapted_english_reference_path).resolve()
    output = Path(output_path).resolve()
    sources = {
        "mixture": mixture,
        "processed": processed,
        "raw_english_reference": raw_reference,
        "aligned_english_reference": aligned_reference,
        "adapted_english_reference": adapted_reference,
    }
    progress = _Progress(progress_callback, ctx, "Проверка сохранности речи")
    try:
        _, _, duration = _validate_sources(list(sources.values()), cfg)
    except ValueError as exc:
        # Restoring an EN+RU source without valid consensus is unsafe.
        _copy_if_needed(processed, output)
        report = _base_report(
            "post_primary_pass",
            cfg,
            sources,
            float(sf.info(str(processed)).duration),
            [],
        )
        report["summary"] = {
            "modified": False,
            "fallback": "processed",
            "reason": str(exc),
            "confirmed_interval_count": 0,
        }
        report["application"] = {"output": _audio_identity(output)}
        if report_path is not None:
            atomic_json(Path(report_path), report)
        progress.emit(100.0, "Сохранён обработанный результат")
        return report
    scopes = _normalise_scopes(analysis_intervals, duration)
    report = _base_report("post_primary_pass", cfg, sources, duration, scopes)
    if not cfg["enabled"] or not scopes:
        _copy_if_needed(processed, output)
        report["confirmed_intervals"] = []
        report["rejected_candidates"] = []
        report["summary"] = {
            "modified": False,
            "reason": "disabled" if not cfg["enabled"] else "empty_analysis_scope",
            "confirmed_interval_count": 0,
        }
        report["application"] = {"output": _audio_identity(output)}
        if report_path is not None:
            atomic_json(Path(report_path), report)
        return report

    def evaluator(arrays: Sequence[np.ndarray], rate: int) -> dict[str, Any]:
        mixture_values, processed_values, raw, aligned, adapted = arrays
        return _primary_decision(
            _spectral_features(mixture_values, rate),
            _spectral_features(
                processed_values,
                rate,
                include_envelopes=False,
                include_periodicity=False,
            ),
            [
                _spectral_features(
                    raw,
                    rate,
                ),
                _spectral_features(
                    aligned,
                    rate,
                    include_envelopes=False,
                    include_periodicity=False,
                ),
                _spectral_features(
                    adapted,
                    rate,
                    include_envelopes=False,
                    include_periodicity=False,
                ),
            ],
            _spectral_features(
                mixture_values - processed_values,
                rate,
                include_periodicity=False,
            ),
            cfg,
        )

    selected_rows, rejected_rows = _scan(
        [mixture, processed, raw_reference, aligned_reference, adapted_reference],
        cfg,
        scopes,
        evaluator,
        progress,
    )
    intervals, vetoed = _merge_selected(selected_rows, duration, cfg, scopes)
    intervals, boundary_vetoed = _trim_to_weak_english_edges(
        mixture,
        processed,
        raw_reference,
        [aligned_reference, adapted_reference],
        intervals,
        cfg,
        scopes,
    )
    vetoed.extend(boundary_vetoed)
    application = _apply_selection(
        processed,
        mixture,
        output,
        intervals,
        cfg,
        progress,
        scopes,
    )
    reason_counts = Counter(
        reason for item in intervals for reason in item.get("reasons", [])
    )
    report["confirmed_intervals"] = intervals
    report["rejected_candidates"] = (
        rejected_rows + vetoed
    )[: cfg["maximum_reported_rejections"]]
    report["summary"] = {
        "modified": bool(application["modified"]),
        "confirmed_interval_count": len(intervals),
        "confirmed_duration_sec": application.get("selected_duration_sec", 0.0),
        "confirmed_window_count": len(selected_rows),
        "rejected_candidate_count": len(rejected_rows) + len(vetoed),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
    }
    report["application"] = application
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


# Explicit aliases make the integration call sites self-documenting while
# keeping the short public names convenient for tests and tools.
protect_second_pass_integrity = guard_second_pass
protect_primary_integrity = guard_primary_pass
