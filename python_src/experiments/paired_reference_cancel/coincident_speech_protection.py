"""Conservative DSP protection for coincident English/Russian words.

DubClean Voice receives an aligned English reference ``R`` and a dubbed
speech stem ``M`` containing both English and Russian speech.  For a short
word that sounds nearly the same in both languages, a separator can remove
both speakers.  This module treats that failure as the classical
echo-canceller "double talk" problem:

* ``P`` is the first-pass DubClean Voice output;
* ``D = M - P`` is diagnostic removed audio (not assumed to be a perfect
  physical English stem);
* a robust local STFT transfer estimates the part of ``D`` explained by
  ``R``;
* only short, high-confidence intervals with unexplained removed speech and
  missing near-end energy are eligible;
* restoration adds only the recoverable spectral residual, never the complete
  EN+RU input stem.

The detector is deliberately conservative.  Mathematically ambiguous cases,
unstable calibration and low-confidence candidates remain unchanged.
Sustained-vocal density is reported for the separate song-protection stage
without changing a word decision solely because preview scope is shorter.

Calibration blocks overlap: every block analyses ``analysis_overlap_sec``
extra audio on both sides but only owns candidates whose centre falls into
its own core span.  Without the overlap a word sitting on a chunk seam was
split into two weak halves and silently lost, and the double-talk context of
events near a seam could not be measured at all.

The applier adds a bounded correction only.  Besides the mask/gain/relative
RMS limits it enforces ``maximum_output_to_input_rms``: over the restored
event the protected stem may never become louder than the dubbed input stem
it came from, so no combination of thresholds can hand back the untouched
EN+RU mixture.
"""
from __future__ import annotations

import math
import os
import shutil
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.storage import atomic_json, utc_now


SCHEMA_VERSION = 3
REPORT_KIND = "dubclean_coincident_speech_protection"
ALGORITHM = "robust_stft_double_talk_v3"


def _defaults() -> dict[str, Any]:
    return {
        "enabled": True,
        "algorithm": ALGORITHM,
        "analysis_sample_rate": 16000,
        "analysis_chunk_sec": 45.0,
        "analysis_overlap_sec": 2.5,
        "frame_sec": 0.032,
        "hop_sec": 0.010,
        "minimum_frequency_hz": 90.0,
        "maximum_frequency_hz": 4200.0,
        "calibration_quantile": 0.75,
        "minimum_calibration_active_sec": 1.5,
        "minimum_calibration_similarity": 0.68,
        "maximum_calibration_gain_mad_db": 3.5,
        "expected_reference_guard_db": 2.0,
        "minimum_reference_rms_db": -46.0,
        "minimum_input_rms_db": -46.0,
        "minimum_removed_rms_db": -48.0,
        "minimum_context_post_rms_db": -48.0,
        "minimum_context_output_share": 0.04,
        "minimum_context_active_share": 0.30,
        "minimum_reference_active_share": 0.60,
        "minimum_removed_reference_similarity": 0.62,
        "minimum_pre_to_post_loss_db": 6.0,
        "minimum_estimated_near_end_share": 0.18,
        "minimum_missing_near_end_share": 0.58,
        "minimum_unexpected_removed_share": 0.22,
        "minimum_recoverable_pre_share": 0.12,
        "minimum_noncollinear_removed_share": 0.12,
        "minimum_speech_band_share": 0.65,
        "isolated_noncollinear_share": 0.24,
        "isolated_near_end_share": 0.32,
        "isolated_missing_share": 0.74,
        "isolated_unexpected_removed_share": 0.34,
        "minimum_interval_sec": 0.10,
        "maximum_interval_sec": 1.60,
        "maximum_merge_gap_sec": 0.08,
        "context_side_sec": 0.30,
        "boundary_padding_sec": 0.035,
        "crossfade_sec": 0.060,
        "minimum_confidence": 0.84,
        "isolated_minimum_confidence": 0.92,
        "target_recovery_fraction": 0.85,
        "maximum_recovery_gain": 0.90,
        "maximum_recovery_mask": 0.85,
        "maximum_recovery_relative_rms": 0.70,
        "maximum_output_to_input_rms": 1.0,
        "sustained_vocal_veto_sec": 2.20,
        "sustained_vocal_gap_sec": 0.32,
        "maximum_intervals_per_minute": 6,
        "maximum_restored_fraction": 0.02,
        "maximum_reported_rejections": 200,
    }


def resolved_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a complete and safely bounded public configuration."""
    defaults = _defaults()
    value = {**defaults, **(config or {})}
    for key, default in defaults.items():
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            continue
        try:
            finite = math.isfinite(float(value[key]))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            value[key] = default
    value["enabled"] = bool(value["enabled"])
    value["algorithm"] = ALGORITHM
    value["analysis_sample_rate"] = max(
        8000, min(24000, int(value["analysis_sample_rate"]))
    )
    value["analysis_chunk_sec"] = max(
        10.0, min(120.0, float(value["analysis_chunk_sec"]))
    )
    # The overlap has to hold a whole event plus both context sides, otherwise
    # a word on a chunk seam is still analysed with a truncated neighbourhood.
    value["analysis_overlap_sec"] = max(
        0.0,
        min(
            float(value["analysis_overlap_sec"]),
            value["analysis_chunk_sec"] * 0.4,
        ),
    )
    value["frame_sec"] = max(0.020, min(0.064, float(value["frame_sec"])))
    value["hop_sec"] = max(
        0.005, min(float(value["hop_sec"]), value["frame_sec"] * 0.75)
    )
    value["minimum_frequency_hz"] = max(
        40.0, float(value["minimum_frequency_hz"])
    )
    value["maximum_frequency_hz"] = max(
        value["minimum_frequency_hz"] + 200.0,
        min(
            float(value["maximum_frequency_hz"]),
            value["analysis_sample_rate"] * 0.49,
        ),
    )
    value["calibration_quantile"] = float(
        np.clip(value["calibration_quantile"], 0.50, 0.95)
    )
    for key in (
        "minimum_calibration_active_sec",
        "maximum_calibration_gain_mad_db",
        "expected_reference_guard_db",
        "minimum_reference_rms_db",
        "minimum_input_rms_db",
        "minimum_removed_rms_db",
        "minimum_context_post_rms_db",
        "minimum_pre_to_post_loss_db",
        "minimum_interval_sec",
        "maximum_interval_sec",
        "maximum_merge_gap_sec",
        "context_side_sec",
        "boundary_padding_sec",
        "crossfade_sec",
        "sustained_vocal_veto_sec",
        "sustained_vocal_gap_sec",
    ):
        value[key] = float(value[key])
    value["minimum_calibration_active_sec"] = max(
        0.1, value["minimum_calibration_active_sec"]
    )
    value["maximum_calibration_gain_mad_db"] = max(
        0.1, value["maximum_calibration_gain_mad_db"]
    )
    value["minimum_interval_sec"] = max(0.04, value["minimum_interval_sec"])
    value["maximum_interval_sec"] = max(
        value["minimum_interval_sec"], value["maximum_interval_sec"]
    )
    for key in (
        "maximum_merge_gap_sec",
        "context_side_sec",
        "boundary_padding_sec",
        "crossfade_sec",
        "sustained_vocal_veto_sec",
        "sustained_vocal_gap_sec",
    ):
        value[key] = max(0.0, value[key])
    for key in (
        "minimum_calibration_similarity",
        "minimum_context_output_share",
        "minimum_context_active_share",
        "minimum_reference_active_share",
        "minimum_removed_reference_similarity",
        "minimum_estimated_near_end_share",
        "minimum_missing_near_end_share",
        "minimum_unexpected_removed_share",
        "minimum_recoverable_pre_share",
        "minimum_noncollinear_removed_share",
        "minimum_speech_band_share",
        "isolated_noncollinear_share",
        "isolated_near_end_share",
        "isolated_missing_share",
        "isolated_unexpected_removed_share",
        "minimum_confidence",
        "isolated_minimum_confidence",
        "target_recovery_fraction",
        "maximum_recovery_gain",
        "maximum_recovery_mask",
        "maximum_recovery_relative_rms",
        "maximum_restored_fraction",
    ):
        value[key] = float(np.clip(value[key], 0.0, 1.0))
    value["isolated_minimum_confidence"] = max(
        value["minimum_confidence"], value["isolated_minimum_confidence"]
    )
    value["maximum_output_to_input_rms"] = float(
        np.clip(value["maximum_output_to_input_rms"], 0.05, 1.0)
    )
    value["maximum_intervals_per_minute"] = max(
        1, int(value["maximum_intervals_per_minute"])
    )
    value["maximum_reported_rejections"] = max(
        0, int(value["maximum_reported_rejections"])
    )
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


def validate_detection_report(
    report: Any,
    mixture_path: str | Path,
    reference_path: str | Path,
    processed_path: str | Path,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Validate that a cached plan belongs to the exact current DSP inputs.

    Cached reports are executable repair plans, not optional presentation
    metadata.  A truncated JSON document, an older schema or a structurally
    valid plan created for other audio must therefore trigger redetection
    instead of either crashing the build or being silently applied.
    """
    if not isinstance(report, dict):
        return False, "invalid_report_type"
    if report.get("schema_version") != SCHEMA_VERSION:
        return False, "schema_mismatch"
    if report.get("report_kind") != REPORT_KIND:
        return False, "report_kind_mismatch"
    if report.get("algorithm") != ALGORITHM:
        return False, "algorithm_mismatch"

    report_config = report.get("config")
    if not isinstance(report_config, dict):
        return False, "invalid_report_config"
    expected_config_source = (
        config if isinstance(config, dict) and config else report_config
    )
    try:
        expected_config = resolved_config(expected_config_source)
        cached_config = resolved_config(report_config)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, "invalid_report_config"
    if cached_config != expected_config:
        return False, "config_mismatch"
    if bool(report.get("enabled")) != bool(expected_config["enabled"]):
        return False, "enabled_state_mismatch"

    sources = report.get("sources")
    if not isinstance(sources, dict):
        return False, "invalid_report_sources"
    expected_sources = {
        "mixture_before_subtraction": _audio_identity(
            Path(mixture_path).resolve()
        ),
        "english_reference": _audio_identity(Path(reference_path).resolve()),
        "first_pass_result": _audio_identity(Path(processed_path).resolve()),
    }
    for key, expected in expected_sources.items():
        if sources.get(key) != expected:
            return False, f"source_mismatch:{key}"

    raw_intervals = report.get("confirmed_intervals")
    raw_blocks = report.get("calibration_blocks")
    raw_frequencies = report.get("frequency_bins_hz")
    if not isinstance(raw_intervals, list):
        return False, "invalid_confirmed_intervals"
    if not isinstance(raw_blocks, list):
        return False, "invalid_calibration_blocks"
    if not isinstance(raw_frequencies, list):
        return False, "invalid_frequency_axis"

    if not expected_config["enabled"]:
        if raw_intervals:
            return False, "disabled_report_has_intervals"
        return True, "valid"

    try:
        frequencies = np.asarray(raw_frequencies, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid_frequency_axis"
    if (
        frequencies.ndim != 1
        or frequencies.size < 2
        or not np.all(np.isfinite(frequencies))
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        return False, "invalid_frequency_axis"

    duration = min(
        float(item["duration_sec"]) for item in expected_sources.values()
    )
    raw_scopes = report.get("analysis_scopes")
    if not isinstance(raw_scopes, list):
        return False, "invalid_analysis_scopes"
    scopes_by_index: dict[int, tuple[float, float]] = {}
    for scope in raw_scopes:
        if not isinstance(scope, dict):
            return False, "invalid_analysis_scope"
        try:
            scope_index = int(scope["scope_index"])
            scope_start = float(scope["start_sec"])
            scope_end = float(scope["end_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "invalid_analysis_scope"
        if (
            scope_index in scopes_by_index
            or not math.isfinite(scope_start)
            or not math.isfinite(scope_end)
            or scope_start < 0.0
            or scope_end <= scope_start
            or scope_end > duration + 0.05
        ):
            return False, "invalid_analysis_scope"
        scopes_by_index[scope_index] = (scope_start, scope_end)

    stable_indices: set[int] = set()
    stable_blocks: dict[int, dict[str, Any]] = {}
    seen_indices: set[int] = set()
    for block in raw_blocks:
        if not isinstance(block, dict):
            return False, "invalid_calibration_block"
        try:
            block_index = int(block["index"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "invalid_calibration_block"
        if block_index in seen_indices:
            return False, "duplicate_calibration_block"
        seen_indices.add(block_index)
        try:
            block_scope_index = int(block["scope_index"])
            block_start = float(block["start_sec"])
            block_end = float(block["end_sec"])
            core_start = float(block["core_start_sec"])
            core_end = float(block["core_end_sec"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "invalid_calibration_block"
        scope_bounds = scopes_by_index.get(block_scope_index)
        if (
            scope_bounds is None
            or not all(
                math.isfinite(value)
                for value in (block_start, block_end, core_start, core_end)
            )
            or block_start < scope_bounds[0] - 0.05
            or block_end > scope_bounds[1] + 0.05
            or block_end <= block_start
            or core_start < block_start - 0.05
            or core_end > block_end + 0.05
            or core_end <= core_start
        ):
            return False, "invalid_calibration_block"
        if not block.get("stable"):
            continue
        try:
            transfer = np.asarray(block.get("transfer_power"), dtype=np.float64)
            transfer_real = np.asarray(
                block.get("transfer_real"), dtype=np.float64
            )
            transfer_imag = np.asarray(
                block.get("transfer_imag"), dtype=np.float64
            )
        except (TypeError, ValueError, OverflowError):
            return False, "invalid_calibration_transfer"
        if (
            transfer.ndim != 1
            or transfer.size != frequencies.size
            or not np.all(np.isfinite(transfer))
            or np.any(transfer <= 0.0)
            or transfer_real.ndim != 1
            or transfer_imag.ndim != 1
            or transfer_real.size != frequencies.size
            or transfer_imag.size != frequencies.size
            or not np.all(np.isfinite(transfer_real))
            or not np.all(np.isfinite(transfer_imag))
        ):
            return False, "invalid_calibration_transfer"
        stable_indices.add(block_index)
        stable_blocks[block_index] = block

    for item in raw_intervals:
        if not isinstance(item, dict):
            return False, "invalid_confirmed_interval"
        try:
            start = float(item["start_sec"])
            end = float(item["end_sec"])
            detected_start = float(item["detected_start_sec"])
            detected_end = float(item["detected_end_sec"])
            confidence = float(item["confidence"])
            block_index = int(item["calibration_block_index"])
            scope_index = int(item["scope_index"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "invalid_confirmed_interval"
        scope_bounds = scopes_by_index.get(scope_index)
        block = stable_blocks.get(block_index)
        maximum_plan_duration = (
            expected_config["maximum_interval_sec"]
            + 2.0 * expected_config["boundary_padding_sec"]
            + max(0.02, expected_config["frame_sec"])
        )
        if (
            not all(
                math.isfinite(value)
                for value in (
                    start,
                    end,
                    detected_start,
                    detected_end,
                    confidence,
                )
            )
            or start < 0.0
            or end <= start
            or end > duration + 0.05
            or end - start > maximum_plan_duration + 1e-6
            or detected_start < start - 0.05
            or detected_end > end + 0.05
            or detected_end <= detected_start
            or not 0.0 <= confidence <= 1.0
            or block_index not in stable_indices
            or scope_bounds is None
            or block is None
            or int(block["scope_index"]) != scope_index
            or start < scope_bounds[0] - 0.05
            or end > scope_bounds[1] + 0.05
            or start < float(block["start_sec"]) - 0.05
            or end > float(block["end_sec"]) + 0.05
            or not (
                float(block["core_start_sec"]) - 0.05
                <= 0.5 * (detected_start + detected_end)
                <= float(block["core_end_sec"]) + 0.05
            )
        ):
            return False, "invalid_confirmed_interval"
    return True, "valid"


def _safe_db(power: np.ndarray | float) -> np.ndarray | float:
    result = 10.0 * np.log10(np.maximum(np.asarray(power), 1e-12))
    return float(result) if result.ndim == 0 else result


def _normalise_intervals(
    intervals: Iterable[tuple[float, float] | dict[str, Any]] | None,
    duration: float,
) -> list[dict[str, Any]]:
    if intervals is None:
        return [{"scope_index": 0, "start_sec": 0.0, "end_sec": duration}]
    result: list[dict[str, Any]] = []
    for index, item in enumerate(intervals):
        if isinstance(item, dict):
            start = float(item.get("start_sec") or 0.0)
            end = float(item.get("end_sec") or start)
            scope_index = int(item.get("scope_index", index))
        else:
            start, end = (float(item[0]), float(item[1]))
            scope_index = index
        start = max(0.0, min(duration, start))
        end = max(start, min(duration, end))
        if end - start >= 0.05:
            result.append(
                {
                    "scope_index": scope_index,
                    "start_sec": start,
                    "end_sec": end,
                }
            )
    return result


def _read_audio_segment(
    path: Path,
    start_sec: float,
    end_sec: float,
    target_rate: int,
) -> np.ndarray:
    target_frames = max(0, int(round((end_sec - start_sec) * target_rate)))
    if target_frames == 0:
        return np.zeros((0, 1), dtype=np.float32)
    with sf.SoundFile(str(path)) as handle:
        source_rate = int(handle.samplerate)
        first = max(0, int(math.floor(start_sec * source_rate)))
        fractional = start_sec - first / source_rate
        needed = int(
            math.ceil((end_sec - start_sec + fractional) * source_rate)
        ) + 4
        handle.seek(min(first, len(handle)))
        values = handle.read(needed, dtype="float32", always_2d=True)
    if source_rate != target_rate:
        values = audio_io.resample(values, source_rate, target_rate)
    offset = max(0, int(round(fractional * target_rate)))
    values = values[offset : offset + target_frames]
    if values.shape[0] < target_frames:
        values = np.pad(
            values,
            ((0, target_frames - values.shape[0]), (0, 0)),
        )
    return np.ascontiguousarray(
        np.nan_to_num(
            values[:target_frames],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        dtype=np.float32,
    )


def _read_mono_segment(
    path: Path,
    start_sec: float,
    end_sec: float,
    target_rate: int,
) -> np.ndarray:
    return np.ascontiguousarray(
        audio_io.to_mono(
            _read_audio_segment(path, start_sec, end_sec, target_rate)
        ),
        dtype=np.float32,
    )


def _stft(
    values: np.ndarray,
    rate: int,
    frame_sec: float,
    hop_sec: float,
    *,
    reconstruction: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    nperseg = max(64, int(round(rate * frame_sec)))
    hop = max(16, int(round(rate * hop_sec)))
    hop = min(hop, nperseg - 1)
    nfft = 1 << int(math.ceil(math.log2(nperseg)))
    frequencies, times, spectrum = signal.stft(
        np.asarray(values, dtype=np.float32),
        fs=rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=nfft,
        boundary="zeros" if reconstruction else None,
        padded=bool(reconstruction),
    )
    return frequencies, times, spectrum, nperseg, hop


def _magnitude_similarity(
    left: np.ndarray, right: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    numerator = np.sum(np.abs(left) * np.abs(right), axis=0)
    denominator = np.sqrt(
        np.sum(np.abs(left) ** 2, axis=0)
        * np.sum(np.abs(right) ** 2, axis=0)
    )
    return np.clip(numerator / np.maximum(denominator, eps), 0.0, 1.0)


def _weighted_margin(
    values: dict[str, float],
    cfg: dict[str, Any],
) -> float:
    strong = {
        "loss": 16.0,
        "near": 0.55,
        "missing": 0.94,
        "unexpected": 0.62,
        "recoverable": 0.42,
        "noncollinear": 0.55,
        "similarity": 0.92,
    }
    minimum = {
        "loss": cfg["minimum_pre_to_post_loss_db"],
        "near": cfg["minimum_estimated_near_end_share"],
        "missing": cfg["minimum_missing_near_end_share"],
        "unexpected": cfg["minimum_unexpected_removed_share"],
        "recoverable": cfg["minimum_recoverable_pre_share"],
        "noncollinear": cfg["minimum_noncollinear_removed_share"],
        "similarity": cfg["minimum_removed_reference_similarity"],
    }
    weights = {
        "loss": 0.12,
        "near": 0.14,
        "missing": 0.16,
        "unexpected": 0.18,
        "recoverable": 0.18,
        "noncollinear": 0.14,
        "similarity": 0.08,
    }
    margin = 0.0
    for key, weight in weights.items():
        span = max(1e-6, strong[key] - minimum[key])
        margin += weight * float(
            np.clip((values[key] - minimum[key]) / span, 0.0, 1.0)
        )
    return float(np.clip(0.55 + 0.45 * margin, 0.0, 1.0))


def _candidate_groups(
    times: np.ndarray,
    mask: np.ndarray,
    hop_sec: float,
    maximum_gap_sec: float,
) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    allowed = hop_sec * 1.6 + maximum_gap_sec
    for index in indices[1:]:
        if float(times[index] - times[groups[-1][-1]]) <= allowed:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [np.asarray(group, dtype=np.int32) for group in groups]


def _analyse_block(
    mixture_path: Path,
    reference_path: Path,
    processed_path: Path,
    start_sec: float,
    end_sec: float,
    core_start_sec: float,
    core_end_sec: float,
    scope_index: int,
    block_index: int,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rate = int(cfg["analysis_sample_rate"])
    mixture = _read_mono_segment(mixture_path, start_sec, end_sec, rate)
    reference = _read_mono_segment(reference_path, start_sec, end_sec, rate)
    processed = _read_mono_segment(processed_path, start_sec, end_sec, rate)
    length = min(mixture.size, reference.size, processed.size)
    mixture = mixture[:length]
    reference = reference[:length]
    processed = processed[:length]
    minimum_frames = max(64, int(round(rate * cfg["frame_sec"])))
    empty_block = {
        "index": block_index,
        "scope_index": scope_index,
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "core_start_sec": round(core_start_sec, 6),
        "core_end_sec": round(min(core_end_sec, end_sec), 6),
        "stable": False,
        "reason": "insufficient_audio",
        "calibration_active_sec": 0.0,
        "reference_gain_db": 0.0,
        "reference_gain_mad_db": 0.0,
        "transfer_power": [],
        "transfer_real": [],
        "transfer_imag": [],
    }
    if length < minimum_frames:
        return empty_block, []

    frequencies, times, mixture_stft, _, hop_frames = _stft(
        mixture, rate, cfg["frame_sec"], cfg["hop_sec"]
    )
    _, _, reference_stft, _, _ = _stft(
        reference, rate, cfg["frame_sec"], cfg["hop_sec"]
    )
    _, _, processed_stft, _, _ = _stft(
        processed, rate, cfg["frame_sec"], cfg["hop_sec"]
    )
    frames = min(
        mixture_stft.shape[1],
        reference_stft.shape[1],
        processed_stft.shape[1],
    )
    if frames == 0:
        return empty_block, []
    mixture_stft = mixture_stft[:, :frames]
    reference_stft = reference_stft[:, :frames]
    processed_stft = processed_stft[:, :frames]
    times = times[:frames]
    removed_stft = mixture_stft - processed_stft
    full_band = frequencies >= max(40.0, cfg["minimum_frequency_hz"] * 0.5)
    speech_band = (
        (frequencies >= cfg["minimum_frequency_hz"])
        & (frequencies <= cfg["maximum_frequency_hz"])
    )
    if not np.any(speech_band):
        return {**empty_block, "reason": "empty_speech_band"}, []

    eps = 1e-12
    mixture_power_bins = np.abs(mixture_stft[speech_band]) ** 2
    reference_power_bins = np.abs(reference_stft[speech_band]) ** 2
    processed_power_bins = np.abs(processed_stft[speech_band]) ** 2
    removed_power_bins = np.abs(removed_stft[speech_band]) ** 2
    mixture_power = np.sum(mixture_power_bins, axis=0)
    reference_power = np.sum(reference_power_bins, axis=0)
    processed_power = np.sum(processed_power_bins, axis=0)
    removed_power = np.sum(removed_power_bins, axis=0)
    full_removed_power = np.sum(np.abs(removed_stft[full_band]) ** 2, axis=0)
    input_db = _safe_db(mixture_power)
    reference_db = _safe_db(reference_power)
    processed_db = _safe_db(processed_power)
    removed_db = _safe_db(removed_power)
    removed_similarity = _magnitude_similarity(
        removed_stft[speech_band],
        reference_stft[speech_band],
    )
    calibration_mask = (
        (reference_db >= cfg["minimum_reference_rms_db"])
        & (input_db >= cfg["minimum_input_rms_db"])
        & (removed_db >= cfg["minimum_removed_rms_db"])
        & (removed_similarity >= cfg["minimum_calibration_similarity"])
    )
    hop_sec = hop_frames / rate
    calibration_active_sec = float(np.sum(calibration_mask) * hop_sec)
    gain_db = _safe_db(
        removed_power[calibration_mask]
        / np.maximum(reference_power[calibration_mask], eps)
    )
    gain_median = (
        float(np.median(gain_db)) if np.asarray(gain_db).size else 0.0
    )
    gain_mad = (
        float(np.median(np.abs(gain_db - gain_median)))
        if np.asarray(gain_db).size
        else float("inf")
    )
    stable = bool(
        calibration_active_sec >= cfg["minimum_calibration_active_sec"]
        and gain_mad <= cfg["maximum_calibration_gain_mad_db"]
    )
    block: dict[str, Any] = {
        "index": block_index,
        "scope_index": scope_index,
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "core_start_sec": round(core_start_sec, 6),
        "core_end_sec": round(min(core_end_sec, end_sec), 6),
        "stable": stable,
        "reason": (
            "stable"
            if stable
            else (
                "insufficient_calibration"
                if calibration_active_sec < cfg["minimum_calibration_active_sec"]
                else "unstable_reference_gain"
            )
        ),
        "calibration_active_sec": round(calibration_active_sec, 4),
        "reference_gain_db": round(gain_median, 4),
        "reference_gain_mad_db": (
            round(gain_mad, 4) if math.isfinite(gain_mad) else None
        ),
        "transfer_power": [],
        "transfer_real": [],
        "transfer_imag": [],
    }
    if not stable:
        return block, []

    ratios = removed_power_bins / np.maximum(reference_power_bins, eps)
    # Very quiet reference bins cannot identify a transfer value.
    reference_floor = np.maximum(
        np.sum(reference_power_bins, axis=0, keepdims=True) * 1e-5,
        eps,
    )
    valid_ratio = (
        calibration_mask[None, :]
        & (reference_power_bins >= reference_floor)
    )
    ratios = np.where(valid_ratio, ratios, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        transfer_power = np.nanquantile(
            ratios, cfg["calibration_quantile"], axis=1
        )
    fallback = float(10.0 ** (gain_median / 10.0))
    transfer_power = np.nan_to_num(
        transfer_power,
        nan=fallback,
        posinf=fallback,
        neginf=fallback,
    )
    transfer_power = np.clip(transfer_power, fallback * 0.02, fallback * 50.0)
    if transfer_power.size >= 5:
        transfer_power = signal.medfilt(transfer_power, kernel_size=5)
        transfer_power = np.maximum(transfer_power, fallback * 0.02)

    complex_ratios = (
        removed_stft[speech_band]
        * np.conj(reference_stft[speech_band])
        / np.maximum(reference_power_bins, eps)
    )
    complex_ratios = np.where(
        valid_ratio,
        complex_ratios,
        np.nan + 1j * np.nan,
    )
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_complex = np.nanmedian(
            np.real(complex_ratios), axis=1
        ) + 1j * np.nanmedian(np.imag(complex_ratios), axis=1)
    valid_phase = (
        np.isfinite(np.real(median_complex))
        & np.isfinite(np.imag(median_complex))
        & (np.abs(median_complex) > 1e-9)
    )
    if np.sum(valid_phase) < 2:
        block["stable"] = False
        block["reason"] = "unstable_complex_transfer"
        return block, []
    bin_indices = np.arange(transfer_power.size, dtype=np.float64)
    unwrapped_phase = np.unwrap(np.angle(median_complex[valid_phase]))
    transfer_phase = np.interp(
        bin_indices,
        bin_indices[valid_phase],
        unwrapped_phase,
    )
    complex_transfer = np.sqrt(transfer_power) * np.exp(
        1j * transfer_phase
    )
    block["transfer_power"] = [
        round(float(value), 9) for value in transfer_power
    ]
    block["transfer_real"] = [
        round(float(value), 9) for value in np.real(complex_transfer)
    ]
    block["transfer_imag"] = [
        round(float(value), 9) for value in np.imag(complex_transfer)
    ]

    guard = float(10.0 ** (cfg["expected_reference_guard_db"] / 10.0))
    expected_reference = transfer_power[:, None] * reference_power_bins * guard
    estimated_near_end = np.maximum(mixture_power_bins - expected_reference, 0.0)
    missing_near_end = np.maximum(estimated_near_end - processed_power_bins, 0.0)
    unexpected_removed = np.maximum(removed_power_bins - expected_reference, 0.0)
    recoverable = np.minimum(missing_near_end, unexpected_removed)
    estimated_near_end_share = np.sum(estimated_near_end, axis=0) / np.maximum(
        mixture_power, eps
    )
    missing_near_end_share = np.sum(missing_near_end, axis=0) / np.maximum(
        np.sum(estimated_near_end, axis=0), eps
    )
    unexpected_removed_share = np.sum(unexpected_removed, axis=0) / np.maximum(
        removed_power, eps
    )
    recoverable_pre_share = np.sum(recoverable, axis=0) / np.maximum(
        mixture_power, eps
    )
    cross = np.sum(
        removed_stft[speech_band]
        * np.conj(reference_stft[speech_band]),
        axis=0,
    )
    removed_coherence = np.abs(cross) ** 2 / np.maximum(
        removed_power * reference_power, eps
    )
    noncollinear_removed_share = np.clip(1.0 - removed_coherence, 0.0, 1.0)
    speech_band_share = removed_power / np.maximum(full_removed_power, eps)
    loss_db = _safe_db(mixture_power / np.maximum(processed_power, eps))
    output_share = processed_power / np.maximum(mixture_power, eps)

    # Keep a deliberately wider probe set for the audit report.  The strict
    # gates are applied to each merged event below; this lets a gain-only
    # English event appear as an explicit rejected candidate instead of
    # silently disappearing from diagnostics.
    frame_mask = (
        (reference_db >= cfg["minimum_reference_rms_db"])
        & (input_db >= cfg["minimum_input_rms_db"])
        & (removed_db >= cfg["minimum_removed_rms_db"])
        & (
            removed_similarity
            >= cfg["minimum_removed_reference_similarity"] * 0.85
        )
        & (loss_db >= cfg["minimum_pre_to_post_loss_db"])
        & (
            estimated_near_end_share
            >= cfg["minimum_estimated_near_end_share"] * 0.60
        )
        & (
            missing_near_end_share
            >= cfg["minimum_missing_near_end_share"] * 0.75
        )
        & (
            unexpected_removed_share
            >= cfg["minimum_unexpected_removed_share"] * 0.60
        )
        & (
            recoverable_pre_share
            >= cfg["minimum_recoverable_pre_share"] * 0.50
        )
        & (
            speech_band_share
            >= cfg["minimum_speech_band_share"] * 0.85
        )
    )
    absolute_times = start_sec + times
    groups = _candidate_groups(
        absolute_times,
        frame_mask,
        hop_sec,
        cfg["maximum_merge_gap_sec"],
    )
    candidates: list[dict[str, Any]] = []
    for group in groups:
        raw_start = max(start_sec, float(absolute_times[group[0]] - cfg["frame_sec"] / 2))
        raw_end = min(end_sec, float(absolute_times[group[-1]] + cfg["frame_sec"] / 2))
        # Neighbouring blocks overlap, so exactly one of them owns each event:
        # the block whose core span contains the event centre.  Otherwise the
        # same word would be restored twice, or split across the seam.
        centre = 0.5 * (raw_start + raw_end)
        if not core_start_sec <= centre < core_end_sec:
            continue
        support_sec = float(group.size * hop_sec)
        duration_sec = raw_end - raw_start
        # Double talk means the English reference is speaking *through* the
        # event.  Measuring this over the group frames alone is meaningless -
        # they are selected by a mask that already requires an active
        # reference - so it is measured over the whole detected span, gaps
        # included.
        interval_frames = (absolute_times >= raw_start) & (
            absolute_times <= raw_end
        )
        reference_active_share = (
            float(
                np.mean(
                    reference_db[interval_frames]
                    >= cfg["minimum_reference_rms_db"]
                )
            )
            if np.any(interval_frames)
            else 0.0
        )
        aggregate = {
            "loss": float(np.quantile(loss_db[group], 0.60)),
            "near": float(np.quantile(estimated_near_end_share[group], 0.60)),
            "missing": float(np.quantile(missing_near_end_share[group], 0.60)),
            "unexpected": float(
                np.quantile(unexpected_removed_share[group], 0.60)
            ),
            "recoverable": float(
                np.quantile(recoverable_pre_share[group], 0.60)
            ),
            "noncollinear": float(
                np.quantile(noncollinear_removed_share[group], 0.60)
            ),
            "similarity": float(np.quantile(removed_similarity[group], 0.40)),
        }
        confidence = _weighted_margin(aggregate, cfg)
        left_context = (
            (absolute_times >= raw_start - cfg["context_side_sec"])
            & (absolute_times < raw_start)
        )
        right_context = (
            (absolute_times > raw_end)
            & (absolute_times <= raw_end + cfg["context_side_sec"])
        )

        def context_active(context_mask: np.ndarray) -> tuple[bool, float]:
            if not np.any(context_mask):
                return False, 0.0
            active = (
                (processed_db[context_mask] >= cfg["minimum_context_post_rms_db"])
                & (
                    output_share[context_mask]
                    >= cfg["minimum_context_output_share"]
                )
            )
            share = float(np.mean(active))
            return share >= cfg["minimum_context_active_share"], share

        left_active, left_share = context_active(left_context)
        right_active, right_share = context_active(right_context)
        contextual = bool(left_active and right_active)
        isolated_strong = bool(
            aggregate["noncollinear"] >= cfg["isolated_noncollinear_share"]
            and aggregate["near"] >= cfg["isolated_near_end_share"]
            and aggregate["missing"] >= cfg["isolated_missing_share"]
            and aggregate["unexpected"]
            >= cfg["isolated_unexpected_removed_share"]
        )
        reasons: list[str] = []
        decision = "confirmed"
        strict_feature_gates = (
            (
                aggregate["similarity"]
                >= cfg["minimum_removed_reference_similarity"],
                "removed_not_reference_related",
            ),
            (
                aggregate["near"]
                >= cfg["minimum_estimated_near_end_share"],
                "insufficient_near_end_evidence",
            ),
            (
                aggregate["missing"]
                >= cfg["minimum_missing_near_end_share"],
                "near_end_not_missing_after_processing",
            ),
            (
                aggregate["unexpected"]
                >= cfg["minimum_unexpected_removed_share"],
                "removed_energy_explained_by_reference",
            ),
            (
                aggregate["recoverable"]
                >= cfg["minimum_recoverable_pre_share"],
                "recoverable_component_too_quiet",
            ),
            (
                aggregate["noncollinear"]
                >= cfg["minimum_noncollinear_removed_share"],
                "reference_only_explains_removed",
            ),
            (
                reference_active_share
                >= cfg["minimum_reference_active_share"],
                "reference_not_active_during_event",
            ),
        )
        for passed, reason in strict_feature_gates:
            if not passed:
                decision = "rejected"
                reasons.append(reason)
        if support_sec < cfg["minimum_interval_sec"]:
            decision = "rejected"
            reasons.append("too_short")
        if duration_sec > cfg["maximum_interval_sec"]:
            decision = "rejected"
            reasons.append("too_long")
        if not contextual and not isolated_strong:
            decision = "rejected"
            reasons.append("insufficient_double_talk_context")
        required_confidence = (
            cfg["minimum_confidence"]
            if contextual
            else cfg["isolated_minimum_confidence"]
        )
        if confidence < required_confidence:
            decision = "rejected"
            reasons.append("low_confidence")
        if decision == "confirmed":
            reasons.extend(
                [
                    "short_deep_output_loss",
                    "unexplained_removed_speech",
                    (
                        "active_translation_context"
                        if contextual
                        else "strong_isolated_double_talk"
                    ),
                ]
            )
        padded_start = max(
            start_sec, raw_start - cfg["boundary_padding_sec"]
        )
        padded_end = min(end_sec, raw_end + cfg["boundary_padding_sec"])
        candidates.append(
            {
                "scope_index": scope_index,
                "calibration_block_index": block_index,
                "start_sec": round(padded_start, 6),
                "end_sec": round(padded_end, 6),
                "detected_start_sec": round(raw_start, 6),
                "detected_end_sec": round(raw_end, 6),
                "duration_sec": round(padded_end - padded_start, 6),
                "support_sec": round(support_sec, 6),
                "confidence": round(confidence, 6),
                "decision": decision,
                "reasons": reasons,
                "context": {
                    "left_active": left_active,
                    "right_active": right_active,
                    "left_active_share": round(left_share, 6),
                    "right_active_share": round(right_share, 6),
                    "mode": "contextual" if contextual else "isolated",
                },
                "features": {
                    "pre_to_post_loss_db": round(aggregate["loss"], 5),
                    "estimated_near_end_share": round(aggregate["near"], 6),
                    "missing_near_end_share": round(aggregate["missing"], 6),
                    "unexpected_removed_share": round(
                        aggregate["unexpected"], 6
                    ),
                    "recoverable_pre_share": round(
                        aggregate["recoverable"], 6
                    ),
                    "noncollinear_removed_share": round(
                        aggregate["noncollinear"], 6
                    ),
                    "removed_reference_similarity": round(
                        aggregate["similarity"], 6
                    ),
                    "reference_active_share": round(
                        reference_active_share, 6
                    ),
                    "speech_band_share": round(
                        float(np.quantile(speech_band_share[group], 0.40)),
                        6,
                    ),
                },
            }
        )
    return block, candidates


_CONFIRMATION_REASONS = frozenset(
    {
        "short_deep_output_loss",
        "unexplained_removed_speech",
        "active_translation_context",
        "strong_isolated_double_talk",
    }
)


def _veto(item: dict[str, Any], reason: str) -> None:
    """Reject a provisionally confirmed candidate, keeping its diagnostics."""
    item["decision"] = "rejected"
    item["reasons"] = [
        value
        for value in item.get("reasons") or []
        if value not in _CONFIRMATION_REASONS
    ]
    item["reasons"].append(reason)


def _apply_global_vetoes(
    candidates: list[dict[str, Any]],
    scopes: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    provisional = [
        item for item in candidates if item.get("decision") == "confirmed"
    ]
    sustained_cluster_count = 0
    for scope in scopes:
        scoped = sorted(
            (
                item
                for item in provisional
                if item["scope_index"] == scope["scope_index"]
            ),
            key=lambda item: item["start_sec"],
        )
        clusters: list[list[dict[str, Any]]] = []
        for item in scoped:
            if (
                clusters
                and item["start_sec"] - clusters[-1][-1]["end_sec"]
                <= cfg["sustained_vocal_gap_sec"]
            ):
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster in clusters:
            span = cluster[-1]["end_sec"] - cluster[0]["start_sec"]
            if span < cfg["sustained_vocal_veto_sec"]:
                continue
            sustained_cluster_count += 1

    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in provisional:
        key = (
            int(item["scope_index"]),
            int(float(item["start_sec"]) // 60.0),
        )
        buckets.setdefault(key, []).append(item)
    limit = cfg["maximum_intervals_per_minute"]
    dense_buckets = sum(1 for bucket in buckets.values() if len(bucket) > limit)
    density_excess = sum(
        max(0, len(bucket) - limit) for bucket in buckets.values()
    )

    # Overlap ownership normally makes this impossible, but floating-point
    # centres right on a core seam can still duplicate one detected event.
    kept: list[dict[str, Any]] = []
    for item in sorted(
        (
            item for item in candidates if item.get("decision") == "confirmed"
        ),
        key=lambda item: (-item["confidence"], item["start_sec"]),
    ):
        if any(
            other["scope_index"] == item["scope_index"]
            and other["calibration_block_index"]
            != item["calibration_block_index"]
            and item["detected_start_sec"] < other["detected_end_sec"]
            and other["detected_start_sec"] < item["detected_end_sec"]
            for other in kept
        ):
            _veto(item, "overlapping_candidate")
            continue
        kept.append(item)

    confirmed = [
        item for item in candidates if item.get("decision") == "confirmed"
    ]
    analysed_duration = sum(
        scope["end_sec"] - scope["start_sec"] for scope in scopes
    )
    budget = analysed_duration * cfg["maximum_restored_fraction"]
    confirmed_duration = sum(
        float(item["end_sec"] - item["start_sec"]) for item in confirmed
    )
    # Scope-dependent limits used to make the same event pass in a full-film
    # scan but fail when only preview scenes were analysed.  Keep them as audit
    # diagnostics only.  Song restoration remains the responsibility of the
    # dedicated song-protection stage, which sees the musical background.
    return {
        "sustained_vocal_threshold_exceeded": bool(
            sustained_cluster_count
        ),
        "sustained_vocal_cluster_count": int(sustained_cluster_count),
        "candidate_density_threshold_exceeded": bool(dense_buckets),
        "dense_minute_bucket_count": int(dense_buckets),
        "candidate_density_excess_count": int(density_excess),
        "restoration_duration_threshold_exceeded": bool(
            confirmed_duration > budget and confirmed
        ),
        "restoration_duration_budget_sec": round(float(budget), 6),
        "confirmed_duration_before_advisory_sec": round(
            float(confirmed_duration), 6
        ),
    }


def detect_coincident_speech(
    mixture_path: str | Path,
    reference_path: str | Path,
    processed_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    analysis_intervals: Iterable[
        tuple[float, float] | dict[str, Any]
    ]
    | None = None,
    report_path: str | Path | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Detect high-confidence short over-subtraction intervals."""
    mixture = Path(mixture_path).resolve()
    reference = Path(reference_path).resolve()
    processed = Path(processed_path).resolve()
    cfg = resolved_config(config)
    sources = {
        "mixture_before_subtraction": _audio_identity(mixture),
        "english_reference": _audio_identity(reference),
        "first_pass_result": _audio_identity(processed),
    }
    duration = min(
        float(sources[key]["duration_sec"])
        for key in sources
    )
    scopes = _normalise_intervals(analysis_intervals, duration)
    if not cfg["enabled"]:
        report = {
            "schema_version": SCHEMA_VERSION,
            "report_kind": REPORT_KIND,
            "algorithm": ALGORITHM,
            "created_at": utc_now(),
            "enabled": False,
            "config": cfg,
            "sources": sources,
            "analysis_scopes": scopes,
            "frequency_bins_hz": [],
            "calibration_blocks": [],
            "confirmed_intervals": [],
            "rejected_candidates": [],
            "summary": {
                "decision": "disabled",
                "modified": False,
                "confirmed_count": 0,
                "rejected_count": 0,
                "analysed_duration_sec": round(
                    sum(item["end_sec"] - item["start_sec"] for item in scopes),
                    6,
                ),
            },
        }
        if report_path is not None:
            atomic_json(Path(report_path), report)
        return report

    blocks: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    chunk_sec = cfg["analysis_chunk_sec"]
    block_index = 0
    total_duration = max(
        1e-6,
        sum(item["end_sec"] - item["start_sec"] for item in scopes),
    )
    completed = 0.0
    overlap_sec = float(cfg["analysis_overlap_sec"])
    for scope in scopes:
        scope_start = float(scope["start_sec"])
        scope_end = float(scope["end_sec"])
        cursor = scope_start
        while cursor < scope_end - 1e-6:
            end = min(scope_end, cursor + chunk_sec)
            block, block_candidates = _analyse_block(
                mixture,
                reference,
                processed,
                max(scope_start, cursor - overlap_sec),
                min(scope_end, end + overlap_sec),
                cursor,
                # The final core has to own everything up to the scope end,
                # including an event whose centre lands exactly on it.
                end if end < scope_end else scope_end + 1.0,
                int(scope["scope_index"]),
                block_index,
                cfg,
            )
            blocks.append(block)
            candidates.extend(block_candidates)
            block_index += 1
            completed += end - cursor
            cursor = end
            if ctx is not None:
                ctx.update(
                    stage="Проверка сохранности совпадающих слов",
                    substage="Анализ речевых фрагментов",
                    progress=min(90.0, completed / total_duration * 90.0),
                    local_progress=min(
                        90.0, completed / total_duration * 90.0
                    ),
                )
                if hasattr(ctx, "check_stop"):
                    ctx.check_stop()

    safety_advisories = _apply_global_vetoes(candidates, scopes, cfg)
    confirmed = sorted(
        (
            item for item in candidates if item.get("decision") == "confirmed"
        ),
        key=lambda item: item["start_sec"],
    )
    rejected = sorted(
        (
            item for item in candidates if item.get("decision") != "confirmed"
        ),
        key=lambda item: (-item["confidence"], item["start_sec"]),
    )
    rejection_counts = Counter(
        reason
        for item in rejected
        for reason in item.get("reasons") or ["unspecified"]
    )
    nfft = 1 << int(
        math.ceil(
            math.log2(
                max(64, int(round(cfg["analysis_sample_rate"] * cfg["frame_sec"])))
            )
        )
    )
    frequencies = np.fft.rfftfreq(nfft, 1.0 / cfg["analysis_sample_rate"])
    speech_frequencies = frequencies[
        (frequencies >= cfg["minimum_frequency_hz"])
        & (frequencies <= cfg["maximum_frequency_hz"])
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "algorithm": ALGORITHM,
        "created_at": utc_now(),
        "enabled": True,
        "config": cfg,
        "sources": sources,
        "analysis_scopes": scopes,
        "frequency_bins_hz": [
            round(float(value), 6) for value in speech_frequencies
        ],
        "calibration_blocks": blocks,
        "confirmed_intervals": confirmed,
        "rejected_candidates": rejected[
            : cfg["maximum_reported_rejections"]
        ],
        "summary": {
            "decision": (
                "confirmed_intervals_found" if confirmed else "no_safe_repair"
            ),
            "modified": False,
            "confirmed_count": len(confirmed),
            "rejected_count": len(rejected),
            "rejected_omitted_count": max(
                0, len(rejected) - cfg["maximum_reported_rejections"]
            ),
            "rejection_reasons": dict(sorted(rejection_counts.items())),
            "stable_calibration_blocks": sum(
                1 for block in blocks if block.get("stable")
            ),
            "calibration_block_count": len(blocks),
            "analysed_duration_sec": round(total_duration, 6),
            "confirmed_duration_sec": round(
                sum(item["end_sec"] - item["start_sec"] for item in confirmed),
                6,
            ),
            "low_confidence_policy": "leave_processed_audio_unchanged",
            "safety_advisories": safety_advisories,
        },
    }
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


def _recovery_waveform(
    mixture: np.ndarray,
    reference: np.ndarray,
    diagnostic_processed: np.ndarray,
    target_processed: np.ndarray,
    rate: int,
    transfer_frequencies: np.ndarray,
    transfer_power: np.ndarray,
    transfer_real: np.ndarray,
    transfer_imag: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    frequencies, _, mixture_stft, nperseg, hop = _stft(
        mixture,
        rate,
        cfg["frame_sec"],
        cfg["hop_sec"],
        reconstruction=True,
    )
    _, _, reference_stft, _, _ = _stft(
        reference,
        rate,
        cfg["frame_sec"],
        cfg["hop_sec"],
        reconstruction=True,
    )
    _, _, diagnostic_stft, _, _ = _stft(
        diagnostic_processed,
        rate,
        cfg["frame_sec"],
        cfg["hop_sec"],
        reconstruction=True,
    )
    _, _, target_stft, _, _ = _stft(
        target_processed,
        rate,
        cfg["frame_sec"],
        cfg["hop_sec"],
        reconstruction=True,
    )
    frames = min(
        mixture_stft.shape[1],
        reference_stft.shape[1],
        diagnostic_stft.shape[1],
        target_stft.shape[1],
    )
    mixture_stft = mixture_stft[:, :frames]
    reference_stft = reference_stft[:, :frames]
    diagnostic_stft = diagnostic_stft[:, :frames]
    target_stft = target_stft[:, :frames]
    removed_stft = mixture_stft - diagnostic_stft
    # Interpolate magnitude and unwrapped phase separately.  Interpolating real
    # and imaginary parts across a phase wrap would artificially collapse the
    # reference estimate and leak English speech into the residual.
    transfer_power_at_rate = np.interp(
        frequencies,
        transfer_frequencies,
        transfer_power,
        left=float(transfer_power[0]),
        right=float(transfer_power[-1]),
    )
    source_complex = transfer_real + 1j * transfer_imag
    source_phase = np.unwrap(np.angle(source_complex))
    transfer_phase = np.interp(
        frequencies,
        transfer_frequencies,
        source_phase,
        left=float(source_phase[0]),
        right=float(source_phase[-1]),
    )
    transfer_complex = np.sqrt(
        np.maximum(transfer_power_at_rate, 0.0)
    ) * np.exp(1j * transfer_phase)
    speech_band = (
        (frequencies >= cfg["minimum_frequency_hz"])
        & (frequencies <= cfg["maximum_frequency_hz"])
    )
    transfer_power_at_rate = np.where(
        speech_band, transfer_power_at_rate, 0.0
    )
    transfer_complex = np.where(speech_band, transfer_complex, 0.0)
    guard = float(10.0 ** (cfg["expected_reference_guard_db"] / 10.0))
    mixture_power = np.abs(mixture_stft) ** 2
    reference_power = np.abs(reference_stft) ** 2
    processed_power = np.abs(target_stft) ** 2
    removed_power = np.abs(removed_stft) ** 2
    expected = transfer_power_at_rate[:, None] * reference_power * guard
    near_end = np.maximum(mixture_power - expected, 0.0)
    missing = np.maximum(near_end - processed_power, 0.0)
    # The amount is still bounded by the conservative power test, but the
    # waveform comes from the complex residual after subtracting the estimated
    # English component.  Scaling ``removed_stft`` directly would return EN and
    # RU in their original proportion.
    residual_stft = (
        removed_stft - transfer_complex[:, None] * reference_stft
    )
    residual_power = np.abs(residual_stft) ** 2
    unexpected = np.minimum(
        np.maximum(removed_power - expected, 0.0),
        residual_power,
    )
    recoverable = np.minimum(missing, unexpected)
    mask = np.sqrt(recoverable / np.maximum(residual_power, 1e-12))
    mask = np.minimum(mask, cfg["maximum_recovery_mask"])
    mask[~speech_band, :] = 0.0
    _, restored = signal.istft(
        residual_stft * mask,
        fs=rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - hop,
        nfft=removed_stft.shape[0] * 2 - 2,
        input_onesided=True,
        boundary=True,
    )
    restored = np.asarray(restored, dtype=np.float32)
    if restored.size < mixture.size:
        restored = np.pad(restored, (0, mixture.size - restored.size))
    return restored[: mixture.size]


def _output_level_gain_cap(
    mixture: np.ndarray,
    target: np.ndarray,
    correction: np.ndarray,
    event_left: int,
    event_right: int,
    maximum_output_to_input_rms: float,
) -> float:
    """Largest gain that keeps the repaired event no louder than the input.

    The dubbed speech stem is the only source the correction can come from,
    so a protected result that is louder than that stem over the same window
    cannot be "recovered near-end speech" any more.  Returning the untouched
    EN+RU mixture is the degenerate case this rules out, whatever the
    detector thresholds did.
    """
    mixture_2d = audio_io.ensure_2d(np.asarray(mixture, dtype=np.float32))
    target_2d = audio_io.ensure_2d(np.asarray(target, dtype=np.float32))
    correction_2d = audio_io.ensure_2d(
        np.asarray(correction, dtype=np.float32)
    )
    channels = target_2d.shape[1]
    mixture_2d = audio_io.match_channels(mixture_2d, channels)
    correction_2d = audio_io.match_channels(correction_2d, channels)
    length = min(
        mixture_2d.shape[0],
        target_2d.shape[0],
        correction_2d.shape[0],
    )
    if event_right <= event_left:
        event = slice(0, length)
    else:
        event = slice(
            max(0, min(length, event_left)),
            max(0, min(length, event_right)),
        )
    if target_2d[event].size == 0:
        return 0.0

    caps: list[float] = []
    for channel in range(channels):
        input_values = np.asarray(
            mixture_2d[event, channel], dtype=np.float64
        )
        target_values = np.asarray(
            target_2d[event, channel], dtype=np.float64
        )
        correction_values = np.asarray(
            correction_2d[event, channel], dtype=np.float64
        )
        limit = maximum_output_to_input_rms * float(
            np.sqrt(np.mean(np.square(input_values)))
        )
        quadratic = float(np.mean(np.square(correction_values)))
        constant = float(np.mean(np.square(target_values))) - limit * limit
        # If the unmodified target is already above the limit, no value in
        # ``[0, cap]`` is guaranteed safe: an anti-correlated correction can
        # create a disjoint valid interval whose lower root is greater than
        # zero.  The caller treats this as an upper cap, so veto instead.
        if constant > 0.0:
            return 0.0
        if quadratic <= 1e-18:
            caps.append(float("inf"))
            continue
        linear = 2.0 * float(np.mean(target_values * correction_values))
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant <= 0.0 or not math.isfinite(discriminant):
            return 0.0
        cap = max(
            0.0,
            (-linear + math.sqrt(discriminant)) / (2.0 * quadratic),
        )
        caps.append(cap if math.isfinite(cap) else 0.0)
    return min(caps, default=0.0)


def _coalesced_intervals(
    intervals: list[dict[str, Any]], crossfade_sec: float
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in sorted(intervals, key=lambda row: row["start_sec"]):
        if (
            result
            and item["start_sec"] - result[-1]["end_sec"]
            <= crossfade_sec * 2.0
        ):
            if item["confidence"] > result[-1]["confidence"]:
                result[-1]["calibration_block_index"] = item[
                    "calibration_block_index"
                ]
            result[-1]["end_sec"] = max(
                result[-1]["end_sec"], item["end_sec"]
            )
            result[-1]["confidence"] = max(
                result[-1]["confidence"], item["confidence"]
            )
            result[-1]["source_interval_count"] += 1
        else:
            result.append(
                {
                    **item,
                    "source_interval_count": 1,
                }
            )
    return result


def _copy_audio_frames(
    reader: sf.SoundFile,
    writer: sf.SoundFile,
    start: int,
    end: int,
    block_frames: int,
) -> None:
    position = start
    while position < end:
        reader.seek(position)
        values = reader.read(
            min(block_frames, end - position),
            dtype="float32",
            always_2d=True,
        )
        if values.size == 0:
            break
        writer.write(values)
        position += values.shape[0]


def apply_coincident_speech_protection(
    mixture_path: str | Path,
    reference_path: str | Path,
    diagnostic_processed_path: str | Path,
    target_processed_path: str | Path,
    output_path: str | Path,
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    report_path: str | Path | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Apply a previously detected repair plan to one first/second-pass voice."""
    mixture = Path(mixture_path).resolve()
    reference = Path(reference_path).resolve()
    diagnostic = Path(diagnostic_processed_path).resolve()
    target = Path(target_processed_path).resolve()
    output = Path(output_path).resolve()
    original_report = report
    if not isinstance(report, dict):
        report = {}
    report_config = report.get("config")
    config_source = (
        config
        if isinstance(config, dict) and config
        else report_config
        if isinstance(report_config, dict)
        else {}
    )
    try:
        cfg = resolved_config(config_source)
    except (KeyError, TypeError, ValueError, OverflowError):
        cfg = resolved_config({})
    # A plan can arrive from a cached JSON file that was truncated or written
    # by an older build.  Missing keys must degrade to "change nothing", never
    # to an exception in the middle of the application pipeline.
    summary = report.get("summary")
    if not isinstance(summary, dict):
        summary = {}
        report["summary"] = summary
    plan_valid, plan_reason = validate_detection_report(
        original_report,
        mixture,
        reference,
        diagnostic,
        cfg,
    )
    intervals = (
        list(report.get("confirmed_intervals") or []) if plan_valid else []
    )
    application = {
        "target": _audio_identity(target),
        "output": str(output),
        "segments": [],
        "modified": False,
    }
    if plan_valid and any(
        float(item["end_sec"])
        > float(application["target"]["duration_sec"]) + 0.05
        for item in intervals
    ):
        plan_valid = False
        plan_reason = "target_duration_mismatch"
        intervals = []
    if not plan_valid:
        summary["application_reason"] = plan_reason
    if not cfg["enabled"] or not intervals:
        output.parent.mkdir(parents=True, exist_ok=True)
        if target != output:
            shutil.copy2(target, output)
        application["output"] = _audio_identity(output)
        report["application"] = application
        summary["modified"] = False
        summary["applied_count"] = 0
        if report_path is not None:
            atomic_json(Path(report_path), report)
        if ctx is not None:
            ctx.update(
                stage="Проверка сохранности совпадающих слов",
                substage="Безопасная замена не требуется",
                progress=100.0,
                local_progress=100.0,
            )
        return report

    transfer_frequencies = np.asarray(
        report.get("frequency_bins_hz") or [], dtype=np.float64
    )
    block_by_index = {
        int(block["index"]): block
        for block in report.get("calibration_blocks") or []
        if (
            block.get("stable")
            and block.get("transfer_power")
            and block.get("transfer_real")
            and block.get("transfer_imag")
            # A transfer curve sampled on a different grid than the reported
            # frequency axis cannot be interpolated; treat it as unusable
            # rather than letting numpy raise deep inside the writer loop.
            and len(block["transfer_power"]) == transfer_frequencies.size
            and len(block["transfer_real"]) == transfer_frequencies.size
            and len(block["transfer_imag"]) == transfer_frequencies.size
            and np.all(np.isfinite(np.asarray(block["transfer_power"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(block["transfer_real"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(block["transfer_imag"], dtype=np.float64)))
        )
    }
    usable = [
        item
        for item in intervals
        if int(item.get("calibration_block_index", -1)) in block_by_index
    ]
    usable = _coalesced_intervals(usable, cfg["crossfade_sec"])
    if not usable or transfer_frequencies.size == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        if target != output:
            shutil.copy2(target, output)
        application["output"] = _audio_identity(output)
        report["application"] = application
        summary["modified"] = False
        summary["applied_count"] = 0
        summary["application_reason"] = "missing_calibration_transfer"
        if report_path is not None:
            atomic_json(Path(report_path), report)
        return report

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.partial"
    )
    applied: list[dict[str, Any]] = []
    try:
        with sf.SoundFile(str(target)) as target_reader:
            rate = int(target_reader.samplerate)
            channels = int(target_reader.channels)
            total_frames = int(len(target_reader))
            with sf.SoundFile(
                str(temporary),
                mode="w",
                samplerate=rate,
                channels=channels,
                format=target_reader.format,
                subtype=target_reader.subtype,
            ) as writer:
                cursor = 0
                block_frames = max(rate, rate * 20)
                for index, item in enumerate(usable):
                    fade = cfg["crossfade_sec"]
                    left = max(
                        cursor,
                        int(
                            math.floor(
                                max(0.0, item["start_sec"] - fade) * rate
                            )
                        ),
                    )
                    right = min(
                        total_frames,
                        int(
                            math.ceil(
                                (item["end_sec"] + fade) * rate
                            )
                        ),
                    )
                    if right <= left:
                        continue
                    _copy_audio_frames(
                        target_reader, writer, cursor, left, block_frames
                    )
                    target_reader.seek(left)
                    target_values = np.nan_to_num(
                        target_reader.read(
                            right - left,
                            dtype="float32",
                            always_2d=True,
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    segment_start = left / rate
                    segment_end = right / rate
                    mixture_channel_values = _read_audio_segment(
                        mixture, segment_start, segment_end, rate
                    )
                    mixture_values = audio_io.to_mono(
                        mixture_channel_values
                    )
                    reference_values = _read_mono_segment(
                        reference, segment_start, segment_end, rate
                    )
                    diagnostic_values = _read_mono_segment(
                        diagnostic, segment_start, segment_end, rate
                    )
                    length = min(
                        target_values.shape[0],
                        mixture_channel_values.shape[0],
                        reference_values.size,
                        diagnostic_values.size,
                    )
                    target_values = target_values[:length]
                    mixture_channel_values = mixture_channel_values[:length]
                    mixture_values = mixture_values[:length]
                    reference_values = reference_values[:length]
                    diagnostic_values = diagnostic_values[:length]
                    block = block_by_index[
                        int(item["calibration_block_index"])
                    ]
                    transfer_power = np.asarray(
                        block["transfer_power"], dtype=np.float64
                    )
                    transfer_real = np.asarray(
                        block["transfer_real"], dtype=np.float64
                    )
                    transfer_imag = np.asarray(
                        block["transfer_imag"], dtype=np.float64
                    )
                    correction = _recovery_waveform(
                        mixture_values,
                        reference_values,
                        diagnostic_values,
                        audio_io.to_mono(target_values),
                        rate,
                        transfer_frequencies,
                        transfer_power,
                        transfer_real,
                        transfer_imag,
                        cfg,
                    )
                    # A degenerate STFT frame must not be able to write NaN
                    # into a deliverable, nor into the JSON diagnostics.
                    correction = np.nan_to_num(
                        correction, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    envelope = np.ones(length, dtype=np.float32)
                    event_left = int(
                        round((item["start_sec"] - segment_start) * rate)
                    )
                    event_right = int(
                        round((item["end_sec"] - segment_start) * rate)
                    )
                    event_left = max(0, min(length, event_left))
                    event_right = max(event_left, min(length, event_right))
                    if event_left > 0:
                        phase = np.linspace(
                            0.0,
                            math.pi * 0.5,
                            event_left,
                            endpoint=True,
                        )
                        envelope[:event_left] = np.sin(phase) ** 2
                    if event_right < length:
                        phase = np.linspace(
                            math.pi * 0.5,
                            0.0,
                            length - event_right,
                            endpoint=True,
                        )
                        envelope[event_right:] = np.sin(phase) ** 2
                    correction *= envelope
                    input_rms = float(
                        np.sqrt(np.mean(mixture_values * mixture_values))
                    )
                    correction_rms = float(
                        np.sqrt(np.mean(correction * correction))
                    )
                    relative_cap = (
                        input_rms
                        * cfg["maximum_recovery_relative_rms"]
                        / correction_rms
                        if correction_rms > 1e-9
                        else 0.0
                    )
                    requested_gain = min(
                        cfg["maximum_recovery_gain"],
                        cfg["target_recovery_fraction"]
                        * float(item["confidence"]),
                        relative_cap,
                    )
                    correction_channels = audio_io.match_channels(
                        correction[:, None], channels
                    )
                    level_cap = _output_level_gain_cap(
                        mixture_channel_values,
                        target_values,
                        correction_channels,
                        0,
                        length,
                        cfg["maximum_output_to_input_rms"],
                    )
                    level_limited = bool(level_cap < requested_gain)
                    gain = max(0.0, min(requested_gain, level_cap))
                    effective_delta_rms = float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    correction_channels * gain,
                                    dtype=np.float64,
                                )
                            )
                        )
                    )
                    combined = np.nan_to_num(
                        target_values + correction_channels * gain,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    peak_before_clip = float(
                        np.max(np.abs(combined)) if combined.size else 0.0
                    )
                    combined = np.clip(combined, -0.999, 0.999)
                    writer.write(combined)
                    cursor = left + length
                    applied.append(
                        {
                            "start_sec": round(item["start_sec"], 6),
                            "end_sec": round(item["end_sec"], 6),
                            "confidence": item["confidence"],
                            "recovery_gain": round(float(gain), 6),
                            "effective_delta_rms": round(
                                effective_delta_rms, 9
                            ),
                            "correction_rms_db": round(
                                float(_safe_db(correction_rms**2)), 4
                            ),
                            "peak_before_clip": round(peak_before_clip, 6),
                            "output_level_gain_cap": round(
                                float(min(level_cap, 1e6)), 6
                            ),
                            "limited_by_output_level_guard": level_limited,
                            "source_interval_count": item[
                                "source_interval_count"
                            ],
                        }
                    )
                    if ctx is not None:
                        ctx.update(
                            stage="Проверка сохранности совпадающих слов",
                            substage="Восстановление подтверждённых фрагментов",
                            progress=90.0
                            + (index + 1) / max(1, len(usable)) * 10.0,
                            local_progress=90.0
                            + (index + 1) / max(1, len(usable)) * 10.0,
                        )
                        if hasattr(ctx, "check_stop"):
                            ctx.check_stop()
                _copy_audio_frames(
                    target_reader, writer, cursor, total_frames, block_frames
                )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    changed = [
        item
        for item in applied
        if item["recovery_gain"] > 0.0
        and item.get("effective_delta_rms", 0.0) > 1e-9
    ]
    application["segments"] = applied
    application["modified"] = bool(changed)
    application["output"] = _audio_identity(output)
    report["application"] = application
    summary["modified"] = bool(changed)
    summary["applied_count"] = len(changed)
    summary["level_limited_count"] = sum(
        1 for item in applied if item["limited_by_output_level_guard"]
    )
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


def protect_coincident_speech(
    mixture_path: str | Path,
    reference_path: str | Path,
    processed_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    analysis_intervals: Iterable[
        tuple[float, float] | dict[str, Any]
    ]
    | None = None,
    report_path: str | Path | None = None,
    ctx: Any | None = None,
) -> dict[str, Any]:
    """Detect and apply protection to a first-pass result."""
    report = detect_coincident_speech(
        mixture_path,
        reference_path,
        processed_path,
        config,
        analysis_intervals=analysis_intervals,
        ctx=ctx,
    )
    return apply_coincident_speech_protection(
        mixture_path,
        reference_path,
        processed_path,
        processed_path,
        output_path,
        report,
        config,
        report_path=report_path,
        ctx=ctx,
    )
