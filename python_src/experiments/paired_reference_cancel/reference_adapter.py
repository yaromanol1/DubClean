from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.storage import atomic_json


DEFAULT_N_FFT = 512
DEFAULT_HOP = 256
DEFAULT_BANDS = 40
PROFILE_SCHEMA_VERSION = 2
PROFILE_MODE = "shared_english_crosspower_scalar_v2"


DEFAULT_CROSSPOWER_CONFIG: dict[str, float | int] = {
    # One second alternating fit/holdout blocks keep both sets distributed
    # across the calibration material instead of validating on one tail only.
    "validation_block_sec": 1.0,
    "min_hz": 120.0,
    "max_hz": 8_000.0,
    "active_dynamic_range_db": 25.0,
    "reference_band_dynamic_range_db": 45.0,
    "min_coherence": 0.12,
    "min_median_coherence": 0.18,
    "min_phase_cosine": 0.50,
    "min_support_bins": 8,
    "min_support_fraction": 0.025,
    "max_gain_mad_db": 2.5,
    "max_fit_holdout_delta_db": 1.25,
    "max_abs_gain_db": 6.0,
    "gain_cap_tolerance_db": 0.75,
    # Sub-dB corrections are more safely left to DubClean Voice itself.  This
    # also makes additive translated speech unable to manufacture a "useful"
    # mastering correction around unity.
    "min_abs_gain_db": 0.5,
    "max_output_peak": 0.995,
}


def _mono(values: np.ndarray) -> np.ndarray:
    data = audio_io.ensure_2d(np.asarray(values, dtype=np.float32))
    if data.shape[1] >= 3:
        return data[:, 2].astype(np.float32)
    return np.mean(data, axis=1).astype(np.float32)


def _rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data) + 1e-12))


def _db(value: float) -> float:
    return 20.0 * math.log10(max(float(value), 1e-9))


def _linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    values, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return _mono(values), int(rate)


def _match_length(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(left), len(right))
    return left[:length], right[:length]


def _stft(values: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    _, _, spectrum = signal.stft(
        np.asarray(values, dtype=np.float32),
        fs=1.0,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    return spectrum.astype(np.complex64)


def _istft(spectrum: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    _, values = signal.istft(
        spectrum,
        fs=1.0,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    values = np.asarray(values, dtype=np.float32)
    if len(values) < length:
        values = np.pad(values, (0, length - len(values)))
    return values[:length].astype(np.float32)


def _mel_scale(freq_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def _hz_from_mel(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    bands: int = DEFAULT_BANDS,
    min_hz: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    max_hz = sample_rate / 2.0
    points = _hz_from_mel(
        np.linspace(_mel_scale(np.array([min_hz]))[0], _mel_scale(np.array([max_hz]))[0], bands + 2)
    )
    freqs = np.linspace(0.0, max_hz, n_fft // 2 + 1)
    filters = np.zeros((bands, len(freqs)), dtype=np.float32)
    for band in range(bands):
        left, center, right = points[band], points[band + 1], points[band + 2]
        up = (freqs - left) / max(center - left, 1e-6)
        down = (right - freqs) / max(right - center, 1e-6)
        filters[band] = np.maximum(0.0, np.minimum(up, down))
    filters /= np.maximum(np.sum(filters, axis=1, keepdims=True), 1e-8)
    return filters.astype(np.float32), points[1:-1].astype(np.float32)


def _band_energy(magnitude: np.ndarray, filters: np.ndarray) -> np.ndarray:
    # magnitude: freq x frames; output: bands x frames
    return np.maximum(filters @ np.asarray(magnitude, dtype=np.float32), 1e-9)


def _band_to_frequency_gain(
    gains_db: np.ndarray,
    sample_rate: int,
    n_fft: int,
    centers_hz: np.ndarray,
) -> np.ndarray:
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    extended_x = np.concatenate([[0.0], centers_hz, [sample_rate / 2.0]])
    extended_y = np.concatenate([[gains_db[0]], gains_db, [gains_db[-1]]])
    gain_db = np.interp(freqs, extended_x, extended_y).astype(np.float32)
    return np.asarray(10.0 ** (gain_db / 20.0), dtype=np.float32)


def _logmel_distance(
    reference: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop: int,
    bands: int,
    quantile: float = 0.25,
) -> dict[str, float]:
    reference, mixture = _match_length(reference, mixture)
    filters, _centers = _mel_filterbank(sample_rate, n_fft, bands)
    ref_mag = np.abs(_stft(reference, n_fft, hop))
    mix_mag = np.abs(_stft(mixture, n_fft, hop))
    ref_band = _band_energy(ref_mag, filters)
    mix_band = _band_energy(mix_mag, filters)
    ref_db = 20.0 * np.log10(ref_band)
    mix_db = 20.0 * np.log10(mix_band)
    ref_energy = np.mean(ref_band, axis=0)
    active = ref_energy > max(float(np.percentile(ref_energy, 70)) * 0.15, 1e-8)
    if int(np.count_nonzero(active)) < 5:
        active = ref_energy > 1e-8
    diff = np.mean(np.abs(ref_db[:, active] - mix_db[:, active]), axis=0)
    if len(diff) == 0:
        return {"median": 0.0, "lower_quantile": 0.0, "frames": 0.0}
    return {
        "median": float(np.median(diff)),
        "lower_quantile": float(np.quantile(diff, quantile)),
        "frames": float(len(diff)),
    }


def estimate_envelope_delay(
    reference: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    frame_sec: float = 0.04,
    max_lag_sec: float = 1.5,
) -> tuple[float, float]:
    reference, mixture = _match_length(reference, mixture)
    frame = max(1, int(round(sample_rate * frame_sec)))
    usable = (len(reference) // frame) * frame
    if usable < frame * 10:
        return 0.0, 0.0
    ref_env = np.sqrt(
        np.mean(reference[:usable].reshape(-1, frame).astype(np.float64) ** 2, axis=1)
        + 1e-12
    )
    mix_env = np.sqrt(
        np.mean(mixture[:usable].reshape(-1, frame).astype(np.float64) ** 2, axis=1)
        + 1e-12
    )
    ref_env = np.log1p(ref_env * 30.0)
    mix_env = np.log1p(mix_env * 30.0)
    ref_env -= float(np.mean(ref_env))
    mix_env -= float(np.mean(mix_env))
    ref_norm = float(np.linalg.norm(ref_env))
    mix_norm = float(np.linalg.norm(mix_env))
    if ref_norm < 1e-8 or mix_norm < 1e-8:
        return 0.0, 0.0
    ref_env /= ref_norm
    mix_env /= mix_norm
    max_lag = int(round(max_lag_sec / frame_sec))
    corr = signal.correlate(mix_env, ref_env, mode="full", method="fft")
    center = len(ref_env) - 1
    window = corr[center - max_lag : center + max_lag + 1]
    if len(window) == 0:
        return 0.0, 0.0
    peak = int(np.argmax(np.abs(window)))
    lag_frames = peak - max_lag
    confidence = float(abs(window[peak]))
    return float(lag_frames * frame_sec), confidence


def _adapter_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(DEFAULT_CROSSPOWER_CONFIG)
    if config:
        for key in merged:
            if key in config:
                merged[key] = config[key]
    return merged


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return float("nan")
    values = values[usable]
    weights = weights[usable]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    midpoint = float(np.sum(weights)) * 0.5
    index = min(int(np.searchsorted(np.cumsum(weights), midpoint)), len(values) - 1)
    return float(values[index])


def _alternating_validation_masks(
    frames: int,
    sample_rate: int,
    hop: int,
    block_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(frames)
    block_frames = max(8, int(round(float(block_sec) * sample_rate / max(hop, 1))))
    blocks = indices // block_frames
    fit = blocks % 2 == 0
    holdout = ~fit
    # Very short calibration clips still get a real disjoint holdout.
    if int(np.count_nonzero(holdout)) < 8:
        split = max(8, min(frames - 8, int(round(frames * 0.60))))
        fit = indices < split
        holdout = ~fit
    return fit, holdout


def _crosspower_split_estimate(
    reference_spectrum: np.ndarray,
    mixture_spectrum: np.ndarray,
    frame_mask: np.ndarray,
    sample_rate: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    selected = np.flatnonzero(frame_mask)
    if len(selected) < 8:
        return {
            "usable": False,
            "reason": "insufficient_frames",
            "support_bins": 0,
            "candidate_bins": 0,
            "support_fraction": 0.0,
        }

    max_hz = min(float(settings["max_hz"]), sample_rate / 2.0)
    frequencies = np.linspace(0.0, sample_rate / 2.0, reference_spectrum.shape[0])
    frequency_mask = (frequencies >= float(settings["min_hz"])) & (frequencies <= max_hz)
    if not np.any(frequency_mask):
        return {
            "usable": False,
            "reason": "empty_frequency_range",
            "support_bins": 0,
            "candidate_bins": 0,
            "support_fraction": 0.0,
        }

    # Silence must not participate in the cross-power estimate.  The threshold
    # is relative to the calibration material, so quiet but valid stems remain
    # measurable without an absolute loudness assumption.
    frame_power = np.mean(
        np.abs(reference_spectrum[frequency_mask][:, selected]) ** 2,
        axis=0,
    )
    active_floor = float(np.percentile(frame_power, 70.0)) * _linear(
        -float(settings["active_dynamic_range_db"])
    ) ** 2
    active = frame_power >= max(active_floor, 1e-18)
    selected = selected[active]
    if len(selected) < 8:
        return {
            "usable": False,
            "reason": "insufficient_active_frames",
            "support_bins": 0,
            "candidate_bins": 0,
            "support_fraction": 0.0,
        }

    ref = reference_spectrum[:, selected].astype(np.complex128, copy=False)
    mix = mixture_spectrum[:, selected].astype(np.complex128, copy=False)
    ref_power = np.mean(np.abs(ref) ** 2, axis=1)
    mix_power = np.mean(np.abs(mix) ** 2, axis=1)
    # E[Y * conj(X)] isolates the common English component.  Independent RU
    # speech contributes energy to Syy but converges to zero in this cross term.
    cross_power = np.mean(mix * np.conj(ref), axis=1)
    coherence = np.abs(cross_power) ** 2 / np.maximum(ref_power * mix_power, 1e-30)
    transfer = np.real(cross_power) / np.maximum(ref_power, 1e-30)
    phase_cosine = np.real(cross_power) / np.maximum(np.abs(cross_power), 1e-30)

    ref_power_db = 10.0 * np.log10(np.maximum(ref_power, 1e-30))
    finite_band_power = ref_power_db[frequency_mask & np.isfinite(ref_power_db)]
    if len(finite_band_power) == 0:
        return {
            "usable": False,
            "reason": "silent_reference",
            "support_bins": 0,
            "candidate_bins": 0,
            "support_fraction": 0.0,
        }
    relative_floor = float(np.max(finite_band_power)) - float(
        settings["reference_band_dynamic_range_db"]
    )
    candidate = (
        frequency_mask
        & np.isfinite(ref_power_db)
        & (ref_power_db >= relative_floor)
    )
    support = (
        candidate
        & np.isfinite(coherence)
        & np.isfinite(transfer)
        & (coherence >= float(settings["min_coherence"]))
        & (phase_cosine >= float(settings["min_phase_cosine"]))
        & (transfer > 1e-6)
    )
    candidate_bins = int(np.count_nonzero(candidate))
    support_bins = int(np.count_nonzero(support))
    support_fraction = float(support_bins / max(candidate_bins, 1))
    if support_bins == 0:
        return {
            "usable": False,
            "reason": "no_coherent_common_component",
            "support_bins": 0,
            "candidate_bins": candidate_bins,
            "support_fraction": 0.0,
        }

    gain_values_db = 20.0 * np.log10(transfer[support])
    normalized_power = ref_power[support] / max(float(np.max(ref_power[support])), 1e-30)
    weights = np.sqrt(np.maximum(normalized_power, 1e-12)) * coherence[support]
    gain_db = _weighted_median(gain_values_db, weights)
    mad_db = _weighted_median(np.abs(gain_values_db - gain_db), weights)
    median_coherence = _weighted_median(coherence[support], weights)
    usable = (
        support_bins >= int(settings["min_support_bins"])
        and support_fraction >= float(settings["min_support_fraction"])
        and median_coherence >= float(settings["min_median_coherence"])
    )
    return {
        "usable": bool(usable),
        "reason": "ok" if usable else "insufficient_coherent_support",
        "gain_db": float(gain_db),
        "mad_db": float(mad_db),
        "median_coherence": float(median_coherence),
        "support_bins": support_bins,
        "candidate_bins": candidate_bins,
        "support_fraction": support_fraction,
        "active_frames": int(len(selected)),
    }


@dataclass
class ReferenceAdapterProfile:
    sample_rate: int
    n_fft: int
    hop: int
    bands: int
    band_centers_hz: list[float]
    band_gains_db: list[float]
    delay_sec: float
    delay_confidence: float
    distance_before: dict[str, float]
    distance_after_estimate: dict[str, float]
    schema_version: int = PROFILE_SCHEMA_VERSION
    mode: str = PROFILE_MODE
    summary: dict[str, Any] = field(default_factory=dict)
    common_gain_db: float = 0.0
    valid: bool = True
    validation_reason: str = "ok"

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ReferenceAdapterProfile":
        raw = data if isinstance(data, dict) else {}

        def invalid(reason: str) -> "ReferenceAdapterProfile":
            try:
                raw_rate = max(1, int(raw.get("sample_rate") or 1))
            except (TypeError, ValueError, OverflowError):
                raw_rate = 1
            try:
                raw_schema = int(raw.get("schema_version") or 0)
            except (TypeError, ValueError, OverflowError):
                raw_schema = 0
            return cls(
                sample_rate=raw_rate,
                n_fft=DEFAULT_N_FFT,
                hop=DEFAULT_HOP,
                bands=1,
                band_centers_hz=[1.0],
                band_gains_db=[0.0],
                delay_sec=0.0,
                delay_confidence=0.0,
                distance_before={},
                distance_after_estimate={},
                schema_version=raw_schema,
                mode=str(raw.get("mode") or ""),
                summary=(
                    dict(raw.get("summary") or {})
                    if isinstance(raw.get("summary"), dict)
                    else {}
                ),
                common_gain_db=0.0,
                valid=False,
                validation_reason=reason,
            )

        if not isinstance(data, dict):
            return invalid("profile_is_not_an_object")
        try:
            schema_version = int(data.get("schema_version") or 0)
            mode = str(data.get("mode") or "")
            sample_rate = int(data["sample_rate"])
            n_fft = int(data["n_fft"])
            hop = int(data["hop"])
            bands = int(data["bands"])
            centers = [float(value) for value in data["band_centers_hz"]]
            gains = [float(value) for value in data["band_gains_db"]]
            summary = dict(data.get("summary") or {})
        except (KeyError, TypeError, ValueError, OverflowError):
            return invalid("malformed_profile")

        trusted_global = (
            mode.startswith("global_mastering_speechfree_")
            or mode.startswith("neural_global_mastering_speechfree_")
        )
        if not (
            (schema_version == PROFILE_SCHEMA_VERSION and mode == PROFILE_MODE)
            or (schema_version == 1 and trusted_global)
        ):
            return invalid("unsupported_schema_or_mode")
        if (
            sample_rate <= 0
            or n_fft < 16
            or hop <= 0
            or hop >= n_fft
            or bands <= 0
            or len(centers) != bands
            or len(gains) != bands
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(gains))
            or np.any(np.diff(np.asarray(centers, dtype=np.float64)) <= 0.0)
        ):
            return invalid("invalid_profile_dimensions")
        hard_gain_limit = 18.0 if trusted_global else 6.05
        if float(np.max(np.abs(gains))) > hard_gain_limit:
            return invalid("unsafe_gain")
        if mode == PROFILE_MODE and float(np.max(gains) - np.min(gains)) > 0.05:
            return invalid("non_scalar_algorithmic_profile")
        common_gain_db = float(
            data.get("common_gain_db")
            if data.get("common_gain_db") is not None
            else np.median(gains)
        )
        if not math.isfinite(common_gain_db):
            return invalid("invalid_common_gain")
        if abs(common_gain_db) > hard_gain_limit:
            return invalid("unsafe_common_gain")
        if mode == PROFILE_MODE and abs(common_gain_db - float(np.median(gains))) > 0.05:
            return invalid("inconsistent_common_gain")
        return cls(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop=hop,
            bands=bands,
            band_centers_hz=centers,
            band_gains_db=gains,
            delay_sec=float(data.get("delay_sec") or 0.0),
            delay_confidence=float(data.get("delay_confidence") or 0.0),
            distance_before=dict(data.get("distance_before") or {}),
            distance_after_estimate=dict(data.get("distance_after_estimate") or {}),
            schema_version=schema_version,
            mode=mode,
            summary=summary,
            common_gain_db=common_gain_db,
        )

    def to_json(self) -> dict[str, Any]:
        summary = dict(self.summary)
        summary.setdefault("usable", False)
        summary.setdefault("reason", self.validation_reason)
        summary.setdefault("median_gain_db", float(self.common_gain_db))
        summary.setdefault("min_gain_db", float(np.min(self.band_gains_db)))
        summary.setdefault("max_gain_db", float(np.max(self.band_gains_db)))
        return {
            "schema_version": int(self.schema_version),
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "bands": self.bands,
            "band_centers_hz": self.band_centers_hz,
            "band_gains_db": self.band_gains_db,
            "common_gain_db": float(self.common_gain_db),
            "delay_sec": self.delay_sec,
            "delay_confidence": self.delay_confidence,
            "distance_before": self.distance_before,
            "distance_after_estimate": self.distance_after_estimate,
            "summary": summary,
        }


def _profile_application_safety(
    profile: ReferenceAdapterProfile,
) -> tuple[bool, str]:
    if not profile.valid:
        return False, profile.validation_reason
    summary = profile.summary or {}
    if profile.mode == PROFILE_MODE:
        if not bool(summary.get("usable")):
            return False, str(summary.get("reason") or "profile_not_validated")
        if summary.get("reason") not in {None, "", "ok"}:
            return False, str(summary.get("reason"))
        if summary.get("verdict") not in {None, "", "ok"}:
            return False, str(summary.get("verdict"))
        return True, "ok"
    if (
        profile.mode.startswith("global_mastering_speechfree_")
        or profile.mode.startswith("neural_global_mastering_speechfree_")
    ):
        # During global profile fitting the summary does not exist yet; after
        # fitting, an explicit rejection must always fail closed.
        if summary.get("usable") is False:
            return False, str(summary.get("verdict") or "global_profile_rejected")
        verdict = summary.get("verdict")
        if verdict not in {None, "", "ok"}:
            return False, str(verdict)
        return True, "ok"
    return False, "unsupported_profile_mode"


def fit_algorithmic_profile(
    reference: np.ndarray,
    mixture: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int | None = None,
    hop: int | None = None,
    bands: int | None = None,
    quantile: float = 0.20,
    config: dict[str, Any] | None = None,
) -> ReferenceAdapterProfile:
    del quantile  # Retained for backwards-compatible callers; no magnitude quantiles.
    settings = _adapter_config(config)
    n_fft = int(n_fft if n_fft is not None else (config or {}).get("n_fft", DEFAULT_N_FFT))
    hop = int(hop if hop is not None else (config or {}).get("hop", DEFAULT_HOP))
    bands = int(bands if bands is not None else (config or {}).get("bands", DEFAULT_BANDS))
    reference, mixture = _match_length(
        np.asarray(reference, dtype=np.float32),
        np.asarray(mixture, dtype=np.float32),
    )
    _filters, centers_hz = _mel_filterbank(sample_rate, n_fft, bands)
    delay, delay_confidence = estimate_envelope_delay(reference, mixture, sample_rate)

    reason = "ok"
    fit: dict[str, Any] = {}
    holdout: dict[str, Any] = {}
    gain_db = 0.0
    if len(reference) < max(n_fft * 4, int(sample_rate * 0.25)):
        reason = "insufficient_audio"
    elif not np.all(np.isfinite(reference)) or not np.all(np.isfinite(mixture)):
        reason = "non_finite_audio"
    elif _rms(reference) < 1e-8:
        reason = "silent_reference"
    else:
        ref_spectrum = _stft(reference, n_fft, hop)
        mix_spectrum = _stft(mixture, n_fft, hop)
        fit_mask, holdout_mask = _alternating_validation_masks(
            ref_spectrum.shape[1],
            sample_rate,
            hop,
            float(settings["validation_block_sec"]),
        )
        fit = _crosspower_split_estimate(
            ref_spectrum, mix_spectrum, fit_mask, sample_rate, settings
        )
        holdout = _crosspower_split_estimate(
            ref_spectrum, mix_spectrum, holdout_mask, sample_rate, settings
        )
        if not fit.get("usable"):
            reason = f"fit_{fit.get('reason') or 'unusable'}"
        elif not holdout.get("usable"):
            reason = f"holdout_{holdout.get('reason') or 'unusable'}"
        else:
            fit_gain_db = float(fit["gain_db"])
            holdout_gain_db = float(holdout["gain_db"])
            if (
                float(fit.get("mad_db") or 0.0) > float(settings["max_gain_mad_db"])
                or float(holdout.get("mad_db") or 0.0) > float(settings["max_gain_mad_db"])
            ):
                reason = "unstable_frequency_estimates"
            elif abs(fit_gain_db - holdout_gain_db) > float(
                settings["max_fit_holdout_delta_db"]
            ):
                reason = "fit_holdout_disagreement"
            elif abs(fit_gain_db) > (
                float(settings["max_abs_gain_db"])
                + float(settings["gain_cap_tolerance_db"])
            ):
                reason = "gain_out_of_safe_range"
            elif abs(fit_gain_db) < float(settings["min_abs_gain_db"]):
                reason = "gain_not_needed"
            else:
                gain_db = float(
                    np.clip(
                        fit_gain_db,
                        -float(settings["max_abs_gain_db"]),
                        float(settings["max_abs_gain_db"]),
                    )
                )

    usable = reason == "ok"
    if not usable:
        gain_db = 0.0
    gains = np.full(bands, gain_db, dtype=np.float32)
    adapted = reference * np.float32(_linear(gain_db))
    before = _logmel_distance(
        reference, mixture, sample_rate, n_fft=n_fft, hop=hop, bands=bands
    )
    after = _logmel_distance(
        adapted, mixture, sample_rate, n_fft=n_fft, hop=hop, bands=bands
    )
    fit_gain = float(fit.get("gain_db") or 0.0)
    holdout_gain = float(holdout.get("gain_db") or 0.0)
    baseline_error = abs(holdout_gain)
    validation_improvement = (
        (1.0 - abs(holdout_gain - gain_db) / max(baseline_error, 0.25)) * 100.0
        if usable
        else 0.0
    )
    support = {
        "fit_bins": int(fit.get("support_bins") or 0),
        "holdout_bins": int(holdout.get("support_bins") or 0),
        "fit_candidate_bins": int(fit.get("candidate_bins") or 0),
        "holdout_candidate_bins": int(holdout.get("candidate_bins") or 0),
        "fit_fraction": float(fit.get("support_fraction") or 0.0),
        "holdout_fraction": float(holdout.get("support_fraction") or 0.0),
    }
    summary = {
        "verdict": "ok" if usable else reason,
        "usable": usable,
        "reason": reason,
        "recommended_mode": "algorithmic" if usable else "raw",
        "median_gain_db": float(gain_db),
        "min_gain_db": float(gain_db),
        "max_gain_db": float(gain_db),
        "raw_fit_gain_db": fit_gain,
        "holdout_gain_db": holdout_gain,
        "gain_mad_db": float(fit.get("mad_db") or 0.0),
        "mad_db": float(fit.get("mad_db") or 0.0),
        "holdout_gain_mad_db": float(holdout.get("mad_db") or 0.0),
        "fit_holdout_delta_db": abs(fit_gain - holdout_gain),
        "median_coherence": float(fit.get("median_coherence") or 0.0),
        "holdout_median_coherence": float(holdout.get("median_coherence") or 0.0),
        "support": support,
        "support_bins": min(support["fit_bins"], support["holdout_bins"]),
        "support_fraction": min(support["fit_fraction"], support["holdout_fraction"]),
        "max_abs_gain_db": float(settings["max_abs_gain_db"]),
        # Kept for the existing preview selector.  This is a held-out
        # cross-transfer validation metric, not an EN+RU magnitude comparison.
        "distance_reduction_percent": float(validation_improvement),
    }
    return ReferenceAdapterProfile(
        sample_rate=int(sample_rate),
        n_fft=n_fft,
        hop=hop,
        bands=bands,
        band_centers_hz=[float(value) for value in centers_hz],
        band_gains_db=[float(value) for value in gains],
        common_gain_db=float(gain_db),
        delay_sec=float(delay),
        delay_confidence=float(delay_confidence),
        distance_before=before,
        distance_after_estimate=after,
        summary=summary,
    )


def apply_profile(
    reference: np.ndarray,
    profile: ReferenceAdapterProfile,
) -> np.ndarray:
    values = np.asarray(reference, dtype=np.float32)
    safe, _reason = _profile_application_safety(profile)
    if not safe or not np.all(np.isfinite(values)):
        return values.copy()
    if profile.mode == PROFILE_MODE:
        # A scalar multiply is exactly linear and independent of chunk size.
        return (values * np.float32(_linear(profile.common_gain_db))).astype(np.float32)
    spectrum = _stft(values, profile.n_fft, profile.hop)
    gains_db = np.asarray(profile.band_gains_db, dtype=np.float32)
    centers = np.asarray(profile.band_centers_hz, dtype=np.float32)
    frequency_gain = _band_to_frequency_gain(
        gains_db, profile.sample_rate, profile.n_fft, centers
    )
    # Deliberately no peak normalization: signal-dependent normalization made
    # the old result depend on block boundaries and changed voice pitch/timbre
    # downstream.  File APIs reject a profile if its linear result would clip.
    return _istft(
        spectrum * frequency_gain[:, None],
        profile.n_fft,
        profile.hop,
        len(values),
    ).astype(np.float32)


def _atomic_copy(source: Path, target: Path) -> None:
    source = Path(source).resolve()
    target = Path(target).resolve()
    if source == target:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.copy.partial")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _write_flac_atomic(path: Path, values: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    sf.write(str(temporary), values, sample_rate, format="FLAC", subtype="PCM_16")
    os.replace(temporary, path)


def adapt_reference_file(
    reference_path: Path,
    mixture_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
    n_fft: int | None = None,
    hop: int | None = None,
    bands: int | None = None,
    config: dict[str, Any] | None = None,
    calibration_reference_path: Path | None = None,
) -> dict[str, Any]:
    reference_path = Path(reference_path).resolve()
    calibration_reference_path = Path(
        calibration_reference_path or reference_path
    ).resolve()
    mixture_path = Path(mixture_path).resolve()
    output_path = Path(output_path).resolve()
    source, source_rate = _read_mono(reference_path)
    calibration_reference, calibration_rate = _read_mono(calibration_reference_path)
    mixture, mix_rate = _read_mono(mixture_path)
    if mix_rate != calibration_rate:
        mixture = audio_io.resample(mixture[:, None], mix_rate, calibration_rate)[:, 0]
    calibration_reference, mixture = _match_length(calibration_reference, mixture)
    profile = fit_algorithmic_profile(
        calibration_reference,
        mixture,
        calibration_rate,
        n_fft=n_fft,
        hop=hop,
        bands=bands,
        config=config,
    )
    settings = _adapter_config(config)
    safe, application_reason = _profile_application_safety(profile)
    if source_rate != profile.sample_rate:
        safe = False
        application_reason = "sample_rate_mismatch"
    adapted = apply_profile(source, profile) if safe else source.copy()
    peak = float(np.max(np.abs(adapted))) if len(adapted) else 0.0
    if safe and peak > float(settings["max_output_peak"]):
        safe = False
        application_reason = "would_clip"
        adapted = source.copy()
        peak = float(np.max(np.abs(adapted))) if len(adapted) else 0.0
    if safe:
        _write_flac_atomic(output_path, adapted, source_rate)
    else:
        # Identity means identity: preserve the exact raw encoded file instead
        # of introducing another PCM quantisation pass.
        _atomic_copy(reference_path, output_path)

    report = profile.to_json()
    if not safe:
        report["summary"] = dict(report.get("summary") or {})
        report["summary"]["usable"] = False
        report["summary"]["recommended_mode"] = "raw"
        report["summary"]["reason"] = application_reason
        report["summary"]["verdict"] = application_reason
    report.update(
        {
            "reference": str(reference_path),
            "calibration_reference": str(calibration_reference_path),
            "mixture": str(mixture_path),
            "output": str(output_path),
            "applied": bool(safe),
            "application_reason": application_reason,
            "reference_rms_db": _db(_rms(source)),
            "adapted_rms_db": _db(_rms(adapted)),
            "mixture_rms_db": _db(_rms(mixture)),
            "peak": peak,
        }
    )
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


def adapt_reference_file_with_profile(
    reference_path: Path,
    output_path: Path,
    profile_data: dict[str, Any],
    *,
    report_path: Path | None = None,
    block_sec: float = 60.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_path = Path(reference_path).resolve()
    output_path = Path(output_path).resolve()
    profile = ReferenceAdapterProfile.from_json(profile_data)
    settings = _adapter_config(config)
    info = sf.info(str(reference_path))
    rate = int(info.samplerate)
    safe, application_reason = _profile_application_safety(profile)
    if rate != profile.sample_rate:
        safe = False
        application_reason = "sample_rate_mismatch"
    total = int(info.frames)
    peak = 0.0
    if not safe:
        _atomic_copy(reference_path, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.partial")
        block_frames = max(rate, int(round(rate * block_sec)))
        margin = max(profile.n_fft * 4, int(round(rate * 0.25)))
        clipped = False
        try:
            with sf.SoundFile(str(reference_path)) as reader:
                with sf.SoundFile(
                    str(temporary),
                    mode="w",
                    samplerate=rate,
                    channels=1,
                    format="FLAC",
                    subtype="PCM_16",
                ) as writer:
                    cursor = 0
                    while cursor < total:
                        frames = min(block_frames, total - cursor)
                        read_start = max(0, cursor - margin)
                        read_stop = min(total, cursor + frames + margin)
                        reader.seek(read_start)
                        block = reader.read(
                            read_stop - read_start,
                            dtype="float32",
                            always_2d=True,
                        )
                        adapted = apply_profile(_mono(block), profile)
                        trim_left = cursor - read_start
                        segment = adapted[trim_left : trim_left + frames]
                        if len(segment) < frames:
                            segment = np.pad(segment, (0, frames - len(segment)))
                        block_peak = (
                            float(np.max(np.abs(segment))) if len(segment) else 0.0
                        )
                        peak = max(peak, block_peak)
                        if block_peak > float(settings["max_output_peak"]):
                            clipped = True
                            break
                        writer.write(segment[:, None])
                        cursor += frames
            if clipped:
                application_reason = "would_clip"
                safe = False
                temporary.unlink(missing_ok=True)
                _atomic_copy(reference_path, output_path)
            else:
                os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "mode": "reference_adapter_profile_application_v2",
        "reference": str(reference_path),
        "output": str(output_path),
        "profile_summary": (
            profile_data.get("summary") or {}
            if isinstance(profile_data, dict)
            else {}
        ),
        "sample_rate": rate,
        "duration_sec": float(total / max(rate, 1)),
        "peak": peak,
        "applied": bool(safe),
        "application_reason": application_reason,
    }
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return report


def mismatch_report(
    reference_path: Path,
    mixture_path: Path,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    bands: int = DEFAULT_BANDS,
) -> dict[str, Any]:
    reference, ref_rate = _read_mono(Path(reference_path))
    mixture, mix_rate = _read_mono(Path(mixture_path))
    if mix_rate != ref_rate:
        mixture = audio_io.resample(mixture[:, None], mix_rate, ref_rate)[:, 0]
    reference, mixture = _match_length(reference, mixture)
    profile = fit_algorithmic_profile(
        reference, mixture, ref_rate, n_fft=n_fft, hop=hop, bands=bands
    )
    data = profile.to_json()
    reduction = float(data["summary"]["distance_reduction_percent"])
    delay = abs(float(data["delay_sec"]))
    gain_span = float(data["summary"]["max_gain_db"] - data["summary"]["min_gain_db"])
    if delay > 0.35 and profile.delay_confidence > 0.50:
        verdict = "alignment_broken"
    elif reduction >= 30.0 or gain_span >= 10.0:
        verdict = "masterings_differ"
    elif reduction >= 15.0:
        verdict = "mild_mastering_difference"
    else:
        verdict = "matched_or_unreliable"
    data.update(
        {
            "reference": str(Path(reference_path).resolve()),
            "mixture": str(Path(mixture_path).resolve()),
            "duration_sec": float(len(reference) / max(ref_rate, 1)),
            "verdict": verdict,
        }
    )
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mixture", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if args.diagnose_only:
        report = mismatch_report(args.reference, args.mixture)
        atomic_json(args.report, report)
    else:
        if args.output is None:
            raise SystemExit("--output is required unless --diagnose-only is used")
        report = adapt_reference_file(args.reference, args.mixture, args.output, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
