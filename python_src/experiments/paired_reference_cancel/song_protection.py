"""Conservative protection for sustained songs in a dubbed film mix.

The speech extractor can classify sung vocals as speech.  This module does not
try to recognise music from VAD or waveform correlation.  It requires four
independent observations over a sustained interval: an AudioSet classifier on
the English original, music behind the vocal, duration, and a clear loss of
vocal-band energy in the processed result.  Confirmed song-only spans can then
be restored from the aligned English original, while translated dialogue
inside a song remains in the processed result.
"""
from __future__ import annotations

import math
import os
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.storage import atomic_json, root_path, utc_now


SCHEMA_VERSION = 7
PREVIEW_CONTRACT_SCHEMA_VERSION = 1
PREVIEW_CONTRACT_KIND = "dubclean_preview_song_contract"
PREVIEW_CONTRACT_NAME = "song_protection_report.json"

AUDIOSET_SPEECH_INDICES = tuple(range(0, 16))
AUDIOSET_SINGING_INDICES = (
    27,  # Singing
    28,  # Choir
    29,  # Yodeling
    30,  # Chant
    31,  # Mantra
    32,  # Male singing
    33,  # Female singing
    34,  # Child singing
    35,  # Synthetic singing
    36,  # Rapping
    37,  # Humming
    254,  # Vocal music
    255,  # A capella
    266,  # Song
)
AUDIOSET_MUSIC_INDICES = (
    137,  # Music
    216,  # Pop music
    219,  # Rock music
    229,  # Country
    235,  # Jazz
    237,  # Classical music
    239,  # Electronic music
    246,  # Ambient music
    252,  # Music for children
    254,  # Vocal music
    256,  # Music of Africa
    258,  # Christian music
    260,  # Music of Asia
    266,  # Song
    267,  # Background music
    268,  # Theme music
)


def _defaults() -> dict[str, Any]:
    return {
        "enabled": True,
        "classifier_backend": "efficientat_mn04_audioset",
        "classifier_model_path": (
            "./third_party_models/efficientat/"
            "efficientat_mn04_audioset_waveform.pt"
        ),
        "classifier_sample_rate": 32000,
        "classifier_input_sec": 10.0,
        "classifier_batch_size": 12,
        "analysis_sample_rate": 8000,
        "window_sec": 10.0,
        "step_sec": 5.0,
        "minimum_interval_sec": 12.0,
        "maximum_gap_sec": 5.0,
        "preview_scene_sec": 35.0,
        "preview_min_sec": 30.0,
        "preview_max_sec": 40.0,
        "crossfade_sec": 0.35,
        "minimum_confidence": 0.78,
        "minimum_input_rms_db": -42.0,
        "minimum_classifier_singing_score": 0.12,
        "minimum_classifier_direct_singing_score": 0.07,
        "minimum_classifier_music_score": 0.65,
        "minimum_music_presence": 0.48,
        "minimum_speech_stem_share": 0.18,
        "minimum_relative_vocal_removal_db": 3.0,
        "minimum_removed_vocal_share": 0.25,
        # Alternative damage evidence for learned M&E whose mastering makes
        # the relative vocal/backing ratio unreliable.  Every strong gate must
        # pass; these values never weaken the standard relative-damage route.
        "minimum_strong_removed_vocal_share": 0.95,
        "minimum_strong_music_presence": 0.40,
        "minimum_strong_music_background_rms_db": -42.0,
        "minimum_strong_music_tonality": 0.80,
        "minimum_strong_speech_stem_share": 0.02,
        "minimum_strong_vocal_continuity": 0.35,
        "minimum_strong_vocal_tonality": 0.95,
        "minimum_strong_singing_over_speech_margin": 0.03,
        # A strict >= .95 window can anchor a sustained group whose adjacent
        # windows show the same learned song/music evidence and nearly the same
        # programme loss.  This does not lower the strict per-window route.
        "minimum_anchor_support_removed_vocal_share": 0.92,
        "minimum_anchor_support_removed_program_share": 0.93,
        "minimum_anchor_group_removed_vocal_share": 0.94,
        "minimum_anchor_group_removed_program_share": 0.94,
        "minimum_anchor_group_support_windows": 3,
        "minimum_anchor_group_span_sec": 12.0,
        "maximum_anchor_group_missed_windows": 1,
        "maximum_anchor_context_extension_sec": 20.0,
        "minimum_track_music_similarity": 0.78,
        "minimum_track_matched_window_share": 0.65,
        "dialogue_window_sec": 1.5,
        "dialogue_step_sec": 0.5,
        "minimum_dialogue_interval_sec": 0.6,
        "dialogue_merge_gap_sec": 0.5,
        "dialogue_boundary_margin_sec": 0.15,
        "synchronization_guard_sec": 0.85,
        "dialogue_timeline_guard_sec": 0.0,
        "minimum_dialogue_nonmatching_share": 0.30,
        "minimum_dialogue_score": 0.38,
        "strong_dialogue_score": 0.60,
        "minimum_dialogue_speech_likeness": 0.20,
        "minimum_translation_voice_share": 0.12,
        "minimum_translation_voice_program_share": 0.55,
        "minimum_translation_voice_vocal_to_backing_db": 6.0,
        "translation_voice_vocal_to_backing_ambiguity_margin_db": 0.5,
        "minimum_translation_voice_speech_likeness": 0.18,
        "minimum_translation_voice_rms_db": -42.0,
        "minimum_translation_voice_majority_window_share": 0.60,
        "minimum_ambiguous_nonmatching_share": 0.38,
        "ambiguous_dialogue_score_margin": 0.08,
        "dialogue_baseline_margin": 0.10,
        "dialogue_mad_multiplier": 2.5,
        "minimum_dialogue_stem_rms_db": -42.0,
    }


def resolved_config(config: dict[str, Any] | None) -> dict[str, Any]:
    value = {**_defaults(), **(config or {})}
    value["classifier_sample_rate"] = max(8000, int(value["classifier_sample_rate"]))
    value["classifier_input_sec"] = max(1.0, float(value["classifier_input_sec"]))
    value["classifier_batch_size"] = max(1, int(value["classifier_batch_size"]))
    value["analysis_sample_rate"] = max(4000, int(value["analysis_sample_rate"]))
    value["window_sec"] = max(0.5, float(value["window_sec"]))
    value["step_sec"] = max(0.25, min(float(value["step_sec"]), value["window_sec"]))
    value["minimum_interval_sec"] = max(value["window_sec"], float(value["minimum_interval_sec"]))
    value["maximum_gap_sec"] = max(0.0, float(value["maximum_gap_sec"]))
    value["preview_min_sec"] = max(5.0, float(value["preview_min_sec"]))
    value["preview_max_sec"] = max(
        value["preview_min_sec"], float(value["preview_max_sec"])
    )
    value["preview_scene_sec"] = float(
        np.clip(
            float(value["preview_scene_sec"]),
            value["preview_min_sec"],
            value["preview_max_sec"],
        )
    )
    value["crossfade_sec"] = max(0.0, float(value["crossfade_sec"]))
    value["dialogue_window_sec"] = max(0.5, float(value["dialogue_window_sec"]))
    value["dialogue_step_sec"] = max(
        0.1,
        min(float(value["dialogue_step_sec"]), value["dialogue_window_sec"]),
    )
    value["minimum_dialogue_interval_sec"] = max(
        0.1, float(value["minimum_dialogue_interval_sec"])
    )
    value["dialogue_merge_gap_sec"] = max(
        0.0, float(value["dialogue_merge_gap_sec"])
    )
    value["dialogue_boundary_margin_sec"] = max(
        0.0, float(value["dialogue_boundary_margin_sec"])
    )
    value["synchronization_guard_sec"] = max(
        0.0, float(value["synchronization_guard_sec"])
    )
    value["dialogue_timeline_guard_sec"] = max(
        0.0, float(value["dialogue_timeline_guard_sec"])
    )
    for key in (
        "minimum_confidence",
        "minimum_classifier_singing_score",
        "minimum_classifier_direct_singing_score",
        "minimum_classifier_music_score",
        "minimum_music_presence",
        "minimum_speech_stem_share",
        "minimum_removed_vocal_share",
        "minimum_strong_removed_vocal_share",
        "minimum_strong_music_presence",
        "minimum_strong_music_tonality",
        "minimum_strong_speech_stem_share",
        "minimum_strong_vocal_continuity",
        "minimum_strong_vocal_tonality",
        "minimum_strong_singing_over_speech_margin",
        "minimum_anchor_support_removed_vocal_share",
        "minimum_anchor_support_removed_program_share",
        "minimum_anchor_group_removed_vocal_share",
        "minimum_anchor_group_removed_program_share",
        "minimum_track_music_similarity",
        "minimum_track_matched_window_share",
        "minimum_dialogue_nonmatching_share",
        "minimum_dialogue_score",
        "strong_dialogue_score",
        "minimum_dialogue_speech_likeness",
        "minimum_translation_voice_share",
        "minimum_translation_voice_program_share",
        "minimum_translation_voice_speech_likeness",
        "minimum_translation_voice_majority_window_share",
        "minimum_ambiguous_nonmatching_share",
    ):
        value[key] = float(np.clip(value[key], 0.0, 1.0))
    value["ambiguous_dialogue_score_margin"] = max(
        0.0, float(value["ambiguous_dialogue_score_margin"])
    )
    value["minimum_strong_music_background_rms_db"] = float(
        value["minimum_strong_music_background_rms_db"]
    )
    value["minimum_anchor_group_support_windows"] = max(
        1, int(value["minimum_anchor_group_support_windows"])
    )
    value["minimum_anchor_group_span_sec"] = max(
        0.0, float(value["minimum_anchor_group_span_sec"])
    )
    value["maximum_anchor_group_missed_windows"] = max(
        0, int(value["maximum_anchor_group_missed_windows"])
    )
    value["maximum_anchor_context_extension_sec"] = max(
        0.0,
        float(value["maximum_anchor_context_extension_sec"]),
    )
    value["minimum_translation_voice_vocal_to_backing_db"] = float(
        value["minimum_translation_voice_vocal_to_backing_db"]
    )
    value["translation_voice_vocal_to_backing_ambiguity_margin_db"] = max(
        0.0,
        float(
            value[
                "translation_voice_vocal_to_backing_ambiguity_margin_db"
            ]
        ),
    )
    value["dialogue_baseline_margin"] = max(
        0.0, float(value["dialogue_baseline_margin"])
    )
    value["dialogue_mad_multiplier"] = max(
        0.0, float(value["dialogue_mad_multiplier"])
    )
    return value


def dialogue_timeline_guard(
    config: dict[str, Any] | None,
    *,
    manual_voice_delay_sec: float = 0.0,
    synchronize_speech: bool = False,
) -> float:
    """Resolve the same conservative dialogue guard for preview and full."""
    cfg = resolved_config(config)
    guard = abs(float(manual_voice_delay_sec or 0.0))
    if synchronize_speech:
        guard = max(
            guard,
            float(cfg["synchronization_guard_sec"]),
        )
    return float(guard)


class AudioSetSongClassifier:
    """Offline EfficientAT classifier used as mandatory singing evidence."""

    backend = "efficientat_mn04_audioset"

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch

        self.torch = torch
        self.sample_rate = int(cfg["classifier_sample_rate"])
        self.input_frames = int(
            round(self.sample_rate * float(cfg["classifier_input_sec"]))
        )
        self.model_path = root_path(str(cfg["classifier_model_path"]))
        if not self.model_path.is_file() or self.model_path.stat().st_size <= 0:
            raise FileNotFoundError(self.model_path)
        # Loading from bytes avoids torch.jit's Windows path limitation when
        # the portable directory contains Cyrillic characters.
        self.model = torch.jit.load(
            BytesIO(self.model_path.read_bytes()), map_location="cpu"
        ).eval()

    @staticmethod
    def _union_probability(values: np.ndarray) -> float:
        clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - clipped))

    def predict_many(self, waveforms: list[np.ndarray]) -> list[dict[str, float]]:
        if not waveforms:
            return []
        prepared: list[np.ndarray] = []
        for waveform in waveforms:
            value = _mono(waveform)
            if value.size < self.input_frames:
                value = np.pad(value, (0, self.input_frames - value.size))
            prepared.append(value[: self.input_frames])
        batch = self.torch.from_numpy(np.stack(prepared).astype(np.float32))
        with self.torch.no_grad():
            scores = np.asarray(self.model(batch).cpu(), dtype=np.float32)
        if scores.shape != (len(prepared), 527):
            raise RuntimeError(
                f"Классификатор песен вернул форму {scores.shape}, ожидалась "
                f"({len(prepared)}, 527)."
            )
        result: list[dict[str, float]] = []
        for values in scores:
            result.append(
                {
                    "singing_score": round(
                        self._union_probability(values[list(AUDIOSET_SINGING_INDICES)]),
                        4,
                    ),
                    "direct_singing_score": round(float(values[27]), 4),
                    "music_score": round(
                        self._union_probability(values[list(AUDIOSET_MUSIC_INDICES)]),
                        4,
                    ),
                    "speech_score": round(
                        float(np.max(values[list(AUDIOSET_SPEECH_INDICES)])), 4
                    ),
                    "vocal_music_score": round(float(values[254]), 4),
                    "song_score": round(float(values[266]), 4),
                }
            )
        return result


def _mono(data: np.ndarray) -> np.ndarray:
    value = np.asarray(data, dtype=np.float32)
    if value.ndim == 2:
        value = np.mean(value, axis=1)
    return np.nan_to_num(value, copy=False)


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(data, dtype=np.float32)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(
        data,
        int(target_rate) // divisor,
        int(source_rate) // divisor,
    ).astype(np.float32)


def _read_window(
    handle: sf.SoundFile,
    start_sec: float,
    duration_sec: float,
    target_rate: int,
) -> np.ndarray:
    start = max(0, int(round(start_sec * handle.samplerate)))
    frames = max(1, int(round(duration_sec * handle.samplerate)))
    handle.seek(min(start, len(handle)))
    data = _mono(handle.read(frames, dtype="float32", always_2d=True))
    if data.size < frames:
        data = np.pad(data, (0, frames - data.size))
    return _resample(data, int(handle.samplerate), target_rate)


def _spectral_features(data: np.ndarray, rate: int) -> dict[str, float]:
    values = _mono(data)
    rms = float(np.sqrt(np.mean(values * values) + 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    n_fft = 512 if rate <= 12000 else 1024
    hop = max(1, n_fft // 4)
    if values.size < n_fft:
        values = np.pad(values, (0, n_fft - values.size))
    frequencies, _, spectrum = signal.stft(
        values,
        fs=rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.maximum(np.abs(spectrum).astype(np.float64) ** 2, 1e-12)
    vocal_mask = (frequencies >= 120.0) & (frequencies <= min(3500.0, rate * 0.46))
    backing_mask = (
        ((frequencies >= 45.0) & (frequencies < 120.0))
        | ((frequencies > 3500.0) & (frequencies <= min(7600.0, rate * 0.48)))
    )
    full_mask = (frequencies >= 45.0) & (frequencies <= min(7600.0, rate * 0.48))
    vocal = power[vocal_mask]
    backing = power[backing_mask] if np.any(backing_mask) else power[full_mask]
    full = power[full_mask]
    vocal_energy = float(np.mean(vocal))
    backing_energy = float(np.mean(backing))
    full_energy = float(np.mean(full))

    frame_energy = np.mean(vocal, axis=0)
    active_floor = max(float(np.percentile(frame_energy, 30.0)) * 1.7, 1e-10)
    active = frame_energy > active_floor
    if not np.any(active):
        active = frame_energy > 1e-10
    selected = vocal[:, active] if np.any(active) else vocal
    flatness = np.exp(np.mean(np.log(selected), axis=0)) / np.maximum(
        np.mean(selected, axis=0), 1e-12
    )
    peak_count = min(8, max(1, selected.shape[0] // 8))
    peak_share = np.sum(
        np.partition(selected, -peak_count, axis=0)[-peak_count:], axis=0
    ) / np.maximum(np.sum(selected, axis=0), 1e-12)
    tonal_frames = (flatness < 0.38) & (peak_share > 0.24)
    vocal_continuity = float(np.mean(active))
    vocal_tonality = float(
        np.clip(
            0.55 * np.mean(tonal_frames)
            + 0.25 * np.mean(1.0 - np.clip(flatness, 0.0, 1.0))
            + 0.20 * np.mean(np.clip(peak_share / 0.45, 0.0, 1.0)),
            0.0,
            1.0,
        )
    )
    backing_flatness = np.exp(np.mean(np.log(full), axis=0)) / np.maximum(
        np.mean(full, axis=0), 1e-12
    )
    music_tonality = float(np.mean(1.0 - np.clip(backing_flatness, 0.0, 1.0)))
    return {
        "rms_db": rms_db,
        "vocal_energy": vocal_energy,
        "backing_energy": backing_energy,
        "full_energy": full_energy,
        "vocal_continuity": vocal_continuity,
        "vocal_tonality": vocal_tonality,
        "music_tonality": music_tonality,
    }


def _magnitude_spectrum(data: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    values = _mono(data)
    n_fft = 512 if rate <= 12000 else 1024
    hop = max(1, n_fft // 4)
    if values.size < n_fft:
        values = np.pad(values, (0, n_fft - values.size))
    frequencies, _, spectrum = signal.stft(
        values,
        fs=rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    return frequencies, np.maximum(np.abs(spectrum).astype(np.float64), 1e-10)


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def _chroma_profile(frequencies: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    profile = np.zeros(12, dtype=np.float64)
    usable = (frequencies >= 55.0) & (frequencies <= 4000.0)
    for frequency, energy in zip(
        frequencies[usable],
        np.mean(magnitude[usable], axis=1),
        strict=True,
    ):
        midi = 69.0 + 12.0 * math.log2(max(float(frequency), 1e-6) / 440.0)
        profile[int(round(midi)) % 12] += float(energy)
    return profile


def _track_music_similarity(
    original: np.ndarray,
    dubbed: np.ndarray,
    rate: int,
    *,
    original_background: np.ndarray | None = None,
    dubbed_speech: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare aligned music beds using spectral/harmonic features.

    When the available stems are supplied, the English side is the restored
    music/effects bed and the dubbed side is a conservative magnitude-STFT
    residual after removing its speech/vocal stem.  No raw waveform
    correlation is used.
    """

    original_values = (
        original_background if original_background is not None else original
    )
    frequencies, original_magnitude = _magnitude_spectrum(original_values, rate)
    _, dubbed_magnitude = _magnitude_spectrum(dubbed, rate)
    speech_subtraction_gain: float | None = None
    comparison_basis = "full_mix_spectral_features"
    if dubbed_speech is not None:
        _, dubbed_speech_magnitude = _magnitude_spectrum(dubbed_speech, rate)
        frames = min(
            original_magnitude.shape[1],
            dubbed_magnitude.shape[1],
            dubbed_speech_magnitude.shape[1],
        )
        original_magnitude = original_magnitude[:, :frames]
        dubbed_magnitude = dubbed_magnitude[:, :frames]
        dubbed_speech_magnitude = dubbed_speech_magnitude[:, :frames]
        speech_subtraction_gain = float(
            np.sum(dubbed_magnitude * dubbed_speech_magnitude)
            / max(np.sum(dubbed_speech_magnitude * dubbed_speech_magnitude), 1e-12)
        )
        speech_subtraction_gain = float(
            np.clip(speech_subtraction_gain, 0.15, 4.0)
        )
        dubbed_magnitude = np.maximum(
            dubbed_magnitude
            - speech_subtraction_gain * dubbed_speech_magnitude,
            1e-10,
        )
        comparison_basis = "english_background_vs_dubbed_spectral_residual"
    else:
        frames = min(original_magnitude.shape[1], dubbed_magnitude.shape[1])
        original_magnitude = original_magnitude[:, :frames]
        dubbed_magnitude = dubbed_magnitude[:, :frames]

    original_magnitude = original_magnitude[:, :frames]
    dubbed_magnitude = dubbed_magnitude[:, :frames]
    usable = (frequencies >= 55.0) & (frequencies <= min(4000.0, rate * 0.48))
    original_log = np.log1p(30.0 * original_magnitude[usable])
    dubbed_log = np.log1p(30.0 * dubbed_magnitude[usable])
    envelope_similarity = _cosine_similarity(
        np.median(original_log, axis=1),
        np.median(dubbed_log, axis=1),
    )
    frame_similarities = [
        _cosine_similarity(original_log[:, index], dubbed_log[:, index])
        for index in range(frames)
    ]
    frame_similarity = (
        float(np.median(frame_similarities)) if frame_similarities else 0.0
    )
    hop = max(1, (512 if rate <= 12000 else 1024) // 4)
    block_frames = max(1, int(round(rate / hop)))
    block_similarities: list[float] = []
    for start in range(0, frames, block_frames):
        end = min(frames, start + block_frames)
        if end - start < max(2, block_frames // 3):
            continue
        block_similarities.append(
            _cosine_similarity(
                np.median(original_log[:, start:end], axis=1),
                np.median(dubbed_log[:, start:end], axis=1),
            )
        )
    matched_window_share = (
        float(np.mean(np.asarray(block_similarities) >= 0.72))
        if block_similarities
        else 0.0
    )
    chroma_similarity = _cosine_similarity(
        _chroma_profile(frequencies, original_magnitude),
        _chroma_profile(frequencies, dubbed_magnitude),
    )
    original_energy = np.sum(original_magnitude[usable], axis=0)
    dubbed_energy = np.sum(dubbed_magnitude[usable], axis=0)
    original_onsets = np.maximum(
        np.diff(np.log1p(original_energy), prepend=np.log1p(original_energy[:1])),
        0.0,
    )
    dubbed_onsets = np.maximum(
        np.diff(np.log1p(dubbed_energy), prepend=np.log1p(dubbed_energy[:1])),
        0.0,
    )
    if (
        float(np.linalg.norm(original_onsets)) <= 1e-8
        and float(np.linalg.norm(dubbed_onsets)) <= 1e-8
    ):
        onset_similarity = 1.0
    else:
        onset_similarity = _cosine_similarity(original_onsets, dubbed_onsets)
    combined = float(
        np.clip(
            0.32 * envelope_similarity
            + 0.28 * frame_similarity
            + 0.22 * chroma_similarity
            + 0.10 * onset_similarity
            + 0.08 * matched_window_share,
            0.0,
            1.0,
        )
    )
    result: dict[str, float | str] = {
        "music_similarity": round(combined, 4),
        "spectral_envelope_similarity": round(envelope_similarity, 4),
        "spectral_frame_similarity": round(frame_similarity, 4),
        "chroma_similarity": round(chroma_similarity, 4),
        "onset_envelope_similarity": round(onset_similarity, 4),
        "matched_window_share": round(matched_window_share, 4),
        "comparison_basis": comparison_basis,
    }
    if speech_subtraction_gain is not None:
        result["dubbed_speech_subtraction_gain"] = round(
            speech_subtraction_gain, 4
        )
    return result


def _spectral_dynamics(
    magnitude: np.ndarray,
    rate: int,
) -> dict[str, float]:
    """Measure speech-like modulation and spectral changes without VAD."""

    values = np.asarray(magnitude, dtype=np.float64)
    frame_energy = np.sum(values, axis=0)
    active_floor = max(float(np.percentile(frame_energy, 35.0)) * 1.5, 1e-9)
    active = frame_energy > active_floor
    normalised_spectrum = values / np.maximum(
        np.sum(values, axis=0, keepdims=True), 1e-10
    )
    spectral_flux = (
        float(
            np.mean(
                np.sum(
                    np.maximum(np.diff(normalised_spectrum, axis=1), 0.0),
                    axis=0,
                )
            )
        )
        if normalised_spectrum.shape[1] > 1
        else 0.0
    )
    flux_score = float(np.clip(spectral_flux / 0.28, 0.0, 1.0))
    active_indices = np.flatnonzero(active)
    adjacent_similarities = [
        _cosine_similarity(
            normalised_spectrum[:, left],
            normalised_spectrum[:, right],
        )
        for left, right in zip(
            active_indices[:-1], active_indices[1:], strict=True
        )
        if right == left + 1
    ]
    spectral_stability = (
        float(np.median(adjacent_similarities))
        if adjacent_similarities
        else 1.0
    )
    active_energy = frame_energy[active]
    activity_variation = (
        float(
            np.clip(
                np.std(active_energy)
                / max(float(np.mean(active_energy)), 1e-10)
                / 1.5,
                0.0,
                1.0,
            )
        )
        if active_energy.size
        else 0.0
    )
    hop = max(1, (512 if rate <= 12000 else 1024) // 4)
    envelope = np.log1p(frame_energy)
    envelope = envelope - float(np.mean(envelope))
    modulation_frequency = np.fft.rfftfreq(
        max(envelope.size, 1), d=hop / float(rate)
    )
    modulation_power = np.abs(np.fft.rfft(envelope)) ** 2
    speech_modulation = (modulation_frequency >= 2.0) & (
        modulation_frequency <= 8.0
    )
    useful_modulation = (modulation_frequency >= 0.5) & (
        modulation_frequency <= 12.0
    )
    modulation_score = float(
        np.sum(modulation_power[speech_modulation])
        / max(np.sum(modulation_power[useful_modulation]), 1e-12)
    )
    speech_likeness = float(
        np.clip(
            0.35 * modulation_score
            + 0.30 * flux_score
            + 0.20 * (1.0 - spectral_stability)
            + 0.15 * activity_variation,
            0.0,
            1.0,
        )
    )
    return {
        "speech_likeness": speech_likeness,
        "speech_band_modulation": modulation_score,
        "spectral_flux": spectral_flux,
        "spectral_stability": spectral_stability,
        "activity_variation": activity_variation,
    }


def _stem_difference_features(
    original_speech: np.ndarray,
    dubbed_speech: np.ndarray,
    rate: int,
    translation_voice: np.ndarray | None = None,
    dubbed_program: np.ndarray | None = None,
) -> dict[str, Any]:
    frequencies, original_magnitude = _magnitude_spectrum(original_speech, rate)
    _, dubbed_magnitude = _magnitude_spectrum(dubbed_speech, rate)
    frames = min(original_magnitude.shape[1], dubbed_magnitude.shape[1])
    usable = (frequencies >= 90.0) & (frequencies <= min(3800.0, rate * 0.48))
    original_magnitude = original_magnitude[usable, :frames]
    dubbed_magnitude = dubbed_magnitude[usable, :frames]
    scale = float(
        np.sum(original_magnitude * dubbed_magnitude)
        / max(np.sum(original_magnitude * original_magnitude), 1e-12)
    )
    scale = float(np.clip(scale, 0.05, 20.0))
    matched_original = original_magnitude * scale
    nonmatching = np.maximum(dubbed_magnitude - matched_original, 0.0)
    nonmatching_share = float(
        np.sum(nonmatching) / max(np.sum(dubbed_magnitude), 1e-12)
    )
    spectral_similarity = _cosine_similarity(
        np.log1p(40.0 * matched_original),
        np.log1p(40.0 * dubbed_magnitude),
    )
    residual_dynamics = _spectral_dynamics(nonmatching, rate)
    speech_likeness = float(residual_dynamics["speech_likeness"])
    dubbed_values = _mono(dubbed_speech).astype(np.float64, copy=False)
    dubbed_rms_linear = float(
        np.sqrt(np.mean(dubbed_values * dubbed_values) + 1e-12)
    )
    dubbed_rms = float(
        20.0
        * math.log10(
            max(
                dubbed_rms_linear,
                1e-8,
            )
        )
    )
    residual_dialogue_score = float(
        np.clip(
            0.55 * nonmatching_share
            + 0.25 * (1.0 - spectral_similarity)
            + 0.20 * speech_likeness,
            0.0,
            1.0,
        )
    )
    translation_available = translation_voice is not None
    translation_rms_db = -160.0
    translation_share = 0.0
    translation_program_share = 0.0
    translation_vocal_to_backing_db = -160.0
    dubbed_program_rms_db = -160.0
    translation_evidence_score = 0.0
    translation_dynamics = {
        "speech_likeness": 0.0,
        "speech_band_modulation": 0.0,
        "spectral_flux": 0.0,
        "spectral_stability": 1.0,
        "activity_variation": 0.0,
    }
    if translation_voice is not None:
        translation_values = _mono(translation_voice).astype(
            np.float64, copy=False
        )
        translation_rms_linear = float(
            np.sqrt(np.mean(translation_values * translation_values) + 1e-12)
        )
        translation_rms_db = float(
            20.0 * math.log10(max(translation_rms_linear, 1e-8))
        )
        translation_spectral = _spectral_features(
            translation_values,
            rate,
        )
        translation_vocal_to_backing_db = _ratio_db(
            translation_spectral["vocal_energy"],
            translation_spectral["backing_energy"],
        )
        translation_share = float(
            np.clip(
                translation_rms_linear / max(dubbed_rms_linear, 1e-8),
                0.0,
                1.0,
            )
        )
        if dubbed_program is not None:
            program_values = _mono(dubbed_program).astype(
                np.float64, copy=False
            )
            comparable_frames = min(
                translation_values.size,
                program_values.size,
            )
            if comparable_frames > 0:
                program_rms_linear = float(
                    np.sqrt(
                        np.mean(
                            program_values[:comparable_frames]
                            * program_values[:comparable_frames]
                        )
                        + 1e-12
                    )
                )
                dubbed_program_rms_db = float(
                    20.0 * math.log10(max(program_rms_linear, 1e-8))
                )
                # Root-energy share is deliberately gain-only. It does not
                # correlate waveforms, so leaked singing cannot look like
                # translated dialogue just because it dominates a speech stem.
                translation_program_share = float(
                    np.clip(
                        translation_rms_linear
                        / max(program_rms_linear, 1e-8),
                        0.0,
                        1.0,
                    )
                )
        translation_frequencies, translation_magnitude = _magnitude_spectrum(
            translation_values, rate
        )
        translation_usable = (
            (translation_frequencies >= 90.0)
            & (
                translation_frequencies
                <= min(3800.0, rate * 0.48)
            )
        )
        translation_dynamics = _spectral_dynamics(
            translation_magnitude[translation_usable],
            rate,
        )
        translation_evidence_score = float(
            np.clip(
                0.65 * np.clip(translation_share / 0.35, 0.0, 1.0)
                + 0.35 * float(translation_dynamics["speech_likeness"]),
                0.0,
                1.0,
            )
        )
    dialogue_score = max(
        residual_dialogue_score,
        translation_evidence_score,
    )
    return {
        "dialogue_score": round(dialogue_score, 4),
        "residual_dialogue_score": round(residual_dialogue_score, 4),
        "nonmatching_share": round(nonmatching_share, 4),
        "spectral_similarity": round(spectral_similarity, 4),
        "speech_likeness": round(speech_likeness, 4),
        "speech_band_modulation": round(
            float(residual_dynamics["speech_band_modulation"]), 4
        ),
        "spectral_flux": round(float(residual_dynamics["spectral_flux"]), 4),
        "spectral_stability": round(
            float(residual_dynamics["spectral_stability"]), 4
        ),
        "activity_variation": round(
            float(residual_dynamics["activity_variation"]), 4
        ),
        "matched_gain": round(scale, 4),
        "dubbed_stem_rms_db": round(dubbed_rms, 3),
        "translation_voice_available": translation_available,
        "translation_voice_rms_db": round(translation_rms_db, 3),
        "translation_voice_share": round(translation_share, 4),
        "dubbed_program_rms_db": round(dubbed_program_rms_db, 3),
        "translation_voice_program_share": round(
            translation_program_share, 4
        ),
        "translation_voice_vocal_to_backing_db": round(
            translation_vocal_to_backing_db, 3
        ),
        "translation_voice_speech_likeness": round(
            float(translation_dynamics["speech_likeness"]), 4
        ),
        "translation_voice_speech_band_modulation": round(
            float(translation_dynamics["speech_band_modulation"]), 4
        ),
        "translation_voice_spectral_flux": round(
            float(translation_dynamics["spectral_flux"]), 4
        ),
        "translation_voice_spectral_stability": round(
            float(translation_dynamics["spectral_stability"]), 4
        ),
        "translation_voice_activity_variation": round(
            float(translation_dynamics["activity_variation"]), 4
        ),
        "translation_voice_evidence_score": round(
            translation_evidence_score, 4
        ),
    }


def _merge_dialogue_windows(
    windows: list[dict[str, Any]],
    song_start: float,
    song_end: float,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    selected = [item for item in windows if item.get("block_restoration")]
    if not selected:
        return []
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in selected:
        if (
            current
            and float(item["start_sec"]) - float(current[-1]["end_sec"])
            > float(cfg["dialogue_merge_gap_sec"])
        ):
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    result: list[dict[str, Any]] = []
    timeline_guard = float(cfg["dialogue_timeline_guard_sec"])
    margin = float(cfg["dialogue_boundary_margin_sec"]) + timeline_guard
    for group in groups:
        start = max(song_start, float(group[0]["start_sec"]) - margin)
        end = min(song_end, float(group[-1]["end_sec"]) + margin)
        if end - start < float(cfg["minimum_dialogue_interval_sec"]):
            continue
        confirmed_dialogue = any(
            bool(item.get("dialogue_candidate")) for item in group
        )
        translation_voice_evidence = any(
            bool(item["features"].get("translation_voice_available"))
            and float(
                item["features"].get("translation_voice_evidence_score") or 0.0
            )
            >= float(cfg["minimum_dialogue_score"])
            for item in group
        )
        result.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "timeline_guard_sec": round(timeline_guard, 3),
                "decision": (
                    "keep_processed_translation"
                    if confirmed_dialogue
                    else "keep_processed_low_confidence"
                ),
                "reason": (
                    (
                        "очищенная русская речевая дорожка и спектральная "
                        "динамика подтверждают переводную речь внутри песни"
                        if translation_voice_evidence
                        else (
                            "в дорожке перевода найдена дополнительная речь, "
                            "не совпадающая с английским вокальным слоем"
                        )
                    )
                    if confirmed_dialogue
                    else (
                        "есть несовпадающая голосовая активность, но уверенности "
                        "недостаточно; автоматическая замена отключена"
                    )
                ),
                "mean_dialogue_score": round(
                    float(
                        np.mean(
                            [item["features"]["dialogue_score"] for item in group]
                        )
                    ),
                    4,
                ),
                "mean_nonmatching_share": round(
                    float(
                        np.mean(
                            [
                                item["features"]["nonmatching_share"]
                                for item in group
                            ]
                        )
                    ),
                    4,
                ),
                "mean_speech_likeness": round(
                    float(
                        np.mean(
                            [
                                item["features"]["speech_likeness"]
                                for item in group
                            ]
                        )
                    ),
                    4,
                ),
                "mean_translation_voice_share": round(
                    float(
                        np.mean(
                            [
                                item["features"].get(
                                    "translation_voice_share", 0.0
                                )
                                for item in group
                            ]
                        )
                    ),
                    4,
                ),
                "mean_translation_voice_program_share": round(
                    float(
                        np.mean(
                            [
                                item["features"].get(
                                    "translation_voice_program_share",
                                    0.0,
                                )
                                for item in group
                            ]
                        )
                    ),
                    4,
                ),
                "mean_translation_voice_evidence_score": round(
                    float(
                        np.mean(
                            [
                                item["features"].get(
                                    "translation_voice_evidence_score", 0.0
                                )
                                for item in group
                            ]
                        )
                    ),
                    4,
                ),
                "evidence_sources": (
                    [
                        "translation_voice_energy_share",
                        "translation_voice_program_share",
                        "translation_voice_spectral_dynamics",
                        "aligned_speech_spectral_difference",
                    ]
                    if translation_voice_evidence
                    else ["aligned_speech_spectral_difference"]
                ),
            }
        )
    return result


def _restoration_segments(
    song_start: float,
    song_end: float,
    dialogue_intervals: list[dict[str, Any]],
    source_kind: str,
) -> list[dict[str, Any]]:
    cursor = song_start
    result: list[dict[str, Any]] = []
    for dialogue in dialogue_intervals:
        start = max(song_start, float(dialogue["start_sec"]))
        end = min(song_end, float(dialogue["end_sec"]))
        if start - cursor > 0.05:
            result.append(
                {
                    "start_sec": round(cursor, 3),
                    "end_sec": round(start, 3),
                    "restore_source": source_kind,
                }
            )
        cursor = max(cursor, end)
    if song_end - cursor > 0.05:
        result.append(
            {
                "start_sec": round(cursor, 3),
                "end_sec": round(song_end, 3),
                "restore_source": source_kind,
            }
        )
    return result


def _ratio_db(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12))


def _window_decision(
    source: dict[str, float],
    speech: dict[str, float],
    background: dict[str, float],
    processed: dict[str, float],
    classifier: dict[str, float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    source_relative_vocal = _ratio_db(source["vocal_energy"], source["backing_energy"])
    processed_relative_vocal = _ratio_db(
        processed["vocal_energy"], processed["backing_energy"]
    )
    relative_removal_db = source_relative_vocal - processed_relative_vocal
    speech_share = float(
        np.clip(speech["vocal_energy"] / max(source["vocal_energy"], 1e-12), 0.0, 1.0)
    )
    background_level_share = float(
        np.clip(background["full_energy"] / max(source["full_energy"], 1e-12), 0.0, 1.0)
    )
    music_presence = float(
        np.clip(
            0.55 * background_level_share
            + 0.45 * background["music_tonality"],
            0.0,
            1.0,
        )
    )
    removed_vocal_share = float(
        np.clip(
            1.0 - processed["vocal_energy"] / max(source["vocal_energy"], 1e-12),
            0.0,
            1.0,
        )
    )
    source_program_energy = float(
        source.get(
            "full_energy",
            float(source["vocal_energy"]) + float(source["backing_energy"]),
        )
    )
    processed_program_energy = float(
        processed.get(
            "full_energy",
            float(processed["vocal_energy"])
            + float(processed["backing_energy"]),
        )
    )
    removed_program_share = float(
        np.clip(
            1.0
            - processed_program_energy / max(source_program_energy, 1e-12),
            0.0,
            1.0,
        )
    )
    singing_over_speech_margin = float(
        classifier["singing_score"] - classifier["speech_score"]
    )
    common_checks = {
        "audible_input": source["rms_db"] >= float(cfg["minimum_input_rms_db"]),
        "learned_singing": classifier["singing_score"]
        >= float(cfg["minimum_classifier_singing_score"]),
        "learned_direct_singing": classifier["direct_singing_score"]
        >= float(cfg["minimum_classifier_direct_singing_score"]),
        "learned_music": classifier["music_score"]
        >= float(cfg["minimum_classifier_music_score"]),
    }
    standard_checks = {
        "music_background": music_presence >= float(cfg["minimum_music_presence"]),
        "speech_stem_contains_vocal": speech_share >= float(cfg["minimum_speech_stem_share"]),
        "vocal_removal_is_clear": relative_removal_db >= float(cfg["minimum_relative_vocal_removal_db"]),
        "removed_vocal_share": removed_vocal_share >= float(cfg["minimum_removed_vocal_share"]),
    }
    strong_checks = {
        "strong_music_background": music_presence
        >= float(cfg["minimum_strong_music_presence"]),
        "strong_audible_music_background": background.get("rms_db", -160.0)
        >= float(cfg["minimum_strong_music_background_rms_db"]),
        "strong_music_tonality": background["music_tonality"]
        >= float(cfg["minimum_strong_music_tonality"]),
        "strong_speech_stem_evidence": speech_share
        >= float(cfg["minimum_strong_speech_stem_share"]),
        "strong_vocal_continuity": speech["vocal_continuity"]
        >= float(cfg["minimum_strong_vocal_continuity"]),
        "strong_vocal_tonality": speech["vocal_tonality"]
        >= float(cfg["minimum_strong_vocal_tonality"]),
        "strong_absolute_vocal_removal": removed_vocal_share
        >= float(cfg["minimum_strong_removed_vocal_share"]),
        "strong_singing_over_speech": singing_over_speech_margin
        >= float(cfg["minimum_strong_singing_over_speech_margin"]),
    }
    strong_non_damage_checks = {
        key: value
        for key, value in strong_checks.items()
        if key
        not in {
            "strong_absolute_vocal_removal",
            # This route exists specifically for the case where the processed
            # programme and its vocal were both removed. Requiring the damaged
            # M&E output to remain audible would make that evidence impossible
            # to use in a short preview scene.
            "strong_audible_music_background",
        }
    }
    programme_and_vocal_loss_checks = {
        "anchor_support_removed_vocal_share": removed_vocal_share
        >= float(cfg["minimum_anchor_support_removed_vocal_share"]),
        "anchor_support_removed_program_share": removed_program_share
        >= float(cfg["minimum_anchor_support_removed_program_share"]),
    }
    programme_and_vocal_loss_support = bool(
        all(common_checks.values())
        and all(strong_non_damage_checks.values())
        and all(programme_and_vocal_loss_checks.values())
    )
    programme_and_vocal_loss_anchor = bool(
        programme_and_vocal_loss_support
        and strong_checks["strong_absolute_vocal_removal"]
    )
    context_boundary_checks = {
        "context_audible_input": common_checks["audible_input"],
        "context_learned_singing": common_checks["learned_singing"],
        "context_learned_direct_singing": common_checks[
            "learned_direct_singing"
        ],
        "context_learned_music": common_checks["learned_music"],
        "context_strong_music_background": strong_checks[
            "strong_music_background"
        ],
        "context_strong_audible_music_background": strong_checks[
            "strong_audible_music_background"
        ],
        "context_strong_music_tonality": strong_checks[
            "strong_music_tonality"
        ],
        "context_strong_singing_over_speech": strong_checks[
            "strong_singing_over_speech"
        ],
    }
    context_boundary_window = bool(
        all(
            value
            for key, value in context_boundary_checks.items()
            if key != "context_strong_audible_music_background"
        )
    )
    standard_route = bool(all(standard_checks.values()))
    strong_absolute_route = bool(all(strong_checks.values()))
    candidate = bool(
        all(common_checks.values())
        and (standard_route or strong_absolute_route)
    )
    if standard_route and strong_absolute_route:
        damage_evidence_route = "standard_relative_and_strong_absolute"
    elif standard_route:
        damage_evidence_route = "standard_relative_damage"
    elif strong_absolute_route:
        damage_evidence_route = "strong_absolute_damage"
    else:
        damage_evidence_route = "none"

    common_margins = [
        np.clip(
            (
                classifier["singing_score"]
                - float(cfg["minimum_classifier_singing_score"])
            )
            / 0.35,
            0.0,
            1.0,
        ),
        np.clip(
            (
                classifier["direct_singing_score"]
                - float(cfg["minimum_classifier_direct_singing_score"])
            )
            / 0.25,
            0.0,
            1.0,
        ),
        np.clip(
            (
                classifier["music_score"]
                - float(cfg["minimum_classifier_music_score"])
            )
            / 0.30,
            0.0,
            1.0,
        ),
    ]
    standard_margins = [
        np.clip((music_presence - float(cfg["minimum_music_presence"])) / 0.30, 0.0, 1.0),
        np.clip((speech_share - float(cfg["minimum_speech_stem_share"])) / 0.50, 0.0, 1.0),
        np.clip((relative_removal_db - float(cfg["minimum_relative_vocal_removal_db"])) / 8.0, 0.0, 1.0),
        np.clip((removed_vocal_share - float(cfg["minimum_removed_vocal_share"])) / 0.50, 0.0, 1.0),
    ]
    strong_margins = [
        np.clip(
            (
                music_presence
                - float(cfg["minimum_strong_music_presence"])
            )
            / 0.20,
            0.0,
            1.0,
        ),
        np.clip(
            (
                float(background.get("rms_db", -160.0))
                - float(cfg["minimum_strong_music_background_rms_db"])
            )
            / 12.0,
            0.0,
            1.0,
        ),
        np.clip(
            (
                float(background["music_tonality"])
                - float(cfg["minimum_strong_music_tonality"])
            )
            / 0.20,
            0.0,
            1.0,
        ),
        np.clip(
            (
                speech_share
                - float(cfg["minimum_strong_speech_stem_share"])
            )
            / 0.16,
            0.0,
            1.0,
        ),
        np.clip(
            (
                float(speech["vocal_tonality"])
                - float(cfg["minimum_strong_vocal_tonality"])
            )
            / 0.05,
            0.0,
            1.0,
        ),
        np.clip(
            (
                float(speech["vocal_continuity"])
                - float(cfg["minimum_strong_vocal_continuity"])
            )
            / 0.45,
            0.0,
            1.0,
        ),
        np.clip(
            (
                removed_vocal_share
                - float(cfg["minimum_strong_removed_vocal_share"])
            )
            / max(
                0.05,
                1.0
                - float(cfg["minimum_strong_removed_vocal_share"]),
            ),
            0.0,
            1.0,
        ),
        np.clip(
            (
                singing_over_speech_margin
                - float(
                    cfg["minimum_strong_singing_over_speech_margin"]
                )
            )
            / 0.25,
            0.0,
            1.0,
        ),
    ]
    programme_and_vocal_loss_margins = [
        *strong_margins[:6],
        np.clip(
            (
                removed_vocal_share
                - float(cfg["minimum_anchor_support_removed_vocal_share"])
            )
            / max(
                0.05,
                1.0
                - float(cfg["minimum_anchor_support_removed_vocal_share"]),
            ),
            0.0,
            1.0,
        ),
        np.clip(
            (
                removed_program_share
                - float(cfg["minimum_anchor_support_removed_program_share"])
            )
            / max(
                0.05,
                1.0
                - float(cfg["minimum_anchor_support_removed_program_share"]),
            ),
            0.0,
            1.0,
        ),
        strong_margins[-1],
    ]

    def route_confidence(route_margins: list[float]) -> float:
        return float(
            np.clip(
                0.72
                + 0.28
                * float(np.mean([*common_margins, *route_margins])),
                0.0,
                1.0,
            )
        )

    eligible_confidences: list[float] = []
    if all(common_checks.values()) and standard_route:
        eligible_confidences.append(route_confidence(standard_margins))
    if all(common_checks.values()) and strong_absolute_route:
        eligible_confidences.append(route_confidence(strong_margins))
    confidence = max(eligible_confidences, default=0.0)
    programme_and_vocal_loss_confidence = (
        route_confidence(programme_and_vocal_loss_margins)
        if programme_and_vocal_loss_support
        else 0.0
    )
    checks = {
        **common_checks,
        **standard_checks,
        **strong_checks,
        **programme_and_vocal_loss_checks,
        **context_boundary_checks,
        "standard_relative_damage_route": standard_route,
        "strong_absolute_damage_route": strong_absolute_route,
        "programme_and_vocal_loss_support_route": (
            programme_and_vocal_loss_support
        ),
        "programme_and_vocal_loss_strict_anchor": (
            programme_and_vocal_loss_anchor
        ),
        "context_boundary_window": context_boundary_window,
        "context_boundary_extension": False,
    }
    return {
        "candidate": candidate,
        "confidence": round(confidence, 4),
        "programme_and_vocal_loss_support_confidence": round(
            programme_and_vocal_loss_confidence, 4
        ),
        "damage_evidence_route": damage_evidence_route,
        "checks": checks,
        "features": {
            "input_rms_db": round(source["rms_db"], 3),
            "vocal_continuity": round(speech["vocal_continuity"], 4),
            "vocal_tonality": round(speech["vocal_tonality"], 4),
            "music_presence": round(music_presence, 4),
            "music_background_rms_db": round(
                background.get("rms_db", -160.0), 3
            ),
            "music_background_tonality": round(
                background["music_tonality"], 4
            ),
            "speech_stem_share": round(speech_share, 4),
            "relative_vocal_removal_db": round(relative_removal_db, 3),
            "removed_vocal_share": round(removed_vocal_share, 4),
            "removed_program_share": round(removed_program_share, 4),
            "singing_over_speech_margin": round(
                singing_over_speech_margin, 4
            ),
            "standard_relative_damage_evidence": float(standard_route),
            "strong_absolute_damage_evidence": float(
                strong_absolute_route
            ),
            "programme_and_vocal_loss_evidence": float(
                programme_and_vocal_loss_support
            ),
            **classifier,
        },
    }


def _window_grid_index(
    item: dict[str, Any],
    cfg: dict[str, Any],
) -> int:
    try:
        return int(item["index"])
    except (KeyError, TypeError, ValueError):
        return int(
            round(
                float(item.get("start_sec") or 0.0)
                / max(float(cfg["step_sec"]), 1e-6)
            )
        )


def _promote_anchor_backed_groups(
    windows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    translation_voice_available: bool,
) -> None:
    """Promote only dense moderate-damage windows around a strict anchor.

    The strict >= .95 route remains unchanged.  A moderate window is usable
    only as part of a dense, sustained group with a strict anchor and matching
    full-programme loss.  This keeps uniform gain changes and isolated peaks
    from becoming restoration permission.
    """

    support_windows = sorted(
        (
            item
            for item in windows
            if bool(
                (item.get("checks") or {}).get(
                    "programme_and_vocal_loss_support_route"
                )
            )
        ),
        key=lambda item: (
            _window_grid_index(item, cfg),
            float(item.get("start_sec") or 0.0),
        ),
    )
    if not support_windows:
        return

    maximum_missed = int(cfg["maximum_anchor_group_missed_windows"])
    provisional_groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in support_windows:
        if current:
            previous_index = _window_grid_index(current[-1], cfg)
            current_index = _window_grid_index(item, cfg)
            if current_index - previous_index - 1 > maximum_missed:
                provisional_groups.append(current)
                current = []
        current.append(item)
    if current:
        provisional_groups.append(current)

    windows_by_index = {
        _window_grid_index(item, cfg): item
        for item in windows
    }
    group_removed_threshold = float(
        cfg["minimum_anchor_group_removed_vocal_share"]
    )
    group_program_threshold = float(
        cfg["minimum_anchor_group_removed_program_share"]
    )
    minimum_support_windows = int(cfg["minimum_anchor_group_support_windows"])
    minimum_span = float(cfg["minimum_anchor_group_span_sec"])
    maximum_context_extension = float(
        cfg["maximum_anchor_context_extension_sec"]
    )

    for group_number, group in enumerate(provisional_groups, start=1):
        indices = [_window_grid_index(item, cfg) for item in group]
        first_index = min(indices)
        last_index = max(indices)
        missing_windows = max(
            0,
            last_index - first_index + 1 - len(set(indices)),
        )
        core_start = min(float(item["start_sec"]) for item in group)
        core_end = max(float(item["end_sec"]) for item in group)
        core_span = core_end - core_start
        removed_vocal = np.asarray(
            [
                float((item.get("features") or {}).get(
                    "removed_vocal_share", 0.0
                ))
                for item in group
            ],
            dtype=np.float64,
        )
        removed_program = np.asarray(
            [
                float((item.get("features") or {}).get(
                    "removed_program_share", 0.0
                ))
                for item in group
            ],
            dtype=np.float64,
        )
        anchor_count = sum(
            bool(
                (item.get("checks") or {}).get(
                    "programme_and_vocal_loss_strict_anchor"
                )
            )
            for item in group
        )
        vocal_median = float(np.median(removed_vocal))
        vocal_mean = float(np.mean(removed_vocal))
        program_median = float(np.median(removed_program))
        program_mean = float(np.mean(removed_program))
        failures: list[str] = []
        if len(group) < minimum_support_windows:
            failures.append("недостаточно опорных окон")
        if core_span < minimum_span:
            failures.append("недостаточная длительность группы")
        if missing_windows > maximum_missed:
            failures.append("слишком разреженная сетка окон")
        if anchor_count < 1:
            failures.append("нет строгого окна-якоря")
        if vocal_median < group_removed_threshold:
            failures.append("медианная потеря вокала недостаточна")
        if vocal_mean < group_removed_threshold:
            failures.append("средняя потеря вокала недостаточна")
        if program_median < group_program_threshold:
            failures.append("медианная потеря программы недостаточна")
        if program_mean < group_program_threshold:
            failures.append("средняя потеря программы недостаточна")
        if not translation_voice_available:
            failures.append(
                "нет русской речевой дорожки для защиты переводного диалога"
            )

        evaluation: dict[str, Any] = {
            "group_id": group_number,
            "eligible": not failures,
            "support_window_count": len(group),
            "strict_anchor_window_count": anchor_count,
            "missed_grid_windows": missing_windows,
            "core_start_sec": round(core_start, 3),
            "core_end_sec": round(core_end, 3),
            "core_span_sec": round(core_span, 3),
            "removed_vocal_share_median": round(vocal_median, 4),
            "removed_vocal_share_mean": round(vocal_mean, 4),
            "removed_program_share_median": round(program_median, 4),
            "removed_program_share_mean": round(program_mean, 4),
            "translation_voice_available": bool(
                translation_voice_available
            ),
            "failures": failures,
        }
        for item in group:
            item["anchor_backed_group_evaluation"] = evaluation
        if failures:
            continue

        extended_start = core_start
        extended_end = core_end
        context_windows: list[dict[str, Any]] = []
        if maximum_context_extension > 0.0:
            left_context = windows_by_index.get(first_index - 1)
            if left_context is not None and bool(
                (left_context.get("checks") or {}).get(
                    "context_boundary_window"
                )
            ):
                candidate_start = max(
                    core_start - maximum_context_extension,
                    float(left_context["start_sec"]),
                )
                if candidate_start < extended_start:
                    extended_start = candidate_start
                    context_windows.append(
                        {
                            "side": "left",
                            "index": _window_grid_index(left_context, cfg),
                            "start_sec": round(
                                float(left_context["start_sec"]), 3
                            ),
                            "end_sec": round(
                                float(left_context["end_sec"]), 3
                            ),
                        }
                    )
                    left_context["checks"][
                        "context_boundary_extension"
                    ] = True
                    left_context["context_boundary_extension"] = {
                        "group_id": group_number,
                        "side": "left",
                    }
            right_context = windows_by_index.get(last_index + 1)
            if right_context is not None and bool(
                (right_context.get("checks") or {}).get(
                    "context_boundary_window"
                )
            ):
                candidate_end = min(
                    core_end + maximum_context_extension,
                    float(right_context["end_sec"]),
                )
                if candidate_end > extended_end:
                    extended_end = candidate_end
                    context_windows.append(
                        {
                            "side": "right",
                            "index": _window_grid_index(right_context, cfg),
                            "start_sec": round(
                                float(right_context["start_sec"]), 3
                            ),
                            "end_sec": round(
                                float(right_context["end_sec"]), 3
                            ),
                        }
                    )
                    right_context["checks"][
                        "context_boundary_extension"
                    ] = True
                    right_context["context_boundary_extension"] = {
                        "group_id": group_number,
                        "side": "right",
                    }

        group_report = {
            **evaluation,
            "route": "programme_and_vocal_loss",
            "extended_start_sec": round(extended_start, 3),
            "extended_end_sec": round(extended_end, 3),
            "context_boundary_extension": {
                "applied": bool(context_windows),
                "maximum_sec_per_side": round(
                    maximum_context_extension, 3
                ),
                "left_extension_sec": round(
                    core_start - extended_start, 3
                ),
                "right_extension_sec": round(
                    extended_end - core_end, 3
                ),
                "windows": context_windows,
                "reason": (
                    "граница расширена только по соседнему окну с "
                    "подтверждёнными пением и музыкальным фоном"
                    if context_windows
                    else "безопасные соседние окна не подтверждены"
                ),
            },
        }
        for item in group:
            item["anchor_backed_group"] = group_report
            item.setdefault("checks", {})[
                "anchor_backed_group_route"
            ] = True
            if not bool(item.get("candidate")):
                item["candidate"] = True
                item["confidence"] = float(
                    item.get(
                        "programme_and_vocal_loss_support_confidence",
                        0.0,
                    )
                )
                item["damage_evidence_route"] = (
                    "programme_and_vocal_loss"
                )


def _merge_windows(
    windows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    translation_voice_available: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _promote_anchor_backed_groups(
        windows,
        cfg,
        translation_voice_available=translation_voice_available,
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in (value for value in windows if value["candidate"]):
        if current and float(item["start_sec"]) - float(current[-1]["end_sec"]) > float(cfg["maximum_gap_sec"]):
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)

    windows_by_index = {
        _window_grid_index(item, cfg): item
        for item in windows
    }
    maximum_context_extension = float(
        cfg["maximum_anchor_context_extension_sec"]
    )
    confirmed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for group in groups:
        anchor_group_reports_by_id: dict[int, dict[str, Any]] = {}
        for item in group:
            anchor_group_report = item.get("anchor_backed_group")
            if anchor_group_report:
                anchor_group_reports_by_id.setdefault(
                    int(anchor_group_report["group_id"]),
                    anchor_group_report,
                )
        anchor_group_reports = list(anchor_group_reports_by_id.values())
        damage_start = float(group[0]["start_sec"])
        damage_end = float(group[-1]["end_sec"])
        start = damage_start
        end = damage_end
        if anchor_group_reports:
            start = min(
                start,
                min(
                    float(item["extended_start_sec"])
                    for item in anchor_group_reports
                ),
            )
            end = max(
                end,
                max(
                    float(item["extended_end_sec"])
                    for item in anchor_group_reports
                ),
            )
        context_windows: list[dict[str, Any]] = []
        first_index = min(_window_grid_index(item, cfg) for item in group)
        last_index = max(_window_grid_index(item, cfg) for item in group)
        if maximum_context_extension > 0.0:
            left_limit = damage_start - maximum_context_extension
            context_index = first_index - 1
            while True:
                context_item = windows_by_index.get(context_index)
                if context_item is None or not bool(
                    (context_item.get("checks") or {}).get(
                        "context_boundary_window"
                    )
                ):
                    break
                candidate_start = max(
                    left_limit,
                    float(context_item["start_sec"]),
                )
                if candidate_start < start:
                    start = candidate_start
                    context_windows.append(
                        {
                            "side": "left",
                            "index": context_index,
                            "start_sec": round(
                                float(context_item["start_sec"]), 3
                            ),
                            "end_sec": round(
                                float(context_item["end_sec"]), 3
                            ),
                        }
                    )
                    context_item.setdefault("checks", {})[
                        "context_boundary_extension"
                    ] = True
                if candidate_start > float(context_item["start_sec"]):
                    break
                context_index -= 1

            right_limit = damage_end + maximum_context_extension
            context_index = last_index + 1
            while True:
                context_item = windows_by_index.get(context_index)
                if context_item is None or not bool(
                    (context_item.get("checks") or {}).get(
                        "context_boundary_window"
                    )
                ):
                    break
                candidate_end = min(
                    right_limit,
                    float(context_item["end_sec"]),
                )
                if candidate_end > end:
                    end = candidate_end
                    context_windows.append(
                        {
                            "side": "right",
                            "index": context_index,
                            "start_sec": round(
                                float(context_item["start_sec"]), 3
                            ),
                            "end_sec": round(
                                float(context_item["end_sec"]), 3
                            ),
                        }
                    )
                    context_item.setdefault("checks", {})[
                        "context_boundary_extension"
                    ] = True
                if candidate_end < float(context_item["end_sec"]):
                    break
                context_index += 1
        damage_duration = damage_end - damage_start
        duration = end - start
        confidence = float(np.mean([float(item["confidence"]) for item in group]))
        features = {
            key: round(float(np.mean([item["features"][key] for item in group])), 4)
            for key in group[0]["features"]
        }
        damage_routes = sorted(
            {
                str(item.get("damage_evidence_route") or "none")
                for item in group
            }
            - {"none"}
        )
        route_window_counts = {
            route: sum(
                1
                for item in group
                if str(item.get("damage_evidence_route") or "") == route
            )
            for route in damage_routes
        }
        reasons = [
            "EfficientAT подтвердил устойчивое пение и музыку",
            "музыкальный фон подтверждён отдельной дорожкой",
            (
                "после обработки подтверждено почти полное абсолютное "
                "удаление вокала"
                if any(
                    "strong_absolute" in route
                    for route in damage_routes
                )
                else (
                    "после обработки заметно уменьшилась вокальная "
                    "составляющая относительно музыкального фона"
                )
            ),
        ]
        if "programme_and_vocal_loss" in damage_routes:
            reasons.append(
                "строгое окно-якорь и плотная группа подтвердили "
                "устойчивую потерю программы и вокала"
            )
        anchor_context_extensions = [
            item["context_boundary_extension"]
            for item in anchor_group_reports
            if bool(
                (item.get("context_boundary_extension") or {}).get(
                    "applied"
                )
            )
        ]
        context_extension_applied = bool(
            start < damage_start - 1e-6
            or end > damage_end + 1e-6
        )
        context_extension = {
            "applied": context_extension_applied,
            "maximum_sec_per_side": round(
                maximum_context_extension, 3
            ),
            "left_extension_sec": round(damage_start - start, 3),
            "right_extension_sec": round(end - damage_end, 3),
            "windows": context_windows,
            "reason": (
                "граница расширена по непрерывным соседним окнам с "
                "подтверждёнными пением и музыкой"
                if context_extension_applied
                else "безопасные соседние окна не подтверждены"
            ),
            "anchor_groups": anchor_context_extensions,
        }
        if context_extension["applied"]:
            reasons.append(
                "границы расширены на соседний подтверждённый музыкальный "
                "контекст без учёта его как доказательства повреждения"
            )
        interval = {
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "duration_sec": round(duration, 3),
            "damage_evidence_duration_sec": round(
                damage_duration, 3
            ),
            "confidence": round(confidence, 4),
            "features": features,
            "damage_evidence_routes": damage_routes,
            "damage_evidence_window_counts": route_window_counts,
            "reasons": reasons,
            "context_boundary_extension": context_extension,
        }
        if anchor_group_reports:
            interval["anchor_backed_groups"] = anchor_group_reports
        if damage_duration < float(cfg["minimum_interval_sec"]):
            interval["decision"] = "rejected"
            interval["rejection_reason"] = "участок слишком короткий для безопасного восстановления песни"
            rejected.append(interval)
        elif confidence < float(cfg["minimum_confidence"]):
            interval["decision"] = "rejected"
            interval["rejection_reason"] = "недостаточная уверенность"
            rejected.append(interval)
        else:
            interval["decision"] = "build_safe_restoration_plan"
            confirmed.append(interval)
    return confirmed, rejected


def scan_song_candidates(
    original_detection_mix: Path,
    config: dict[str, Any] | None = None,
    *,
    report_path: Path | None = None,
    ctx: Any = None,
    classifier: Any = None,
) -> dict[str, Any]:
    """Find sustained singing candidates on the aligned English original.

    This deliberately is only a *preview scene picker*.  It never authorises a
    restoration by itself: :func:`detect_song_intervals` still has to confirm
    the music bed, vocal damage and translated-dialogue safety on every
    selected preview scene.  Keeping the scan classifier-only makes it
    possible to inspect the whole film without extracting full-film speech
    stems, while ensuring that the learned model never sees the dubbed track.
    """
    cfg = resolved_config(config)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "purpose": "preview_scene_selection_only",
        "enabled": bool(cfg["enabled"]),
        "source": str(original_detection_mix.resolve()),
        "config": cfg,
        "candidate_intervals": [],
        "rejected_candidates": [],
        "window_decisions": [],
    }
    if not cfg["enabled"]:
        report["summary"] = "Защита песен отключена в конфигурации."
        if report_path is not None:
            atomic_json(report_path, report)
        return report
    if (
        not original_detection_mix.is_file()
        or original_detection_mix.stat().st_size <= 0
    ):
        raise RuntimeError(
            "Не найдена выровненная английская дорожка для поиска песен."
        )
    try:
        song_classifier = classifier or AudioSetSongClassifier(cfg)
    except Exception as error:
        report["classifier"] = {
            "backend": str(cfg["classifier_backend"]),
            "available": False,
            "error": str(error),
        }
        report["summary"] = (
            "Классификатор песен недоступен; песенные сцены не добавлены."
        )
        if report_path is not None:
            atomic_json(report_path, report)
        return report
    report["classifier"] = {
        "backend": str(
            getattr(song_classifier, "backend", cfg["classifier_backend"])
        ),
        "available": True,
        "model_path": str(
            getattr(song_classifier, "model_path", cfg["classifier_model_path"])
        ),
        "sample_rate": int(
            getattr(
                song_classifier,
                "sample_rate",
                cfg["classifier_sample_rate"],
            )
        ),
        "input_sec": float(cfg["classifier_input_sec"]),
    }

    with sf.SoundFile(str(original_detection_mix)) as detection_file:
        duration = len(detection_file) / detection_file.samplerate
        starts = np.arange(
            0.0,
            max(0.01, duration - float(cfg["window_sec"]) + 1e-6),
            float(cfg["step_sec"]),
        ).tolist()
        if not starts and duration > 0:
            starts = [0.0]
        classifier_rate = int(
            getattr(
                song_classifier,
                "sample_rate",
                cfg["classifier_sample_rate"],
            )
        )
        classifier_scores: list[dict[str, float]] = []
        batch_size = int(cfg["classifier_batch_size"])
        for batch_start in range(0, len(starts), batch_size):
            if ctx is not None:
                ctx.check_stop()
            batch_starts = starts[batch_start : batch_start + batch_size]
            waveforms = [
                _read_window(
                    detection_file,
                    float(start),
                    min(float(cfg["window_sec"]), duration - float(start)),
                    classifier_rate,
                )
                for start in batch_starts
            ]
            classifier_scores.extend(song_classifier.predict_many(waveforms))
            if ctx is not None:
                checked = min(batch_start + len(batch_starts), len(starts))
                ctx.update(
                    stage="Поиск песенных сцен для превью",
                    substage=(
                        f"EfficientAT: проверено {checked} из {len(starts)} окон"
                    ),
                    local_progress=checked / max(len(starts), 1) * 100.0,
                )
        if len(classifier_scores) != len(starts):
            raise RuntimeError(
                "Классификатор песен вернул неполный набор результатов."
            )
        windows: list[dict[str, Any]] = []
        feature_rate = int(cfg["analysis_sample_rate"])
        for index, (start, scores) in enumerate(
            zip(starts, classifier_scores, strict=True)
        ):
            window_duration = min(float(cfg["window_sec"]), duration - start)
            if window_duration <= 0:
                continue
            source_features = _spectral_features(
                _read_window(
                    detection_file,
                    float(start),
                    window_duration,
                    feature_rate,
                ),
                feature_rate,
            )
            checks = {
                "audible_input": source_features["rms_db"]
                >= float(cfg["minimum_input_rms_db"]),
                "learned_singing": scores["singing_score"]
                >= float(cfg["minimum_classifier_singing_score"]),
                "learned_direct_singing": scores["direct_singing_score"]
                >= float(cfg["minimum_classifier_direct_singing_score"]),
                "learned_music": scores["music_score"]
                >= float(cfg["minimum_classifier_music_score"]),
            }
            margins = [
                np.clip(
                    (
                        scores["singing_score"]
                        - float(cfg["minimum_classifier_singing_score"])
                    )
                    / 0.35,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (
                        scores["direct_singing_score"]
                        - float(
                            cfg["minimum_classifier_direct_singing_score"]
                        )
                    )
                    / 0.25,
                    0.0,
                    1.0,
                ),
                np.clip(
                    (
                        scores["music_score"]
                        - float(cfg["minimum_classifier_music_score"])
                    )
                    / 0.30,
                    0.0,
                    1.0,
                ),
            ]
            confidence = (
                float(
                    np.clip(
                        0.78 + 0.22 * float(np.mean(margins)),
                        0.0,
                        1.0,
                    )
                )
                if all(checks.values())
                else 0.0
            )
            windows.append(
                {
                    "index": index,
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(start + window_duration), 3),
                    "candidate": bool(all(checks.values())),
                    "confidence": round(confidence, 4),
                    "checks": checks,
                    "features": {
                        "input_rms_db": round(source_features["rms_db"], 3),
                        **scores,
                    },
                }
            )

    confirmed, rejected = _merge_windows(windows, cfg)
    for item in confirmed:
        item["decision"] = "add_preview_scene_for_full_verification"
        item["reasons"] = [
            "EfficientAT подтвердил устойчивое пение",
            "EfficientAT подтвердил музыкальный фон",
            "длительность достаточна для проверки в превью",
        ]
    report["duration_sec"] = round(float(duration), 3)
    report["analysed_windows"] = len(windows)
    report["candidate_intervals"] = confirmed
    report["rejected_candidates"] = rejected
    report["window_decisions"] = windows
    report["summary"] = (
        f"Для превью найдено песенных кандидатов: {len(confirmed)}."
        if confirmed
        else "Устойчивые песенные кандидаты для превью не найдены."
    )
    if report_path is not None:
        atomic_json(report_path, report)
    return report


def _build_restoration_plan(
    confirmed: list[dict[str, Any]],
    original_detection_mix: Path,
    dubbed_mix: Path,
    original_speech_stem: Path | None,
    dubbed_speech_stem: Path,
    translation_voice_stem: Path | None,
    music_background: Path,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if not confirmed:
        return []
    if (
        original_speech_stem is None
        or not original_speech_stem.is_file()
        or original_speech_stem.stat().st_size <= 0
    ):
        for interval in confirmed:
            interval["restoration_source"] = "none"
            interval["dialogue_detection"] = {
                "available": False,
                "reason": "не найдена выровненная английская речевая дорожка",
            }
            interval["restoration_segments"] = []
        return []

    rate = int(cfg["analysis_sample_rate"])
    all_segments: list[dict[str, Any]] = []
    with (
        sf.SoundFile(str(original_detection_mix)) as original_mix_file,
        sf.SoundFile(str(dubbed_mix)) as dubbed_mix_file,
        sf.SoundFile(str(original_speech_stem)) as original_speech_file,
        sf.SoundFile(str(dubbed_speech_stem)) as dubbed_speech_file,
        (
            sf.SoundFile(str(translation_voice_stem))
            if translation_voice_stem is not None
            else nullcontext(None)
        ) as translation_voice_file,
        sf.SoundFile(str(music_background)) as music_background_file,
    ):
        for song_index, interval in enumerate(confirmed):
            song_start = float(interval["start_sec"])
            song_end = float(interval["end_sec"])
            song_duration = max(0.0, song_end - song_start)
            original_mix = _read_window(
                original_mix_file, song_start, song_duration, rate
            )
            dubbed_mix_values = _read_window(
                dubbed_mix_file, song_start, song_duration, rate
            )
            original_background = _read_window(
                music_background_file, song_start, song_duration, rate
            )
            dubbed_song_speech = _read_window(
                dubbed_speech_file, song_start, song_duration, rate
            )
            comparison = _track_music_similarity(
                original_mix,
                dubbed_mix_values,
                rate,
                original_background=original_background,
                dubbed_speech=dubbed_song_speech,
            )
            same_music = (
                float(comparison["music_similarity"])
                >= float(cfg["minimum_track_music_similarity"])
                and float(comparison["matched_window_share"])
                >= float(cfg["minimum_track_matched_window_share"])
            )
            interval["track_comparison"] = comparison
            if not same_music:
                # A mismatch is not permission to restore another complete
                # mix.  It means that we cannot prove the English song bed is
                # the one heard in the dub, so the conservative action is to
                # leave the already processed result untouched.
                interval["restoration_source"] = "none"
                interval["restoration_source_reason"] = (
                    "музыкальные признаки различаются или совпадение "
                    "недостаточно уверенное; автоматическая замена отключена"
                )
                interval["dialogue_detection"] = {
                    "available": False,
                    "reason": (
                        "сравнение музыкальных дорожек не подтвердило "
                        "безопасное восстановление английского оригинала"
                    ),
                }
                interval["restoration_segments"] = []
                continue
            if translation_voice_file is None:
                interval["restoration_source"] = "none"
                interval["restoration_source_reason"] = (
                    "не найдена очищенная русская речевая дорожка; "
                    "автоматическая замена отключена"
                )
                interval["dialogue_detection"] = {
                    "available": False,
                    "reason": (
                        "без очищенной русской речевой дорожки нельзя "
                        "надёжно исключить переводной диалог внутри песни"
                    ),
                }
                interval["restoration_segments"] = []
                continue
            source_kind = "original_aligned"

            window_sec = min(float(cfg["dialogue_window_sec"]), song_duration)
            fine_starts = np.arange(
                song_start,
                max(song_start + 1e-6, song_end - window_sec + 1e-6),
                float(cfg["dialogue_step_sec"]),
            ).tolist()
            if not fine_starts and song_duration > 0:
                fine_starts = [song_start]
            dialogue_windows: list[dict[str, Any]] = []
            for start in fine_starts:
                duration = min(window_sec, song_end - float(start))
                if duration <= 0:
                    continue
                features = _stem_difference_features(
                    _read_window(
                        original_speech_file, float(start), duration, rate
                    ),
                    _read_window(dubbed_speech_file, float(start), duration, rate),
                    rate,
                    (
                        _read_window(
                            translation_voice_file,
                            float(start),
                            duration,
                            rate,
                        )
                        if translation_voice_file is not None
                        else None
                    ),
                    _read_window(
                        dubbed_mix_file,
                        float(start),
                        duration,
                        rate,
                    ),
                )
                dialogue_windows.append(
                    {
                        "start_sec": round(float(start), 3),
                        "end_sec": round(float(start + duration), 3),
                        "features": features,
                    }
                )
            scores = np.asarray(
                [
                    item["features"]["dialogue_score"]
                    for item in dialogue_windows
                ],
                dtype=np.float64,
            )
            baseline = float(np.median(scores)) if scores.size else 0.0
            mad = (
                float(np.median(np.abs(scores - baseline))) if scores.size else 0.0
            )
            adaptive_threshold = max(
                float(cfg["minimum_dialogue_score"]),
                baseline + float(cfg["dialogue_baseline_margin"]),
                baseline + float(cfg["dialogue_mad_multiplier"]) * mad,
            )
            translation_vocal_dominance_threshold = float(
                cfg["minimum_translation_voice_vocal_to_backing_db"]
            )
            translation_vocal_dominance_ambiguity_margin = float(
                cfg[
                    "translation_voice_vocal_to_backing_ambiguity_margin_db"
                ]
            )
            translation_vocal_dominance_ambiguity_floor = (
                translation_vocal_dominance_threshold
                - translation_vocal_dominance_ambiguity_margin
            )
            for item in dialogue_windows:
                features = item["features"]
                score = float(features["dialogue_score"])
                translation_vocal_dominance = float(
                    features[
                        "translation_voice_vocal_to_backing_db"
                    ]
                )
                audible_difference = bool(
                    float(features["dubbed_stem_rms_db"])
                    >= float(cfg["minimum_dialogue_stem_rms_db"])
                )
                translation_voice_evidence = bool(
                    features["translation_voice_available"]
                    and float(features["translation_voice_rms_db"])
                    >= float(cfg["minimum_translation_voice_rms_db"])
                    and float(features["translation_voice_share"])
                    >= float(cfg["minimum_translation_voice_share"])
                    and float(
                        features["translation_voice_program_share"]
                    )
                    >= float(
                        cfg["minimum_translation_voice_program_share"]
                    )
                    and float(
                        features[
                            "translation_voice_vocal_to_backing_db"
                        ]
                    )
                    >= float(
                        translation_vocal_dominance_threshold
                    )
                    and float(features["translation_voice_speech_likeness"])
                    >= float(
                        cfg["minimum_translation_voice_speech_likeness"]
                    )
                )
                residual_difference_evidence = bool(
                    float(features["nonmatching_share"])
                    >= float(cfg["minimum_dialogue_nonmatching_share"])
                    and float(features["speech_likeness"])
                    >= float(cfg["minimum_dialogue_speech_likeness"])
                )
                item["dialogue_candidate"] = bool(
                    audible_difference
                    # Residual stem differences remain useful corroboration,
                    # but cannot block restoration without substantial cleaned
                    # Russian speech in the full dubbed programme.
                    and translation_voice_evidence
                    and (
                        score >= adaptive_threshold
                        or score >= float(cfg["strong_dialogue_score"])
                    )
                )
                item["ambiguous_content"] = bool(
                    audible_difference
                    and not item["dialogue_candidate"]
                    and features["translation_voice_available"]
                    and float(
                        features["translation_voice_program_share"]
                    )
                    >= float(
                        cfg["minimum_translation_voice_program_share"]
                    )
                    and float(
                        features[
                            "translation_voice_vocal_to_backing_db"
                        ]
                    )
                    >= float(
                        translation_vocal_dominance_ambiguity_floor
                    )
                    and (
                        (
                            float(features["nonmatching_share"])
                            >= float(
                                cfg["minimum_ambiguous_nonmatching_share"]
                            )
                        )
                        or (
                            features["translation_voice_available"]
                            and float(features["translation_voice_rms_db"])
                            >= float(cfg["minimum_translation_voice_rms_db"])
                            and float(features["translation_voice_share"])
                            >= float(cfg["minimum_translation_voice_share"])
                        )
                    )
                    and (
                        score
                        >= baseline
                        + float(cfg["ambiguous_dialogue_score_margin"])
                        or score >= float(cfg["minimum_dialogue_score"])
                    )
                )
                item["evidence"] = {
                    "residual_spectral_difference": residual_difference_evidence,
                    "translation_voice_program_share": bool(
                        features["translation_voice_available"]
                        and float(
                            features["translation_voice_program_share"]
                        )
                        >= float(
                            cfg[
                                "minimum_translation_voice_program_share"
                            ]
                        )
                    ),
                    "translation_voice_vocal_to_backing": bool(
                        features["translation_voice_available"]
                        and translation_vocal_dominance
                        >= translation_vocal_dominance_threshold
                    ),
                    "translation_voice_vocal_to_backing_ambiguous": bool(
                        features["translation_voice_available"]
                        and translation_vocal_dominance
                        >= translation_vocal_dominance_ambiguity_floor
                        and translation_vocal_dominance
                        < translation_vocal_dominance_threshold
                    ),
                    "translation_voice": translation_voice_evidence,
                    "adaptive_score_passed": score >= adaptive_threshold,
                    "strong_absolute_score_passed": (
                        score >= float(cfg["strong_dialogue_score"])
                    ),
                }
                item["block_restoration"] = bool(
                    item["dialogue_candidate"] or item["ambiguous_content"]
                )
            translation_evidence_windows = [
                item
                for item in dialogue_windows
                if bool((item.get("evidence") or {}).get("translation_voice"))
                and float(
                    item["features"].get(
                        "translation_voice_evidence_score", 0.0
                    )
                )
                >= float(cfg["minimum_dialogue_score"])
            ]
            translation_voice_window_share = (
                len(translation_evidence_windows)
                / max(len(dialogue_windows), 1)
            )
            full_interval_translation_block = bool(
                dialogue_windows
                and translation_voice_window_share
                >= float(
                    cfg[
                        "minimum_translation_voice_majority_window_share"
                    ]
                )
            )
            if full_interval_translation_block:
                blocked_intervals = [
                    {
                        "start_sec": round(song_start, 3),
                        "end_sec": round(song_end, 3),
                        "duration_sec": round(song_duration, 3),
                        "decision": "keep_processed_translation",
                        "reason": (
                            "очищенная русская речь устойчиво присутствует "
                            "в большей части песенного интервала; английский "
                            "микс не подставляется"
                        ),
                        "mean_dialogue_score": round(
                            float(
                                np.mean(
                                    [
                                        item["features"]["dialogue_score"]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "mean_nonmatching_share": round(
                            float(
                                np.mean(
                                    [
                                        item["features"][
                                            "nonmatching_share"
                                        ]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "mean_speech_likeness": round(
                            float(
                                np.mean(
                                    [
                                        item["features"][
                                            "translation_voice_speech_likeness"
                                        ]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "mean_translation_voice_share": round(
                            float(
                                np.mean(
                                    [
                                        item["features"][
                                            "translation_voice_share"
                                        ]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "mean_translation_voice_program_share": round(
                            float(
                                np.mean(
                                    [
                                        item["features"][
                                            "translation_voice_program_share"
                                        ]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "mean_translation_voice_evidence_score": round(
                            float(
                                np.mean(
                                    [
                                        item["features"][
                                            "translation_voice_evidence_score"
                                        ]
                                        for item in translation_evidence_windows
                                    ]
                                )
                            ),
                            4,
                        ),
                        "evidence_sources": [
                            "translation_voice_energy_share",
                            "translation_voice_program_share",
                            "translation_voice_spectral_dynamics",
                            "sustained_majority_of_song_windows",
                        ],
                    }
                ]
            else:
                blocked_intervals = _merge_dialogue_windows(
                    dialogue_windows, song_start, song_end, cfg
                )
            segments = _restoration_segments(
                song_start, song_end, blocked_intervals, source_kind
            )
            for segment in segments:
                segment["song_index"] = song_index
                segment["reason"] = (
                    "музыкальные признаки оригинала и перевода совпадают"
                )
            interval["restoration_source"] = source_kind
            interval["restoration_source_reason"] = (
                "музыкальные признаки совпадают; сохраняется английский оригинал"
            )
            interval["dialogue_detection"] = {
                "available": True,
                "baseline_score": round(baseline, 4),
                "median_absolute_deviation": round(mad, 4),
                "adaptive_threshold": round(adaptive_threshold, 4),
                "translation_voice_window_share": round(
                    translation_voice_window_share, 4
                ),
                "translation_voice_majority_threshold": float(
                    cfg["minimum_translation_voice_majority_window_share"]
                ),
                "translation_voice_program_share_threshold": float(
                    cfg["minimum_translation_voice_program_share"]
                ),
                "translation_voice_vocal_to_backing_db_threshold": float(
                    cfg[
                        "minimum_translation_voice_vocal_to_backing_db"
                    ]
                ),
                "translation_voice_vocal_to_backing_db_ambiguity_margin": (
                    translation_vocal_dominance_ambiguity_margin
                ),
                "translation_voice_vocal_to_backing_db_ambiguity_floor": (
                    translation_vocal_dominance_ambiguity_floor
                ),
                "full_interval_translation_block": (
                    full_interval_translation_block
                ),
                "windows": dialogue_windows,
                "confirmed_intervals": [
                    item
                    for item in blocked_intervals
                    if item["decision"] == "keep_processed_translation"
                ],
                "low_confidence_intervals": [
                    item
                    for item in blocked_intervals
                    if item["decision"] == "keep_processed_low_confidence"
                ],
                "blocked_restoration_intervals": blocked_intervals,
            }
            interval["restoration_segments"] = segments
            all_segments.extend(segments)
    return all_segments


def detect_song_intervals(
    original_detection_mix: Path,
    dubbed_mix: Path,
    speech_stem: Path,
    music_background: Path,
    processed_mix: Path,
    config: dict[str, Any] | None = None,
    *,
    report_path: Path | None = None,
    ctx: Any = None,
    classifier: Any = None,
    original_speech_stem: Path | None = None,
    translation_voice_stem: Path | None = None,
) -> dict[str, Any]:
    cfg = resolved_config(config)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "enabled": bool(cfg["enabled"]),
        "sources": {
            "original_detection_mix": str(original_detection_mix.resolve()),
            "dubbed_mix": str(dubbed_mix.resolve()),
            "original_speech_stem": (
                str(original_speech_stem.resolve())
                if original_speech_stem is not None
                else ""
            ),
            "translation_voice_stem": (
                str(translation_voice_stem.resolve())
                if translation_voice_stem is not None
                else ""
            ),
            "speech_stem": str(speech_stem.resolve()),
            "music_background": str(music_background.resolve()),
            "processed_mix": str(processed_mix.resolve()),
        },
        "config": cfg,
        "confirmed_intervals": [],
        "rejected_candidates": [],
    }
    if not cfg["enabled"]:
        report["summary"] = "Защита песен отключена в конфигурации."
        if report_path is not None:
            atomic_json(report_path, report)
        return report
    for path in (
        original_detection_mix,
        dubbed_mix,
        speech_stem,
        music_background,
        processed_mix,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Не найден аудиофайл для защиты песен: {path}")
    if original_speech_stem is not None and (
        not original_speech_stem.is_file()
        or original_speech_stem.stat().st_size <= 0
    ):
        original_speech_stem = None
    if translation_voice_stem is not None and (
        not translation_voice_stem.is_file()
        or translation_voice_stem.stat().st_size <= 0
    ):
        translation_voice_stem = None
    try:
        song_classifier = classifier or AudioSetSongClassifier(cfg)
    except Exception as error:
        report["classifier"] = {
            "backend": str(cfg["classifier_backend"]),
            "available": False,
            "error": str(error),
        }
        report["summary"] = (
            "Классификатор песен недоступен; автоматическая замена не выполнена."
        )
        if report_path is not None:
            atomic_json(report_path, report)
        return report
    report["classifier"] = {
        "backend": str(getattr(song_classifier, "backend", cfg["classifier_backend"])),
        "available": True,
        "model_path": str(
            getattr(song_classifier, "model_path", cfg["classifier_model_path"])
        ),
        "sample_rate": int(
            getattr(song_classifier, "sample_rate", cfg["classifier_sample_rate"])
        ),
    }

    with (
        sf.SoundFile(str(original_detection_mix)) as detection_file,
        sf.SoundFile(str(dubbed_mix)) as source_file,
        sf.SoundFile(str(speech_stem)) as speech_file,
        sf.SoundFile(str(music_background)) as background_file,
        sf.SoundFile(str(processed_mix)) as processed_file,
    ):
        duration = min(
            len(detection_file) / detection_file.samplerate,
            len(source_file) / source_file.samplerate,
            len(speech_file) / speech_file.samplerate,
            len(background_file) / background_file.samplerate,
            len(processed_file) / processed_file.samplerate,
        )
        starts = np.arange(
            0.0,
            max(0.01, duration - float(cfg["window_sec"]) + 1e-6),
            float(cfg["step_sec"]),
        ).tolist()
        if not starts and duration > 0:
            starts = [0.0]
        classifier_scores: list[dict[str, float]] = []
        batch_size = int(cfg["classifier_batch_size"])
        classifier_rate = int(
            getattr(song_classifier, "sample_rate", cfg["classifier_sample_rate"])
        )
        for batch_start in range(0, len(starts), batch_size):
            if ctx is not None:
                ctx.check_stop()
            batch_starts = starts[batch_start : batch_start + batch_size]
            # The learned classifier sees only the English original aligned
            # to the dubbed timeline. Damage measurements below still compare
            # the untouched dubbed full mix with the processed result.
            waveforms = [
                _read_window(
                    detection_file,
                    float(start),
                    min(float(cfg["window_sec"]), duration - float(start)),
                    classifier_rate,
                )
                for start in batch_starts
            ]
            classifier_scores.extend(song_classifier.predict_many(waveforms))
            if ctx is not None:
                checked = min(batch_start + len(batch_starts), len(starts))
                ctx.update(
                    stage="Защита песенных фрагментов",
                    substage=(
                        f"Классификатор: проверено {checked} из {len(starts)} окон"
                    ),
                    local_progress=checked / max(len(starts), 1) * 45.0,
                )
        if len(classifier_scores) != len(starts):
            raise RuntimeError(
                "Классификатор песен вернул неполный набор результатов."
            )
        windows: list[dict[str, Any]] = []
        for index, start in enumerate(starts):
            if ctx is not None:
                ctx.check_stop()
            window_duration = min(float(cfg["window_sec"]), duration - start)
            if window_duration <= 0:
                continue
            rate = int(cfg["analysis_sample_rate"])
            decision = _window_decision(
                _spectral_features(_read_window(source_file, start, window_duration, rate), rate),
                _spectral_features(_read_window(speech_file, start, window_duration, rate), rate),
                _spectral_features(_read_window(background_file, start, window_duration, rate), rate),
                _spectral_features(_read_window(processed_file, start, window_duration, rate), rate),
                classifier_scores[index],
                cfg,
            )
            windows.append(
                {
                    "index": index,
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(start + window_duration), 3),
                    **decision,
                }
            )
            if ctx is not None and (index % 10 == 0 or index + 1 == len(starts)):
                ctx.update(
                    stage="Защита песенных фрагментов",
                    substage=(
                        f"Проверка удаления вокала: {index + 1} из {len(starts)} окон"
                    ),
                    local_progress=45.0
                    + (index + 1) / max(len(starts), 1) * 55.0,
                )
    confirmed, rejected = _merge_windows(
        windows,
        cfg,
        translation_voice_available=translation_voice_stem is not None,
    )
    restoration_segments = _build_restoration_plan(
        confirmed,
        original_detection_mix,
        dubbed_mix,
        original_speech_stem,
        speech_stem,
        translation_voice_stem,
        music_background,
        cfg,
    )
    report["duration_sec"] = round(float(duration), 3)
    report["analysed_windows"] = len(windows)
    report["confirmed_intervals"] = confirmed
    report["rejected_candidates"] = rejected
    report["window_decisions"] = windows
    report["restoration_segments"] = restoration_segments
    report["summary"] = (
        f"Подтверждено песенных интервалов: {len(confirmed)}."
        if confirmed
        else "Песенные интервалы с достаточной уверенностью не найдены; звук не заменён."
    )
    if report_path is not None:
        atomic_json(report_path, report)
    return report


def apply_song_protection(
    processed_mix: Path,
    dubbed_mix: Path,
    destination: Path,
    intervals: list[dict[str, Any]],
    *,
    original_mix: Path | None = None,
    crossfade_sec: float,
    ctx: Any = None,
) -> Path:
    """Restore each confirmed span from its selected untouched full mix."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed_info = sf.info(str(processed_mix))
    dubbed_info = sf.info(str(dubbed_mix))
    original_path = original_mix or dubbed_mix
    original_info = sf.info(str(original_path))
    if (
        int(processed_info.samplerate) != int(dubbed_info.samplerate)
        or int(processed_info.samplerate) != int(original_info.samplerate)
    ):
        raise RuntimeError("Защита песен требует одинаковой частоты аудиодорожек.")
    rate = int(processed_info.samplerate)
    channels = int(processed_info.channels)
    total_frames = int(processed_info.frames)
    fades = max(0, int(round(float(crossfade_sec) * rate)))
    spans: list[tuple[int, int, str]] = []
    for item in intervals:
        if float(item.get("end_sec") or 0.0) <= float(
            item.get("start_sec") or 0.0
        ):
            continue
        source_kind = str(item.get("restore_source") or "dubbed")
        if source_kind not in {"dubbed", "original_aligned"}:
            raise RuntimeError(
                f"Неизвестный источник восстановления песни: {source_kind}."
            )
        spans.append(
            (
                max(0, int(round(float(item["start_sec"]) * rate))),
                min(total_frames, int(round(float(item["end_sec"]) * rate))),
                source_kind,
            )
        )
    if original_mix is None and any(
        source_kind == "original_aligned" for _, _, source_kind in spans
    ):
        raise RuntimeError(
            "Для восстановления песни не передана выровненная английская дорожка."
        )
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    cursor = 0
    try:
        with sf.SoundFile(str(processed_mix)) as processed_reader:
            with sf.SoundFile(str(dubbed_mix)) as dubbed_reader:
                with sf.SoundFile(str(original_path)) as original_reader:
                    with sf.SoundFile(
                    str(temporary),
                    mode="w",
                    samplerate=rate,
                    channels=channels,
                    format="FLAC",
                    subtype="PCM_24",
                    ) as writer:
                        blocksize = rate * 20
                        while cursor < total_frames:
                            if ctx is not None:
                                ctx.check_stop()
                            frames = min(blocksize, total_frames - cursor)
                            processed = processed_reader.read(
                                frames, dtype="float32", always_2d=True
                            )
                            dubbed = dubbed_reader.read(
                                frames, dtype="float32", always_2d=True
                            )
                            original = original_reader.read(
                                frames, dtype="float32", always_2d=True
                            )
                            if len(dubbed) < frames:
                                dubbed = np.vstack(
                                    [
                                        dubbed,
                                        np.zeros(
                                            (
                                                frames - len(dubbed),
                                                int(dubbed_info.channels),
                                            ),
                                            dtype=np.float32,
                                        ),
                                    ]
                                )
                            if len(original) < frames:
                                original = np.vstack(
                                    [
                                        original,
                                        np.zeros(
                                            (
                                                frames - len(original),
                                                int(original_info.channels),
                                            ),
                                            dtype=np.float32,
                                        ),
                                    ]
                                )
                            dubbed = audio_io.match_channels(dubbed, channels)
                            original = audio_io.match_channels(original, channels)
                            dubbed_alpha = np.zeros(frames, dtype=np.float32)
                            original_alpha = np.zeros(frames, dtype=np.float32)
                            block_end = cursor + frames
                            for start, end, source_kind in spans:
                                left = max(cursor, start)
                                right = min(block_end, end)
                                if right <= left:
                                    continue
                                positions = np.arange(left, right, dtype=np.int64)
                                weight = np.ones(len(positions), dtype=np.float32)
                                if fades:
                                    weight = np.minimum(
                                        weight,
                                        np.clip(
                                            (positions - start) / fades, 0.0, 1.0
                                        ),
                                    )
                                    weight = np.minimum(
                                        weight,
                                        np.clip(
                                            (end - positions) / fades, 0.0, 1.0
                                        ),
                                    )
                                offset = left - cursor
                                alpha = (
                                    original_alpha
                                    if source_kind == "original_aligned"
                                    else dubbed_alpha
                                )
                                alpha[offset : offset + len(weight)] = np.maximum(
                                    alpha[offset : offset + len(weight)], weight
                                )
                            source_sum = dubbed_alpha + original_alpha
                            total_alpha = np.clip(source_sum, 0.0, 1.0)
                            normalizer = np.maximum(source_sum, 1.0)
                            dubbed_weight = dubbed_alpha / normalizer
                            original_weight = original_alpha / normalizer
                            mixed = (
                                processed * (1.0 - total_alpha[:, None])
                                + dubbed[:frames] * dubbed_weight[:, None]
                                + original[:frames] * original_weight[:, None]
                            )
                            # Do not apply a limiter or gain to protected cores:
                            # a unit source weight must preserve the untouched
                            # selected mix. SoundFile only performs the format's
                            # unavoidable PCM conversion.
                            writer.write(mixed)
                            cursor += frames
                            if ctx is not None:
                                ctx.update(
                                    stage="Защита песенных фрагментов",
                                    substage="Плавное восстановление песенных участков",
                                    local_progress=cursor
                                    / max(total_frames, 1)
                                    * 100.0,
                                )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
