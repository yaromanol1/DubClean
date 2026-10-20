#!/usr/bin/env python
"""Global mastering profile from speech-free shared M&E segments.

Идея: оригинальная EN-дорожка и EN-подложка внутри перевода различаются
мастерингом/каналом. Вместо оценки профиля по речевым сценам (где RU-речь
загрязняет измерение и приходится защищаться квантилями, см.
``reference_adapter.fit_algorithmic_profile``) профиль оценивается по
10--20 участкам БЕЗ речи в обеих версиях, где музыка/эффекты обязаны
совпадать. Один глобальный профиль затем применяется ко всему референсу
существующим ``reference_adapter.adapt_reference_file_with_profile`` —
эмитируемый JSON совместим с ``ReferenceAdapterProfile``.

Ограничения by design (важно не расширять бездумно):
- профиль строго линейный и статический: gain + гладкая mel-EQ кривая,
  ограниченная ``max_abs_gain_db``.  Никакой компрессии, шума, реверба —
  такие операции сигнало-зависимы, не коммутируют с speech separation и
  могут испортить референс сильнее, чем помочь;
- смещения/скорость НЕ являются частью профиля: выравнивание — работа
  alignment; здесь локальный offset только измеряется и служит гейтом.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.alignment import normalized_xcorr  # noqa: E402
from experiments.paired_reference_cancel.reference_adapter import (
    ReferenceAdapterProfile,
    _band_energy,
    _logmel_distance,
    _mel_filterbank,
    _mono,
    _stft,
    apply_profile,
)
from experiments.paired_reference_cancel.storage import atomic_json

PROFILE_MODE = "global_mastering_speechfree_v2"

# Дефолты подобраны под 48 кГц полные миксы фильмов.
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 512
DEFAULT_BANDS = 40
DEFAULT_WINDOW_SEC = 6.0
DEFAULT_HOP_SEC = 3.0
# Совпадает с application_pipeline.MIN_SPEECH_RMS_DB по смыслу: тише — «речи нет».
DEFAULT_SPEECH_RMS_DB_MAX = -48.0
DEFAULT_MIX_RMS_DB_MIN = -50.0
DEFAULT_MAX_REFINED_OFFSET_MS = 60.0
DEFAULT_MIN_COHERENCE = 0.55
DEFAULT_MAX_SEGMENTS = 20
DEFAULT_MIN_SPACING_SEC = 45.0
DEFAULT_MAX_ABS_GAIN_DB = 12.0
# A profile that spends a material part of its bands on the safety rail is not
# a measured channel any more: the real correction lies outside the modelled
# range.  Applying such a curve used to turn a very loud mismatch into an
# apparently valid profile (the old Bronson report was exactly +12 dB in the
# median).  Refuse it and let the downstream path keep the raw reference.
DEFAULT_SATURATION_MARGIN_DB = 0.25
DEFAULT_MAX_SATURATED_BAND_FRACTION = 0.20


def _rms_db(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return 20.0 * math.log10(math.sqrt(float(np.mean(data * data)) + 1e-12) + 1e-12)


def _frame_rms_db(values: np.ndarray, frame: int) -> np.ndarray:
    usable = (len(values) // frame) * frame
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    frames = values[:usable].astype(np.float64).reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return (20.0 * np.log10(rms + 1e-12)).astype(np.float32)


@dataclass
class SpeechFreeSegment:
    start_sec: float
    duration_sec: float
    refined_offset_ms: float
    coherence: float
    original_rms_db: float
    dubbed_rms_db: float
    role: str = "fit"  # fit | holdout

    def to_json(self) -> dict[str, Any]:
        return {
            "start_sec": round(self.start_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "refined_offset_ms": round(self.refined_offset_ms, 2),
            "coherence": round(self.coherence, 4),
            "original_rms_db": round(self.original_rms_db, 2),
            "dubbed_rms_db": round(self.dubbed_rms_db, 2),
            "role": self.role,
        }


@dataclass
class SegmentSelectionReport:
    segments: list[SpeechFreeSegment]
    windows_total: int = 0
    rejected_speech: int = 0
    rejected_quiet: int = 0
    rejected_offset: int = 0
    rejected_coherence: int = 0
    rejected_spacing: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_json() for segment in self.segments],
            "windows_total": self.windows_total,
            "rejected_speech": self.rejected_speech,
            "rejected_quiet": self.rejected_quiet,
            "rejected_offset": self.rejected_offset,
            "rejected_coherence": self.rejected_coherence,
            "rejected_spacing": self.rejected_spacing,
        }


def _band_log_energy(values: np.ndarray, sample_rate: int, n_fft: int, hop: int, filters: np.ndarray) -> np.ndarray:
    magnitude = np.abs(_stft(values, n_fft, hop))
    return 20.0 * np.log10(_band_energy(magnitude, filters))


def _spectral_coherence(
    original: np.ndarray,
    dubbed: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop: int,
    filters: np.ndarray,
) -> float:
    """Корреляция лог-энергий band×frame: чувствительна к «другой музыке»,
    нечувствительна к статическому EQ/gain (вычитаем средние)."""
    ref_db = _band_log_energy(original, sample_rate, n_fft, hop, filters)
    mix_db = _band_log_energy(dubbed, sample_rate, n_fft, hop, filters)
    frames = min(ref_db.shape[1], mix_db.shape[1])
    if frames < 8:
        return 0.0
    a = ref_db[:, :frames] - ref_db[:, :frames].mean(axis=1, keepdims=True)
    b = mix_db[:, :frames] - mix_db[:, :frames].mean(axis=1, keepdims=True)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return 0.0
    return float(np.sum(a * b) / denominator)


def select_speech_free_segments(
    original_mix: np.ndarray,
    dubbed_mix: np.ndarray,
    original_speech: np.ndarray,
    dubbed_speech: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    speech_rms_db_max: float = DEFAULT_SPEECH_RMS_DB_MAX,
    mix_rms_db_min: float = DEFAULT_MIX_RMS_DB_MIN,
    max_refined_offset_ms: float = DEFAULT_MAX_REFINED_OFFSET_MS,
    min_coherence: float = DEFAULT_MIN_COHERENCE,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    min_spacing_sec: float = DEFAULT_MIN_SPACING_SEC,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> SegmentSelectionReport:
    """Отбирает окна без речи в ОБЕИХ версиях с когерентной M&E.

    Входы уже должны быть на одной временной шкале (original прогнан через
    warp_original_to_dubbed_timeline).  Речевые стемы служат VAD-гейтом: RU-
    диктор поверх титров (речь только в dubbed) отбрасывает окно точно так же,
    как EN-речь в оригинале.
    """
    length = min(len(original_mix), len(dubbed_mix), len(original_speech), len(dubbed_speech))
    window = int(round(window_sec * sample_rate))
    step = max(1, int(round(hop_sec * sample_rate)))
    filters, _centers = _mel_filterbank(sample_rate, n_fft, DEFAULT_BANDS)
    report = SegmentSelectionReport(segments=[])
    candidates: list[SpeechFreeSegment] = []
    max_offset_sec = max(0.4, 2.0 * max_refined_offset_ms / 1000.0)
    for start in range(0, max(0, length - window + 1), step):
        stop = start + window
        report.windows_total += 1
        original_speech_db = _rms_db(original_speech[start:stop])
        dubbed_speech_db = _rms_db(dubbed_speech[start:stop])
        if original_speech_db > speech_rms_db_max or dubbed_speech_db > speech_rms_db_max:
            report.rejected_speech += 1
            continue
        original_db = _rms_db(original_mix[start:stop])
        dubbed_db = _rms_db(dubbed_mix[start:stop])
        if original_db < mix_rms_db_min or dubbed_db < mix_rms_db_min:
            report.rejected_quiet += 1
            continue
        offset_sec, _score = normalized_xcorr(
            dubbed_mix[start:stop], original_mix[start:stop], sample_rate, max_offset_sec
        )
        if abs(offset_sec) * 1000.0 > max_refined_offset_ms:
            report.rejected_offset += 1
            continue
        shift = int(round(offset_sec * sample_rate))
        dubbed_start = min(max(0, start + shift), max(0, length - window))
        coherence = _spectral_coherence(
            original_mix[start:stop],
            dubbed_mix[dubbed_start : dubbed_start + window],
            sample_rate,
            n_fft=n_fft,
            hop=hop,
            filters=filters,
        )
        if coherence < min_coherence:
            report.rejected_coherence += 1
            continue
        candidates.append(
            SpeechFreeSegment(
                start_sec=start / sample_rate,
                duration_sec=window / sample_rate,
                refined_offset_ms=offset_sec * 1000.0,
                coherence=coherence,
                original_rms_db=original_db,
                dubbed_rms_db=dubbed_db,
            )
        )
    # Лучшие по когерентности, разнесённые по фильму: профиль не должен быть
    # оценкой одной длинной музыкальной темы.
    candidates.sort(key=lambda segment: -segment.coherence)
    chosen: list[SpeechFreeSegment] = []
    for candidate in candidates:
        if any(abs(candidate.start_sec - other.start_sec) < min_spacing_sec for other in chosen):
            report.rejected_spacing += 1
            continue
        chosen.append(candidate)
        if len(chosen) >= max_segments:
            break
    chosen.sort(key=lambda segment: segment.start_sec)
    # Каждый третий сегмент — holdout: на нём профиль только проверяется.
    for index, segment in enumerate(chosen):
        segment.role = "holdout" if index % 3 == 2 else "fit"
    if chosen and not any(segment.role == "holdout" for segment in chosen):
        chosen[-1].role = "holdout"
    report.segments = chosen
    return report


def _segment_band_gains_db(
    original: np.ndarray,
    dubbed: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int,
    hop: int,
    filters: np.ndarray,
) -> np.ndarray | None:
    ref_db = _band_log_energy(original, sample_rate, n_fft, hop, filters)
    mix_db = _band_log_energy(dubbed, sample_rate, n_fft, hop, filters)
    frames = min(ref_db.shape[1], mix_db.shape[1])
    if frames < 8:
        return None
    ref_db = ref_db[:, :frames]
    mix_db = mix_db[:, :frames]
    gains = np.zeros(ref_db.shape[0], dtype=np.float32)
    for band in range(ref_db.shape[0]):
        # Только информативные кадры полосы: тихие кадры измеряют шум пола.
        active = ref_db[band] >= max(float(np.percentile(ref_db[band], 85)) - 28.0, -80.0)
        if int(np.count_nonzero(active)) < 4:
            active = np.ones(frames, dtype=bool)
        gains[band] = float(np.median(mix_db[band, active] - ref_db[band, active]))
    return gains


def estimate_global_profile(
    original_mix: np.ndarray,
    dubbed_mix: np.ndarray,
    sample_rate: int,
    selection: SegmentSelectionReport,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    bands: int = DEFAULT_BANDS,
    max_abs_gain_db: float = DEFAULT_MAX_ABS_GAIN_DB,
) -> dict[str, Any]:
    """Один глобальный профиль по fit-сегментам + валидация на holdout.

    Возвращаемый JSON читается ``ReferenceAdapterProfile.from_json`` и потому
    применяется существующим ``adapt_reference_file_with_profile`` без правок.
    """
    filters, centers_hz = _mel_filterbank(sample_rate, n_fft, bands)
    fit_segments = [segment for segment in selection.segments if segment.role == "fit"]
    holdout_segments = [segment for segment in selection.segments if segment.role == "holdout"]
    per_segment: list[np.ndarray] = []
    length = min(len(original_mix), len(dubbed_mix))

    def _pair(segment: SpeechFreeSegment) -> tuple[np.ndarray, np.ndarray]:
        start = int(round(segment.start_sec * sample_rate))
        window = int(round(segment.duration_sec * sample_rate))
        shift = int(round(segment.refined_offset_ms / 1000.0 * sample_rate))
        dubbed_start = min(max(0, start + shift), max(0, length - window))
        return (
            original_mix[start : start + window],
            dubbed_mix[dubbed_start : dubbed_start + window],
        )

    for segment in fit_segments:
        original, dubbed = _pair(segment)
        gains = _segment_band_gains_db(original, dubbed, sample_rate, n_fft=n_fft, hop=hop, filters=filters)
        if gains is not None:
            per_segment.append(gains)

    if per_segment:
        stacked = np.stack(per_segment)
        band_gains = np.median(stacked, axis=0)
        band_spread_iqr = np.subtract(*np.percentile(stacked, [75, 25], axis=0))
    else:
        band_gains = np.zeros(bands, dtype=np.float32)
        band_spread_iqr = np.zeros(bands, dtype=np.float32)
    band_gains = gaussian_filter1d(band_gains.astype(np.float32), sigma=1.1, mode="nearest")
    band_gains = np.clip(band_gains, -abs(max_abs_gain_db), abs(max_abs_gain_db))

    profile: dict[str, Any] = {
        "schema_version": 1,
        "mode": PROFILE_MODE,
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "hop": int(hop),
        "bands": int(bands),
        "band_centers_hz": [float(value) for value in centers_hz],
        "band_gains_db": [float(value) for value in band_gains],
        # Поля ReferenceAdapterProfile: смещение здесь не оценивается глобально,
        # локальные offsets — гейт отбора, а не часть профиля.
        "delay_sec": 0.0,
        "delay_confidence": 0.0,
        "distance_before": {},
        "distance_after_estimate": {},
        "selection": selection.to_json(),
        "band_spread_iqr_db": [float(value) for value in band_spread_iqr],
    }

    # Holdout-валидация: профиль обязан улучшать сегменты, которые не видел.
    improvements: list[float] = []
    for segment in holdout_segments:
        original, dubbed = _pair(segment)
        adapted = apply_profile(original, ReferenceAdapterProfile.from_json(profile))
        before = _logmel_distance(original, dubbed, sample_rate, n_fft=n_fft, hop=hop, bands=bands)
        after = _logmel_distance(adapted, dubbed, sample_rate, n_fft=n_fft, hop=hop, bands=bands)
        if not profile["distance_before"]:
            profile["distance_before"] = before
            profile["distance_after_estimate"] = after
        # Keep the sign.  Clamping negative values to zero hid profiles that
        # actively made unseen windows worse.
        improvements.append(
            (1.0 - after["median"] / max(before["median"], 1e-8)) * 100.0
        )

    holdout_improvement = float(np.median(improvements)) if improvements else 0.0
    median_iqr = float(np.median(band_spread_iqr)) if len(band_spread_iqr) else 0.0
    gain_span = float(np.max(band_gains) - np.min(band_gains)) if len(band_gains) else 0.0
    median_gain = float(np.median(band_gains)) if len(band_gains) else 0.0
    saturated = (
        np.abs(band_gains)
        >= max(0.0, abs(float(max_abs_gain_db)) - DEFAULT_SATURATION_MARGIN_DB)
    )
    saturated_fraction = float(np.mean(saturated)) if len(band_gains) else 0.0
    profile_saturated = bool(
        saturated_fraction > DEFAULT_MAX_SATURATED_BAND_FRACTION
        or abs(median_gain)
        >= max(0.0, abs(float(max_abs_gain_db)) - DEFAULT_SATURATION_MARGIN_DB)
    )
    if len(fit_segments) < 4 or len(holdout_segments) < 1:
        verdict = "insufficient_segments"
    elif profile_saturated:
        verdict = "saturated_profile"
    elif median_iqr > 6.0:
        verdict = "unstable_profile"
    elif gain_span < 3.0 and abs(float(np.median(band_gains))) < 1.5:
        verdict = "matched_channels"
    elif holdout_improvement < 10.0:
        verdict = "no_holdout_gain"
    else:
        verdict = "ok"
    profile["summary"] = {
        "verdict": verdict,
        "usable": verdict == "ok",
        "recommended_mode": "global" if verdict == "ok" else "raw",
        "fit_segments": len(fit_segments),
        "holdout_segments": len(holdout_segments),
        "holdout_improvement_percent": holdout_improvement,
        "median_gain_db": median_gain,
        "min_gain_db": float(np.min(band_gains)) if len(band_gains) else 0.0,
        "max_gain_db": float(np.max(band_gains)) if len(band_gains) else 0.0,
        "median_band_iqr_db": median_iqr,
        "saturated_band_fraction": saturated_fraction,
        "profile_saturated": profile_saturated,
        "max_abs_gain_db": float(max_abs_gain_db),
        # Совместимость с текущим selection-кодом pipeline, который читает
        # distance_reduction_percent из summary алгоритмического профиля.
        "distance_reduction_percent": holdout_improvement,
    }
    return profile


def _read_mono_window(path: Path, start_frame: int, frames: int) -> np.ndarray:
    with sf.SoundFile(str(path)) as handle:
        start_frame = min(max(0, start_frame), max(0, handle.frames - 1))
        handle.seek(start_frame)
        block = handle.read(min(frames, handle.frames - start_frame), dtype="float32", always_2d=True)
    return _mono(block)


def _stream_envelope_db(path: Path, frame_sec: float) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as handle:
        rate = int(handle.samplerate)
        frame = max(1, int(round(rate * frame_sec)))
        tail = np.zeros(0, dtype=np.float32)
        while True:
            block = handle.read(frame * 512, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            mono = np.concatenate([tail, _mono(block)])
            usable = (len(mono) // frame) * frame
            if usable:
                chunks.append(_frame_rms_db(mono[:usable], frame))
            tail = mono[usable:]
    envelope = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return envelope, rate


def fit_global_profile_for_files(
    original_mix_path: Path,
    dubbed_mix_path: Path,
    original_speech_path: Path,
    dubbed_speech_path: Path,
    *,
    report_path: Path | None = None,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    min_coherence: float = DEFAULT_MIN_COHERENCE,
    max_abs_gain_db: float = DEFAULT_MAX_ABS_GAIN_DB,
) -> dict[str, Any]:
    """Файловый вариант для полных фильмов: потоковые огибающие вместо чтения
    полутора гигабайт в память; аудио читается только для окон-кандидатов.

    ВАЖНО: original_mix уже должен быть на шкале dubbed (после warp по
    alignment_map), как aligned_en_speech в текущем full-пайплайне.
    """
    frame_sec = 0.05
    original_env, original_rate = _stream_envelope_db(Path(original_mix_path), frame_sec)
    dubbed_env, dubbed_rate = _stream_envelope_db(Path(dubbed_mix_path), frame_sec)
    original_speech_env, speech_rate_a = _stream_envelope_db(Path(original_speech_path), frame_sec)
    dubbed_speech_env, speech_rate_b = _stream_envelope_db(Path(dubbed_speech_path), frame_sec)
    rates = {original_rate, dubbed_rate, speech_rate_a, speech_rate_b}
    if len(rates) != 1:
        raise RuntimeError(f"Все дорожки должны иметь один sample_rate, получено: {sorted(rates)}")
    sample_rate = original_rate
    frames_per_window = int(round(window_sec / frame_sec))
    step = max(1, int(round(hop_sec / frame_sec)))
    total = min(len(original_env), len(dubbed_env), len(original_speech_env), len(dubbed_speech_env))

    selection = SegmentSelectionReport(segments=[])
    window_frames = int(round(window_sec * sample_rate))
    filters, _centers = _mel_filterbank(sample_rate, DEFAULT_N_FFT, DEFAULT_BANDS)
    max_offset_sec = max(0.4, 2.0 * DEFAULT_MAX_REFINED_OFFSET_MS / 1000.0)
    candidates: list[SpeechFreeSegment] = []
    for env_start in range(0, max(0, total - frames_per_window + 1), step):
        env_stop = env_start + frames_per_window
        selection.windows_total += 1
        if (
            float(np.max(original_speech_env[env_start:env_stop])) > DEFAULT_SPEECH_RMS_DB_MAX
            or float(np.max(dubbed_speech_env[env_start:env_stop])) > DEFAULT_SPEECH_RMS_DB_MAX
        ):
            selection.rejected_speech += 1
            continue
        original_db = float(np.median(original_env[env_start:env_stop]))
        dubbed_db = float(np.median(dubbed_env[env_start:env_stop]))
        if original_db < DEFAULT_MIX_RMS_DB_MIN or dubbed_db < DEFAULT_MIX_RMS_DB_MIN:
            selection.rejected_quiet += 1
            continue
        start_frame = int(round(env_start * frame_sec * sample_rate))
        original_window = _read_mono_window(Path(original_mix_path), start_frame, window_frames)
        dubbed_window = _read_mono_window(Path(dubbed_mix_path), start_frame, window_frames)
        if min(len(original_window), len(dubbed_window)) < window_frames // 2:
            selection.rejected_quiet += 1
            continue
        offset_sec, _score = normalized_xcorr(dubbed_window, original_window, sample_rate, max_offset_sec)
        if abs(offset_sec) * 1000.0 > DEFAULT_MAX_REFINED_OFFSET_MS:
            selection.rejected_offset += 1
            continue
        shifted_dubbed = _read_mono_window(
            Path(dubbed_mix_path), start_frame + int(round(offset_sec * sample_rate)), window_frames
        )
        coherence = _spectral_coherence(
            original_window,
            shifted_dubbed,
            sample_rate,
            n_fft=DEFAULT_N_FFT,
            hop=DEFAULT_HOP,
            filters=filters,
        )
        if coherence < min_coherence:
            selection.rejected_coherence += 1
            continue
        candidates.append(
            SpeechFreeSegment(
                start_sec=start_frame / sample_rate,
                duration_sec=window_sec,
                refined_offset_ms=offset_sec * 1000.0,
                coherence=coherence,
                original_rms_db=original_db,
                dubbed_rms_db=dubbed_db,
            )
        )
    candidates.sort(key=lambda segment: -segment.coherence)
    chosen: list[SpeechFreeSegment] = []
    for candidate in candidates:
        if any(abs(candidate.start_sec - other.start_sec) < DEFAULT_MIN_SPACING_SEC for other in chosen):
            selection.rejected_spacing += 1
            continue
        chosen.append(candidate)
        if len(chosen) >= max_segments:
            break
    chosen.sort(key=lambda segment: segment.start_sec)
    for index, segment in enumerate(chosen):
        segment.role = "holdout" if index % 3 == 2 else "fit"
    if chosen and not any(segment.role == "holdout" for segment in chosen):
        chosen[-1].role = "holdout"
    selection.segments = chosen

    # Preserve the actual film positions for reports and listening audits.
    # The compact arrays below need synthetic 0, 6, 12... coordinates, but
    # those must never replace the source timeline in the exported JSON.
    source_selection_json = selection.to_json()

    # Для оценки читаем только отобранные окна: склеиваем их в компактные
    # массивы и пересчитываем стартовые позиции сегментов.
    joined_original: list[np.ndarray] = []
    joined_dubbed: list[np.ndarray] = []
    cursor = 0
    for segment in chosen:
        start_frame = int(round(segment.start_sec * sample_rate))
        shift = int(round(segment.refined_offset_ms / 1000.0 * sample_rate))
        joined_original.append(_read_mono_window(Path(original_mix_path), start_frame, window_frames))
        joined_dubbed.append(_read_mono_window(Path(dubbed_mix_path), start_frame + shift, window_frames))
        segment.start_sec = cursor / sample_rate
        segment.refined_offset_ms = 0.0
        cursor += window_frames
    original_joined = np.concatenate(joined_original) if joined_original else np.zeros(0, dtype=np.float32)
    dubbed_joined = np.concatenate(joined_dubbed) if joined_dubbed else np.zeros(0, dtype=np.float32)
    profile = estimate_global_profile(
        original_joined,
        dubbed_joined,
        sample_rate,
        selection,
        max_abs_gain_db=max_abs_gain_db,
    )
    profile["selection"] = source_selection_json
    profile.update(
        {
            "original_mix": str(Path(original_mix_path).resolve()),
            "dubbed_mix": str(Path(dubbed_mix_path).resolve()),
            "original_speech": str(Path(original_speech_path).resolve()),
            "dubbed_speech": str(Path(dubbed_speech_path).resolve()),
        }
    )
    if report_path is not None:
        atomic_json(Path(report_path), profile)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-mix", required=True, type=Path, help="Полный original mix, уже на шкале dubbed")
    parser.add_argument("--dubbed-mix", required=True, type=Path)
    parser.add_argument("--original-speech", required=True, type=Path, help="Речевой стем original (VAD-гейт)")
    parser.add_argument("--dubbed-speech", required=True, type=Path, help="Речевой стем dubbed (VAD-гейт)")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    parser.add_argument("--min-coherence", type=float, default=DEFAULT_MIN_COHERENCE)
    args = parser.parse_args()
    profile = fit_global_profile_for_files(
        args.original_mix,
        args.dubbed_mix,
        args.original_speech,
        args.dubbed_speech,
        report_path=args.report,
        max_segments=args.max_segments,
        min_coherence=args.min_coherence,
    )
    print(json.dumps(profile["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
