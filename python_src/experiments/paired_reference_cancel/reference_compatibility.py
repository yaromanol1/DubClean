"""Mastering-invariant compatibility analysis for original/dubbed references.

The analyser is intentionally independent from the subtraction models.  It
produces a persisted routing contract; applying that contract is a separate
user action handled by the application API.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel.storage import Store, atomic_json, read_json, utc_now


SCHEMA_VERSION = 1


def _analysis_cfg(store: Store) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "window_sec": 20.0,
        "step_sec": 10.0,
        "max_lag_sec": 1.5,
        "mel_bands": 24,
        "min_reference_activity": 0.08,
        "min_active_windows": 3,
        "low_confidence": 0.42,
        "timeline_offset_tolerance_sec": 0.010,
        "timeline_drift_tolerance_sec": 0.025,
        "thresholds": {"high": 0.72, "mid": 0.48},
        "weights": {
            "logmel_envelope": 0.36,
            "gcc_lag_stability": 0.22,
            "vad_spectral": 0.32,
            "reference_activity": 0.10,
        },
    }
    configured = store.cfg.get("reference_compatibility") or {}
    merged = {**defaults, **configured}
    merged["thresholds"] = {**defaults["thresholds"], **(configured.get("thresholds") or {})}
    merged["weights"] = {**defaults["weights"], **(configured.get("weights") or {})}
    return merged


def _mono(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return np.nan_to_num(data, copy=False)


def _read_interval(handle: sf.SoundFile, start_sec: float, duration_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * handle.samplerate)))
    frames = max(1, int(round(duration_sec * handle.samplerate)))
    handle.seek(min(start, max(0, len(handle) - 1)))
    return _mono(handle.read(frames, dtype="float32", always_2d=True))


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return data
    divisor = math.gcd(int(source_rate), int(target_rate))
    return signal.resample_poly(data, target_rate // divisor, source_rate // divisor).astype(np.float32)


def _mel_filters(rate: int, n_fft: int, bands: int) -> np.ndarray:
    def hz_to_mel(value: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)

    def mel_to_hz(value: np.ndarray) -> np.ndarray:
        return 700.0 * (np.power(10.0, value / 2595.0) - 1.0)

    low = float(hz_to_mel(70.0))
    high = float(hz_to_mel(min(rate * 0.48, 3800.0)))
    points = mel_to_hz(np.linspace(low, high, bands + 2))
    bins = np.clip(np.floor((n_fft + 1) * points / rate).astype(int), 0, n_fft // 2)
    filters = np.zeros((bands, n_fft // 2 + 1), dtype=np.float32)
    for index in range(bands):
        left, center, right = bins[index : index + 3]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for position in range(left, min(center, filters.shape[1])):
            filters[index, position] = (position - left) / max(1, center - left)
        for position in range(center, min(right, filters.shape[1])):
            filters[index, position] = (right - position) / max(1, right - center)
    return filters


def _logmel(data: np.ndarray, rate: int, bands: int) -> tuple[np.ndarray, np.ndarray]:
    n_fft = 512 if rate >= 8000 else 256
    hop = max(1, int(round(rate * 0.02)))
    if data.size < n_fft:
        data = np.pad(data, (0, n_fft - data.size))
    _, _, spectrum = signal.stft(
        data,
        fs=rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum).astype(np.float32) ** 2
    mel = _mel_filters(rate, n_fft, bands) @ power
    logmel = np.log1p(20.0 * mel).T
    energy = np.sqrt(np.maximum(np.mean(power, axis=0), 1e-12)).astype(np.float32)
    return logmel.astype(np.float32), energy


def _standardize(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float64)
    deviation = float(np.std(data))
    if data.size < 2 or deviation < 1e-8:
        return np.zeros_like(data)
    return (data - float(np.mean(data))) / deviation


def _best_envelope_lag(
    original: np.ndarray, dubbed: np.ndarray, max_frames: int
) -> tuple[float, int]:
    original = _standardize(original)
    dubbed = _standardize(dubbed)
    best_score, best_lag = -1.0, 0
    for lag in range(-max_frames, max_frames + 1):
        if lag >= 0:
            left, right = original[lag:], dubbed[: dubbed.size - lag or None]
        else:
            left, right = original[: original.size + lag], dubbed[-lag:]
        count = min(left.size, right.size)
        if count < 8:
            continue
        score = float(np.mean(left[:count] * right[:count]))
        if score > best_score:
            best_score, best_lag = score, lag
    return float(np.clip((best_score + 1.0) * 0.5, 0.0, 1.0)), best_lag


def _gcc_phat_lag(left: np.ndarray, right: np.ndarray, rate: int, max_lag_sec: float) -> float | None:
    count = min(left.size, right.size)
    if count < max(256, rate // 2):
        return None
    left = left[:count] - float(np.mean(left[:count]))
    right = right[:count] - float(np.mean(right[:count]))
    if float(np.sqrt(np.mean(left * left))) < 1e-5 or float(np.sqrt(np.mean(right * right))) < 1e-5:
        return None
    size = 1 << int(math.ceil(math.log2(max(2, count * 2 - 1))))
    cross = np.fft.rfft(left, size) * np.conj(np.fft.rfft(right, size))
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.irfft(cross, size)
    max_shift = min(int(round(max_lag_sec * rate)), size // 2 - 1)
    corr = np.concatenate((corr[-max_shift:], corr[: max_shift + 1]))
    shift = int(np.argmax(np.abs(corr))) - max_shift
    return float(shift / rate)


def _lag_stability(original: np.ndarray, dubbed: np.ndarray, rate: int, max_lag_sec: float) -> tuple[float, list[float]]:
    segment_frames = max(rate * 3, 1)
    count = min(original.size, dubbed.size)
    lags: list[float] = []
    for start in range(0, max(1, count - segment_frames + 1), segment_frames):
        lag = _gcc_phat_lag(
            original[start : start + segment_frames],
            dubbed[start : start + segment_frames],
            rate,
            max_lag_sec,
        )
        if lag is not None:
            lags.append(lag)
    if not lags:
        return 0.0, []
    values = np.asarray(lags, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    coverage = min(1.0, len(lags) / max(1.0, count / segment_frames))
    stability = math.exp(-mad / 0.045) * coverage
    return float(np.clip(stability, 0.0, 1.0)), [round(item, 5) for item in lags]


def _spectral_similarity(
    original: np.ndarray,
    dubbed: np.ndarray,
    original_energy: np.ndarray,
    lag_frames: int,
) -> tuple[float, float]:
    if lag_frames >= 0:
        left, right, energy = original[lag_frames:], dubbed[: dubbed.shape[0] - lag_frames or None], original_energy[lag_frames:]
    else:
        left, right, energy = original[: original.shape[0] + lag_frames], dubbed[-lag_frames:], original_energy[: original_energy.size + lag_frames]
    count = min(left.shape[0], right.shape[0], energy.size)
    if count < 8:
        return 0.0, 0.0
    left, right, energy = left[:count], right[:count], energy[:count]
    floor = float(np.percentile(energy, 55.0))
    active = energy > max(floor, 1e-5)
    activity = float(np.mean(active))
    if int(np.sum(active)) < 4:
        return 0.0, activity
    left = left[active]
    right = right[active]
    left = left - np.mean(left, axis=1, keepdims=True)
    right = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-7)
    return float(np.clip(np.median((cosine + 1.0) * 0.5), 0.0, 1.0)), activity


def _map_original_interval(alignment: dict[str, Any], dubbed_start: float, duration: float) -> tuple[float, float] | None:
    center = dubbed_start + duration * 0.5
    manual = float(alignment.get("manual_correction_sec") or 0.0)
    segments = [item for item in (alignment.get("segments") or []) if item.get("usable", True)]
    for item in segments:
        dub_a = float(item.get("dubbed_start") or 0.0)
        dub_b = float(item.get("dubbed_end") or dub_a)
        if dub_a <= center <= dub_b and dub_b > dub_a:
            orig_a = float(item.get("original_start") or 0.0)
            orig_b = float(item.get("original_end") or orig_a)
            ratio = (orig_b - orig_a) / (dub_b - dub_a)
            mapped = orig_a + (dubbed_start - dub_a) * ratio + manual
            return max(0.0, mapped), max(0.1, duration * max(0.85, min(1.15, ratio)))
    global_map = alignment.get("global") or {}
    ratio = float(global_map.get("speed_ratio") or 1.0)
    # ``offset_sec`` is a delay: original_time = dubbed_time - offset.  Adding
    # it here doubled the error instead of removing it whenever a map arrived
    # without usable segments.
    offset = float(global_map.get("offset_sec") or 0.0)
    return max(0.0, dubbed_start * ratio - offset + manual), max(0.1, duration * ratio)


def _window_features(
    original: np.ndarray,
    dubbed: np.ndarray,
    rate: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    count = min(original.size, dubbed.size)
    original, dubbed = original[:count], dubbed[:count]
    original_logmel, original_energy = _logmel(original, rate, int(cfg["mel_bands"]))
    dubbed_logmel, _ = _logmel(dubbed, rate, int(cfg["mel_bands"]))
    frame_count = min(original_logmel.shape[0], dubbed_logmel.shape[0])
    original_logmel, dubbed_logmel = original_logmel[:frame_count], dubbed_logmel[:frame_count]
    original_energy = original_energy[:frame_count]
    original_envelope = np.mean(original_logmel, axis=1)
    dubbed_envelope = np.mean(dubbed_logmel, axis=1)
    max_frames = max(1, int(round(float(cfg["max_lag_sec"]) / 0.02)))
    envelope_score, lag_frames = _best_envelope_lag(original_envelope, dubbed_envelope, max_frames)
    lag_stability, lags = _lag_stability(original, dubbed, rate, float(cfg["max_lag_sec"]))
    # ``_gcc_phat_lag`` answers "where does the original sit inside the dubbed
    # window"; the rest of the product speaks in delays, where
    # original_time = dubbed_time - delay.  Hence the sign flip.
    window_offset = -float(np.median(lags)) if lags else 0.0
    spectral_score, activity = _spectral_similarity(
        original_logmel, dubbed_logmel, original_energy, lag_frames
    )
    activity_score = float(np.clip(activity / max(float(cfg["min_reference_activity"]), 1e-4), 0.0, 1.0))
    weights = cfg["weights"]
    score = (
        float(weights["logmel_envelope"]) * envelope_score
        + float(weights["gcc_lag_stability"]) * lag_stability
        + float(weights["vad_spectral"]) * spectral_score
        + float(weights["reference_activity"]) * activity_score
    )
    return {
        "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "logmel_envelope_correlation": round(envelope_score, 4),
        "best_logmel_lag_sec": round(lag_frames * 0.02, 4),
        "gcc_phat_lag_stability": round(lag_stability, 4),
        "gcc_phat_lags_sec": lags,
        "timeline_offset_sec": round(window_offset, 4),
        "vad_spectral_similarity": round(spectral_score, 4),
        "reference_activity": round(activity, 4),
    }


def _timeline_action(
    offset_median_sec: float, offset_drift_sec: float, cfg: dict[str, Any]
) -> str:
    """Whether the pair still needs its timeline measured again.

    This is deliberately not part of the quality band.  A dub that steps by a
    tenth of a second mid-reel matches the original on every mastering cue and
    scores HIGH, so folding the two together is what let a 190 ms staircase be
    reported as "дорожки хорошо соответствуют друг другу".

    The offsets are residuals: they are measured after the current alignment
    map has already been applied, so a map that describes the pair correctly
    leaves nothing here and a fabricated one shows its whole error.
    """
    drift_tolerance = max(0.0, float(cfg.get("timeline_drift_tolerance_sec", 0.025)))
    offset_tolerance = max(0.0, float(cfg.get("timeline_offset_tolerance_sec", 0.010)))
    if abs(float(offset_drift_sec)) > drift_tolerance:
        return "realign_local"
    if abs(float(offset_median_sec)) > offset_tolerance:
        return "realign_global"
    return "none"


def _route(
    score: float,
    confidence: float,
    lag_stability: float,
    cfg: dict[str, Any],
) -> dict[str, str]:
    """Choose the subtraction model and the mastering reference adapter.

    ``alignment`` here names the level/EQ adapter that ``automatic_assembly``
    turns into ``reference_adapter``; the timeline is a separate question,
    answered by ``_timeline_action``.
    """
    high = float(cfg["thresholds"]["high"])
    mid = float(cfg["thresholds"]["mid"])
    if confidence < float(cfg["low_confidence"]):
        return {"band": "PASSTHROUGH", "model": "passthrough", "alignment": "none"}
    if score >= high:
        return {"band": "HIGH", "model": "dubclean_voice", "alignment": "none"}
    if score >= mid:
        alignment = "algorithmic" if lag_stability >= 0.62 else "local_auto"
        return {"band": "MID", "model": "dubclean_voice", "alignment": alignment}
    return {"band": "LOW", "model": "passthrough", "alignment": "none"}


def analyze_reference_compatibility(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
) -> dict[str, Any]:
    pair = store.load_pair(project_id, pair_id)
    root = store.pair_dir(project_id, pair_id)
    original_path = root / "extracted" / "original_proxy.wav"
    dubbed_path = root / "extracted" / "dubbed_proxy.wav"
    alignment_path = root / "alignment" / "alignment_map.json"
    if not original_path.is_file() or not dubbed_path.is_file() or not alignment_path.is_file():
        raise RuntimeError("Сначала извлеките и сопоставьте дорожки.")

    cfg = _analysis_cfg(store)
    alignment = read_json(alignment_path, {}) or {}
    window_sec = max(4.0, float(cfg["window_sec"]))
    step_sec = max(1.0, float(cfg["step_sec"]))
    windows: list[dict[str, Any]] = []
    ctx.update(force=True, stage="Анализ соответствия", substage="Подготовка окон", progress=1.0, current_file=pair.get("name", ""))

    with sf.SoundFile(str(original_path)) as original_file, sf.SoundFile(str(dubbed_path)) as dubbed_file:
        rate = int(dubbed_file.samplerate)
        duration = float(len(dubbed_file) / dubbed_file.samplerate)
        starts = np.arange(0.0, max(0.01, duration - window_sec * 0.5), step_sec).tolist()
        if not starts:
            starts = [0.0]
        for index, dubbed_start in enumerate(starts):
            ctx.check_stop()
            current_duration = min(window_sec, max(0.1, duration - dubbed_start))
            mapped = _map_original_interval(alignment, dubbed_start, current_duration)
            if mapped is None:
                continue
            original_start, original_duration = mapped
            original = _read_interval(original_file, original_start, original_duration)
            dubbed = _read_interval(dubbed_file, dubbed_start, current_duration)
            original = _resample(original, int(original_file.samplerate), rate)
            if original.size != dubbed.size and original.size > 1 and dubbed.size > 1:
                original = signal.resample(original, dubbed.size).astype(np.float32)
            features = _window_features(original, dubbed, rate, cfg)
            activity = float(features["reference_activity"])
            confidence = float(np.clip(
                0.45 * min(1.0, activity / max(float(cfg["min_reference_activity"]), 1e-4))
                + 0.35 * float(features["gcc_phat_lag_stability"])
                + 0.20 * float(features["logmel_envelope_correlation"]),
                0.0,
                1.0,
            ))
            route = _route(
                float(features["score"]), confidence,
                float(features["gcc_phat_lag_stability"]), cfg,
            )
            windows.append({
                "index": index,
                "dubbed_start_sec": round(float(dubbed_start), 3),
                "dubbed_end_sec": round(float(dubbed_start + current_duration), 3),
                "original_start_sec": round(float(original_start), 3),
                "original_end_sec": round(float(original_start + original_duration), 3),
                "band": route["band"],
                "confidence": round(confidence, 4),
                "recommended_model": route["model"],
                "recommended_alignment": route["alignment"],
                "features": features,
            })
            ctx.update(
                stage="Анализ соответствия",
                substage=f"Окно {index + 1} из {len(starts)}",
                progress=2.0 + 94.0 * (index + 1) / len(starts),
                current_file=pair.get("name", ""),
            )

    active = [
        item for item in windows
        if float((item.get("features") or {}).get("reference_activity") or 0.0)
        >= float(cfg["min_reference_activity"])
    ]
    warnings: list[str] = []
    offsets = np.asarray(
        [float((item["features"] or {}).get("timeline_offset_sec") or 0.0) for item in active],
        dtype=np.float64,
    )
    timeline_offset = float(np.median(offsets)) if offsets.size else 0.0
    # The spread between windows, never inside one: a delay that holds for a
    # whole reel and then steps looks perfectly stable locally, which is how a
    # 190 ms staircase used to pass as a match.
    timeline_drift = (
        float(np.percentile(offsets, 90.0) - np.percentile(offsets, 10.0))
        if offsets.size >= 4
        else 0.0
    )
    if len(active) < int(cfg["min_active_windows"]):
        band, confidence, model, alignment = "PASSTHROUGH", 0.0, "passthrough", "none"
        timeline = "none"
        summary = "Недостаточно активной исходной речи для надёжного анализа; русская речь будет сохранена без изменений."
        warnings.append("Недостаточно данных в активных речевых окнах.")
    else:
        scores = np.asarray([float(item["features"]["score"]) for item in active])
        stabilities = np.asarray([float(item["features"]["gcc_phat_lag_stability"]) for item in active])
        window_confidence = np.asarray([float(item["confidence"]) for item in active])
        aggregate_score = float(np.median(scores))
        dispersion = float(np.median(np.abs(scores - aggregate_score)))
        coverage = min(1.0, len(active) / max(1.0, len(windows) * 0.25))
        confidence = float(np.clip(np.median(window_confidence) * coverage * math.exp(-dispersion / 0.22), 0.0, 1.0))
        route = _route(
            aggregate_score,
            confidence,
            float(np.median(stabilities)),
            cfg,
        )
        band, model, alignment = route["band"], route["model"], route["alignment"]
        timeline = _timeline_action(timeline_offset, timeline_drift, cfg)
        if band == "HIGH":
            summary = "Дорожки хорошо соответствуют друг другу."
        elif band == "MID":
            summary = "Есть различия мастеринга, рекомендуется алгоритмическое выравнивание." if alignment == "algorithmic" else "Есть локальные различия, рекомендуется автоматическое выравнивание по участкам."
        elif band == "LOW":
            summary = "Исходная речь слабо соответствует переводу; русская речь будет сохранена без вычитания."
            warnings.append("Надёжное вычитание для этих дорожек не рекомендовано.")
        else:
            summary = "Надёжное вычитание невозможно, русская речь будет сохранена без изменений."
        # Mastering similarity and timeline correctness are different questions.
        # A dub can match the original in every spectral cue and still sit a
        # tenth of a second away from it, and that is exactly the case whose
        # music and effects bed used to be assembled on the wrong timeline.
        if timeline == "realign_local":
            summary += (
                f" Смещение перевода меняется по фильму (разброс "
                f"{timeline_drift * 1000.0:.0f} мс): постройте карту "
                "сопоставления заново."
            )
            warnings.append(
                "Смещение перевода относительно оригинала непостоянно: разброс "
                f"{timeline_drift * 1000.0:.0f} мс при текущей карте. Общая "
                "ручная поправка такие ступени не исправляет — переcоберите "
                "сопоставление дорожек."
            )
        elif timeline == "realign_global":
            summary += (
                f" Перевод смещён относительно оригинала на "
                f"{timeline_offset * 1000.0:.0f} мс: постройте карту "
                "сопоставления заново."
            )
            warnings.append(
                "Перевод смещён относительно оригинала на "
                f"{timeline_offset * 1000.0:.0f} мс при текущей карте "
                "сопоставления."
            )

    aggregate_features = {
        "analysed_windows": len(windows),
        "active_windows": len(active),
        "window_sec": window_sec,
        "step_sec": step_sec,
        "score_median": round(float(np.median([item["features"]["score"] for item in active])) if active else 0.0, 4),
        "logmel_envelope_correlation_median": round(float(np.median([item["features"]["logmel_envelope_correlation"] for item in active])) if active else 0.0, 4),
        "gcc_phat_lag_stability_median": round(float(np.median([item["features"]["gcc_phat_lag_stability"] for item in active])) if active else 0.0, 4),
        "timeline_offset_median_sec": round(timeline_offset, 4),
        "timeline_offset_drift_sec": round(timeline_drift, 4),
        "vad_spectral_similarity_median": round(float(np.median([item["features"]["vad_spectral_similarity"] for item in active])) if active else 0.0, 4),
        "reference_activity_median": round(float(np.median([item["features"]["reference_activity"] for item in active])) if active else 0.0, 4),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "band": band,
        "confidence": round(confidence, 4),
        "recommended_model": model,
        "recommended_alignment": alignment,
        # The product has one production subtraction model.  What the user can
        # meaningfully choose after this analysis is whether to run the
        # extraction module once more over the cleaned Russian voice.  MID
        # material benefits most often: it still goes through the production
        # model, but mastering differences leave more soft foreign artefacts.
        "recommended_speech_extraction_second_pass": band == "MID",
        # The timeline verdict is separate from the mastering band on purpose:
        # it names an action on the alignment map, not on the reference adapter.
        "recommended_timeline_action": timeline,
        "summary": summary,
        "features": aggregate_features,
        "warnings": warnings,
    }
    routing = {
        "schema_version": SCHEMA_VERSION,
        "created_at": result["created_at"],
        "window_sec": window_sec,
        "step_sec": step_sec,
        "thresholds": cfg["thresholds"],
        "weights": cfg["weights"],
        "aggregate": result,
        "windows": windows,
    }
    analysis_root = root / "analysis"
    routing_path = analysis_root / "routing_map.json"
    atomic_json(routing_path, routing)
    result["routing_map"] = str(routing_path.resolve())
    pair["reference_compatibility_analysis"] = result
    # A fresh analysis invalidates only the previously applied recommendation,
    # never the user's current manual selection or any processing artefacts.
    pair.pop("reference_compatibility_applied", None)
    pair.pop("reference_compatibility_decision", None)
    store.save_pair(pair)
    ctx.update(force=True, stage="Анализ соответствия", substage="Оценка сохранена", progress=100.0, current_file=pair.get("name", ""))
    return result


def apply_reference_compatibility_recommendations(
    store: Store,
    project_id: str,
    pair_id: str,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the current analysis as the effective processing decision."""
    pair = store.load_pair(project_id, pair_id)
    result = dict(analysis or pair.get("reference_compatibility_analysis") or {})
    if result.get("band") not in {"HIGH", "MID", "LOW", "PASSTHROUGH"}:
        raise RuntimeError("Сначала выполните анализ соответствия дорожек.")

    pair_root = store.pair_dir(project_id, pair_id).resolve()
    routing_path = Path(str(result.get("routing_map") or "")).resolve()
    try:
        routing_path.relative_to(pair_root)
    except ValueError as error:
        raise RuntimeError("Карта маршрутизации находится вне проекта.") from error
    if not routing_path.is_file():
        raise RuntimeError("Карта маршрутизации анализа отсутствует. Запустите анализ заново.")

    model = str(result.get("recommended_model") or "passthrough")
    if model not in {"dubclean_voice", "passthrough"}:
        model = "passthrough"
    alignment = str(result.get("recommended_alignment") or "none")
    if alignment not in {"none", "algorithmic", "local_auto"}:
        alignment = "none"
    second_pass = bool(
        result.get(
            "recommended_speech_extraction_second_pass",
            result.get("band") == "MID",
        )
    )

    applied = {
        **result,
        "recommended_model": model,
        "recommended_alignment": alignment,
        "recommended_speech_extraction_second_pass": second_pass,
        "routing_map": str(routing_path),
        "applied_at": utc_now(),
    }
    manual_second_pass_key = (
        "reference_compatibility_manual_speech_extraction_second_pass"
    )
    if manual_second_pass_key not in pair:
        pair[manual_second_pass_key] = bool(
            pair.get("speech_extraction_second_pass", False)
        )
    pair["reference_compatibility_applied"] = applied
    pair["reference_compatibility_effective_model"] = model
    pair["speech_extraction_second_pass"] = second_pass
    pair["reference_compatibility_decision"] = {
        "choice": "recommendations",
        "analysis_created_at": result.get("created_at"),
        "speech_extraction_second_pass": second_pass,
        "decided_at": utc_now(),
    }
    store.save_pair(pair)
    return applied
