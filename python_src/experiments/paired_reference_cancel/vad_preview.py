"""Deterministic, local Silero-VAD selection for DubClean preview scenes.

The module deliberately knows nothing about the separator models.  It analyses
only the dubbed timeline and returns a compact, serialisable description that
the application pipeline can use for every before/after preview layer.
"""
from __future__ import annotations

import math
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf
from scipy import signal


SILERO_VAD_VERSION = "6.2.0"
TARGET_SAMPLE_RATE = 16_000
ZONE_WINDOW_SEC = 180.0
PREFERRED_PREVIEW_SEC = 30.0
MIN_PREVIEW_SEC = 10.0
CONTEXT_SEC = 0.75
MERGE_GAP_SEC = 0.80

VAD_PARAMETERS = {
    "sampling_rate": TARGET_SAMPLE_RATE,
    "threshold": 0.50,
    "neg_threshold": 0.35,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 350,
    "speech_pad_ms": 80,
    "merge_gap_sec": MERGE_GAP_SEC,
    "preview_duration_sec": PREFERRED_PREVIEW_SEC,
    "context_sec": CONTEXT_SEC,
}

ZONE_SPECS = (
    ("start", None),
    ("25_percent", 0.25),
    ("50_percent", 0.50),
    ("75_percent", 0.75),
    ("90_percent", 0.90),
)


def build_analysis_zones(duration_sec: float) -> list[dict[str, Any]]:
    """Return the five requested windows, always clamped to the source file."""
    duration = max(0.0, float(duration_sec))
    result: list[dict[str, Any]] = []
    for name, percentage in ZONE_SPECS:
        window_duration = min(ZONE_WINDOW_SEC, duration)
        if name == "start":
            start = 0.0
        elif window_duration <= 0.0:
            start = 0.0
        else:
            center = duration * float(percentage)
            start = center - window_duration / 2.0
            start = min(max(0.0, start), max(0.0, duration - window_duration))
        end = min(duration, start + window_duration)
        result.append(
            {
                "zone": name,
                "percentage": percentage,
                "analysis_start_sec": round(start, 3),
                "analysis_end_sec": round(end, 3),
                "analysis_duration_sec": round(max(0.0, end - start), 3),
            }
        )
    return result


@lru_cache(maxsize=1)
def _load_model() -> Any:
    # The wheel ships this JIT model locally.  Loading from bytes also avoids a
    # Windows torch.jit limitation with non-ASCII portable-folder names.
    import importlib.resources as resources
    import torch
    import silero_vad.data

    model_bytes = resources.files(silero_vad.data).joinpath("silero_vad.jit").read_bytes()
    return torch.jit.load(BytesIO(model_bytes), map_location="cpu")


def _mono(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _resample(values: np.ndarray, rate: int) -> np.ndarray:
    if rate == TARGET_SAMPLE_RATE:
        return values.astype(np.float32, copy=False)
    divisor = math.gcd(int(rate), TARGET_SAMPLE_RATE)
    return signal.resample_poly(values, TARGET_SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)


def _read_window(path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    with sf.SoundFile(str(path)) as reader:
        rate = int(reader.samplerate)
        start_frame = max(0, min(len(reader), int(round(start_sec * rate))))
        end_frame = max(start_frame, min(len(reader), int(round(end_sec * rate))))
        reader.seek(start_frame)
        values = reader.read(end_frame - start_frame, dtype="float32", always_2d=True)
    return _resample(_mono(values), rate)


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for left, right in sorted((float(a), float(b)) for a, b in intervals if b > a):
        if merged and left - merged[-1][1] <= MERGE_GAP_SEC:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(round(left, 3), round(right, 3)) for left, right in merged]


def _speech_overlap(intervals: Sequence[tuple[float, float]], left: float, right: float) -> tuple[float, float]:
    total = 0.0
    longest = 0.0
    for start, end in intervals:
        overlap = max(0.0, min(right, end) - max(left, start))
        total += overlap
        longest = max(longest, overlap)
    return total, longest


def _candidate_starts(
    zone_start: float,
    zone_end: float,
    preview_duration: float,
    intervals: Sequence[tuple[float, float]],
    allowed_intervals: Sequence[tuple[float, float]] | None,
) -> list[float]:
    upper = zone_end - preview_duration
    if upper < zone_start - 1e-6:
        return []
    starts = {zone_start, upper}
    for start, end in intervals:
        starts.update((start - CONTEXT_SEC, end - preview_duration + CONTEXT_SEC, (start + end - preview_duration) / 2.0))
    # A sparse deterministic grid catches connected dialogue that starts in the
    # middle of a long interval without making selection depend on randomness.
    cursor = zone_start
    while cursor <= upper + 1e-6:
        starts.add(cursor)
        cursor += 0.5
    candidates: list[float] = []
    for raw in starts:
        value = min(max(zone_start, raw), upper)
        if allowed_intervals and not any(
            value >= allowed_start - 1e-4 and value + preview_duration <= allowed_end + 1e-4
            for allowed_start, allowed_end in allowed_intervals
        ):
            continue
        candidates.append(round(value, 3))
    return sorted(set(candidates))


def _select_candidate(
    zone: dict[str, Any],
    intervals: Sequence[tuple[float, float]],
    probabilities: np.ndarray,
    allowed_intervals: Sequence[tuple[float, float]] | None,
) -> dict[str, Any] | None:
    start = float(zone["analysis_start_sec"])
    end = float(zone["analysis_end_sec"])
    available = end - start
    if available <= 0.0:
        return None
    preview_duration = min(PREFERRED_PREVIEW_SEC, available)
    if available >= MIN_PREVIEW_SEC:
        preview_duration = max(MIN_PREVIEW_SEC, preview_duration)
    starts = _candidate_starts(start, end, preview_duration, intervals, allowed_intervals)
    if not starts:
        return None
    frame_sec = 512.0 / TARGET_SAMPLE_RATE
    best: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    center = (start + end) / 2.0
    for candidate_start in starts:
        candidate_end = candidate_start + preview_duration
        speech_duration, longest = _speech_overlap(intervals, candidate_start, candidate_end)
        left_frame = max(0, int(math.floor((candidate_start - start) / frame_sec)))
        right_frame = min(len(probabilities), int(math.ceil((candidate_end - start) / frame_sec)))
        local = probabilities[left_frame:right_frame]
        confidence = float(np.mean(local)) if len(local) else 0.0
        # Speech coverage dominates; confidence and an unbroken phrase resolve
        # ties.  The last key is only a stable tie-breaker, not a fixed point.
        score = (speech_duration / preview_duration) * 100.0 + longest * 0.35 + confidence * 5.0
        key = (score, confidence, -abs((candidate_start + candidate_end) / 2.0 - center))
        value = {
            "preview_start_sec": round(candidate_start, 3),
            "preview_end_sec": round(candidate_end, 3),
            "preview_duration_sec": round(preview_duration, 3),
            "speech_duration_sec": round(speech_duration, 3),
            "speech_ratio": round(speech_duration / preview_duration, 5),
            "candidate_confidence": round(confidence, 5),
        }
        if best is None or key > best[0]:
            best = (key, value)
    return best[1] if best else None


def _probabilities(audio: np.ndarray) -> np.ndarray:
    import torch

    model = _load_model()
    if hasattr(model, "reset_states"):
        model.reset_states()
    values = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    result: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(values), 512):
            block = values[offset: offset + 512]
            if len(block) < 512:
                block = torch.nn.functional.pad(block, (0, 512 - len(block)))
            result.append(float(model(block, TARGET_SAMPLE_RATE).item()))
    return np.asarray(result, dtype=np.float32)


def _timestamps(audio: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    model = _load_model()
    if hasattr(model, "reset_states"):
        model.reset_states()
    stamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
        model,
        threshold=threshold,
        sampling_rate=TARGET_SAMPLE_RATE,
        min_speech_duration_ms=int(VAD_PARAMETERS["min_speech_duration_ms"]),
        min_silence_duration_ms=int(VAD_PARAMETERS["min_silence_duration_ms"]),
        speech_pad_ms=int(VAD_PARAMETERS["speech_pad_ms"]),
        neg_threshold=float(VAD_PARAMETERS["neg_threshold"]),
        return_seconds=True,
    )
    return _merge_intervals((float(item["start"]), float(item["end"])) for item in stamps)


def analyse_zone(
    audio_path: Path,
    zone: dict[str, Any],
    allowed_intervals: Sequence[tuple[float, float]] | None = None,
    *,
    probability_reader: Callable[[np.ndarray], np.ndarray] | None = None,
    timestamp_reader: Callable[[np.ndarray, float], Sequence[tuple[float, float]]] | None = None,
) -> dict[str, Any]:
    """Analyse one zone and return JSON-safe metadata plus a stable selection."""
    value = dict(zone)
    start = float(value["analysis_start_sec"])
    end = float(value["analysis_end_sec"])
    value.update({
        "silero_vad_version": SILERO_VAD_VERSION,
        "vad_parameters": dict(VAD_PARAMETERS),
        "fallback": {"used": False, "reason": None},
    })
    if end <= start:
        value.update({"status": "zone_unavailable", "selection_reason": "Аудиофайл не содержит эту временную зону."})
        return value
    audio = _read_window(audio_path, start, end)
    if len(audio) < 512:
        value.update({"status": "zone_unavailable", "selection_reason": "Зона слишком короткая для VAD."})
        return value
    probabilities = (probability_reader or _probabilities)(audio)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    value["average_speech_confidence"] = round(float(np.mean(probabilities)), 5)
    value["max_speech_confidence"] = round(float(np.max(probabilities)), 5)
    reader = timestamp_reader or _timestamps
    intervals = list(reader(audio, float(VAD_PARAMETERS["threshold"])))
    local_zone = {**value, "analysis_start_sec": 0.0, "analysis_end_sec": end - start}
    local_allowed = (
        [(max(0.0, left - start), min(end - start, right - start)) for left, right in allowed_intervals]
        if allowed_intervals
        else None
    )
    chosen = _select_candidate(local_zone, intervals, probabilities, local_allowed)
    reliable = bool(chosen and chosen["speech_duration_sec"] >= min(1.5, chosen["preview_duration_sec"] * 0.12) and chosen["speech_ratio"] >= 0.12)
    mode = "default"
    if not reliable:
        # First retain a genuine high-probability fragment even if it is short:
        # quiet dialogue should be exposed as low-confidence, not discarded.
        if chosen and value["max_speech_confidence"] >= 0.35 and chosen["speech_duration_sec"] >= 0.25:
            reliable = True
            mode = "maximum_probability"
        else:
            soft_intervals = list(reader(audio, 0.35))
            soft_chosen = _select_candidate(local_zone, soft_intervals, probabilities, local_allowed)
            if soft_chosen and soft_chosen["speech_duration_sec"] >= 0.50:
                intervals, chosen, reliable, mode = soft_intervals, soft_chosen, True, "soft_threshold"
    value["speech_intervals"] = [
        {"start_sec": round(start + left, 3), "end_sec": round(start + right, 3)}
        for left, right in intervals
    ]
    value["speech_interval_count"] = len(intervals)
    if not reliable or chosen is None:
        value.update({
            "status": "no_speech_found",
            "selection_reason": "Silero VAD не подтвердил пригодную речевую область в этой зоне.",
        })
        return value
    chosen["preview_start_sec"] = round(start + float(chosen["preview_start_sec"]), 3)
    chosen["preview_end_sec"] = round(start + float(chosen["preview_end_sec"]), 3)
    in_preview = [
        {
            "start_sec": max(float(item["start_sec"]), float(chosen["preview_start_sec"])),
            "end_sec": min(float(item["end_sec"]), float(chosen["preview_end_sec"])),
        }
        for item in value["speech_intervals"]
        if item["end_sec"] > chosen["preview_start_sec"] and item["start_sec"] < chosen["preview_end_sec"]
    ]
    value.update(chosen)
    value["speech_region_start_sec"] = min((item["start_sec"] for item in in_preview), default=None)
    value["speech_region_end_sec"] = max((item["end_sec"] for item in in_preview), default=None)
    value["status"] = "speech_selected" if mode == "default" else "speech_selected_low_confidence"
    value["selection_reason"] = {
        "default": "Выбран стабильный 10–30-секундный участок с максимальной долей подтверждённой речи.",
        "maximum_probability": "Выбран участок с максимальной вероятностью речи; уверенность ниже обычной.",
        "soft_threshold": "Выбран участок после повторного VAD-анализа с более мягким порогом.",
    }[mode]
    return value


def analyse_file(
    audio_path: Path,
    duration_sec: float,
    allowed_intervals: Sequence[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    return [analyse_zone(audio_path, zone, allowed_intervals) for zone in build_analysis_zones(duration_sec)]


def waveform_svg(
    audio_path: Path,
    window_start_sec: float,
    window_end_sec: float,
    *,
    highlights: Sequence[tuple[float, float]] | None = None,
    marker_start_sec: float | None = None,
    marker_end_sec: float | None = None,
    width: int = 900,
    height: int = 90,
) -> str:
    """Render a self-contained peak waveform of one window as inline SVG.

    All times are in the audio file's own timeline.  ``highlights`` (VAD speech
    regions) are drawn as marks along the bottom; the optional marker pair frames
    the selected preview window.  Returns "" if the window cannot be read, so
    callers can safely inline the result without a try/except of their own.
    """
    try:
        start = float(window_start_sec)
        end = float(window_end_sec)
        if not (end > start):
            return ""
        with sf.SoundFile(str(audio_path)) as reader:
            rate = int(reader.samplerate)
            first = max(0, min(len(reader), int(round(start * rate))))
            last = max(first, min(len(reader), int(round(end * rate))))
            if last <= first:
                return ""
            reader.seek(first)
            data = reader.read(last - first, dtype="float32", always_2d=True)
        mono = _mono(data)
        if mono.size == 0:
            return ""
        bars = min(600, max(1, mono.size))
        bounds = (np.arange(bars + 1, dtype=np.int64) * mono.size) // bars
        peaks = np.maximum.reduceat(np.abs(mono), bounds[:bars])
        scale = float(np.percentile(np.abs(mono), 99.5)) or 1.0
        span = end - start
        mid = height / 2.0
        bar_w = width / bars

        def x_of(value: float) -> float:
            return (min(end, max(start, float(value))) - start) / span * width

        pieces: list[str] = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" role="img" '
            f'style="background:#eef1f6;border-radius:8px;display:block;margin:8px 0">'
        ]
        for index in range(bars):
            bar_h = min(1.0, float(peaks[index]) / scale) * (height - 6)
            pieces.append(
                f'<rect x="{index * bar_w:.2f}" y="{mid - bar_h / 2.0:.2f}" '
                f'width="{max(0.6, bar_w - 0.4):.2f}" height="{max(1.0, bar_h):.2f}" fill="#2c9361"/>'
            )
        if marker_start_sec is not None and marker_end_sec is not None:
            x0, x1 = x_of(marker_start_sec), x_of(marker_end_sec)
            pieces.append(f'<rect x="{x0:.1f}" y="0" width="{max(1.0, x1 - x0):.1f}" height="{height}" fill="#2c9361" opacity="0.16"/>')
            pieces.append(f'<rect x="{x0:.1f}" y="0" width="1.6" height="{height}" fill="#c0392b"/>')
            pieces.append(f'<rect x="{x1 - 1.6:.1f}" y="0" width="1.6" height="{height}" fill="#c0392b"/>')
        for left, right in highlights or ():
            a, b = x_of(left), x_of(right)
            if b > a:
                pieces.append(f'<rect x="{a:.1f}" y="{height - 6}" width="{max(1.0, b - a):.1f}" height="5" fill="#e67e22"/>')
        pieces.append("</svg>")
        return "".join(pieces)
    except Exception:
        return ""
