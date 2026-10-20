from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SCHEMA_VERSION = 2
ALGORITHM = "conservative_copy_move_contract_v2"

SEPARATION_SCHEMA_VERSION = 1
SEPARATION_ALGORITHM = "stem_vs_mixture_passthrough_detector_v1"

# Moving "speech" is only meaningful when the stem being moved actually is
# speech.  When the extractor returns the mixture unchanged, every segment the
# detector finds contains the film's music and effects too, and shifting those
# segments shreds a continuous background into hundreds of displaced fragments.
# Measured over the shipped projects, a genuinely separated stem and a
# passthrough stem sit far apart on all three axes, so the gate demands that
# every one of them agree before it disables synchronization:
#
#   pair                 correlation   level drop dB   residual dB
#   separated                  0.70            5.09          -2.4
#   separated                  0.81            2.95          -4.6
#   passthrough                0.97            0.48         -12.5
#   passthrough                0.97            0.47         -12.8
#   passthrough                0.98            0.20         -14.7
DEFAULT_SEPARATION_GUARD: dict[str, float] = {
    "analysis_sample_rate": 16000.0,
    "window_sec": 8.0,
    "step_sec": 45.0,
    "minimum_window_rms_db": -50.0,
    "minimum_windows": 8.0,
    "passthrough_correlation": 0.93,
    "passthrough_level_drop_db": 1.5,
    "passthrough_residual_ratio_db": -9.0,
    # A window whose residual is this close to the mixture is one the extractor
    # emptied: it heard no speech and said so.
    "silence_evidence_db": -3.0,
    "passthrough_max_silence_share": 0.03,
}


@dataclass(frozen=True)
class SpeechSegment:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class SegmentMove:
    source_start_sec: float
    source_end_sec: float
    destination_start_sec: float
    reference_start_sec: float | None
    shift_sec: float
    matched: bool


def _mono(block: np.ndarray) -> np.ndarray:
    values = np.asarray(block, dtype=np.float32)
    if values.ndim == 1:
        return values
    if values.shape[1] >= 3:
        # The centre channel normally contains dialogue in surround material.
        return values[:, 2]
    return values.mean(axis=1)


def _read_window(path: Path, start_sec: float, duration_sec: float, rate: int) -> np.ndarray:
    """Read one mono window and bring it to ``rate`` for comparison."""
    from scipy import signal as _signal

    with sf.SoundFile(str(path)) as reader:
        source_rate = int(reader.samplerate)
        reader.seek(min(int(round(start_sec * source_rate)), max(reader.frames - 1, 0)))
        block = reader.read(
            int(round(duration_sec * source_rate)), dtype="float32", always_2d=True
        )
    values = _mono(block).astype(np.float64, copy=False)
    if values.size == 0 or source_rate == rate:
        return values
    return _signal.resample_poly(values, rate, source_rate)


def _rms_db(values: np.ndarray) -> float:
    if values.size == 0:
        return -180.0
    return 20.0 * math.log10(
        max(float(np.sqrt(np.mean(values * values))), 1e-12)
    )


def evaluate_stem_separation(
    mixture_path: str | Path,
    stem_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure whether a speech stem really separated speech from the mixture.

    Three scale-different views of the same question are taken on a sparse grid
    of windows and reduced by median, so one loud scene cannot decide the run:

    ``correlation``
        Waveform correlation between mixture and stem.  Scale invariant, so it
        still answers the question when the extractor applied a global gain.
    ``level_drop_db``
        How much quieter the stem is than the mixture.
    ``residual_ratio_db``
        Level of ``mixture - stem`` relative to the mixture.  A real film
        background is a few dB under the mixture; a passthrough leaves nothing.

    The verdict is ``separated`` unless all three agree on passthrough, because
    the expensive mistake is disabling synchronization on material that would
    have synchronized correctly.
    """
    settings = {**DEFAULT_SEPARATION_GUARD, **(config or {})}
    rate = int(settings["analysis_sample_rate"])
    window_sec = float(settings["window_sec"])
    step_sec = max(float(settings["step_sec"]), window_sec)
    mixture = Path(mixture_path)
    stem = Path(stem_path)
    report: dict[str, Any] = {
        "schema_version": SEPARATION_SCHEMA_VERSION,
        "algorithm": SEPARATION_ALGORITHM,
        "mixture": str(mixture.resolve()),
        "stem": str(stem.resolve()),
        "config": {key: float(value) for key, value in settings.items()},
        "separated": True,
        "windows": 0,
    }
    if not mixture.is_file() or not stem.is_file():
        report["reason"] = "missing_input"
        return report
    mixture_info = sf.info(str(mixture))
    duration = min(mixture_info.duration, sf.info(str(stem)).duration)
    # ``_mono`` mirrors the extractor's own channel choice, so the comparison
    # asks exactly "did the model change what it was given".  The layout is
    # recorded because it decides how much programme a passthrough drags along:
    # a stereo downmix carries the whole soundtrack, a 5.1 centre mostly dialogue.
    report["mixture_channels"] = int(mixture_info.channels)
    correlations: list[float] = []
    level_drops: list[float] = []
    residuals: list[float] = []
    position = min(60.0, max(0.0, duration * 0.05))
    while position + window_sec <= duration:
        mixture_window = _read_window(mixture, position, window_sec, rate)
        stem_window = _read_window(stem, position, window_sec, rate)
        size = min(mixture_window.size, stem_window.size)
        position += step_sec
        if size == 0:
            continue
        mixture_window = mixture_window[:size]
        stem_window = stem_window[:size]
        mixture_db = _rms_db(mixture_window)
        if mixture_db < float(settings["minimum_window_rms_db"]):
            continue
        centred_mixture = mixture_window - float(np.mean(mixture_window))
        centred_stem = stem_window - float(np.mean(stem_window))
        denominator = math.sqrt(
            float(np.dot(centred_mixture, centred_mixture))
            * float(np.dot(centred_stem, centred_stem))
        )
        # A window the extractor emptied has no defined correlation with its
        # mixture, and dropping it threw away the one measurement that proves
        # separation happened.  Silence carries none of the mixture, so zero is
        # the honest value rather than a missing row.
        correlations.append(
            float(np.dot(centred_mixture, centred_stem)) / denominator
            if denominator > 0.0
            else 0.0
        )
        level_drops.append(mixture_db - _rms_db(stem_window))
        residuals.append(_rms_db(mixture_window - stem_window) - mixture_db)
    report["windows"] = len(correlations)
    if len(correlations) < int(settings["minimum_windows"]):
        report["reason"] = "not_enough_measurable_windows"
        return report
    median_correlation = float(np.median(correlations))
    median_level_drop = float(np.median(level_drops))
    median_residual = float(np.median(residuals))
    # All three medians shift towards passthrough on material that simply is
    # speech from end to end — a dialogue reel, or the speech-densest scenes a
    # preview deliberately selects.  What no passthrough can produce is a window
    # it emptied, so one is required before the verdict is allowed to stand.
    silence_windows = int(
        np.sum(np.asarray(residuals) >= float(settings["silence_evidence_db"]))
    )
    silence_share = silence_windows / max(len(residuals), 1)
    report["measured"] = {
        "median_correlation": round(median_correlation, 4),
        "median_level_drop_db": round(median_level_drop, 3),
        "median_residual_ratio_db": round(median_residual, 3),
        "emptied_windows": silence_windows,
        "emptied_window_share": round(silence_share, 4),
    }
    passthrough = (
        median_correlation >= float(settings["passthrough_correlation"])
        and median_level_drop <= float(settings["passthrough_level_drop_db"])
        and median_residual <= float(settings["passthrough_residual_ratio_db"])
        and silence_share <= float(settings["passthrough_max_silence_share"])
    )
    report["separated"] = not passthrough
    report["reason"] = (
        "stem_repeats_the_mixture" if passthrough else "stem_carries_separated_speech"
    )
    if not passthrough and silence_share > float(
        settings["passthrough_max_silence_share"]
    ):
        report["separation_evidence"] = "extractor_emptied_speech_free_windows"
    return report


def _frame_levels(path: Path, frame_sec: float = 0.02) -> tuple[np.ndarray, int]:
    levels: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as reader:
        rate = int(reader.samplerate)
        frame_size = max(1, int(round(rate * frame_sec)))
        blocksize = frame_size * 1000
        for block in reader.blocks(
            blocksize=blocksize, dtype="float32", always_2d=True
        ):
            mono = _mono(block)
            remainder = len(mono) % frame_size
            if remainder:
                mono = np.pad(mono, (0, frame_size - remainder))
            frames = mono.reshape(-1, frame_size).astype(np.float64, copy=False)
            rms = np.sqrt(np.maximum(np.mean(frames * frames, axis=1), 1e-12))
            levels.append(
                (20.0 * np.log10(np.maximum(rms, 1e-9))).astype(np.float32)
            )
    return (
        np.concatenate(levels) if levels else np.empty(0, dtype=np.float32),
        rate,
    )


def _segments_from_levels(
    levels_db: np.ndarray,
    frame_sec: float = 0.02,
    *,
    minimum_speech_sec: float = 0.08,
    join_gap_sec: float = 0.24,
    margin_sec: float = 0.10,
) -> tuple[list[SpeechSegment], float]:
    if len(levels_db) == 0:
        return [], -50.0
    finite = levels_db[np.isfinite(levels_db)]
    if len(finite) == 0:
        return [], -50.0
    noise = float(np.percentile(finite, 20))
    speech = float(np.percentile(finite, 92))
    # Adaptive, but bounded: quiet words must survive and extractor hiss must
    # not turn the entire film into one continuous segment.
    threshold = float(np.clip(noise + 0.28 * max(8.0, speech - noise), -52.0, -32.0))
    active = levels_db >= threshold
    if not np.any(active):
        return [], threshold

    indices = np.flatnonzero(active)
    max_gap_frames = max(1, int(round(join_gap_sec / frame_sec)))
    minimum_frames = max(1, int(round(minimum_speech_sec / frame_sec)))
    margin_frames = max(0, int(round(margin_sec / frame_sec)))
    raw: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        value = int(value)
        if value - previous > max_gap_frames:
            raw.append((start, previous + 1))
            start = value
        previous = value
    raw.append((start, previous + 1))

    expanded: list[tuple[int, int]] = []
    for left, right in raw:
        if right - left < minimum_frames:
            continue
        left = max(0, left - margin_frames)
        right = min(len(levels_db), right + margin_frames)
        if expanded and left <= expanded[-1][1]:
            expanded[-1] = (expanded[-1][0], max(expanded[-1][1], right))
        else:
            expanded.append((left, right))
    return [
        SpeechSegment(left * frame_sec, right * frame_sec)
        for left, right in expanded
    ], threshold


def detect_speech_segments(path: str | Path) -> tuple[list[SpeechSegment], dict[str, Any]]:
    source = Path(path)
    levels, rate = _frame_levels(source)
    segments, threshold = _segments_from_levels(levels)
    return segments, {
        "sample_rate": rate,
        "frame_sec": 0.02,
        "threshold_db": round(threshold, 2),
        "segment_count": len(segments),
    }


def plan_segment_moves(
    russian_segments: list[SpeechSegment],
    english_segments: list[SpeechSegment],
    *,
    maximum_advance_sec: float = 0.85,
    desired_lag_sec: float = 0.10,
    match_radius_sec: float = 2.50,
) -> list[SegmentMove]:
    """Match utterance onsets monotonically and only move RU earlier.

    Russian voice-over often starts after the English phrase has begun.  We
    preserve the Russian waveform and duration; only an onset delay is reduced.
    Moving later or time-stretching is intentionally forbidden because both can
    make an already-clean voice sound synthetic.
    """
    moves: list[SegmentMove] = []
    reference_cursor = 0
    previous_destination_end = 0.0
    for russian in russian_segments:
        best_index: int | None = None
        best_score = float("inf")
        for index in range(reference_cursor, len(english_segments)):
            english = english_segments[index]
            start_delta = russian.start_sec - english.start_sec
            if english.start_sec > russian.end_sec + match_radius_sec:
                break
            if abs(start_delta) > match_radius_sec:
                continue
            duration_ratio = russian.duration_sec / max(english.duration_sec, 0.08)
            if not 0.20 <= duration_ratio <= 5.0:
                continue
            # Prefer an English onset just before the Russian voice-over, while
            # still allowing near-simultaneous dubbing.
            direction_penalty = 0.45 if start_delta < -0.15 else 0.0
            score = abs(start_delta - 0.45) + 0.18 * abs(math.log(duration_ratio))
            score += direction_penalty
            if score < best_score:
                best_score = score
                best_index = index

        destination = russian.start_sec
        reference_start: float | None = None
        matched = best_index is not None
        if best_index is not None:
            english = english_segments[best_index]
            reference_start = english.start_sec
            requested = english.start_sec + desired_lag_sec
            # Advancing only: a translation that already starts before EN is not
            # delayed, and an extreme voice-over lag is corrected gradually.
            destination = max(
                russian.start_sec - maximum_advance_sec,
                min(russian.start_sec, requested),
            )
            reference_cursor = best_index + 1

        # Never reorder or overlap Russian phrases.  A tiny gap avoids clicks
        # after reconstruction.
        destination = max(destination, previous_destination_end + 0.015)
        if destination > russian.start_sec:
            destination = russian.start_sec
        shift = destination - russian.start_sec
        previous_destination_end = destination + russian.duration_sec
        moves.append(
            SegmentMove(
                source_start_sec=russian.start_sec,
                source_end_sec=russian.end_sec,
                destination_start_sec=destination,
                reference_start_sec=reference_start,
                shift_sec=shift,
                matched=matched,
            )
        )
    return moves


def _copy_segment(
    reader: sf.SoundFile,
    writer: sf.SoundFile,
    move: SegmentMove,
    rate: int,
    *,
    fade_sec: float,
) -> dict[str, int]:
    """Move one segment on top of an already copied source file.

    The source interval is removed and the destination interval is replaced,
    rather than added to.  This is important for overlapping source and
    destination ranges: addition would leave a delayed echo of the utterance.
    """
    source_start = max(0, int(round(move.source_start_sec * rate)))
    source_end = min(len(reader), int(round(move.source_end_sec * rate)))
    destination_start = max(0, int(round(move.destination_start_sec * rate)))
    frames = max(0, source_end - source_start)
    destination_end = min(len(reader), destination_start + frames)
    frames = min(frames, destination_end - destination_start)
    if frames <= 0:
        return {
            "source_start": source_start,
            "source_end": source_start,
            "destination_start": destination_start,
            "destination_end": destination_start,
            "fade_frames": 0,
        }

    reader.seek(source_start)
    moved = reader.read(frames, dtype="float32", always_2d=True)
    frames = len(moved)
    source_end = source_start + frames
    destination_end = destination_start + frames
    region_start = min(source_start, destination_start)
    region_end = max(source_end, destination_end)

    writer.seek(region_start)
    region = writer.read(
        region_end - region_start,
        dtype="float32",
        always_2d=True,
    )
    fade = min(frames // 2, max(1, int(round(rate * fade_sec))))

    source_left = source_start - region_start
    source_right = source_left + frames
    keep_source = np.zeros(frames, dtype=np.float32)
    if fade:
        phase = np.linspace(0.0, math.pi / 2.0, fade, dtype=np.float32)
        keep_source[:fade] = np.cos(phase)
        keep_source[-fade:] = np.sin(phase)
    region[source_left:source_right] *= keep_source[:, None]

    destination_left = destination_start - region_start
    destination_right = destination_left + frames
    insert_gain = np.ones(frames, dtype=np.float32)
    keep_destination = np.zeros(frames, dtype=np.float32)
    if fade:
        phase = np.linspace(0.0, math.pi / 2.0, fade, dtype=np.float32)
        insert_gain[:fade] = np.sin(phase)
        keep_destination[:fade] = np.cos(phase)
        insert_gain[-fade:] = np.cos(phase)
        keep_destination[-fade:] = np.sin(phase)
    region[destination_left:destination_right] = (
        region[destination_left:destination_right] * keep_destination[:, None]
        + moved * insert_gain[:, None]
    )
    writer.seek(region_start)
    writer.write(np.clip(region, -1.0, 1.0))
    return {
        "source_start": source_start,
        "source_end": source_end,
        "destination_start": destination_start,
        "destination_end": destination_end,
        "fade_frames": fade,
    }


def _normalise_preservation_intervals(
    intervals: list[SpeechSegment | tuple[float, float] | dict[str, Any]] | None,
) -> list[SpeechSegment]:
    normalised: list[SpeechSegment] = []
    for item in intervals or []:
        if isinstance(item, SpeechSegment):
            start, end = item.start_sec, item.end_sec
        elif isinstance(item, dict):
            start = float(item.get("start_sec", item.get("start", 0.0)))
            end = float(item.get("end_sec", item.get("end", start)))
        else:
            start, end = float(item[0]), float(item[1])
        start = max(0.0, float(start))
        end = max(start, float(end))
        if end > start:
            normalised.append(SpeechSegment(start, end))
    normalised.sort(key=lambda value: (value.start_sec, value.end_sec))
    merged: list[SpeechSegment] = []
    for interval in normalised:
        if merged and interval.start_sec <= merged[-1].end_sec:
            merged[-1] = SpeechSegment(
                merged[-1].start_sec,
                max(merged[-1].end_sec, interval.end_sec),
            )
        else:
            merged.append(interval)
    return merged


def _intersects_preservation(
    move: SegmentMove,
    intervals: list[SpeechSegment],
    *,
    margin_sec: float,
) -> bool:
    source_start = move.source_start_sec - margin_sec
    source_end = move.source_end_sec + margin_sec
    destination_start = move.destination_start_sec - margin_sec
    destination_end = destination_start + (move.source_end_sec - move.source_start_sec)
    destination_end += 2.0 * margin_sec
    for interval in intervals:
        source_overlap = (
            source_start < interval.end_sec and interval.start_sec < source_end
        )
        destination_overlap = (
            destination_start < interval.end_sec
            and interval.start_sec < destination_end
        )
        if source_overlap or destination_overlap:
            return True
    return False


def _copy_source_to_float_wave(source: Path, destination: Path) -> dict[str, int]:
    with sf.SoundFile(str(source)) as reader:
        metadata = {
            "frames": len(reader),
            "sample_rate": int(reader.samplerate),
            "channels": int(reader.channels),
        }
        with sf.SoundFile(
            str(destination),
            mode="w",
            samplerate=metadata["sample_rate"],
            channels=metadata["channels"],
            format="WAV",
            subtype="FLOAT",
        ) as writer:
            for block in reader.blocks(
                blocksize=max(1, metadata["sample_rate"] * 10),
                dtype="float32",
                always_2d=True,
            ):
                writer.write(block)
    return metadata


def _merged_sample_ranges(
    applications: list[dict[str, int]],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in applications:
        ranges.extend(
            [
                (item["source_start"], item["source_end"]),
                (item["destination_start"], item["destination_end"]),
            ]
        )
    ranges = sorted((left, right) for left, right in ranges if right > left)
    merged: list[tuple[int, int]] = []
    for left, right in ranges:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _validate_synchronization_contract(
    original: Path,
    candidate: Path,
    *,
    metadata: dict[str, int],
    shifted_move_count: int,
    applications: list[dict[str, int]],
    skipped_count: int,
    preservation_intervals: list[SpeechSegment],
) -> dict[str, Any]:
    """Verify duration, unchanged coverage and one destination per move."""
    reasons: list[str] = []
    with sf.SoundFile(str(original)) as source, sf.SoundFile(str(candidate)) as result:
        if (
            len(result) != metadata["frames"]
            or int(result.samplerate) != metadata["sample_rate"]
            or int(result.channels) != metadata["channels"]
        ):
            reasons.append("metadata_mismatch")

        modified = _merged_sample_ranges(applications)
        source.seek(0)
        result.seek(0)
        input_energy = 0.0
        output_energy = 0.0
        unchanged_max_error = 0.0
        cursor = 0
        range_index = 0
        blocksize = max(1, metadata["sample_rate"] * 10)
        while cursor < min(len(source), len(result)):
            count = min(blocksize, len(source) - cursor, len(result) - cursor)
            before = source.read(count, dtype="float32", always_2d=True)
            after = result.read(count, dtype="float32", always_2d=True)
            input_energy += float(np.sum(before.astype(np.float64) ** 2))
            output_energy += float(np.sum(after.astype(np.float64) ** 2))
            changed = np.zeros(count, dtype=bool)
            while range_index < len(modified) and modified[range_index][1] <= cursor:
                range_index += 1
            check_index = range_index
            while check_index < len(modified) and modified[check_index][0] < cursor + count:
                left, right = modified[check_index]
                changed[max(0, left - cursor):min(count, right - cursor)] = True
                check_index += 1
            if np.any(~changed):
                unchanged_max_error = max(
                    unchanged_max_error,
                    float(np.max(np.abs(after[~changed] - before[~changed]))),
                )
            cursor += count

        if unchanged_max_error > 1e-7:
            reasons.append("unmodified_audio_changed")
        energy_ratio = output_energy / max(input_energy, 1e-20)
        if input_energy > 1e-12 and not 0.50 <= energy_ratio <= 1.50:
            reasons.append("global_energy_out_of_bounds")

        move_checks: list[dict[str, Any]] = []
        for item in applications:
            frames = item["source_end"] - item["source_start"]
            fade = item["fade_frames"]
            core_left = min(frames, fade)
            core_right = max(core_left, frames - fade)
            relative_error = 0.0
            if core_right > core_left:
                source.seek(item["source_start"] + core_left)
                result.seek(item["destination_start"] + core_left)
                expected = source.read(
                    core_right - core_left, dtype="float32", always_2d=True
                )
                actual = result.read(
                    core_right - core_left, dtype="float32", always_2d=True
                )
                denominator = float(
                    np.sqrt(np.mean(expected.astype(np.float64) ** 2))
                )
                error = float(
                    np.sqrt(np.mean((actual - expected).astype(np.float64) ** 2))
                )
                relative_error = error / max(denominator, 1e-8)
                if relative_error > 0.03:
                    reasons.append("moved_segment_destination_mismatch")
            residual_ratio = 0.0
            residual_correlation = 0.0
            # The destination normally precedes and partially overlaps the
            # source.  Its trailing source-only part must not retain a second
            # correlated copy of the moved utterance.
            vacated_left = max(item["source_start"], item["destination_end"])
            vacated_right = item["source_end"]
            vacated_left += fade
            vacated_right -= fade
            if vacated_right > vacated_left:
                source.seek(vacated_left)
                result.seek(vacated_left)
                original_tail = source.read(
                    vacated_right - vacated_left,
                    dtype="float32",
                    always_2d=True,
                ).astype(np.float64)
                output_tail = result.read(
                    vacated_right - vacated_left,
                    dtype="float32",
                    always_2d=True,
                ).astype(np.float64)
                original_rms = float(np.sqrt(np.mean(original_tail ** 2)))
                output_rms = float(np.sqrt(np.mean(output_tail ** 2)))
                residual_ratio = output_rms / max(original_rms, 1e-8)
                denominator = float(
                    np.sqrt(np.sum(original_tail ** 2) * np.sum(output_tail ** 2))
                )
                if denominator > 1e-12:
                    residual_correlation = float(
                        np.sum(original_tail * output_tail) / denominator
                    )
                if residual_ratio > 0.25 and residual_correlation > 0.90:
                    reasons.append("moved_segment_left_duplicate_at_source")
            move_checks.append(
                {
                    "source_start_frame": item["source_start"],
                    "destination_start_frame": item["destination_start"],
                    "frames": frames,
                    "destination_relative_rms_error": round(relative_error, 8),
                    "source_residual_rms_ratio": round(residual_ratio, 8),
                    "source_residual_correlation": round(residual_correlation, 8),
                }
            )

        preservation_max_error = 0.0
        for interval in preservation_intervals:
            left = max(0, int(round(interval.start_sec * metadata["sample_rate"])))
            right = min(
                metadata["frames"],
                int(round(interval.end_sec * metadata["sample_rate"])),
            )
            if right <= left:
                continue
            source.seek(left)
            result.seek(left)
            remaining = right - left
            while remaining:
                count = min(blocksize, remaining)
                before = source.read(count, dtype="float32", always_2d=True)
                after = result.read(count, dtype="float32", always_2d=True)
                preservation_max_error = max(
                    preservation_max_error,
                    float(np.max(np.abs(after - before))) if len(before) else 0.0,
                )
                remaining -= count
        if preservation_max_error > 1e-7:
            reasons.append("preservation_interval_changed")

    if len(applications) + skipped_count != shifted_move_count:
        reasons.append("move_coverage_mismatch")
    modified_frames = sum(right - left for left, right in _merged_sample_ranges(applications))
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "duration_frames": metadata["frames"],
        "modified_frames": modified_frames,
        "untouched_frames": max(0, metadata["frames"] - modified_frames),
        "unchanged_max_abs_error": unchanged_max_error,
        "preservation_max_abs_error": preservation_max_error,
        "global_energy_ratio": round(float(energy_ratio), 8),
        "shifted_moves_accounted_for": len(applications) + skipped_count,
        "move_checks": move_checks,
    }


def _encode_flac(source_wave: Path, destination: Path) -> None:
    with sf.SoundFile(str(source_wave)) as reader:
        with sf.SoundFile(
            str(destination),
            mode="w",
            samplerate=int(reader.samplerate),
            channels=int(reader.channels),
            format="FLAC",
            subtype="PCM_24",
        ) as writer:
            for block in reader.blocks(
                blocksize=max(1, int(reader.samplerate) * 10),
                dtype="float32",
                always_2d=True,
            ):
                writer.write(block)


def synchronize_voice_file(
    russian_voice: str | Path,
    aligned_english_speech: str | Path,
    destination: str | Path,
    *,
    maximum_advance_sec: float = 0.85,
    preservation_intervals: list[
        SpeechSegment | tuple[float, float] | dict[str, Any]
    ] | None = None,
    crossfade_sec: float = 0.015,
) -> dict[str, Any]:
    russian_path = Path(russian_voice)
    english_path = Path(aligned_english_speech)
    output = Path(destination)
    russian_segments, russian_detection = detect_speech_segments(russian_path)
    english_segments, english_detection = detect_speech_segments(english_path)
    moves = plan_segment_moves(
        russian_segments,
        english_segments,
        maximum_advance_sec=maximum_advance_sec,
    )
    preserved = _normalise_preservation_intervals(preservation_intervals)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_wave = output.with_name(f".{output.stem}.{os.getpid()}.partial.wav")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    applications: list[dict[str, int]] = []
    skipped_moves: list[dict[str, Any]] = []
    fallback = False
    try:
        metadata = _copy_source_to_float_wave(russian_path, temporary_wave)
        shifted = [move for move in moves if move.shift_sec < -1e-4]
        with sf.SoundFile(str(russian_path)) as reader, sf.SoundFile(
            str(temporary_wave), mode="r+"
        ) as writer:
            for index, move in enumerate(shifted):
                if _intersects_preservation(
                    move,
                    preserved,
                    margin_sec=max(0.0, crossfade_sec),
                ):
                    skipped_moves.append(
                        {
                            "move_index": index,
                            "reason": "preservation_interval",
                            "move": asdict(move),
                        }
                    )
                    continue
                application = _copy_segment(
                    reader,
                    writer,
                    move,
                    metadata["sample_rate"],
                    fade_sec=max(0.0, crossfade_sec),
                )
                if application["source_end"] <= application["source_start"]:
                    skipped_moves.append(
                        {
                            "move_index": index,
                            "reason": "empty_segment",
                            "move": asdict(move),
                        }
                    )
                    continue
                application["move_index"] = index
                applications.append(application)
            writer.flush()
        try:
            contract = _validate_synchronization_contract(
                russian_path,
                temporary_wave,
                metadata=metadata,
                shifted_move_count=len(shifted),
                applications=applications,
                skipped_count=len(skipped_moves),
                preservation_intervals=preserved,
            )
        except Exception as error:  # pragma: no cover - defensive I/O fallback
            contract = {
                "valid": False,
                "reasons": [f"validation_error:{type(error).__name__}"],
            }
        if not contract.get("valid", False):
            fallback = True
            temporary_wave.unlink(missing_ok=True)
            _copy_source_to_float_wave(russian_path, temporary_wave)
        _encode_flac(temporary_wave, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_wave.unlink(missing_ok=True)

    shifts = [abs(move.shift_sec) for move in shifted]
    report = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "mode": "neural_stems_conservative_onset_alignment",
        "russian_detection": russian_detection,
        "english_detection": english_detection,
        "matched_segments": sum(1 for move in moves if move.matched),
        "shifted_segments": len(shifted),
        "mean_advance_sec": round(float(np.mean(shifts)) if shifts else 0.0, 4),
        "median_advance_sec": round(float(np.median(shifts)) if shifts else 0.0, 4),
        "maximum_advance_sec": maximum_advance_sec,
        "crossfade_sec": max(0.0, crossfade_sec),
        "preservation_intervals": [asdict(interval) for interval in preserved],
        "applied_shifted_segments": len(applications) if not fallback else 0,
        "skipped_shifted_segments": len(skipped_moves),
        "skipped_moves": skipped_moves,
        "contract": contract,
        "fallback_to_unsynchronized": fallback,
        "moves": [asdict(move) for move in moves],
        "output": str(output.resolve()),
    }
    return report
