"""Production workflow for applying a trained subtractor to one film pair."""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import subprocess
import time
import ctypes
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel import (
    audio_mix,
    audio_io,
    coincident_speech_protection,
    pipeline,
    song_protection,
    speech_integrity_guard,
    speech_synchronizer,
    vad_preview,
)
from experiments.paired_reference_cancel.global_mastering_neural import (
    fit_neural_global_profile_for_files,
)
from experiments.paired_reference_cancel.global_mastering_profile import (
    PROFILE_MODE as GLOBAL_MASTERING_PROFILE_MODE,
    fit_global_profile_for_files,
)
from experiments.paired_reference_cancel.neural_reference_adapter import (
    adapt_reference_file_neural,
)
from experiments.paired_reference_cancel.reference_adapter import (
    adapt_reference_file,
    adapt_reference_file_with_profile,
)
from experiments.paired_reference_cancel.speech_synchronizer import (
    synchronize_voice_file,
)
from experiments.paired_reference_cancel.storage import (
    Store,
    atomic_json,
    read_json,
    root_path,
    utc_now,
)

SCENE_SELECTION_VERSION = 9
REFERENCE_LEVEL_ALIGNMENT_CACHE_VERSION = 2
DEFAULT_NEURAL_REFERENCE_ADAPTER: Path | None = None
DEFAULT_GLOBAL_MASTERING_ADAPTER: Path | None = None

# A 30-second speech stem quieter than this is treated as "no speech here".
MIN_SPEECH_RMS_DB = -48.0
# Normalised waveform cross-correlation (±1.5 s) between the isolated original
# English and the dubbed EN+RU speech.  Reference subtraction can only remove a
# phase-coherent copy of the reference, so a peak this low means the dubbed track
# carries no cancellable English (different master / re-recorded soundtrack).
# Empirically a genuine shared voice-over reaches 0.3-0.8 here; five "Бронсон"
# scenes sat at 0.04-0.09 — indistinguishable from noise — and the model
# correctly passed them through unchanged.
FEASIBLE_SPEECH_CORRELATION = 0.20


# Internal artifact names are deliberately stable because cache sidecars and
# resumable full-film builds depend on them.  Files exposed to the user are
# published separately with the same concise Russian labels as the product UI.
USER_AUDIO_ARTIFACT_NAMES: dict[str, str] = {
    "original_en_speech": "Иностранная речь из оригинала.flac",
    "dubbed_en_ru_speech": "Речь перевода до очистки.flac",
    "russian_voice": "Очищенная русская речь.flac",
    "russian_voice_second_pass": "После повторного прохода.flac",
    "music_effects": "Музыка и эффекты.flac",
    "result_without_speech_synchronization": "До синхронизации речи.flac",
}


def _mono(data: np.ndarray) -> np.ndarray:
    values = audio_io.ensure_2d(np.asarray(data, dtype=np.float32))
    if values.shape[1] >= 3:
        return values[:, 2].astype(np.float32)
    return np.mean(values, axis=1).astype(np.float32)


def _rms_db(data: np.ndarray) -> float:
    values = np.asarray(data, dtype=np.float64)
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(values * values) + 1e-12)), 1e-9))


def _linear_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _applied_reference_routing(pair: dict[str, Any]) -> dict[str, Any]:
    value = dict(pair.get("reference_compatibility_applied") or {})
    return value if value.get("band") in {"HIGH", "MID", "LOW", "PASSTHROUGH"} else {}


def _routing_document(applied: dict[str, Any]) -> dict[str, Any]:
    value = str(applied.get("routing_map") or "").strip()
    path = Path(value).resolve() if value else None
    if path is None or not path.is_file():
        return {}
    return read_json(path, {}) or {}


def _copy_audio(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    # A passthrough copy is not a model inference.  Never leave a stale model
    # fingerprint that could make a later non-passthrough run reuse this file.
    destination.with_name(destination.name + ".source.json").unlink(missing_ok=True)
    return destination


class _SequentialAudioReader:
    """Serve small forward slices of a stream from large sequential reads.

    Block readers used to ask the decoder for exactly the slice they needed,
    ten seconds at a time.  On one real film that pattern made libsndfile 1.2.0
    fail at the same second of every attempt, while reading the identical bytes
    in thirty-second blocks decoded the whole file — re-encoding the artifact
    did not help, so the trigger is the read shape rather than the data.

    Reading in large blocks and slicing in memory sidesteps it, and is what the
    access pattern wanted anyway: the pass is strictly forward, so the decoder
    is never asked to seek at all.
    """

    def __init__(self, path: Path, block_sec: float = 30.0) -> None:
        self._reader = sf.SoundFile(str(path))
        self._block = max(1, int(round(block_sec * int(self._reader.samplerate))))
        self._buffer = np.zeros((0, int(self._reader.channels)), dtype=np.float32)
        self._position = 0

    @property
    def samplerate(self) -> int:
        return int(self._reader.samplerate)

    @property
    def channels(self) -> int:
        return int(self._reader.channels)

    def __len__(self) -> int:
        return len(self._reader)

    def read_at(self, start: int, frames: int) -> np.ndarray:
        """Return ``frames`` starting at ``start``, padding past the end.

        Only forward movement is supported; a request that points backwards
        reopens nothing and simply yields silence, because the routing pass
        never asks for it.
        """
        if frames <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        if start < self._position:
            return np.zeros((frames, self.channels), dtype=np.float32)
        while self._position + len(self._buffer) < start + frames:
            values = self._reader.read(
                self._block, dtype="float32", always_2d=True
            )
            if values.shape[0] == 0:
                break
            self._buffer = (
                values if self._buffer.shape[0] == 0
                else np.concatenate((self._buffer, values))
            )
        offset = start - self._position
        if offset > 0:
            self._buffer = self._buffer[offset:]
            self._position = start
        chunk = self._buffer[:frames]
        # Keep only what a later slice may still need.
        self._buffer = self._buffer[chunk.shape[0]:]
        self._position += chunk.shape[0]
        if chunk.shape[0] < frames:
            chunk = np.pad(chunk, ((0, frames - chunk.shape[0]), (0, 0)))
        return chunk

    def close(self) -> None:
        self._reader.close()


def _route_passthrough_windows(
    processed: Path,
    mixture: Path,
    routing: dict[str, Any],
    destination: Path,
) -> Path:
    """Replace unsafe windows with the untouched EN+RU speech, streaming.

    LOW/PASSTHROUGH windows are conservatively preserved instead of being sent
    through a model that the analysis does not consider reliable for them.
    """
    passthrough = [
        item for item in (routing.get("windows") or [])
        if item.get("recommended_model") == "passthrough"
        or item.get("band") == "PASSTHROUGH"
    ]
    if not passthrough:
        return processed
    windows = [
        (float(item.get("dubbed_start_sec") or 0.0), float(item.get("dubbed_end_sec") or 0.0))
        for item in passthrough
        if float(item.get("dubbed_end_sec") or 0.0) > float(item.get("dubbed_start_sec") or 0.0)
    ]
    if not windows:
        return processed
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial.flac")
    block_frames = 48000 * 10
    with sf.SoundFile(str(processed)) as processed_file, sf.SoundFile(str(mixture)) as mixture_file:
        rate = int(processed_file.samplerate)
        channels = int(processed_file.channels)
        block_frames = rate * 10
        fade = max(1, int(round(rate * 0.08)))
        with sf.SoundFile(
            str(temporary), mode="w", samplerate=rate, channels=channels,
            format="FLAC", subtype="PCM_16",
        ) as writer:
            position = 0
            while position < len(processed_file):
                processed_file.seek(position)
                mixture_file.seek(min(position, max(0, len(mixture_file) - 1)))
                clean = processed_file.read(block_frames, dtype="float32", always_2d=True)
                original = mixture_file.read(clean.shape[0], dtype="float32", always_2d=True)
                if original.shape[0] < clean.shape[0]:
                    original = np.pad(original, ((0, clean.shape[0] - original.shape[0]), (0, 0)))
                original = audio_io.match_channels(original, channels)
                mask = np.zeros((clean.shape[0], 1), dtype=np.float32)
                block_start = position / rate
                block_end = (position + clean.shape[0]) / rate
                for start_sec, end_sec in windows:
                    if end_sec <= block_start or start_sec >= block_end:
                        continue
                    start = max(0, int(round(start_sec * rate)) - position)
                    end = min(clean.shape[0], int(round(end_sec * rate)) - position)
                    if end <= start:
                        continue
                    local = np.ones(end - start, dtype=np.float32)
                    edge = min(fade, max(0, (end - start) // 2))
                    if edge:
                        ramp = np.sin(np.linspace(0.0, math.pi * 0.5, edge, endpoint=True)) ** 2
                        if start_sec >= block_start:
                            local[:edge] = ramp
                        if end_sec <= block_end:
                            local[-edge:] = ramp[::-1]
                    mask[start:end, 0] = np.maximum(mask[start:end, 0], local)
                writer.write(clean * (1.0 - mask) + original * mask)
                position += clean.shape[0]
    os.replace(temporary, destination)
    return destination


def _configured_semantic_checkpoint(
    store: Store,
    config_key: str,
    fallback_name: str,
) -> Path | None:
    """Resolve a bundled semantic checkpoint without depending on install path."""
    compatibility = store.cfg.get("reference_compatibility") or {}
    configured = compatibility.get(config_key)
    values = configured if isinstance(configured, list) else [configured]
    values.append(f"semantic_ru_separator/{fallback_name}")
    for value in values:
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = store.models_root / candidate
        candidate = candidate.resolve()
        if (
            candidate.is_file()
            and candidate.suffix.casefold() == ".pt"
            and candidate.stat().st_size > 1_000_000
        ):
            return candidate
    return None


def _route_semantic_model_windows(
    primary: Path | None,
    mixture: Path,
    routing: dict[str, Any],
    destination: Path,
) -> Path:
    """Blend DubClean Voice and passthrough output using the window map.

    Analysis windows overlap by design. Each model receives a smooth weight in
    its window and conflicting recommendations are cross-faded. Samples not
    covered by the map are passed through, which is the conservative fallback.
    """
    route_sources = {
        "dubclean_voice": primary,
        "passthrough": mixture,
    }
    windows = []
    for item in routing.get("windows") or []:
        model = str(item.get("recommended_model") or "passthrough")
        if model not in route_sources or route_sources[model] is None:
            model = "passthrough"
        start = float(item.get("dubbed_start_sec") or 0.0)
        end = float(item.get("dubbed_end_sec") or 0.0)
        if end > start:
            windows.append((start, end, model))
    if not windows:
        return _copy_audio(mixture, destination)

    available = {
        name: path for name, path in route_sources.items()
        if path is not None and path.is_file()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial.flac")
    readers = {
        name: _SequentialAudioReader(path) for name, path in available.items()
    }
    try:
        mixture_reader = readers["passthrough"]
        rate = int(mixture_reader.samplerate)
        channels = int(mixture_reader.channels)
        total_frames = len(mixture_reader)
        block_frames = rate * 10
        fade_frames = max(1, int(round(rate * 0.08)))
        with sf.SoundFile(
            str(temporary), mode="w", samplerate=rate, channels=channels,
            format="FLAC", subtype="PCM_16",
        ) as writer:
            position = 0
            while position < total_frames:
                frames = min(block_frames, total_frames - position)
                audio: dict[str, np.ndarray] = {}
                for name, reader in readers.items():
                    source_rate = int(reader.samplerate)
                    source_position = int(
                        round((position / max(rate, 1)) * source_rate)
                    )
                    source_frames = int(
                        round((frames / max(rate, 1)) * source_rate)
                    )
                    if source_position >= len(reader) or source_frames <= 0:
                        values = np.zeros(
                            (frames, int(reader.channels)), dtype=np.float32
                        )
                    else:
                        values = reader.read_at(source_position, source_frames)
                        if source_rate != rate:
                            values = audio_io.resample(
                                values, source_rate, rate
                            )
                    if values.shape[0] < frames:
                        values = np.pad(values, ((0, frames - values.shape[0]), (0, 0)))
                    elif values.shape[0] > frames:
                        values = values[:frames]
                    audio[name] = audio_io.match_channels(values, channels)

                weights = {
                    name: np.zeros((frames, 1), dtype=np.float32)
                    for name in route_sources
                }
                block_start = position / rate
                block_end = (position + frames) / rate
                for start_sec, end_sec, model in windows:
                    if end_sec <= block_start or start_sec >= block_end:
                        continue
                    start = max(0, int(round(start_sec * rate)) - position)
                    end = min(frames, int(round(end_sec * rate)) - position)
                    if end <= start:
                        continue
                    local = np.ones(end - start, dtype=np.float32)
                    edge = min(fade_frames, max(0, (end - start) // 2))
                    if edge:
                        ramp = np.sin(np.linspace(0.0, math.pi * 0.5, edge)) ** 2
                        local[:edge] = ramp
                        local[-edge:] = np.minimum(local[-edge:], ramp[::-1])
                    weights[model][start:end, 0] = np.maximum(
                        weights[model][start:end, 0], local
                    )

                total_weight = sum(weights.values())
                uncovered = total_weight[:, 0] <= 1e-6
                weights["passthrough"][uncovered, 0] = 1.0
                total_weight = np.maximum(sum(weights.values()), 1e-6)
                output = np.zeros((frames, channels), dtype=np.float32)
                for name, weight in weights.items():
                    source_name = name if name in audio else "passthrough"
                    output += audio[source_name] * (weight / total_weight)
                writer.write(np.clip(output, -1.0, 1.0))
                position += frames
    finally:
        for reader in readers.values():
            reader.close()
    os.replace(temporary, destination)
    return destination


def _db_from_rms(values: np.ndarray) -> np.ndarray:
    return (20.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-9))).astype(
        np.float32
    )


def _frame_rms_series(path: Path, frame_sec: float = 0.05) -> tuple[np.ndarray, float]:
    """Return mono RMS per fixed time frame without loading the whole film."""
    info = sf.info(str(path))
    rate = int(info.samplerate)
    frame_size = max(1, int(round(rate * frame_sec)))
    block_size = frame_size * 400
    carry = np.empty((0, int(info.channels)), dtype=np.float32)
    chunks: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as reader:
        while True:
            block = reader.read(block_size, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            if len(carry):
                block = np.concatenate([carry, block], axis=0)
            usable = (len(block) // frame_size) * frame_size
            if usable:
                mono = _mono(block[:usable]).reshape(-1, frame_size)
                chunks.append(
                    np.sqrt(np.mean(mono.astype(np.float64) ** 2, axis=1) + 1e-12).astype(
                        np.float32
                    )
                )
            carry = block[usable:]
    if not chunks:
        return np.zeros(0, dtype=np.float32), frame_sec
    return np.concatenate(chunks).astype(np.float32), frame_sec


def _program_rms_db(path: Path, frame_sec: float = 0.05) -> float | None:
    """Return the integrated programme RMS used for transparent level matching."""
    values, _ = _frame_rms_series(path, frame_sec)
    if len(values) == 0:
        return None
    power = float(np.mean(values.astype(np.float64) ** 2))
    if not math.isfinite(power) or power <= 1e-18:
        return None
    return float(10.0 * math.log10(power))


def _gated_level_db(
    path: Path,
    *,
    frame_sec: float = 0.05,
    gate_below_peak_db: float = 24.0,
) -> dict[str, Any]:
    """Return a robust active-frame level without scanning audio in memory.

    This is deliberately content-gated: long credits, pauses and different
    film durations must not make a translated programme look artificially
    quiet.  It is not labelled LUFS, because no BS.1770 K-weighting is applied.
    """
    values, actual_frame_sec = _frame_rms_series(path, frame_sec)
    if len(values) < 10:
        return {"ok": False, "reason": "too_few_frames", "active_seconds": 0.0}
    levels = _db_from_rms(values)
    audible = levels > -80.0
    if int(np.count_nonzero(audible)) < 10:
        return {"ok": False, "reason": "inaudible", "active_seconds": 0.0}
    peak_reference = float(np.percentile(levels[audible], 95))
    threshold = max(-60.0, peak_reference - float(gate_below_peak_db))
    active = levels >= threshold
    if int(np.count_nonzero(active)) < 10:
        return {"ok": False, "reason": "too_few_active_frames", "active_seconds": 0.0}
    active_power = float(np.mean(np.square(values[active].astype(np.float64))))
    return {
        "ok": True,
        "level_db": float(10.0 * math.log10(max(active_power, 1e-18))),
        "median_frame_db": float(np.median(levels[active])),
        "gate_db": float(threshold),
        "active_seconds": float(np.count_nonzero(active) * actual_frame_sec),
    }


def _active_signal_ratio_db(
    numerator_path: Path,
    speech_path: Path,
    *,
    frame_sec: float = 0.05,
) -> dict[str, Any]:
    """Measure ``numerator / speech`` loudness on frames where speech is active."""
    numerator_rms, numerator_frame_sec = _frame_rms_series(numerator_path, frame_sec)
    speech_rms, speech_frame_sec = _frame_rms_series(speech_path, frame_sec)
    frame_sec = float(max(numerator_frame_sec, speech_frame_sec))
    length = min(len(numerator_rms), len(speech_rms))
    if length == 0:
        return {
            "ok": False,
            "reason": "empty_audio",
            "active_seconds": 0.0,
        }
    numerator_rms = numerator_rms[:length]
    speech_rms = speech_rms[:length]
    speech_db = _db_from_rms(speech_rms)
    audible = speech_db > -80.0
    if int(np.count_nonzero(audible)) < 10:
        return {
            "ok": False,
            "reason": "no_speech_frames",
            "active_seconds": 0.0,
        }
    threshold_db = max(
        MIN_SPEECH_RMS_DB,
        float(np.percentile(speech_db[audible], 95)) - 24.0,
    )
    active = speech_db >= threshold_db
    if int(np.count_nonzero(active)) < 10:
        # Last-resort fallback for very sparse speech: measure the loudest 5%
        # of frames rather than pretending the ratio is known.
        threshold_db = float(np.percentile(speech_db[audible], 95))
        active = speech_db >= threshold_db
    active_count = int(np.count_nonzero(active))
    if active_count < 10:
        return {
            "ok": False,
            "reason": "too_few_active_frames",
            "active_seconds": float(active_count * frame_sec),
            "speech_threshold_db": threshold_db,
        }
    numerator_db = _db_from_rms(numerator_rms)
    ratio = numerator_db[active] - speech_db[active]
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) == 0:
        return {
            "ok": False,
            "reason": "invalid_ratio",
            "active_seconds": float(active_count * frame_sec),
            "speech_threshold_db": threshold_db,
        }
    return {
        "ok": True,
        "ratio_median_db": float(np.median(ratio)),
        "ratio_p25_db": float(np.percentile(ratio, 25)),
        "ratio_p75_db": float(np.percentile(ratio, 75)),
        "active_seconds": float(active_count * frame_sec),
        "speech_threshold_db": float(threshold_db),
    }


def _paired_background_level_match(
    target_program_path: Path,
    target_speech_path: Path,
    current_background_path: Path,
    *,
    frame_sec: float = 0.05,
) -> dict[str, Any]:
    """Match M&E to the source dub on the *same* time frames.

    Independent content gates are not comparable: the source programme gate
    tends to select loud dialogue while the restored-background gate tends to
    select music and effects.  Subtracting those two integrated powers can
    therefore make M&E several dB too quiet.  Prefer source frames with little
    extracted speech; on those frames the dubbed programme is the best
    available estimate of the intended background level.
    """
    programme_rms, programme_frame_sec = _frame_rms_series(
        target_program_path, frame_sec
    )
    speech_rms, speech_frame_sec = _frame_rms_series(target_speech_path, frame_sec)
    background_rms, background_frame_sec = _frame_rms_series(
        current_background_path, frame_sec
    )
    length = min(len(programme_rms), len(speech_rms), len(background_rms))
    actual_frame_sec = float(
        max(programme_frame_sec, speech_frame_sec, background_frame_sec)
    )
    if length < 10:
        return {"ok": False, "reason": "too_few_paired_frames", "active_seconds": 0.0}

    programme_rms = programme_rms[:length].astype(np.float64)
    speech_rms = speech_rms[:length].astype(np.float64)
    background_rms = background_rms[:length].astype(np.float64)
    programme_db = _db_from_rms(programme_rms)
    speech_db = _db_from_rms(speech_rms)
    background_db = _db_from_rms(background_rms)

    audible_programme = programme_db > -80.0
    audible_background = background_db > -80.0
    if int(np.count_nonzero(audible_programme)) < 10:
        return {"ok": False, "reason": "inaudible_programme", "active_seconds": 0.0}
    programme_gate = max(
        -60.0,
        float(np.percentile(programme_db[audible_programme], 95)) - 30.0,
    )
    programme_active = programme_db >= programme_gate

    audible_speech = speech_db > -80.0
    if int(np.count_nonzero(audible_speech)) >= 10:
        speech_gate = max(
            MIN_SPEECH_RMS_DB,
            float(np.percentile(speech_db[audible_speech], 95)) - 24.0,
        )
    else:
        speech_gate = MIN_SPEECH_RMS_DB
    speech_to_programme_db = speech_db - programme_db
    background_only = (
        programme_active
        & audible_background
        & ((speech_db < speech_gate) | (speech_to_programme_db <= -14.0))
    )

    minimum_frames = max(10, int(round(1.0 / max(actual_frame_sec, 1e-6))))
    method = "paired_low_speech_frames"
    target_background_rms = programme_rms
    if int(np.count_nonzero(background_only)) < minimum_frames:
        # A dialogue-heavy sample may have no clean M&E-only interval.  In
        # that case estimate the residual power per paired frame.  Unlike the
        # old global subtraction, both powers refer to exactly the same time.
        background_only = programme_active & audible_background
        target_background_rms = np.sqrt(
            np.maximum(
                np.square(programme_rms) - np.square(speech_rms),
                np.square(programme_rms) * 0.01,
            )
        )
        method = "paired_residual_power_fallback"

    count = int(np.count_nonzero(background_only))
    if count < 10:
        return {
            "ok": False,
            "reason": "too_few_background_frames",
            "active_seconds": float(count * actual_frame_sec),
        }

    target_power = float(
        np.mean(np.square(target_background_rms[background_only]))
    )
    current_power = float(np.mean(np.square(background_rms[background_only])))
    raw_gain_db = 10.0 * math.log10(
        max(target_power, 1e-18) / max(current_power, 1e-18)
    )
    return {
        "ok": True,
        "method": method,
        "raw_gain_db": float(raw_gain_db),
        "target_background_db": float(
            10.0 * math.log10(max(target_power, 1e-18))
        ),
        "current_background_db": float(
            10.0 * math.log10(max(current_power, 1e-18))
        ),
        "active_seconds": float(count * actual_frame_sec),
        "frame_count": count,
        "programme_gate_db": float(programme_gate),
        "speech_gate_db": float(speech_gate),
    }


def _paired_programme_prediction(
    target_program_path: Path,
    current_background_path: Path,
    current_voice_path: Path,
    *,
    background_gain_db: float,
    voice_gain_db: float,
    frame_sec: float = 0.05,
) -> dict[str, Any]:
    """Estimate final programme power on source-active paired frames."""
    programme_rms, programme_frame_sec = _frame_rms_series(
        target_program_path, frame_sec
    )
    background_rms, background_frame_sec = _frame_rms_series(
        current_background_path, frame_sec
    )
    voice_rms, voice_frame_sec = _frame_rms_series(current_voice_path, frame_sec)
    length = min(len(programme_rms), len(background_rms), len(voice_rms))
    actual_frame_sec = float(
        max(programme_frame_sec, background_frame_sec, voice_frame_sec)
    )
    if length < 10:
        return {"ok": False, "reason": "too_few_paired_frames"}
    programme_rms = programme_rms[:length].astype(np.float64)
    background_rms = background_rms[:length].astype(np.float64)
    voice_rms = voice_rms[:length].astype(np.float64)
    programme_db = _db_from_rms(programme_rms)
    audible = programme_db > -80.0
    if int(np.count_nonzero(audible)) < 10:
        return {"ok": False, "reason": "inaudible_programme"}
    gate_db = max(
        -60.0,
        float(np.percentile(programme_db[audible], 95)) - 30.0,
    )
    active = programme_db >= gate_db
    count = int(np.count_nonzero(active))
    if count < 10:
        return {"ok": False, "reason": "too_few_active_frames"}
    target_power = float(np.mean(np.square(programme_rms[active])))
    predicted_power = float(
        np.mean(
            np.square(
                background_rms[active] * _linear_gain(background_gain_db)
            )
            + np.square(voice_rms[active] * _linear_gain(voice_gain_db))
        )
    )
    target_db = float(10.0 * math.log10(max(target_power, 1e-18)))
    predicted_db = float(10.0 * math.log10(max(predicted_power, 1e-18)))
    return {
        "ok": True,
        "target_dubbed_active_db": target_db,
        "predicted_before_boost_db": predicted_db,
        "raw_boost_db": target_db - predicted_db,
        "active_seconds": float(count * actual_frame_sec),
        "frame_count": count,
        "programme_gate_db": float(gate_db),
    }


def _calculate_mix_balance(
    background_path: Path,
    original_speech_path: Path,
    russian_voice_path: Path,
    original_mix_path: Path | None = None,
    *,
    dubbed_speech_path: Path | None = None,
    dubbed_mix_path: Path | None = None,
    max_adjust_db: float = 12.0,
    max_program_adjust_db: float = 6.0,
) -> dict[str, Any]:
    """Preserve the source dub's dialogue/M&E balance on paired time frames."""
    target_speech_path = dubbed_speech_path or original_speech_path
    target_mix_path = dubbed_mix_path or original_mix_path
    target_voice = _gated_level_db(target_speech_path)
    current_voice = _gated_level_db(russian_voice_path)
    target_program = _gated_level_db(target_mix_path) if target_mix_path else {"ok": False}
    current_background = _gated_level_db(background_path)
    report: dict[str, Any] = {
        "schema_version": 3,
        "mode": "paired_source_dub_balance",
        "target_dubbed_speech_level": target_voice,
        "current_clean_russian_voice_level": current_voice,
        "target_dubbed_programme_level": target_program,
        "current_background_level": current_background,
        "max_adjust_db": float(max_adjust_db),
        "max_program_adjust_db": float(max_program_adjust_db),
        "background_gain_db": 0.0,
        "voice_gain_db": 0.0,
        "enabled": bool(target_voice.get("ok") and current_voice.get("ok")),
    }
    if not report["enabled"]:
        report["reason"] = "not_enough_reliable_speech_frames"
        return report
    raw_voice_gain = float(target_voice["level_db"]) - float(current_voice["level_db"])
    # A hot isolated stem may legitimately need a small reduction, but never
    # repeat the previous multi-dB dialogue attenuation.
    voice_gain = max(-1.5, min(float(max_adjust_db), raw_voice_gain))
    report["raw_voice_gain_db"] = raw_voice_gain
    report["voice_gain_db"] = voice_gain
    report["voice_gain_clamped"] = abs(voice_gain - raw_voice_gain) > 1e-6

    background_gain = 0.0
    boost_only = 0.0
    if target_program.get("ok") and current_background.get("ok"):
        background_match = _paired_background_level_match(
            target_mix_path,
            target_speech_path,
            background_path,
        )
        report["background_level_match"] = background_match
        if background_match.get("ok"):
            raw_background_gain = float(background_match["raw_gain_db"])
            background_gain = max(
                -float(max_adjust_db),
                min(float(max_adjust_db), raw_background_gain),
            )
        else:
            raw_background_gain = 0.0
        prediction = _paired_programme_prediction(
            target_mix_path,
            background_path,
            russian_voice_path,
            background_gain_db=background_gain,
            voice_gain_db=voice_gain,
        )
        if prediction.get("ok"):
            boost_only = max(
                0.0,
                min(float(max_program_adjust_db), float(prediction["raw_boost_db"])),
            )
        report["program_level_match"] = {
            "enabled": bool(prediction.get("ok")),
            **prediction,
            "raw_background_gain_db": float(raw_background_gain),
            "background_gain_db": float(background_gain),
            "boost_only_gain_db": float(boost_only),
            "speech_attenuation_forbidden": True,
        }
    else:
        report["background_level_match"] = {
            "ok": False,
            "reason": "programme_level_unavailable",
        }
        report["program_level_match"] = {"enabled": False, "reason": "programme_level_unavailable"}
    report["background_gain_db"] = background_gain + boost_only
    report["voice_gain_db"] = voice_gain + boost_only
    report["clamped"] = bool(
        report.get("voice_gain_clamped")
        or abs(background_gain - float((report.get("program_level_match") or {}).get("raw_background_gain_db", background_gain))) > 1e-6
    )
    return report


def _correlation_and_lag(
    left: np.ndarray,
    right: np.ndarray,
    sample_rate: int,
    max_lag_sec: float = 0.25,
) -> tuple[float, float]:
    target_rate = 2000
    left = signal.resample_poly(np.asarray(left, dtype=np.float32), target_rate, sample_rate)
    right = signal.resample_poly(np.asarray(right, dtype=np.float32), target_rate, sample_rate)
    length = min(len(left), len(right))
    if length < target_rate:
        # Callers unpack ``(correlation, lag)``; a scalar would crash them.
        return 0.0, 0.0
    left = left[:length] - float(np.mean(left[:length]))
    right = right[:length] - float(np.mean(right[:length]))
    left /= max(float(np.linalg.norm(left)), 1e-8)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    radius = int(round(max_lag_sec * target_rate))
    values = signal.correlate(right, left, mode="full", method="fft")
    center = length - 1
    window = np.abs(values[center - radius : center + radius + 1])
    peak = int(np.argmax(window))
    return float(window[peak]), float(peak - radius) / target_rate


def _max_correlation(left: np.ndarray, right: np.ndarray, sample_rate: int) -> float:
    return _correlation_and_lag(left, right, sample_rate)[0]


def _write(path: Path, data: np.ndarray, rate: int, channels: int | None = None) -> str:
    values = audio_io.ensure_2d(np.asarray(data, dtype=np.float32))
    if channels is not None:
        values = audio_io.match_channels(values, channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(values, -1.0, 1.0), rate, format="FLAC", subtype="PCM_16")
    return str(path.resolve())


def _matched_audio_blocks(
    background_reader: sf.SoundFile,
    voice_reader: sf.SoundFile,
    blocksize: int,
):
    yield from audio_mix.matched_audio_blocks(
        background_reader,
        voice_reader,
        blocksize,
    )


def _mix_audio_files(
    background: Path,
    voice: Path,
    destination: Path,
    *,
    background_gain_db: float = 0.0,
    voice_gain_db: float = 0.0,
    voice_delay_sec: float = 0.0,
    report_path: Path | None = None,
    ctx: Any = None,
    progress_substage: str = "",
    progress_base: float = 0.0,
    progress_span: float = 100.0,
) -> Path:
    """Stream M&E + voice without loading a feature film into RAM."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    background_info = sf.info(str(background))
    voice_info = sf.info(str(voice))
    if int(background_info.samplerate) != int(voice_info.samplerate):
        resampled_voice = destination.with_name(
            f".{destination.stem}.voice_{int(background_info.samplerate)}hz."
            f"{os.getpid()}.flac"
        )
        command = [
            audio_io.check_ffmpeg(),
            "-y",
            "-i",
            str(voice),
            "-vn",
            "-ar",
            str(int(background_info.samplerate)),
            "-c:a",
            "flac",
            "-compression_level",
            "5",
            str(resampled_voice),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Не удалось привести русскую речь к частоте фоновой дорожки: "
                + completed.stderr[-2000:]
        )
        try:
            return _mix_audio_files(
                background,
                resampled_voice,
                destination,
                background_gain_db=background_gain_db,
                voice_gain_db=voice_gain_db,
                voice_delay_sec=voice_delay_sec,
                report_path=report_path,
                ctx=ctx,
                progress_substage=progress_substage,
                progress_base=progress_base,
                progress_span=progress_span,
            )
        finally:
            resampled_voice.unlink(missing_ok=True)
    def publish_progress(local_progress: float) -> None:
        if ctx is None:
            return
        ctx.update(
            stage="Финальная сводка",
            substage=progress_substage or "Запись русской дорожки",
            progress=min(
                progress_base + progress_span,
                progress_base
                + local_progress / 100.0 * progress_span,
            ),
            local_progress=local_progress,
            current_file=destination.name,
        )

    mix_report = audio_mix.mix_audio_files(
        background,
        voice,
        destination,
        background_gain_db=background_gain_db,
        voice_gain_db=voice_gain_db,
        voice_delay_sec=voice_delay_sec,
        check_stop=(
            (lambda: pipeline._check_stop(ctx))
            if ctx is not None
            else None
        ),
        progress=publish_progress if ctx is not None else None,
    )
    if report_path is not None:
        atomic_json(report_path, mix_report)
    return destination


def _delay_voice_file(
    voice: Path,
    delay_sec: float,
    ctx: Any = None,
    *,
    force_recompute: bool = False,
) -> Path:
    """Return a copy of the voice track shifted by ``delay_sec`` (manual nudge).

    Positive delay pushes the Russian voice later (leading silence); negative
    pulls it earlier (trims the head and pads the tail so the length is kept).
    Returns the original path unchanged for a zero shift, so the default path
    is untouched.
    """
    shift = float(delay_sec or 0.0)
    if abs(shift) < 1e-4:
        return voice
    info = sf.info(str(voice))
    rate = int(info.samplerate)
    shift_samples = int(round(abs(shift) * rate))
    if shift_samples <= 0:
        return voice
    destination = voice.with_name(f"{voice.stem}.delay_{int(round(shift * 1000))}ms.flac")
    if not force_recompute and _cached_model_output_valid_for(destination, [voice]):
        return destination
    data, _ = sf.read(str(voice), dtype="float32", always_2d=True)
    channels = data.shape[1]
    if shift > 0:
        shifted = np.concatenate(
            [np.zeros((shift_samples, channels), dtype=np.float32), data], axis=0
        )
    else:
        trimmed = data[shift_samples:]
        shifted = np.concatenate(
            [trimmed, np.zeros((shift_samples, channels), dtype=np.float32)], axis=0
        )
    sf.write(str(destination), shifted, rate, format="FLAC", subtype="PCM_24")
    _record_model_output_for(destination, [voice])
    return destination


def _checkpoint_fingerprint(checkpoint: Path) -> dict[str, Any]:
    """Identify the exact checkpoint *file*, not just its path.

    Rolling checkpoint names (``*_best.pt``, ``*_last.pt``) are overwritten in
    place after every epoch, so a path comparison alone would happily reuse a
    model output produced by older weights."""
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _song_protection_preview_identity(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fingerprint the classifier and every threshold used by preview safety."""
    resolved = song_protection.resolved_config(config)
    model_path = root_path(str(resolved["classifier_model_path"]))
    return {
        # Invalidates preview artefacts created by the former full-track
        # preview scan.  Preview song analysis is now strictly scene-local.
        "analysis_scope": "selected_preview_scenes_only_v1",
        "schema_version": int(song_protection.SCHEMA_VERSION),
        "config": resolved,
        "classifier_model": (
            _checkpoint_fingerprint(model_path)
            if model_path.is_file()
            else {
                "path": str(model_path.resolve()),
                "missing": True,
            }
        ),
    }


def _coincident_speech_protection_identity(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fingerprint every DSP threshold that can change protected speech."""
    return {
        "schema_version": int(coincident_speech_protection.SCHEMA_VERSION),
        "algorithm": coincident_speech_protection.ALGORITHM,
        "config": coincident_speech_protection.resolved_config(config),
    }


def _speech_integrity_guard_identity(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fingerprint guard code and every threshold used by cached audio."""
    module_path = Path(str(speech_integrity_guard.__file__)).resolve()
    return {
        "schema_version": int(speech_integrity_guard.SCHEMA_VERSION),
        "algorithm": speech_integrity_guard.ALGORITHM,
        "config": speech_integrity_guard.resolved_config(config),
        "module_source": _checkpoint_fingerprint(module_path),
    }


def _load_integrity_report(path: Path, mode: str) -> dict[str, Any] | None:
    """Read only a complete report produced by the current guard contract."""
    try:
        report = read_json(path, None)
    except RuntimeError:
        return None
    if not isinstance(report, dict):
        return None
    if report.get("schema_version") != speech_integrity_guard.SCHEMA_VERSION:
        return None
    if report.get("report_kind") != speech_integrity_guard.REPORT_KIND:
        return None
    if report.get("algorithm") != speech_integrity_guard.ALGORITHM:
        return None
    if report.get("mode") != mode:
        return None
    intervals = report.get("confirmed_intervals")
    if intervals is not None and not isinstance(intervals, list):
        return None
    return report


def _guard_preservation_intervals(
    *reports: dict[str, Any] | None,
) -> list[dict[str, float]]:
    """Return a deterministic union of intervals restored by active guards."""
    values: list[tuple[float, float]] = []
    for report in reports:
        for item in (report or {}).get("confirmed_intervals") or []:
            try:
                start = max(0.0, float(item.get("start_sec", 0.0)))
                end = max(start, float(item.get("end_sec", start)))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end > start:
                values.append((start, end))
    merged: list[list[float]] = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        {"start_sec": round(start, 6), "end_sec": round(end, 6)}
        for start, end in merged
    ]


def _load_valid_coincident_report(
    path: Path,
    mixture: Path,
    reference: Path,
    processed: Path,
    config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Load a cached repair plan only when it is readable and still applicable."""
    try:
        report = read_json(path, None)
    except RuntimeError:
        return None
    try:
        valid, _ = coincident_speech_protection.validate_detection_report(
            report,
            mixture,
            reference,
            processed,
            config,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return report if valid and isinstance(report, dict) else None


def _cached_model_output_valid(output: Path, checkpoint: Path) -> bool:
    """A cached model output may be reused only when its sidecar proves it was
    produced by exactly this checkpoint file (same path, size and mtime)."""
    if not output.is_file() or output.stat().st_size == 0:
        return False
    sidecar = output.with_name(output.name + ".source.json")
    record = read_json(sidecar, {}) or {}
    return record.get("checkpoint") == _checkpoint_fingerprint(checkpoint)


def _record_model_output(output: Path, checkpoint: Path) -> None:
    sidecar = output.with_name(output.name + ".source.json")
    atomic_json(sidecar, {"checkpoint": _checkpoint_fingerprint(checkpoint)})


def _cached_model_output_valid_for(
    output: Path,
    checkpoints: list[Path],
    *,
    recipe: str = "",
) -> bool:
    if not output.is_file() or output.stat().st_size == 0:
        return False
    sidecar = output.with_name(output.name + ".source.json")
    record = read_json(sidecar, {}) or {}
    # An artifact whose build recipe is part of its identity (the final film,
    # whose muxing options decide whether players can open it) must not be
    # reused across a recipe change, even though its inputs did not move.
    if str(record.get("recipe") or "") != str(recipe or ""):
        return False
    expected = [_checkpoint_fingerprint(checkpoint) for checkpoint in checkpoints]
    if record.get("checkpoints") == expected:
        return True
    if len(checkpoints) == 1 and record.get("checkpoint") == expected[0]:
        return True
    return False


def _record_model_output_for(
    output: Path,
    checkpoints: list[Path],
    *,
    recipe: str = "",
) -> None:
    sidecar = output.with_name(output.name + ".source.json")
    payload: dict[str, Any] = {
        "checkpoints": [_checkpoint_fingerprint(checkpoint) for checkpoint in checkpoints]
    }
    if recipe:
        payload["recipe"] = str(recipe)
    atomic_json(sidecar, payload)


def _remux_recipe_signature(remux_cfg: dict[str, Any]) -> str:
    """Identify the muxing recipe a cached final film was produced with."""
    return "|".join(
        [
            audio_io.REMUX_RECIPE_VERSION,
            str(remux_cfg.get("remux_container") or "mkv"),
            str(remux_cfg.get("remux_audio_codec") or "aac"),
            str(remux_cfg.get("remux_audio_bitrate") or ""),
            str(remux_cfg.get("remux_audio_channels") or ""),
            str(remux_cfg.get("remux_audio_sample_rate") or ""),
        ]
    )


def _full_output_cache_reusable(
    output: Path,
    dependencies: list[Path],
    *,
    force_recompute: bool,
    recipe: str = "",
) -> bool:
    """Return whether one full-film artifact may be reused for this run.

    ``force`` on the full-film API is a production rebuild request, not merely
    an override of the preview feasibility guard.  Keeping that policy in one
    helper makes it difficult for a newly added full-film phase to
    accidentally ignore the request.
    """
    return (
        not force_recompute
        and _cached_model_output_valid_for(output, dependencies, recipe=recipe)
    )


def _atomic_json_if_changed(path: Path, payload: dict[str, Any]) -> None:
    """Keep a parameter file's mtime stable so it can safely key a cache."""
    if path.is_file() and read_json(path, None) == payload:
        return
    atomic_json(path, payload)


def _final_user_audio_name(pair: dict[str, Any]) -> str:
    """Return a readable final-track name derived from the dubbed source."""
    dubbed = dict((pair.get("sources") or {}).get("dubbed") or {})
    source_value = str(dubbed.get("path") or "").strip()
    source_stem = Path(source_value).stem.strip() if source_value else ""
    base = source_stem or str(pair.get("name") or "Итоговая дорожка").strip()
    # Keep ample room for the surrounding project path on Windows.
    safe_base = audio_io.safe_filename(base, fallback="Итоговая дорожка")[:120].rstrip(" .")
    return f"{safe_base} · DubClean.flac"


def _publish_user_audio_files(
    pair: dict[str, Any],
    full_root: Path,
    intermediates: dict[str, str] | None = None,
    *,
    audio: Path | None = None,
) -> dict[str, str]:
    """Publish readable user-facing links without renaming cache artifacts.

    NTFS hard links avoid duplicating multi-gigabyte FLAC files.  The copy
    fallback keeps the portable product usable on filesystems without hard
    link support.  A failed cosmetic publication never invalidates the actual
    processing result.
    """
    candidates: dict[str, tuple[Path, str]] = {}
    for key, value in (intermediates or {}).items():
        friendly_name = USER_AUDIO_ARTIFACT_NAMES.get(str(key))
        if friendly_name:
            candidates[str(key)] = (Path(str(value)).resolve(), friendly_name)
    if audio is not None:
        candidates["audio"] = (audio.resolve(), _final_user_audio_name(pair))
    if not candidates:
        return {}

    public_root = full_root / "Дорожки DubClean"
    published: dict[str, str] = {}
    for key, (source, friendly_name) in candidates.items():
        if not source.is_file() or source.stat().st_size <= 0:
            continue
        try:
            public_root.mkdir(parents=True, exist_ok=True)
            destination = public_root / audio_io.safe_filename(
                friendly_name,
                fallback=f"{key}.flac",
            )
            if destination.is_file():
                try:
                    if os.path.samefile(source, destination):
                        published[key] = str(destination.resolve())
                        continue
                except OSError:
                    pass
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.partial"
            )
            if temporary.exists():
                temporary.unlink()
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, destination)
            published[key] = str(destination.resolve())
        except OSError:
            continue
    return published


def _publish_full_application_progress(
    store: Store,
    pair: dict[str, Any],
    full_root: Path,
    method: str,
    *,
    intermediates: dict[str, str] | None = None,
    audio: Path | None = None,
    movie: Path | None = None,
    state: str = "running",
) -> None:
    """Persist completed full-length artifacts while the task is still running.

    A full film can take a long time, but every completed FLAC is already a
    useful, resumable result.  Publishing only non-empty, atomically completed
    files keeps stopped/failed runs inspectable without ever advertising
    ``.partial`` outputs.
    """
    previous = pair.get("application_full_progress")
    payload = (
        dict(previous)
        if isinstance(previous, dict)
        and previous.get("method") == method
        and Path(str(previous.get("root") or "")).resolve() == full_root.resolve()
        else {}
    )
    current_intermediates = dict(payload.get("intermediates") or {})
    for key, value in (intermediates or {}).items():
        path = Path(str(value)).resolve()
        if path.is_file() and path.stat().st_size > 0:
            current_intermediates[str(key)] = str(path)
    # Drop stale paths (for example after a user removed one artifact manually).
    current_intermediates = {
        key: value
        for key, value in current_intermediates.items()
        if Path(str(value)).is_file() and Path(str(value)).stat().st_size > 0
    }
    user_files = _publish_user_audio_files(
        pair,
        full_root,
        current_intermediates,
        audio=audio,
    )
    payload.update(
        {
            "schema_version": 1,
            "method": method,
            "root": str(full_root.resolve()),
            "state": state,
            "updated_at": utc_now(),
            "intermediates": current_intermediates,
            "user_files": user_files,
        }
    )
    if audio is not None and audio.is_file() and audio.stat().st_size > 0:
        payload["audio"] = str(audio.resolve())
    if movie is not None and movie.is_file() and movie.stat().st_size > 0:
        payload["movie"] = str(movie.resolve())
    pair["application_full_progress"] = payload
    store.save_pair(pair)


class _ProgressRangeContext:
    """Delegate a local operation while keeping task-wide progress separate."""

    def __init__(self, parent: Any, base: float, span: float):
        self.parent = parent
        self.base = float(base)
        self.span = float(span)
        self._high_water = float(base)

    @property
    def task(self):
        return self.parent.task

    @property
    def child_pid(self):
        return self.parent.child_pid

    @child_pid.setter
    def child_pid(self, value):
        self.parent.child_pid = value

    def check_stop(self):
        return self.parent.check_stop()

    def heartbeat(self):
        heartbeat = getattr(self.parent, "heartbeat", None)
        return heartbeat() if heartbeat is not None else None

    def log(self, message):
        return self.parent.log(message)

    def update(self, force: bool = False, **values):
        local = max(
            0.0,
            min(
                100.0,
                float(
                    values.get(
                        "local_progress",
                        values.get("progress", 0.0),
                    )
                    or 0.0
                ),
            ),
        )
        operation_progress = max(
            0.0,
            min(100.0, float(values.get("progress", local) or 0.0)),
        )
        self._high_water = max(
            self._high_water,
            self.base + operation_progress / 100.0 * self.span,
        )
        values["progress"] = self._high_water
        values["local_progress"] = local
        return self.parent.update(force=force, **values)


def _speech_extractor_dependencies(store: Store, model_role: str = "original") -> list[Path]:
    """Return the concrete files that define the selected speech extractor.

    The extractor uses a small text pointer named ``last_best_checkpoint``.
    Tracking both that pointer and the checkpoint it names keeps cached second
    pass outputs honest when the model is replaced in place.
    """
    model_cfg = store.cfg["speech_extraction"]
    checkpoint_key = (
        "dubbed_checkpoint_dir" if model_role == "dubbed" else "checkpoint_dir"
    )
    checkpoint_dir = root_path(
        model_cfg.get(checkpoint_key) or model_cfg["checkpoint_dir"]
    )
    pointer = checkpoint_dir / "last_best_checkpoint"
    if not pointer.is_file():
        raise RuntimeError("Модуль извлечения речи не найден.")
    checkpoint_name = pointer.read_text(encoding="utf-8").strip()
    checkpoint = (checkpoint_dir / checkpoint_name).resolve()
    if not checkpoint.is_file():
        raise RuntimeError("Сохранение модуля извлечения речи не найдено.")
    runner_value = str(model_cfg.get("runner") or "").strip()
    runner = (
        root_path(runner_value)
        if runner_value
        else (Path(__file__).resolve().parent / "speech_stem_runner.py")
    )
    if not runner.is_file():
        raise RuntimeError("Файл запуска модуля извлечения речи не найден.")
    # The runner and its block-processing contract materially affect every
    # stem.  Previously only the checkpoint was fingerprinted, so changing
    # chunk overlap/crossfades could silently reuse audio built by the old
    # hard-concatenation algorithm.
    runtime_root = Path(getattr(store, "runtime_root", checkpoint_dir)).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    contract_path = runtime_root / (
        f"speech_extraction_{model_role}_processing_contract.json"
    )
    _atomic_json_if_changed(
        contract_path,
        {
            "schema_version": 2,
            "algorithm": "mossformer_context_overlap_add_v2",
            "backend": str(model_cfg.get("backend") or ""),
            "model_name": str(model_cfg.get("model_name") or ""),
            "sample_rate": int(model_cfg.get("sample_rate", 48000)),
            "chunk_sec": float(model_cfg.get("chunk_sec", 20.0)),
            "context_sec": float(model_cfg.get("context_sec", 1.0)),
            "model_role": model_role,
        },
    )
    # Keep the actual model checkpoint last for compatibility with diagnostics
    # which expose it as the selected extractor weight.
    return [runner, contract_path, pointer, checkpoint]


def _full_speech_stem_dependencies(
    store: Store,
    source: Path,
    model_role: str,
) -> list[Path]:
    """Fingerprint every input that defines a full-film speech stem."""
    return [
        source.resolve(),
        *_speech_extractor_dependencies(store, model_role),
    ]


def _ensure_full_speech_stem(
    store: Store,
    source: Path,
    shared_cache: Path,
    destination: Path,
    ctx: Any,
    pair_name: str,
    stage: str,
    *,
    model_role: str,
    progress_base: float,
    progress_span: float,
    force_recompute: bool,
) -> Path:
    """Return a proven-current full speech stem or run the extractor.

    Legacy shared stems without a sidecar are deliberately not trusted.  A
    valid sidecar records the source audio plus the extractor pointer and the
    exact checkpoint it names, so replacing either model or input invalidates
    the cached stem automatically.
    """
    dependencies = _full_speech_stem_dependencies(store, source, model_role)
    for candidate in (shared_cache, destination):
        if _full_output_cache_reusable(
            candidate,
            dependencies,
            force_recompute=force_recompute,
        ):
            return candidate

    destination.parent.mkdir(parents=True, exist_ok=True)
    pipeline._run_speech_extractor(
        store,
        source,
        destination,
        ctx,
        pair_name,
        stage,
        progress_base=progress_base,
        progress_span=progress_span,
        model_role=model_role,
    )
    _record_model_output_for(destination, dependencies)
    return destination


def _resolve_neural_reference_adapter_checkpoint(
    parameters: dict[str, Any] | None = None,
) -> Path | None:
    value = str((parameters or {}).get("reference_adapter_checkpoint") or "").strip()
    checkpoint = Path(value).resolve() if value else DEFAULT_NEURAL_REFERENCE_ADAPTER
    if checkpoint is None:
        return None
    if "global_mastering" in checkpoint.name.casefold() or any(
        "global_mastering" in parent.name.casefold() for parent in checkpoint.parents
    ):
        return None
    return checkpoint if checkpoint.is_file() else None


def _resolve_global_mastering_checkpoint(
    parameters: dict[str, Any] | None = None,
) -> Path | None:
    value = str(
        (parameters or {}).get("global_mastering_checkpoint")
        or (parameters or {}).get("reference_adapter_checkpoint")
        or ""
    ).strip()
    checkpoint = Path(value).resolve() if value else DEFAULT_GLOBAL_MASTERING_ADAPTER
    if checkpoint is None:
        return None
    if not checkpoint.is_file():
        return None
    is_global = "global_mastering" in checkpoint.name.casefold() or any(
        "global_mastering" in parent.name.casefold() for parent in checkpoint.parents
    )
    return checkpoint if is_global else None


def _application_source_signature(pair: dict[str, Any]) -> dict[str, Any]:
    """Stable identity of the currently selected audio sources/streams.

    Preview candidate caches are valid only for the exact selected streams.  A
    different RU track in the same video file must rebuild candidate scan,
    speech stems and model outputs instead of reusing previous materials.
    """
    extraction = pair.get("extraction") or {}
    extracted_files = extraction.get("files") or {}
    signature: dict[str, Any] = {"roles": {}}
    for role in ("original", "dubbed"):
        source = (pair.get("sources") or {}).get(role) or {}
        extracted = extracted_files.get(role) or {}
        signature["roles"][role] = {
            "path": str(source.get("path") or extracted.get("source_path") or ""),
            "stream_index": int(source.get("stream_index") if source.get("stream_index") is not None else extracted.get("stream_index") or -1),
            "source_sha256": str(extracted.get("source_sha256") or ""),
            "source_size": int(extracted.get("source_size") or 0),
        }
    return signature


def _valid_audio_file(path: Path | str | None) -> bool:
    """Return True only for a complete, readable audio artifact."""
    if not path:
        return False
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            return False
        info = sf.info(str(candidate))
        return int(info.frames) > 0 and int(info.samplerate) > 0
    except (OSError, RuntimeError, ValueError):
        return False


def _preparation_identity(
    pair: dict[str, Any],
    checkpoint: Path,
    background_checkpoint: Path | None,
    applied_routing: dict[str, Any],
    speech_second_pass: bool = False,
    speech_second_pass_dependencies: list[Path] | None = None,
    song_protection_identity: dict[str, Any] | None = None,
    coincident_speech_protection_identity: dict[str, Any] | None = None,
    reference_level_alignment_identity: dict[str, Any] | None = None,
    speech_integrity_guard_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "scene_selection_version": SCENE_SELECTION_VERSION,
        "source_signature": _application_source_signature(pair),
        "checkpoint": _checkpoint_fingerprint(checkpoint),
        "background_checkpoint": (
            _checkpoint_fingerprint(background_checkpoint)
            if background_checkpoint is not None
            else None
        ),
        "routing": {
            "band": str(applied_routing.get("band") or ""),
            "recommended_model": str(
                applied_routing.get("recommended_model") or ""
            ),
            "recommended_alignment": str(
                applied_routing.get("recommended_alignment") or ""
            ),
        },
        "speech_second_pass": {
            "enabled": bool(speech_second_pass),
            "extractor": (
                [
                    _checkpoint_fingerprint(path)
                    for path in (speech_second_pass_dependencies or [])
                ]
                if speech_second_pass
                else []
            ),
        },
        "song_protection": song_protection_identity or {},
        "coincident_speech_protection": (
            coincident_speech_protection_identity or {}
        ),
        "reference_level_alignment": (
            reference_level_alignment_identity or {}
        ),
        "speech_integrity_guard": speech_integrity_guard_identity or {},
    }


def _compatible_preparation_phases(
    previous: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any]:
    """Keep only phases whose inputs/models still match the current run."""
    previous_identity = previous.get("identity") or {}
    phases = dict(previous.get("phases") or {})
    if previous_identity.get("scene_selection_version") != identity.get(
        "scene_selection_version"
    ):
        return {}
    if previous_identity.get("source_signature") != identity.get("source_signature"):
        return {}
    if previous_identity.get("song_protection") != identity.get(
        "song_protection"
    ):
        # The local song-analysis contract and thresholds affect the final
        # scene artefacts.  Do not reuse previews made under another contract.
        return {}
    if previous_identity.get(
        "coincident_speech_protection"
    ) != identity.get("coincident_speech_protection"):
        # Speech extraction itself is still current.  Every subtraction,
        # second-pass and playback artifact must be reconsidered.
        return {
            key: value
            for key, value in phases.items()
            if key == "speech"
        }
    if previous_identity.get("speech_integrity_guard") != identity.get(
        "speech_integrity_guard"
    ):
        # Extracted scene speech remains valid; every destructive stage and
        # all downstream mixes/synchronization must be reconsidered.
        return {
            key: value
            for key, value in phases.items()
            if key == "speech"
        }
    if (
        previous_identity.get("checkpoint") != identity.get("checkpoint")
        or previous_identity.get("routing") != identity.get("routing")
    ):
        return {key: value for key, value in phases.items() if key == "speech"}
    if previous_identity.get("background_checkpoint") != identity.get(
        "background_checkpoint"
    ):
        return {
            key: value
            for key, value in phases.items()
            if key in {"speech", "subtraction"}
        }
    if previous_identity.get("speech_second_pass") != identity.get(
        "speech_second_pass"
    ):
        return {
            key: value
            for key, value in phases.items()
            if key in {"speech", "subtraction", "background"}
        }
    return phases


def _begin_application_preparation(
    store: Store,
    project_id: str,
    pair_id: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    pair = store.load_pair(project_id, pair_id)
    previous = pair.get("application_preparation") or {}
    preparation = {
        "schema_version": 1,
        "identity": identity,
        "phases": _compatible_preparation_phases(previous, identity),
        "updated_at": utc_now(),
    }
    pair["application_preparation"] = preparation
    store.save_pair(pair)
    return preparation


def _save_application_preparation_phase(
    store: Store,
    project_id: str,
    pair_id: str,
    identity: dict[str, Any],
    phase: str,
    artifacts: dict[str, Path | str],
) -> dict[str, Any]:
    resolved = {key: str(Path(path).resolve()) for key, path in artifacts.items()}
    if not resolved or not all(_valid_audio_file(path) for path in resolved.values()):
        raise RuntimeError(
            f"Этап {phase} не может быть сохранён: один из аудиофайлов не готов."
        )
    pair = store.load_pair(project_id, pair_id)
    previous = pair.get("application_preparation") or {}
    phases = _compatible_preparation_phases(previous, identity)
    phases[phase] = {
        "status": "completed",
        "completed_at": utc_now(),
        "artifacts": resolved,
    }
    preparation = {
        "schema_version": 1,
        "identity": identity,
        "phases": phases,
        "updated_at": utc_now(),
    }
    pair["application_preparation"] = preparation
    store.save_pair(pair)
    return preparation


def _reuse_existing_preparation(
    store: Store, project_id: str, pair: dict[str, Any], ctx: Any
) -> bool:
    """Hard-link immutable extracted tracks and copy an existing alignment map."""
    wanted = pair.get("sources") or {}
    for project in store.list_projects():
        if project["id"] == project_id:
            continue
        for candidate in store.list_pairs(project["id"]):
            sources = candidate.get("sources") or {}
            if any(
                store._source_identity(sources.get(role))
                != store._source_identity(wanted.get(role))
                for role in ("original", "dubbed")
            ):
                continue
            candidate_root = store.pair_dir(project["id"], candidate["id"])
            files = [
                candidate_root / "extracted" / "original.flac",
                candidate_root / "extracted" / "dubbed.flac",
                candidate_root / "extracted" / "original_proxy.wav",
                candidate_root / "extracted" / "dubbed_proxy.wav",
                candidate_root / "extracted" / "metadata.json",
                candidate_root / "alignment" / "alignment_map.json",
            ]
            if not all(path.is_file() and path.stat().st_size > 0 for path in files):
                continue
            root = store.pair_dir(project_id, pair["id"])
            (root / "extracted").mkdir(parents=True, exist_ok=True)
            (root / "alignment").mkdir(parents=True, exist_ok=True)
            for stale in (root / "extracted").glob(".*.partial.*"):
                stale.unlink(missing_ok=True)
            missing_source = False
            for name in (
                "original.flac", "dubbed.flac", "original_proxy.wav",
                "dubbed_proxy.wav", "metadata.json",
            ):
                source = candidate_root / "extracted" / name
                destination = root / "extracted" / name
                if not source.is_file() or source.stat().st_size == 0:
                    missing_source = True
                    break
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.unlink(missing_ok=True)
                if source.suffix.casefold() == ".json":
                    shutil.copy2(source, destination)
                else:
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
            if missing_source:
                shutil.rmtree(root / "extracted", ignore_errors=True)
                shutil.rmtree(root / "alignment", ignore_errors=True)
                continue
            reusable_stems = {
                candidate_root / "stems" / "original" / "en_speech.flac":
                    root / "stems" / "original" / "en_speech.flac",
                candidate_root / "stems" / "dubbed" / "speech_en_ru.flac":
                    root / "stems" / "dubbed" / "speech_en_ru.flac",
            }
            for source, destination in reusable_stems.items():
                if not source.is_file() or source.stat().st_size == 0:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.unlink(missing_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            for source in (candidate_root / "alignment").glob("*.json"):
                shutil.copy2(source, root / "alignment" / source.name)
            extraction = read_json(root / "extracted" / "metadata.json", {}) or {}
            for role in ("original", "dubbed"):
                file_info = (extraction.get("files") or {}).get(role) or {}
                file_info["path"] = str((root / "extracted" / f"{role}.flac").resolve())
                file_info["proxy_path"] = str(
                    (root / "extracted" / f"{role}_proxy.wav").resolve()
                )
            atomic_json(root / "extracted" / "metadata.json", extraction)
            pair["extraction"] = extraction
            pair["alignment"] = candidate.get("alignment")
            pair["stages"]["2"] = {
                "status": "completed",
                "message": f"Переиспользованы извлечённые дорожки из «{candidate['name']}».",
            }
            pair["stages"]["3"] = {
                "status": "completed",
                "message": f"Переиспользована карта сопоставления из «{candidate['name']}».",
            }
            pair["application_reused_preparation"] = {
                "project_id": project["id"],
                "pair_id": candidate["id"],
                "pair_name": candidate["name"],
                "reused_at": utc_now(),
            }
            store.save_pair(pair)
            ctx.update(
                stage="Подготовленные дорожки найдены",
                substage=f"Используется проверенная подготовка «{candidate['name']}»",
                progress=10.0,
                local_progress=100.0,
                current_file=pair["name"],
            )
            return True
    return False


def _run_process(command: list[str], ctx: Any, stage: str, pair_name: str) -> list[str]:
    process = subprocess.Popen(
        command,
        cwd=str(pipeline.PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ctx.child_pid = process.pid
    lines: list[str] = []
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            pipeline._check_stop(ctx)
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
            lines = lines[-200:]
            if line.startswith("[status]"):
                ctx.update(
                    stage=stage,
                    substage=line.removeprefix("[status]").strip(),
                    current_file=pair_name,
                )
            elif line.startswith("[progress]"):
                progress_text = line.removeprefix("[progress]").strip()
                local_progress: float | None = None
                match = re.match(
                    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*"
                    r"([0-9]+(?:\.[0-9]+)?)\s*(?:sec|s|с)?\s*$",
                    progress_text,
                    flags=re.IGNORECASE,
                )
                if match:
                    processed = float(match.group(1))
                    total = float(match.group(2))
                    if total > 0:
                        local_progress = min(
                            100.0, max(0.0, processed / total * 100.0)
                        )
                ctx.update(
                    stage=stage,
                    substage=(
                        f"Обработано {match.group(1)} из {match.group(2)} с"
                        if match
                        else progress_text
                    ),
                    **(
                        {"local_progress": local_progress}
                        if local_progress is not None
                        else {}
                    ),
                    current_file=pair_name,
                )
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        ctx.child_pid = None
    if code:
        raise RuntimeError(f"{stage} завершилось с ошибкой.\n" + "\n".join(lines)[-6000:])
    return lines


def subtractor_runner_name(checkpoint: Path) -> str:
    """Program that can actually load this subtraction checkpoint.

    The production model is published under its public name
    ``models/semantic_ru_separator/dubclean_voice.pt``, so the choice must be
    made from the folder that defines the format, not from the research file
    name the weights used to carry.  Only legacy paired-subtraction
    checkpoints kept inside a project still use the older program.
    """
    name = checkpoint.name.casefold()
    if (
        checkpoint.parent.name.casefold() == "semantic_ru_separator"
        or name.startswith("semantic_ru_separator")
        or "semantic_" in name
        or name == "dubclean_voice.pt"
    ):
        return "semantic_separator_infer.py"
    return "reference_subtractor_infer.py"


def _is_semantic_subtractor(checkpoint: Path) -> bool:
    """Use the same checkpoint-format decision as the inference launcher."""
    return subtractor_runner_name(checkpoint) == "semantic_separator_infer.py"


def _run_subtractor(
    store: Store,
    mixture: Path,
    reference: Path,
    destination: Path,
    checkpoint: Path,
    ctx: Any,
    pair_name: str,
    stage: str,
    progress_base: float | None = None,
    progress_span: float = 0.0,
) -> None:
    python = root_path(store.cfg["training"]["python"])
    runner = Path(__file__).with_name(subtractor_runner_name(checkpoint))
    command = [
        str(python),
        str(runner),
        "--checkpoint",
        str(checkpoint),
        "--mixture",
        str(mixture),
        "--reference",
        str(reference),
        "--output",
        str(destination),
        "--min-cuda-free-gb",
        str(
            float(
                (store.cfg.get("compute") or {}).get(
                    "subtraction_min_cuda_free_gb", 1.0
                )
            )
        ),
        "--device",
        str((store.cfg.get("compute") or {}).get("device", "auto")),
    ]
    start_progress = {"progress": progress_base} if progress_base is not None else {}
    ctx.update(
        stage=stage,
        substage="Подготовка операции",
        local_progress=0.0,
        current_file=pair_name,
        **start_progress,
    )
    _run_process(command, ctx, stage, pair_name)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Модель не создала выходную русскую речевую дорожку.")
    end_progress = (
        {"progress": progress_base + progress_span}
        if progress_base is not None
        else {}
    )
    ctx.update(
        stage=stage,
        substage="Готово",
        local_progress=100.0,
        current_file=pair_name,
        **end_progress,
    )


def _run_background_restorer(
    store: Store,
    mixture: Path,
    reference: Path,
    destination: Path,
    checkpoint: Path,
    ctx: Any,
    pair_name: str,
    stage: str,
    progress_base: float | None = None,
    progress_span: float = 0.0,
) -> None:
    python = root_path(store.cfg["training"]["python"])
    runner = Path(__file__).with_name("background_restorer_infer.py")
    command = [
        str(python),
        str(runner),
        "--checkpoint",
        str(checkpoint),
        "--mixture",
        str(mixture),
        "--reference",
        str(reference),
        "--output",
        str(destination),
        "--min-cuda-free-gb",
        str(
            float(
                (store.cfg.get("compute") or {}).get(
                    "background_restoration_min_cuda_free_gb", 1.0
                )
            )
        ),
        "--device",
        str((store.cfg.get("compute") or {}).get("device", "auto")),
    ]
    start_progress = {"progress": progress_base} if progress_base is not None else {}
    ctx.update(
        stage=stage,
        substage="Подготовка операции",
        local_progress=0.0,
        current_file=pair_name,
        **start_progress,
    )
    _run_process(command, ctx, stage, pair_name)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Модель не создала дорожку музыки и эффектов.")
    end_progress = (
        {"progress": progress_base + progress_span}
        if progress_base is not None
        else {}
    )
    ctx.update(
        stage=stage,
        substage="Готово",
        local_progress=100.0,
        current_file=pair_name,
        **end_progress,
    )


def _run_music_separator(
    store: Store,
    source: Path,
    output_dir: Path,
    ctx: Any,
    pair_name: str,
) -> dict[str, Path]:
    config = store.cfg["music_separation"]
    model_dir = root_path(config["model_dir"])
    model_path = model_dir / str(config["model_name"])
    control_path = model_path.with_name(model_path.name + ".aria2")
    expected_size = int(config.get("model_size_bytes") or 0)
    if model_path.is_file() and expected_size and model_path.stat().st_size > expected_size:
        model_path.unlink()
    if (
        not model_path.is_file()
        or control_path.is_file()
        or (expected_size and model_path.stat().st_size != expected_size)
    ):
        model_dir.mkdir(parents=True, exist_ok=True)
        download_pid_file = store.runtime_root / "separator_model_download.pid"
        try:
            download_pid = int(download_pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            download_pid = 0
        while download_pid and (
            not model_path.is_file()
            or control_path.is_file()
            or model_path.stat().st_size < expected_size
        ):
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, download_pid)
            if not handle:
                break
            ctypes.windll.kernel32.CloseHandle(handle)
            pipeline._check_stop(ctx)
            actual = model_path.stat().st_size
            ctx.update(
                stage="Установка модели музыки и эффектов",
                substage=(
                    f"Загружено {actual / 1024 / 1024:.0f} из "
                    f"{expected_size / 1024 / 1024:.0f} МБ"
                ),
                current_file=pair_name,
            )
            time.sleep(2.0)
        ctx.update(
            stage="Установка модели музыки и эффектов",
            substage=(
                f"Загружено {model_path.stat().st_size / 1024 / 1024:.0f} из "
                f"{expected_size / 1024 / 1024:.0f} МБ"
                if model_path.is_file() and expected_size
                else "Начинается загрузка модели"
            ),
            current_file=pair_name,
        )
        _run_process(
            [
                "aria2c.exe",
                "--continue=true",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=4M",
                "--retry-wait=2",
                "--max-tries=0",
                "--summary-interval=5",
                "--dir",
                str(model_dir),
                "--out",
                model_path.name,
                str(config["download_url"]),
            ],
            ctx,
            "Установка модели музыки и эффектов",
            pair_name,
        )
    if (
        not model_path.is_file()
        or control_path.is_file()
        or (expected_size and model_path.stat().st_size != expected_size)
    ):
        actual = model_path.stat().st_size if model_path.is_file() else 0
        raise RuntimeError(
            f"Модель сопровождения загружена не полностью: {actual} из {expected_size} байт."
        )
    command = [
        str(Path(str(config["python"])).resolve()),
        str(Path(str(config["runner"])).resolve()),
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--model",
        str(config["model_name"]),
        "--model-dir",
        str(model_dir),
        "--chunk-sec",
        str(int(config.get("chunk_sec", 120))),
    ]
    lines = _run_process(
        command,
        ctx,
        "Специализированное разделение голоса и сопровождения",
        pair_name,
    )
    payload = next(
        (
            json.loads(line.removeprefix("[result]").strip())
            for line in reversed(lines)
            if line.startswith("[result]")
        ),
        None,
    )
    if not payload:
        raise RuntimeError("Модель разделения не сообщила пути к результатам.")
    result: dict[str, Path] = {}
    for value in payload.get("outputs") or []:
        path = Path(value)
        name = path.name.casefold()
        if "instrumental" in name or "no_vocals" in name:
            result["instrumental"] = path
        elif "vocals" in name or "vocal" in name:
            result["vocals"] = path
    if "instrumental" not in result:
        raise RuntimeError("Модель разделения не создала дорожку музыки и эффектов.")
    return result


def _alignment_segment_speed_is_safe(
    alignment: dict[str, Any], segment: dict[str, Any]
) -> bool:
    """Reject stale map intervals that would audibly stretch preview audio."""
    raw_limit = (alignment.get("summary") or {}).get(
        "processing_max_speed_deviation", 0.06
    )
    raw_speed = segment.get("speed_ratio")
    try:
        limit = max(0.0, float(raw_limit))
        speed = float(1.0 if raw_speed is None else raw_speed)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(limit)
        and math.isfinite(speed)
        and speed > 0.0
        and abs(speed - 1.0) <= limit
    )


def _alignment_segment_for_range(
    alignment: dict[str, Any], start_sec: float, duration_sec: float
) -> dict[str, Any] | None:
    end_sec = start_sec + duration_sec
    segments = [
        item for item in alignment.get("segments", [])
        if float(item.get("dubbed_end") or 0.0) > float(item.get("dubbed_start") or 0.0)
    ]
    exact = [
        item for item in segments
        if item.get("usable", True) is not False
        if _alignment_segment_speed_is_safe(alignment, item)
        if float(item["dubbed_start"]) <= start_sec + 1e-3
        and float(item["dubbed_end"]) >= end_sec - 1e-3
    ]
    if exact:
        return max(exact, key=lambda item: float(item.get("confidence") or 0.0))
    return None


def _candidate_from_selection(
    alignment: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any] | None:
    start = float(selection.get("preview_start_sec") or 0.0)
    duration = float(selection.get("preview_duration_sec") or 0.0)
    segment = _alignment_segment_for_range(alignment, start, duration)
    if segment is None or duration <= 0.0:
        return None
    speed = float(segment.get("speed_ratio") or 1.0)
    correction = float(alignment.get("manual_correction_sec") or 0.0)
    original_start = float(segment["original_start"]) + (
        start - float(segment["dubbed_start"])
    ) * speed + correction
    return {
        **selection,
        "dubbed_start_sec": round(start, 3),
        # A mapping just before the head of the original file stays negative:
        # ``pipeline._read_interval`` zero-pads negative starts, whereas
        # clamping to 0 would silently shift the reference by that amount.
        "original_start_sec": round(original_start, 3),
        "duration_sec": round(duration, 3),
        "speed_ratio": speed,
        "alignment_confidence": float(segment.get("confidence") or 0.0),
        "alignment_segment_id": segment.get("id"),
    }


# Human-readable justification for each positional-fallback trigger. The reason
# code itself is machine-checkable and stored in the manifest.
_FALLBACK_REASON_TEXT = {
    "no_speech_found": (
        "Silero VAD не подтвердил речь; использован безопасный "
        "позиционный выбор."
    ),
    "vad_error": (
        "Silero VAD завершился с ошибкой в этой зоне; использован "
        "позиционный резервный выбор."
    ),
    "vad_window_unmapped": (
        "Silero VAD нашёл речь, но выбранный участок не покрывается одним сегментом "
        "карты сопоставления; использован резервный выбор в той же зоне."
    ),
    "zone_unavailable": (
        "Зона недоступна для VAD; использован позиционный резервный выбор."
    ),
}


def _legacy_zone_fallback(
    alignment: dict[str, Any],
    zone: dict[str, Any],
    reason: str = "no_speech_found",
) -> dict[str, Any] | None:
    """Use the former positional picker only after VAD failed for this zone.

    ``reason`` records *why* VAD did not yield a usable selection so the manifest
    never presents a fallback as a confident VAD choice.
    """
    left = float(zone["analysis_start_sec"])
    right = float(zone["analysis_end_sec"])
    preferred = left if zone["zone"] == "start" else (left + right) / 2.0
    options: list[tuple[float, dict[str, Any], float, float]] = []
    for segment in alignment.get("segments", []):
        if segment.get("usable", True) is False:
            continue
        if not _alignment_segment_speed_is_safe(alignment, segment):
            continue
        seg_left = max(left, float(segment.get("dubbed_start") or 0.0))
        seg_right = min(right, float(segment.get("dubbed_end") or 0.0))
        span = seg_right - seg_left
        if span <= 0.0:
            continue
        duration = min(vad_preview.PREFERRED_PREVIEW_SEC, span)
        if span >= vad_preview.MIN_PREVIEW_SEC:
            duration = max(vad_preview.MIN_PREVIEW_SEC, duration)
        start = min(max(seg_left, preferred - duration / 2.0), seg_right - duration)
        options.append((span, segment, start, duration))
    if not options:
        return None
    _, segment, start, duration = max(options, key=lambda item: (item[0], float(item[1].get("confidence") or 0.0)))
    fallback = dict(zone)
    fallback.update({
        "preview_start_sec": round(start, 3),
        "preview_end_sec": round(start + duration, 3),
        "preview_duration_sec": round(duration, 3),
        "speech_duration_sec": 0.0,
        "speech_ratio": 0.0,
        "speech_region_start_sec": None,
        "speech_region_end_sec": None,
        "status": "fallback_used",
        "selection_reason": _FALLBACK_REASON_TEXT.get(reason, _FALLBACK_REASON_TEXT["no_speech_found"]),
        "fallback": {"used": True, "reason": reason},
    })
    return _candidate_from_selection(alignment, fallback)


def _emergency_zone_fallback(zone: dict[str, Any]) -> dict[str, Any]:
    """Last-resort preview when the alignment map has no usable interval.

    This intentionally remains inside the requested VAD zone.  It keeps the
    job and all five UI cards alive, while its explicit status prevents an
    uncertain original/dubbed correspondence from being presented as VAD
    speech selection.
    """
    left = float(zone.get("analysis_start_sec") or 0.0)
    right = max(left, float(zone.get("analysis_end_sec") or left))
    available = right - left
    duration = min(vad_preview.PREFERRED_PREVIEW_SEC, available)
    if available >= vad_preview.MIN_PREVIEW_SEC:
        duration = max(vad_preview.MIN_PREVIEW_SEC, duration)
    anchor = left if zone.get("zone") == "start" else (left + right) / 2.0
    start = min(max(left, anchor - duration / 2.0), max(left, right - duration))
    return {
        **zone,
        "preview_start_sec": round(start, 3),
        "preview_end_sec": round(start + duration, 3),
        "preview_duration_sec": round(duration, 3),
        "dubbed_start_sec": round(start, 3),
        # This is only a best-effort timeline coordinate; downstream metrics
        # retain alignment_confidence=0 and subtraction_feasible=False.
        "original_start_sec": round(start, 3),
        "duration_sec": round(duration, 3),
        "speed_ratio": 1.0,
        "alignment_confidence": 0.0,
        "alignment_segment_id": None,
        "speech_duration_sec": float(zone.get("speech_duration_sec") or 0.0),
        "speech_ratio": float(zone.get("speech_ratio") or 0.0),
        "speech_region_start_sec": zone.get("speech_region_start_sec"),
        "speech_region_end_sec": zone.get("speech_region_end_sec"),
        "status": "fallback_used",
        "selection_reason": (
            "Нет пригодного участка карты сопоставления; создано превью "
            "внутри исходной зоны с пометкой резервного выбора."
        ),
        "fallback": {"used": True, "reason": "alignment_unavailable"},
    }


def _vad_candidates(
    dubbed_path: Path, duration: float, alignment: dict[str, Any], ctx: Any | None = None,
    pair_name: str = "",
) -> list[dict[str, Any]]:
    allowed = [
        (float(item["dubbed_start"]), float(item["dubbed_end"]))
        for item in alignment.get("segments", [])
        if item.get("usable", True) is not False
        if _alignment_segment_speed_is_safe(alignment, item)
        if float(item.get("dubbed_end") or 0.0) - float(item.get("dubbed_start") or 0.0) >= 1.0
    ]
    candidates: list[dict[str, Any]] = []
    for index, zone in enumerate(vad_preview.build_analysis_zones(duration), 1):
        if ctx is not None:
            ctx.update(
                stage="Silero VAD: поиск речевых зон",
                substage=f"Анализ зоны {zone['zone']} ({index}/5)",
                progress=10.0 + index * 2.0,
                local_progress=index / 5.0 * 100.0,
                current_file=pair_name,
            )
        try:
            selection = vad_preview.analyse_zone(dubbed_path, zone, allowed)
        except Exception as error:  # one bad VAD zone must not stop the film
            selection = {
                **zone,
                "status": "no_speech_found",
                "selection_reason": f"Silero VAD недоступен для зоны: {error}",
                "fallback": {"used": False, "reason": "vad_error"},
                "speech_intervals": [],
                "speech_interval_count": 0,
                "average_speech_confidence": None,
                "max_speech_confidence": None,
                "vad_parameters": dict(vad_preview.VAD_PARAMETERS),
                "silero_vad_version": vad_preview.SILERO_VAD_VERSION,
            }
        candidate = _candidate_from_selection(alignment, selection)
        if candidate is None:
            # Preserve *why* VAD did not map so the manifest reason is accurate:
            # a genuine no-speech zone, a VAD error, or speech that VAD found but
            # whose window is not covered by a single alignment segment.
            status = selection.get("status")
            if (selection.get("fallback") or {}).get("reason") == "vad_error":
                fallback_reason = "vad_error"
            elif status in ("speech_selected", "speech_selected_low_confidence"):
                fallback_reason = "vad_window_unmapped"
            elif status == "zone_unavailable":
                fallback_reason = "zone_unavailable"
            else:
                fallback_reason = "no_speech_found"
            candidate = _legacy_zone_fallback(alignment, selection, reason=fallback_reason)
        if candidate is None:
            # Keep the zone in metadata even when its timeline cannot be mapped.
            selection["status"] = "zone_unavailable"
            selection["selection_reason"] = "Нет пригодного участка карты сопоставления для этой зоны."
            candidates.append(selection)
        else:
            candidates.append(candidate)
    return candidates


def _preselect_candidates(
    root: Path,
    candidates: list[dict[str, Any]],
    rate: int,
    per_zone: int = 5,
    ctx: Any | None = None,
    pair_name: str = "",
    semantic_mode: bool = False,
) -> list[dict[str, Any]]:
    """Cheaply scan many aligned full-mix windows before neural speech extraction."""
    original_speech = root / "stems" / "original" / "en_speech.flac"
    dubbed_speech = root / "stems" / "dubbed" / "speech_en_ru.flac"
    use_speech_stems = bool(
        semantic_mode and original_speech.is_file() and dubbed_speech.is_file()
    )
    original_path = (
        original_speech if use_speech_stems else root / "extracted" / "original.flac"
    )
    dubbed_path = (
        dubbed_speech if use_speech_stems else root / "extracted" / "dubbed.flac"
    )
    zone_order = list(dict.fromkeys(str(item["zone"]) for item in candidates))
    ranked: dict[str, list[tuple[float, dict[str, Any]]]] = {zone: [] for zone in zone_order}
    for index, item in enumerate(candidates):
        if ctx is not None:
            pipeline._check_stop(ctx)
        duration = float(item["duration_sec"])
        target_len = int(round(duration * rate))
        original, original_rate = pipeline._read_interval(
            original_path,
            float(item["original_start_sec"]),
            duration * float(item["speed_ratio"]),
        )
        dubbed, dubbed_rate = pipeline._read_interval(
            dubbed_path, float(item["dubbed_start_sec"]), duration
        )
        original = pipeline._resample_exact(
            _mono(original)[:, None], original_rate, rate, target_len
        )[:, 0]
        dubbed = pipeline._resample_exact(
            _mono(dubbed)[:, None], dubbed_rate, rate, target_len
        )[:, 0]
        correlation, lag = _correlation_and_lag(
            original, dubbed, rate, max_lag_sec=1.5
        )
        original_db, dubbed_db = _rms_db(original), _rms_db(dubbed)
        value = {
            **item,
            "raw_mix_correlation": round(correlation, 4),
            "raw_mix_delay_sec": round(lag, 4),
            "original_mix_rms_db": round(original_db, 2),
            "dubbed_mix_rms_db": round(dubbed_db, 2),
        }
        score = (
            0.5 * original_db
            + 0.5 * dubbed_db
            + float(item["alignment_confidence"])
            if use_speech_stems
            else correlation * 100.0
            + 0.1 * max(-60.0, original_db)
            + 0.05 * max(-60.0, dubbed_db)
            + float(item["alignment_confidence"])
        )
        ranked[str(item["zone"])].append((score, value))
        if ctx is not None:
            ctx.update(
                stage="Поиск сцен с той же английской репликой",
                substage=f"Проверен участок {index + 1} из {len(candidates)}",
                progress=10.0 + (index + 1) / max(len(candidates), 1) * 10.0,
                local_progress=(index + 1) / max(len(candidates), 1) * 100.0,
                current_file=pair_name,
            )
    selected: list[dict[str, Any]] = []
    for zone in zone_order:
        choices = sorted(ranked[zone], key=lambda row: row[0], reverse=True)
        valid = [
            row for row in choices
            if row[1]["original_mix_rms_db"] >= -58.0
            and row[1]["dubbed_mix_rms_db"] >= -55.0
        ]
        selected.extend(row[1] for row in (valid or choices)[:per_zone])
    return selected


def _song_preview_candidates(
    scan_report: dict[str, Any],
    duration_sec: float,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Turn full-track classifier intervals into bounded preview windows."""
    cfg = song_protection.resolved_config(config)
    preferred = float(cfg["preview_scene_sec"])
    minimum = float(cfg["preview_min_sec"])
    maximum = float(cfg["preview_max_sec"])
    scan_step = float(cfg["step_sec"])
    film_duration = max(0.0, float(duration_sec))
    result: list[dict[str, Any]] = []
    sequence = 0

    def grid_floor(value: float) -> float:
        return max(
            0.0,
            math.floor((max(0.0, value) + 1e-9) / scan_step) * scan_step,
        )

    def grid_ceil(value: float) -> float:
        return max(
            0.0,
            math.ceil((max(0.0, value) - 1e-9) / scan_step) * scan_step,
        )

    def bounded_grid_start(value: float, scene_duration: float) -> float:
        # The classifier evaluates a fixed global grid.  Keeping preview starts
        # on that same grid makes the card represent the exact analysed region.
        latest = max(0.0, film_duration - scene_duration)
        latest_grid = grid_floor(latest)
        return min(grid_floor(value), latest_grid)

    for candidate_index, interval in enumerate(
        scan_report.get("candidate_intervals") or [], 1
    ):
        song_start = max(0.0, float(interval.get("start_sec") or 0.0))
        song_end = min(
            film_duration,
            max(song_start, float(interval.get("end_sec") or song_start)),
        )
        if song_end <= song_start:
            continue
        song_duration = song_end - song_start
        # One 30–40 second scene is preferable.  Sustained longer material is
        # tiled with a small overlap so every portion remains listenable.
        if song_duration <= maximum:
            scene_duration = min(
                film_duration,
                max(minimum, min(maximum, max(preferred, song_duration))),
            )
            grid_start = bounded_grid_start(song_start, scene_duration)
            # A candidate can begin between classifier steps when a report was
            # imported from an older build.  Expand, where possible, without
            # exceeding the documented 40-second preview bound.
            scene_duration = min(
                maximum,
                max(scene_duration, song_end - grid_start),
            )
            starts = [bounded_grid_start(grid_start, scene_duration)]
        else:
            scene_duration = min(maximum, max(minimum, preferred))
            stride = max(minimum, scene_duration - scan_step)
            stride = max(
                scan_step,
                round(stride / scan_step) * scan_step,
            )
            starts = []
            cursor = bounded_grid_start(song_start, scene_duration)
            while cursor + scene_duration < song_end - 1e-6:
                if not starts or abs(cursor - starts[-1]) > 1e-6:
                    starts.append(cursor)
                cursor += stride
            tail = bounded_grid_start(
                grid_ceil(max(song_start, song_end - scene_duration)),
                scene_duration,
            )
            if not starts or abs(tail - starts[-1]) > 1e-6:
                starts.append(tail)
            if not starts:
                starts.append(bounded_grid_start(song_start, scene_duration))
            starts.sort()
            # Remove a rare redundant tile introduced by clamping at the end
            # of a file while preserving the global scan grid.
            unique_starts: list[float] = []
            for start in starts:
                if not unique_starts or abs(start - unique_starts[-1]) > 1e-6:
                    unique_starts.append(start)
                if start + scene_duration >= song_end - 1e-6:
                    break
            starts = unique_starts
        for tile_index, start in enumerate(starts, 1):
            sequence += 1
            end = min(film_duration, start + scene_duration)
            result.append(
                {
                    "zone": f"song_{sequence:02d}",
                    "preview_kind": "song_candidate",
                    "preview_start_sec": round(start, 3),
                    "preview_end_sec": round(end, 3),
                    "preview_duration_sec": round(end - start, 3),
                    "dubbed_start_sec": round(start, 3),
                    # The preview batch reads this item from the already
                    # aligned English full mix, so both timelines are equal.
                    "original_start_sec": round(start, 3),
                    "duration_sec": round(end - start, 3),
                    "speed_ratio": 1.0,
                    "alignment_confidence": 1.0,
                    "alignment_segment_id": "full_aligned_original",
                    "speech_duration_sec": 0.0,
                    "speech_ratio": 0.0,
                    "speech_region_start_sec": None,
                    "speech_region_end_sec": None,
                    "status": "song_candidate_selected",
                    "selection_reason": (
                        "EfficientAT нашёл продолжительный песенный участок; "
                        "окончательное решение выполняется по всем защитным "
                        "проверкам внутри превью."
                    ),
                    "fallback": {"used": False, "reason": None},
                    "song_candidate": {
                        "candidate_index": candidate_index,
                        "tile_index": tile_index,
                        "start_sec": round(song_start, 3),
                        "end_sec": round(song_end, 3),
                        "confidence": interval.get("confidence"),
                        "features": interval.get("features") or {},
                        "reasons": interval.get("reasons") or [],
                    },
                }
            )
    return result


def _deduplicate_selected_song_scenes(
    selected: list[dict[str, Any]],
    scan_step_sec: float = 5.0,
) -> list[dict[str, Any]]:
    """Avoid only exact grid-aligned coverage of a song preview.

    A partially overlapping speech card is not equivalent to the classifier
    window: it can hide the beginning of a song and used to make a real
    candidate disappear from the preview.
    """
    speech = [
        item for item in selected
        if item.get("preview_kind") != "song_candidate"
    ]
    songs = [
        item for item in selected
        if item.get("preview_kind") == "song_candidate"
    ]
    accepted: list[dict[str, Any]] = list(speech)
    for song in songs:
        song_start = float(song.get("dubbed_start_sec") or 0.0)
        song_end = song_start + float(song.get("duration_sec") or 0.0)
        duplicate: dict[str, Any] | None = None
        for existing in accepted:
            existing_start = float(existing.get("dubbed_start_sec") or 0.0)
            existing_end = existing_start + float(
                existing.get("duration_sec") or 0.0
            )
            step = max(1e-6, float(scan_step_sec))
            starts_on_scan_grid = abs(
                existing_start / step - round(existing_start / step)
            ) <= 1e-4
            covers_whole_candidate = (
                existing_start <= song_start + 1e-4
                and existing_end >= song_end - 1e-4
            )
            if starts_on_scan_grid and covers_whole_candidate:
                duplicate = existing
                break
        if duplicate is None:
            # A song reference must never receive the local speech-only lag.
            song["local_speech_delay_sec"] = 0.0
            accepted.append(song)
            continue
        candidate = dict(song.get("song_candidate") or {})
        duplicate.setdefault("song_candidates", []).append(candidate)
        if duplicate.get("preview_kind") != "song_candidate":
            duplicate["preview_kind"] = "speech_and_song_candidate"
    return accepted


def _make_candidate_batches(
    root: Path,
    candidates: list[dict[str, Any]],
    rate: int,
    *,
    aligned_original_path: Path | None = None,
) -> tuple[Path, Path, list[tuple[int, int]]]:
    original_path = root / "extracted" / "original.flac"
    dubbed_path = root / "extracted" / "dubbed.flac"
    gap = np.zeros(rate, dtype=np.float32)
    original_parts: list[np.ndarray] = []
    dubbed_parts: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for item in candidates:
        target_len = int(round(float(item["duration_sec"]) * rate))
        if (
            item.get("preview_kind") == "song_candidate"
            and aligned_original_path is not None
        ):
            original, original_rate = pipeline._read_interval(
                aligned_original_path,
                float(item["dubbed_start_sec"]),
                float(item["duration_sec"]),
            )
        else:
            original, original_rate = pipeline._read_interval(
                original_path,
                float(item["original_start_sec"]),
                float(item["duration_sec"]) * float(item["speed_ratio"]),
            )
        dubbed, dubbed_rate = pipeline._read_interval(
            dubbed_path,
            float(item["dubbed_start_sec"]),
            float(item["duration_sec"]),
        )
        original = pipeline._resample_exact(
            _mono(original)[:, None], original_rate, rate, target_len
        )[:, 0]
        dubbed = pipeline._resample_exact(_mono(dubbed)[:, None], dubbed_rate, rate, target_len)[:, 0]
        original_parts.extend([original, gap])
        dubbed_parts.extend([dubbed, gap])
        ranges.append((cursor, cursor + target_len))
        cursor += target_len + len(gap)
    preview_root = root / "application" / "candidate_scan"
    return (
        Path(_write(preview_root / "original_candidates.flac", np.concatenate(original_parts), rate)),
        Path(_write(preview_root / "dubbed_candidates.flac", np.concatenate(dubbed_parts), rate)),
        ranges,
    )


SCENE_MEDIA_DELAY_GUARD: dict[str, float] = {
    "max_delay_sec": 0.25,
    "min_delay_sec": 0.008,
    # The floor only keeps hopeless windows out of the verification; what
    # decides is whether moving the bed measurably improves the agreement,
    # and a real reel offset clears that test at confidences this low.
    "min_confidence": 0.15,
    "min_agreement_gain": 0.02,
}


def _scene_media_delay_guard(store: Store) -> dict[str, float]:
    settings = dict(SCENE_MEDIA_DELAY_GUARD)
    configured = (store.cfg.get("alignment") or {}).get(
        "scene_media_delay_guard"
    ) or {}
    for key, value in configured.items():
        if key not in settings:
            continue
        try:
            settings[key] = float(value)
        except (TypeError, ValueError):
            continue
    return settings


def _measure_scene_media_delays(
    original_mix: Path,
    dubbed_mix: Path,
    selected: list[dict[str, Any]],
    rate: int,
    settings: dict[str, float],
) -> None:
    """Set ``local_media_delay_sec`` per scene, and only when it is verified.

    The music and effects bed is cut from the original and then mixed under the
    Russian voice, which lives on the dubbed timeline.  The alignment map is
    what puts the two on one clock; this is the residual check that the scene
    really did land there, because a coarse map (one reel boundary inside the
    scene, or a single global offset standing in for a wandering one) leaves
    tens of milliseconds behind, and the bed carries them into the mix.

    Nothing is shifted on a hunch: the delay is applied only if it is confident,
    large enough to matter, and measurably improves the agreement between the
    two mixes — otherwise the scene keeps the timeline the map gave it.
    """
    original, original_rate = sf.read(str(original_mix), dtype="float32", always_2d=True)
    dubbed, dubbed_rate = sf.read(str(dubbed_mix), dtype="float32", always_2d=True)
    if original_rate != rate:
        original = audio_io.resample(original, original_rate, rate)
    if dubbed_rate != rate:
        dubbed = audio_io.resample(dubbed, dubbed_rate, rate)
    original_mono = _mono(original)
    dubbed_mono = _mono(dubbed)
    limit = max(0.0, float(settings["max_delay_sec"]))
    floor = max(0.0, float(settings["min_delay_sec"]))
    for item in selected:
        left, right = item["_range"]
        item["local_media_delay_sec"] = 0.0
        item["media_delay_report"] = {"applied": False, "reason": "not_measured"}
        source = original_mono[left:right]
        target = dubbed_mono[left:right]
        if source.size < rate or target.size < rate:
            continue
        estimate = pipeline.estimate_global_offset(source, target, rate, limit)
        delay = float(estimate.offset_sec)
        report = {
            "applied": False,
            "delay_sec": round(delay, 4),
            "confidence": round(float(estimate.confidence), 4),
            "reason": "measured",
        }
        if float(estimate.confidence) < float(settings["min_confidence"]):
            report["reason"] = "low_confidence"
        elif abs(delay) < floor:
            report["reason"] = "below_floor"
        else:
            shift = int(round(delay * rate))
            moved = np.zeros_like(source)
            if shift > 0:
                moved[shift:] = source[:-shift] if shift < source.size else 0.0
            elif shift < 0:
                moved[:shift] = source[-shift:] if -shift < source.size else 0.0
            else:
                moved = source
            before = pipeline._envelope_agreement(source, target, rate)
            after = pipeline._envelope_agreement(moved, target, rate)
            report["agreement_before"] = (
                round(float(before), 4) if math.isfinite(before) else None
            )
            report["agreement_after"] = (
                round(float(after), 4) if math.isfinite(after) else None
            )
            if (
                math.isfinite(before)
                and math.isfinite(after)
                and after >= before + float(settings["min_agreement_gain"])
            ):
                item["local_media_delay_sec"] = round(delay, 4)
                report["applied"] = True
            else:
                report["reason"] = "no_measurable_gain"
        item["media_delay_report"] = report


def _select_speech_scenes(
    candidates: list[dict[str, Any]],
    original_speech: Path,
    dubbed_speech: Path,
    ranges: list[tuple[int, int]],
    rate: int,
    semantic_mode: bool = False,
) -> list[dict[str, Any]]:
    original, _ = sf.read(str(original_speech), dtype="float32", always_2d=True)
    dubbed, _ = sf.read(str(dubbed_speech), dtype="float32", always_2d=True)
    zone_order = list(dict.fromkeys(str(item["zone"]) for item in candidates))
    ranked: dict[str, list[tuple[float, dict[str, Any]]]] = {zone: [] for zone in zone_order}
    for item, (left, right) in zip(candidates, ranges):
        en = _mono(original[left:right])
        both = _mono(dubbed[left:right])
        en_db = _rms_db(en)
        both_db = _rms_db(both)
        block_frames = max(1, int(round(rate * 0.10)))
        block_count = max(1, len(en) // block_frames)
        trimmed = en[: block_count * block_frames].reshape(block_count, block_frames)
        block_db = 10.0 * np.log10(
            np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12
        )
        active_fraction = float(np.mean(block_db >= -42.0))
        peak_speech_db = float(np.max(block_db))
        correlation, lag = _correlation_and_lag(
            en, both, rate, max_lag_sec=1.5
        )
        # The semantic separator needs clearly audible dialogue, not phase
        # coherence.  The legacy subtractor still keeps coherence dominant.
        speech_score = (
            0.4 * en_db + 0.6 * both_db + 10.0 * correlation
            if semantic_mode
            else 100.0 * correlation + min(en_db, both_db)
        )
        value = {
            **item,
            "english_speech_rms_db": round(en_db, 2),
            "combined_speech_rms_db": round(both_db, 2),
            "speech_correlation": round(correlation, 4),
            "local_speech_delay_sec": round(lag, 4),
            "speech_score": round(speech_score, 3),
            "speech_active_fraction": round(active_fraction, 4),
            "peak_speech_rms_db": round(peak_speech_db, 2),
            "short_speech_score": round(
                peak_speech_db - 18.0 * abs(active_fraction - 0.12), 3
            ),
            "subtraction_feasible": bool(
                not bool((item.get("fallback") or {}).get("used"))
                and (semantic_mode or correlation >= FEASIBLE_SPEECH_CORRELATION)
            ),
            "_range": (left, right),
        }
        ranked[str(item["zone"])].append((speech_score, value))
    selected: list[dict[str, Any]] = []
    for zone in zone_order:
        choices = sorted(ranked[zone], key=lambda row: row[0], reverse=True)
        present = [
            row for row in choices
            if row[1]["english_speech_rms_db"] >= MIN_SPEECH_RMS_DB
            and row[1]["combined_speech_rms_db"] >= MIN_SPEECH_RMS_DB
        ]
        if not present:
            # VAD has already chosen the best available timeline segment.  The
            # separator's own stem can be quiet or imperfect, but that must not
            # turn one weak zone into a failed preview job.
            chosen = choices[0][1]
            chosen["subtraction_feasible"] = False
            chosen["speech_profile"] = (
                "VAD-выбор; выделенная речь низкого уровня"
            )
            selected.append(chosen)
            continue
        feasible = [row for row in present if row[1]["subtraction_feasible"]]
        # Prefer a scene where the English is actually cancellable; otherwise
        # keep the best available so the operator can still hear that the pair
        # is unsuitable, with ``subtraction_feasible`` flagged False.
        chosen = (feasible or present)[0][1]
        chosen["speech_profile"] = "Silero VAD: подтверждённая речевая зона"
        selected.append(chosen)
    return selected


def _selected_batch(
    source: Path,
    selected: list[dict[str, Any]],
    destination: Path,
    rate: int,
    shift_key: str | None = None,
) -> tuple[Path, list[tuple[int, int]]]:
    data, source_rate = sf.read(str(source), dtype="float32", always_2d=True)
    if source_rate != rate:
        data = audio_io.resample(data, source_rate, rate)
    gap = np.zeros((rate, data.shape[1]), dtype=np.float32)
    parts: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for item in selected:
        left, right = item["_range"]
        part = data[left:right].copy()
        if shift_key:
            shift = int(round(float(item.get(shift_key) or 0.0) * rate))
            shifted = np.zeros_like(part)
            if shift > 0:
                shifted[shift:] = part[:-shift] if shift < len(part) else 0.0
            elif shift < 0:
                shifted[:shift] = part[-shift:] if -shift < len(part) else 0.0
            else:
                shifted = part
            part = shifted
        parts.extend([part, gap])
        ranges.append((cursor, cursor + len(part)))
        cursor += len(part) + len(gap)
    _write(destination, np.concatenate(parts, axis=0), rate)
    return destination, ranges


def _split_demo_files(
    root: Path,
    selected: list[dict[str, Any]],
    files: dict[str, Path],
    ranges: list[tuple[int, int]],
    rate: int,
    semantic_mode: bool = False,
) -> list[dict[str, Any]]:
    loaded: dict[str, np.ndarray] = {}
    for key, path in files.items():
        values, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if source_rate != rate:
            values = audio_io.resample(values, source_rate, rate)
        loaded[key] = values
    scenes: list[dict[str, Any]] = []
    for index, (item, (left, right)) in enumerate(zip(selected, ranges), 1):
        scene_root = root / "application" / "previews" / f"scene_{index:02d}"
        scene_files: dict[str, str] = {}
        for key, values in loaded.items():
            scene_files[key] = _write(scene_root / f"{key}.flac", values[left:right], rate)
        reference = _mono(loaded["original_en_speech"][left:right])
        speech_input = _mono(loaded["dubbed_en_ru_speech"][left:right])
        speech_metric_key = (
            "model_ru_voice_before_coincident_protection"
            if "model_ru_voice_before_coincident_protection" in loaded
            else "model_ru_voice"
        )
        speech_output = _mono(loaded[speech_metric_key][left:right])
        dirty_input = _mono(loaded["dubbed_mix"][left:right])
        dirty_output = _mono(loaded["direct_dirty_result"][left:right])
        speech_before = _correlation_and_lag(
            reference, speech_input, rate, max_lag_sec=1.5
        )[0]
        speech_after = _correlation_and_lag(
            reference, speech_output, rate, max_lag_sec=1.5
        )[0]
        direct_before = _correlation_and_lag(
            reference, dirty_input, rate, max_lag_sec=1.5
        )[0]
        direct_after = _correlation_and_lag(
            reference, dirty_output, rate, max_lag_sec=1.5
        )[0]
        speech_change = _rms_db(speech_input - speech_output) - _rms_db(speech_input)
        direct_change = _rms_db(dirty_input - dirty_output) - _rms_db(dirty_input)
        result_metrics = {
            "speech_reference_before": round(speech_before, 4),
            "speech_reference_after": round(speech_after, 4),
            "speech_reduction_percent": round(
                max(0.0, 1.0 - speech_after / max(speech_before, 1e-8)) * 100.0,
                1,
            ),
            "direct_reference_before": round(direct_before, 4),
            "direct_reference_after": round(direct_after, 4),
            "direct_reduction_percent": round(
                max(0.0, 1.0 - direct_after / max(direct_before, 1e-8)) * 100.0,
                1,
            ),
            "speech_model_change_db": round(speech_change, 2),
            "direct_model_change_db": round(direct_change, 2),
            "speech_model_effective": bool(
                speech_change >= -30.0
                if semantic_mode
                else speech_before >= 0.04 and speech_after <= speech_before * 0.8
            ),
            "direct_model_effective": bool(
                direct_change >= -30.0
                if semantic_mode
                else direct_before >= 0.04 and direct_after <= direct_before * 0.8
            ),
            "subtraction_feasible": bool(
                not bool((item.get("fallback") or {}).get("used"))
                and (semantic_mode or item.get("subtraction_feasible"))
            ),
        }
        if "algorithmic_en_speech" in loaded and "model_ru_voice_algorithmic" in loaded:
            adapted_reference = _mono(loaded["algorithmic_en_speech"][left:right])
            algorithmic_metric_key = (
                "model_ru_voice_algorithmic_before_coincident_protection"
                if (
                    "model_ru_voice_algorithmic_before_coincident_protection"
                    in loaded
                )
                else "model_ru_voice_algorithmic"
            )
            algorithmic_output = _mono(
                loaded[algorithmic_metric_key][left:right]
            )
            adapted_before = _correlation_and_lag(
                adapted_reference, speech_input, rate, max_lag_sec=1.5
            )[0]
            adapted_after = _correlation_and_lag(
                adapted_reference, algorithmic_output, rate, max_lag_sec=1.5
            )[0]
            algorithmic_change = (
                _rms_db(speech_input - algorithmic_output) - _rms_db(speech_input)
            )
            result_metrics.update(
                {
                    "algorithmic_reference_before": round(adapted_before, 4),
                    "algorithmic_reference_after": round(adapted_after, 4),
                    "algorithmic_reduction_percent": round(
                        max(0.0, 1.0 - adapted_after / max(adapted_before, 1e-8))
                        * 100.0,
                        1,
                    ),
                    "algorithmic_model_change_db": round(algorithmic_change, 2),
                    "algorithmic_model_effective": bool(
                        algorithmic_change >= -30.0
                        and (
                            adapted_before < 0.04
                            or adapted_after <= adapted_before * 0.9
                        )
                    ),
                }
            )
        if "neural_en_speech" in loaded and "model_ru_voice_neural" in loaded:
            neural_reference = _mono(loaded["neural_en_speech"][left:right])
            neural_metric_key = (
                "model_ru_voice_neural_before_coincident_protection"
                if "model_ru_voice_neural_before_coincident_protection" in loaded
                else "model_ru_voice_neural"
            )
            neural_output = _mono(loaded[neural_metric_key][left:right])
            neural_before = _correlation_and_lag(
                neural_reference, speech_input, rate, max_lag_sec=1.5
            )[0]
            neural_after = _correlation_and_lag(
                neural_reference, neural_output, rate, max_lag_sec=1.5
            )[0]
            neural_change = (
                _rms_db(speech_input - neural_output) - _rms_db(speech_input)
            )
            result_metrics.update(
                {
                    "neural_reference_before": round(neural_before, 4),
                    "neural_reference_after": round(neural_after, 4),
                    "neural_reduction_percent": round(
                        max(0.0, 1.0 - neural_after / max(neural_before, 1e-8))
                        * 100.0,
                        1,
                    ),
                    "neural_model_change_db": round(neural_change, 2),
                    "neural_model_effective": bool(
                        neural_change >= -30.0
                        and (
                            neural_before < 0.04
                            or neural_after <= neural_before * 0.9
                        )
                    ),
                }
            )
        if "global_en_speech" in loaded and "model_ru_voice_global" in loaded:
            global_reference = _mono(loaded["global_en_speech"][left:right])
            global_metric_key = (
                "model_ru_voice_global_before_coincident_protection"
                if "model_ru_voice_global_before_coincident_protection" in loaded
                else "model_ru_voice_global"
            )
            global_output = _mono(loaded[global_metric_key][left:right])
            global_before = _correlation_and_lag(
                global_reference, speech_input, rate, max_lag_sec=1.5
            )[0]
            global_after = _correlation_and_lag(
                global_reference, global_output, rate, max_lag_sec=1.5
            )[0]
            global_change = (
                _rms_db(speech_input - global_output) - _rms_db(speech_input)
            )
            result_metrics.update(
                {
                    "global_reference_before": round(global_before, 4),
                    "global_reference_after": round(global_after, 4),
                    "global_reduction_percent": round(
                        max(0.0, 1.0 - global_after / max(global_before, 1e-8))
                        * 100.0,
                        1,
                    ),
                    "global_model_change_db": round(global_change, 2),
                    "global_model_effective": bool(
                        global_change >= -30.0
                        and (
                            global_before < 0.04
                            or global_after <= global_before * 0.9
                        )
                    ),
                }
            )
        public_item = {key: value for key, value in item.items() if not key.startswith("_")}
        scenes.append(
            {
                "id": f"scene_{index:02d}",
                **public_item,
                "result_metrics": result_metrics,
                "files": scene_files,
            }
        )
    return scenes


def _preview_scene_scan_overlaps(
    scene: dict[str, Any],
    scan_report: dict[str, Any],
) -> list[dict[str, Any]]:
    start = float(scene.get("dubbed_start_sec") or 0.0)
    end = start + float(scene.get("duration_sec") or 0.0)
    result: list[dict[str, Any]] = []
    for interval in scan_report.get("candidate_intervals") or []:
        interval_start = float(interval.get("start_sec") or 0.0)
        interval_end = float(interval.get("end_sec") or interval_start)
        overlap_start = max(start, interval_start)
        overlap_end = min(end, interval_end)
        if overlap_end <= overlap_start:
            continue
        result.append(
            {
                "start_sec": round(interval_start, 3),
                "end_sec": round(interval_end, 3),
                "overlap_start_sec": round(overlap_start, 3),
                "overlap_end_sec": round(overlap_end, 3),
                "confidence": interval.get("confidence"),
                "features": interval.get("features") or {},
                "reasons": interval.get("reasons") or [],
            }
        )
    return result


def _preview_song_variant_contracts(
    files: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Describe every full-pipeline voice/reference combination in a scene."""

    def existing(key: str) -> str:
        value = files.get(key)
        return str(Path(str(value)).resolve()) if _valid_audio_file(value) else ""

    variants: dict[str, dict[str, Any]] = {}

    def add_variant(
        name: str,
        *,
        reference_mode: str,
        speech_pass: str,
        detection_reference_key: str,
        detection_voice_key: str,
        translation_voice_key: str,
        synchronized_voice_key: str,
    ) -> None:
        detection_reference = existing(detection_reference_key)
        detection_voice = existing(detection_voice_key)
        translation_voice = existing(translation_voice_key)
        unsynchronized_voice = existing(detection_voice_key)
        synchronized_voice = existing(synchronized_voice_key)
        if not all(
            (detection_reference, detection_voice, translation_voice)
        ):
            return
        playback_voices: list[dict[str, Any]] = []
        if unsynchronized_voice:
            playback_voices.append(
                {
                    "path": unsynchronized_voice,
                    "synchronized": False,
                    "allowed_sync_modes": ["off", "manual"],
                }
            )
        if synchronized_voice:
            playback_voices.append(
                {
                    "path": synchronized_voice,
                    "synchronized": True,
                    "allowed_sync_modes": ["conservative"],
                }
            )
        variants[name] = {
            "reference_mode": reference_mode,
            "speech_pass": speech_pass,
            "detection_reference": detection_reference,
            "detection_voice": detection_voice,
            # Translation dialogue evidence deliberately stays on the first
            # pass, matching the full-film pipeline.
            "translation_voice": translation_voice,
            "playback_voices": playback_voices,
        }

    add_variant(
        "raw_first_pass",
        reference_mode="raw",
        speech_pass="first",
        detection_reference_key="original_song_speech",
        detection_voice_key="model_ru_voice",
        translation_voice_key="model_ru_voice",
        synchronized_voice_key="synchronized_ru_voice",
    )
    add_variant(
        "algorithmic_first_pass",
        reference_mode="algorithmic",
        speech_pass="first",
        detection_reference_key="algorithmic_en_speech",
        detection_voice_key="model_ru_voice_algorithmic",
        translation_voice_key="model_ru_voice_algorithmic",
        synchronized_voice_key="synchronized_ru_voice_algorithmic",
    )
    add_variant(
        "raw_second_pass",
        reference_mode="raw",
        speech_pass="second",
        detection_reference_key="original_song_speech",
        detection_voice_key="model_ru_voice_second_pass",
        translation_voice_key="model_ru_voice",
        synchronized_voice_key="synchronized_ru_voice_second_pass",
    )
    add_variant(
        "algorithmic_second_pass",
        reference_mode="algorithmic",
        speech_pass="second",
        detection_reference_key="algorithmic_en_speech",
        detection_voice_key="model_ru_voice_algorithmic_second_pass",
        translation_voice_key="model_ru_voice_algorithmic",
        synchronized_voice_key=(
            "synchronized_ru_voice_algorithmic_second_pass"
        ),
    )
    return variants


def _protect_preview_song_scenes(
    root: Path,
    scenes: list[dict[str, Any]],
    config: dict[str, Any] | None,
    ctx: Any,
    pair_name: str,
) -> dict[str, Any]:
    """Analyse and protect every already-selected preview scene locally."""
    cfg = song_protection.resolved_config(config)
    aggregate: dict[str, Any] = {
        "schema_version": int(song_protection.SCHEMA_VERSION),
        "created_at": utc_now(),
        "scope": "application_preview",
        "analysis_scope": "selected_preview_scenes_only",
        "config": cfg,
        "scenes": [],
    }
    analysed_scenes = list(scenes)
    for scene_position, scene in enumerate(analysed_scenes, 1):
        if ctx is not None:
            ctx.check_stop()
            ctx.update(
                stage="Защита песен в превью",
                substage=(
                    f"Проверка сцены {scene_position} "
                    f"из {len(analysed_scenes)}"
                ),
                progress=94.0
                + scene_position / max(len(analysed_scenes), 1) * 1.0,
                local_progress=0.0,
                current_file=pair_name,
            )
        files = scene.get("files") or {}
        scene_root = root / "application" / "previews" / str(scene["id"])
        scene_root.mkdir(parents=True, exist_ok=True)
        contract_path = (
            scene_root / song_protection.PREVIEW_CONTRACT_NAME
        )
        precomputed_report_path = (
            scene_root / "song_protection_precomputed.json"
        )
        precomputed_mix_path = (
            scene_root / "song_detection_mix_raw.flac"
        )
        precomputed_protected_path = (
            scene_root / "rebuilt_result_song_protected.flac"
        )
        variants = _preview_song_variant_contracts(files)
        required = (
            "original_song_mix",
            "dubbed_mix",
            "dubbed_en_ru_speech",
            "specialized_me",
        )
        missing = [
            key
            for key in required
            if not _valid_audio_file(files.get(key))
        ]
        default_variant_name = (
            "raw_second_pass"
            if "raw_second_pass" in variants
            else "raw_first_pass"
        )
        default_variant = variants.get(default_variant_name)
        if missing or default_variant is None:
            detection_report = {
                "schema_version": int(song_protection.SCHEMA_VERSION),
                "enabled": bool(cfg["enabled"]),
                "confirmed_intervals": [],
                "restoration_segments": [],
                "summary": (
                    "Защита песни не выполнена: отсутствуют слои "
                    + ", ".join(missing)
                    + "."
                ),
            }
        else:
            # This precomputed card is only the raw default.  The dynamic API
            # reruns the decision with the exact reference/pass/timing selected
            # by the user, so a raw plan is never copied to another variant.
            audio_mix.mix_audio_files(
                Path(files["specialized_me"]),
                Path(default_variant["detection_voice"]),
                precomputed_mix_path,
            )
            detection_cfg = dict(cfg)
            detection_cfg["dialogue_timeline_guard_sec"] = (
                song_protection.dialogue_timeline_guard(
                    detection_cfg,
                    manual_voice_delay_sec=0.0,
                    synchronize_speech=False,
                )
            )
            detection_report = song_protection.detect_song_intervals(
                Path(files["original_song_mix"]),
                Path(files["dubbed_mix"]),
                Path(files["dubbed_en_ru_speech"]),
                Path(files["specialized_me"]),
                precomputed_mix_path,
                detection_cfg,
                report_path=precomputed_report_path,
                ctx=ctx,
                original_speech_stem=Path(
                    default_variant["detection_reference"]
                ),
                translation_voice_stem=Path(
                    default_variant["translation_voice"]
                ),
            )
        restoration_segments = detection_report.get("restoration_segments") or []
        protected_outputs: dict[str, str] = {}
        before_outputs: dict[str, str] = {}
        if default_variant is not None and precomputed_mix_path.is_file():
            files["rebuilt_result_before_song_protection"] = str(
                precomputed_mix_path.resolve()
            )
            files["rebuilt_result"] = str(precomputed_mix_path.resolve())
            before_outputs["rebuilt_result"] = str(
                precomputed_mix_path.resolve()
            )
            if restoration_segments:
                song_protection.apply_song_protection(
                    precomputed_mix_path,
                    Path(files["dubbed_mix"]),
                    precomputed_protected_path,
                    restoration_segments,
                    original_mix=Path(files["original_song_mix"]),
                    crossfade_sec=float(cfg["crossfade_sec"]),
                    ctx=ctx,
                )
                files["rebuilt_result"] = str(
                    precomputed_protected_path.resolve()
                )
                protected_outputs["rebuilt_result"] = str(
                    precomputed_protected_path.resolve()
                )
        scene_offset = float(scene.get("dubbed_start_sec") or 0.0)

        def globalised(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for item in items:
                value = dict(item)
                value["local_start_sec"] = item.get("start_sec")
                value["local_end_sec"] = item.get("end_sec")
                value["start_sec"] = round(
                    scene_offset + float(item.get("start_sec") or 0.0), 3
                )
                value["end_sec"] = round(
                    scene_offset + float(item.get("end_sec") or 0.0), 3
                )
                result.append(value)
            return result

        scene_state = {
            "analysis_scope": "scene_local",
            "report": str(contract_path.resolve()),
            "precomputed_report": str(
                precomputed_report_path.resolve()
            ),
            "precomputed_variant": default_variant_name,
            "dynamic_variants": sorted(variants),
            "classifier": detection_report.get("classifier") or {},
            "confirmed_intervals": globalised(
                detection_report.get("confirmed_intervals") or []
            ),
            "restoration_segments": globalised(restoration_segments),
            "protected": bool(restoration_segments),
            "summary": detection_report.get("summary"),
            "reasons": [
                reason
                for interval in detection_report.get("confirmed_intervals") or []
                for reason in interval.get("reasons") or []
            ],
            "before_outputs": before_outputs,
            "protected_outputs": protected_outputs,
        }
        detection_report["preview_scene"] = {
            "id": scene["id"],
            "global_start_sec": round(scene_offset, 3),
            "global_end_sec": round(
                scene_offset + float(scene.get("duration_sec") or 0.0), 3
            ),
            "analysis_scope": "scene_local",
            "protected_outputs": protected_outputs,
            "before_outputs": before_outputs,
            "variant": default_variant_name,
        }
        atomic_json(precomputed_report_path, detection_report)
        contract = {
            "schema_version": int(
                song_protection.PREVIEW_CONTRACT_SCHEMA_VERSION
            ),
            "report_kind": song_protection.PREVIEW_CONTRACT_KIND,
            "report_name": song_protection.PREVIEW_CONTRACT_NAME,
            "created_at": utc_now(),
            "scene_id": str(scene["id"]),
            "scene_directory": str(scene_root.resolve()),
            "config": cfg,
            "sources": {
                "original_detection_mix": str(
                    Path(files["original_song_mix"]).resolve()
                )
                if _valid_audio_file(files.get("original_song_mix"))
                else "",
                "dubbed_mix": str(Path(files["dubbed_mix"]).resolve())
                if _valid_audio_file(files.get("dubbed_mix"))
                else "",
                "dubbed_speech_stem": str(
                    Path(files["dubbed_en_ru_speech"]).resolve()
                )
                if _valid_audio_file(files.get("dubbed_en_ru_speech"))
                else "",
                "music_background": str(
                    Path(files["specialized_me"]).resolve()
                )
                if _valid_audio_file(files.get("specialized_me"))
                else "",
            },
            "variants": variants,
            "precomputed": {
                "variant": default_variant_name,
                "report": str(precomputed_report_path.resolve()),
            },
        }
        atomic_json(contract_path, contract)
        files["song_protection_report"] = str(contract_path.resolve())
        scene["song_protection"] = scene_state
        aggregate["scenes"].append({"id": scene["id"], **scene_state})

    aggregate["summary"] = {
        "analysed_scene_count": len(analysed_scenes),
        "candidate_scene_count": sum(
            1
            for scene in analysed_scenes
            if bool(
                (scene.get("song_protection") or {}).get(
                    "confirmed_intervals"
                )
            )
        ),
        "protected_scene_count": sum(
            1
            for scene in analysed_scenes
            if bool((scene.get("song_protection") or {}).get("protected"))
        ),
        "restoration_segment_count": sum(
            len(
                (scene.get("song_protection") or {}).get(
                    "restoration_segments"
                )
                or []
            )
            for scene in analysed_scenes
        ),
    }
    report_path = root / "application" / "preview_song_protection_report.json"
    atomic_json(report_path, aggregate)
    aggregate["report"] = str(report_path.resolve())
    return aggregate


def _write_vad_preview_report(root: Path, scenes: list[dict[str, Any]]) -> Path:
    """Create a standalone local page for listening to all VAD-selected clips."""
    rows: list[str] = []
    for scene in scenes:
        files = scene.get("files") or {}
        scene_id = str(scene.get("id") or "scene")
        audio = f"previews/{scene_id}/dubbed_mix.flac"
        speech_start = scene.get("speech_region_start_sec")
        speech_end = scene.get("speech_region_end_sec")
        details = {
            "Зона": scene.get("zone"),
            "Окно анализа": f"{scene.get('analysis_start_sec')}–{scene.get('analysis_end_sec')} с",
            "Превью": f"{scene.get('preview_start_sec')}–{scene.get('preview_end_sec')} с",
            "Речь": f"{speech_start}–{speech_end} с" if speech_start is not None else "не подтверждена",
            "Доля речи": scene.get("speech_ratio"),
            "Статус": scene.get("status"),
            "Резервный выбор": (
                scene.get("fallback") or {}
            ).get("used", False),
        }
        meta = "".join(
            f"<dt>{escape(str(key))}</dt><dd>{escape(str(value))}</dd>"
            for key, value in details.items()
        )
        # Waveform of the exact preview clip, with the confirmed speech region
        # highlighted, so the operator sees what VAD selected without playing it.
        clip_path = files.get("dubbed_mix")
        preview_start = float(scene.get("preview_start_sec") or 0.0)
        clip_duration = float(scene.get("preview_duration_sec") or scene.get("duration_sec") or 0.0)
        highlights: list[tuple[float, float]] = []
        if speech_start is not None and speech_end is not None:
            highlights = [(float(speech_start) - preview_start, float(speech_end) - preview_start)]
        wave = (
            vad_preview.waveform_svg(Path(clip_path), 0.0, clip_duration, highlights=highlights)
            if clip_path and clip_duration > 0.0
            else ""
        )
        rows.append(
            f"<section><h2>{escape(scene_id)} — {escape(str(scene.get('zone')))}</h2>"
            f"{wave}"
            f"<audio controls preload=\"metadata\" src=\"{escape(audio)}\"></audio>"
            f"<dl>{meta}</dl><p>{escape(str(scene.get('selection_reason') or ''))}</p></section>"
        )
    document = """<!doctype html><html lang=\"ru\"><meta charset=\"utf-8\">
<title>DubClean — проверка Silero VAD</title><style>
body{font:16px system-ui,sans-serif;max-width:960px;margin:32px auto;padding:0 18px;color:#172033}
section{border:1px solid #d9dee8;border-radius:12px;padding:16px;margin:14px 0}h1{margin-bottom:4px}
audio{width:100%;margin:8px 0}dl{display:grid;grid-template-columns:180px 1fr;gap:5px 14px}dt{font-weight:650}dd{margin:0}
</style><h1>Проверка превью Silero VAD</h1><p>Все интервалы указаны в таймлайне дорожки перевода.</p>""" + "\n".join(rows) + "</html>"
    destination = root / "application" / "vad_preview_report.html"
    destination.write_text(document, encoding="utf-8")
    return destination


def _average_metric(
    scenes: list[dict[str, Any]], key: str, *, minimum_before_key: str | None = None
) -> float | None:
    values: list[float] = []
    for scene in scenes:
        metrics = scene.get("result_metrics") or {}
        if minimum_before_key is not None:
            if float(metrics.get(minimum_before_key) or 0.0) < 0.04:
                continue
        if metrics.get(key) is None:
            continue
        values.append(float(metrics[key]))
    if not values:
        return None
    return float(sum(values) / len(values))


def _select_reference_adapter_from_preview(
    scenes: list[dict[str, Any]],
    algorithmic_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usable = [
        scene
        for scene in scenes
        if (
            (scene.get("result_metrics") or {}).get("algorithmic_reduction_percent")
            is not None
            or (scene.get("result_metrics") or {}).get("neural_reduction_percent")
            is not None
            or (scene.get("result_metrics") or {}).get("global_reduction_percent")
            is not None
        )
        and max(
            float((scene.get("result_metrics") or {}).get("speech_reference_before") or 0.0),
            float((scene.get("result_metrics") or {}).get("algorithmic_reference_before") or 0.0),
            float((scene.get("result_metrics") or {}).get("neural_reference_before") or 0.0),
            float((scene.get("result_metrics") or {}).get("global_reference_before") or 0.0),
        )
        >= 0.04
    ]
    raw_reduction = _average_metric(
        usable, "speech_reduction_percent", minimum_before_key="speech_reference_before"
    )
    algorithmic_reduction = _average_metric(
        usable,
        "algorithmic_reduction_percent",
        minimum_before_key="algorithmic_reference_before",
    )
    raw_change = _average_metric(usable, "speech_model_change_db")
    algorithmic_change = _average_metric(usable, "algorithmic_model_change_db")
    raw_after = _average_metric(
        usable, "speech_reference_after", minimum_before_key="speech_reference_before"
    )
    algorithmic_after = _average_metric(
        usable,
        "algorithmic_reference_after",
        minimum_before_key="algorithmic_reference_before",
    )
    neural_reduction = _average_metric(
        usable,
        "neural_reduction_percent",
        minimum_before_key="neural_reference_before",
    )
    neural_change = _average_metric(usable, "neural_model_change_db")
    neural_after = _average_metric(
        usable,
        "neural_reference_after",
        minimum_before_key="neural_reference_before",
    )
    global_reduction = _average_metric(
        usable,
        "global_reduction_percent",
        minimum_before_key="global_reference_before",
    )
    global_change = _average_metric(usable, "global_model_change_db")
    global_after = _average_metric(
        usable,
        "global_reference_after",
        minimum_before_key="global_reference_before",
    )
    enough_data = len(usable) >= 3
    candidates: list[dict[str, Any]] = []

    def add_candidate(
        name: str,
        reduction: float | None,
        change: float | None,
        after: float | None,
        *,
        safety_usable: bool = True,
    ) -> None:
        if reduction is None or raw_reduction is None:
            return
        delta = reduction - raw_reduction
        not_over_destructive = (
            change is not None
            and raw_change is not None
            and change >= raw_change - 4.0
        )
        lower_leakage = (
            after is not None
            and raw_after is not None
            and after <= raw_after * 0.95
        )
        usable_candidate = bool(
            safety_usable
            and
            enough_data
            and delta >= 5.0
            and not_over_destructive
            and (lower_leakage or reduction >= 50.0)
        )
        candidates.append(
            {
                "name": name,
                "reduction": reduction,
                "delta": delta,
                "change": change,
                "after": after,
                "usable": usable_candidate,
                "safety_usable": bool(safety_usable),
            }
        )

    algorithmic_safety_usable = bool(
        ((algorithmic_profile or {}).get("summary") or {}).get("usable")
    )
    add_candidate(
        "algorithmic",
        algorithmic_reduction,
        algorithmic_change,
        algorithmic_after,
        safety_usable=algorithmic_safety_usable,
    )
    add_candidate("neural", neural_reduction, neural_change, neural_after)
    add_candidate("global", global_reduction, global_change, global_after)
    usable_candidates = [candidate for candidate in candidates if candidate["usable"]]
    selected_candidate = (
        max(usable_candidates, key=lambda item: (float(item["delta"]), float(item["reduction"])))
        if usable_candidates
        else None
    )
    recommended = str(selected_candidate["name"]) if selected_candidate else "raw"
    reduction_delta = (
        float(selected_candidate["delta"]) if selected_candidate is not None else None
    )
    return {
        "schema_version": 1,
        "recommended": recommended,
        "reason": (
            f"{recommended}_preview_reduced_en_leakage"
            if selected_candidate is not None
            else (
                "algorithmic_common_component_not_confident"
                if not algorithmic_safety_usable
                else "not_enough_comparable_scenes"
                if not enough_data
                else "adapters_preview_not_better_enough"
            )
        ),
        "usable_scenes": len(usable),
        "raw_reduction_percent_avg": (
            round(raw_reduction, 2) if raw_reduction is not None else None
        ),
        "algorithmic_reduction_percent_avg": (
            round(algorithmic_reduction, 2)
            if algorithmic_reduction is not None
            else None
        ),
        "reduction_delta_percent": (
            round(reduction_delta, 2) if reduction_delta is not None else None
        ),
        "raw_model_change_db_avg": (
            round(raw_change, 2) if raw_change is not None else None
        ),
        "algorithmic_model_change_db_avg": (
            round(algorithmic_change, 2)
            if algorithmic_change is not None
            else None
        ),
        "raw_reference_after_avg": round(raw_after, 4) if raw_after is not None else None,
        "algorithmic_reference_after_avg": (
            round(algorithmic_after, 4) if algorithmic_after is not None else None
        ),
        "neural_reduction_percent_avg": (
            round(neural_reduction, 2) if neural_reduction is not None else None
        ),
        "neural_model_change_db_avg": (
            round(neural_change, 2) if neural_change is not None else None
        ),
        "neural_reference_after_avg": (
            round(neural_after, 4) if neural_after is not None else None
        ),
        "global_reduction_percent_avg": (
            round(global_reduction, 2) if global_reduction is not None else None
        ),
        "global_model_change_db_avg": (
            round(global_change, 2) if global_change is not None else None
        ),
        "global_reference_after_avg": (
            round(global_after, 4) if global_after is not None else None
        ),
        "candidates": candidates,
    }


def _make_video_previews(
    store: Store,
    pair: dict[str, Any],
    scenes: list[dict[str, Any]],
    ctx: Any,
) -> None:
    video = Path(pair["sources"]["dubbed"]["path"]).resolve()
    for index, scene in enumerate(scenes, 1):
        audio = Path(scene["files"]["rebuilt_result"]).resolve()
        preview_duration = max(
            1.0,
            float(
                scene.get("preview_duration_sec")
                or scene.get("duration_sec")
                or 15.0
            ),
        )
        destination = audio.with_name("rebuilt_preview.mp4")
        if (
            not destination.is_file()
            or destination.stat().st_size == 0
            or destination.stat().st_mtime < audio.stat().st_mtime
        ):
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.partial.mp4"
            )
            command = [
                audio_io.check_ffmpeg(),
                "-y",
                "-ss",
                str(float(scene["dubbed_start_sec"])),
                "-i",
                str(video),
                "-i",
                str(audio),
                "-t",
                f"{preview_duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "25",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
            progress_context = _ProgressRangeContext(
                ctx,
                95.0 + (index - 1) / max(len(scenes), 1) * 4.0,
                4.0 / max(len(scenes), 1),
            )
            pipeline._run_ffmpeg_progress(
                command,
                preview_duration,
                progress_context,
                f"Видеопредпросмотр {index} из {len(scenes)}",
                video.name,
            )
            os.replace(temporary, destination)
        scene["files"]["video_preview"] = str(destination.resolve())


def _feasibility_verdict(
    scenes: list[dict[str, Any]], semantic_mode: bool = False
) -> dict[str, Any]:
    """Honest, pair-level answer to «можно ли вообще удалить английскую речь».

    A subtractor removes only a phase-coherent copy of the reference.  If no
    scene reaches the feasibility correlation, the dubbed soundtrack is a
    different master and full-film processing will not remove any English —
    the operator must know this before committing hours of GPU time."""
    total = len(scenes)
    feasible = [
        scene for scene in scenes
        if bool((scene.get("result_metrics") or {}).get("subtraction_feasible")
                or scene.get("subtraction_feasible"))
    ]
    effective = [
        scene for scene in scenes
        if bool((scene.get("result_metrics") or {}).get("speech_model_effective"))
    ]
    correlations = sorted(
        float(scene.get("speech_correlation") or 0.0) for scene in scenes
    )
    median_correlation = (
        correlations[len(correlations) // 2] if correlations else 0.0
    )
    ready = bool(total == 5) if semantic_mode else bool(feasible) and bool(effective)
    if semantic_mode:
        verdict = (
            "Используется смысловое разделение RU/EN: фазовое совпадение мастер-версий "
            "не требуется. Прослушайте пять разговорных сцен и подтвердите сборку фильма."
        )
    elif ready:
        verdict = (
            f"Английская речь удаляема на {len(effective)} из {total} сцен. "
            "Прослушайте превью и подтвердите сборку фильма вручную."
        )
    elif feasible:
        verdict = (
            "Английский референс когерентен, но измеримого удаления в превью не "
            "получено. Не запускайте полный фильм без прослушивания."
        )
    else:
        verdict = (
            "Оригинал и дубляж фазово некогерентны (общий английский слой не "
            f"обнаружен, медианная связь {median_correlation:.2f}). Вычитание не "
            "удалит английскую речь на этой паре — сборка не даст нужного результата."
        )
    return {
        "scenes_total": total,
        "scenes_subtraction_feasible": len(feasible),
        "scenes_model_effective": len(effective),
        "median_reference_correlation": round(median_correlation, 4),
        "ready_for_full_film": ready,
        "verdict": verdict,
    }


def _full_run_blocked(preview: dict[str, Any] | None, force: bool) -> str | None:
    """Refuse a multi-hour full-film run when the demo proved it is pointless.

    A preview without a feasibility block (older run) is allowed for backward
    compatibility; a fresh one that measured no cancellable English is blocked
    unless the operator explicitly overrides with ``force``."""
    if force:
        return None
    if (preview or {}).get("model_kind") == "semantic_ru_separator":
        return None
    feasibility = (preview or {}).get("feasibility") or {}
    if "ready_for_full_film" in feasibility and not feasibility["ready_for_full_film"]:
        return (
            "Сборка остановлена: превью не показало удаления английской речи. "
            f"{feasibility.get('verdict', '')} "
            "Если всё же нужно обработать фильм, повторите запуск с параметром force."
        ).strip()
    return None


def _song_protection_completion_status(report: dict[str, Any]) -> str:
    """Produce the user-facing completion status without hiding model errors."""
    if not bool(report.get("enabled", True)):
        return "Защита песенных фрагментов отключена"
    classifier = report.get("classifier") or {}
    if classifier.get("available") is False:
        return "Классификатор песен недоступен; автоматическая защита не выполнена"
    restoration_segments = report.get("restoration_segments") or []
    if restoration_segments:
        return f"Защищено песенных участков: {len(restoration_segments)}"
    if report.get("confirmed_intervals"):
        return "Песни найдены, но безопасных участков для замены нет"
    return "Уверенные песенные интервалы не найдены"


def _ensure_full_speech_stems(
    store: Store,
    root: Path,
    original_path: Path,
    dubbed_path: Path,
    ctx: Any,
    pair_name: str,
    *,
    progress_base: float = 70.0,
) -> tuple[Path, Path]:
    original_speech = root / "stems" / "original" / "en_speech.flac"
    dubbed_speech = root / "stems" / "dubbed" / "speech_en_ru.flac"
    original_speech.parent.mkdir(parents=True, exist_ok=True)
    dubbed_speech.parent.mkdir(parents=True, exist_ok=True)
    if not original_speech.is_file() or original_speech.stat().st_size == 0:
        pipeline._run_speech_extractor(
            store,
            original_path,
            original_speech,
            ctx,
            pair_name,
            "Полный фильм: выделение английской речи для выравнивания качества",
            progress_base=progress_base,
            progress_span=10.0,
        )
    if not dubbed_speech.is_file() or dubbed_speech.stat().st_size == 0:
        pipeline._run_speech_extractor(
            store,
            dubbed_path,
            dubbed_speech,
            ctx,
            pair_name,
            "Полный фильм: выделение речи перевода для выравнивания качества",
            progress_base=progress_base + 10.0,
            progress_span=10.0,
            model_role="dubbed",
        )
    return original_speech, dubbed_speech


def _fit_or_load_global_mastering_profile(
    store: Store,
    root: Path,
    original_path: Path,
    dubbed_path: Path,
    alignment: dict[str, Any],
    rate: int,
    duration: float,
    ctx: Any,
    pair_name: str,
    checkpoint: Path | None,
) -> tuple[dict[str, Any], Path, dict[str, str]]:
    global_root = root / "application" / "global_mastering"
    global_root.mkdir(parents=True, exist_ok=True)
    original_speech, dubbed_speech = _ensure_full_speech_stems(
        store,
        root,
        original_path,
        dubbed_path,
        ctx,
        pair_name,
        progress_base=64.0,
    )
    aligned_original_mix = _align_to_dubbed(
        original_path,
        global_root / "aligned_original_mix.flac",
        alignment,
        rate,
        duration,
        1,
        ctx,
        pair_name,
        "Выравнивание качества: сопоставление оригинального звука",
    )
    aligned_original_speech = _align_to_dubbed(
        original_speech,
        global_root / "aligned_original_speech.flac",
        alignment,
        rate,
        duration,
        1,
        ctx,
        pair_name,
        "Выравнивание качества: сопоставление английской речи",
    )
    algorithmic_report = global_root / "global_profile_algorithmic.json"
    if checkpoint is None:
        report_path = algorithmic_report
        cached_profile = read_json(report_path, {}) if report_path.is_file() else {}
        if (
            not report_path.is_file()
            or report_path.stat().st_size == 0
            or (cached_profile or {}).get("mode") != GLOBAL_MASTERING_PROFILE_MODE
        ):
            ctx.update(
                stage="Выравнивание качества дорожек",
                substage="Оценка единого профиля по участкам без речи",
                progress=82.0,
                local_progress=0.0,
                current_file=pair_name,
            )
            profile = fit_global_profile_for_files(
                aligned_original_mix,
                dubbed_path,
                aligned_original_speech,
                dubbed_speech,
                report_path=report_path,
            )
        else:
            profile = cached_profile or {}
    else:
        report_path = global_root / f"global_profile_{checkpoint.stem}.json"
        if not _cached_model_output_valid_for(report_path, [checkpoint]):
            ctx.update(
                stage="Выравнивание качества дорожек",
                substage="Нейросетевая дооценка единого профиля по участкам без речи",
                progress=82.0,
                local_progress=0.0,
                current_file=pair_name,
            )
            profile = fit_neural_global_profile_for_files(
                aligned_original_mix,
                dubbed_path,
                aligned_original_speech,
                dubbed_speech,
                checkpoint,
                report_path=report_path,
                algorithmic_report_path=algorithmic_report,
            )
            _record_model_output_for(report_path, [checkpoint])
        else:
            profile = read_json(report_path, {}) or {}
    return profile, report_path, {
        "aligned_original_mix": str(aligned_original_mix.resolve()),
        "aligned_original_speech": str(aligned_original_speech.resolve()),
        "dubbed_speech": str(dubbed_speech.resolve()),
        "algorithmic_report": str(algorithmic_report.resolve()),
    }


def _global_profile_usable(profile: dict[str, Any] | None) -> bool:
    """Only a measured, holdout-validated profile may touch the reference.

    Older reports did not have an explicit ``usable`` field, so ``verdict=ok``
    remains readable for migration.  New reports always carry both values.
    """
    summary = (profile or {}).get("summary") or {}
    if "usable" in summary:
        return bool(summary["usable"])
    return str(summary.get("verdict") or "") == "ok"


def prepare_application(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    pair = store.load_pair(project_id, pair_id)
    applied_routing = _applied_reference_routing(pair)
    root = store.pair_dir(project_id, pair_id)
    checkpoint = Path(str(parameters.get("checkpoint") or "")).resolve()
    if not checkpoint.is_file() or not checkpoint.name.lower().endswith(".pt"):
        raise RuntimeError("Выберите существующее сохранение обученной модели.")
    semantic_mode = _is_semantic_subtractor(checkpoint)
    background_checkpoint_value = str(parameters.get("background_checkpoint") or "")
    background_checkpoint = (
        Path(background_checkpoint_value).resolve()
        if background_checkpoint_value
        else None
    )
    if background_checkpoint is not None and (
        not background_checkpoint.is_file()
        or not background_checkpoint.name.lower().endswith(".pt")
    ):
        raise RuntimeError("Выберите установленную модель музыки и эффектов.")
    neural_adapter_checkpoint = _resolve_neural_reference_adapter_checkpoint(parameters)
    global_adapter_checkpoint = _resolve_global_mastering_checkpoint(parameters)
    speech_second_pass = bool(
        parameters.get(
            "speech_extraction_second_pass",
            pair.get("speech_extraction_second_pass", False),
        )
    )
    speech_second_pass_dependencies = (
        _speech_extractor_dependencies(store, "original")
        if speech_second_pass
        else []
    )
    song_cfg = dict(store.cfg.get("song_protection") or {})
    song_preview_identity = _song_protection_preview_identity(song_cfg)
    coincident_cfg = dict(
        store.cfg.get("coincident_speech_protection") or {}
    )
    coincident_identity = _coincident_speech_protection_identity(
        coincident_cfg
    )
    integrity_cfg = dict(store.cfg.get("speech_integrity_guard") or {})
    integrity_identity = _speech_integrity_guard_identity(integrity_cfg)
    synchronization_cfg = dict(
        store.cfg.get("speech_synchronization") or {}
    )
    reference_level_cfg = dict(
        store.cfg.get("reference_speech_level_alignment") or {}
    )
    reference_level_identity = {
        "schema_version": REFERENCE_LEVEL_ALIGNMENT_CACHE_VERSION,
        "algorithm": "robust_common_component_cross_power_v1",
        "config": reference_level_cfg,
    }
    song_preview_cache_token = hashlib.sha256(
        json.dumps(
            song_preview_identity,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    reference_level_cache_token = hashlib.sha256(
        json.dumps(
            reference_level_identity,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]

    _reuse_existing_preparation(store, project_id, pair, ctx)
    pair = store.load_pair(project_id, pair_id)
    extraction_ready = (
        pair.get("stages", {}).get("2", {}).get("status") == "completed"
        and (root / "extracted" / "original.flac").is_file()
        and (root / "extracted" / "dubbed.flac").is_file()
        and (root / "extracted" / "original_proxy.wav").is_file()
        and (root / "extracted" / "dubbed_proxy.wav").is_file()
        and (root / "extracted" / "metadata.json").is_file()
    )
    if not extraction_ready:
        from experiments.paired_reference_cancel.task_worker import (
            ProgressPhaseContext,
        )

        pipeline.extract_pair(
            store,
            project_id,
            pair_id,
            ProgressPhaseContext(ctx, 0.0, 10.0, "Подготовка · извлечение"),
        )
    pair = store.load_pair(project_id, pair_id)
    alignment_ready = (
        pair.get("stages", {}).get("3", {}).get("status") == "completed"
        and (root / "alignment" / "alignment_map.json").is_file()
    )
    if not alignment_ready:
        from experiments.paired_reference_cancel.task_worker import (
            ProgressPhaseContext,
        )

        pipeline.align_pair(
            store,
            project_id,
            pair_id,
            ProgressPhaseContext(ctx, 10.0, 10.0, "Подготовка · сопоставление"),
        )
    pair = store.load_pair(project_id, pair_id)
    pair["alignment_review"] = {
        "accepted": True,
        "note": "Автоматически принято для подготовки превью; карта показана в рабочем разделе.",
        "reviewed_at": utc_now(),
    }
    pair["stages"]["4"] = {"status": "completed", "message": "Карта принята для превью."}
    store.save_pair(pair)
    pair = store.load_pair(project_id, pair_id)
    preparation_identity = _preparation_identity(
        pair,
        checkpoint,
        background_checkpoint,
        applied_routing,
        speech_second_pass,
        speech_second_pass_dependencies,
        song_preview_identity,
        coincident_identity,
        reference_level_identity,
        integrity_identity,
    )
    _begin_application_preparation(
        store, project_id, pair_id, preparation_identity
    )
    alignment = pipeline._processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )

    original_path = root / "extracted" / "original.flac"
    dubbed_path = root / "extracted" / "dubbed.flac"
    dubbed_info = sf.info(str(dubbed_path))
    rate = int(dubbed_info.samplerate)
    duration = float(dubbed_info.duration)
    scan_root = root / "application" / "candidate_scan"
    selection_path = scan_root / "selection.json"
    source_signature = _application_source_signature(pair)
    previous_selection = read_json(selection_path, {}) or {}
    selection_reusable = (
        previous_selection.get("version") == SCENE_SELECTION_VERSION
        and previous_selection.get("source_signature") == source_signature
        and previous_selection.get("candidates")
    )
    if selection_reusable:
        candidates = previous_selection["candidates"]
        ctx.update(
            stage="Поиск сцен с той же английской репликой",
            substage="Используются уже выбранные речевые участки",
            progress=20.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
    else:
        shutil.rmtree(scan_root, ignore_errors=True)
        scan_root.mkdir(parents=True, exist_ok=True)
        candidates = _vad_candidates(
            dubbed_path, duration, alignment, ctx=ctx, pair_name=pair["name"]
        )
        candidates = [
            item if item.get("duration_sec") else _emergency_zone_fallback(item)
            for item in candidates
        ]
    atomic_json(
        selection_path,
        {
            "version": SCENE_SELECTION_VERSION,
            "source_signature": source_signature,
            "candidates": candidates,
        },
    )
    original_batch, dubbed_batch, ranges = _make_candidate_batches(
        root,
        candidates,
        rate,
    )
    original_speech = scan_root / "original_speech_candidates.flac"
    dubbed_speech = scan_root / "dubbed_speech_candidates.flac"
    original_speech_dependencies = [
        original_batch,
        *_speech_extractor_dependencies(store, "original"),
    ]
    dubbed_speech_dependencies = [
        dubbed_batch,
        *_speech_extractor_dependencies(store, "dubbed"),
    ]
    if not _cached_model_output_valid_for(
        original_speech,
        original_speech_dependencies,
    ):
        pipeline._run_speech_extractor(
            store, original_batch, original_speech, ctx, pair["name"],
            "Подготовка речевых сцен: английская речь",
            progress_base=20.0,
            progress_span=30.0,
        )
        _record_model_output_for(
            original_speech,
            original_speech_dependencies,
        )
    else:
        ctx.update(
            stage="Подготовка речевых сцен",
            substage="Используется уже выделенная английская речь",
            progress=35.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
    if not _cached_model_output_valid_for(
        dubbed_speech,
        dubbed_speech_dependencies,
    ):
        pipeline._run_speech_extractor(
            store, dubbed_batch, dubbed_speech, ctx, pair["name"],
            "Подготовка речевых сцен: речь оригинала и перевода",
            progress_base=50.0,
            progress_span=30.0,
            model_role="dubbed",
        )
        _record_model_output_for(
            dubbed_speech,
            dubbed_speech_dependencies,
        )
    else:
        ctx.update(
            stage="Подготовка речевых сцен",
            substage="Используется уже выделенная речь оригинала и перевода",
            progress=55.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
    selected = _select_speech_scenes(
        candidates,
        original_speech,
        dubbed_speech,
        ranges,
        rate,
        semantic_mode,
    )
    _measure_scene_media_delays(
        original_batch,
        dubbed_batch,
        selected,
        rate,
        _scene_media_delay_guard(store),
    )
    selected_original_speech, selected_ranges = _selected_batch(
        original_speech,
        selected,
        scan_root / "selected_original_speech.flac",
        rate,
        shift_key=None if semantic_mode else "local_speech_delay_sec",
    )
    selected_original_background_speech, _ = _selected_batch(
        original_speech,
        selected,
        scan_root / "selected_original_background_speech.flac",
        rate,
        shift_key="local_speech_delay_sec",
    )
    selected_original_song_speech, _ = _selected_batch(
        original_speech,
        selected,
        scan_root / "selected_original_song_speech.flac",
        rate,
        shift_key="local_media_delay_sec",
    )
    selected_dubbed_speech, _ = _selected_batch(
        dubbed_speech, selected, scan_root / "selected_dubbed_speech.flac", rate
    )
    selected_original_mix, _ = _selected_batch(
        original_batch,
        selected,
        scan_root / "selected_original_mix.flac",
        rate,
        shift_key="local_speech_delay_sec",
    )
    # The music and effects bed is built from this pair and then mixed under a
    # voice that lives on the dubbed clock.  Cutting it without the residual
    # scene delay left the bed up to a tenth of a second away from the dialogue
    # it belongs to — the local speech lag is the wrong correction here (it
    # measures how the dubbing actor is placed against the English one), which
    # is why the two song batches carry their own, media-derived delay.
    selected_original_song_mix, _ = _selected_batch(
        original_batch,
        selected,
        scan_root / "selected_original_song_mix.flac",
        rate,
        shift_key="local_media_delay_sec",
    )
    selected_dubbed_mix, _ = _selected_batch(
        dubbed_batch, selected, scan_root / "selected_dubbed_mix.flac", rate
    )
    _save_application_preparation_phase(
        store,
        project_id,
        pair_id,
        preparation_identity,
        "speech",
        {
            "original_en_speech": selected_original_speech,
            "original_background_speech": selected_original_background_speech,
            "original_song_speech": selected_original_song_speech,
            "dubbed_en_ru_speech": selected_dubbed_speech,
            "original_mix": selected_original_mix,
            "original_song_mix": selected_original_song_mix,
            "dubbed_mix": selected_dubbed_mix,
        },
    )

    model_root = (
        scan_root
        / "model_results"
        / (
            f"selection_v{SCENE_SELECTION_VERSION}"
            f"_song_{song_preview_cache_token}"
        )
        / checkpoint.stem
    )
    model_root.mkdir(parents=True, exist_ok=True)
    adapter_root = (
        scan_root
        / "reference_adapter"
        / (
            f"selection_v{SCENE_SELECTION_VERSION}"
            f"_song_{song_preview_cache_token}"
            f"_level_{reference_level_cache_token}"
        )
    )
    adapter_root.mkdir(parents=True, exist_ok=True)
    algorithmic_reference = adapter_root / "selected_original_speech_algorithmic.flac"
    algorithmic_report_path = adapter_root / "algorithmic_reference_report.json"
    if (
        not algorithmic_reference.is_file()
        or algorithmic_reference.stat().st_size == 0
        or not algorithmic_report_path.is_file()
    ):
        ctx.update(
            stage="Адаптация английского референса",
            substage="Алгоритмическая оценка канала по пяти сценам",
            progress=80.0,
            local_progress=0.0,
            current_file=pair["name"],
        )
        algorithmic_report = adapt_reference_file(
            selected_original_speech,
            selected_dubbed_speech,
            algorithmic_reference,
            report_path=algorithmic_report_path,
            calibration_reference_path=selected_original_background_speech,
            config=reference_level_cfg,
        )
    else:
        algorithmic_report = read_json(algorithmic_report_path, {}) or {}
    neural_reference: Path | None = None
    neural_report: dict[str, Any] | None = None
    neural_report_path: Path | None = None
    if neural_adapter_checkpoint is not None:
        neural_reference = adapter_root / "selected_original_speech_neural.flac"
        neural_report_path = adapter_root / "neural_reference_report.json"
        if not _cached_model_output_valid_for(neural_reference, [neural_adapter_checkpoint]):
            ctx.update(
                stage="Адаптация английского референса",
                substage="Автоматическая подгонка по пяти сценам",
                progress=81.0,
                local_progress=0.0,
                current_file=pair["name"],
            )
            neural_report = adapt_reference_file_neural(
                selected_original_speech,
                selected_dubbed_speech,
                neural_adapter_checkpoint,
                neural_reference,
                report_path=neural_report_path,
            )
            _record_model_output_for(neural_reference, [neural_adapter_checkpoint])
        else:
            neural_report = read_json(neural_report_path, {}) if neural_report_path else {}
    # Preview work stays bounded to the five already-selected scenes.  Global
    # mastering and full-track song analysis belong to build_full_application.
    global_reference: Path | None = None
    global_profile: dict[str, Any] | None = None
    global_profile_path: Path | None = None
    global_apply_report_path: Path | None = None
    global_profile_paths: dict[str, str] = {}
    ru_voice = model_root / "model_ru_voice.flac"
    passthrough_preview = applied_routing.get("recommended_model") == "passthrough"
    if passthrough_preview:
        _copy_audio(selected_dubbed_speech, ru_voice)
    elif not _cached_model_output_valid(ru_voice, checkpoint):
        ctx.update(progress=82.0, local_progress=0.0)
        _run_subtractor(
            store, selected_dubbed_speech, selected_original_speech, ru_voice,
            checkpoint, ctx, pair["name"], "Модель: удаление EN из речевого стема",
        )
        _record_model_output(ru_voice, checkpoint)
    algorithmic_ru_voice = model_root / "model_ru_voice_algorithmic_reference.flac"
    if passthrough_preview:
        _copy_audio(selected_dubbed_speech, algorithmic_ru_voice)
    elif not _cached_model_output_valid_for(
        algorithmic_ru_voice,
        [checkpoint, algorithmic_reference],
    ):
        ctx.update(progress=84.0, local_progress=0.0)
        _run_subtractor(
            store,
            selected_dubbed_speech,
            algorithmic_reference,
            algorithmic_ru_voice,
            checkpoint,
            ctx,
            pair["name"],
            "Модель: удаление EN с адаптированным референсом",
        )
        _record_model_output_for(
            algorithmic_ru_voice,
            [checkpoint, algorithmic_reference],
        )
    neural_ru_voice: Path | None = None
    if neural_adapter_checkpoint is not None and neural_reference is not None:
        neural_ru_voice = model_root / "model_ru_voice_neural_reference.flac"
        if passthrough_preview:
            _copy_audio(selected_dubbed_speech, neural_ru_voice)
        elif not _cached_model_output_valid_for(
            neural_ru_voice, [checkpoint, neural_adapter_checkpoint]
        ):
            ctx.update(progress=85.0, local_progress=0.0)
            _run_subtractor(
                store,
                selected_dubbed_speech,
                neural_reference,
                neural_ru_voice,
                checkpoint,
                ctx,
                pair["name"],
                "Модель: удаление EN с автоматически подогнанным референсом",
            )
            _record_model_output_for(neural_ru_voice, [checkpoint, neural_adapter_checkpoint])
    global_ru_voice: Path | None = None
    if global_reference is not None and global_profile_path is not None:
        global_ru_voice = model_root / "model_ru_voice_global_reference.flac"
        global_output_dependencies = [checkpoint, global_profile_path]
        if passthrough_preview:
            _copy_audio(selected_dubbed_speech, global_ru_voice)
        elif not _cached_model_output_valid_for(
            global_ru_voice, global_output_dependencies
        ):
            ctx.update(progress=85.5, local_progress=0.0)
            _run_subtractor(
                store,
                selected_dubbed_speech,
                global_reference,
                global_ru_voice,
                checkpoint,
                ctx,
                pair["name"],
                "Модель: удаление EN с глобально подогнанным референсом",
            )
            _record_model_output_for(global_ru_voice, global_output_dependencies)
    direct = model_root / "model_direct_dirty_mix.flac"
    if passthrough_preview:
        _copy_audio(selected_dubbed_mix, direct)
    elif not _cached_model_output_valid(direct, checkpoint):
        ctx.update(progress=86.0, local_progress=0.0)
        _run_subtractor(
            store, selected_dubbed_mix, selected_original_mix, direct,
            checkpoint, ctx, pair["name"], "Удаление EN из полного звука",
        )
        _record_model_output(direct, checkpoint)

    coincident_parameters_path = (
        model_root / "coincident_speech_protection_parameters.json"
    )
    _atomic_json_if_changed(
        coincident_parameters_path,
        coincident_identity,
    )
    coincident_module_path = Path(
        str(coincident_speech_protection.__file__)
    ).resolve()
    integrity_parameters_path = (
        model_root / "speech_integrity_guard_parameters.json"
    )
    _atomic_json_if_changed(
        integrity_parameters_path,
        integrity_identity,
    )
    integrity_module_path = Path(
        str(speech_integrity_guard.__file__)
    ).resolve()
    preview_analysis_ranges = [
        {
            "scope_index": index,
            "start_sec": left / rate,
            "end_sec": right / rate,
        }
        for index, (left, right) in enumerate(selected_ranges)
    ]
    raw_first_pass_voices: dict[str, Path] = {
        "selected": ru_voice,
        "algorithmic": algorithmic_ru_voice,
        **({"neural": neural_ru_voice} if neural_ru_voice is not None else {}),
        **({"global": global_ru_voice} if global_ru_voice is not None else {}),
    }
    first_pass_references: dict[str, Path] = {
        "selected": selected_original_speech,
        "algorithmic": algorithmic_reference,
        **({"neural": neural_reference} if neural_reference is not None else {}),
        **({"global": global_reference} if global_reference is not None else {}),
    }
    protected_first_pass_voices: dict[str, Path] = {}
    coincident_protected_first_pass_voices: dict[str, Path] = {}
    coincident_preview_reports: dict[str, dict[str, Any]] = {}
    integrity_preview_reports: dict[str, dict[str, Any]] = {}
    primary_integrity_report_data: dict[str, dict[str, Any]] = {}
    second_integrity_report_data: dict[str, dict[str, Any]] = {}
    for index, (variant_name, raw_voice) in enumerate(
        raw_first_pass_voices.items()
    ):
        reference_voice = first_pass_references[variant_name]
        protected_voice = raw_voice.with_name(
            f"{raw_voice.stem}_coincident_protected.flac"
        )
        protection_report_path = raw_voice.with_name(
            f"coincident_speech_protection_{variant_name}.json"
        )
        protection_input_dependencies = [
            raw_voice,
            selected_dubbed_speech,
            reference_voice,
            coincident_parameters_path,
        ]
        cached_protection_report = _load_valid_coincident_report(
            protection_report_path,
            selected_dubbed_speech,
            reference_voice,
            raw_voice,
            coincident_cfg,
        )
        protection_dependencies = [
            *protection_input_dependencies,
            protection_report_path,
        ]
        if (
            cached_protection_report is None
            or not _cached_model_output_valid_for(
                protected_voice, protection_dependencies
            )
        ):
            protection_report = (
                coincident_speech_protection.protect_coincident_speech(
                    selected_dubbed_speech,
                    reference_voice,
                    raw_voice,
                    protected_voice,
                    coincident_cfg,
                    analysis_intervals=preview_analysis_ranges,
                    report_path=protection_report_path,
                    ctx=_ProgressRangeContext(
                        ctx,
                        86.1 + min(index, 2) * 0.2,
                        0.4,
                    ),
                )
            )
            _record_model_output_for(
                protected_voice, protection_dependencies
            )
        else:
            protection_report = cached_protection_report
        coincident_protected_first_pass_voices[variant_name] = protected_voice
        coincident_preview_reports[variant_name] = {
            "report": str(protection_report_path.resolve()),
            "summary": protection_report.get("summary") or {},
        }
        integrity_voice = protected_voice.with_name(
            f"{protected_voice.stem}_integrity_protected.flac"
        )
        integrity_report_path = protected_voice.with_name(
            f"speech_integrity_primary_{variant_name}.json"
        )
        integrity_input_dependencies = [
            protected_voice,
            selected_dubbed_speech,
            selected_original_mix,
            selected_original_speech,
            reference_voice,
            integrity_parameters_path,
            integrity_module_path,
        ]
        cached_integrity_report = _load_integrity_report(
            integrity_report_path,
            "post_primary_pass",
        )
        integrity_dependencies = [
            *integrity_input_dependencies,
            integrity_report_path,
        ]
        if (
            cached_integrity_report is None
            or not _cached_model_output_valid_for(
                integrity_voice,
                integrity_dependencies,
            )
        ):
            primary_integrity_report = speech_integrity_guard.guard_primary_pass(
                selected_dubbed_speech,
                protected_voice,
                selected_original_mix,
                selected_original_speech,
                reference_voice,
                integrity_voice,
                integrity_cfg,
                analysis_intervals=preview_analysis_ranges,
                report_path=integrity_report_path,
                ctx=_ProgressRangeContext(
                    ctx,
                    86.35 + min(index, 2) * 0.18,
                    0.35,
                ),
            )
            _record_model_output_for(
                integrity_voice,
                integrity_dependencies,
            )
        else:
            primary_integrity_report = cached_integrity_report
        protected_first_pass_voices[variant_name] = integrity_voice
        primary_integrity_report_data[variant_name] = primary_integrity_report
        integrity_preview_reports[variant_name] = {
            "primary_report": str(integrity_report_path.resolve()),
            "primary_summary": primary_integrity_report.get("summary") or {},
        }

    ru_voice = protected_first_pass_voices["selected"]
    algorithmic_ru_voice = protected_first_pass_voices["algorithmic"]
    neural_ru_voice = protected_first_pass_voices.get("neural")
    global_ru_voice = protected_first_pass_voices.get("global")

    second_pass_voices: dict[str, Path] = {}
    raw_second_pass_voices: dict[str, Path] = {}
    coincident_protected_second_pass_voices: dict[str, Path] = {}
    if speech_second_pass:
        second_pass_sources: tuple[tuple[str, Path | None], ...] = (
            ("selected", ru_voice),
            ("algorithmic", algorithmic_ru_voice),
            ("neural", neural_ru_voice),
            ("global", global_ru_voice),
        )
        for index, (variant_name, source_voice) in enumerate(second_pass_sources):
            if source_voice is None or not source_voice.is_file():
                continue
            raw_output_voice = model_root / (
                "model_ru_voice_second_pass.flac"
                if variant_name == "selected"
                else f"model_ru_voice_{variant_name}_second_pass.flac"
            )
            dependencies = [source_voice, *speech_second_pass_dependencies]
            if not _cached_model_output_valid_for(
                raw_output_voice, dependencies
            ):
                pipeline._run_speech_extractor(
                    store,
                    source_voice,
                    raw_output_voice,
                    ctx,
                    pair["name"],
                    "Повторный проход модуля извлечения речи",
                    progress_base=86.5 + min(index, 2) * 0.5,
                    progress_span=1.0,
                    model_role="original",
                )
                _record_model_output_for(raw_output_voice, dependencies)
            raw_second_pass_voices[variant_name] = raw_output_voice
            reference_voice = first_pass_references[variant_name]
            protected_output_voice = raw_output_voice.with_name(
                f"{raw_output_voice.stem}_coincident_protected.flac"
            )
            second_report_path = raw_output_voice.with_name(
                "coincident_speech_protection_"
                f"{variant_name}_second_pass.json"
            )
            protection_input_dependencies = [
                raw_output_voice,
                selected_dubbed_speech,
                reference_voice,
                coincident_parameters_path,
                coincident_module_path,
            ]
            cached_second_report = _load_valid_coincident_report(
                second_report_path,
                selected_dubbed_speech,
                reference_voice,
                raw_output_voice,
                coincident_cfg,
            )
            protection_dependencies = [
                *protection_input_dependencies,
                second_report_path,
            ]
            if (
                cached_second_report is None
                or not _cached_model_output_valid_for(
                    protected_output_voice,
                    protection_dependencies,
                )
            ):
                second_report = (
                    coincident_speech_protection.protect_coincident_speech(
                        selected_dubbed_speech,
                        reference_voice,
                        raw_output_voice,
                        protected_output_voice,
                        coincident_cfg,
                        analysis_intervals=preview_analysis_ranges,
                        report_path=second_report_path,
                        ctx=_ProgressRangeContext(ctx, 88.0, 0.4),
                    )
                )
                _record_model_output_for(
                    protected_output_voice,
                    protection_dependencies,
                )
            else:
                second_report = cached_second_report
            coincident_preview_reports[
                f"{variant_name}_second_pass"
            ] = {
                "report": str(second_report_path.resolve()),
                "summary": second_report.get("summary") or {},
            }
            integrity_output_voice = protected_output_voice.with_name(
                f"{protected_output_voice.stem}_integrity_protected.flac"
            )
            coincident_protected_second_pass_voices[
                variant_name
            ] = protected_output_voice
            integrity_second_report_path = protected_output_voice.with_name(
                f"speech_integrity_second_pass_{variant_name}.json"
            )
            integrity_second_input_dependencies = [
                source_voice,
                protected_output_voice,
                integrity_parameters_path,
                integrity_module_path,
            ]
            cached_integrity_second_report = _load_integrity_report(
                integrity_second_report_path,
                "post_second_pass",
            )
            integrity_second_dependencies = [
                *integrity_second_input_dependencies,
                integrity_second_report_path,
            ]
            if (
                cached_integrity_second_report is None
                or not _cached_model_output_valid_for(
                    integrity_output_voice,
                    integrity_second_dependencies,
                )
            ):
                second_integrity_report = (
                    speech_integrity_guard.guard_second_pass(
                        source_voice,
                        protected_output_voice,
                        integrity_output_voice,
                        integrity_cfg,
                        analysis_intervals=preview_analysis_ranges,
                        report_path=integrity_second_report_path,
                        ctx=_ProgressRangeContext(
                            ctx,
                            88.35 + min(index, 2) * 0.18,
                            0.35,
                        ),
                    )
                )
                _record_model_output_for(
                    integrity_output_voice,
                    integrity_second_dependencies,
                )
            else:
                second_integrity_report = cached_integrity_second_report
            second_integrity_report_data[variant_name] = second_integrity_report
            integrity_preview_reports[variant_name].update(
                {
                    "second_pass_report": str(
                        integrity_second_report_path.resolve()
                    ),
                    "second_pass_summary": (
                        second_integrity_report.get("summary") or {}
                    ),
                }
            )
            second_pass_voices[variant_name] = integrity_output_voice

    _save_application_preparation_phase(
        store,
        project_id,
        pair_id,
        preparation_identity,
        "subtraction",
        {
            "model_ru_voice": ru_voice,
            "model_ru_voice_algorithmic": algorithmic_ru_voice,
            "model_ru_voice_before_coincident_protection": (
                raw_first_pass_voices["selected"]
            ),
            "model_ru_voice_algorithmic_before_coincident_protection": (
                raw_first_pass_voices["algorithmic"]
            ),
            "model_ru_voice_before_integrity_guard": (
                coincident_protected_first_pass_voices["selected"]
            ),
            "model_ru_voice_algorithmic_before_integrity_guard": (
                coincident_protected_first_pass_voices["algorithmic"]
            ),
            "direct_dirty_result": direct,
            **{
                f"model_ru_voice_{name}_second_pass": path
                for name, path in second_pass_voices.items()
            },
            **{
                f"model_ru_voice_{name}_second_pass_before_coincident_protection": path
                for name, path in raw_second_pass_voices.items()
            },
            **{
                f"model_ru_voice_{name}_second_pass_before_integrity_guard": path
                for name, path in coincident_protected_second_pass_voices.items()
            },
        },
    )
    ctx.update(progress=90.0, local_progress=0.0)
    if background_checkpoint is not None:
        background_root = (
            scan_root
            / "background_results"
            / (
                f"selection_v{SCENE_SELECTION_VERSION}"
                f"_song_{song_preview_cache_token}"
            )
            / background_checkpoint.stem
        )
        background_root.mkdir(parents=True, exist_ok=True)
        instrumental = background_root / "learned_music_effects.flac"
        background_dependencies = [
            background_checkpoint,
            selected_original_song_mix,
            selected_original_song_speech,
        ]
        if not _cached_model_output_valid_for(
            instrumental,
            background_dependencies,
        ):
            _run_background_restorer(
                store,
                selected_original_song_mix,
                selected_original_song_speech,
                instrumental,
                background_checkpoint,
                ctx,
                pair["name"],
                "Восстановление музыки и эффектов",
            )
            _record_model_output_for(
                instrumental,
                background_dependencies,
            )
    else:
        separator_root = (
            scan_root
            / "specialized_separator"
            / (
                f"selection_v{SCENE_SELECTION_VERSION}"
                f"_song_{song_preview_cache_token}"
            )
        )
        separated = _run_music_separator(
            store,
            selected_original_song_mix,
            separator_root,
            ctx,
            pair["name"],
        )
        instrumental = separated["instrumental"]
    _save_application_preparation_phase(
        store,
        project_id,
        pair_id,
        preparation_identity,
        "background",
        {"specialized_me": instrumental},
    )
    me_data, me_rate = sf.read(str(instrumental), dtype="float32", always_2d=True)
    ru_data, ru_rate = sf.read(str(ru_voice), dtype="float32", always_2d=True)
    length = min(
        len(me_data),
        int(round(len(ru_data) * me_rate / max(ru_rate, 1))),
    )
    me_data = me_data[:length]

    preview_mix_balance: dict[str, Any] = {"schema_version": 2, "variants": {}}

    def _write_balanced_preview_mix(
        voice_path: Path,
        output_path: Path,
        variant_name: str,
    ) -> dict[str, Any]:
        voice_data, voice_rate = sf.read(
            str(voice_path), dtype="float32", always_2d=True
        )
        voice_data = pipeline._resample_exact(
            voice_data,
            voice_rate,
            me_rate,
            length,
        )
        balance = _calculate_mix_balance(
            instrumental,
            selected_original_speech,
            voice_path,
            selected_original_mix,
            dubbed_speech_path=selected_dubbed_speech,
            dubbed_mix_path=selected_dubbed_mix,
        )
        mixed = (
            me_data * _linear_gain(float(balance.get("background_gain_db", 0.0)))
            + audio_io.match_channels(voice_data, me_data.shape[1])
            * _linear_gain(float(balance.get("voice_gain_db", 0.0)))
        )
        _write(output_path, mixed, me_rate)
        preview_mix_balance["variants"][variant_name] = balance
        return balance

    rebuilt_path = model_root / "model_ru_plus_specialized_me.flac"
    _write_balanced_preview_mix(ru_voice, rebuilt_path, "selected")
    algorithmic_rebuilt_path = model_root / "model_ru_algorithmic_ref_plus_specialized_me.flac"
    _write_balanced_preview_mix(
        algorithmic_ru_voice,
        algorithmic_rebuilt_path,
        "algorithmic",
    )
    neural_rebuilt_path: Path | None = None
    if neural_ru_voice is not None and neural_ru_voice.is_file():
        neural_rebuilt_path = model_root / "model_ru_neural_ref_plus_specialized_me.flac"
        _write_balanced_preview_mix(
            neural_ru_voice,
            neural_rebuilt_path,
            "neural",
        )
    global_rebuilt_path: Path | None = None
    if global_ru_voice is not None and global_ru_voice.is_file():
        global_rebuilt_path = model_root / "model_ru_global_ref_plus_specialized_me.flac"
        _write_balanced_preview_mix(
            global_ru_voice,
            global_rebuilt_path,
            "global",
        )

    second_pass_rebuilt: dict[str, Path] = {}
    for variant_name, variant_voice in second_pass_voices.items():
        output_mix = model_root / (
            "model_ru_second_pass_plus_specialized_me.flac"
            if variant_name == "selected"
            else f"model_ru_{variant_name}_ref_second_pass_plus_specialized_me.flac"
        )
        _write_balanced_preview_mix(
            variant_voice,
            output_mix,
            f"{variant_name}_second_pass",
        )
        second_pass_rebuilt[variant_name] = output_mix

    synchronized_ru_voice = model_root / "model_ru_voice_synchronized.flac"
    synchronization = synchronize_voice_file(
        ru_voice,
        selected_original_speech,
        synchronized_ru_voice,
        maximum_advance_sec=float(
            synchronization_cfg.get("maximum_advance_sec", 0.85)
        ),
        preservation_intervals=_guard_preservation_intervals(
            primary_integrity_report_data.get("selected")
        ),
        crossfade_sec=float(
            synchronization_cfg.get("crossfade_sec", 0.015)
        ),
    )
    synchronized_rebuilt_path = (
        model_root / "model_ru_synchronized_plus_specialized_me.flac"
    )
    _write_balanced_preview_mix(
        synchronized_ru_voice,
        synchronized_rebuilt_path,
        "synchronized",
    )
    atomic_json(model_root / "speech_synchronization.json", synchronization)
    atomic_json(model_root / "mix_balance.json", preview_mix_balance)

    # The editor may combine conservative timing with a prepared reference.
    # Build the matching synchronized voice for every available subtraction
    # variant; otherwise the browser has to choose between the selected
    # reference adapter and synchronization, and conservative mode can appear
    # enabled while playing an unsynchronized voice file.
    synchronized_variants: dict[str, Path] = {}
    for variant_name, variant_voice in (
        ("algorithmic", algorithmic_ru_voice),
        ("neural", neural_ru_voice),
        ("global", global_ru_voice),
    ):
        if variant_voice is None or not variant_voice.is_file():
            continue
        variant_output = model_root / f"model_ru_voice_{variant_name}_synchronized.flac"
        variant_report = synchronize_voice_file(
            variant_voice,
            selected_original_speech,
            variant_output,
            maximum_advance_sec=float(
                synchronization_cfg.get("maximum_advance_sec", 0.85)
            ),
            preservation_intervals=_guard_preservation_intervals(
                primary_integrity_report_data.get(variant_name)
            ),
            crossfade_sec=float(
                synchronization_cfg.get("crossfade_sec", 0.015)
            ),
        )
        atomic_json(
            model_root / f"speech_synchronization_{variant_name}.json",
            variant_report,
        )
        synchronized_variants[variant_name] = variant_output

    synchronized_second_pass_variants: dict[str, Path] = {}
    synchronized_second_pass_rebuilt: dict[str, Path] = {}
    for variant_name, variant_voice in second_pass_voices.items():
        variant_output = model_root / (
            "model_ru_voice_second_pass_synchronized.flac"
            if variant_name == "selected"
            else f"model_ru_voice_{variant_name}_second_pass_synchronized.flac"
        )
        variant_report = synchronize_voice_file(
            variant_voice,
            selected_original_speech,
            variant_output,
            maximum_advance_sec=float(
                synchronization_cfg.get("maximum_advance_sec", 0.85)
            ),
            preservation_intervals=_guard_preservation_intervals(
                primary_integrity_report_data.get(variant_name),
                second_integrity_report_data.get(variant_name),
            ),
            crossfade_sec=float(
                synchronization_cfg.get("crossfade_sec", 0.015)
            ),
        )
        atomic_json(
            model_root / f"speech_synchronization_{variant_name}_second_pass.json",
            variant_report,
        )
        synchronized_second_pass_variants[variant_name] = variant_output
        output_mix = model_root / (
            "model_ru_second_pass_synchronized_plus_specialized_me.flac"
            if variant_name == "selected"
            else f"model_ru_{variant_name}_ref_second_pass_synchronized_plus_specialized_me.flac"
        )
        _write_balanced_preview_mix(
            variant_output,
            output_mix,
            f"{variant_name}_second_pass_synchronized",
        )
        synchronized_second_pass_rebuilt[variant_name] = output_mix

    scene_files = {
        "original_mix": selected_original_mix,
        "original_song_mix": selected_original_song_mix,
        "dubbed_mix": selected_dubbed_mix,
        "original_en_speech": selected_original_speech,
        "original_song_speech": selected_original_song_speech,
        "algorithmic_en_speech": algorithmic_reference,
        "dubbed_en_ru_speech": selected_dubbed_speech,
        "model_ru_voice": ru_voice,
        "model_ru_voice_algorithmic": algorithmic_ru_voice,
        "model_ru_voice_before_coincident_protection": (
            raw_first_pass_voices["selected"]
        ),
        "model_ru_voice_algorithmic_before_coincident_protection": (
            raw_first_pass_voices["algorithmic"]
        ),
        "model_ru_voice_before_integrity_guard": (
            coincident_protected_first_pass_voices["selected"]
        ),
        "model_ru_voice_algorithmic_before_integrity_guard": (
            coincident_protected_first_pass_voices["algorithmic"]
        ),
        "direct_dirty_result": direct,
        "specialized_me": instrumental,
        "rebuilt_result": rebuilt_path,
        "algorithmic_rebuilt_result": algorithmic_rebuilt_path,
        "synchronized_ru_voice": synchronized_ru_voice,
        "synchronized_rebuilt_result": synchronized_rebuilt_path,
    }
    if neural_reference is not None and neural_reference.is_file():
        scene_files["neural_en_speech"] = neural_reference
    if neural_ru_voice is not None and neural_ru_voice.is_file():
        scene_files["model_ru_voice_neural"] = neural_ru_voice
        scene_files[
            "model_ru_voice_neural_before_coincident_protection"
        ] = raw_first_pass_voices["neural"]
        scene_files[
            "model_ru_voice_neural_before_integrity_guard"
        ] = coincident_protected_first_pass_voices["neural"]
    if neural_rebuilt_path is not None and neural_rebuilt_path.is_file():
        scene_files["neural_rebuilt_result"] = neural_rebuilt_path
    if global_reference is not None and global_reference.is_file():
        scene_files["global_en_speech"] = global_reference
    if global_ru_voice is not None and global_ru_voice.is_file():
        scene_files["model_ru_voice_global"] = global_ru_voice
        scene_files[
            "model_ru_voice_global_before_coincident_protection"
        ] = raw_first_pass_voices["global"]
        scene_files[
            "model_ru_voice_global_before_integrity_guard"
        ] = coincident_protected_first_pass_voices["global"]
    if global_rebuilt_path is not None and global_rebuilt_path.is_file():
        scene_files["global_rebuilt_result"] = global_rebuilt_path
    for variant_name, variant_output in synchronized_variants.items():
        scene_files[f"synchronized_ru_voice_{variant_name}"] = variant_output
    for variant_name, variant_output in second_pass_voices.items():
        key = (
            "model_ru_voice_second_pass"
            if variant_name == "selected"
            else f"model_ru_voice_{variant_name}_second_pass"
        )
        scene_files[key] = variant_output
        raw_key = (
            "model_ru_voice_second_pass_before_coincident_protection"
            if variant_name == "selected"
            else (
                f"model_ru_voice_{variant_name}_second_pass_"
                "before_coincident_protection"
            )
        )
        scene_files[raw_key] = raw_second_pass_voices[variant_name]
        before_integrity_key = (
            "model_ru_voice_second_pass_before_integrity_guard"
            if variant_name == "selected"
            else (
                f"model_ru_voice_{variant_name}_second_pass_"
                "before_integrity_guard"
            )
        )
        scene_files[before_integrity_key] = (
            coincident_protected_second_pass_voices[variant_name]
        )
    for variant_name, variant_output in second_pass_rebuilt.items():
        key = (
            "second_pass_rebuilt_result"
            if variant_name == "selected"
            else f"{variant_name}_second_pass_rebuilt_result"
        )
        scene_files[key] = variant_output
    for variant_name, variant_output in synchronized_second_pass_variants.items():
        key = (
            "synchronized_ru_voice_second_pass"
            if variant_name == "selected"
            else f"synchronized_ru_voice_{variant_name}_second_pass"
        )
        scene_files[key] = variant_output
    for variant_name, variant_output in synchronized_second_pass_rebuilt.items():
        key = (
            "synchronized_second_pass_rebuilt_result"
            if variant_name == "selected"
            else f"synchronized_{variant_name}_second_pass_rebuilt_result"
        )
        scene_files[key] = variant_output
    scenes = _split_demo_files(
        root, selected, scene_files, selected_ranges, rate, semantic_mode
    )
    coincident_scene_reports: list[dict[str, Any]] = []
    for scene_index, (scene, (left, right)) in enumerate(
        zip(scenes, selected_ranges), 1
    ):
        batch_start = left / rate
        batch_end = right / rate
        variants: dict[str, Any] = {}
        for variant_name, state in coincident_preview_reports.items():
            source_report = read_json(Path(state["report"]), {}) or {}

            def localised(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
                result: list[dict[str, Any]] = []
                for item in items:
                    start = max(batch_start, float(item["start_sec"]))
                    end = min(batch_end, float(item["end_sec"]))
                    if end <= start:
                        continue
                    result.append(
                        {
                            **item,
                            "batch_start_sec": item["start_sec"],
                            "batch_end_sec": item["end_sec"],
                            "start_sec": round(start - batch_start, 6),
                            "end_sec": round(end - batch_start, 6),
                        }
                    )
                return result

            variants[variant_name] = {
                "report": state["report"],
                "summary": state["summary"],
                "confirmed_intervals": localised(
                    source_report.get("confirmed_intervals") or []
                ),
                "rejected_candidates": localised(
                    source_report.get("rejected_candidates") or []
                ),
            }
        scene_report_path = (
            root
            / "application"
            / "previews"
            / f"scene_{scene_index:02d}"
            / "coincident_speech_protection.json"
        )
        scene_report = {
            "schema_version": int(
                coincident_speech_protection.SCHEMA_VERSION
            ),
            "report_kind": (
                "dubclean_preview_coincident_speech_protection"
            ),
            "scene_id": scene["id"],
            "variants": variants,
        }
        atomic_json(scene_report_path, scene_report)
        scene["coincident_speech_protection"] = {
            "report": str(scene_report_path.resolve()),
            "confirmed_count": sum(
                len(item["confirmed_intervals"])
                for item in variants.values()
            ),
        }
        coincident_scene_reports.append(
            scene["coincident_speech_protection"]
        )
    preview_coincident_report_path = (
        root / "application" / "preview_coincident_speech_protection_report.json"
    )
    preview_coincident_report = {
        "schema_version": int(
            coincident_speech_protection.SCHEMA_VERSION
        ),
        "report_kind": "dubclean_preview_coincident_speech_protection",
        "algorithm": coincident_speech_protection.ALGORITHM,
        "analysis_scope": "selected_preview_scenes_only",
        "config": coincident_identity["config"],
        "variants": coincident_preview_reports,
        "scenes": coincident_scene_reports,
    }
    atomic_json(
        preview_coincident_report_path,
        preview_coincident_report,
    )
    preview_song_report = _protect_preview_song_scenes(
        root,
        scenes,
        song_cfg,
        ctx,
        pair["name"],
    )
    speech_scenes = [
        scene
        for scene in scenes
        if scene.get("preview_kind") != "song_candidate"
    ]
    adapter_selection = _select_reference_adapter_from_preview(
        speech_scenes,
        algorithmic_report,
    )
    _make_video_previews(store, pair, scenes, ctx)
    vad_review_report = _write_vad_preview_report(root, speech_scenes)
    manifest = {
        "schema_version": 8,
        "processing_scope": "speech_and_song_scenes",
        "speech_extraction_second_pass": {
            "prepared": bool(second_pass_voices),
            "enabled_by_default": bool(speech_second_pass),
        },
        "created_at": utc_now(),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "background_checkpoint": (
            str(background_checkpoint) if background_checkpoint is not None else ""
        ),
        "background_checkpoint_name": (
            background_checkpoint.name if background_checkpoint is not None else ""
        ),
        "model_kind": "semantic_ru_separator" if semantic_mode else "phase_subtractor",
        "separator_model": (
            background_checkpoint.name
            if background_checkpoint is not None
            else store.cfg["music_separation"]["model_name"]
        ),
        "scene_duration_sec": vad_preview.PREFERRED_PREVIEW_SEC,
        "scene_count": len(scenes),
        "speech_scene_count": len(speech_scenes),
        "song_scene_count": sum(
            1
            for scene in scenes
            if bool((scene.get("song_protection") or {}).get("protected"))
        ),
        "song_analysis_scope": "selected_preview_scenes_only",
        "coincident_speech_protection": {
            "enabled": bool(
                coincident_identity["config"].get("enabled")
            ),
            "algorithm": coincident_speech_protection.ALGORITHM,
            "analysis_scope": "selected_preview_scenes_only",
            "report": str(preview_coincident_report_path.resolve()),
            "variants": coincident_preview_reports,
        },
        "speech_integrity_guard": {
            "enabled": bool(integrity_identity["config"].get("enabled")),
            "algorithm": speech_integrity_guard.ALGORITHM,
            "schema_version": int(speech_integrity_guard.SCHEMA_VERSION),
            "analysis_scope": "selected_preview_scenes_only",
            "parameters": str(integrity_parameters_path.resolve()),
            "variants": integrity_preview_reports,
        },
        "scene_selection_policy": (
            "Silero VAD 6.2.0 локально анализирует пять фиксированных зон: начало, "
            "25%, 50%, 75% и 90%. Внутри каждой зоны выбран стабильный участок с "
            "максимальной долей речи; позиционный алгоритм используется "
            "только как резервный выбор. "
            "EfficientAT и защита песен проверяют только эти выбранные сцены. "
            "Полный анализ песен выполняется при финальной сборке."
        ),
        "vad": {
            "engine": "silero-vad",
            "version": vad_preview.SILERO_VAD_VERSION,
            "model_loading": "Локальный файл silero_vad/data/silero_vad.jit из установленного пакета.",
            "parameters": vad_preview.VAD_PARAMETERS,
            "review_report": str(vad_review_report.resolve()),
        },
        "alignment": alignment.get("summary") or {},
        "song_protection": {
            **preview_song_report,
            "dynamic_mix_contract": {
                "endpoint": "/api/preview-mix",
                "enabled_parameter": "song_protection=1",
                "before_parameter": "song_protection=0",
                "report_parameter": "song_report",
                "original_parameter": "song_original",
                "dubbed_parameter": "song_dubbed",
                "detection_voice_parameter": "song_detection_voice",
                "translation_voice_parameter": "song_translation_voice",
                "sync_mode_parameter": "sync_mode",
                "sync_modes": ["off", "manual", "conservative"],
                "report_schema_version": int(
                    song_protection.PREVIEW_CONTRACT_SCHEMA_VERSION
                ),
            },
        },
        "reference_compatibility": applied_routing,
        "mix_balance": {
            "report": str((model_root / "mix_balance.json").resolve()),
            "variants": preview_mix_balance["variants"],
        },
        "speech_synchronization_recommendation": synchronization,
        "scenes": scenes,
        "feasibility": _feasibility_verdict(speech_scenes, semantic_mode),
        "methods": {
            "speech_rebuild": (
                "DubClean Voice отделяет перевод; затем добавляются "
                "восстановленные музыка и эффекты"
                if background_checkpoint is not None
                else "Разделение EN+RU → RU; затем RU + музыка и эффекты"
            ),
            "reference_adapter_algorithmic": (
                "Сверяет уровень только общей английской речи. "
                "При низкой уверенности сохраняет исходный референс."
            ),
            "reference_adapter_neural": (
                "Автоматическая подгонка меняет только частотную "
                "окраску и уровень английского референса перед вычитанием."
            ),
            "reference_adapter_global": (
                "Выравнивание качества по участкам без речи оценивает "
                "единый профиль качества/эквализации и применяет его ко всей "
                "английской речевой дорожке перед вычитанием."
            ),
            "direct_mix": "Разделение речи прямо из полного звука",
            "speech_synchronization": (
                "Опционально: границы реплик из нейросетевых речевых слоёв; "
                "русский голос сдвигается раньше не более чем на 0,85 с без "
                "изменения скорости и тембра."
            ),
        },
        "reference_adapter": {
            "algorithmic": algorithmic_report,
            "preview_report": str(algorithmic_report_path.resolve()),
            "neural": neural_report or {},
            "neural_checkpoint": (
                str(neural_adapter_checkpoint.resolve())
                if neural_adapter_checkpoint is not None
                else ""
            ),
            "neural_preview_report": (
                str(neural_report_path.resolve())
                if neural_report_path is not None
                else ""
            ),
            "global": global_profile or {},
            "global_checkpoint": (
                str(global_adapter_checkpoint.resolve())
                if global_adapter_checkpoint is not None
                else ""
            ),
            "global_profile_report": (
                str(global_profile_path.resolve())
                if global_profile_path is not None
                else ""
            ),
            "global_preview_report": (
                str(global_apply_report_path.resolve())
                if global_apply_report_path is not None
                else ""
            ),
            "global_profile_inputs": global_profile_paths,
            "selection": adapter_selection,
        },
        "warning": (
            "Автоматическая метрика не заменяет прослушивание. Если корреляция "
            "английского референса после модели не уменьшилась хотя бы на 20%, "
            "сцена помечается как неэффективная."
        ),
    }
    manifest_path = root / "application" / "preview_manifest.json"
    atomic_json(manifest_path, manifest)
    pair = store.load_pair(project_id, pair_id)
    pair["application_preview"] = manifest
    pair["application_checkpoint"] = str(checkpoint)
    pair["application_background_checkpoint"] = (
        str(background_checkpoint) if background_checkpoint is not None else ""
    )
    pair["speech_extraction_second_pass"] = bool(speech_second_pass)
    store.save_pair(pair)
    _save_application_preparation_phase(
        store,
        project_id,
        pair_id,
        preparation_identity,
        "preview",
        {
            "rebuilt_result": rebuilt_path,
            "algorithmic_rebuilt_result": algorithmic_rebuilt_path,
            "synchronized_rebuilt_result": synchronized_rebuilt_path,
            **{
                f"{name}_second_pass_rebuilt_result": path
                for name, path in second_pass_rebuilt.items()
            },
            **{
                f"{name}_second_pass_synchronized_rebuilt_result": path
                for name, path in synchronized_second_pass_rebuilt.items()
            },
        },
    )
    ctx.update(
        progress=100.0,
        local_progress=100.0,
        stage="Превью готово",
        substage="",
    )
    return manifest


def _source_language(pair: dict[str, Any], role: str) -> str:
    """Language tag of the stream this role was extracted from.

    Guessing is worse than admitting ignorance here: a wrong tag makes players
    pick the wrong default track, while ``und`` simply leaves the choice open.
    """
    source = dict((pair.get("sources") or {}).get(role) or {})
    try:
        wanted = int(source.get("stream_index"))
    except (TypeError, ValueError):
        return "und"
    for stream in (source.get("probe") or {}).get("streams") or []:
        if int(stream.get("index", -1)) == wanted:
            language = str(stream.get("language") or "").strip()
            return language or "und"
    return "und"


def _inverted_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    """Turn a map onto the dubbed timeline into a map onto the original one.

    ``_align_to_dubbed`` reads a segment as "for output span
    [dubbed_start, dubbed_end) read the source from original_start onwards at
    speed_ratio".  The inverse swaps the two roles: the output span becomes the
    original one, the source position becomes the dubbed one, and the rate is
    reciprocal.  The manual correction is folded into the output span because
    it applies to the original side in the forward direction.
    """
    correction = float(alignment.get("manual_correction_sec") or 0.0)
    inverted: list[dict[str, Any]] = []
    for item in alignment.get("segments") or []:
        try:
            dubbed_start = float(item["dubbed_start"])
            dubbed_end = float(item["dubbed_end"])
            original_start = float(item["original_start"])
            raw_speed = item.get("speed_ratio")
            speed = float(1.0 if raw_speed is None else raw_speed)
        except (KeyError, TypeError, ValueError):
            continue
        if dubbed_end <= dubbed_start or not math.isfinite(speed) or speed <= 0.0:
            continue
        span = (dubbed_end - dubbed_start) * speed
        entry = dict(item)
        entry["dubbed_start"] = original_start + correction
        entry["dubbed_end"] = original_start + correction + span
        entry["original_start"] = dubbed_start
        entry["original_end"] = dubbed_end
        entry["duration"] = round(span, 6)
        entry["speed_ratio"] = 1.0 / speed
        inverted.append(entry)
    inverted.sort(key=lambda entry: float(entry["dubbed_start"]))
    return {
        "schema_version": alignment.get("schema_version"),
        "method": f"inverted_{alignment.get('method') or 'alignment'}",
        "summary": dict(alignment.get("summary") or {}),
        "manual_correction_sec": 0.0,
        "segments": inverted,
    }


def _alignment_coverage(alignment: dict[str, Any], duration: float) -> float:
    """Share of the output timeline a map can actually fill with real audio."""
    if duration <= 0.0:
        return 0.0
    covered = 0.0
    for item in alignment.get("segments") or []:
        if item.get("usable", True) is False:
            continue
        if not _alignment_segment_speed_is_safe(alignment, item):
            continue
        start = max(0.0, float(item.get("dubbed_start") or 0.0))
        end = min(duration, float(item.get("dubbed_end") or 0.0))
        if end > start:
            covered += end - start
    return min(1.0, covered / duration)


def _align_to_dubbed(
    source: Path,
    destination: Path,
    alignment: dict[str, Any],
    target_rate: int,
    target_duration: float,
    channels: int,
    ctx: Any,
    pair_name: str,
    stage: str,
    progress_base: float = 0.0,
    progress_span: float = 75.0,
) -> Path:
    segments = alignment.get("segments") or []
    correction = float(alignment.get("manual_correction_sec") or 0.0)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    writer = sf.SoundFile(
        str(temporary), "w", samplerate=target_rate, channels=channels,
        format="FLAC", subtype="PCM_16"
    )
    cursor = 0
    total_frames = int(round(target_duration * target_rate))
    chunk_frames = max(target_rate, int(round(target_rate * 60.0)))
    try:
        for index, item in enumerate(segments):
            start = max(0, int(round(float(item["dubbed_start"]) * target_rate)))
            stop = min(total_frames, int(round(float(item["dubbed_end"]) * target_rate)))
            if stop <= start:
                continue
            if start > cursor:
                writer.write(np.zeros((start - cursor, channels), dtype=np.float32))
                cursor = start
            if start < cursor:
                start = cursor
            raw_speed = item.get("speed_ratio")
            try:
                speed = float(1.0 if raw_speed is None else raw_speed)
            except (TypeError, ValueError):
                speed = float("nan")
            safe_speed = _alignment_segment_speed_is_safe(alignment, item)
            while start < stop:
                frames = min(chunk_frames, stop - start)
                if item.get("usable") and safe_speed:
                    source_start = float(item["original_start"]) + (
                        start / target_rate - float(item["dubbed_start"])
                    ) * speed + correction
                    values, source_rate = pipeline._read_interval(
                        source,
                        source_start,
                        frames / target_rate * speed,
                        None if channels == 1 else channels,
                    )
                    if channels == 1:
                        values = _mono(values)[:, None]
                    values = pipeline._resample_exact(
                        values, source_rate, target_rate, frames
                    )
                else:
                    values = np.zeros((frames, channels), dtype=np.float32)
                writer.write(np.clip(values, -1.0, 1.0))
                start += frames
                cursor += frames
                ctx.update(
                    stage=stage,
                    substage=(
                        f"Единая карта · {cursor / target_rate:.0f} "
                        f"из {target_duration:.0f} с"
                        if len(segments) == 1
                        else f"Участок {index + 1} из {len(segments)}"
                    ),
                    progress=min(
                        progress_base + progress_span,
                        progress_base
                        + cursor / max(total_frames, 1) * progress_span,
                    ),
                    local_progress=min(
                        100.0,
                        cursor / max(total_frames, 1) * 100.0,
                    ),
                    current_file=pair_name,
                )
        if cursor < total_frames:
            writer.write(np.zeros((total_frames - cursor, channels), dtype=np.float32))
    finally:
        writer.close()
    os.replace(temporary, destination)
    ctx.update(
        stage=stage,
        substage="Готово",
        progress=progress_base + progress_span,
        local_progress=100.0,
        current_file=pair_name,
    )
    return destination


def build_full_application(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    ctx.update(
        force=True,
        stage="Сборка полного фильма",
        substage="Проверка подготовленных материалов",
        progress=0.0,
        local_progress=0.0,
    )
    pair = store.load_pair(project_id, pair_id)
    applied_routing = _applied_reference_routing(pair)
    routing_document = _routing_document(applied_routing)
    if not pair.get("application_preview"):
        raise RuntimeError("Сначала подготовьте и прослушайте превью.")
    force_recompute = bool(parameters.get("force"))
    blocked = _full_run_blocked(
        pair.get("application_preview"), force_recompute
    )
    if blocked:
        raise RuntimeError(blocked)
    method = str(parameters.get("method") or "")
    if method not in {"voice_only", "speech_rebuild", "direct_mix"}:
        raise ValueError("Выберите доступный способ сборки звука.")
    balance_final_mix = (
        bool(parameters.get("balance_final_mix", True))
        and method == "speech_rebuild"
    )
    checkpoint = Path(str(pair.get("application_checkpoint") or "")).resolve()
    if not checkpoint.is_file():
        raise RuntimeError("Сохранение обученной модели больше не найдено.")
    background_checkpoint_value = str(
        pair.get("application_background_checkpoint") or ""
    )
    background_checkpoint = (
        Path(background_checkpoint_value).resolve()
        if background_checkpoint_value
        else None
    )
    root = store.pair_dir(project_id, pair_id)
    alignment = pipeline._processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )
    dubbed_path = root / "extracted" / "dubbed.flac"
    original_path = root / "extracted" / "original.flac"
    dubbed_info = sf.info(str(dubbed_path))
    rate = int(dubbed_info.samplerate)
    duration = float(dubbed_info.duration)
    synchronize_speech = bool(parameters.get("synchronize_speech", False)) and method in {
        "voice_only",
        "speech_rebuild",
    }
    speech_second_pass = bool(
        parameters.get(
            "speech_extraction_second_pass",
            pair.get("speech_extraction_second_pass", False),
        )
    ) and method in {"voice_only", "speech_rebuild"}
    # Manual editor trims. Defaults of 0 keep the automatic mix identical, so a
    # run that does not touch the faders behaves exactly as before.
    manual_voice_gain_db = max(-12.0, min(12.0, float(parameters.get("voice_gain_db") or 0.0)))
    manual_background_gain_db = max(
        -12.0, min(12.0, float(parameters.get("background_gain_db") or 0.0))
    )
    manual_voice_delay_sec = max(-2.0, min(2.0, float(parameters.get("voice_delay_sec") or 0.0)))
    # Conservative synchronization already moves individual utterances.  A
    # stale global editor delay must never be layered on top of those local
    # moves; manual delay belongs exclusively to the non-conservative mode.
    if synchronize_speech:
        manual_voice_delay_sec = 0.0
    # Which file supplies the picture.  Everything the pipeline produces lives
    # on the dubbed timeline, so choosing the original's picture additionally
    # requires retiming the finished track through the inverse map.
    video_source = str(parameters.get("video_source") or "dubbed").strip().lower()
    if video_source not in {"dubbed", "original"}:
        raise RuntimeError("Неизвестный источник видео для итогового фильма.")
    include_all_tracks = bool(parameters.get("include_all_tracks", True))
    reference_selection_mode = str(
        parameters.get("reference_selection_mode") or ""
    ).strip().lower()
    if reference_selection_mode and reference_selection_mode not in {
        "raw", "algorithmic"
    }:
        raise RuntimeError("Неизвестный режим выбора референса.")
    reference_adapter_mode = (
        reference_selection_mode
        or str(parameters.get("reference_adapter") or "auto").lower()
    )
    if reference_adapter_mode not in {
        "none", "raw", "algorithmic", "neural", "global", "global_auto", "auto"
    }:
        raise RuntimeError("Неизвестный режим адаптации референса.")
    neural_adapter_checkpoint = _resolve_neural_reference_adapter_checkpoint(parameters)
    if reference_adapter_mode == "neural" and neural_adapter_checkpoint is None:
        raise RuntimeError("Модуль автоматической подгонки не установлен.")
    global_adapter_checkpoint = _resolve_global_mastering_checkpoint(parameters)
    full_root = root / "application" / "full" / method
    full_root.mkdir(parents=True, exist_ok=True)
    alignment_dependency = full_root / "alignment_parameters.json"
    _atomic_json_if_changed(alignment_dependency, alignment)
    coincident_cfg = dict(
        store.cfg.get("coincident_speech_protection") or {}
    )
    coincident_identity = _coincident_speech_protection_identity(
        coincident_cfg
    )
    coincident_parameters_path = (
        full_root / "coincident_speech_protection_parameters.json"
    )
    _atomic_json_if_changed(
        coincident_parameters_path,
        coincident_identity,
    )
    coincident_module_path = Path(
        str(coincident_speech_protection.__file__)
    ).resolve()
    integrity_cfg = dict(store.cfg.get("speech_integrity_guard") or {})
    integrity_identity = _speech_integrity_guard_identity(integrity_cfg)
    integrity_parameters_path = (
        full_root / "speech_integrity_guard_parameters.json"
    )
    _atomic_json_if_changed(integrity_parameters_path, integrity_identity)
    integrity_module_path = Path(
        str(speech_integrity_guard.__file__)
    ).resolve()
    synchronization_cfg = dict(store.cfg.get("speech_synchronization") or {})
    synchronization_parameters_path = (
        full_root / "speech_synchronization_parameters.json"
    )
    _atomic_json_if_changed(
        synchronization_parameters_path,
        {
            "schema_version": int(speech_synchronizer.SCHEMA_VERSION),
            "algorithm": speech_synchronizer.ALGORITHM,
            "config": synchronization_cfg,
            "module_source": _checkpoint_fingerprint(
                Path(str(speech_synchronizer.__file__)).resolve()
            ),
        },
    )
    live_intermediates: dict[str, str] = {}
    # Measured only on the stem paths; direct_mix has no speech stem to judge.
    separation_report: dict[str, Any] = {}
    coincident_report: dict[str, Any] | None = None
    coincident_second_pass_report: dict[str, Any] | None = None
    primary_integrity_report: dict[str, Any] | None = None
    second_integrity_report: dict[str, Any] | None = None
    _publish_full_application_progress(
        store, pair, full_root, method, intermediates=live_intermediates
    )

    if method in {"voice_only", "speech_rebuild"}:
        cached_original_speech = root / "stems" / "original" / "en_speech.flac"
        cached_dubbed_speech = root / "stems" / "dubbed" / "speech_en_ru.flac"
        original_speech = _ensure_full_speech_stem(
            store,
            original_path,
            cached_original_speech,
            full_root / "original_en_speech.flac",
            ctx,
            pair["name"],
            "Полный фильм: выделение английской речи",
            model_role="original",
            progress_base=0.0,
            progress_span=18.0,
            force_recompute=force_recompute,
        )
        ctx.update(
            stage="Выделение английской речи",
            substage="Дорожка готова",
            progress=18.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        live_intermediates["original_en_speech"] = str(original_speech.resolve())
        _publish_full_application_progress(
            store, pair, full_root, method, intermediates=live_intermediates
        )
        dubbed_speech = _ensure_full_speech_stem(
            store,
            dubbed_path,
            cached_dubbed_speech,
            full_root / "dubbed_en_ru_speech.flac",
            ctx,
            pair["name"],
            "Полный фильм: выделение общей речи EN+RU",
            model_role="dubbed",
            progress_base=18.0,
            progress_span=18.0,
            force_recompute=force_recompute,
        )
        ctx.update(
            stage="Выделение речи перевода",
            substage="Дорожка готова",
            progress=36.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        live_intermediates["dubbed_en_ru_speech"] = str(dubbed_speech.resolve())
        _publish_full_application_progress(
            store, pair, full_root, method, intermediates=live_intermediates
        )
        aligned_reference = full_root / "aligned_en_speech.flac"
        aligned_reference_dependencies = [
            original_speech,
            alignment_dependency,
        ]
        if not _full_output_cache_reusable(
            aligned_reference,
            aligned_reference_dependencies,
            force_recompute=force_recompute,
        ):
            _align_to_dubbed(
                original_speech, aligned_reference, alignment,
                rate, duration, 1, ctx, pair["name"], "Сопоставление английской речи",
                progress_base=36.0,
                progress_span=8.0,
            )
            _record_model_output_for(
                aligned_reference, aligned_reference_dependencies
            )
        live_intermediates["aligned_en_speech"] = str(aligned_reference.resolve())
        ctx.update(
            stage="Сопоставление английской речи",
            substage="Готово",
            progress=44.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        _publish_full_application_progress(
            store, pair, full_root, method, intermediates=live_intermediates
        )
        aligned_original_program = full_root / "aligned_original_program.flac"
        original_program_dependencies = [
            original_path,
            alignment_dependency,
        ]
        if not _full_output_cache_reusable(
            aligned_original_program,
            original_program_dependencies,
            force_recompute=force_recompute,
        ):
            _align_to_dubbed(
                original_path,
                aligned_original_program,
                alignment,
                rate,
                duration,
                int(sf.info(str(original_path)).channels),
                ctx,
                pair["name"],
                "Сопоставление исходной английской дорожки",
                progress_base=44.0,
                progress_span=1.0,
            )
            _record_model_output_for(
                aligned_original_program,
                original_program_dependencies,
            )
        live_intermediates["aligned_original_program"] = str(
            aligned_original_program.resolve()
        )
        _publish_full_application_progress(
            store, pair, full_root, method, intermediates=live_intermediates
        )
        reference_for_subtraction = aligned_reference
        reference_adapter_report: dict[str, Any] = {
            "requested": reference_adapter_mode,
            "selected": "raw",
            "reason": "disabled",
        }
        if reference_adapter_mode in {
            "algorithmic", "neural", "global", "global_auto", "auto"
        }:
            preview_reference_adapter = (
                (pair.get("application_preview") or {}).get("reference_adapter", {})
            )
            preview_adapter = preview_reference_adapter.get("algorithmic", {})
            preview_adapter_summary = preview_adapter.get("summary") or {}
            algorithmic_profile_usable = bool(
                preview_adapter_summary.get("usable")
            )
            preview_neural = preview_reference_adapter.get("neural", {})
            preview_global = preview_reference_adapter.get("global", {})
            preview_selection = preview_reference_adapter.get("selection") or {}
            reduction = float(
                ((preview_adapter.get("summary") or {}).get("distance_reduction_percent") or 0.0)
            )
            neural_checkpoint_value = str(
                preview_reference_adapter.get("neural_checkpoint") or ""
            ).strip()
            if neural_checkpoint_value and Path(neural_checkpoint_value).is_file():
                neural_adapter_checkpoint = Path(neural_checkpoint_value).resolve()
            global_checkpoint_value = str(
                preview_reference_adapter.get("global_checkpoint") or ""
            ).strip()
            if global_checkpoint_value and Path(global_checkpoint_value).is_file():
                global_adapter_checkpoint = Path(global_checkpoint_value).resolve()
            recommended = str(preview_selection.get("recommended") or "").lower()
            selected_adapter = "raw"
            if reference_adapter_mode in {"algorithmic", "neural", "global"}:
                selected_adapter = reference_adapter_mode
            elif reference_adapter_mode == "global_auto":
                selected_adapter = "global" if recommended == "global" else "raw"
            elif reference_adapter_mode == "auto" and recommended in {"algorithmic", "neural", "global"}:
                selected_adapter = recommended
            unsafe_algorithmic_reason = ""
            if selected_adapter == "algorithmic" and not algorithmic_profile_usable:
                unsafe_algorithmic_reason = str(
                    preview_adapter_summary.get("reason")
                    or "common_component_not_confident"
                )
                selected_adapter = "raw"
            if selected_adapter == "neural" and neural_adapter_checkpoint is None:
                selected_adapter = "raw"
            if selected_adapter == "global" and (
                not preview_global or not _global_profile_usable(preview_global)
            ):
                if reference_adapter_mode == "global":
                    verdict = str(
                        ((preview_global.get("summary") or {}).get("verdict"))
                        or "profile_missing"
                    )
                    raise RuntimeError(
                        "Алгоритмическая подгонка безопасно отклонена "
                        f"(причина: {verdict}). Оставьте исходный режим или заново подготовьте превью."
                    )
                selected_adapter = "raw"
            use_algorithmic = selected_adapter == "algorithmic"
            use_neural = selected_adapter == "neural"
            use_global = selected_adapter == "global"
            reference_adapter_report = {
                "requested": reference_adapter_mode,
                "selected": selected_adapter,
                "preview_distance_reduction_percent": reduction,
                "preview_selection": preview_selection,
                "neural_checkpoint": (
                    str(neural_adapter_checkpoint.resolve())
                    if neural_adapter_checkpoint is not None
                    else ""
                ),
                "neural_epoch": preview_neural.get("epoch"),
                "global_checkpoint": (
                    str(global_adapter_checkpoint.resolve())
                    if global_adapter_checkpoint is not None
                    else ""
                ),
                "global_epoch": (preview_global.get("neural_global_mastering") or {}).get("epoch"),
                "reason": (
                    f"algorithmic_profile_rejected_{unsafe_algorithmic_reason}"
                    if unsafe_algorithmic_reason
                    else
                    "forced"
                    if reference_adapter_mode in {"algorithmic", "neural", "global"}
                    else (
                        f"preview_scenes_recommended_{selected_adapter}"
                        if selected_adapter != "raw"
                        else "preview_scenes_recommended_raw"
                    )
                ),
            }
            if use_algorithmic:
                adapted_reference = full_root / "aligned_en_speech_algorithmic.flac"
                adapted_report = full_root / "aligned_en_speech_algorithmic.report.json"
                algorithmic_profile_dependency = (
                    full_root / "aligned_en_speech_algorithmic.profile.json"
                )
                _atomic_json_if_changed(
                    algorithmic_profile_dependency,
                    preview_adapter,
                )
                algorithmic_dependencies = [
                    aligned_reference,
                    algorithmic_profile_dependency,
                ]
                if (
                    not adapted_report.is_file()
                    or not _full_output_cache_reusable(
                        adapted_reference,
                        algorithmic_dependencies,
                        force_recompute=force_recompute,
                    )
                ):
                    ctx.update(
                        stage="Адаптация английского референса",
                        substage="Применение профиля к полному фильму",
                        progress=45.0,
                        local_progress=0.0,
                        current_file=pair["name"],
                    )
                    adapt_reference_file_with_profile(
                        aligned_reference,
                        adapted_reference,
                        preview_adapter,
                        report_path=adapted_report,
                    )
                    _record_model_output_for(
                        adapted_reference,
                        algorithmic_dependencies,
                    )
                reference_for_subtraction = adapted_reference
                reference_adapter_report["reference"] = str(adapted_reference.resolve())
                reference_adapter_report["report"] = str(adapted_report.resolve())
            elif use_neural:
                assert neural_adapter_checkpoint is not None
                adapted_reference = full_root / "aligned_en_speech_neural.flac"
                adapted_report = full_root / "aligned_en_speech_neural.report.json"
                neural_dependencies = [
                    aligned_reference,
                    dubbed_speech,
                    neural_adapter_checkpoint,
                ]
                if (
                    not adapted_report.is_file()
                    or not _full_output_cache_reusable(
                        adapted_reference,
                        neural_dependencies,
                        force_recompute=force_recompute,
                    )
                ):
                    ctx.update(
                        stage="Адаптация английского референса",
                        substage="Автоматическая подгонка по полному фильму",
                        progress=45.0,
                        local_progress=0.0,
                        current_file=pair["name"],
                    )
                    adapt_reference_file_neural(
                        aligned_reference,
                        dubbed_speech,
                        neural_adapter_checkpoint,
                        adapted_reference,
                        report_path=adapted_report,
                    )
                    _record_model_output_for(
                        adapted_reference,
                        neural_dependencies,
                    )
                reference_for_subtraction = adapted_reference
                reference_adapter_report["reference"] = str(adapted_reference.resolve())
                reference_adapter_report["report"] = str(adapted_report.resolve())
            elif use_global:
                adapted_reference = full_root / "aligned_en_speech_global.flac"
                adapted_report = full_root / "aligned_en_speech_global.apply_report.json"
                global_profile_report_value = str(
                    preview_reference_adapter.get("global_profile_report") or ""
                ).strip()
                global_profile_report = (
                    Path(global_profile_report_value).resolve()
                    if global_profile_report_value
                    else None
                )
                global_profile_dependency = (
                    full_root / "aligned_en_speech_global.profile.json"
                )
                _atomic_json_if_changed(
                    global_profile_dependency,
                    preview_global,
                )
                profile_dependencies = [
                    aligned_reference,
                    global_profile_dependency,
                ]
                if (
                    global_profile_report is not None
                    and global_profile_report.is_file()
                ):
                    profile_dependencies.append(global_profile_report)
                if (
                    not adapted_report.is_file()
                    or not _full_output_cache_reusable(
                        adapted_reference,
                        profile_dependencies,
                        force_recompute=force_recompute,
                    )
                ):
                    ctx.update(
                        stage="Выравнивание качества дорожек",
                        substage="Применение единого профиля к английской речи",
                        progress=45.0,
                        local_progress=0.0,
                        current_file=pair["name"],
                    )
                    adapt_reference_file_with_profile(
                        aligned_reference,
                        adapted_reference,
                        preview_global,
                        report_path=adapted_report,
                    )
                    _record_model_output_for(
                        adapted_reference,
                        profile_dependencies,
                    )
                reference_for_subtraction = adapted_reference
                reference_adapter_report["reference"] = str(adapted_reference.resolve())
                reference_adapter_report["report"] = str(adapted_report.resolve())
                reference_adapter_report["profile_report"] = preview_reference_adapter.get(
                    "global_profile_report", ""
                )
        ctx.update(
            stage="Выравнивание качества дорожек",
            substage="Готово",
            progress=50.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        selected_reference_adapter = str(reference_adapter_report.get("selected") or "raw")
        if selected_reference_adapter == "algorithmic":
            ru_voice = full_root / "russian_voice_algorithmic_reference.flac"
            output_checkpoints = [checkpoint]
        elif selected_reference_adapter == "neural":
            ru_voice = full_root / "russian_voice_neural_reference.flac"
            output_checkpoints = (
                [checkpoint, neural_adapter_checkpoint]
                if neural_adapter_checkpoint is not None
                else [checkpoint]
            )
        elif selected_reference_adapter == "global":
            ru_voice = full_root / "russian_voice_global_reference.flac"
            global_profile_report_value = str(
                reference_adapter_report.get("profile_report") or ""
            ).strip()
            global_profile_report = (
                Path(global_profile_report_value).resolve()
                if global_profile_report_value
                else None
            )
            output_checkpoints = [checkpoint]
            if global_profile_report is not None and global_profile_report.is_file():
                output_checkpoints.append(global_profile_report)
        else:
            ru_voice = full_root / "russian_voice.flac"
            output_checkpoints = [checkpoint]
        window_routes = routing_document.get("windows") or []
        needs_primary_semantic = not applied_routing or any(
            item.get("recommended_model") == "dubclean_voice"
            for item in window_routes
        )
        aggregate_passthrough = applied_routing.get("recommended_model") == "passthrough"
        if aggregate_passthrough and not needs_primary_semantic:
            ru_voice = full_root / "russian_voice_passthrough.flac"
            _copy_audio(dubbed_speech, ru_voice)
            reference_adapter_report["subtraction_model"] = "passthrough"
            reference_adapter_report["reason"] = "applied_reference_compatibility"
        else:
            if routing_document:
                primary_checkpoint = _configured_semantic_checkpoint(
                    store,
                    "production_checkpoints",
                    "dubclean_voice.pt",
                )
                primary_output: Path | None = None
                route_dependencies = [dubbed_speech]
                routing_contract = full_root / "semantic_window_routing_contract.json"
                _atomic_json_if_changed(
                    routing_contract,
                    {
                        "schema_version": 2,
                        "sample_rate_policy": "timeline_resample_before_blend",
                    },
                )
                route_dependencies.append(routing_contract)
                routing_path_value = str(applied_routing.get("routing_map") or "").strip()
                routing_path = Path(routing_path_value).resolve() if routing_path_value else None
                if routing_path is not None and routing_path.is_file():
                    route_dependencies.append(routing_path)

                if needs_primary_semantic:
                    if primary_checkpoint is None:
                        raise RuntimeError("Модель DubClean Voice не найдена.")
                    primary_output = ru_voice.with_name(
                        f"{ru_voice.stem}_dubclean_voice.flac"
                    )
                    primary_dependencies = [
                        dubbed_speech,
                        reference_for_subtraction,
                        primary_checkpoint,
                    ]
                    if not _full_output_cache_reusable(
                        primary_output,
                        primary_dependencies,
                        force_recompute=force_recompute,
                    ):
                        _run_subtractor(
                            store, dubbed_speech, reference_for_subtraction, primary_output,
                            primary_checkpoint, ctx, pair["name"],
                            "Полный фильм: DubClean Voice",
                            progress_base=50.0,
                            progress_span=18.0,
                        )
                        _record_model_output_for(
                            primary_output,
                            primary_dependencies,
                        )
                    route_dependencies.extend(
                        [primary_output, primary_checkpoint]
                    )

                routed_voice = full_root / "russian_voice_routed.flac"
                if not _full_output_cache_reusable(
                    routed_voice,
                    route_dependencies,
                    force_recompute=force_recompute,
                ):
                    _route_semantic_model_windows(
                        primary_output, dubbed_speech,
                        routing_document, routed_voice,
                    )
                    _record_model_output_for(routed_voice, route_dependencies)
                ru_voice = routed_voice
                counts = {
                    name: sum(
                        1 for item in window_routes
                        if item.get("recommended_model") == name
                    )
                    for name in ("dubclean_voice", "passthrough")
                }
                reference_adapter_report["window_routing_applied"] = True
                reference_adapter_report["window_model_counts"] = counts
                reference_adapter_report["subtraction_model"] = "window_routing"
                reference_adapter_report["subtraction_checkpoints"] = [
                    path.name for path in (primary_checkpoint,)
                    if path is not None and path in route_dependencies
                ]
            else:
                ru_voice_dependencies = [
                    dubbed_speech,
                    reference_for_subtraction,
                    *output_checkpoints,
                ]
                if not _full_output_cache_reusable(
                    ru_voice,
                    ru_voice_dependencies,
                    force_recompute=force_recompute,
                ):
                    _run_subtractor(
                        store, dubbed_speech, reference_for_subtraction, ru_voice,
                        checkpoint, ctx, pair["name"], "Полный фильм: удаление английской речи",
                        progress_base=50.0,
                        progress_span=18.0,
                    )
                    _record_model_output_for(
                        ru_voice,
                        ru_voice_dependencies,
                    )
                reference_adapter_report["subtraction_model"] = "dubclean_voice"
                reference_adapter_report["subtraction_checkpoint"] = checkpoint.name
        ctx.update(
            stage="DubClean Voice",
            substage="Очистка речи завершена",
            progress=68.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        raw_ru_voice = ru_voice
        protected_ru_voice = (
            full_root / "russian_voice_coincident_protected.flac"
        )
        coincident_report_path = (
            full_root / "coincident_speech_protection_report.json"
        )
        coincident_input_dependencies = [
            raw_ru_voice,
            dubbed_speech,
            reference_for_subtraction,
            coincident_parameters_path,
        ]
        cached_coincident_report = _load_valid_coincident_report(
            coincident_report_path,
            dubbed_speech,
            reference_for_subtraction,
            raw_ru_voice,
            coincident_cfg,
        )
        coincident_dependencies = [
            *coincident_input_dependencies,
            coincident_report_path,
        ]
        if (
            cached_coincident_report is None
            or not _full_output_cache_reusable(
                protected_ru_voice,
                coincident_dependencies,
                force_recompute=force_recompute,
            )
        ):
            coincident_report = (
                coincident_speech_protection.protect_coincident_speech(
                    dubbed_speech,
                    reference_for_subtraction,
                    raw_ru_voice,
                    protected_ru_voice,
                    coincident_cfg,
                    report_path=coincident_report_path,
                    ctx=_ProgressRangeContext(ctx, 68.0, 1.0),
                )
            )
            _record_model_output_for(
                protected_ru_voice, coincident_dependencies
            )
        else:
            coincident_report = cached_coincident_report
        primary_integrity_voice = (
            full_root / "russian_voice_integrity_protected.flac"
        )
        primary_integrity_report_path = (
            full_root / "speech_integrity_primary_report.json"
        )
        primary_integrity_input_dependencies = [
            protected_ru_voice,
            dubbed_speech,
            aligned_original_program,
            aligned_reference,
            reference_for_subtraction,
            integrity_parameters_path,
            integrity_module_path,
        ]
        cached_primary_integrity_report = _load_integrity_report(
            primary_integrity_report_path,
            "post_primary_pass",
        )
        primary_integrity_dependencies = [
            *primary_integrity_input_dependencies,
            primary_integrity_report_path,
        ]
        if (
            cached_primary_integrity_report is None
            or not _full_output_cache_reusable(
                primary_integrity_voice,
                primary_integrity_dependencies,
                force_recompute=force_recompute,
            )
        ):
            primary_integrity_report = (
                speech_integrity_guard.guard_primary_pass(
                    dubbed_speech,
                    protected_ru_voice,
                    aligned_original_program,
                    aligned_reference,
                    reference_for_subtraction,
                    primary_integrity_voice,
                    integrity_cfg,
                    report_path=primary_integrity_report_path,
                    ctx=_ProgressRangeContext(ctx, 68.8, 0.8),
                )
            )
            _record_model_output_for(
                primary_integrity_voice,
                primary_integrity_dependencies,
            )
        else:
            primary_integrity_report = cached_primary_integrity_report
        ru_voice = primary_integrity_voice
        intermediates = {
            **live_intermediates,
            "aligned_en_speech": str(aligned_reference.resolve()),
            "aligned_original_program": str(
                aligned_original_program.resolve()
            ),
            "reference_for_subtraction": str(reference_for_subtraction.resolve()),
            "russian_voice": str(ru_voice.resolve()),
            "russian_voice_before_coincident_protection": str(
                raw_ru_voice.resolve()
            ),
            "russian_voice_before_integrity_guard": str(
                protected_ru_voice.resolve()
            ),
            "coincident_speech_protection_report": str(
                coincident_report_path.resolve()
            ),
            "speech_integrity_primary_report": str(
                primary_integrity_report_path.resolve()
            ),
        }
        _publish_full_application_progress(
            store, pair, full_root, method, intermediates=intermediates
        )
        if reference_adapter_report.get("selected") == "algorithmic":
            intermediates["aligned_en_speech_algorithmic"] = str(
                reference_for_subtraction.resolve()
            )
        if reference_adapter_report.get("selected") == "neural":
            intermediates["aligned_en_speech_neural"] = str(
                reference_for_subtraction.resolve()
            )
        if reference_adapter_report.get("selected") == "global":
            intermediates["aligned_en_speech_global"] = str(
                reference_for_subtraction.resolve()
            )
        processed_voice = ru_voice
        if speech_second_pass:
            raw_second_pass_voice = (
                full_root / "russian_voice_second_pass.flac"
            )
            speech_dependencies = [
                ru_voice,
                *_speech_extractor_dependencies(store, "original"),
            ]
            if not _full_output_cache_reusable(
                raw_second_pass_voice,
                speech_dependencies,
                force_recompute=force_recompute,
            ):
                pipeline._run_speech_extractor(
                    store,
                    ru_voice,
                    raw_second_pass_voice,
                    ctx,
                    pair["name"],
                    "Повторный проход модуля извлечения речи",
                    progress_base=69.6,
                    progress_span=1.4,
                    model_role="original",
                )
                _record_model_output_for(
                    raw_second_pass_voice, speech_dependencies
                )
            second_pass_voice = (
                full_root
                / "russian_voice_second_pass_coincident_protected.flac"
            )
            coincident_second_pass_report_path = (
                full_root
                / "coincident_speech_protection_second_pass_report.json"
            )
            second_protection_input_dependencies = [
                raw_second_pass_voice,
                dubbed_speech,
                reference_for_subtraction,
                coincident_parameters_path,
                coincident_module_path,
            ]
            cached_second_pass_report = _load_valid_coincident_report(
                coincident_second_pass_report_path,
                dubbed_speech,
                reference_for_subtraction,
                raw_second_pass_voice,
                coincident_cfg,
            )
            second_protection_dependencies = [
                *second_protection_input_dependencies,
                coincident_second_pass_report_path,
            ]
            if (
                cached_second_pass_report is None
                or not _full_output_cache_reusable(
                    second_pass_voice,
                    second_protection_dependencies,
                    force_recompute=force_recompute,
                )
            ):
                coincident_second_pass_report = (
                    coincident_speech_protection.protect_coincident_speech(
                        dubbed_speech,
                        reference_for_subtraction,
                        raw_second_pass_voice,
                        second_pass_voice,
                        coincident_cfg,
                        report_path=coincident_second_pass_report_path,
                        ctx=_ProgressRangeContext(ctx, 71.0, 0.4),
                    )
                )
                _record_model_output_for(
                    second_pass_voice,
                    second_protection_dependencies,
                )
            else:
                coincident_second_pass_report = cached_second_pass_report
            second_pass_before_integrity_voice = second_pass_voice
            second_integrity_voice = (
                full_root / "russian_voice_second_pass_integrity_protected.flac"
            )
            second_integrity_report_path = (
                full_root / "speech_integrity_second_pass_report.json"
            )
            second_integrity_input_dependencies = [
                ru_voice,
                second_pass_before_integrity_voice,
                primary_integrity_report_path,
                integrity_parameters_path,
                integrity_module_path,
            ]
            cached_second_integrity_report = _load_integrity_report(
                second_integrity_report_path,
                "post_second_pass",
            )
            second_integrity_dependencies = [
                *second_integrity_input_dependencies,
                second_integrity_report_path,
            ]
            if (
                cached_second_integrity_report is None
                or not _full_output_cache_reusable(
                    second_integrity_voice,
                    second_integrity_dependencies,
                    force_recompute=force_recompute,
                )
            ):
                second_integrity_report = (
                    speech_integrity_guard.guard_second_pass(
                        ru_voice,
                        second_pass_before_integrity_voice,
                        second_integrity_voice,
                        integrity_cfg,
                        report_path=second_integrity_report_path,
                        ctx=_ProgressRangeContext(ctx, 71.4, 0.6),
                    )
                )
                _record_model_output_for(
                    second_integrity_voice,
                    second_integrity_dependencies,
                )
            else:
                second_integrity_report = cached_second_integrity_report
            second_pass_voice = second_integrity_voice
            intermediates["russian_voice_second_pass"] = str(
                second_pass_voice.resolve()
            )
            intermediates[
                "russian_voice_second_pass_before_coincident_protection"
            ] = str(raw_second_pass_voice.resolve())
            intermediates[
                "coincident_speech_protection_second_pass_report"
            ] = str(coincident_second_pass_report_path.resolve())
            intermediates[
                "russian_voice_second_pass_before_integrity_guard"
            ] = str(second_pass_before_integrity_voice.resolve())
            intermediates[
                "speech_integrity_second_pass_report"
            ] = str(second_integrity_report_path.resolve())
            processed_voice = second_pass_voice
            _publish_full_application_progress(
                store, pair, full_root, method, intermediates=intermediates
            )
        ctx.update(
            stage=(
                "Повторный проход модуля извлечения речи"
                if speech_second_pass
                else "Подготовка очищенной русской речи"
            ),
            substage="Готово",
            progress=72.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        voice_for_result = processed_voice
        synchronization_report: dict[str, Any] | None = None
        # Synchronization copies and moves whole segments of the voice stem.
        # That is the intended edit only while the stem really is speech: when
        # the extractor handed its input back unchanged, the "speech segments"
        # cover the entire soundtrack, and moving hundreds of them shreds a
        # continuous background into displaced fragments.  Measure before
        # moving anything.
        separation_report.update(
            speech_synchronizer.evaluate_stem_separation(
                dubbed_path,
                dubbed_speech,
                dict(synchronization_cfg.get("separation_guard") or {}),
            )
        )
        separation_report_path = full_root / "speech_stem_separation.json"
        atomic_json(separation_report_path, separation_report)
        intermediates["speech_stem_separation_report"] = str(
            separation_report_path.resolve()
        )
        if synchronize_speech and not separation_report.get("separated", True):
            measured = separation_report.get("measured") or {}
            ctx.log(
                "Синхронизация речи отключена: выделение речи вернуло исходную "
                "дорожку почти без изменений (совпадение "
                f"{measured.get('median_correlation', 0):.3f}, остаток "
                f"{measured.get('median_residual_ratio_db', 0):.1f} дБ). "
                "Перенос реплик сдвинул бы вместе с ними музыку и эффекты."
            )
            synchronize_speech = False
        if synchronize_speech:
            synchronized_voice = full_root / "russian_voice_synchronized.flac"
            synchronization_path = full_root / "speech_synchronization.json"
            active_integrity_report_paths = [primary_integrity_report_path]
            if speech_second_pass:
                active_integrity_report_paths.append(
                    second_integrity_report_path
                )
            synchronization_dependencies = [
                processed_voice,
                aligned_reference,
                *active_integrity_report_paths,
                synchronization_parameters_path,
                Path(str(speech_synchronizer.__file__)).resolve(),
            ]
            if not _full_output_cache_reusable(
                synchronized_voice,
                synchronization_dependencies,
                force_recompute=force_recompute,
            ):
                synchronization_report = synchronize_voice_file(
                    processed_voice,
                    aligned_reference,
                    synchronized_voice,
                    maximum_advance_sec=float(
                        synchronization_cfg.get(
                            "maximum_advance_sec", 0.85
                        )
                    ),
                    preservation_intervals=_guard_preservation_intervals(
                        primary_integrity_report,
                        second_integrity_report,
                    ),
                    crossfade_sec=float(
                        synchronization_cfg.get("crossfade_sec", 0.015)
                    ),
                )
                atomic_json(synchronization_path, synchronization_report)
                _record_model_output_for(
                    synchronized_voice,
                    synchronization_dependencies,
                )
            else:
                synchronization_report = read_json(synchronization_path, {}) or {}
            intermediates["russian_voice_synchronized"] = str(
                synchronized_voice.resolve()
            )
            intermediates["speech_synchronization_report"] = str(
                synchronization_path.resolve()
            )
            voice_for_result = synchronized_voice
            _publish_full_application_progress(
                store, pair, full_root, method, intermediates=intermediates
            )
        ctx.update(
            stage="Синхронизация готовой речи",
            substage="Готово" if synchronize_speech else "Не требуется",
            progress=74.0,
            local_progress=100.0,
            current_file=pair["name"],
        )
        if method == "voice_only":
            final = voice_for_result
        else:
            if background_checkpoint is None or not background_checkpoint.is_file():
                raise RuntimeError(
                    "Для основной сборки не выбрана обученная модель восстановления фона. "
                    "Пересоздайте превью после проверки установленных моделей."
                )
            learned_me = full_root / "learned_music_effects_original_timeline.flac"
            # The sidecar records exactly which checkpoint file produced the
            # cached background track. This also invalidates when the manifest is missing
            # (a crashed earlier run) or when a rolling checkpoint name was
            # overwritten in place with new weights.
            background_dependencies = [
                original_path,
                original_speech,
                background_checkpoint,
            ]
            if not _full_output_cache_reusable(
                learned_me,
                background_dependencies,
                force_recompute=force_recompute,
            ):
                learned_me.unlink(missing_ok=True)
                _run_background_restorer(
                    store,
                    original_path,
                    original_speech,
                    learned_me,
                    background_checkpoint,
                    ctx,
                    pair["name"],
                    "Полный фильм: восстановление музыки и эффектов",
                    progress_base=74.0,
                    progress_span=12.0,
                )
                _record_model_output_for(
                    learned_me,
                    background_dependencies,
                )
            intermediates["music_effects_original_timeline"] = str(
                learned_me.resolve()
            )
            _publish_full_application_progress(
                store, pair, full_root, method, intermediates=intermediates
            )
            ctx.update(
                stage="Восстановление музыки и эффектов",
                substage="Фоновая дорожка готова",
                progress=86.0,
                local_progress=100.0,
                current_file=pair["name"],
            )
            instrumental_info = sf.info(str(learned_me))
            aligned_me = full_root / "aligned_music_effects.flac"
            aligned_me_dependencies = [
                learned_me,
                alignment_dependency,
            ]
            if not _full_output_cache_reusable(
                aligned_me,
                aligned_me_dependencies,
                force_recompute=force_recompute,
            ):
                _align_to_dubbed(
                    learned_me, aligned_me,
                    alignment, rate, duration, int(instrumental_info.channels), ctx,
                    pair["name"], "Сопоставление музыки и эффектов",
                    progress_base=86.0,
                    progress_span=4.0,
                )
                _record_model_output_for(
                    aligned_me,
                    aligned_me_dependencies,
                )
            intermediates["music_effects"] = str(aligned_me.resolve())
            _publish_full_application_progress(
                store, pair, full_root, method, intermediates=intermediates
            )
            ctx.update(
                stage="Сопоставление музыки и эффектов",
                substage="Готово",
                progress=90.0,
                local_progress=100.0,
                current_file=pair["name"],
            )
            mix_balance_path = full_root / "mix_balance.json"
            baseline_balance: dict[str, Any] = {
                "schema_version": 2,
                "enabled": False,
                "reason": "disabled",
                "background_gain_db": 0.0,
                "voice_gain_db": 0.0,
            }
            final_balance: dict[str, Any] = dict(baseline_balance)
            if balance_final_mix:
                ctx.update(
                    stage="Финальная сводка",
                    substage="Измерение баланса фон/речь",
                    progress=90.0,
                    local_progress=0.0,
                )
                baseline_balance = _calculate_mix_balance(
                    aligned_me,
                    aligned_reference,
                    processed_voice,
                    original_path,
                    dubbed_speech_path=dubbed_speech,
                    dubbed_mix_path=dubbed_path,
                )
                final_balance = (
                    _calculate_mix_balance(
                        aligned_me,
                        aligned_reference,
                        voice_for_result,
                        original_path,
                        dubbed_speech_path=dubbed_speech,
                        dubbed_mix_path=dubbed_path,
                    )
                    if synchronize_speech
                    else dict(baseline_balance)
                )
            _atomic_json_if_changed(
                mix_balance_path,
                {
                    "schema_version": 2,
                    "enabled": balance_final_mix,
                    "baseline_without_speech_synchronization": baseline_balance,
                    "final": final_balance,
                },
            )
            baseline_final = full_root / "russian_voice_plus_music_effects.flac"
            ctx.update(
                stage="Финальная сводка",
                substage="Запись варианта без синхронизации речи",
                progress=91.0,
                local_progress=0.0,
                current_file=pair["name"],
            )
            baseline_voice = processed_voice
            baseline_mix_settings = full_root / "mix_baseline_parameters.json"
            _atomic_json_if_changed(
                baseline_mix_settings,
                {
                    "background_gain_db": float(
                        baseline_balance.get("background_gain_db", 0.0)
                    )
                    + manual_background_gain_db,
                    "voice_gain_db": float(
                        baseline_balance.get("voice_gain_db", 0.0)
                    )
                    + manual_voice_gain_db,
                    "manual_voice_delay_sec": manual_voice_delay_sec,
                },
            )
            baseline_dependencies = [
                aligned_me,
                processed_voice,
                baseline_mix_settings,
            ]
            if not _full_output_cache_reusable(
                baseline_final,
                baseline_dependencies,
                force_recompute=force_recompute,
            ):
                _mix_audio_files(
                    aligned_me,
                    baseline_voice,
                    baseline_final,
                    background_gain_db=float(
                        baseline_balance.get("background_gain_db", 0.0)
                    )
                    + manual_background_gain_db,
                    voice_gain_db=float(
                        baseline_balance.get("voice_gain_db", 0.0)
                    )
                    + manual_voice_gain_db,
                    voice_delay_sec=manual_voice_delay_sec,
                    report_path=full_root / "mix_baseline_write_report.json",
                    ctx=ctx,
                    progress_substage="Запись варианта без синхронизации речи",
                    progress_base=91.0,
                    progress_span=3.0,
                )
                _record_model_output_for(
                    baseline_final, baseline_dependencies
                )
            intermediates["result_without_speech_synchronization"] = str(
                baseline_final.resolve()
            )
            _publish_full_application_progress(
                store,
                pair,
                full_root,
                method,
                intermediates=intermediates,
                audio=baseline_final if not synchronize_speech else None,
            )
            if synchronize_speech:
                final = (
                    full_root
                    / "russian_voice_synchronized_plus_music_effects.flac"
                )
                ctx.update(
                    stage="Финальная сводка",
                    substage="Запись финальной русской дорожки",
                    progress=94.0,
                    local_progress=0.0,
                    current_file=pair["name"],
                )
                synchronized_voice = voice_for_result
                final_mix_settings = full_root / "mix_synchronized_parameters.json"
                _atomic_json_if_changed(
                    final_mix_settings,
                    {
                        "background_gain_db": float(
                            final_balance.get("background_gain_db", 0.0)
                        )
                        + manual_background_gain_db,
                        "voice_gain_db": float(
                            final_balance.get("voice_gain_db", 0.0)
                        )
                        + manual_voice_gain_db,
                        "manual_voice_delay_sec": manual_voice_delay_sec,
                    },
                )
                final_dependencies = [
                    aligned_me,
                    voice_for_result,
                    final_mix_settings,
                ]
                if not _full_output_cache_reusable(
                    final,
                    final_dependencies,
                    force_recompute=force_recompute,
                ):
                    _mix_audio_files(
                        aligned_me,
                        synchronized_voice,
                        final,
                        background_gain_db=float(
                            final_balance.get("background_gain_db", 0.0)
                        )
                        + manual_background_gain_db,
                        voice_gain_db=float(
                            final_balance.get("voice_gain_db", 0.0)
                        )
                        + manual_voice_gain_db,
                        voice_delay_sec=manual_voice_delay_sec,
                        report_path=full_root / "mix_synchronized_write_report.json",
                        ctx=ctx,
                        progress_substage="Запись финальной русской дорожки",
                        progress_base=94.0,
                        progress_span=2.0,
                    )
                    _record_model_output_for(final, final_dependencies)
            else:
                final = baseline_final

            song_report_path = full_root / "song_protection_report.json"
            song_cfg = dict(store.cfg.get("song_protection") or {})
            dialogue_timeline_guard_sec = (
                song_protection.dialogue_timeline_guard(
                    song_cfg,
                    manual_voice_delay_sec=manual_voice_delay_sec,
                    synchronize_speech=synchronize_speech,
                )
            )
            song_cfg["dialogue_timeline_guard_sec"] = (
                dialogue_timeline_guard_sec
            )
            aligned_original_for_song_detection = (
                full_root / "aligned_original_mix_for_song_detection_and_restore.flac"
            )
            ctx.update(
                stage="Защита песенных фрагментов",
                substage="Подготовка оригинальной английской дорожки",
                progress=96.0,
                local_progress=0.0,
                current_file=pair["name"],
            )
            song_detection_alignment_dependencies = [
                original_path,
                alignment_dependency,
            ]
            if not _full_output_cache_reusable(
                aligned_original_for_song_detection,
                song_detection_alignment_dependencies,
                force_recompute=force_recompute,
            ):
                _align_to_dubbed(
                    original_path,
                    aligned_original_for_song_detection,
                    alignment,
                    rate,
                    duration,
                    int(dubbed_info.channels),
                    ctx,
                    pair["name"],
                    "Сопоставление английской дорожки для поиска песен",
                    progress_base=96.0,
                    progress_span=0.5,
                )
                _record_model_output_for(
                    aligned_original_for_song_detection,
                    song_detection_alignment_dependencies,
                )
            intermediates["song_detection_original_mix"] = str(
                aligned_original_for_song_detection.resolve()
            )
            neutral_song_detection_mix = (
                full_root / "song_detection_processed_mix_neutral.flac"
            )
            neutral_song_detection_settings = (
                full_root / "song_detection_mix_parameters.json"
            )
            _atomic_json_if_changed(
                neutral_song_detection_settings,
                {
                    "background_gain_db": 0.0,
                    "voice_gain_db": 0.0,
                    "manual_voice_delay_sec": manual_voice_delay_sec,
                    "purpose": "song_damage_decision_only",
                },
            )
            neutral_song_detection_dependencies = [
                aligned_me,
                processed_voice,
                neutral_song_detection_settings,
            ]
            if not _full_output_cache_reusable(
                neutral_song_detection_mix,
                neutral_song_detection_dependencies,
                force_recompute=force_recompute,
            ):
                _mix_audio_files(
                    aligned_me,
                    processed_voice,
                    neutral_song_detection_mix,
                    background_gain_db=0.0,
                    voice_gain_db=0.0,
                    voice_delay_sec=manual_voice_delay_sec,
                    report_path=(
                        full_root / "song_detection_mix_write_report.json"
                    ),
                    ctx=ctx,
                    progress_substage=(
                        "Подготовка звука для проверки сохранности песен"
                    ),
                    progress_base=96.4,
                    progress_span=0.1,
                )
                _record_model_output_for(
                    neutral_song_detection_mix,
                    neutral_song_detection_dependencies,
                )
            intermediates["song_detection_processed_mix"] = str(
                neutral_song_detection_mix.resolve()
            )
            ctx.update(
                stage="Защита песенных фрагментов",
                substage="Поиск продолжительных песенных участков",
                progress=96.5,
                local_progress=0.0,
                current_file=pair["name"],
            )
            song_report = song_protection.detect_song_intervals(
                aligned_original_for_song_detection,
                dubbed_path,
                dubbed_speech,
                aligned_me,
                neutral_song_detection_mix,
                song_cfg,
                report_path=song_report_path,
                ctx=ctx,
                original_speech_stem=reference_for_subtraction,
                translation_voice_stem=ru_voice,
            )
            restoration_segments = song_report.get("restoration_segments") or []
            song_report["restoration_policy"] = {
                "sources": {
                    "original_aligned": str(
                        aligned_original_for_song_detection.resolve()
                    ),
                    "dubbed": str(dubbed_path.resolve()),
                },
                "source_selection": (
                    "original_aligned_only_when_music_features_match"
                ),
                "fallback_when_music_does_not_match": "no_replacement",
                "speech_processing_bypassed_inside_restoration_segments": True,
                "processed_translation_kept_inside_dialogue_intervals": True,
                "low_confidence_content_kept_processed": True,
                "dialogue_timeline_guard_sec": dialogue_timeline_guard_sec,
                "gain_change_inside_confirmed_intervals_db": 0.0,
                "decision_mix_gains_db": {
                    "background": 0.0,
                    "voice": 0.0,
                },
                "crossfade_sec": float(
                    song_cfg.get(
                        "crossfade_sec", 0.35
                    )
                ),
            }
            if restoration_segments:
                restored_outputs: list[str] = []
                unprotected_baseline = baseline_final
                protected_baseline = full_root / (
                    "russian_voice_plus_music_effects_song_protected.flac"
                )
                restored_outputs.append(str(protected_baseline.resolve()))
                song_protection.apply_song_protection(
                    unprotected_baseline,
                    dubbed_path,
                    protected_baseline,
                    restoration_segments,
                    original_mix=aligned_original_for_song_detection,
                    crossfade_sec=float(
                        song_cfg.get(
                            "crossfade_sec", 0.35
                        )
                    ),
                    ctx=ctx,
                )
                intermediates["result_before_song_protection"] = str(
                    unprotected_baseline.resolve()
                )
                baseline_final = protected_baseline
                if synchronize_speech:
                    unprotected_final = final
                    protected_final = full_root / (
                        "russian_voice_synchronized_plus_music_effects_song_protected.flac"
                    )
                    restored_outputs.append(str(protected_final.resolve()))
                    song_protection.apply_song_protection(
                        unprotected_final,
                        dubbed_path,
                        protected_final,
                        restoration_segments,
                        original_mix=aligned_original_for_song_detection,
                        crossfade_sec=float(
                            song_cfg.get(
                                "crossfade_sec", 0.35
                            )
                        ),
                        ctx=ctx,
                    )
                    intermediates["final_before_song_protection"] = str(
                        unprotected_final.resolve()
                    )
                    final = protected_final
                else:
                    final = baseline_final
                song_report["restored_outputs"] = restored_outputs
            atomic_json(song_report_path, song_report)
            intermediates["result_without_speech_synchronization"] = str(
                baseline_final.resolve()
            )
            intermediates["song_protection_report"] = str(
                song_report_path.resolve()
            )
            song_protection_status = _song_protection_completion_status(
                song_report
            )
            classifier_state = song_report.get("classifier") or {}
            if (
                classifier_state.get("available") is False
                and hasattr(ctx, "log")
            ):
                classifier_error = str(
                    classifier_state.get("error") or "причина не указана"
                )
                ctx.log(
                    f"{song_protection_status}. {classifier_error}"
                )
            ctx.update(
                stage="Защита песенных фрагментов",
                substage=song_protection_status,
                progress=98.0,
                local_progress=100.0,
                current_file=pair["name"],
            )
            intermediates["music_effects"] = str(aligned_me.resolve())
            intermediates["mix_balance"] = str(mix_balance_path.resolve())
            intermediates["background_checkpoint"] = str(
                background_checkpoint.resolve()
            )
            _publish_full_application_progress(
                store,
                pair,
                full_root,
                method,
                intermediates=intermediates,
                audio=final,
            )
    else:
        aligned_original = _align_to_dubbed(
            original_path, full_root / "aligned_original_mix.flac", alignment,
            rate, duration, 1, ctx, pair["name"], "Сопоставление полного оригинального звука",
            progress_base=0.0,
            progress_span=35.0,
        )
        dubbed, dubbed_rate = sf.read(str(dubbed_path), dtype="float32", always_2d=True)
        dubbed_mono = full_root / "dubbed_mix_mono.flac"
        _write(dubbed_mono, _mono(dubbed), dubbed_rate)
        final = full_root / "direct_dirty_mix_result.flac"
        _run_subtractor(
            store, dubbed_mono, aligned_original, final,
            checkpoint, ctx, pair["name"], "Обработка полного звука",
            progress_base=35.0,
            progress_span=63.0,
        )
        intermediates = {
            "aligned_original_mix": str(aligned_original.resolve()),
            "dubbed_mix": str(dubbed_mono.resolve()),
        }

    user_files = _publish_user_audio_files(
        pair,
        full_root,
        intermediates,
        audio=final,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "method": method,
        "force_recompute": force_recompute,
        "checkpoint": str(checkpoint),
        "audio": user_files.get("audio", str(final.resolve())),
        "internal_audio": str(final.resolve()),
        "user_files": user_files,
        "intermediates": intermediates,
        "speech_synchronization": {
            "enabled": synchronize_speech,
            "mode": (
                "neural_stems_conservative_onset_alignment"
                if synchronize_speech
                else "disabled"
            ),
            "report": intermediates.get(
                "speech_synchronization_report", ""
            ),
            "preservation_interval_count": len(
                _guard_preservation_intervals(
                    primary_integrity_report,
                    second_integrity_report,
                )
            ),
            "stem_separation": {
                "algorithm": speech_synchronizer.SEPARATION_ALGORITHM,
                "schema_version": int(
                    speech_synchronizer.SEPARATION_SCHEMA_VERSION
                ),
                "report": intermediates.get(
                    "speech_stem_separation_report", ""
                ),
                "separated": bool(separation_report.get("separated", True)),
                "measured": separation_report.get("measured") or {},
            },
        },
        "speech_extraction_second_pass": {
            "enabled": bool(speech_second_pass),
            "artifact": intermediates.get("russian_voice_second_pass", ""),
        },
        "coincident_speech_protection": (
            {
                "enabled": bool(
                    coincident_identity["config"].get("enabled")
                ),
                "algorithm": coincident_speech_protection.ALGORITHM,
                "report": intermediates.get(
                    "coincident_speech_protection_report", ""
                ),
                "summary": (
                    (coincident_report or {}).get("summary") or {}
                ),
                "second_pass_report": intermediates.get(
                    "coincident_speech_protection_second_pass_report",
                    "",
                ),
                "second_pass_summary": (
                    (coincident_second_pass_report or {}).get("summary")
                    or {}
                ),
            }
            if method in {"voice_only", "speech_rebuild"}
            else {
                "enabled": False,
                "algorithm": coincident_speech_protection.ALGORITHM,
                "reason": "direct_mix_has_no_separate_speech_stem",
                "summary": {
                    "modified": False,
                    "confirmed_count": 0,
                },
            }
        ),
        "speech_integrity_guard": (
            {
                "enabled": bool(integrity_identity["config"].get("enabled")),
                "algorithm": speech_integrity_guard.ALGORITHM,
                "schema_version": int(speech_integrity_guard.SCHEMA_VERSION),
                "parameters": str(integrity_parameters_path.resolve()),
                "primary_report": intermediates.get(
                    "speech_integrity_primary_report", ""
                ),
                "primary_summary": (
                    (primary_integrity_report or {}).get("summary") or {}
                ),
                "second_pass_report": intermediates.get(
                    "speech_integrity_second_pass_report", ""
                ),
                "second_pass_summary": (
                    (second_integrity_report or {}).get("summary") or {}
                ),
            }
            if method in {"voice_only", "speech_rebuild"}
            else {
                "enabled": False,
                "algorithm": speech_integrity_guard.ALGORITHM,
                "reason": "direct_mix_has_no_separate_speech_stem",
            }
        ),
        "song_protection": (
            song_report
            if method == "speech_rebuild"
            else {
                "enabled": False,
                "summary": "Для выбранного способа сборки защита полного микса не применяется.",
                "confirmed_intervals": [],
            }
        ),
        "mix_balance": {
            "enabled": balance_final_mix,
            "mode": (
                "match_original_background_to_speech_ratio"
                if balance_final_mix
                else "disabled"
            ),
        },
        "reference_adapter": reference_adapter_report,
        "reference_compatibility": applied_routing,
    }
    _publish_full_application_progress(
        store,
        pair,
        full_root,
        method,
        intermediates=intermediates,
        audio=final,
    )
    if bool(parameters.get("remux", True)):
        dubbed_video = Path(pair["sources"]["dubbed"]["path"]).resolve()
        original_video = Path(pair["sources"]["original"]["path"]).resolve()
        video = dubbed_video
        packaged_audio = final
        packaging_report: dict[str, Any] = {
            "video_source": "dubbed",
            "video_path": str(dubbed_video),
            "include_all_tracks": include_all_tracks,
            "retimed_to_video_timeline": False,
        }
        extra_audio: list[dict[str, str]] = []
        if video_source == "original":
            # The finished track sits on the dubbed timeline; the original's
            # picture runs on its own.  Refuse rather than ship a film whose
            # sound holds only where the map happened to be usable.
            inverted = _inverted_alignment(alignment)
            original_info = sf.info(str(original_path))
            coverage = _alignment_coverage(inverted, float(original_info.duration))
            minimum_coverage = float(
                (store.cfg.get("alignment") or {}).get(
                    "minimum_video_swap_coverage", 0.95
                )
            )
            packaging_report["inverse_map_coverage"] = round(coverage, 5)
            if coverage < minimum_coverage:
                raise RuntimeError(
                    "Нельзя собрать фильм с картинкой оригинала: временная "
                    f"карта покрывает лишь {coverage * 100:.1f}% его "
                    f"длительности (нужно {minimum_coverage * 100:.0f}%). "
                    "Уточните сопоставление дорожек или соберите фильм с "
                    "картинкой перевода."
                )
            video = original_video
            retimed = full_root / "result_audio_original_timeline.flac"
            retimed_dependencies = [final, alignment_dependency]
            if not _full_output_cache_reusable(
                retimed,
                retimed_dependencies,
                force_recompute=force_recompute,
                recipe="inverse_timeline_v1",
            ):
                _align_to_dubbed(
                    final,
                    retimed,
                    inverted,
                    int(original_info.samplerate),
                    float(original_info.duration),
                    int(sf.info(str(final)).channels),
                    ctx,
                    pair["name"],
                    "Перенос готовой дорожки на картинку оригинала",
                    progress_base=98.0,
                    progress_span=1.0,
                )
                _record_model_output_for(
                    retimed, retimed_dependencies, recipe="inverse_timeline_v1"
                )
            packaged_audio = retimed
            intermediates["result_audio_original_timeline"] = str(retimed.resolve())
            packaging_report.update(
                {
                    "video_source": "original",
                    "video_path": str(original_video),
                    "retimed_to_video_timeline": True,
                }
            )
            if include_all_tracks:
                # Mirror of the dubbed-picture case: the other side's programme
                # joins the film, brought onto the timeline the picture uses.
                # Without this, picking the better picture would silently cost
                # the operator the untouched translation to compare against.
                companion = full_root / "dubbed_program_original_timeline.flac"
                companion_dependencies = [dubbed_path, alignment_dependency]
                if not _full_output_cache_reusable(
                    companion,
                    companion_dependencies,
                    force_recompute=force_recompute,
                    recipe="inverse_timeline_v1",
                ):
                    _align_to_dubbed(
                        dubbed_path,
                        companion,
                        inverted,
                        int(original_info.samplerate),
                        float(original_info.duration),
                        int(sf.info(str(dubbed_path)).channels),
                        ctx,
                        pair["name"],
                        "Перенос дорожки перевода на картинку оригинала",
                        progress_base=99.0,
                        progress_span=0.5,
                    )
                    _record_model_output_for(
                        companion,
                        companion_dependencies,
                        recipe="inverse_timeline_v1",
                    )
                intermediates["dubbed_program_original_timeline"] = str(
                    companion.resolve()
                )
                extra_audio.append(
                    {
                        "path": str(companion.resolve()),
                        "title": "Дорожка перевода · синхронизирована",
                        "language": _source_language(pair, "dubbed"),
                    }
                )
        elif include_all_tracks:
            # Only the other side's programme already living on this timeline
            # may join: raw tracks from the other file would play out of sync.
            companion = (
                intermediates.get("aligned_original_program")
                or intermediates.get("aligned_original_mix")
                or intermediates.get("song_detection_original_mix")
            )
            if companion and Path(companion).is_file():
                extra_audio.append(
                    {
                        "path": companion,
                        "title": "Оригинальная дорожка · синхронизирована",
                        "language": _source_language(pair, "original"),
                    }
                )
        packaging_report["extra_tracks"] = [item["title"] for item in extra_audio]
        movie = full_root / "result.mkv"
        temporary = movie.with_name(f".{movie.name}.{os.getpid()}.partial.mkv")
        remux_cfg = dict(store.cfg.get("full_processing") or {})
        command = audio_io.build_remux_command(
            audio_io.check_ffmpeg(),
            video,
            packaged_audio,
            temporary,
            audio_codec=str(remux_cfg.get("remux_audio_codec") or "aac"),
            audio_bitrate=str(remux_cfg.get("remux_audio_bitrate") or ""),
            audio_channels=int(remux_cfg.get("remux_audio_channels") or 0),
            audio_sample_rate=int(remux_cfg.get("remux_audio_sample_rate") or 0),
            track_title=(
                "Чистая русская речь без фона"
                if method == "voice_only"
                else (
                    "Русская дорожка с синхронизацией реплик"
                    if synchronize_speech
                    else "Русская дорожка после удаления английской речи"
                )
            ),
            extra_audio=extra_audio,
            keep_source_audio=include_all_tracks,
        )
        movie_dependencies = [
            video,
            packaged_audio,
            *[Path(item["path"]) for item in extra_audio],
        ]
        movie_recipe = "|".join(
            [
                _remux_recipe_signature(remux_cfg),
                video_source,
                "all_tracks" if include_all_tracks else "clean_only",
            ]
        )
        if not _full_output_cache_reusable(
            movie,
            movie_dependencies,
            force_recompute=force_recompute,
            recipe=movie_recipe,
        ):
            pipeline._run_ffmpeg_progress(
                command,
                # The picture decides how long the muxer runs: with the
                # original's picture the film is not the dubbed length.
                float(sf.info(str(packaged_audio)).duration) or duration,
                _ProgressRangeContext(ctx, 98.0, 2.0),
                "Упаковка итогового фильма",
                video.name,
            )
            os.replace(temporary, movie)
            _record_model_output_for(
                movie,
                movie_dependencies,
                recipe=movie_recipe,
            )
        report["movie"] = str(movie.resolve())
        report["packaging"] = packaging_report
        # The unsynchronised variant differs only by audio.  Re-copying the
        # complete multi-gigabyte video used to double the final packaging time
        # while the UI sat at 99%.  The player already combines one shared
        # picture with arbitrary FLAC tracks, so expose that audio directly.
        if "result_without_speech_synchronization" in intermediates:
            report["audio_without_speech_synchronization"] = user_files.get(
                "result_without_speech_synchronization",
                intermediates["result_without_speech_synchronization"],
            )
        _publish_full_application_progress(
            store,
            pair,
            full_root,
            method,
            intermediates=intermediates,
            audio=final,
            movie=movie,
        )
    manifest_path = full_root / "manifest.json"
    report["manifest"] = str(manifest_path.resolve())
    atomic_json(manifest_path, report)
    pair["application_result"] = report
    _publish_full_application_progress(
        store,
        pair,
        full_root,
        method,
        intermediates=intermediates,
        audio=final,
        movie=Path(str(report.get("movie") or "")) if report.get("movie") else None,
        state="completed",
    )
    store.save_pair(pair)
    ctx.update(
        progress=100.0,
        local_progress=100.0,
        stage="Полный фильм готов",
        substage="",
    )
    return report
