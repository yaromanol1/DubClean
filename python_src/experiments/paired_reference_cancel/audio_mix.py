"""Shared deterministic M&E + voice mixing for preview and full-film output."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import audio_io


def _linear_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def matched_audio_blocks(
    background_reader: sf.SoundFile,
    voice_reader: sf.SoundFile,
    blocksize: int,
    *,
    voice_delay_sec: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield same-length blocks with a sample-exact global voice delay.

    Positive delay moves the voice later; negative delay moves it earlier.
    Both ends are silence-padded and the background length is authoritative.
    """
    rate = int(background_reader.samplerate)
    voice_rate = int(voice_reader.samplerate)
    if rate != voice_rate:
        raise RuntimeError(
            "Сведение требует одинаковой частоты фоновой и речевой дорожек."
        )
    shift_frames = int(round(float(voice_delay_sec) * rate))
    remaining = len(background_reader)
    background_cursor = 0
    voice_channels = int(voice_reader.channels)
    while remaining > 0:
        frames = min(blocksize, remaining)
        background_block = background_reader.read(
            frames,
            dtype="float32",
            always_2d=True,
        )
        expected = len(background_block)
        voice_block = np.zeros(
            (expected, voice_channels),
            dtype=np.float32,
        )
        source_left = background_cursor - shift_frames
        source_right = source_left + expected
        readable_left = max(0, source_left)
        readable_right = min(len(voice_reader), source_right)
        if readable_right > readable_left:
            voice_reader.seek(readable_left)
            values = voice_reader.read(
                readable_right - readable_left,
                dtype="float32",
                always_2d=True,
            )
            destination_left = readable_left - source_left
            destination_right = min(
                expected,
                destination_left + len(values),
            )
            voice_block[destination_left:destination_right] = values[
                : destination_right - destination_left
            ]
        yield background_block, voice_block
        remaining -= expected
        background_cursor += expected


def mix_audio_files(
    background: Path,
    voice: Path,
    destination: Path,
    *,
    background_gain_db: float = 0.0,
    voice_gain_db: float = 0.0,
    voice_delay_sec: float = 0.0,
    clip_limit: float = 0.98,
    check_stop: Callable[[], Any] | None = None,
    progress: Callable[[float], Any] | None = None,
) -> dict[str, Any]:
    """Mix with full-pipeline channel mapping, gains and hard sample clipping."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    background_info = sf.info(str(background))
    voice_info = sf.info(str(voice))
    if int(background_info.samplerate) != int(voice_info.samplerate):
        raise RuntimeError(
            "Сведение требует одинаковой частоты фоновой и речевой дорожек."
        )
    limit = float(np.clip(float(clip_limit), 1e-6, 1.0))
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.partial"
    )
    background_gain = _linear_gain(background_gain_db)
    voice_gain = _linear_gain(voice_gain_db)
    peak = 0.0
    clipped_samples = 0
    total_samples = 0
    processed_frames = 0
    total_frames = max(1, int(background_info.frames))
    try:
        with sf.SoundFile(str(background)) as background_reader:
            with sf.SoundFile(str(voice)) as voice_reader:
                rate = int(background_reader.samplerate)
                channels = int(background_reader.channels)
                with sf.SoundFile(
                    str(temporary),
                    mode="w",
                    samplerate=rate,
                    channels=channels,
                    format="FLAC",
                    subtype="PCM_24",
                ) as writer:
                    for background_block, voice_block in matched_audio_blocks(
                        background_reader,
                        voice_reader,
                        rate * 20,
                        voice_delay_sec=voice_delay_sec,
                    ):
                        if check_stop is not None:
                            check_stop()
                        mixed = (
                            background_gain * background_block
                            + voice_gain
                            * audio_io.match_channels(voice_block, channels)
                        )
                        if len(mixed):
                            peak = max(
                                peak,
                                float(np.max(np.abs(mixed))),
                            )
                            clipped_samples += int(
                                np.count_nonzero(np.abs(mixed) > limit)
                            )
                            total_samples += int(np.size(mixed))
                        writer.write(np.clip(mixed, -limit, limit))
                        processed_frames += len(background_block)
                        if progress is not None:
                            progress(
                                min(
                                    100.0,
                                    processed_frames
                                    / total_frames
                                    * 100.0,
                                )
                            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "background": str(background.resolve()),
        "voice": str(voice.resolve()),
        "destination": str(destination.resolve()),
        "background_gain_db": float(background_gain_db),
        "voice_gain_db": float(voice_gain_db),
        "voice_delay_sec": float(voice_delay_sec),
        "pre_limiter_peak": float(peak),
        "peak_limiter": f"sample_clip_at_{limit:.2f}",
        "limited_samples": int(clipped_samples),
        "total_samples": int(total_samples),
        "limited_samples_ratio": float(
            clipped_samples / max(total_samples, 1)
        ),
        "output_gain_db": 0.0,
        "duration_sec": float(
            int(background_info.frames)
            / max(int(background_info.samplerate), 1)
        ),
        "channels": int(background_info.channels),
        "sample_rate": int(background_info.samplerate),
        "finite": bool(math.isfinite(float(peak))),
    }
