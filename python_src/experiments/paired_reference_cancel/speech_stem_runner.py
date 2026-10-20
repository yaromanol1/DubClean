#!/usr/bin/env python
"""Blockwise 48 kHz speech extraction using ClearVoice MossFormer2.

This runner is intentionally executed by the existing ClearVoice environment.
It never loads a full film into memory and writes its result atomically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

# The pipeline launches this file by its absolute path. In that mode Python only
# adds this script's directory to sys.path, not the python_src package root.
PYTHON_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC_ROOT))

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from experiments.paired_reference_cancel.device_policy import (
    device_status_text,
    select_torch_device,
)


def load_model(checkpoint_dir: Path, cpu: bool):
    import torch
    from clearvoice.network_wrapper import network_wrapper

    wrapper = network_wrapper()
    wrapper.model_name = "MossFormer2_SE_48K"
    wrapper.load_args_se()
    wrapper.args.checkpoint_dir = str(checkpoint_dir)
    wrapper.args.use_cuda = 0 if cpu else 1
    from clearvoice.networks import CLS_MossFormer2_SE_48K

    context = patch("torch.cuda.is_available", return_value=False) if cpu else nullcontext()
    with context:
        return CLS_MossFormer2_SE_48K(wrapper.args)


def select_dialogue_channel(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.shape[1] >= 3:
        return data[:, 2]
    return np.mean(data, axis=1, dtype=np.float32)


def fit_length(data: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(data, dtype=np.float32).reshape(-1)
    if len(value) < length:
        value = np.pad(value, (0, length - len(value)))
    return value[:length]


def enhance(model, mono_48k: np.ndarray) -> np.ndarray:
    import torch
    from clearvoice.utils.decode import decode_one_audio_mossformer2_se_48k

    with torch.inference_mode():
        result = decode_one_audio_mossformer2_se_48k(
            model.model,
            model.device,
            np.asarray(mono_48k, dtype=np.float32).reshape(1, -1),
            model.args,
        )
    return fit_length(result, len(mono_48k))


def _chunk_starts(total_frames: int, chunk_frames: int, overlap_frames: int) -> list[int]:
    """Return core-window starts without creating a redundant final window."""

    total_frames = max(0, int(total_frames))
    chunk_frames = max(1, int(chunk_frames))
    overlap_frames = max(0, min(int(overlap_frames), chunk_frames - 1))
    if total_frames <= chunk_frames:
        return [0]
    step_frames = chunk_frames - overlap_frames
    starts = [0]
    while starts[-1] + chunk_frames < total_frames:
        starts.append(starts[-1] + step_frames)
    return starts


def _equal_power_crossfade(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Blend adjacent estimates with correlation-aware equal-power ramps."""

    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if len(left) != len(right):
        raise ValueError("Crossfade inputs must have equal lengths.")
    if not len(left):
        return np.empty(0, dtype=np.float32)
    if len(left) == 1:
        return ((left + right) * 0.5).astype(np.float32, copy=False)
    phase = np.linspace(0.0, np.pi / 2.0, len(left), dtype=np.float32)
    fade_out = np.cos(phase).astype(np.float32, copy=False)
    fade_in = np.sin(phase).astype(np.float32, copy=False)
    # Adjacent model estimates usually contain the same waveform.  A raw
    # equal-power sum would boost identical material by 3 dB in the middle of
    # every overlap.  Compensate only the measured positive correlation; for
    # unrelated estimates this remains a conventional equal-power crossfade.
    energy_product = float(np.dot(left, left) * np.dot(right, right))
    correlation = 0.0
    if energy_product > 1e-16:
        correlation = float(np.dot(left, right) / np.sqrt(energy_product))
        correlation = min(1.0, max(0.0, correlation))
    normalization = np.sqrt(
        np.maximum(
            1e-8,
            fade_out * fade_out
            + fade_in * fade_in
            + 2.0 * correlation * fade_out * fade_in,
        )
    ).astype(np.float32, copy=False)
    return ((left * fade_out + right * fade_in) / normalization).astype(
        np.float32,
        copy=False,
    )


class _StreamingCrossfadeWriter:
    """Write overlapping chunks while retaining at most one chunk in memory."""

    def __init__(self, writer, overlap_frames: int):
        self.writer = writer
        self.overlap_frames = max(0, int(overlap_frames))
        self.pending = np.empty(0, dtype=np.float32)
        self.frames_written = 0

    def _write(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if not len(values):
            return
        self.writer.write(np.clip(values, -1.0, 1.0)[:, None])
        self.frames_written += len(values)

    def push(self, chunk: np.ndarray, actual_overlap_frames: int) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not len(self.pending):
            self.pending = chunk.copy()
            return
        overlap = max(
            0,
            min(
                int(actual_overlap_frames),
                self.overlap_frames,
                len(self.pending),
                len(chunk),
            ),
        )
        if not overlap:
            self._write(self.pending)
            self.pending = chunk.copy()
            return
        self._write(self.pending[:-overlap])
        self._write(_equal_power_crossfade(self.pending[-overlap:], chunk[:overlap]))
        self.pending = chunk[overlap:].copy()

    def finish(self) -> int:
        self._write(self.pending)
        self.pending = np.empty(0, dtype=np.float32)
        return self.frames_written


def stream_enhance(
    reader,
    writer,
    *,
    model,
    source_sr: int,
    start_sec: float,
    duration_sec: float,
    target_sr: int = 48000,
    chunk_sec: float = 20.0,
    context_sec: float = 1.0,
    enhancer=None,
    progress_callback=None,
) -> int:
    """Enhance an interval with bounded-memory overlap-add.

    ``context_sec`` keeps its original purpose: input outside each core window is
    supplied to the model and then trimmed.  The same duration is also used as
    the overlap between adjacent core estimates, so their independently inferred
    edges are never concatenated directly.
    """

    source_sr = max(1, int(source_sr))
    target_sr = max(1, int(target_sr))
    total_frames = int(round(max(0.0, float(duration_sec)) * target_sr))
    if total_frames <= 0:
        raise ValueError("Пустой интервал для выделения речи.")
    chunk_frames = max(1, int(round(max(0.0, float(chunk_sec)) * target_sr)))
    context_frames = max(0, int(round(max(0.0, float(context_sec)) * target_sr)))
    overlap_frames = min(context_frames, max(0, chunk_frames - 1))
    starts = _chunk_starts(total_frames, chunk_frames, overlap_frames)
    enhance_block = enhancer or enhance
    output = _StreamingCrossfadeWriter(writer, overlap_frames)
    previous_end = 0

    for index, core_start in enumerate(starts):
        core_end = min(total_frames, core_start + chunk_frames)
        process_start = max(0, core_start - context_frames)
        process_end = min(total_frames, core_end + context_frames)
        absolute_start_sec = float(start_sec) + process_start / target_sr
        absolute_end_sec = float(start_sec) + process_end / target_sr
        source_start = int(round(absolute_start_sec * source_sr))
        source_end = int(round(absolute_end_sec * source_sr))
        reader.seek(source_start)
        data = reader.read(
            max(0, source_end - source_start),
            dtype="float32",
            always_2d=True,
        )
        mono = select_dialogue_channel(data)
        if source_sr != target_sr:
            divisor = int(np.gcd(source_sr, target_sr))
            mono = resample_poly(mono, target_sr // divisor, source_sr // divisor)
        expected = process_end - process_start
        mono = fit_length(mono, expected)
        speech = enhance_block(model, mono)
        left = core_start - process_start
        core = fit_length(speech[left : left + (core_end - core_start)], core_end - core_start)
        actual_overlap = max(0, previous_end - core_start) if index else 0
        output.push(core, actual_overlap)
        previous_end = core_end
        if progress_callback is not None:
            progress_callback(
                {
                    "block": index + 1,
                    "blocks": len(starts),
                    "processed_sec": round(core_end / target_sr, 3),
                    "total_sec": round(total_frames / target_sr, 3),
                }
            )

    frames_written = output.finish()
    if frames_written != total_frames:
        raise RuntimeError(
            "Нарушена длина результата MossFormer2: "
            f"ожидалось {total_frames}, записано {frames_written}."
        )
    return len(starts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--chunk-sec", type=float, default=20.0)
    parser.add_argument("--context-sec", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-cuda-free-gb", type=float, default=2.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not (checkpoint_dir / "last_best_checkpoint").is_file():
        raise FileNotFoundError(f"Веса MossFormer2 не найдены: {checkpoint_dir}")

    target_sr = 48000
    with sf.SoundFile(str(source), "r") as reader:
        source_sr = int(reader.samplerate)
        source_total_sec = reader.frames / max(source_sr, 1)
        start_sec = min(max(0.0, args.start), source_total_sec)
        duration_sec = (
            min(max(0.0, args.duration), source_total_sec - start_sec)
            if args.duration > 0
            else source_total_sec - start_sec
        )
    if duration_sec <= 0:
        raise ValueError("Пустой интервал для выделения речи.")

    selected_device, device_details = select_torch_device(
        minimum_free_gib=max(0.0, float(args.min_cuda_free_gb)),
        force_cpu=bool(args.cpu),
        preference=args.device,
    )
    use_cpu = selected_device.type == "cpu"
    print(
        f"[status] Загрузка модуля извлечения речи · "
        f"{device_status_text(device_details)}",
        flush=True,
    )
    model = load_model(checkpoint_dir, use_cpu)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.partial{destination.suffix}"
    )
    chunk_sec = max(5.0, float(args.chunk_sec))
    context_sec = max(0.0, min(2.0, float(args.context_sec)))
    try:
        with sf.SoundFile(
            str(temporary),
            "w",
            samplerate=target_sr,
            channels=1,
            format="FLAC",
            subtype="PCM_16",
        ) as writer, sf.SoundFile(str(source), "r") as reader:
            stream_enhance(
                reader,
                writer,
                model=model,
                source_sr=source_sr,
                start_sec=start_sec,
                duration_sec=duration_sec,
                target_sr=target_sr,
                chunk_sec=chunk_sec,
                context_sec=context_sec,
                progress_callback=lambda payload: print(
                    "[progress] " + json.dumps(payload, ensure_ascii=False),
                    flush=True,
                ),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[result] {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
