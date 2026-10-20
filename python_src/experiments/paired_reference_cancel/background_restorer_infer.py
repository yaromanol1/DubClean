#!/usr/bin/env python
"""Streaming inference for the learned speech-removal/background-restoration model."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.background_restorer_runner import (
    FORMAT,
    BackgroundRestorer,
)
from experiments.paired_reference_cancel.device_policy import (
    device_status_text,
    select_torch_device,
)


def read_audio(path: str | Path, target_rate: int) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != int(target_rate):
        divisor = math.gcd(int(rate), int(target_rate))
        data = np.stack(
            [
                signal.resample_poly(
                    data[:, channel],
                    target_rate // divisor,
                    int(rate) // divisor,
                ).astype(np.float32)
                for channel in range(data.shape[1])
            ],
            axis=1,
        )
    return np.nan_to_num(data, copy=False).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--hop-sec", type=float, default=3.0)
    parser.add_argument("--speech-gate-db", type=float, default=-50.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-cuda-free-gb", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device, device_details = select_torch_device(
        minimum_free_gib=max(0.0, float(args.min_cuda_free_gb)),
        force_cpu=bool(args.cpu),
        preference=args.device,
    )
    print(
        f"[status] Восстановление музыки и эффектов · "
        f"{device_status_text(device_details)}",
        flush=True,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("format") != FORMAT:
        raise RuntimeError("Файл не является моделью восстановления фонового звука.")
    config = state["configuration"]
    model = BackgroundRestorer(
        int(config.get("n_fft", 512)),
        int(config.get("hop_length", 128)),
        int(config.get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    rate = int(args.sample_rate)
    mixture = read_audio(args.mixture, rate)
    reference = read_audio(args.reference, rate).mean(axis=1)
    length = min(len(mixture), len(reference))
    mixture = mixture[:length]
    reference = reference[:length]
    window = int(round(args.window_sec * rate))
    hop = int(round(args.hop_sec * rate))
    overlap = window - hop
    if hop <= 0 or overlap < 0:
        raise ValueError("Некорректный шаг окна.")
    fade = np.ones(window, dtype=np.float64)
    if overlap:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
        fade[:overlap] = ramp
        fade[-overlap:] = ramp[::-1]
    result = np.zeros_like(mixture, dtype=np.float64)
    weights = np.zeros(length, dtype=np.float64)
    starts = list(range(0, max(1, length - overlap), hop))
    gate = 10.0 ** (float(args.speech_gate_db) / 20.0)
    with torch.no_grad():
        for index, start in enumerate(starts):
            stop = min(start + window, length)
            take = stop - start
            reference_chunk = np.zeros(window, dtype=np.float32)
            reference_chunk[:take] = reference[start:stop]
            block = max(1, int(round(rate * 0.10)))
            block_count = max(1, take // block)
            trimmed = reference_chunk[: block_count * block].reshape(
                block_count, block
            )
            # Gate by the loudest 100-ms block, not by the six-second average:
            # otherwise one or two words are mistaken for silence.
            speech_rms = float(
                np.max(
                    np.sqrt(
                        np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12
                    )
                )
            )
            outputs = np.zeros((window, mixture.shape[1]), dtype=np.float32)
            if speech_rms < gate:
                outputs[:take] = mixture[start:stop]
            else:
                reference_tensor = torch.from_numpy(reference_chunk)[None].to(device)
                for channel in range(mixture.shape[1]):
                    mixture_chunk = np.zeros(window, dtype=np.float32)
                    mixture_chunk[:take] = mixture[start:stop, channel]
                    outputs[:, channel] = (
                        model(
                            torch.from_numpy(mixture_chunk)[None].to(device),
                            reference_tensor,
                        )["background"][0]
                        .cpu()
                        .numpy()
                    )
            result[start:stop] += outputs[:take] * fade[:take, None]
            weights[start:stop] += fade[:take]
            if index % 10 == 0 or stop >= length:
                print(f"[progress] {stop / rate:.1f}/{length / rate:.1f} sec", flush=True)
            if stop >= length:
                break
    result /= np.maximum(weights[:, None], 1e-8)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Атомарная запись: см. semantic_separator_infer. Прерванная запись прямо
    # в файл назначения оставляет обрезанный FLAC с правильным заголовком.
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.partial{destination.suffix}"
    )
    try:
        sf.write(
            str(temporary),
            np.clip(result, -1.0, 1.0).astype(np.float32),
            rate,
            format="FLAC",
            subtype="PCM_16",
        )
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Готово: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
