#!/usr/bin/env python
"""Apply a trained speech reference subtractor to full-length tracks.

Inputs are two synchronised files: the EN+RU speech mixture and the aligned
original English speech stem.  The track is processed in overlapping windows;
each window is aligned to the reference by the model itself (GCC-PHAT inside
forward) and the voice outputs are cross-faded back together.

    python reference_subtractor_infer.py \
        --checkpoint .../speech_reference_subtractor_best.pt \
        --mixture mixture.flac --reference en_speech.flac --output ru_voice.flac
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

import importlib.util

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.device_policy import (
    device_status_text,
    select_torch_device,
)

_SPEC = importlib.util.spec_from_file_location(
    "reference_training_runner", Path(__file__).with_name("reference_training_runner.py")
)
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def read_mono(path: str | Path, target_rate: int) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        mono = signal.resample_poly(
            mono, target_rate // divisor, rate // divisor
        ).astype(np.float32)
    return mono


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--hop-sec", type=float, default=4.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-cuda-free-gb", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device, device_details = select_torch_device(
        minimum_free_gib=max(0.0, float(args.min_cuda_free_gb)),
        force_cpu=bool(args.cpu),
        preference=args.device,
    )
    print(f"[status] Очистка перевода · {device_status_text(device_details)}", flush=True)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if state.get("format") not in {
        "speech_reference_subtractor_v3",
        "speech_reference_subtractor_v4",
    }:
        raise RuntimeError("Файл не является сохранением модели парного вычитания.")
    config = state["configuration"]
    model = _RUNNER.ReferenceSubtractor(
        int(config["n_fft"]), int(config["hop_length"]), int(config["base_channels"])
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.sample_rate_hint = int(args.sample_rate)
    model.eval()

    rate = int(args.sample_rate)
    mixture = read_mono(args.mixture, rate)
    reference = read_mono(args.reference, rate)
    length = min(len(mixture), len(reference))
    mixture, reference = mixture[:length], reference[:length]

    window = int(round(args.window_sec * rate))
    hop = int(round(args.hop_sec * rate))
    if hop <= 0 or hop > window:
        raise ValueError("hop-sec должен быть больше нуля и не больше window-sec.")
    fade = np.ones(window, dtype=np.float64)
    overlap = window - hop
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
        fade[:overlap] = ramp
        fade[-overlap:] = ramp[::-1]
    result = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, max(1, length - overlap), hop):
            stop = min(start + window, length)
            mix_chunk = np.zeros(window, dtype=np.float32)
            ref_chunk = np.zeros(window, dtype=np.float32)
            mix_chunk[: stop - start] = mixture[start:stop]
            ref_chunk[: stop - start] = reference[start:stop]
            output = model(
                torch.from_numpy(mix_chunk)[None].to(device),
                torch.from_numpy(ref_chunk)[None].to(device),
            )
            voice = output["voice"][0].cpu().numpy().astype(np.float64)
            take = stop - start
            result[start:stop] += voice[:take] * fade[:take]
            weight[start:stop] += fade[:take]
            if stop >= length:
                break
    result /= np.maximum(weight, 1e-8)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(destination),
        np.clip(result, -1.0, 1.0).astype(np.float32),
        rate,
        format="FLAC",
        subtype="PCM_16",
    )
    print(f"Готово: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
