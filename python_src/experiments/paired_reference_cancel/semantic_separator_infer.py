#!/usr/bin/env python
"""Streaming inference for the mastering-invariant semantic RU separator."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

# This file is launched by its absolute path from the application pipeline.
# Put python_src on sys.path before importing any package modules.
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from experiments.paired_reference_cancel.device_policy import (
    device_status_text,
    select_torch_device,
)


_SPEC = importlib.util.spec_from_file_location(
    "semantic_separator_runner", MODULE_DIR / "semantic_separator_runner.py"
)
_RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)


def load_checkpoint_model(state: dict, device: torch.device) -> torch.nn.Module:
    """Load the single production DubClean Voice architecture."""
    checkpoint_format = state.get("format")
    config = state.get("configuration") or {}
    if checkpoint_format != _RUNNER.FORMAT:
        raise RuntimeError("Файл не является поддерживаемым сохранением RU/EN-разделителя.")
    model = _RUNNER.build_semantic_model(config)
    model = model.to(device)
    model.load_state_dict(state["model_state"])
    return model


def read_mono(path: str | Path, target_rate: int) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if int(rate) != int(target_rate):
        divisor = math.gcd(int(rate), int(target_rate))
        mono = signal.resample_poly(
            mono, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32)
    return mono


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--window-sec", type=float, default=6.0)
    parser.add_argument("--hop-sec", type=float, default=3.0)
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
    model = load_checkpoint_model(state, device)
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
    overlap = window - hop
    fade = np.ones(window, dtype=np.float64)
    if overlap > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(overlap) / overlap)
        fade[:overlap] = ramp
        fade[-overlap:] = ramp[::-1]
    result = np.zeros(length, dtype=np.float64)
    weights = np.zeros(length, dtype=np.float64)
    starts = list(range(0, max(1, length - overlap), hop))
    with torch.no_grad():
        for index, start in enumerate(starts):
            stop = min(start + window, length)
            take = stop - start
            mix_chunk = np.zeros(window, dtype=np.float32)
            ref_chunk = np.zeros(window, dtype=np.float32)
            mix_chunk[:take] = mixture[start:stop]
            ref_chunk[:take] = reference[start:stop]
            mixture_tensor = torch.from_numpy(mix_chunk)[None].to(device)
            reference_tensor = torch.from_numpy(ref_chunk)[None].to(device)
            model_output = model(mixture_tensor, reference_tensor)
            output_tensor = model_output["voice"]
            output = output_tensor[0].cpu().numpy().astype(np.float64)
            result[start:stop] += output[:take] * fade[:take]
            weights[start:stop] += fade[:take]
            if index % 10 == 0 or stop >= length:
                print(
                    f"[progress] {stop / rate:.1f}/{length / rate:.1f} sec",
                    flush=True,
                )
            if stop >= length:
                break
    result /= np.maximum(weights, 1e-8)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Атомарная запись. Убитый на этом месте процесс иначе оставлял FLAC, чей
    # заголовок уже обещает полную длительность, а кадров в нём меньше: файл
    # выглядел готовым, проходил проверку кэша и валил каждую пересборку на
    # одном и том же месте.
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.partial{destination.suffix}"
    )
    try:
        sf.write(
            str(temporary),
            np.clip(result, -1.0, 1.0).astype(np.float32),
            rate,
            format="FLAC",
            subtype="PCM_24",
        )
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Готово: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
