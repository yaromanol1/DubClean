#!/usr/bin/env python
"""Train a reference-conditioned dialogue remover with exact background targets."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.recipe_audio import (
    _best_active_slice,
    _ducking_envelope,
    read_interval_mono,
)
from experiments.paired_reference_cancel.semantic_separator_runner import (
    SemanticRuSeparator,
    _complex_l1,
    _negative_si_sdr,
    _rms,
    _shift,
    _spectral_tilt,
)


FORMAT = "background_restorer_v1"
TRAINING_CLIP_SEC = 4.0


def emit(kind: str, **values) -> None:
    print(json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_audio(path: Path, value: np.ndarray, rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(value, -1.0, 1.0), rate, format="FLAC", subtype="PCM_16")
    return str(path.resolve())


def _voice_mastering(
    value: np.ndarray,
    rate: int,
    tilt_db: float,
    drive: float,
    reverb_wet: float,
    delay_ms: float,
) -> np.ndarray:
    output = _spectral_tilt(value, rate, tilt_db)
    if drive > 1.001:
        output = (np.tanh(output * drive) / np.tanh(drive)).astype(np.float32)
    if reverb_wet > 1e-5:
        length = max(16, int(round(rate * 0.16)))
        impulse = np.exp(-np.arange(length) / max(1.0, rate * 0.045))
        impulse[0] = 1.0
        reverberated = signal.fftconvolve(output, impulse, mode="full")[: len(output)]
        reverberated /= max(float(np.sum(np.abs(impulse))), 1.0)
        output = ((1.0 - reverb_wet) * output + reverb_wet * reverberated).astype(np.float32)
    return _shift(output, int(round(delay_ms * rate / 1000.0)))


def _active_rms(value: np.ndarray) -> float:
    """RMS over the speaking part only, so a short insert surrounded by silence
    is levelled by its actual loudness rather than diluted by the pauses."""
    peak = float(np.max(np.abs(value)))
    if peak < 1e-7:
        return _rms(value)
    active = value[np.abs(value) > 0.05 * peak]
    return _rms(active) if active.size else _rms(value)


def _place_short_insert(voice: np.ndarray, rate: int, recipe: dict) -> np.ndarray:
    """Keep only a short active slice of the voice, surrounded by silence, so the
    model must remove brief single words / short phrases and preserve the
    background in the pauses before and after (the reverb tail is added later by
    the voice mastering chain)."""
    frames = len(voice)
    duration = float(recipe.get("insert_duration_sec", 0.8))
    segment = max(1, min(int(round(duration * rate)), frames))
    left, right = _best_active_slice(voice, segment)
    piece = voice[left:right].copy()
    fade = min(int(round(0.02 * rate)), max(1, len(piece) // 4))
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    piece[:fade] *= ramp
    piece[-fade:] *= ramp[::-1]
    output = np.zeros(frames, dtype=np.float32)
    offset = float(recipe.get("insert_offset_sec", 0.0))
    start = max(0, min(frames - len(piece), int(round(offset * rate))))
    output[start : start + len(piece)] = piece
    return output


def synthesize_background_example(row: dict) -> tuple[np.ndarray, ...]:
    recipe = row["recipe"]
    rate = int(row["sample_rate"])
    duration = float(row["duration_sec"])
    frames = int(round(rate * duration))
    background = read_interval_mono(
        recipe["background_path"],
        float(recipe["background_start_sec"]),
        duration,
        rate,
        frames,
    )
    voice_source = read_interval_mono(
        recipe["voice_path"],
        float(recipe["voice_start_sec"]),
        duration,
        rate,
        frames,
    )
    rng = np.random.default_rng(int(recipe["seed"]))
    # Optionally keep only a short spoken insert (single word / short phrase)
    # with silence before and after.  The reverb tail of the mastering chain
    # then decays into the following pause, matching real dubbing artefacts.
    short_insert = str(recipe.get("voice_scenario", "sustained")) == "short_insert"
    if short_insert and not bool(recipe.get("identity_no_voice")):
        voice_source = _place_short_insert(voice_source, rate, recipe)
    # Vary the background mastering while keeping an exact target.
    background = _spectral_tilt(background, rate, float(rng.uniform(-5.0, 5.0)))
    background_drive = float(rng.uniform(1.0, 1.8))
    background = (
        np.tanh(background * background_drive) / np.tanh(background_drive)
    ).astype(np.float32)
    reference = _voice_mastering(
        voice_source,
        rate,
        float(recipe["reference_tilt_db"]),
        float(rng.uniform(1.0, 2.0)),
        float(recipe["reverb_wet"]) * float(rng.uniform(0.0, 0.35)),
        0.0,
    )
    embedded_voice = _voice_mastering(
        voice_source,
        rate,
        float(recipe["embedded_tilt_db"]),
        float(recipe["compression_drive"]),
        float(recipe["reverb_wet"]),
        float(recipe["voice_delay_ms"]),
    )
    if bool(recipe.get("identity_no_voice")):
        embedded_voice.fill(0.0)
        reference.fill(0.0)
    else:
        desired = max(_rms(background), 10.0 ** (-38.0 / 20.0)) * 10.0 ** (
            float(recipe["voice_to_background_db"]) / 20.0
        )
        # For a short insert, level by the spoken part so voice_to_background_db
        # stays the *local* ratio instead of being diluted by the surrounding
        # silence.  Sustained voices are unaffected (_active_rms ≈ _rms).
        level = _active_rms if short_insert else _rms
        embedded_voice *= desired / max(level(embedded_voice), 1e-7)
        reference *= desired / max(level(reference), 1e-7)
        # Speech extractors leak quiet music.  Add a controlled amount so the
        # model learns not to erase background merely because it is in reference.
        leak_gain = 10.0 ** (float(recipe["reference_leak_db"]) / 20.0)
        reference += background * leak_gain
    target_background = background.copy()
    ducking_db = float(recipe["ducking_db"])
    if ducking_db > 0.05 and np.max(np.abs(embedded_voice)) > 1e-7:
        envelope = _ducking_envelope(embedded_voice, rate)
        target_background *= np.power(10.0, -ducking_db * envelope / 20.0).astype(np.float32)
    mixture = target_background + embedded_voice
    crop_frames = int(round(TRAINING_CLIP_SEC * rate))
    if len(mixture) > crop_frames:
        maximum_start = len(mixture) - crop_frames
        if np.max(np.abs(embedded_voice)) > 1e-7:
            starts = np.linspace(0, maximum_start, num=9, dtype=np.int64)
            start = max(
                (int(value) for value in starts),
                key=lambda value: _rms(
                    embedded_voice[value : value + crop_frames]
                ),
            )
        else:
            start = int(rng.integers(0, maximum_start + 1))
        stop = start + crop_frames
        mixture = mixture[start:stop]
        reference = reference[start:stop]
        embedded_voice = embedded_voice[start:stop]
        target_background = target_background[start:stop]
    peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(reference))), 1e-7)
    rms_scale = (10.0 ** (-18.0 / 20.0)) / max(_rms(mixture), 1e-7)
    scale = min(0.98 / peak, rms_scale)
    return (
        (mixture * scale).astype(np.float32),
        (reference * scale).astype(np.float32),
        (embedded_voice * scale).astype(np.float32),
        (target_background * scale).astype(np.float32),
    )


class BackgroundDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        values = synthesize_background_example(row)
        return tuple(torch.from_numpy(value) for value in values), {
            "id": row["id"],
            "film_name": row["background_film"],
        }


class BackgroundRestorer(SemanticRuSeparator):
    """The same mastering-invariant U-Net, interpreted as a background mask."""

    def forward(self, mixture: torch.Tensor, reference: torch.Tensor) -> dict:
        value = super().forward(mixture, reference)
        return {
            "background": value["voice"],
            "removed_voice": value["reference"],
            "background_spectrum": value["voice_spectrum"],
            "removed_voice_spectrum": value["reference_spectrum"],
        }


def calculate_loss(
    model: BackgroundRestorer,
    output: dict,
    target_voice: torch.Tensor,
    target_background: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    background_spectrum = model.stft(target_background)
    voice_spectrum = model.stft(target_voice)
    background_spectral = _complex_l1(
        output["background_spectrum"], background_spectrum
    )
    voice_spectral = _complex_l1(output["removed_voice_spectrum"], voice_spectrum)
    background_wave = F.l1_loss(output["background"], target_background)
    voice_wave = F.l1_loss(output["removed_voice"], target_voice)
    background_sisdr = _negative_si_sdr(output["background"], target_background)
    residual_energy = torch.mean(
        (output["background"] - target_background) ** 2, dim=-1
    )
    voice_energy = torch.mean(target_voice**2, dim=-1)
    background_energy = torch.mean(target_background**2, dim=-1)
    normalized_error = torch.mean(
        residual_energy / (voice_energy + 0.05 * background_energy + 1e-7)
    )
    # Residual-speech penalty: project the reconstruction error onto the removed
    # voice waveform.  That projected energy is precisely the leftover speech
    # (the "English ghost") in the output background, so we penalise it directly
    # and relative to the voice energy that had to be removed.  This targets the
    # observed failure where speech is attenuated but not fully erased.
    residual = output["background"] - target_background
    voice_dot = torch.sum(residual * target_voice, dim=-1, keepdim=True)
    voice_norm = torch.sum(target_voice * target_voice, dim=-1, keepdim=True) + 1e-8
    # Squared regression coefficient of the error onto the voice: how much of the
    # removed voice (in units of the voice signal itself) survives in the output.
    speech_leak = torch.mean((voice_dot / voice_norm) ** 2)
    # On identity examples target_voice is zero, so this term strongly protects
    # music and effects from unnecessary edits.
    identity = voice_energy < 1e-8
    identity_error = (
        torch.mean(
            residual_energy[identity] / (background_energy[identity] + 1e-7)
        )
        if torch.any(identity)
        else torch.zeros((), device=target_voice.device)
    )
    total = (
        background_spectral
        + 0.45 * voice_spectral
        + 1.20 * background_wave
        + 0.25 * voice_wave
        + 0.003 * background_sisdr
        + 1.50 * normalized_error
        + 1.00 * speech_leak
        + 0.80 * identity_error
    )
    return total, {
        "background_spectral": float(background_spectral.detach().cpu()),
        "voice_spectral": float(voice_spectral.detach().cpu()),
        "background_wave": float(background_wave.detach().cpu()),
        "voice_wave": float(voice_wave.detach().cpu()),
        "background_si_sdr_loss": float(background_sisdr.detach().cpu()),
        "normalized_error": float(normalized_error.detach().cpu()),
        "speech_leak": float(speech_leak.detach().cpu()),
        "identity_error": float(identity_error.detach().cpu()),
    }


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float((-_negative_si_sdr(estimate, target)).detach().cpu())


def _stratified_by_bucket(rows: list[dict], maximum: int) -> list[dict]:
    """Round-robin across voice-loudness buckets so louder voices (the ones that
    left ghosts) are represented in the evaluation sample, not just the head of
    the list."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("hardcase_bucket", "unknown")), []).append(row)
    order = sorted(groups)
    selected: list[dict] = []
    depth = 0
    while len(selected) < maximum and order:
        progressed = False
        for bucket in order:
            group = groups[bucket]
            if depth < len(group):
                selected.append(group[depth])
                progressed = True
                if len(selected) >= maximum:
                    break
        if not progressed:
            break
        depth += 1
    return selected


@torch.no_grad()
def evaluate(model, rows: list[dict], device, maximum: int = 120) -> dict:
    model.eval()
    losses: list[float] = []
    suppression: list[float] = []
    preservation: list[float] = []
    identity_errors: list[float] = []
    leak_ratios: list[float] = []
    by_bucket: dict[str, list[float]] = {}
    selected = _stratified_by_bucket(rows, maximum)
    for left in range(0, len(selected), 4):
        batch_rows = selected[left : left + 4]
        examples = [synthesize_background_example(row) for row in batch_rows]
        tensors = [
            torch.from_numpy(np.stack([example[index] for example in examples])).to(device)
            for index in range(4)
        ]
        mixture, reference, voice, background = tensors
        output = model(mixture, reference)
        loss, _ = calculate_loss(model, output, voice, background)
        losses.extend([float(loss.cpu())] * len(batch_rows))
        for index, row in enumerate(batch_rows):
            source_error = torch.mean((mixture[index] - background[index]) ** 2)
            result_error = torch.mean(
                (output["background"][index] - background[index]) ** 2
            )
            if bool(row["recipe"].get("identity_no_voice")):
                identity_errors.append(float(result_error.cpu()))
            else:
                value = float(
                    10.0
                    * torch.log10((source_error + 1e-9) / (result_error + 1e-9)).cpu()
                )
                suppression.append(value)
                by_bucket.setdefault(str(row.get("hardcase_bucket", "unknown")), []).append(value)
                # Residual speech that survives in the output background.
                residual = output["background"][index] - background[index]
                dot = float(torch.sum(residual * voice[index]).cpu())
                norm = float(torch.sum(voice[index] * voice[index]).cpu()) + 1e-8
                leak_ratios.append((dot / norm) ** 2)
            preservation.append(
                si_sdr(
                    output["background"][index : index + 1],
                    background[index : index + 1],
                )
            )
    residual_speech_leak_db = (
        float(10.0 * np.log10(np.mean(leak_ratios) + 1e-9)) if leak_ratios else 0.0
    )
    return {
        "loss": float(np.mean(losses)),
        "speech_suppression_db": float(np.mean(suppression)) if suppression else 0.0,
        "background_si_sdr_db": float(np.mean(preservation)),
        "identity_mse": float(np.mean(identity_errors)) if identity_errors else 0.0,
        "residual_speech_leak_db": residual_speech_leak_db,
        "bucket_speech_suppression_db": {
            bucket: float(np.mean(values)) for bucket, values in sorted(by_bucket.items())
        },
        "examples": len(selected),
    }


@torch.no_grad()
def save_controls(model, manifest: dict, root: Path, epoch: int, device) -> dict:
    rows = [row for row in manifest["examples"] if row["split"] == "valid"]
    row = next(
        (
            value
            for value in rows[(epoch - 1) % len(rows) :] + rows[: (epoch - 1) % len(rows)]
            if not value["recipe"].get("identity_no_voice")
        ),
        rows[0],
    )
    mixture, reference, voice, background = synthesize_background_example(row)
    output = model(
        torch.from_numpy(mixture)[None].to(device),
        torch.from_numpy(reference)[None].to(device),
    )
    control_root = root / "controls" / f"epoch_{epoch:03d}"
    rate = int(row["sample_rate"])
    files = {
        "input_mix": write_audio(control_root / "input_mix.flac", mixture, rate),
        "speech_reference": write_audio(control_root / "speech_reference.flac", reference, rate),
        "exact_background": write_audio(control_root / "exact_background.flac", background, rate),
        "embedded_voice": write_audio(control_root / "embedded_voice.flac", voice, rate),
        "model_background": write_audio(
            control_root / "model_background.flac",
            output["background"][0].cpu().numpy(),
            rate,
        ),
        "model_removed_voice": write_audio(
            control_root / "model_removed_voice.flac",
            output["removed_voice"][0].cpu().numpy(),
            rate,
        ),
    }
    return {
        "example_id": row["id"],
        "background_film": row["background_film"],
        "voice_film": row["voice_film"],
        "files": files,
    }


def save_checkpoint(path, model, optimizer, epoch, step, manifest, config, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "format": FORMAT,
            "dataset_id": manifest["dataset_id"],
            "epoch": epoch,
            "global_step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "configuration": config,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    config = json.loads(args.config_json)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("target_kind") != "exact_synthetic_dialogue_removal_background_v1":
        raise RuntimeError("Выбран несовместимый датасет восстановления M&E.")
    run_root = Path(args.run_dir)
    checkpoint_root = Path(args.checkpoint_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    random.seed(81637)
    np.random.seed(81637)
    torch.manual_seed(81637)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emit(
        "log",
        message=f"Устройство: {device}"
        + (f" · {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""),
    )
    model = BackgroundRestorer(
        int(config.get("n_fft", 512)),
        int(config.get("hop_length", 256)),
        int(config.get("base_channels", 24)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 0.0004)),
        weight_decay=1e-5,
    )
    start_epoch = 0
    global_step = 0
    history: list[dict] = []
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("format") != FORMAT:
            raise RuntimeError("Сохранение относится к другой модели.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        if state.get("dataset_id") == manifest["dataset_id"]:
            history = list(state.get("history", []))
        else:
            # Fine-tuning on a different (harder) dataset: keep the weights and
            # optimizer state but reset metric history so best-checkpoint
            # selection is not frozen by the easier dataset's lower loss.
            history = []
            emit(
                "log",
                message=(
                    f"Смена датасета {state.get('dataset_id')} -> "
                    f"{manifest['dataset_id']}: история метрик сброшена."
                ),
            )
    rows = manifest["examples"]
    train_rows = [row for row in rows if row["split"] == "train"]
    valid_rows = [row for row in rows if row["split"] == "valid"]
    test_rows = [row for row in rows if row["split"] == "test"]
    workers = int(config.get("data_loader_workers", 3))
    loader = DataLoader(
        BackgroundDataset(train_rows),
        batch_size=int(config.get("batch_size", 8)),
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        pin_memory=device.type == "cuda",
    )
    total_steps = max(1, len(loader) * args.epochs)
    best_loss = min(
        (float(row["validation_loss"]) for row in history), default=float("inf")
    )
    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch + 1
        model.train()
        losses: list[float] = []
        sums: dict[str, float] = {}
        for batch_index, (values, metadata) in enumerate(loader):
            mixture, reference, voice, background = [
                value.to(device, non_blocking=True) for value in values
            ]
            optimizer.zero_grad(set_to_none=True)
            output = model(mixture, reference)
            loss, parts = calculate_loss(model, output, voice, background)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + value
            if batch_index % 5 == 0:
                emit(
                    "progress",
                    stage="Обучение удалению речи с сохранением M&E",
                    substage=f"Проход {epoch}, пакет {batch_index + 1} из {len(loader)}",
                    progress=(
                        (local_epoch * len(loader) + batch_index + 1)
                        / total_steps
                        * 100.0
                    ),
                    current_file=str(metadata.get("film_name", [""])[0]),
                    loss=losses[-1],
                )
        validation = evaluate(model, valid_rows, device)
        test = evaluate(model, test_rows, device)
        controls = save_controls(model, manifest, run_root, epoch, device)
        checkpoint = checkpoint_root / f"{run_root.name}_epoch{epoch}.pt"
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation["loss"],
            "test_loss": test["loss"],
            "validation_speech_suppression_db": validation["speech_suppression_db"],
            "test_speech_suppression_db": test["speech_suppression_db"],
            "validation_background_si_sdr_db": validation["background_si_sdr_db"],
            "test_background_si_sdr_db": test["background_si_sdr_db"],
            "validation_identity_mse": validation["identity_mse"],
            "test_identity_mse": test["identity_mse"],
            "validation_residual_speech_leak_db": validation["residual_speech_leak_db"],
            "test_residual_speech_leak_db": test["residual_speech_leak_db"],
            "validation_bucket_speech_suppression_db": validation["bucket_speech_suppression_db"],
            "test_bucket_speech_suppression_db": test["bucket_speech_suppression_db"],
            "loss_parts": {key: value / max(1, len(losses)) for key, value in sums.items()},
            "control_example": controls,
            "checkpoint": str(checkpoint.resolve()),
            "dataset_id": manifest["dataset_id"],
            "algorithm_version": FORMAT,
        }
        history.append(row)
        save_checkpoint(
            checkpoint, model, optimizer, epoch, global_step, manifest, config, history
        )
        save_checkpoint(
            checkpoint_root / "background_restorer_last.pt",
            model,
            optimizer,
            epoch,
            global_step,
            manifest,
            config,
            history,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(
                checkpoint_root / "background_restorer_best.pt",
                model,
                optimizer,
                epoch,
                global_step,
                manifest,
                config,
                history,
            )
        best = min(history, key=lambda value: float(value["validation_loss"]))
        atomic_json(
            run_root / "history.json",
            {
                "schema_version": 1,
                "run_id": run_root.name,
                "dataset_id": manifest["dataset_id"],
                "algorithm_version": FORMAT,
                "device": str(device),
                "history": history,
                "best_epoch": best,
                "recommended_checkpoint": best["checkpoint"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        emit(
            "log",
            message=(
                f"Эпоха {epoch}: подавление речи {test['speech_suppression_db']:.2f} дБ, "
                f"сохранность фона {test['background_si_sdr_db']:.2f} дБ"
            ),
        )
    emit("complete", history=str((run_root / "history.json").resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
