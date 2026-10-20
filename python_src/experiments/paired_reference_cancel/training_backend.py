"""Small, auditable band-gain model used when only pseudo-targets are available.

This is deliberately not presented as a speech recognizer or separator.  It
learns a global frequency-dependent transfer from the aligned reference to the
common component estimated by the deterministic baseline.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel import audio_io, diagnostics
from experiments.paired_reference_cancel.storage import (
    Store,
    atomic_json,
    read_json,
    root_path,
    utc_now,
)

FORMAT = "paired_band_gain_calibrator_v1"


def _mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.mean(data, axis=1).astype(np.float32), int(sr)


def _band_sos(low: float, high: float, sample_rate: int):
    nyquist = sample_rate / 2.0
    low = max(1.0, low)
    high = min(high, nyquist * 0.98)
    if low >= high:
        return None
    return signal.butter(3, [low / nyquist, high / nyquist], btype="bandpass", output="sos")


def _bands(reference: np.ndarray, sample_rate: int, edges: list[float]) -> list[np.ndarray]:
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        sos = _band_sos(float(low), float(high), sample_rate)
        result.append(
            signal.sosfilt(sos, reference).astype(np.float32)
            if sos is not None
            else np.zeros_like(reference)
        )
    return result


def _loss_and_grad(
    mixture: np.ndarray,
    reference: np.ndarray,
    pseudo_target: np.ndarray,
    sample_rate: int,
    edges: list[float],
    gains: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    length = min(len(mixture), len(reference), len(pseudo_target))
    mixture = mixture[:length]
    reference = reference[:length]
    pseudo_target = pseudo_target[:length]
    common_target = mixture - pseudo_target
    components = _bands(reference, sample_rate, edges)
    predicted = np.zeros(length, dtype=np.float32)
    for gain, component in zip(gains, components):
        predicted += float(gain) * component[:length]
    error = predicted - common_target
    loss = float(np.mean(error.astype(np.float64) ** 2))
    grad = np.asarray(
        [2.0 * np.mean(error * component[:length]) for component in components],
        dtype=np.float64,
    )
    estimate = mixture - predicted
    return loss, grad, estimate.astype(np.float32)


def _checkpoint_payload(
    dataset_id: str,
    run_id: str,
    epoch: int,
    global_step: int,
    edges: list[float],
    gains: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    history: list[dict],
    cfg: dict,
) -> dict:
    return {
        "format": FORMAT,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "algorithm_version": FORMAT,
        "model_state": {"gains": gains.tolist(), "band_edges_hz": edges},
        "optimizer_state": {"name": "Adam", "m": m.tolist(), "v": v.tolist()},
        "scaler_state": {"enabled": False, "scale": 1.0},
        "configuration": {
            "learning_rate": float(cfg["training"]["learning_rate"]),
            "batch_size": int(cfg["training"]["batch_size"]),
        },
        "metrics_history": history,
        "saved_at": utc_now(),
    }


def _atomic_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                gains=np.asarray(payload["model_state"]["gains"], dtype=np.float64),
                optimizer_m=np.asarray(payload["optimizer_state"]["m"], dtype=np.float64),
                optimizer_v=np.asarray(payload["optimizer_state"]["v"], dtype=np.float64),
                metadata=np.asarray(json.dumps(payload, ensure_ascii=False)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict:
    with np.load(str(path), allow_pickle=False) as archive:
        payload = json.loads(str(archive["metadata"].item()))
        payload["model_state"]["gains"] = archive["gains"].astype(float).tolist()
        payload["optimizer_state"]["m"] = archive["optimizer_m"].astype(float).tolist()
        payload["optimizer_state"]["v"] = archive["optimizer_v"].astype(float).tolist()
    if payload.get("format") != FORMAT:
        raise RuntimeError("Checkpoint создан несовместимым алгоритмом.")
    return payload


def _evaluate(rows: list[dict], gains: np.ndarray, edges: list[float], ctx: Any) -> tuple[float, float]:
    losses = []
    improvements = []
    for row in rows:
        ctx.check_stop()
        mixture, sr = _mono(row["files"]["mixture"])
        reference, ref_sr = _mono(row["files"]["aligned_original"])
        target, target_sr = _mono(row["files"]["pseudo_target"])
        if ref_sr != sr:
            reference = np.asarray(audio_io.resample(reference, ref_sr, sr)).reshape(-1)
        if target_sr != sr:
            target = np.asarray(audio_io.resample(target, target_sr, sr)).reshape(-1)
        loss, _grad, estimate = _loss_and_grad(
            mixture, reference, target, sr, edges, gains
        )
        length = min(len(estimate), len(target), len(mixture))
        losses.append(loss)
        improvements.append(
            diagnostics.si_snr(estimate[:length], target[:length])
            - diagnostics.si_snr(mixture[:length], target[:length])
        )
    return (
        float(np.mean(losses)) if losses else float("nan"),
        float(np.mean(improvements)) if improvements else float("nan"),
    )


def _save_control_example(
    row: dict,
    gains: np.ndarray,
    edges: list[float],
    destination: Path,
) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    mixture, sample_rate = _mono(row["files"]["mixture"])
    reference, reference_rate = _mono(row["files"]["aligned_original"])
    target, target_rate = _mono(row["files"]["pseudo_target"])
    if reference_rate != sample_rate:
        reference = np.asarray(
            audio_io.resample(reference, reference_rate, sample_rate)
        ).reshape(-1)
    if target_rate != sample_rate:
        target = np.asarray(audio_io.resample(target, target_rate, sample_rate)).reshape(-1)
    _loss, _grad, estimate = _loss_and_grad(
        mixture, reference, target, sample_rate, edges, gains
    )
    length = min(len(mixture), len(reference), len(target), len(estimate))
    values = {
        "mixture": mixture[:length],
        "aligned_original": reference[:length],
        "baseline_result": target[:length],
        "model_result": estimate[:length],
        "pseudo_target": target[:length],
    }
    files: dict[str, str] = {}
    for name, data in values.items():
        path = destination / f"{name}.flac"
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial.flac")
        try:
            sf.write(
                str(temporary),
                np.clip(data, -1.0, 1.0),
                sample_rate,
                format="FLAC",
                subtype="PCM_24",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        files[name] = str(path.resolve())
    return {
        "example_id": row["id"],
        "film_name": row["film_name"],
        "dubbed_start_sec": row["dubbed_start_sec"],
        "files": files,
        "warning": "Базовый результат и pseudo-target совпадают по происхождению.",
    }


def train_legacy(
    store: Store,
    project_id: str,
    ctx: Any,
    parameters: dict,
) -> dict:
    project = store.load_project(project_id)
    dataset_id = str(parameters.get("dataset_id") or project.get("latest_dataset_id") or "")
    if not dataset_id:
        raise RuntimeError("Сначала подготовьте датасет.")
    manifest_path = store.project_dir(project_id) / "training" / "datasets" / dataset_id / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest:
        raise RuntimeError("Манифест датасета не найден.")
    if manifest.get("target_kind") in {
        "neural_ru_speech_pseudo_target",
        "ru_voice_from_en_ru_speech_minus_en_speech",
        "selected_method_audio_outputs",
    }:
        raise RuntimeError(
            "Этот датасет создан новой компонентной цепочкой MossFormer2. "
            "Старый полосовой калибратор с ней несовместим и намеренно отключён. "
            "Проверьте контрольные сцены на этапе 9; отдельное дообучение "
            "потребует речевой модели и размеченного чистого эталона."
        )
    mode = str(parameters.get("mode", "quick"))
    epochs = int(parameters.get("epochs", 1))
    allowed = (
        [1]
        if mode == "quick"
        else list(store.cfg["training"]["short_epoch_choices"] if mode == "short" else store.cfg["training"]["full_epoch_choices"])
    )
    if epochs not in allowed:
        raise ValueError("Недопустимое количество дополнительных проходов.")
    train_rows = [row for row in manifest["examples"] if row["split"] == "train"]
    valid_rows = [row for row in manifest["examples"] if row["split"] == "valid"]
    if not train_rows:
        raise RuntimeError("В обучающей выборке нет примеров.")
    validation_independent = True
    if not valid_rows:
        valid_rows = [row for row in manifest["examples"] if row["split"] == "test"][
            : max(1, len(train_rows) // 4)
        ]
    if not valid_rows:
        if mode != "quick":
            raise RuntimeError(
                "Нет независимой выборки проверки. Добавьте фильмы или сцены так, "
                "чтобы датасет содержал раздел valid."
            )
        valid_rows = train_rows[:1]
        validation_independent = False
    edges = [float(item) for item in store.cfg["training"]["band_edges_hz"]]
    gains = np.ones(len(edges) - 1, dtype=np.float64)
    m = np.zeros_like(gains)
    v = np.zeros_like(gains)
    history: list[dict] = []
    start_epoch = 0
    global_step = 0
    resume = parameters.get("checkpoint")
    if resume:
        checkpoint_path = Path(str(resume)).resolve()
        checkpoint_root = store.project_dir(project_id) / "checkpoints"
        if not checkpoint_path.is_file() or checkpoint_root.resolve() not in checkpoint_path.parents:
            raise ValueError("Checkpoint должен находиться внутри текущего проекта.")
        state = load_checkpoint(checkpoint_path)
        if state["dataset_id"] != dataset_id:
            raise RuntimeError("Checkpoint относится к другой версии датасета.")
        gains = np.asarray(state["model_state"]["gains"], dtype=np.float64)
        m = np.asarray(state["optimizer_state"]["m"], dtype=np.float64)
        v = np.asarray(state["optimizer_state"]["v"], dtype=np.float64)
        history = list(state.get("metrics_history", []))
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_root = store.project_dir(project_id) / "training" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    lr = float(store.cfg["training"]["learning_rate"])
    beta1, beta2 = 0.9, 0.999
    quick_limit = int(store.cfg["training"]["quick_check_batches"])
    total_batches = min(len(train_rows), quick_limit) if mode == "quick" else len(train_rows)
    for local_epoch in range(epochs):
        epoch = start_epoch + local_epoch + 1
        losses = []
        # The manifest is grouped by film. Shuffle reproducibly per epoch so a
        # quick check does not always inspect the first two clips of one film,
        # while keeping runs debuggable and repeatable.
        epoch_rows = list(train_rows)
        random.Random(f"{dataset_id}:{epoch}").shuffle(epoch_rows)
        for batch_index, row in enumerate(epoch_rows[:total_batches]):
            ctx.check_stop()
            mixture, sr = _mono(row["files"]["mixture"])
            reference, ref_sr = _mono(row["files"]["aligned_original"])
            target, target_sr = _mono(row["files"]["pseudo_target"])
            if ref_sr != sr:
                reference = np.asarray(audio_io.resample(reference, ref_sr, sr)).reshape(-1)
            if target_sr != sr:
                target = np.asarray(audio_io.resample(target, target_sr, sr)).reshape(-1)
            loss, grad, _estimate = _loss_and_grad(
                mixture, reference, target, sr, edges, gains
            )
            global_step += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad * grad
            m_hat = m / (1 - beta1**global_step)
            v_hat = v / (1 - beta2**global_step)
            gains -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            gains = np.clip(gains, 0.0, 8.0)
            losses.append(loss)
            overall = (local_epoch * total_batches + batch_index + 1) / max(epochs * total_batches, 1)
            ctx.update(
                stage="Обучение калибровочной модели",
                substage=f"Проход {epoch}, пакет {batch_index + 1} из {total_batches}",
                progress=overall * 100.0,
                current_file=row["film_name"],
            )
        train_loss = float(np.mean(losses))
        validation_loss, si_snr_improvement = _evaluate(valid_rows, gains, edges, ctx)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "si_snr_improvement_db": si_snr_improvement,
            "dataset_id": dataset_id,
            "algorithm_version": FORMAT,
            "validation_independent": validation_independent,
        }
        history.append(row)
        if mode != "quick":
            row["control_example"] = _save_control_example(
                valid_rows[0],
                gains,
                edges,
                run_root / "controls" / f"epoch_{epoch:03d}",
            )
            if len(history) >= 2:
                previous = history[-2].get("validation_loss")
                row["validation_change"] = (
                    float(validation_loss - previous)
                    if previous is not None and np.isfinite(previous)
                    else None
                )
            else:
                row["validation_change"] = None
            payload = _checkpoint_payload(
                dataset_id, run_id, epoch, global_step, edges, gains, m, v, history, store.cfg
            )
            step_path = store.project_dir(project_id) / "checkpoints" / f"{run_id}_epoch{epoch}.npz"
            _atomic_npz(step_path, payload)
            _atomic_npz(store.project_dir(project_id) / "checkpoints" / "checkpoint_last.npz", payload)
            row["checkpoint"] = str(step_path.resolve())
    valid_values = [item["validation_loss"] for item in history if np.isfinite(item["validation_loss"])]
    if len(valid_values) >= 2:
        relative = (valid_values[-2] - valid_values[-1]) / max(abs(valid_values[-2]), 1e-12)
    else:
        relative = 0.0
    threshold = float(store.cfg["training"]["recommendation_min_relative_improvement"])
    recommendation = (
        "Добавьте независимую выборку проверки перед сохранением модели."
        if not validation_independent
        else (
            "Можно продолжить: validation loss улучшается."
            if relative >= threshold
            else "Остановиться и сравнить на слух: подтверждённого улучшения validation loss нет."
        )
    )
    report = {
        "schema_version": 2,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "algorithm_version": FORMAT,
        "mode": mode,
        "dry_run": mode == "quick",
        "resume_checkpoint": str(resume or ""),
        "epochs_requested": epochs,
        "history": history,
        "best_epoch": min(history, key=lambda item: item["validation_loss"] if np.isfinite(item["validation_loss"]) else float("inf")),
        "recommendation": recommendation,
        "validation_independent": validation_independent,
        "note": "Модель обучается на pseudo-target и не может превзойти качество разметки без чистого эталона.",
        "finished_at": utc_now(),
    }
    atomic_json(run_root / "history.json", report)
    project["latest_training_run"] = run_id
    store.save_project(project)
    for pair in store.list_pairs(project_id):
        if pair["id"] in manifest["pair_ids"]:
            pair["stages"]["8"] = {"status": "completed", "message": "Проверка обучения выполнена."}
            pair["stages"]["9"] = {"status": "available", "message": "Доступно сравнение с базовым алгоритмом."}
            store.save_pair(pair)
    return report


def _apply_gains(reference: np.ndarray, sample_rate: int, edges: list[float], gains: np.ndarray) -> np.ndarray:
    output = np.zeros_like(reference, dtype=np.float32)
    for channel in range(reference.shape[1]):
        components = _bands(reference[:, channel], sample_rate, edges)
        for gain, component in zip(gains, components):
            output[:, channel] += float(gain) * component
    return output


def apply_checkpoint_streaming(
    mixture_path: Path,
    aligned_path: Path,
    confidence_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    cfg: dict,
    ctx: Any,
) -> None:
    state = load_checkpoint(checkpoint_path)
    gains = np.asarray(state["model_state"]["gains"], dtype=np.float64)
    edges = [float(item) for item in state["model_state"]["band_edges_hz"]]
    info = sf.info(str(mixture_path))
    block = int(round(float(cfg["audio"]["processing_block_sec"]) * info.samplerate))
    temporary = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.partial{output_path.suffix}")
    processed = 0
    try:
        with sf.SoundFile(str(mixture_path), "r") as mix, sf.SoundFile(
            str(aligned_path), "r"
        ) as aligned, sf.SoundFile(str(confidence_path), "r") as confidence, sf.SoundFile(
            str(temporary),
            "w",
            samplerate=info.samplerate,
            channels=info.channels,
            format="FLAC",
            subtype="PCM_24",
        ) as output:
            while True:
                ctx.check_stop()
                mixture = mix.read(block, dtype="float32", always_2d=True)
                if not len(mixture):
                    break
                reference = aligned.read(len(mixture), dtype="float32", always_2d=True)
                conf = confidence.read(len(mixture), dtype="float32", always_2d=True)
                # The aligned/confidence tracks are built segment-by-segment
                # and may differ from the mixture by a few samples at the very
                # end; pad with silence/zero-confidence instead of crashing.
                if len(reference) < len(mixture):
                    reference = np.vstack(
                        [reference, np.zeros((len(mixture) - len(reference), reference.shape[1]), dtype=np.float32)]
                    )
                if len(conf) < len(mixture):
                    conf = np.vstack(
                        [conf, np.zeros((len(mixture) - len(conf), conf.shape[1]), dtype=np.float32)]
                    )
                reference = audio_io.match_channels(reference, info.channels).astype(np.float32)
                predicted = _apply_gains(reference, info.samplerate, edges, gains)
                mask = np.clip(conf[: len(mixture), :1], 0.0, 1.0)
                result = mixture - predicted[: len(mixture)] * mask
                output.write(np.clip(result, -1.0, 1.0))
                processed += len(mixture)
                ctx.update(
                    stage="Полная обработка выбранной моделью",
                    substage=f"Обработано {processed / info.samplerate:.1f} из {info.duration:.1f} с",
                    progress=min(100.0, processed / max(info.frames, 1) * 100.0),
                    processed_seconds=processed / info.samplerate,
                    total_seconds=info.duration,
                    current_file=mixture_path.name,
                )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    store: Store,
    project_id: str,
    ctx: Any,
    parameters: dict,
) -> dict:
    """Launch the isolated mastering-invariant semantic RU separator trainer."""
    project = store.load_project(project_id)
    dataset_id = str(parameters.get("dataset_id") or project.get("latest_dataset_id") or "")
    if not dataset_id:
        raise RuntimeError("Сначала подготовьте датасет парного вычитания.")
    manifest_path = (
        store.project_dir(project_id)
        / "training"
        / "datasets"
        / dataset_id
        / "manifest.json"
    )
    manifest = read_json(manifest_path)
    if manifest.get("target_kind") not in {
        "paired_speech_reference_subtraction_supervised_v2",
        "clean_translation_reference_subtraction_supervised_v3",
    }:
        raise RuntimeError(
            "Выбран старый датасет. Подготовьте новый датасет парного вычитания на четырёх фильмах."
        )
    mode = str(parameters.get("mode") or "quick")
    epochs = int(parameters.get("epochs") or 1)
    allowed = (
        [1]
        if mode == "quick"
        else list(
            store.cfg["training"][
                "short_epoch_choices" if mode == "short" else "full_epoch_choices"
            ]
        )
    )
    if mode not in {"quick", "short", "full"} or epochs not in allowed:
        raise ValueError("Недопустимый режим или количество проходов.")
    python_path = root_path(store.cfg["training"]["python"])
    runner_path = root_path(store.cfg["training"]["runner"])
    if not python_path.is_file() or not runner_path.is_file():
        raise RuntimeError("Среда PyTorch или программа обучения не найдена.")
    resume = str(parameters.get("checkpoint") or "")
    checkpoint_root = store.project_dir(project_id) / "checkpoints"
    if resume:
        resume_path = Path(resume).resolve()
        if not resume_path.is_file() or checkpoint_root.resolve() not in resume_path.parents:
            raise ValueError("Сохранение модели должно принадлежать текущему проекту.")
        if resume_path.suffix.lower() != ".pt":
            raise ValueError("Для обученной модели выберите файл .pt.")
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_semantic_" + uuid.uuid4().hex[:6]
    run_root = store.project_dir(project_id) / "training" / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    training_cfg = {
        key: store.cfg["training"][key]
        for key in (
            "n_fft",
            "hop_length",
            "base_channels",
            "learning_rate",
            "batch_size",
            "data_loader_workers",
            "quick_check_batches",
        )
    }
    command = [
        str(python_path),
        str(runner_path),
        "--manifest",
        str(manifest_path),
        "--run-dir",
        str(run_root),
        "--checkpoint-dir",
        str(checkpoint_root),
        "--mode",
        mode,
        "--epochs",
        str(epochs),
        "--config-json",
        json.dumps(training_cfg),
    ]
    if resume:
        command.extend(["--resume", resume])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    ctx.child_pid = process.pid
    tail: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            ctx.check_stop()
            text = line.strip()
            if not text:
                continue
            tail.append(text)
            tail = tail[-30:]
            try:
                event = json.loads(text)
            except ValueError:
                ctx.log(text)
                continue
            if event.get("type") == "progress":
                ctx.update(
                    stage=event.get("stage", "Обучение смысловому разделению RU/EN"),
                    substage=event.get("substage", ""),
                    progress=float(event.get("progress", 0.0)),
                    current_file=event.get("current_file", ""),
                )
            elif event.get("type") == "log":
                ctx.log(str(event.get("message") or ""))
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                "Обучение завершилось с ошибкой. Последние сообщения:\n"
                + "\n".join(tail[-12:])
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        ctx.child_pid = None
    report = read_json(run_root / "history.json")
    if not report:
        raise RuntimeError("Обучение завершилось, но история запуска не создана.")
    project = store.load_project(project_id)
    project["latest_training_run"] = run_id
    store.save_project(project)
    for pair in store.list_pairs(project_id):
        if pair["id"] in manifest.get("pair_ids", []):
            pair["stages"]["9"] = {
                "status": "completed",
                "message": "Контрольное обучение модели парного вычитания выполнено.",
            }
            store.save_pair(pair)
    return report
