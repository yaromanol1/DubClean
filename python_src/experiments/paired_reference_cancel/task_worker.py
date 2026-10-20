#!/usr/bin/env python
"""Background operation worker. One process owns exactly one persisted task."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.storage import (  # noqa: E402
    Store,
    atomic_json,
    read_json,
    utc_now,
)
from experiments.paired_reference_cancel.automatic_assembly import (  # noqa: E402
    resolve_policies,
)


class TaskStopped(RuntimeError):
    pass


class TaskContext:
    def __init__(self, store: Store, project_id: str, task_id: str):
        self.store = store
        self.project_id = project_id
        self.task_id = task_id
        self.task = store.load_task(project_id, task_id)
        self.stop_file = Path(self.task["stop_file"])
        self.log_file = Path(self.task["log_file"])
        self.child_pid: int | None = None
        self._last_write = 0.0
        self._write_lock = threading.Lock()

    def check_stop(self) -> None:
        if self.stop_file.exists():
            raise TaskStopped("Остановлено пользователем.")

    def log(self, message: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H:%M:%S")
        with self.log_file.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"[{stamp}] {message.rstrip()}\n")
        with self._write_lock:
            self.task["heartbeat_at"] = utc_now()
            self.store.save_task(self.task)

    def update(self, force: bool = False, **values: Any) -> None:
        with self._write_lock:
            now = time.monotonic()
            previous_stage = str(self.task.get("stage") or "")
            next_stage = str(values.get("stage") or previous_stage)
            stage_changed = "stage" in values and next_stage != previous_stage
            if "progress" in values:
                try:
                    raw_progress = max(
                        0.0,
                        min(100.0, float(values.get("progress", 0.0) or 0.0)),
                    )
                    if "local_progress" not in values:
                        # Overall progress and operation progress are different
                        # values. A newly announced operation starts locally at
                        # zero; subsequent updates within that same operation
                        # may use its 0..100 progress directly.
                        values["local_progress"] = (
                            0.0 if stage_changed else raw_progress
                        )
                    values["progress"] = max(
                        float(self.task.get("progress", 0.0) or 0.0),
                        raw_progress,
                    )
                except (TypeError, ValueError):
                    pass
            if "local_progress" in values:
                try:
                    values["local_progress"] = max(
                        0.0,
                        min(100.0, float(values.get("local_progress", 0.0) or 0.0)),
                    )
                except (TypeError, ValueError):
                    values.pop("local_progress", None)
            elif stage_changed:
                values["local_progress"] = 0.0
            self.task.update(values)
            self.task["heartbeat_at"] = utc_now()
            if force or now - self._last_write >= 0.2 or float(values.get("progress", -1)) >= 100:
                self.store.save_task(self.task)
                self._last_write = now

    def heartbeat(self) -> None:
        """Persist liveness while a single CPU-heavy segment is processing."""
        with self._write_lock:
            self.task["heartbeat_at"] = utc_now()
            self.store.save_task(self.task)


class ProgressPhaseContext:
    """Map local 0..100 progress into an explicit overall task interval."""

    def __init__(self, parent: TaskContext, base: float, span: float, label: str):
        self.parent = parent
        self.base = float(base)
        self.span = float(span)
        self.label = label
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
        return self.parent.heartbeat()

    def log(self, message):
        return self.parent.log(message)

    def update(self, force: bool = False, **values):
        has_overall_progress = "progress" in values
        local = max(
            0.0,
            min(
                100.0,
                float(
                    values.get(
                        "local_progress",
                        values.get(
                            "progress",
                            self.parent.task.get("local_progress", 0.0),
                        ),
                    )
                    or 0.0
                ),
            ),
        )
        if has_overall_progress:
            mapped_source = max(
                0.0,
                min(100.0, float(values.get("progress", 0.0) or 0.0)),
            )
            mapped = self.base + mapped_source / 100.0 * self.span
            self._high_water = max(self._high_water, mapped)
        stage = str(values.get("stage") or "").strip()
        values["stage"] = f"{self.label}: {stage}" if stage else self.label
        values["local_progress"] = local
        values["progress"] = self._high_water
        return self.parent.update(force=force, **values)


def _mark_pair_failed(store: Store, task: dict, message: str) -> None:
    pair_id = task.get("pair_id")
    if not pair_id:
        return
    try:
        pair = store.load_pair(task["project_id"], pair_id)
    except (FileNotFoundError, RuntimeError):
        return
    pair.setdefault("errors", []).append(message)
    operation_stage = {
        "extract": 2,
        "align": 3,
        "speech_test": 5,
        "speech_extract": 5,
        "components_test": 6,
        "components_build": 6,
        "preview": 7,
        "finalize": 10,
        "application_prepare": 5,
        "application_full": 10,
        "application_auto": 10,
    }.get(task["operation"])
    if task["operation"] == "application_tracks":
        # This combined operation can fail either while extracting or while
        # aligning.  Attribute the error to the phase that was actually active
        # so the UI never claims that extraction succeeded when it did not.
        operation_stage = (
            3
            if pair.get("stages", {}).get("2", {}).get("status") == "completed"
            else 2
        )
    if operation_stage:
        pair["stages"][str(operation_stage)] = {"status": "error", "message": message}
    store.save_pair(pair)


def execute(store: Store, ctx: TaskContext) -> Any:
    task = ctx.task
    operation = task["operation"]
    project_id = task["project_id"]
    pair_id = task.get("pair_id")
    parameters = task.get("parameters") or {}
    if operation == "application_auto":
        from experiments.paired_reference_cancel import application_pipeline, pipeline
        from experiments.paired_reference_cancel.reference_compatibility import (
            analyze_reference_compatibility,
            apply_reference_compatibility_recommendations,
        )

        pair = store.load_pair(project_id, pair_id)
        root = store.pair_dir(project_id, pair_id)
        extraction_ready = (
            pair.get("stages", {}).get("2", {}).get("status") == "completed"
            and (root / "extracted" / "original.flac").is_file()
            and (root / "extracted" / "dubbed.flac").is_file()
            and (root / "extracted" / "original_proxy.wav").is_file()
            and (root / "extracted" / "dubbed_proxy.wav").is_file()
            and (root / "extracted" / "metadata.json").is_file()
        )
        if not extraction_ready:
            pipeline.extract_pair(
                store,
                project_id,
                pair_id,
                ProgressPhaseContext(ctx, 0.0, 14.0, "Автосборка · извлечение"),
            )
        pair = store.load_pair(project_id, pair_id)
        alignment_ready = (
            pair.get("stages", {}).get("3", {}).get("status") == "completed"
            and (root / "alignment" / "alignment_map.json").is_file()
        )
        if not alignment_ready:
            pipeline.align_pair(
                store,
                project_id,
                pair_id,
                ProgressPhaseContext(ctx, 14.0, 6.0, "Автосборка · сопоставление"),
            )

        # The automatic path uses the same sparse, multi-scene correction as
        # the step-2 "Авто" action. Different file lengths are intentionally
        # ignored: the decision comes from matching scenes across the film.
        pair = store.load_pair(project_id, pair_id)
        alignment_path = root / "alignment" / "alignment_map.json"
        alignment_map = read_json(alignment_path, {}) or {}
        try:
            estimate = pipeline.estimate_sparse_manual_correction(
                root / "extracted" / "original_proxy.wav",
                root / "extracted" / "dubbed_proxy.wav",
                alignment_map,
                dict(store.cfg.get("alignment") or {}),
            )
            minimum_confidence = float(
                (store.cfg.get("alignment") or {}).get(
                    "auto_correction_min_confidence", 0.3
                )
            )
            correction = float(estimate.get("correction_sec") or 0.0)
            if float(estimate.get("confidence") or 0.0) >= minimum_confidence:
                previous = float(alignment_map.get("manual_correction_sec") or 0.0)
                if abs(previous - correction) >= 0.0005:
                    pair = store.invalidate_pair_derivatives(
                        project_id,
                        pair_id,
                        from_stage=4,
                        reason="automatic_alignment_correction",
                    )
                alignment_map["manual_correction_sec"] = correction
                atomic_json(alignment_path, alignment_map)
                pair["alignment_map"] = alignment_map
                pair["alignment_manual_correction_sec"] = correction
                pair["alignment_auto_correction"] = estimate
                store.save_pair(pair)
            else:
                ctx.log(
                    "Автоматический остаточный сдвиг не применён: "
                    "недостаточная уверенность."
                )
        except RuntimeError as error:
            ctx.log(f"Автоматический остаточный сдвиг не применён: {error}")
        ctx.update(
            force=True,
            stage="Автосборка · сопоставление",
            substage="Сдвиг дорожек проверен",
            progress=24.0,
        )

        analysis = analyze_reference_compatibility(
            store,
            project_id,
            pair_id,
            ProgressPhaseContext(ctx, 24.0, 10.0, "Автосборка · анализ"),
        )
        applied_recommendations = apply_reference_compatibility_recommendations(
            store, project_id, pair_id, analysis
        )
        automatic_settings = resolve_policies(parameters, applied_recommendations)
        parameters["reference_alignment_policy"] = automatic_settings[
            "reference_alignment_policy"
        ]
        parameters["speech_second_pass_policy"] = automatic_settings[
            "speech_second_pass_policy"
        ]
        parameters["speech_extraction_second_pass"] = bool(
            automatic_settings["speech_extraction_second_pass"]
        )
        parameters["subtraction_model"] = "dubclean_voice"
        parameters["reference_adapter"] = automatic_settings["reference_adapter"]
        pair = store.load_pair(project_id, pair_id)
        pair["speech_extraction_second_pass"] = bool(
            automatic_settings["speech_extraction_second_pass"]
        )
        pair["automatic_assembly_settings"] = automatic_settings
        store.save_pair(pair)

        preview = application_pipeline.prepare_application(
            store,
            project_id,
            pair_id,
            ProgressPhaseContext(ctx, 34.0, 21.0, "Автосборка · подготовка"),
            parameters,
        )
        # Settings the operator chose explicitly win over the analysis.  Only
        # the two policy settings have an "Автоматически" option, and those are
        # already resolved above; the rest are concrete answers, so overwriting
        # them here made the one-click film ignore its own defaults screen.
        synchronization = (
            preview.get("speech_synchronization_recommendation") or {}
        )
        parameters.setdefault(
            "synchronize_speech",
            bool(
                int(synchronization.get("shifted_segments") or 0) > 0
                and int(synchronization.get("matched_segments") or 0) > 0
            ),
        )
        parameters.setdefault("balance_final_mix", True)
        parameters.setdefault("voice_gain_db", 0.0)
        parameters.setdefault("background_gain_db", 0.0)
        parameters.setdefault("voice_delay_sec", 0.0)
        # A global delay and per-phrase moves cannot both apply; the full-film
        # builder enforces this too, but the automatic path must not send a
        # contradictory pair in the first place.
        if parameters.get("synchronize_speech"):
            parameters["voice_delay_sec"] = 0.0
        result = application_pipeline.build_full_application(
            store,
            project_id,
            pair_id,
            ProgressPhaseContext(ctx, 55.0, 45.0, "Автосборка · фильм"),
            parameters,
        )
        ctx.update(
            force=True,
            stage="Полный фильм готов",
            substage="Автоматический конвейер завершён",
            progress=100.0,
            local_progress=100.0,
        )
        return result
    if operation == "application_tracks":
        ctx.update(
            force=True,
            stage="Подготовка дорожек",
            substage="Проверка извлечения и сопоставления",
            progress=0.5,
            current_file="",
        )
        from experiments.paired_reference_cancel import pipeline

        pair = store.load_pair(project_id, pair_id)
        root = store.pair_dir(project_id, pair_id)
        extraction_ready = (
            pair.get("stages", {}).get("2", {}).get("status") == "completed"
            and (root / "extracted" / "original.flac").is_file()
            and (root / "extracted" / "dubbed.flac").is_file()
            and (root / "extracted" / "original_proxy.wav").is_file()
            and (root / "extracted" / "dubbed_proxy.wav").is_file()
            and (root / "extracted" / "metadata.json").is_file()
        )
        if not extraction_ready:
            pipeline.extract_pair(
                store,
                project_id,
                pair_id,
                ProgressPhaseContext(ctx, 0.0, 82.0, "Подготовка дорожек · извлечение"),
            )
        else:
            ctx.update(
                force=True,
                stage="Подготовка дорожек",
                substage="Извлечённые дорожки готовы",
                progress=82.0,
                local_progress=100.0,
            )
        pair = store.load_pair(project_id, pair_id)
        alignment_ready = (
            pair.get("stages", {}).get("3", {}).get("status") == "completed"
            and (root / "alignment" / "alignment_map.json").is_file()
        )
        if not alignment_ready:
            pipeline.align_pair(
                store,
                project_id,
                pair_id,
                ProgressPhaseContext(ctx, 82.0, 18.0, "Подготовка дорожек · сопоставление"),
            )
        ctx.update(
            force=True,
            stage="Подготовка дорожек",
            substage="Дорожки извлечены и сопоставлены",
            progress=100.0,
            local_progress=100.0,
        )
        return {"tracks_ready": True}
    if operation == "application_reference_analysis":
        ctx.update(
            force=True,
            stage="Анализ соответствия",
            substage="Загрузка извлечённых дорожек",
            progress=0.5,
            current_file="",
        )
        from experiments.paired_reference_cancel.reference_compatibility import (
            analyze_reference_compatibility,
        )

        return analyze_reference_compatibility(store, project_id, pair_id, ctx)
    if operation in {"application_prepare", "application_full"}:
        ctx.update(
            force=True,
            stage=(
                "Подготовка полного фильма"
                if operation == "application_full"
                else "Подготовка нейрообработки"
            ),
            substage="Загрузка модулей обработки",
            progress=0.5,
            current_file="",
        )
        from experiments.paired_reference_cancel import application_pipeline

        if operation == "application_prepare":
            return application_pipeline.prepare_application(
                store, project_id, pair_id, ctx, parameters
            )
        return application_pipeline.build_full_application(
            store, project_id, pair_id, ctx, parameters
        )
    if operation == "train":
        from experiments.paired_reference_cancel.training_backend import train

        return train(store, project_id, ctx, parameters)

    from experiments.paired_reference_cancel import pipeline

    if operation == "extract":
        return pipeline.extract_pair(store, project_id, pair_id, ctx)
    if operation == "align":
        return pipeline.align_pair(store, project_id, pair_id, ctx)
    if operation == "speech_test":
        return pipeline.extract_original_speech_test(store, project_id, pair_id, ctx, parameters)
    if operation == "speech_extract":
        return pipeline.extract_original_speech_pair(store, project_id, pair_id, ctx)
    if operation == "target_speech_extract":
        return pipeline.extract_clean_translation_speech_pair(
            store, project_id, pair_id, ctx
        )
    if operation == "components_test":
        return pipeline.build_components_test(store, project_id, pair_id, ctx)
    if operation == "components_build":
        return pipeline.build_components_pair(store, project_id, pair_id, ctx)
    if operation == "preview":
        return pipeline.make_previews(store, project_id, pair_id, ctx, parameters)
    if operation == "dataset":
        return pipeline.prepare_reference_subtraction_dataset(
            store,
            project_id,
            list(parameters.get("pair_ids") or []),
            ctx,
            parameters,
        )
    if operation == "finalize":
        return pipeline.finalize_pair(store, project_id, pair_id, ctx, parameters)
    raise ValueError(f"Неизвестная операция: {operation}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    store = Store()
    ctx = TaskContext(store, args.project_id, args.task_id)
    # A task id is never reused.  Therefore an existing stop marker is a real
    # cancellation request, not stale state to erase.  Clearing it here let a
    # just-stopped "starting" worker resurrect itself and run to completion.
    if ctx.task.get("state") not in {"queued", "starting"}:
        return 3
    if ctx.stop_file.exists():
        ctx.task.update(
            {
                "state": "stopped",
                "stage": "Остановлено",
                "substage": "Остановлено до запуска фонового процесса.",
                "pid": None,
                "error": None,
                "finished_at": utc_now(),
                "heartbeat_at": utc_now(),
            }
        )
        ctx.update(force=True)
        return 2
    ctx.task.update(
        {
            "state": "running",
            "pid": os.getpid(),
            "started_at": ctx.task.get("started_at") or utc_now(),
            "heartbeat_at": utc_now(),
            "error": None,
        }
    )
    ctx.update(force=True)
    ctx.check_stop()
    ctx.log(f"Запущена операция «{ctx.task['operation']}».")
    heartbeat_stop = threading.Event()

    def keep_heartbeat() -> None:
        while not heartbeat_stop.wait(10.0):
            ctx.heartbeat()

    heartbeat_thread = threading.Thread(target=keep_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = execute(store, ctx)
        ctx.task.update(
            {
                "state": "completed",
                "progress": 100.0,
                "local_progress": 100.0,
                "stage": "Завершено",
                "substage": "",
                "eta_seconds": 0.0,
                "result": result,
                "finished_at": utc_now(),
                "heartbeat_at": utc_now(),
                "pid": None,
            }
        )
        ctx.log("Операция успешно завершена.")
        ctx.update(force=True)
        return 0
    except TaskStopped as exc:
        ctx.task.update(
            {
                "state": "stopped",
                "stage": "Остановлено",
                "substage": str(exc),
                "error": None,
                "finished_at": utc_now(),
                "heartbeat_at": utc_now(),
                "pid": None,
            }
        )
        ctx.log(str(exc))
        ctx.update(force=True)
        return 2
    except BaseException as exc:  # noqa: BLE001 - must persist worker failures
        message = f"{type(exc).__name__}: {exc}"
        ctx.task.update(
            {
                "state": "failed",
                "stage": "Ошибка",
                "substage": "",
                "error": message,
                "finished_at": utc_now(),
                "heartbeat_at": utc_now(),
                "pid": None,
            }
        )
        ctx.log(message)
        ctx.log(traceback.format_exc())
        _mark_pair_failed(store, ctx.task, message)
        ctx.update(force=True)
        return 1
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
