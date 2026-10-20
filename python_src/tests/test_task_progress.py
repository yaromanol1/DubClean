from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.paired_reference_cancel.task_worker import (
    ProgressPhaseContext,
    TaskContext,
    _mark_pair_failed,
)


class _TaskStore:
    def __init__(self, root: Path) -> None:
        self.task = {
            "id": "task",
            "project_id": "project",
            "state": "running",
            "stage": "Начало",
            "progress": 0.0,
            "local_progress": 0.0,
            "stop_file": str(root / "task.stop"),
            "log_file": str(root / "task.log"),
        }

    def load_task(self, _project_id: str, _task_id: str) -> dict:
        return self.task

    def save_task(self, task: dict) -> None:
        self.task = task


class _PairStore:
    def __init__(self, extraction_completed: bool) -> None:
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "errors": [],
            "stages": {
                "2": {
                    "status": "completed" if extraction_completed else "running",
                    "message": "",
                },
                "3": {"status": "blocked", "message": ""},
            },
        }

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair


class TaskProgressTests(unittest.TestCase):
    def test_local_progress_can_restart_while_overall_progress_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _TaskStore(Path(temp_dir))
            ctx = TaskContext(store, "project", "task")
            ctx.update(
                force=True,
                stage="Первая операция",
                progress=65.0,
                local_progress=65.0,
            )
            ctx.update(
                force=True,
                stage="Вторая операция",
                progress=8.0,
                local_progress=8.0,
            )

        self.assertEqual(store.task["progress"], 65.0)
        self.assertEqual(store.task["local_progress"], 8.0)

    def test_phase_context_maps_overall_and_keeps_local_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _TaskStore(Path(temp_dir))
            parent = TaskContext(store, "project", "task")
            phase = ProgressPhaseContext(parent, 40.0, 30.0, "Этап")
            phase.update(force=True, stage="Операция", progress=20.0)

        self.assertEqual(store.task["progress"], 46.0)
        self.assertEqual(store.task["local_progress"], 20.0)
        self.assertEqual(store.task["stage"], "Этап: Операция")

    def test_new_operation_without_local_value_starts_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _TaskStore(Path(temp_dir))
            ctx = TaskContext(store, "project", "task")
            ctx.update(
                force=True,
                stage="Первая операция",
                progress=40.0,
                local_progress=90.0,
            )
            ctx.update(force=True, stage="Вторая операция", progress=55.0)

        self.assertEqual(store.task["progress"], 55.0)
        self.assertEqual(store.task["local_progress"], 0.0)

    def test_phase_local_only_update_does_not_advance_overall_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _TaskStore(Path(temp_dir))
            parent = TaskContext(store, "project", "task")
            phase = ProgressPhaseContext(parent, 40.0, 30.0, "Этап")
            phase.update(
                force=True,
                stage="Операция",
                progress=20.0,
                local_progress=20.0,
            )
            phase.update(
                force=True,
                stage="Операция",
                local_progress=100.0,
            )

        self.assertEqual(store.task["progress"], 46.0)
        self.assertEqual(store.task["local_progress"], 100.0)

    def test_track_failure_is_attributed_to_extraction_when_not_completed(self) -> None:
        store = _PairStore(extraction_completed=False)

        _mark_pair_failed(
            store,
            {
                "project_id": "project",
                "pair_id": "pair",
                "operation": "application_tracks",
            },
            "Ошибка извлечения",
        )

        self.assertEqual(store.pair["stages"]["2"]["status"], "error")
        self.assertEqual(store.pair["stages"]["3"]["status"], "blocked")

    def test_track_failure_is_attributed_to_alignment_after_extraction(self) -> None:
        store = _PairStore(extraction_completed=True)

        _mark_pair_failed(
            store,
            {
                "project_id": "project",
                "pair_id": "pair",
                "operation": "application_tracks",
            },
            "Ошибка сопоставления",
        )

        self.assertEqual(store.pair["stages"]["2"]["status"], "completed")
        self.assertEqual(store.pair["stages"]["3"]["status"], "error")


if __name__ == "__main__":
    unittest.main()
