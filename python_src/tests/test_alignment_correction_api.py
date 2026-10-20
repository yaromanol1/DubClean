from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_reference_cancel import app as app_module


class _CorrectionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cfg = {"alignment": {"auto_correction_points": 7}}
        self.invalidations = 0
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "alignment_manual_correction_sec": 0.0,
            "alignment_map": {"manual_correction_sec": 0.0, "segments": []},
            "application_preview": {"scenes": [{"id": "scene"}]},
            "application_result": {"movie": "result.mkv"},
            "stages": {
                "2": {"status": "completed"},
                "3": {"status": "completed"},
                "4": {"status": "completed"},
                "5": {"status": "completed"},
            },
        }
        alignment_dir = self.pair_dir("project", "pair") / "alignment"
        alignment_dir.mkdir(parents=True)
        (alignment_dir / "alignment_map.json").write_text(
            json.dumps(self.pair["alignment_map"]), encoding="utf-8"
        )
        extracted_dir = self.pair_dir("project", "pair") / "extracted"
        extracted_dir.mkdir(parents=True)
        (extracted_dir / "original_proxy.wav").write_bytes(b"proxy")
        (extracted_dir / "dubbed_proxy.wav").write_bytes(b"proxy")

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair

    def invalidate_pair_derivatives(self, *_args, **_kwargs) -> dict:
        self.invalidations += 1
        self.pair.pop("application_preview", None)
        self.pair.pop("application_result", None)
        return self.pair


class AlignmentCorrectionApiTests(unittest.TestCase):
    def test_saving_existing_offset_does_not_invalidate_ready_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _CorrectionStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                response = client.post(
                    "/api/projects/project/pairs/pair/alignment-correction",
                    json={"correction_sec": 0.0},
                )

            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.get_json()["changed"])
            self.assertEqual(store.invalidations, 0)
            self.assertIn("application_preview", store.pair)
            self.assertIn("application_result", store.pair)

    def test_changed_offset_still_invalidates_dependent_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _CorrectionStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                response = client.post(
                    "/api/projects/project/pairs/pair/alignment-correction",
                    json={"correction_sec": 1.25},
                )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["changed"])
            self.assertEqual(store.invalidations, 1)
            self.assertNotIn("application_preview", store.pair)
            self.assertNotIn("application_result", store.pair)

    def test_auto_correction_runs_sparse_estimator_without_persisting_draft(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _CorrectionStore(Path(temp_dir))
            estimate = {
                "correction_sec": -2.25,
                "confidence": 0.81,
                "analysed_points": 7,
                "accepted_points": 6,
                "points": [],
                "method": "sparse_multi_scene_residual_v1",
            }
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
                patch.object(app_module, "estimate_sparse_manual_correction", return_value=estimate) as estimator,
            ):
                client = app_module.app.test_client()
                response = client.post(
                    "/api/projects/project/pairs/pair/alignment-auto-correction"
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["correction_sec"], -2.25)
            self.assertEqual(store.invalidations, 0)
            self.assertEqual(store.pair["alignment_manual_correction_sec"], 0.0)
            estimator.assert_called_once()


if __name__ == "__main__":
    unittest.main()
