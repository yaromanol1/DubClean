from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_reference_cancel import app as app_module


class _DecisionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = {"id": "project", "application_mode": True}
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "reference_compatibility_analysis": {
                "created_at": "2026-07-20T00:00:00Z",
                "band": "MID",
                "confidence": 0.75,
                "recommended_model": "dubclean_voice",
                "recommended_alignment": "algorithmic",
                "summary": "Есть различия мастеринга.",
                "features": {},
                "warnings": [],
                "routing_map": str(root / "pair" / "analysis" / "routing_map.json"),
            },
        }
        routing_path = Path(self.pair["reference_compatibility_analysis"]["routing_map"])
        routing_path.parent.mkdir(parents=True)
        routing_path.write_text(json.dumps({"windows": []}), encoding="utf-8")

    def load_project(self, _project_id: str) -> dict:
        return self.project

    def list_pairs(self, _project_id: str) -> list[dict]:
        return [self.pair]

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair

    def save_project(self, project: dict) -> None:
        self.project = project


class ReferenceAnalysisDecisionApiTests(unittest.TestCase):
    def test_second_speech_extraction_pass_is_explicitly_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            with patch.object(app_module, "cfg_store", store):
                client = app_module.app.test_client()
                enabled = client.patch(
                    "/api/applications/project",
                    json={"speech_extraction_second_pass": True},
                )
                disabled = client.patch(
                    "/api/applications/project",
                    json={"speech_extraction_second_pass": False},
                )

            self.assertEqual(enabled.status_code, 200)
            self.assertTrue(enabled.get_json()["pair"]["speech_extraction_second_pass"])
            self.assertEqual(disabled.status_code, 200)
            self.assertFalse(store.pair["speech_extraction_second_pass"])

    def test_product_exposes_only_dubclean_voice(self) -> None:
        options = app_module._subtraction_model_options()

        self.assertEqual([item["id"] for item in options], ["dubclean_voice"])
        self.assertEqual(
            options[0]["available"],
            Path(options[0]["path"]).is_file(),
        )
        self.assertEqual(options[0]["label"], "DubClean Voice")
        if os.environ.get("DUBCLEAN_SOURCE_ONLY") != "1":
            self.assertTrue(options[0]["path"].endswith("dubclean_voice.pt"))

    def test_manual_model_is_persisted_and_removed_model_cannot_be_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
                patch.object(
                    app_module,
                    "_subtraction_checkpoint",
                    side_effect=lambda choice: (
                        ("dubclean_voice", Path("dubclean_voice.pt"))
                        if choice == "dubclean_voice"
                        else (_ for _ in ()).throw(
                            ValueError("Неизвестная модель очистки перевода.")
                        )
                    ),
                ),
            ):
                client = app_module.app.test_client()
                manual = client.post(
                    "/api/applications/project/reference-analysis/manual",
                    json={
                        "manual_alignment": "none",
                        "manual_model": "dubclean_voice",
                    },
                )
                unavailable = client.post(
                    "/api/applications/project/reference-analysis/manual",
                    json={"manual_model": "removed_model"},
                )

            self.assertEqual(manual.status_code, 200)
            self.assertEqual(manual.get_json()["manual_model"], "dubclean_voice")
            self.assertEqual(
                store.pair["reference_compatibility_effective_model"],
                "dubclean_voice",
            )
            self.assertEqual(unavailable.status_code, 400)

    def test_apply_and_return_to_manual_are_explicit_persisted_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                applied = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={"manual_alignment": "none"},
                )
                manual = client.post(
                    "/api/applications/project/reference-analysis/manual"
                )

            self.assertEqual(applied.status_code, 200)
            self.assertEqual(
                applied.get_json()["applied"]["recommended_alignment"],
                "algorithmic",
            )
            self.assertEqual(manual.status_code, 200)
            self.assertEqual(manual.get_json()["decision"]["choice"], "manual")
            self.assertEqual(manual.get_json()["manual_alignment"], "none")
            self.assertNotIn("reference_compatibility_applied", store.pair)
            self.assertEqual(
                store.pair["reference_compatibility_decision"]["analysis_created_at"],
                store.pair["reference_compatibility_analysis"]["created_at"],
            )

    def test_return_to_manual_restores_second_pass_value_before_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            store.pair["reference_compatibility_analysis"][
                "recommended_speech_extraction_second_pass"
            ] = True
            store.pair["speech_extraction_second_pass"] = False
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                applied = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={"manual_speech_extraction_second_pass": False},
                )
                restored = client.post(
                    "/api/applications/project/reference-analysis/manual",
                    json={},
                )

            self.assertEqual(applied.status_code, 200)
            self.assertTrue(store.pair["reference_compatibility_analysis"]["recommended_speech_extraction_second_pass"])
            self.assertEqual(restored.status_code, 200)
            self.assertFalse(restored.get_json()["speech_extraction_second_pass"])
            self.assertFalse(store.pair["speech_extraction_second_pass"])

    def test_reanalysis_and_reapply_do_not_overwrite_saved_manual_second_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            store.pair["reference_compatibility_analysis"][
                "recommended_speech_extraction_second_pass"
            ] = True
            store.pair["speech_extraction_second_pass"] = False
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                first = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={
                        "manual_alignment": "none",
                        "manual_speech_extraction_second_pass": False,
                    },
                )
                self.assertEqual(first.status_code, 200)
                self.assertTrue(store.pair["speech_extraction_second_pass"])

                # A fresh analysis invalidates the applied recommendation but
                # must retain the choice that existed before recommendations.
                store.pair.pop("reference_compatibility_applied", None)
                store.pair.pop("reference_compatibility_decision", None)
                store.pair["reference_compatibility_analysis"][
                    "created_at"
                ] = "2026-07-21T00:00:00Z"
                reapplied = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={
                        # The old UI sent the currently effective second-pass
                        # value back on the second Apply.
                        "manual_alignment": "none",
                        "manual_speech_extraction_second_pass": True,
                    },
                )
                restored = client.post(
                    "/api/applications/project/reference-analysis/manual",
                    json={},
                )

            self.assertEqual(reapplied.status_code, 200)
            self.assertEqual(restored.status_code, 200)
            self.assertEqual(restored.get_json()["manual_alignment"], "none")
            self.assertFalse(restored.get_json()["speech_extraction_second_pass"])
            self.assertFalse(
                store.pair[
                    "reference_compatibility_manual_speech_extraction_second_pass"
                ]
            )

    def test_repeated_apply_manual_apply_keeps_both_choices_reversible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _DecisionStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_active_tasks_for_pair", return_value=[]),
            ):
                client = app_module.app.test_client()
                first = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={"manual_alignment": "none"},
                )
                manual = client.post(
                    "/api/applications/project/reference-analysis/manual"
                )
                second = client.post(
                    "/api/applications/project/reference-analysis/apply",
                    json={"manual_alignment": "algorithmic"},
                )

            self.assertEqual(first.status_code, 200)
            self.assertEqual(manual.get_json()["manual_alignment"], "none")
            self.assertEqual(second.status_code, 200)
            self.assertEqual(
                store.pair["reference_compatibility_manual_alignment"],
                "algorithmic",
            )
            self.assertEqual(
                store.pair["reference_compatibility_decision"]["choice"],
                "recommendations",
            )

    def test_partial_phase_is_not_reported_when_recorded_file_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            present = root / "present.flac"
            present.write_bytes(b"audio")
            pair = {
                "application_preparation": {
                    "schema_version": 1,
                    "phases": {
                        "speech": {
                            "status": "completed",
                            "artifacts": {
                                "present": str(present),
                                "missing": str(root / "missing.flac"),
                            },
                        }
                    },
                }
            }

            value, missing = app_module._sanitize_application_preparation(pair, root)

            self.assertEqual(value["phases"], {})
            self.assertEqual(missing, [str(root / "missing.flac")])


if __name__ == "__main__":
    unittest.main()
