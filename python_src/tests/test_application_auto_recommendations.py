from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_reference_cancel import (
    application_pipeline,
    pipeline,
    reference_compatibility,
    task_worker,
)


class _AutoStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.cfg = {
            "alignment": {
                "auto_correction_min_confidence": 0.3,
            }
        }
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "stages": {
                "2": {"status": "completed"},
                "3": {"status": "completed"},
            },
        }

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair

    def invalidate_pair_derivatives(
        self, _project_id: str, _pair_id: str, **_values
    ) -> dict:
        return self.pair


class _AutoContext:
    def __init__(self) -> None:
        self.task = {
            "operation": "application_auto",
            "project_id": "project",
            "pair_id": "pair",
            "parameters": {
                "checkpoint": "dubclean_voice.pt",
                "background_checkpoint": "dubclean_me.pt",
                "reference_alignment_policy": "auto",
                "speech_second_pass_policy": "auto",
            },
        }
        self.child_pid = None
        self.progress: list[float] = []

    def check_stop(self) -> None:
        return None

    def heartbeat(self) -> None:
        return None

    def log(self, _message: str) -> None:
        return None

    def update(self, **values) -> None:
        self.progress.append(float(values.get("progress") or 0.0))


class AutomaticRecommendationWorkflowTests(unittest.TestCase):
    def test_full_auto_uses_analysis_alignment_sync_and_mix_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _AutoStore(Path(temp_dir))
            root = store.pair_dir("project", "pair")
            extracted = root / "extracted"
            alignment = root / "alignment"
            extracted.mkdir(parents=True)
            alignment.mkdir(parents=True)
            for name in (
                "original.flac",
                "dubbed.flac",
                "original_proxy.wav",
                "dubbed_proxy.wav",
                "metadata.json",
            ):
                (extracted / name).write_bytes(b"x")
            (alignment / "alignment_map.json").write_text(
                json.dumps(
                    {
                        "manual_correction_sec": 0.0,
                        "global": {"offset_sec": 0.0, "speed_ratio": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            ctx = _AutoContext()
            prepared_parameters: dict = {}
            full_parameters: dict = {}

            def fake_prepare(_store, _project, _pair, _ctx, parameters):
                prepared_parameters.update(parameters)
                return {
                    "speech_synchronization_recommendation": {
                        "shifted_segments": 3,
                        "matched_segments": 4,
                    }
                }

            def fake_full(_store, _project, _pair, _ctx, parameters):
                full_parameters.update(parameters)
                return {"result": "ok"}

            with (
                patch.object(
                    pipeline,
                    "estimate_sparse_manual_correction",
                    return_value={
                        "correction_sec": 0.125,
                        "confidence": 0.9,
                    },
                ),
                patch.object(
                    reference_compatibility,
                    "analyze_reference_compatibility",
                    return_value={
                        "band": "MID",
                        "recommended_model": "dubclean_voice",
                        "recommended_alignment": "algorithmic",
                    },
                ),
                patch.object(
                    reference_compatibility,
                    "apply_reference_compatibility_recommendations",
                    return_value={
                        "recommended_model": "dubclean_voice",
                        "recommended_alignment": "algorithmic",
                        "recommended_speech_extraction_second_pass": True,
                    },
                ),
                patch.object(
                    application_pipeline,
                    "prepare_application",
                    side_effect=fake_prepare,
                ),
                patch.object(
                    application_pipeline,
                    "build_full_application",
                    side_effect=fake_full,
                ),
            ):
                result = task_worker.execute(store, ctx)

            self.assertEqual(result, {"result": "ok"})
            self.assertEqual(prepared_parameters["subtraction_model"], "dubclean_voice")
            self.assertEqual(prepared_parameters["reference_adapter"], "algorithmic")
            self.assertTrue(prepared_parameters["speech_extraction_second_pass"])
            self.assertTrue(full_parameters["speech_extraction_second_pass"])
            self.assertEqual(
                store.pair["automatic_assembly_settings"]["reference_alignment_policy"],
                "auto",
            )
            self.assertTrue(store.pair["speech_extraction_second_pass"])
            self.assertTrue(full_parameters["synchronize_speech"])
            self.assertTrue(full_parameters["balance_final_mix"])
            self.assertEqual(full_parameters["voice_gain_db"], 0.0)
            self.assertEqual(full_parameters["background_gain_db"], 0.0)
            self.assertEqual(full_parameters["voice_delay_sec"], 0.0)
            self.assertEqual(store.pair["alignment_manual_correction_sec"], 0.125)
            self.assertEqual(ctx.progress[-1], 100.0)


if __name__ == "__main__":
    unittest.main()
