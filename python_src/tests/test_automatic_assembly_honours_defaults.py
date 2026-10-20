"""Одна кнопка «Собрать фильм полностью» обязана слушать те же настройки.

Из анализа берутся только те два параметра, у которых в настройках есть
вариант «Автоматически». Всё остальное — прямые ответы человека, и раньше
автосборка их молча переписывала: громкости обнулялись, синхронизация решалась
по рекомендации, а выбор видео и состава дорожек вообще не доезжал.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from experiments.paired_reference_cancel import app as app_module
from experiments.paired_reference_cancel.automatic_assembly import resolve_policies


ROOT = Path(__file__).parents[2]
WORKER_SOURCE = (
    ROOT
    / "python_src"
    / "experiments"
    / "paired_reference_cancel"
    / "task_worker.py"
).read_text(encoding="utf-8")


class AutoEndpointForwardsChoicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def _start(self, body: dict) -> dict:
        # The bundled background model is what the panel always sends; an empty
        # value is rejected on purpose, so it cannot stand in for "не указано".
        payload = {
            "background_checkpoint": str(
                ROOT / "models" / "background_restorer" / "dubclean_me.pt"
            ),
            **body,
        }
        with mock.patch.object(
            app_module, "_start_application_task", return_value=("ok", 202)
        ) as start:
            response = self.client.post(
                "/api/applications/p1/auto", json=payload
            )
        if not start.call_args:
            self.fail(
                f"задача не создана: {response.status_code} "
                f"{(response.get_json() or {}).get('error', '')}"
            )
        return start.call_args.args[2]

    def test_packaging_choices_reach_the_worker(self) -> None:
        parameters = self._start(
            {"video_source": "original", "include_all_tracks": False}
        )

        self.assertEqual(parameters["video_source"], "original")
        self.assertFalse(parameters["include_all_tracks"])

    def test_levels_reach_the_worker_instead_of_being_zeroed(self) -> None:
        parameters = self._start(
            {"voice_gain_db": 3.0, "background_gain_db": -3.0}
        )

        self.assertEqual(parameters["voice_gain_db"], 3.0)
        self.assertEqual(parameters["background_gain_db"], -3.0)

    def test_levels_are_clamped_like_the_step_by_step_route(self) -> None:
        parameters = self._start(
            {"voice_gain_db": 99.0, "background_gain_db": -99.0}
        )

        self.assertEqual(parameters["voice_gain_db"], 12.0)
        self.assertEqual(parameters["background_gain_db"], -12.0)

    def test_synchronization_is_only_sent_when_answered(self) -> None:
        # Absent means "no answer": the worker then falls back to the preview
        # recommendation instead of inventing an off switch.
        self.assertNotIn("synchronize_speech", self._start({}))
        self.assertTrue(self._start({"synchronize_speech": True})["synchronize_speech"])
        self.assertFalse(
            self._start({"synchronize_speech": False})["synchronize_speech"]
        )

    def test_unknown_video_source_is_rejected(self) -> None:
        response = self.client.post(
            "/api/applications/p1/auto", json={"video_source": "auto"}
        )

        self.assertEqual(response.status_code, 400)

    def test_defaults_stay_safe_when_nothing_is_sent(self) -> None:
        parameters = self._start({})

        self.assertEqual(parameters["video_source"], "dubbed")
        self.assertTrue(parameters["include_all_tracks"])
        self.assertEqual(parameters["method"], "speech_rebuild")
        self.assertTrue(parameters["remux"])


class WorkerKeepsAnsweredSettingsTests(unittest.TestCase):
    def test_answered_settings_are_no_longer_overwritten(self) -> None:
        auto_block = WORKER_SOURCE.split('if operation == "application_auto":', 1)[
            1
        ].split('if operation == "application_tracks":', 1)[0]

        # Whitespace-insensitive: the calls are formatted both on one line and
        # across several, and the test is about which call is used, not layout.
        compact = re.sub(r"\s+", "", auto_block)
        for key in (
            "synchronize_speech",
            "balance_final_mix",
            "voice_gain_db",
            "background_gain_db",
            "voice_delay_sec",
        ):
            with self.subTest(key=key):
                self.assertIn(f'parameters.setdefault("{key}"', compact)

        # Unconditional assignment is what silently discarded the answers.
        for key in ("synchronize_speech", "voice_gain_db", "background_gain_db"):
            with self.subTest(key=key):
                self.assertNotIn(f'parameters["{key}"] = ', auto_block)

    def test_recommendation_still_fills_an_unanswered_synchronization(self) -> None:
        auto_block = WORKER_SOURCE.split('if operation == "application_auto":', 1)[1]

        self.assertIn('synchronization.get("shifted_segments")', auto_block)
        self.assertIn('synchronization.get("matched_segments")', auto_block)

    def test_a_global_delay_never_rides_on_top_of_phrase_moves(self) -> None:
        auto_block = WORKER_SOURCE.split('if operation == "application_auto":', 1)[1]

        self.assertIn(
            'if parameters.get("synchronize_speech"):\n'
            '            parameters["voice_delay_sec"] = 0.0',
            auto_block,
        )

    def test_only_the_two_policy_settings_follow_the_analysis(self) -> None:
        # These are the ones whose settings screen offers «Автоматически»;
        # inheriting a recommendation anywhere else contradicts a direct answer.
        recommendations = {
            "recommended_alignment": "algorithmic",
            "recommended_speech_extraction_second_pass": True,
        }

        forced_off = resolve_policies(
            {
                "reference_alignment_policy": "off",
                "speech_second_pass_policy": "off",
            },
            recommendations,
        )
        self.assertEqual(forced_off["reference_adapter"], "raw")
        self.assertFalse(forced_off["speech_extraction_second_pass"])

        followed = resolve_policies(
            {
                "reference_alignment_policy": "auto",
                "speech_second_pass_policy": "auto",
            },
            recommendations,
        )
        self.assertEqual(followed["reference_adapter"], "algorithmic")
        self.assertTrue(followed["speech_extraction_second_pass"])


class AutoRequestFromTheUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html_source = (
            ROOT / "docs" / "концепт интерфейса" / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")

    def test_ui_sends_every_answered_setting_on_the_automatic_route(self) -> None:
        call = self.html_source.split("/auto` : `/api/applications/${id}/prepare`", 1)[
            1
        ].split("});", 1)[0]

        self.assertIn("video_source: autoPackaging.videoSource,", call)
        self.assertIn("include_all_tracks: autoPackaging.includeAllTracks,", call)
        self.assertIn("synchronize_speech: this.state.syncMode === 2,", call)
        self.assertIn("voice_gain_db: Number(this.state.speech || 0),", call)
        self.assertIn("background_gain_db: Number(this.state.background || 0),", call)
        self.assertIn("balance_final_mix: this.state.balanceFinalMix !== false,", call)

    def test_both_routes_read_the_same_balance_setting(self) -> None:
        # The step-by-step route used to hard-code it off while the one-click
        # route hard-coded it on, so the same film sounded different.
        self.assertEqual(
            self.html_source.count("this.state.balanceFinalMix !== false"), 2
        )
        self.assertNotIn("balance_final_mix:false,", self.html_source)

    def test_balance_is_offered_as_a_saved_setting(self) -> None:
        self.assertIn(
            "label:'Подгонять баланс под исходный дубляж',", self.html_source
        )
        self.assertIn(
            "(i)=>this.setPreference({ balanceFinalMix: i === 0 })",
            self.html_source,
        )
        self.assertIn(
            "if (typeof saved.balanceFinalMix === 'boolean')", self.html_source
        )
        self.assertIn("balanceFinalMix: state.balanceFinalMix,", self.html_source)

    def test_packaging_uses_the_same_resolution_as_every_other_screen(self) -> None:
        # A third copy of the "best quality" rule could disagree with the one
        # the preview screen shows.
        self.assertEqual(self.html_source.count("this.packagingChoice()"), 3)

    def test_step_by_step_request_is_left_untouched(self) -> None:
        call = self.html_source.split("/auto` : `/api/applications/${id}/prepare`", 1)[
            1
        ].split("});", 1)[0]

        # The preparation route takes none of this: it only builds the preview.
        self.assertIn("...(autoAssemble ? {", call)
        self.assertIn("} : {}),", call)


if __name__ == "__main__":
    unittest.main()
