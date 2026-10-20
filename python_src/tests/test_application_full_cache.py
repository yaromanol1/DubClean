from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.paired_reference_cancel import (
    app as app_module,
    pipeline,
)
from experiments.paired_reference_cancel.application_pipeline import (
    _ensure_full_speech_stem,
    _final_user_audio_name,
    _full_output_cache_reusable,
    _full_speech_stem_dependencies,
    _publish_user_audio_files,
    _record_model_output_for,
    _song_protection_completion_status,
    build_full_application,
)


class _TaskStartStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "application_preview": {"scenes": [{"id": "scene"}]},
            "application_result": {"movie": "stale-result.mkv"},
            "application_full_progress": {
                "state": "completed",
                "movie": "stale-result.mkv",
            },
        }
        self.saved: dict | None = None

    def load_project(self, _project_id: str) -> dict:
        return {"id": "project", "application_mode": True}

    def list_pairs(self, _project_id: str) -> list[dict]:
        return [self.pair]

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def save_pair(self, pair: dict) -> None:
        self.pair = pair
        self.saved = dict(pair)

    def create_task(
        self,
        _project_id: str,
        operation: str,
        _pair_id: str,
        parameters: dict,
    ) -> dict:
        return {
            "id": "task",
            "operation": operation,
            "parameters": dict(parameters),
        }


class _SpeechStore:
    def __init__(self, root: Path) -> None:
        original_dir = root / "original_model"
        dubbed_dir = root / "dubbed_model"
        original_dir.mkdir()
        dubbed_dir.mkdir()
        for directory, payload in (
            (original_dir, b"original-weights"),
            (dubbed_dir, b"dubbed-weights"),
        ):
            (directory / "last_best_checkpoint").write_text(
                "model.pt",
                encoding="utf-8",
            )
            (directory / "model.pt").write_bytes(payload)
        self.cfg = {
            "speech_extraction": {
                "checkpoint_dir": str(original_dir),
                "dubbed_checkpoint_dir": str(dubbed_dir),
            }
        }


class _Context:
    child_pid = None


class UserFacingAudioNamesTests(unittest.TestCase):
    def test_final_name_uses_dubbed_source_and_dubclean_marker(self) -> None:
        pair = {
            "name": "Название проекта",
            "sources": {
                "dubbed": {"path": r"E:\Фильмы\Однажды: в Ирландии.mkv"}
            },
        }

        self.assertEqual(
            _final_user_audio_name(pair),
            "Однажды_ в Ирландии · DubClean.flac",
        )

    def test_publisher_keeps_internal_files_and_creates_russian_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_root = root / "application" / "full" / "speech_rebuild"
            full_root.mkdir(parents=True)
            original_speech = full_root / "original_en_speech.flac"
            russian_voice = full_root / "russian_voice.flac"
            final = full_root / "result.flac"
            original_speech.write_bytes(b"original speech")
            russian_voice.write_bytes(b"russian voice")
            final.write_bytes(b"final")
            pair = {
                "name": "Проект",
                "sources": {"dubbed": {"path": r"E:\Фильмы\Фильм.mkv"}},
            }

            published = _publish_user_audio_files(
                pair,
                full_root,
                {
                    "original_en_speech": str(original_speech),
                    "russian_voice": str(russian_voice),
                    "technical_report": str(full_root / "report.json"),
                },
                audio=final,
            )

            self.assertTrue(original_speech.is_file())
            self.assertTrue(russian_voice.is_file())
            self.assertEqual(
                Path(published["original_en_speech"]).name,
                "Иностранная речь из оригинала.flac",
            )
            self.assertEqual(
                Path(published["russian_voice"]).name,
                "Очищенная русская речь.flac",
            )
            self.assertEqual(
                Path(published["audio"]).name,
                "Фильм · DubClean.flac",
            )
            self.assertNotIn("technical_report", published)
            self.assertEqual(Path(published["audio"]).read_bytes(), b"final")


class ApplicationFullStartTests(unittest.TestCase):
    def test_start_hides_previous_result_and_live_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _TaskStartStore(Path(temp_dir))
            with (
                patch.object(app_module, "cfg_store", store),
                patch.object(app_module, "_ensure_no_duplicate"),
                patch.object(app_module, "scheduler_tick"),
                patch.object(
                    app_module,
                    "_sanitize_application_preview",
                    return_value=(store.pair["application_preview"], []),
                ),
                app_module.app.test_request_context(),
            ):
                app_module._start_application_task(
                    "project",
                    "application_full",
                    {"method": "speech_rebuild"},
                )

            self.assertIsNotNone(store.saved)
            self.assertNotIn("application_result", store.saved)
            self.assertNotIn("application_full_progress", store.saved)
            self.assertIn("application_preview", store.saved)

    def test_full_api_forwards_explicit_reference_selection_and_manual_mix(
        self,
    ) -> None:
        for selection in ("raw", "algorithmic"):
            with self.subTest(selection=selection):
                with patch.object(
                    app_module,
                    "_start_application_task",
                    return_value={"ok": True},
                ) as start:
                    response = app_module.app.test_client().post(
                        "/api/applications/project/full",
                        json={
                            "method": "speech_rebuild",
                            "balance_final_mix": False,
                            "reference_selection_mode": selection,
                        },
                    )

                self.assertEqual(response.status_code, 200)
                parameters = start.call_args.args[2]
                self.assertEqual(
                    parameters["reference_selection_mode"],
                    selection,
                )
                self.assertEqual(parameters["reference_adapter"], selection)
                self.assertFalse(parameters["balance_final_mix"])


class PreviewSanitizationTests(unittest.TestCase):
    def test_song_aware_preview_requires_existing_protection_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "application" / "preview_manifest.json"
            scene_root = root / "application" / "previews" / "scene_01"
            scene_root.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            video = scene_root / "video.mp4"
            audio = scene_root / "rebuilt.flac"
            video.write_bytes(b"video")
            audio.write_bytes(b"audio")
            report = root / "application" / "preview_song_protection_report.json"
            pair = {
                "application_preview": {
                    "schema_version": 5,
                    "processing_scope": "speech_and_song_scenes",
                    "song_protection": {
                        "report": str(report),
                        "scenes": [],
                    },
                    "scenes": [
                        {
                            "id": "scene_01",
                            "files": {
                                "video_preview": str(video),
                                "rebuilt_result": str(audio),
                            },
                        }
                    ],
                }
            }

            missing_preview, missing = app_module._sanitize_application_preview(
                pair, root
            )

            self.assertIsNone(missing_preview)
            self.assertIn(str(report), missing)

            report.write_text("{}", encoding="utf-8")
            valid_preview, missing = app_module._sanitize_application_preview(
                pair, root
            )

            self.assertIsNotNone(valid_preview)
            self.assertEqual(
                valid_preview["song_protection"]["report"],
                str(report.resolve()),
            )
            self.assertEqual(missing, [])


class FullSpeechStemCacheTests(unittest.TestCase):
    def test_source_or_extractor_change_invalidates_full_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _SpeechStore(root)
            source = root / "source.flac"
            output = root / "speech.flac"
            source.write_bytes(b"source-v1")
            output.write_bytes(b"speech-v1")
            dependencies = _full_speech_stem_dependencies(
                store,
                source,
                "original",
            )
            _record_model_output_for(output, dependencies)

            self.assertTrue(
                _full_output_cache_reusable(
                    output,
                    dependencies,
                    force_recompute=False,
                )
            )
            self.assertFalse(
                _full_output_cache_reusable(
                    output,
                    dependencies,
                    force_recompute=True,
                )
            )

            source.write_bytes(b"source-v2-with-different-size")
            changed_source_dependencies = _full_speech_stem_dependencies(
                store,
                source,
                "original",
            )
            self.assertFalse(
                _full_output_cache_reusable(
                    output,
                    changed_source_dependencies,
                    force_recompute=False,
                )
            )

            _record_model_output_for(output, changed_source_dependencies)
            original_checkpoint = (
                Path(store.cfg["speech_extraction"]["checkpoint_dir"])
                / "model.pt"
            )
            original_checkpoint.write_bytes(
                b"changed-original-weights-with-different-size"
            )
            changed_model_dependencies = _full_speech_stem_dependencies(
                store,
                source,
                "original",
            )
            self.assertFalse(
                _full_output_cache_reusable(
                    output,
                    changed_model_dependencies,
                    force_recompute=False,
                )
            )

    def test_dubbed_stem_tracks_the_dubbed_extractor_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _SpeechStore(root)
            source = root / "dubbed.flac"
            output = root / "dubbed-speech.flac"
            source.write_bytes(b"dubbed-source")
            output.write_bytes(b"dubbed-speech")
            dependencies = _full_speech_stem_dependencies(
                store,
                source,
                "dubbed",
            )
            _record_model_output_for(output, dependencies)

            self.assertEqual(
                dependencies[-1].parent,
                Path(
                    store.cfg["speech_extraction"]["dubbed_checkpoint_dir"]
                ).resolve(),
            )
            dubbed_checkpoint = dependencies[-1]
            dubbed_checkpoint.write_bytes(
                b"changed-dubbed-weights-with-different-size"
            )
            self.assertFalse(
                _full_output_cache_reusable(
                    output,
                    _full_speech_stem_dependencies(
                        store,
                        source,
                        "dubbed",
                    ),
                    force_recompute=False,
                )
            )

    def test_force_runs_extractor_even_when_sidecar_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _SpeechStore(root)
            source = root / "source.flac"
            shared = root / "shared.flac"
            output = root / "full.flac"
            source.write_bytes(b"source")
            output.write_bytes(b"cached-speech")
            dependencies = _full_speech_stem_dependencies(
                store,
                source,
                "original",
            )
            _record_model_output_for(output, dependencies)

            def fake_extract(
                _store,
                _source,
                destination,
                *_args,
                **_kwargs,
            ) -> None:
                destination.write_bytes(b"fresh-forced-speech")

            with patch.object(
                pipeline,
                "_run_speech_extractor",
                side_effect=fake_extract,
            ) as extractor:
                reused = _ensure_full_speech_stem(
                    store,
                    source,
                    shared,
                    output,
                    _Context(),
                    "pair",
                    "stage",
                    model_role="original",
                    progress_base=0.0,
                    progress_span=18.0,
                    force_recompute=False,
                )
                self.assertEqual(reused, output)
                extractor.assert_not_called()

                rebuilt = _ensure_full_speech_stem(
                    store,
                    source,
                    shared,
                    output,
                    _Context(),
                    "pair",
                    "stage",
                    model_role="original",
                    progress_base=0.0,
                    progress_span=18.0,
                    force_recompute=True,
                )

            self.assertEqual(rebuilt, output)
            self.assertEqual(output.read_bytes(), b"fresh-forced-speech")
            extractor.assert_called_once()
            self.assertTrue(
                _full_output_cache_reusable(
                    output,
                    _full_speech_stem_dependencies(
                        store,
                        source,
                        "original",
                    ),
                    force_recompute=False,
                )
            )

    def test_full_builder_routes_cache_checks_through_force_policy(self) -> None:
        source = inspect.getsource(build_full_application)

        self.assertIn(
            'force_recompute = bool(parameters.get("force"))',
            source,
        )
        self.assertIn("_full_output_cache_reusable(", source)
        self.assertNotIn("_cached_model_output_valid(", source)
        self.assertNotIn("_cached_model_output_valid_for(", source)


class SongProtectionStatusTests(unittest.TestCase):
    def test_classifier_failure_is_not_reported_as_no_songs(self) -> None:
        status = _song_protection_completion_status(
            {
                "enabled": True,
                "classifier": {
                    "available": False,
                    "error": "checkpoint missing",
                },
                "confirmed_intervals": [],
                "restoration_segments": [],
            }
        )

        self.assertIn("Классификатор песен недоступен", status)
        self.assertNotIn("не найдены", status)


if __name__ == "__main__":
    unittest.main()
