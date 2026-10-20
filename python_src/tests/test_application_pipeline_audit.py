from __future__ import annotations

import tempfile
import unittest
import inspect
import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.application_pipeline import (
    _begin_application_preparation,
    _calculate_mix_balance,
    _candidate_from_selection,
    _compatible_preparation_phases,
    _deduplicate_selected_song_scenes,
    _is_semantic_subtractor,
    _make_candidate_batches,
    _matched_audio_blocks,
    _protect_preview_song_scenes,
    _preview_song_variant_contracts,
    _save_application_preparation_phase,
    _song_preview_candidates,
    build_full_application,
    prepare_application,
)
from experiments.paired_reference_cancel.pipeline import _read_interval


class _PreparationStore:
    def __init__(self) -> None:
        self.pair = {"id": "pair", "project_id": "project"}

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair.copy()

    def save_pair(self, pair: dict) -> None:
        self.pair = pair


class PreparationResumeTests(unittest.TestCase):
    def test_completed_phases_persist_and_model_change_keeps_speech(self) -> None:
        rate = 8_000
        identity = {
            "source_signature": {"roles": {"original": 1, "dubbed": 2}},
            "checkpoint": {"path": "dubclean_voice", "size": 1, "mtime_ns": 1},
            "background_checkpoint": {"path": "background", "size": 1, "mtime_ns": 1},
            "routing": {"band": "MID"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            speech = root / "speech.flac"
            russian = root / "russian.flac"
            sf.write(speech, np.zeros(rate, dtype=np.float32), rate)
            sf.write(russian, np.zeros(rate, dtype=np.float32), rate)
            store = _PreparationStore()

            _begin_application_preparation(store, "project", "pair", identity)
            _save_application_preparation_phase(
                store, "project", "pair", identity, "speech", {"speech": speech}
            )
            _save_application_preparation_phase(
                store, "project", "pair", identity, "subtraction", {"voice": russian}
            )
            _begin_application_preparation(store, "project", "pair", identity)

            self.assertEqual(
                set(store.pair["application_preparation"]["phases"]),
                {"speech", "subtraction"},
            )

            changed_identity = {
                **identity,
                "checkpoint": {"path": "epoch35", "size": 2, "mtime_ns": 2},
            }
            _begin_application_preparation(
                store, "project", "pair", changed_identity
            )

            self.assertEqual(
                set(store.pair["application_preparation"]["phases"]),
                {"speech"},
            )

    def test_preview_preparation_never_starts_full_film_analysis(self) -> None:
        source = inspect.getsource(prepare_application)

        self.assertNotIn("_fit_or_load_global_mastering_profile(", source)
        self.assertIn('"processing_scope": "speech_and_song_scenes"', source)
        self.assertNotIn("scan_song_candidates(", source)
        self.assertNotIn("_song_preview_candidates(", source)
        self.assertNotIn("song_candidate_scan", source)
        self.assertNotIn("full_track_scan", source)
        self.assertIn(
            '"song_analysis_scope": "selected_preview_scenes_only"',
            source,
        )
        self.assertIn(
            "detect_song_intervals(",
            inspect.getsource(build_full_application),
        )

    def test_starting_new_preview_keeps_previous_completed_result(self) -> None:
        app_path = Path(__file__).parents[1] / "experiments" / "paired_reference_cancel" / "app.py"
        app_source = app_path.read_text(encoding="utf-8")
        tree = ast.parse(app_source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_start_application_task"
        )
        source = ast.get_source_segment(app_source, function) or ""
        prepare_branch = source.split(
            'elif operation == "application_prepare":',
            1,
        )[1].split("cfg_store.save_pair(pair)", 1)[0]

        self.assertNotIn('"application_result"', prepare_branch)
        self.assertNotIn('"application_full_progress"', prepare_branch)

    def test_second_pass_toggle_reuses_completed_base_phases(self) -> None:
        phases = {
            "speech": {"status": "completed"},
            "subtraction": {"status": "completed"},
            "background": {"status": "completed"},
            "preview": {"status": "completed"},
        }
        previous = {
            "identity": {
                "source_signature": {"source": "same"},
                "checkpoint": {"path": "dubclean_voice"},
                "background_checkpoint": {"path": "background"},
                "routing": {"band": "HIGH"},
                "speech_second_pass": {"enabled": False, "extractor": []},
            },
            "phases": phases,
        }
        enabled_identity = {
            **previous["identity"],
            "speech_second_pass": {
                "enabled": True,
                "extractor": [{"path": "speech-extractor"}],
            },
        }

        compatible = _compatible_preparation_phases(previous, enabled_identity)

        self.assertEqual(
            set(compatible),
            {"speech", "subtraction", "background"},
        )
        self.assertNotIn("preview", compatible)

    def test_second_pass_uses_original_speech_extractor_and_separate_artifact(self) -> None:
        source = inspect.getsource(prepare_application)

        self.assertIn('"Повторный проход модуля извлечения речи"', source)
        self.assertIn('model_role="original"', source)
        self.assertIn('"model_ru_voice_second_pass.flac"', source)

    def test_song_config_change_invalidates_all_concatenated_preview_phases(self) -> None:
        previous = {
            "identity": {
                "source_signature": {"source": "same"},
                "checkpoint": {"path": "voice"},
                "background_checkpoint": {"path": "background"},
                "routing": {"band": "HIGH"},
                "speech_second_pass": {"enabled": False, "extractor": []},
                "song_protection": {"config": {"minimum_confidence": 0.78}},
            },
            "phases": {
                "speech": {"status": "completed"},
                "subtraction": {"status": "completed"},
                "background": {"status": "completed"},
                "preview": {"status": "completed"},
            },
        }
        changed = {
            **previous["identity"],
            "song_protection": {
                "config": {"minimum_confidence": 0.82}
            },
        }

        self.assertEqual(
            _compatible_preparation_phases(previous, changed),
            {},
        )

    def test_production_dubclean_voice_checkpoint_is_semantic(self) -> None:
        checkpoint = (
            Path("models")
            / "semantic_ru_separator"
            / "dubclean_voice.pt"
        )

        self.assertTrue(_is_semantic_subtractor(checkpoint))


class ProductUiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).parents[2]
        cls.html_source = (
            root
            / "docs"
            / "концепт интерфейса"
            / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")
        cls.app_source = (
            root
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "app.py"
        ).read_text(encoding="utf-8")
        cls.pipeline_source = inspect.getsource(build_full_application)

    def test_only_song_aware_preview_schema_is_current(self) -> None:
        source = self.html_source

        self.assertIn("Number(preview.schema_version || 0) >= 7", source)
        self.assertIn(
            "preview.processing_scope === 'speech_and_song_scenes'",
            source,
        )
        self.assertIn(
            "preview.song_analysis_scope === "
            "'selected_preview_scenes_only'",
            source,
        )
        self.assertIn("songProtection.report", source)
        self.assertIn("songContract.endpoint === '/api/preview-mix'", source)
        self.assertIn(
            "songContract.detection_voice_parameter === 'song_detection_voice'",
            source,
        )
        self.assertIn(
            "songContract.translation_voice_parameter === 'song_translation_voice'",
            source,
        )
        self.assertIn(
            "songContract.sync_mode_parameter === 'sync_mode'",
            source,
        )
        self.assertIn(
            "songContract.sync_modes.join(',') === 'off,manual,conservative'",
            source,
        )
        self.assertIn(
            "Number(songContract.report_schema_version || 0) >= 1",
            source,
        )

    def test_full_result_prefers_readable_user_audio_files(self) -> None:
        source = self.html_source

        self.assertIn("...(savedResult.user_files || {})", source)
        self.assertIn("...(liveResult.user_files || {})", source)
        self.assertIn(
            "resultUserFiles.russian_voice || resultIntermediates.russian_voice",
            source,
        )
        self.assertIn("resultUserFiles.audio || result.audio", source)

    def test_full_run_uses_explicit_reference_and_manual_mix_levels(self) -> None:
        source = self.html_source

        self.assertIn("const refModes = ['raw','algorithmic'];", source)
        self.assertIn(
            "reference_selection_mode:referenceSelectionMode",
            source,
        )
        # Балансом теперь владеет настройка, а не жёсткое значение в запросе:
        # раньше пошаговая сборка всегда слала false, а автоматическая — true,
        # и один и тот же фильм звучал по-разному.
        self.assertIn("balance_final_mix:this.state.balanceFinalMix !== false", source)
        self.assertNotIn("balance_final_mix:false", source)
        self.assertNotIn("const refModes = ['none','auto'];", source)

    def test_unusable_reference_level_profile_is_fail_closed_in_ui_and_full(
        self,
    ) -> None:
        source = self.html_source

        self.assertIn(
            "previewAppliedSettings: { speech:0, background:0, delay:0, syncMode:2, refMode:0",
            source,
        )
        self.assertIn("syncMode: 2, refMode: 0", source)
        self.assertIn("adapterSummary.usable === true", source)
        self.assertIn("const referenceSelectionMode = refModes[safeRequestedRefMode] ?? 'raw';", source)
        self.assertNotIn("Рекомендуемое выравнивание", source)
        self.assertIn(
            'if selected_adapter == "algorithmic" and not algorithmic_profile_usable:',
            self.pipeline_source,
        )

    def test_new_pair_uses_one_confirmed_probe_snapshot(self) -> None:
        create_project = self.html_source.split(
            "async createBackendProject() {",
            1,
        )[1].split(
            "bestSemanticModel(",
            1,
        )[0]

        self.assertIn(
            "const originalPath = String(originalProbe.path || '').trim();",
            create_project,
        )
        self.assertIn(
            ": String(dubbedProbe.path || '').trim();",
            create_project,
        )
        self.assertIn(
            "original_path: originalPath,",
            create_project,
        )
        self.assertIn(
            "dubbed_path: dubbedPath,",
            create_project,
        )
        self.assertNotIn(
            "original_path: this.state.originalPath",
            create_project,
        )
        self.assertNotIn(
            "dubbed_path: this.state.dubbedPath",
            create_project,
        )

    def test_new_project_does_not_open_hidden_file_pickers_from_submit(self) -> None:
        create_project = self.html_source.split(
            "async createBackendProject() {",
            1,
        )[1].split(
            "bestSemanticModel(",
            1,
        )[0]

        self.assertNotIn("await this.probeSingleFile()", create_project)
        self.assertNotIn("await this.probeFile('original')", create_project)
        self.assertNotIn("await this.probeFile('dubbed')", create_project)
        self.assertIn(
            "Сначала выберите исходный фильм.",
            create_project,
        )
        self.assertIn(
            "Сначала выберите фильм с переводом.",
            create_project,
        )

    def test_late_project_poll_cannot_restore_project_into_new_form(self) -> None:
        refresh = self.html_source.split(
            "async refreshActiveApplication() {",
            1,
        )[1].split(
            "applicationStatePatch(activeApplication) {",
            1,
        )[0]

        self.assertIn(
            "if (this.state.activeApplication?.project?.id !== id) return;",
            refresh,
        )

    def test_late_poll_cannot_restore_stale_running_task(self) -> None:
        refresh = self.html_source.split(
            "async refreshActiveApplication() {",
            1,
        )[1].split(
            "applicationStatePatch(activeApplication) {",
            1,
        )[0]

        self.assertIn(
            "if (pollSequence !== this.__applicationPollSequence) return;",
            refresh,
        )
        self.assertIn(
            "['failed','error','interrupted'].includes(task.state)",
            refresh,
        )
        self.assertIn("this.showNotice(message);", refresh)
        self.assertNotIn("catch (_) {}", refresh)

    def test_background_poll_does_not_queue_duplicate_requests(self) -> None:
        refresh = self.html_source.split(
            "async refreshActiveApplication() {",
            1,
        )[1].split(
            "applicationStatePatch(activeApplication) {",
            1,
        )[0]
        catalog = self.html_source.split(
            "async refreshCatalog() {",
            1,
        )[1].split(
            "async refreshActiveApplication() {",
            1,
        )[0]

        self.assertIn("this.__activeRefreshPromise", refresh)
        self.assertIn("this.__catalogRefreshPromise", catalog)
        self.assertNotIn("await this.refreshCatalog();", refresh)
        self.assertIn("this.updateApplicationCatalogSummary(activeForState);", refresh)

    def test_track_failure_remains_visible_with_retry_available(self) -> None:
        source = self.html_source

        self.assertIn(
            "const failedMatchTask = !matchTask",
            source,
        )
        self.assertIn(
            "failedMatchTask.error || failedMatchTask.substage",
            source,
        )
        self.assertIn(
            'value="{{ matchTaskErrorVisible }}"',
            source,
        )
        self.assertIn("{{ matchTaskError }}", source)
        self.assertIn(
            "matchPrepareAction:()=> matchTask ? this.stopPrepareTracks() : this.prepareTracks()",
            source,
        )

    def test_project_creation_is_guarded_against_double_submit(self) -> None:
        create_project = self.html_source.split(
            "async createBackendProject() {",
            1,
        )[1].split(
            "bestSemanticModel(",
            1,
        )[0]

        self.assertIn("if (this.state.creatingProject) return;", create_project)
        self.assertIn("this.setState({ creatingProject:true });", create_project)
        self.assertIn("this.setState({ creatingProject:false });", create_project)
        self.assertIn(
            'disabled="{{ projectActionDisabled }}"',
            self.html_source,
        )

    def test_repeated_track_extraction_never_starts_hidden_full_build(self) -> None:
        prepare_tracks = self.html_source.split(
            "async prepareTracks() {",
            1,
        )[1].split(
            "async stopPrepareTracks() {",
            1,
        )[0]

        self.assertIn(
            "await this.api(`/api/applications/${id}/tracks`, {",
            prepare_tracks,
        )
        self.assertNotIn(
            "`/api/applications/${id}/auto`",
            prepare_tracks,
        )
        self.assertIn(
            "this.go('match');",
            prepare_tracks,
        )

    def test_source_changes_reset_previous_full_assembly_choice(self) -> None:
        source = self.html_source
        application_patch = source.split(
            "applicationStatePatch(activeApplication) {",
            1,
        )[1].split(
            "async openBackendProject(id) {",
            1,
        )[0]
        select_track = source.split(
            "selectTrackOption(trackIndex, optionIndex, keepMenuOpen=false) {",
            1,
        )[1].split(
            "async createBackendProject() {",
            1,
        )[0]
        save_sources = source.split(
            "async saveProjectSources() {",
            1,
        )[1].split(
            "startProjectAction() {",
            1,
        )[0]
        swap_tracks = source.split(
            "async swapTracks() {",
            1,
        )[1].split(
            "renderVals() {",
            1,
        )[0]

        self.assertIn("autoAssemble: false,", application_patch)
        self.assertNotIn(
            "autoAssemble: !!activeApplication?.project?.auto_assemble",
            application_patch,
        )
        self.assertIn("autoAssemble:false", select_track)
        self.assertIn(
            "body:JSON.stringify({ auto_assemble:false })",
            save_sources,
        )
        self.assertIn("autoAssemble: false,", swap_tracks)
        self.assertIn(
            "body:JSON.stringify({ auto_assemble:false })",
            swap_tracks,
        )

    def test_switching_distinct_pair_to_single_never_keeps_an_ambiguous_file(
        self,
    ) -> None:
        source_mode = self.html_source.split(
            "setSourceMode(mode) {",
            1,
        )[1].split(
            "selectTrackOption(",
            1,
        )[0]

        self.assertIn(
            "if (originalPath && dubbedPath && originalPath !== dubbedPath)",
            source_mode,
        )
        self.assertIn("originalProbe:null", source_mode)
        self.assertIn("dubbedProbe:null", source_mode)
        self.assertIn("originalPath:''", source_mode)
        self.assertIn("dubbedPath:''", source_mode)
        self.assertIn(
            "Выберите один файл, содержащий оригинал и перевод.",
            source_mode,
        )

    def test_source_draft_is_not_overwritten_by_background_poll(self) -> None:
        refresh = self.html_source.split(
            "async refreshActiveApplication() {",
            1,
        )[1].split(
            "applicationStatePatch(activeApplication) {",
            1,
        )[0]
        matching = self.html_source.split(
            "// ---- MATCHING ----",
            1,
        )[1].split(
            "// ---- PREVIEW ----",
            1,
        )[0]

        self.assertIn("if (this.__sourceMutationInFlight) return;", refresh)
        self.assertIn(
            "this.state.sourceDraftActive && this.state.activeApplication",
            refresh,
        )
        self.assertIn(
            "pair: this.state.activeApplication.pair",
            refresh,
        )
        self.assertIn(
            "String(s.originalPath || srcOriginal?.path || '')",
            matching,
        )
        self.assertIn(
            "String(s.dubbedPath || srcDubbed?.path || '')",
            matching,
        )
        save_sources = self.html_source.split(
            "async saveProjectSources() {",
            1,
        )[1].split(
            "startProjectAction() {",
            1,
        )[0]
        self.assertIn(
            "this.setState({ sourceDraftActive:false });",
            save_sources,
        )

    def test_export_playback_waits_for_final_video_but_folder_does_not(
        self,
    ) -> None:
        export_rendering = self.html_source.split(
            "const finalVideoReady = !!result.movie;",
            1,
        )[1].split(
            "// ---- SETTINGS ----",
            1,
        )[0]

        self.assertIn(
            "playReady:!!item.filePath && finalVideoReady",
            export_rendering,
        )
        self.assertIn(
            "playDisabled:!row.playReady, openDisabled:!row.ready",
            export_rendering,
        )
        self.assertIn(
            "play:()=>row.playReady && this.selectExportPreviewTrack(row.key)",
            export_rendering,
        )
        self.assertIn(
            'disabled="{{ e.playDisabled }}"',
            self.html_source,
        )
        self.assertIn(
            'disabled="{{ e.openDisabled }}"',
            self.html_source,
        )
        self.assertIn(
            "exportPreviewReady: finalVideoReady && !!exportPictureUrl",
            self.html_source,
        )

    def test_final_screen_hides_internal_song_protection_diagnostics(
        self,
    ) -> None:
        export_template = self.html_source.split(
            "<!-- SCREEN 7: СБОРКА ФИЛЬМА -->",
            1,
        )[1].split(
            "<!-- SCREEN 8: SETTINGS -->",
            1,
        )[0]
        export_rendering = self.html_source.split(
            "const finalVideoReady = !!result.movie;",
            1,
        )[1].split(
            "// ---- SETTINGS ----",
            1,
        )[0]

        self.assertNotIn("Защита песен", export_template)
        self.assertNotIn("JSON-отчёт", export_template)
        self.assertNotIn("songProtectionStatusVisible", export_template)
        self.assertNotIn("До защиты песен", export_rendering)
        self.assertNotIn("before_song_protection", export_rendering)
        self.assertNotIn("Итоговый звук после защиты песен", self.html_source)
        self.assertNotIn("'before-song-protection'", self.html_source)

    def test_pair_replacement_persists_inferred_source_mode(self) -> None:
        replace_pair = self.app_source.split(
            "def api_replace_pair(project_id: str, pair_id: str):",
            1,
        )[1].split(
            '@app.post("/api/projects/<project_id>/pairs/<pair_id>/material-variants/',
            1,
        )[0]

        self.assertIn(
            '"single" if _same_local_path(original_path, dubbed_path) else "pair"',
            replace_pair,
        )
        self.assertIn(
            'project["application_source_mode"] = inferred_source_mode',
            replace_pair,
        )
        self.assertIn("cfg_store.save_project(project)", replace_pair)

    def test_rebuilt_preview_urls_include_manifest_generation(self) -> None:
        source = self.html_source
        preview_rendering = source.split(
            "// ---- PREVIEW ----",
            1,
        )[1].split(
            "// ---- PROCESSING ----",
            1,
        )[0]

        self.assertIn(
            "const previewManifest = "
            "s.activeApplication?.pair?.application_preview || {};",
            preview_rendering,
        )
        self.assertIn(
            "previewManifest.created_at || 'preview'",
            preview_rendering,
        )
        self.assertGreaterEqual(
            preview_rendering.count(
                "encodeURIComponent(previewMediaRevision)",
            ),
            4,
        )
        self.assertNotIn(
            "v=${encodeURIComponent(s.previewAppliedToken || 0)}",
            preview_rendering,
        )

    def test_full_backend_preserves_explicit_reference_selection(self) -> None:
        self.assertIn(
            '"reference_selection_mode": reference_selection_mode',
            self.app_source,
        )
        self.assertIn(
            'parameters.get("reference_selection_mode")',
            self.pipeline_source,
        )

    def test_algorithmic_preview_never_selects_hidden_global_or_neural_variant(
        self,
    ) -> None:
        source = self.html_source

        self.assertIn("files.model_ru_voice_algorithmic", source)
        self.assertIn("files.synchronized_ru_voice_algorithmic", source)
        self.assertNotIn("files.model_ru_voice_global", source)
        self.assertNotIn("files.synchronized_ru_voice_global", source)
        self.assertNotIn("files.model_ru_voice_neural", source)
        self.assertNotIn("files.synchronized_ru_voice_neural", source)
        self.assertNotIn(
            "files.model_ru_voice_algorithmic || files.model_ru_voice",
            source,
        )
        self.assertNotIn(
            "files.synchronized_ru_voice_algorithmic || files.synchronized_ru_voice",
            source,
        )
        self.assertIn("const algorithmicBaseReady = scenes.every", source)
        self.assertIn("const selectedVariantsReady = scenes.every", source)
        self.assertIn(
            "files.synchronized_ru_voice_algorithmic_second_pass",
            source,
        )

    def test_song_preview_sends_selected_voice_evidence_and_sync_mode(
        self,
    ) -> None:
        source = self.html_source

        self.assertIn("isSongPreviewScene(scene)", source)
        self.assertNotIn(
            "beforeSongProtection\n        && files.song_protection_report",
            source,
        )
        self.assertIn("song_detection_voice=${encodeURIComponent(", source)
        self.assertIn("song_translation_voice=${encodeURIComponent(", source)
        self.assertIn("sync_mode=${encodeURIComponent(syncMode)}", source)
        self.assertIn("['off','manual','conservative']", source)
        self.assertIn(
            'request.args.get("song_detection_voice", "")',
            self.app_source,
        )
        self.assertIn(
            'request.args.get("song_translation_voice", "")',
            self.app_source,
        )
        self.assertIn(
            'request.args.get("sync_mode", "off")',
            self.app_source,
        )
        self.assertIn(
            "secondPass=previewSecondPassVisible",
            source,
        )

    def test_partial_song_overlap_is_not_labelled_as_a_song_card(self) -> None:
        source = self.html_source
        helper = source.split(
            "const isSongPreviewScene = (scene={}) => !!(",
            1,
        )[1].split("const scenes =", 1)[0]

        self.assertIn("scene.song_protection?.protected", helper)
        self.assertIn("scene.preview_kind", helper)
        self.assertNotIn("full_track_candidates", helper)

    def test_song_damage_decision_is_independent_of_user_mix_gains(self) -> None:
        self.assertIn(
            "preview-mix-v7-neutral-song-decision-gains",
            self.app_source,
        )
        self.assertIn(
            '"detection_background_gain_db": 0.0',
            self.app_source,
        )
        self.assertIn(
            '"detection_voice_gain_db": 0.0',
            self.app_source,
        )
        self.assertIn(
            'full_root / "song_detection_processed_mix_neutral.flac"',
            self.pipeline_source,
        )
        self.assertIn(
            '"purpose": "song_damage_decision_only"',
            self.pipeline_source,
        )
        self.assertIn(
            '"decision_mix_gains_db"',
            self.pipeline_source,
        )


class SongPreviewSelectionTests(unittest.TestCase):
    def test_every_selected_preview_scene_is_analysed_locally(self) -> None:
        rate = 8_000
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenes: list[dict] = []
            for index, start_sec in enumerate((12.0, 48.0), 1):
                scene_root = (
                    root
                    / "application"
                    / "previews"
                    / f"scene_{index:02d}"
                )
                scene_root.mkdir(parents=True)
                files: dict[str, str] = {}
                for key in (
                    "original_song_mix",
                    "dubbed_mix",
                    "dubbed_en_ru_speech",
                    "specialized_me",
                    "original_song_speech",
                    "model_ru_voice",
                ):
                    path = scene_root / f"{key}.flac"
                    sf.write(
                        path,
                        np.full(rate, index * 0.01, dtype=np.float32),
                        rate,
                    )
                    files[key] = str(path)
                scenes.append(
                    {
                        "id": f"scene_{index:02d}",
                        "dubbed_start_sec": start_sec,
                        "duration_sec": 1.0,
                        "files": files,
                    }
                )

            local_report = {
                "schema_version": 1,
                "enabled": True,
                "classifier": {"backend": "test"},
                "confirmed_intervals": [],
                "restoration_segments": [],
                "summary": "Песня не подтверждена.",
            }
            with patch(
                "experiments.paired_reference_cancel.application_pipeline."
                "song_protection.detect_song_intervals",
                side_effect=lambda *args, **kwargs: dict(local_report),
            ) as detector:
                report = _protect_preview_song_scenes(
                    root,
                    scenes,
                    {},
                    None,
                    "pair",
                )

        self.assertEqual(detector.call_count, len(scenes))
        analysed_sources = {
            Path(call.args[0]).parent.name
            for call in detector.call_args_list
        }
        self.assertEqual(analysed_sources, {"scene_01", "scene_02"})
        self.assertEqual(
            report["analysis_scope"],
            "selected_preview_scenes_only",
        )
        self.assertEqual(report["summary"]["analysed_scene_count"], 2)
        self.assertNotIn("full_track_scan", report)
        self.assertTrue(
            all(
                scene["song_protection"]["analysis_scope"] == "scene_local"
                for scene in scenes
            )
        )

    def test_song_contract_maps_second_pass_detection_to_first_pass_dialogue(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files: dict[str, str] = {}
            for key in (
                "original_song_speech",
                "algorithmic_en_speech",
                "model_ru_voice",
                "model_ru_voice_algorithmic",
                "model_ru_voice_second_pass",
                "model_ru_voice_algorithmic_second_pass",
                "synchronized_ru_voice",
                "synchronized_ru_voice_algorithmic",
                "synchronized_ru_voice_second_pass",
                "synchronized_ru_voice_algorithmic_second_pass",
            ):
                path = root / f"{key}.flac"
                sf.write(
                    path,
                    np.zeros(800, dtype=np.float32),
                    8_000,
                )
                files[key] = str(path)

            variants = _preview_song_variant_contracts(files)

        raw_second = variants["raw_second_pass"]
        algorithmic_second = variants["algorithmic_second_pass"]
        self.assertEqual(
            Path(raw_second["detection_voice"]).name,
            "model_ru_voice_second_pass.flac",
        )
        self.assertEqual(
            Path(raw_second["translation_voice"]).name,
            "model_ru_voice.flac",
        )
        self.assertEqual(
            Path(algorithmic_second["detection_reference"]).name,
            "algorithmic_en_speech.flac",
        )
        self.assertEqual(
            Path(algorithmic_second["translation_voice"]).name,
            "model_ru_voice_algorithmic.flac",
        )
        synchronized = raw_second["playback_voices"][1]
        self.assertTrue(synchronized["synchronized"])
        self.assertEqual(
            synchronized["allowed_sync_modes"],
            ["conservative"],
        )

    def test_real_song_zones_create_grid_aligned_scenes_without_partial_dedup(
        self,
    ) -> None:
        report = {
            "candidate_intervals": [
                {"start_sec": 710.0, "end_sec": 745.0, "confidence": 0.9},
                {"start_sec": 1025.0, "end_sec": 1055.0, "confidence": 0.9},
                {"start_sec": 1245.0, "end_sec": 1275.0, "confidence": 0.9},
            ]
        }
        songs = _song_preview_candidates(report, 1400.0, {})
        speech_scene = {
            "zone": "50_percent",
            "preview_kind": "speech",
            "dubbed_start_sec": 717.8,
            "duration_sec": 30.0,
        }

        selected = _deduplicate_selected_song_scenes(
            [speech_scene, *songs]
        )

        self.assertEqual(len(songs), 3)
        self.assertTrue(
            all(
                30.0 <= float(item["duration_sec"]) <= 40.0
                for item in songs
            )
        )
        self.assertTrue(
            all(
                abs(float(item["dubbed_start_sec"]) / 5.0
                    - round(float(item["dubbed_start_sec"]) / 5.0))
                < 1e-6
                for item in songs
            )
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0]["preview_kind"], "speech")
        starts = [
            round(float(item["dubbed_start_sec"]), 1)
            for item in selected[1:]
        ]
        self.assertEqual(starts, [710.0, 1025.0, 1245.0])

    def test_grid_aligned_speech_scene_covering_whole_song_is_deduplicated(
        self,
    ) -> None:
        report = {
            "candidate_intervals": [
                {"start_sec": 710.0, "end_sec": 745.0, "confidence": 0.9},
            ]
        }
        song = _song_preview_candidates(report, 1400.0, {})[0]
        speech_scene = {
            "zone": "50_percent",
            "preview_kind": "speech",
            "dubbed_start_sec": 710.0,
            "duration_sec": 35.0,
        }

        selected = _deduplicate_selected_song_scenes(
            [speech_scene, song],
            scan_step_sec=5.0,
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected[0]["preview_kind"],
            "speech_and_song_candidate",
        )
        self.assertEqual(len(selected[0]["song_candidates"]), 1)


class CandidateMappingTests(unittest.TestCase):
    def test_regular_preview_batch_reads_the_original_track_directly(self) -> None:
        rate = 8_000
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            extracted = root / "extracted"
            extracted.mkdir(parents=True)
            original = np.linspace(-0.5, 0.5, rate * 3, dtype=np.float32)
            dubbed = np.linspace(0.5, -0.5, rate * 3, dtype=np.float32)
            sf.write(extracted / "original.flac", original, rate)
            sf.write(extracted / "dubbed.flac", dubbed, rate)

            original_batch, dubbed_batch, ranges = _make_candidate_batches(
                root,
                [
                    {
                        "preview_kind": "speech",
                        "original_start_sec": 0.5,
                        "dubbed_start_sec": 0.5,
                        "duration_sec": 1.0,
                        "speed_ratio": 1.0,
                    }
                ],
                rate,
            )

            self.assertTrue(original_batch.is_file())
            self.assertTrue(dubbed_batch.is_file())
            self.assertEqual(ranges, [(0, rate)])
            self.assertEqual(sf.info(str(original_batch)).frames, rate * 2)

    def test_negative_original_start_is_preserved_not_clamped(self) -> None:
        # A manual correction near the head of the original file can map a
        # preview window before the start of the original.  The mapping must
        # stay truthful: ``pipeline._read_interval`` zero-pads negative starts,
        # while clamping to 0 silently shifts the reference by that amount.
        alignment = {
            "manual_correction_sec": -5.0,
            "segments": [
                {
                    "id": 0,
                    "dubbed_start": 0.0,
                    "dubbed_end": 40.0,
                    "original_start": 2.0,
                    "speed_ratio": 1.0,
                    "confidence": 0.9,
                    "usable": True,
                }
            ],
        }
        selection = {"preview_start_sec": 0.0, "preview_duration_sec": 30.0}

        candidate = _candidate_from_selection(alignment, selection)

        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(candidate["original_start_sec"], -3.0, places=3)

    def test_read_interval_zero_pads_negative_start(self) -> None:
        rate = 8_000
        ramp = np.linspace(0.0, 1.0, rate * 2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ramp.wav"
            sf.write(path, ramp, rate, subtype="FLOAT")

            data, read_rate = _read_interval(path, -0.5, 1.0)

        self.assertEqual(read_rate, rate)
        self.assertEqual(len(data), rate)
        half = rate // 2
        self.assertTrue(np.all(data[:half] == 0.0))
        np.testing.assert_allclose(data[half:, 0], ramp[:half], atol=1e-6)


class MatchedAudioBlocksTests(unittest.TestCase):
    def test_short_voice_tail_is_padded_with_silence_not_stretched(self) -> None:
        # Voice 3 s, background 5 s, block size 2 s: the block covering 2..4 s
        # only has 1 s of voice left.  The old code FFT-resampled that second
        # to two seconds, slowing the last phrases down and moving the click
        # from 2.5 s to 3.0 s.  The correct behaviour is silence padding.
        rate = 8_000
        background = np.full(5 * rate, 0.5, dtype=np.float32)
        voice = np.zeros(3 * rate, dtype=np.float32)
        click_index = int(2.5 * rate)
        voice[click_index] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            background_path = root / "background.wav"
            voice_path = root / "voice.wav"
            sf.write(background_path, background, rate)
            sf.write(voice_path, voice, rate)

            with sf.SoundFile(str(background_path)) as background_reader:
                with sf.SoundFile(str(voice_path)) as voice_reader:
                    blocks = list(
                        _matched_audio_blocks(
                            background_reader, voice_reader, blocksize=2 * rate
                        )
                    )

        for background_block, voice_block in blocks:
            self.assertEqual(len(background_block), len(voice_block))
        joined_voice = np.concatenate([voice_block for _, voice_block in blocks])
        self.assertEqual(len(joined_voice), len(background))
        peak_index = int(np.argmax(np.abs(joined_voice[:, 0])))
        self.assertEqual(peak_index, click_index)
        self.assertEqual(float(np.max(np.abs(joined_voice[3 * rate :]))), 0.0)


class MixBalanceClampTests(unittest.TestCase):
    def test_extreme_ratio_correction_is_clamped(self) -> None:
        # Original speech much louder than the Russian voice produces a huge
        # negative raw background correction; it must be limited to
        # ``max_adjust_db`` and reported as clamped.
        rate = 8_000
        seconds = 3.0
        time = np.arange(int(rate * seconds), dtype=np.float32) / rate
        background = 0.05 * np.sin(2.0 * np.pi * 173.0 * time)
        original_speech = 0.5 * np.sin(2.0 * np.pi * 311.0 * time)
        russian_voice = 0.02 * np.sin(2.0 * np.pi * 401.0 * time)
        original_mix = background + original_speech

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "background": root / "background.wav",
                "original_speech": root / "original_speech.wav",
                "russian_voice": root / "russian_voice.wav",
                "original_mix": root / "original_mix.wav",
            }
            sf.write(paths["background"], background, rate)
            sf.write(paths["original_speech"], original_speech, rate)
            sf.write(paths["russian_voice"], russian_voice, rate)
            sf.write(paths["original_mix"], original_mix, rate)

            report = _calculate_mix_balance(
                paths["background"],
                paths["original_speech"],
                paths["russian_voice"],
                paths["original_mix"],
                dubbed_speech_path=paths["original_speech"],
                dubbed_mix_path=paths["original_mix"],
            )

        self.assertTrue(report["enabled"])
        self.assertTrue(report["clamped"])
        level = report["program_level_match"]
        self.assertLessEqual(abs(level["background_gain_db"]), 12.0 + 1e-6)
        self.assertGreater(report["raw_voice_gain_db"], 12.0)
        self.assertGreaterEqual(report["voice_gain_db"], -1.5)
        self.assertTrue(level["speech_attenuation_forbidden"])


if __name__ == "__main__":
    unittest.main()
