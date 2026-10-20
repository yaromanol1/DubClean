from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import app as app_module
from experiments.paired_reference_cancel import song_protection


class PreviewSongMixApiTests(unittest.TestCase):
    def _audio(
        self,
        path: Path,
        value: float,
        seconds: float = 2.0,
    ) -> Path:
        rate = 8000
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            path,
            np.full(int(rate * seconds), value, dtype=np.float32),
            rate,
            subtype=(
                "PCM_16"
                if path.suffix.casefold() == ".flac"
                else "FLOAT"
            ),
        )
        return path

    def _contract(
        self,
        scene: Path,
        *,
        original: Path,
        dubbed: Path,
        dubbed_speech: Path,
        background: Path,
        detection_reference: Path,
        detection_voice: Path,
        translation_voice: Path,
        playback_voice: Path,
        schema_version: int | None = None,
    ) -> Path:
        report = scene / song_protection.PREVIEW_CONTRACT_NAME
        report.write_text(
            json.dumps(
                {
                    "schema_version": (
                        song_protection.PREVIEW_CONTRACT_SCHEMA_VERSION
                        if schema_version is None
                        else schema_version
                    ),
                    "report_kind": (
                        song_protection.PREVIEW_CONTRACT_KIND
                    ),
                    "report_name": (
                        song_protection.PREVIEW_CONTRACT_NAME
                    ),
                    "scene_id": scene.name,
                    "scene_directory": str(scene.resolve()),
                    "config": {
                        "enabled": True,
                        "crossfade_sec": 0.1,
                    },
                    "sources": {
                        "original_detection_mix": str(original.resolve()),
                        "dubbed_mix": str(dubbed.resolve()),
                        "dubbed_speech_stem": str(
                            dubbed_speech.resolve()
                        ),
                        "music_background": str(background.resolve()),
                    },
                    "variants": {
                        "raw_first_pass": {
                            "reference_mode": "raw",
                            "speech_pass": "first",
                            "detection_reference": str(
                                detection_reference.resolve()
                            ),
                            "detection_voice": str(
                                detection_voice.resolve()
                            ),
                            "translation_voice": str(
                                translation_voice.resolve()
                            ),
                            "playback_voices": [
                                {
                                    "path": str(
                                        playback_voice.resolve()
                                    ),
                                    "synchronized": False,
                                    "allowed_sync_modes": [
                                        "off",
                                        "manual",
                                    ],
                                }
                            ],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return report

    def _fake_mux(self, commands: list[list[str]]):
        def fake_run(command, **_kwargs):
            values = [str(value) for value in command]
            commands.append(values)
            destination = Path(values[-1])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"preview" * 512)
            return SimpleNamespace(returncode=0, stderr=b"")

        return fake_run

    def _query(
        self,
        *,
        video: Path,
        voice: Path,
        background: Path,
        report: Path,
        original: Path,
        dubbed: Path,
        detection_voice: Path,
        translation_voice: Path,
        sync_mode: str = "off",
        delay: str = "0",
        voice_gain: str = "0",
        background_gain: str = "0",
    ) -> dict[str, str]:
        return {
            "video": str(video),
            "voice": str(voice),
            "background": str(background),
            "voice_gain": voice_gain,
            "background_gain": background_gain,
            "delay": delay,
            "song_protection": "1",
            "song_report": str(report),
            "song_original": str(original),
            "song_dubbed": str(dubbed),
            "song_detection_voice": str(detection_voice),
            "song_translation_voice": str(translation_voice),
            "sync_mode": sync_mode,
        }

    def test_dynamic_protection_uses_selected_detection_layers_and_exact_mix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene = root / "scene_01"
            scene.mkdir()
            runtime = root / "runtime"
            video = scene / "rebuilt_preview.mp4"
            video.write_bytes(b"video")
            playback_voice = self._audio(
                scene / "playback_voice.wav", 0.9
            )
            detection_voice = self._audio(
                scene / "detection_voice.wav", 0.1
            )
            translation_voice = self._audio(
                scene / "translation_voice.wav", 0.15
            )
            background = self._audio(scene / "background.wav", 0.4)
            original = self._audio(scene / "original.wav", 0.7)
            dubbed = self._audio(scene / "dubbed.wav", 0.5)
            dubbed_speech = self._audio(
                scene / "dubbed_speech.wav", 0.2
            )
            detection_reference = self._audio(
                scene / "detection_reference.wav", 0.3
            )
            report = self._contract(
                scene,
                original=original,
                dubbed=dubbed,
                dubbed_speech=dubbed_speech,
                background=background,
                detection_reference=detection_reference,
                detection_voice=detection_voice,
                translation_voice=translation_voice,
                playback_voice=playback_voice,
            )
            commands: list[list[str]] = []
            observed: dict[str, object] = {}

            def fake_detect(
                original_mix,
                dubbed_mix,
                speech_stem,
                music_background,
                processed_mix,
                _config,
                **kwargs,
            ):
                values, _ = sf.read(
                    str(processed_mix),
                    dtype="float32",
                    always_2d=True,
                )
                observed["detection_mean"] = float(np.mean(values))
                observed["original_speech_stem"] = kwargs[
                    "original_speech_stem"
                ]
                observed["translation_voice_stem"] = kwargs[
                    "translation_voice_stem"
                ]
                return {
                    "schema_version": song_protection.SCHEMA_VERSION,
                    "config": _config,
                    "sources": {},
                    "confirmed_intervals": [],
                    "restoration_segments": [
                        {
                            "start_sec": 0.5,
                            "end_sec": 1.5,
                            "restore_source": "original_aligned",
                        }
                    ],
                    "summary": "confirmed",
                }

            def fake_apply(
                source,
                _dubbed,
                destination,
                _segments,
                **_kwargs,
            ):
                values, rate = sf.read(
                    str(source),
                    dtype="float32",
                    always_2d=True,
                )
                observed["playback_peak"] = float(
                    np.max(np.abs(values))
                )
                sf.write(
                    str(destination),
                    values,
                    rate,
                    format="FLAC",
                    subtype="PCM_24",
                )
                return destination

            with (
                patch.object(
                    app_module,
                    "_safe_public_file",
                    side_effect=lambda value: Path(value).resolve(),
                ),
                patch.object(app_module, "RUNTIME_DIR", runtime),
                patch.object(
                    app_module,
                    "check_ffmpeg",
                    return_value="ffmpeg",
                ),
                patch.object(
                    app_module.song_protection,
                    "detect_song_intervals",
                    side_effect=fake_detect,
                ) as detector,
                patch.object(
                    app_module.song_protection,
                    "apply_song_protection",
                    side_effect=fake_apply,
                ),
                patch.object(
                    app_module.subprocess,
                    "run",
                    side_effect=self._fake_mux(commands),
                ),
            ):
                response = app_module.app.test_client().get(
                    "/api/preview-mix",
                    query_string=self._query(
                        video=video,
                        voice=playback_voice,
                        background=background,
                        report=report,
                        original=original,
                        dubbed=dubbed,
                        detection_voice=detection_voice,
                        translation_voice=translation_voice,
                        voice_gain="-6",
                        background_gain="-12",
                    ),
                )

            status = response.status_code
            body = response.get_data(as_text=True)
            header = response.headers.get(
                "X-DubClean-Song-Protection"
            )
            response.close()

        self.assertEqual(status, 200, body)
        self.assertEqual(header, "applied")
        self.assertEqual(len(commands), 1)
        detector.assert_called_once()
        self.assertAlmostEqual(
            float(observed["detection_mean"]),
            0.5,
            places=3,
        )
        self.assertNotAlmostEqual(
            float(observed["playback_peak"]),
            float(observed["detection_mean"]),
            places=2,
        )
        self.assertEqual(
            observed["original_speech_stem"],
            detection_reference.resolve(),
        )
        self.assertEqual(
            observed["translation_voice_stem"],
            translation_voice.resolve(),
        )

    def test_disabled_protection_never_runs_song_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "runtime"
            video = root / "video.mp4"
            video.write_bytes(b"video")
            voice = self._audio(root / "voice.wav", 0.2)
            background = self._audio(root / "background.wav", 0.3)
            commands: list[list[str]] = []
            with (
                patch.object(
                    app_module,
                    "_safe_public_file",
                    side_effect=lambda value: Path(value).resolve(),
                ),
                patch.object(app_module, "RUNTIME_DIR", runtime),
                patch.object(
                    app_module,
                    "check_ffmpeg",
                    return_value="ffmpeg",
                ),
                patch.object(
                    app_module.song_protection,
                    "detect_song_intervals",
                ) as detector,
                patch.object(
                    app_module.subprocess,
                    "run",
                    side_effect=self._fake_mux(commands),
                ),
            ):
                response = app_module.app.test_client().get(
                    "/api/preview-mix",
                    query_string={
                        "video": str(video),
                        "voice": str(voice),
                        "background": str(background),
                        "song_protection": "0",
                    },
                )
            status = response.status_code
            header = response.headers.get(
                "X-DubClean-Song-Protection"
            )
            response.close()

        self.assertEqual(status, 200)
        self.assertEqual(header, "disabled")
        self.assertEqual(len(commands), 1)
        detector.assert_not_called()

    def test_invalid_contract_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "scene_01"
            scene.mkdir()
            video = scene / "video.mp4"
            video.write_bytes(b"video")
            files = {
                key: self._audio(scene / f"{key}.wav", 0.1)
                for key in (
                    "voice",
                    "background",
                    "original",
                    "dubbed",
                    "dubbed_speech",
                    "reference",
                    "detection",
                    "translation",
                )
            }
            report = self._contract(
                scene,
                original=files["original"],
                dubbed=files["dubbed"],
                dubbed_speech=files["dubbed_speech"],
                background=files["background"],
                detection_reference=files["reference"],
                detection_voice=files["detection"],
                translation_voice=files["translation"],
                playback_voice=files["voice"],
                schema_version=999,
            )
            with patch.object(
                app_module,
                "_safe_public_file",
                side_effect=lambda value: Path(value).resolve(),
            ):
                response = app_module.app.test_client().get(
                    "/api/preview-mix",
                    query_string=self._query(
                        video=video,
                        voice=files["voice"],
                        background=files["background"],
                        report=report,
                        original=files["original"],
                        dubbed=files["dubbed"],
                        detection_voice=files["detection"],
                        translation_voice=files["translation"],
                    ),
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "версия или имя",
            response.get_data(as_text=True),
        )

    def test_any_cross_scene_contract_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene = root / "scene_01"
            other_scene = root / "scene_02"
            scene.mkdir()
            other_scene.mkdir()
            video = scene / "video.mp4"
            video.write_bytes(b"video")
            playback_voice = self._audio(
                other_scene / "voice.wav", 0.1
            )
            background = self._audio(scene / "background.wav", 0.1)
            original = self._audio(scene / "original.wav", 0.1)
            dubbed = self._audio(scene / "dubbed.wav", 0.1)
            dubbed_speech = self._audio(
                scene / "dubbed_speech.wav", 0.1
            )
            reference = self._audio(scene / "reference.wav", 0.1)
            detection = self._audio(scene / "detection.wav", 0.1)
            translation = self._audio(scene / "translation.wav", 0.1)
            report = self._contract(
                scene,
                original=original,
                dubbed=dubbed,
                dubbed_speech=dubbed_speech,
                background=background,
                detection_reference=reference,
                detection_voice=detection,
                translation_voice=translation,
                playback_voice=playback_voice,
            )
            with patch.object(
                app_module,
                "_safe_public_file",
                side_effect=lambda value: Path(value).resolve(),
            ):
                response = app_module.app.test_client().get(
                    "/api/preview-mix",
                    query_string=self._query(
                        video=video,
                        voice=playback_voice,
                        background=background,
                        report=report,
                        original=original,
                        dubbed=dubbed,
                        detection_voice=detection,
                        translation_voice=translation,
                    ),
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "одной папке сцены",
            response.get_data(as_text=True),
        )

    def test_non_manual_mode_rejects_hidden_delay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            scene = Path(temp_dir) / "scene_01"
            scene.mkdir()
            video = scene / "video.mp4"
            video.write_bytes(b"video")
            files = {
                key: self._audio(scene / f"{key}.wav", 0.1)
                for key in (
                    "voice",
                    "background",
                    "original",
                    "dubbed",
                    "dubbed_speech",
                    "reference",
                    "detection",
                    "translation",
                )
            }
            report = self._contract(
                scene,
                original=files["original"],
                dubbed=files["dubbed"],
                dubbed_speech=files["dubbed_speech"],
                background=files["background"],
                detection_reference=files["reference"],
                detection_voice=files["detection"],
                translation_voice=files["translation"],
                playback_voice=files["voice"],
            )
            with patch.object(
                app_module,
                "_safe_public_file",
                side_effect=lambda value: Path(value).resolve(),
            ):
                response = app_module.app.test_client().get(
                    "/api/preview-mix",
                    query_string=self._query(
                        video=video,
                        voice=files["voice"],
                        background=files["background"],
                        report=report,
                        original=files["original"],
                        dubbed=files["dubbed"],
                        detection_voice=files["detection"],
                        translation_voice=files["translation"],
                        sync_mode="off",
                        delay="0.25",
                    ),
                )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "только в ручном режиме",
            response.get_data(as_text=True),
        )


if __name__ == "__main__":
    unittest.main()
