from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import app as app_module
from experiments.paired_reference_cancel import quality_comparison_builder as builder


class QualityComparisonBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.rate = 8000
        self.windows = tuple(
            builder.ComparisonWindow(
                case_id=f"case-{index + 1:02d}",
                file_stem=f"case_{index + 1:02d}_test",
                event_start_sec=index + 0.20,
                event_end_sec=index + 0.40,
                clip_start_sec=float(index),
                clip_duration_sec=0.80,
            )
            for index in range(7)
        )
        frames = self.rate * 8
        timeline = np.arange(frames, dtype=np.float64) / self.rate
        background = np.column_stack(
            (
                0.015 * np.sin(2.0 * np.pi * 97.0 * timeline),
                0.012 * np.sin(2.0 * np.pi * 131.0 * timeline),
            )
        ).astype(np.float32)
        before_voice = np.zeros((frames, 1), dtype=np.float32)
        after_voice = np.zeros((frames, 1), dtype=np.float32)
        for window in self.windows:
            left = int(round(window.event_start_sec * self.rate))
            right = int(round(window.event_end_sec * self.rate))
            local = np.arange(right - left, dtype=np.float64) / self.rate
            after_voice[left:right, 0] = (
                0.08 * np.sin(2.0 * np.pi * 220.0 * local)
            ).astype(np.float32)
        self.before_voice = self.root / "before_voice.wav"
        self.after_voice = self.root / "after_voice.wav"
        self.before_mix = self.root / "before_mix.wav"
        self.after_mix = self.root / "after_mix.wav"
        sf.write(
            self.before_voice, before_voice, self.rate, format="WAV", subtype="FLOAT"
        )
        sf.write(
            self.after_voice, after_voice, self.rate, format="WAV", subtype="FLOAT"
        )
        sf.write(
            self.before_mix, background + before_voice, self.rate, format="WAV", subtype="FLOAT"
        )
        sf.write(
            self.after_mix, background + after_voice, self.rate, format="WAV", subtype="FLOAT"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _test_extractor(
        source: Path,
        destination: Path,
        window: builder.ComparisonWindow,
    ) -> None:
        values, rate = builder._read_interval(
            source, window.clip_start_sec, window.clip_duration_sec
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, values[:, :2], rate, format="WAV", subtype="PCM_16")

    def test_builds_exactly_seven_mix_clips_and_voice_metrics(self) -> None:
        output = self.root / "comparison"

        manifest = builder.build_comparison(
            before_voice=self.before_voice,
            before_mix=self.before_mix,
            after_voice=self.after_voice,
            after_mix=self.after_mix,
            output_dir=output,
            windows=self.windows,
            after_dependencies=(),
            clip_extractor=self._test_extractor,
        )

        self.assertTrue(manifest["published"])
        self.assertEqual(len(manifest["intervals"]), 7)
        self.assertEqual(manifest["summary"]["улучшено"], 7)
        self.assertEqual(
            manifest["intervals"][0]["metrics"]["before_dbfs"],
            "цифровая тишина",
        )
        self.assertIsInstance(
            manifest["intervals"][0]["metrics"]["after_dbfs"], float
        )
        self.assertEqual(len(list((output / "before").glob("*.wav"))), 7)
        self.assertEqual(len(list((output / "after").glob("*.wav"))), 7)
        report = json.loads((output / "build_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["mix_contract"]["valid"])
        self.assertLess(
            report["mix_contract"]["background_residual_max_abs"], 1e-6
        )

    def test_different_background_is_rejected_and_comparison_stays_hidden(self) -> None:
        wrong_mix = self.root / "wrong_after_mix.wav"
        values, rate = sf.read(
            self.after_mix, always_2d=True, dtype="float32"
        )
        values[:, 0] += 0.01
        sf.write(wrong_mix, values, rate, format="WAV", subtype="FLOAT")
        output = self.root / "comparison"

        with self.assertRaisesRegex(ValueError, "разным фоном"):
            builder.build_comparison(
                before_voice=self.before_voice,
                before_mix=self.before_mix,
                after_voice=self.after_voice,
                after_mix=wrong_mix,
                output_dir=output,
                windows=self.windows,
                after_dependencies=(),
                clip_extractor=self._test_extractor,
            )

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["published"])
        self.assertEqual(manifest["publication_status"], builder.PUBLICATION_WAITING)

    def test_after_older_than_processing_code_is_rejected(self) -> None:
        dependency = self.root / "new_processing.py"
        dependency.write_text("# new", encoding="utf-8")
        newer = max(self.after_voice.stat().st_mtime, self.after_mix.stat().st_mtime) + 10
        os.utime(dependency, (newer, newer))
        output = self.root / "comparison"

        with self.assertRaisesRegex(ValueError, "старше действующего кода"):
            builder.build_comparison(
                before_voice=self.before_voice,
                before_mix=self.before_mix,
                after_voice=self.after_voice,
                after_mix=self.after_mix,
                output_dir=output,
                windows=self.windows,
                after_dependencies=(dependency,),
                clip_extractor=self._test_extractor,
            )

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["published"])


class UnpublishedQualityComparisonRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "quality_comparisons"
        folder = self.root / "waiting"
        (folder / "audio").mkdir(parents=True)
        (folder / "audio" / "old.wav").write_bytes(b"RIFF-old")
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "published": False,
                    "publication_status": builder.PUBLICATION_WAITING,
                    "title": "Старое сравнение",
                    "intervals": [
                        {
                            "start_sec": 1.0,
                            "end_sec": 2.0,
                            "before_audio": "audio/old.wav",
                            "after_audio": "audio/old.wav",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.root_patch = mock.patch.object(
            app_module, "_quality_comparisons_root", return_value=self.root
        )
        self.root_patch.start()
        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def test_unpublished_manifest_and_audio_are_not_exposed(self) -> None:
        catalog = self.client.get("/api/quality-comparisons")
        detail = self.client.get("/api/quality-comparisons/waiting")
        audio = self.client.get(
            "/api/quality-comparisons/waiting/audio/audio/old.wav"
        )

        self.assertEqual(catalog.get_json(), {"comparisons": []})
        self.assertEqual(detail.status_code, 409)
        self.assertEqual(audio.status_code, 404)
        self.assertNotIn(b"RIFF-old", audio.data)


if __name__ == "__main__":
    unittest.main()
