from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import yaml

from experiments.paired_reference_cancel.speech_synchronizer import (
    ALGORITHM,
    SCHEMA_VERSION,
    SpeechSegment,
    plan_segment_moves,
    synchronize_voice_file,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "paired_reference_cancel"
    / "config.yaml"
)


class ConservativeSpeechSynchronizationTests(unittest.TestCase):
    def test_algorithm_identity_matches_portable_cache_config(self) -> None:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        synchronization = config["speech_synchronization"]

        self.assertTrue(synchronization["enabled"])
        self.assertEqual(synchronization["algorithm"], ALGORITHM)
        self.assertEqual(synchronization["schema_version"], SCHEMA_VERSION)
        self.assertGreater(synchronization["maximum_advance_sec"], 0.0)
        self.assertGreater(synchronization["crossfade_sec"], 0.0)
        self.assertTrue(synchronization["preserve_unmatched_audio"])
        self.assertTrue(
            synchronization["fallback_to_unsynchronized_on_contract_failure"]
        )

    @staticmethod
    def _phrase(
        target: np.ndarray,
        rate: int,
        start: float,
        end: float,
        *,
        amplitude: float = 0.35,
        frequency: float = 220.0,
    ) -> None:
        left = int(round(start * rate))
        right = int(round(end * rate))
        phase = np.arange(right - left, dtype=np.float32)
        target[left:right, 0] = amplitude * np.sin(
            2.0 * np.pi * frequency * phase / rate
        )

    def test_moves_delayed_phrases_earlier_without_delaying_early_speech(self) -> None:
        english = [
            SpeechSegment(1.0, 1.6),
            SpeechSegment(4.0, 4.7),
            SpeechSegment(7.0, 7.6),
        ]
        russian = [
            SpeechSegment(1.45, 1.95),
            SpeechSegment(5.20, 5.75),
            SpeechSegment(6.90, 7.45),
        ]

        moves = plan_segment_moves(russian, english, maximum_advance_sec=0.85)

        self.assertAlmostEqual(moves[0].destination_start_sec, 1.10, places=6)
        self.assertAlmostEqual(moves[0].shift_sec, -0.35, places=6)
        # The large lag is corrected conservatively, not snapped by 1.1 s.
        self.assertAlmostEqual(moves[1].destination_start_sec, 4.35, places=6)
        self.assertAlmostEqual(moves[1].shift_sec, -0.85, places=6)
        # Speech which is already early is never delayed.
        self.assertAlmostEqual(moves[2].destination_start_sec, 6.90, places=6)
        self.assertAlmostEqual(moves[2].shift_sec, 0.0, places=6)

    def test_file_report_and_audio_contain_real_local_moves(self) -> None:
        rate = 8_000
        duration_sec = 7.0
        frames = int(rate * duration_sec)
        english = np.zeros((frames, 1), dtype=np.float32)
        russian = np.zeros((frames, 1), dtype=np.float32)

        self._phrase(english, rate, 1.0, 1.6)
        self._phrase(english, rate, 4.0, 4.7)
        self._phrase(russian, rate, 1.45, 1.95)
        self._phrase(russian, rate, 5.20, 5.75)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            english_path = root / "english.flac"
            russian_path = root / "russian.flac"
            output_path = root / "synchronized.flac"
            sf.write(english_path, english, rate, subtype="PCM_24")
            sf.write(russian_path, russian, rate, subtype="PCM_24")

            report = synchronize_voice_file(
                russian_path,
                english_path,
                output_path,
            )
            synchronized, _ = sf.read(
                output_path, dtype="float32", always_2d=True
            )

        self.assertEqual(report["mode"], "neural_stems_conservative_onset_alignment")
        self.assertEqual(report["algorithm"], ALGORITHM)
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["matched_segments"], 2)
        self.assertEqual(report["shifted_segments"], 2)
        self.assertEqual(report["applied_shifted_segments"], 2)
        self.assertFalse(report["fallback_to_unsynchronized"])
        self.assertTrue(report["contract"]["valid"])
        self.assertGreater(report["mean_advance_sec"], 0.2)
        self.assertGreater(float(np.max(np.abs(synchronized[int(1.0 * rate):int(1.3 * rate)]))), 0.05)
        # The original delayed location is no longer the start of the first phrase.
        self.assertLess(float(np.max(np.abs(synchronized[int(1.85 * rate):int(2.05 * rate)]))), 0.05)

    def test_quiet_undetected_speech_is_preserved_in_place(self) -> None:
        rate = 8_000
        frames = rate * 6
        english = np.zeros((frames, 1), dtype=np.float32)
        russian = np.zeros((frames, 1), dtype=np.float32)
        self._phrase(english, rate, 1.0, 1.6)
        self._phrase(russian, rate, 1.45, 1.95)
        # This phrase is below the detector's -52 dB lower bound.  The old
        # empty-canvas reconstruction silently deleted it.
        self._phrase(
            russian,
            rate,
            3.0,
            3.5,
            amplitude=0.001,
            frequency=330.0,
        )

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            english_path = root / "english.flac"
            russian_path = root / "russian.flac"
            output_path = root / "synchronized.flac"
            sf.write(english_path, english, rate, subtype="PCM_24")
            sf.write(russian_path, russian, rate, subtype="PCM_24")
            report = synchronize_voice_file(russian_path, english_path, output_path)
            synchronized, _ = sf.read(output_path, dtype="float32", always_2d=True)

        quiet = slice(int(3.05 * rate), int(3.45 * rate))
        before_rms = float(np.sqrt(np.mean(russian[quiet, 0] ** 2)))
        after_rms = float(np.sqrt(np.mean(synchronized[quiet, 0] ** 2)))
        self.assertFalse(report["fallback_to_unsynchronized"])
        self.assertGreater(after_rms, before_rms * 0.95)
        self.assertLess(after_rms, before_rms * 1.05)

    def test_shifted_segment_is_removed_then_inserted_exactly_once(self) -> None:
        rate = 8_000
        frames = rate * 4
        english = np.zeros((frames, 1), dtype=np.float32)
        russian = np.zeros((frames, 1), dtype=np.float32)
        self._phrase(english, rate, 0.8, 1.4)
        self._phrase(russian, rate, 1.25, 1.85)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            english_path = root / "english.flac"
            russian_path = root / "russian.flac"
            output_path = root / "synchronized.flac"
            sf.write(english_path, english, rate, subtype="PCM_24")
            sf.write(russian_path, russian, rate, subtype="PCM_24")
            report = synchronize_voice_file(russian_path, english_path, output_path)
            synchronized, _ = sf.read(output_path, dtype="float32", always_2d=True)

        moved = next(move for move in report["moves"] if move["shift_sec"] < 0.0)
        destination_left = int(round(moved["destination_start_sec"] * rate))
        destination_right = destination_left + int(
            round((moved["source_end_sec"] - moved["source_start_sec"]) * rate)
        )
        source_right = int(round(moved["source_end_sec"] * rate))
        fade = int(round(report["crossfade_sec"] * rate))
        destination = synchronized[
            destination_left + fade:destination_right - fade, 0
        ]
        vacated_tail = synchronized[
            destination_right + fade:max(destination_right + fade, source_right - fade),
            0,
        ]
        source_energy = float(np.sum(russian[:, 0].astype(np.float64) ** 2))
        output_energy = float(np.sum(synchronized[:, 0].astype(np.float64) ** 2))
        self.assertEqual(report["applied_shifted_segments"], 1)
        self.assertGreater(float(np.max(np.abs(destination))), 0.20)
        self.assertGreater(len(vacated_tail), 0)
        self.assertLess(float(np.max(np.abs(vacated_tail))), 1e-3)
        # A duplicated copy would make this ratio approach two.
        self.assertGreater(output_energy / source_energy, 0.90)
        self.assertLess(output_energy / source_energy, 1.10)

    def test_preservation_interval_prevents_a_move_and_remains_unchanged(self) -> None:
        rate = 8_000
        frames = rate * 4
        english = np.zeros((frames, 1), dtype=np.float32)
        russian = np.zeros((frames, 1), dtype=np.float32)
        self._phrase(english, rate, 0.8, 1.4)
        self._phrase(russian, rate, 1.25, 1.85)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            english_path = root / "english.flac"
            russian_path = root / "russian.flac"
            output_path = root / "synchronized.flac"
            sf.write(english_path, english, rate, subtype="PCM_24")
            sf.write(russian_path, russian, rate, subtype="PCM_24")
            report = synchronize_voice_file(
                russian_path,
                english_path,
                output_path,
                preservation_intervals=[SpeechSegment(1.20, 1.90)],
            )
            source, _ = sf.read(russian_path, dtype="float32", always_2d=True)
            synchronized, _ = sf.read(output_path, dtype="float32", always_2d=True)

        self.assertEqual(report["applied_shifted_segments"], 0)
        self.assertEqual(report["skipped_shifted_segments"], 1)
        self.assertEqual(report["skipped_moves"][0]["reason"], "preservation_interval")
        np.testing.assert_allclose(synchronized, source, atol=2e-7)

    def test_contract_failure_falls_back_to_original_unsynchronized_audio(self) -> None:
        rate = 8_000
        frames = rate * 4
        english = np.zeros((frames, 1), dtype=np.float32)
        russian = np.zeros((frames, 1), dtype=np.float32)
        self._phrase(english, rate, 0.8, 1.4)
        self._phrase(russian, rate, 1.25, 1.85)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            english_path = root / "english.flac"
            russian_path = root / "russian.flac"
            output_path = root / "synchronized.flac"
            sf.write(english_path, english, rate, subtype="PCM_24")
            sf.write(russian_path, russian, rate, subtype="PCM_24")
            with patch(
                "experiments.paired_reference_cancel.speech_synchronizer."
                "_validate_synchronization_contract",
                return_value={"valid": False, "reasons": ["forced_test_failure"]},
            ):
                report = synchronize_voice_file(
                    russian_path,
                    english_path,
                    output_path,
                )
            source, _ = sf.read(russian_path, dtype="float32", always_2d=True)
            synchronized, _ = sf.read(output_path, dtype="float32", always_2d=True)

        self.assertTrue(report["fallback_to_unsynchronized"])
        self.assertEqual(report["applied_shifted_segments"], 0)
        self.assertEqual(report["contract"]["reasons"], ["forced_test_failure"])
        np.testing.assert_allclose(synchronized, source, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
