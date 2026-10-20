from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import audio_mix
from experiments.paired_reference_cancel.application_pipeline import (
    _calculate_mix_balance,
    _mix_audio_files,
)


class ApplicationMixBalanceTests(unittest.TestCase):
    @staticmethod
    def _write_paths(
        root: Path,
        rate: int,
        *,
        background: np.ndarray,
        source_speech: np.ndarray,
        russian_voice: np.ndarray,
        source_mix: np.ndarray,
    ) -> dict[str, Path]:
        paths = {
            "background": root / "background.wav",
            "original_speech": root / "original_speech.wav",
            "russian_voice": root / "russian_voice.wav",
            "original_mix": root / "original_mix.wav",
        }
        sf.write(paths["background"], background, rate)
        sf.write(paths["original_speech"], source_speech, rate)
        sf.write(paths["russian_voice"], russian_voice, rate)
        sf.write(paths["original_mix"], source_mix, rate)
        return paths

    def test_neutral_mix_matches_original_programme_level(self) -> None:
        rate = 8_000
        seconds = 3.0
        time = np.arange(int(rate * seconds), dtype=np.float32) / rate
        background = 0.035 * np.sin(2.0 * np.pi * 173.0 * time)
        original_speech = 0.11 * np.sin(2.0 * np.pi * 311.0 * time)
        russian_voice = 0.065 * np.sin(2.0 * np.pi * 401.0 * time)
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

        level = report["program_level_match"]
        self.assertTrue(report["enabled"])
        self.assertTrue(level["enabled"])
        self.assertGreater(report["voice_gain_db"], 0.0)
        self.assertGreaterEqual(report["voice_gain_db"], -1.5)
        self.assertAlmostEqual(
            level["predicted_before_boost_db"] + level["boost_only_gain_db"],
            level["target_dubbed_active_db"],
            places=5,
        )
        self.assertTrue(level["speech_attenuation_forbidden"])

    def test_background_is_matched_on_the_same_low_speech_frames(self) -> None:
        rate = 8_000
        seconds = 4.0
        time = np.arange(int(rate * seconds), dtype=np.float32) / rate
        target_background = 0.08 * np.sin(2.0 * np.pi * 173.0 * time)
        restored_background = 0.02 * np.sin(2.0 * np.pi * 173.0 * time)
        speech = np.zeros_like(time)
        speech[len(speech) // 2 :] = (
            0.10
            * np.sin(2.0 * np.pi * 311.0 * time[len(speech) // 2 :])
        )
        source_mix = target_background + speech

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._write_paths(
                Path(temp_dir),
                rate,
                background=restored_background,
                source_speech=speech,
                russian_voice=speech,
                source_mix=source_mix,
            )
            report = _calculate_mix_balance(
                paths["background"],
                paths["original_speech"],
                paths["russian_voice"],
                paths["original_mix"],
                dubbed_speech_path=paths["original_speech"],
                dubbed_mix_path=paths["original_mix"],
            )

        match = report["background_level_match"]
        self.assertTrue(match["ok"])
        self.assertEqual(match["method"], "paired_low_speech_frames")
        self.assertAlmostEqual(match["raw_gain_db"], 12.04, delta=0.2)
        self.assertGreater(report["background_gain_db"], 11.5)

    def test_matching_background_is_not_reduced_by_dialogue_gate(self) -> None:
        rate = 8_000
        seconds = 4.0
        time = np.arange(int(rate * seconds), dtype=np.float32) / rate
        background = 0.07 * np.sin(2.0 * np.pi * 173.0 * time)
        speech = np.zeros_like(time)
        speech[rate : 3 * rate] = (
            0.20 * np.sin(2.0 * np.pi * 311.0 * time[rate : 3 * rate])
        )
        source_mix = background + speech

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = self._write_paths(
                Path(temp_dir),
                rate,
                background=background,
                source_speech=speech,
                russian_voice=speech,
                source_mix=source_mix,
            )
            report = _calculate_mix_balance(
                paths["background"],
                paths["original_speech"],
                paths["russian_voice"],
                paths["original_mix"],
                dubbed_speech_path=paths["original_speech"],
                dubbed_mix_path=paths["original_mix"],
            )

        self.assertAlmostEqual(
            report["background_level_match"]["raw_gain_db"], 0.0, delta=0.15
        )
        self.assertGreaterEqual(report["background_gain_db"], -0.15)

    def test_preview_and_full_mix_share_delay_gain_channel_and_clip_math(
        self,
    ) -> None:
        rate = 8_000
        frames = rate * 2
        background = np.column_stack(
            (
                np.full(frames, 0.55, dtype=np.float32),
                np.full(frames, 0.45, dtype=np.float32),
            )
        )
        voice = np.full(frames, 0.65, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            background_path = root / "background.wav"
            voice_path = root / "voice.wav"
            preview_path = root / "preview.flac"
            full_path = root / "full.flac"
            sf.write(
                background_path,
                background,
                rate,
                subtype="FLOAT",
            )
            sf.write(voice_path, voice, rate, subtype="FLOAT")

            audio_mix.mix_audio_files(
                background_path,
                voice_path,
                preview_path,
                background_gain_db=1.5,
                voice_gain_db=2.0,
                voice_delay_sec=0.25,
            )
            _mix_audio_files(
                background_path,
                voice_path,
                full_path,
                background_gain_db=1.5,
                voice_gain_db=2.0,
                voice_delay_sec=0.25,
            )
            preview, _ = sf.read(
                preview_path,
                dtype="float32",
                always_2d=True,
            )
            full, _ = sf.read(
                full_path,
                dtype="float32",
                always_2d=True,
            )

        np.testing.assert_array_equal(preview, full)
        delay_frames = rate // 4
        expected_background_left = min(
            0.98,
            0.55 * 10.0 ** (1.5 / 20.0),
        )
        self.assertAlmostEqual(
            float(preview[delay_frames - 1, 0]),
            expected_background_left,
            places=5,
        )
        self.assertAlmostEqual(
            float(np.max(preview[delay_frames:, 0])),
            0.98,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
