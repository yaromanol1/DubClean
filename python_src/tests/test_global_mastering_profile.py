from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.global_mastering_profile import (
    SegmentSelectionReport,
    SpeechFreeSegment,
    estimate_global_profile,
    fit_global_profile_for_files,
)
from experiments.paired_reference_cancel.application_pipeline import _global_profile_usable


class GlobalMasteringProfileTests(unittest.TestCase):
    sample_rate = 8_000
    segment_sec = 1.0

    def make_fixture(
        self,
        fit_gain_db: float,
        *,
        holdout_gain_db: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, SegmentSelectionReport]:
        rng = np.random.default_rng(20260717)
        frames = int(self.sample_rate * self.segment_sec)
        original_parts: list[np.ndarray] = []
        dubbed_parts: list[np.ndarray] = []
        segments: list[SpeechFreeSegment] = []
        roles = ["fit", "fit", "holdout", "fit", "fit", "holdout"]
        for index, role in enumerate(roles):
            # Coloured, non-stationary audio exercises all mel bands more
            # realistically than a single tone while staying deterministic.
            noise = rng.standard_normal(frames).astype(np.float32)
            envelope = np.linspace(0.25, 1.0, frames, dtype=np.float32)
            original = noise * envelope * np.float32(0.03)
            gain_db = (
                holdout_gain_db
                if role == "holdout" and holdout_gain_db is not None
                else fit_gain_db
            )
            dubbed = original * np.float32(10.0 ** (float(gain_db) / 20.0))
            original_parts.append(original)
            dubbed_parts.append(dubbed)
            segments.append(
                SpeechFreeSegment(
                    start_sec=index * self.segment_sec,
                    duration_sec=self.segment_sec,
                    refined_offset_ms=0.0,
                    coherence=0.99,
                    original_rms_db=-30.0,
                    dubbed_rms_db=-30.0 + float(gain_db),
                    role=role,
                )
            )
        return (
            np.concatenate(original_parts),
            np.concatenate(dubbed_parts),
            SegmentSelectionReport(segments=segments),
        )

    def profile(self, fit_gain_db: float, *, holdout_gain_db: float | None = None):
        original, dubbed, selection = self.make_fixture(
            fit_gain_db, holdout_gain_db=holdout_gain_db
        )
        return estimate_global_profile(
            original,
            dubbed,
            self.sample_rate,
            selection,
            n_fft=256,
            hop=128,
            bands=20,
        )

    def test_stable_gain_profile_is_usable(self) -> None:
        profile = self.profile(6.0)
        summary = profile["summary"]
        self.assertEqual(summary["verdict"], "ok")
        self.assertTrue(summary["usable"])
        self.assertEqual(summary["recommended_mode"], "global")
        self.assertAlmostEqual(summary["median_gain_db"], 6.0, delta=0.35)
        self.assertGreater(summary["holdout_improvement_percent"], 80.0)

    def test_saturated_profile_is_rejected(self) -> None:
        profile = self.profile(18.0)
        summary = profile["summary"]
        self.assertEqual(summary["verdict"], "saturated_profile")
        self.assertFalse(summary["usable"])
        self.assertTrue(summary["profile_saturated"])
        self.assertEqual(summary["recommended_mode"], "raw")

    def test_harmful_holdout_keeps_negative_improvement_and_rejects(self) -> None:
        profile = self.profile(6.0, holdout_gain_db=-6.0)
        summary = profile["summary"]
        self.assertEqual(summary["verdict"], "no_holdout_gain")
        self.assertFalse(summary["usable"])
        self.assertLess(summary["holdout_improvement_percent"], 0.0)

    def test_pipeline_accepts_only_validated_profile(self) -> None:
        self.assertTrue(_global_profile_usable({"summary": {"verdict": "ok"}}))
        self.assertTrue(
            _global_profile_usable({"summary": {"verdict": "ok", "usable": True}})
        )
        self.assertFalse(
            _global_profile_usable(
                {"summary": {"verdict": "saturated_profile", "usable": False}}
            )
        )
        self.assertFalse(_global_profile_usable({"summary": {"verdict": "matched_channels"}}))

    def test_file_report_preserves_source_timeline_positions(self) -> None:
        rate = 2_000
        seconds = 120
        rng = np.random.default_rng(91)
        original = rng.standard_normal(rate * seconds).astype(np.float32) * 0.03
        dubbed = original * np.float32(10.0 ** (3.0 / 20.0))
        speech = np.zeros_like(original)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "original": root / "original.wav",
                "dubbed": root / "dubbed.wav",
                "original_speech": root / "original_speech.wav",
                "dubbed_speech": root / "dubbed_speech.wav",
            }
            sf.write(paths["original"], original, rate)
            sf.write(paths["dubbed"], dubbed, rate)
            sf.write(paths["original_speech"], speech, rate)
            sf.write(paths["dubbed_speech"], speech, rate)
            with mock.patch(
                "experiments.paired_reference_cancel.global_mastering_profile.normalized_xcorr",
                return_value=(0.0, 1.0),
            ), mock.patch(
                "experiments.paired_reference_cancel.global_mastering_profile._spectral_coherence",
                return_value=0.99,
            ):
                profile = fit_global_profile_for_files(
                    paths["original"],
                    paths["dubbed"],
                    paths["original_speech"],
                    paths["dubbed_speech"],
                    max_segments=4,
                )
        starts = [item["start_sec"] for item in profile["selection"]["segments"]]
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(any(value >= 45.0 for value in starts), starts)


if __name__ == "__main__":
    unittest.main()
