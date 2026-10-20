"""The gate that refuses to move a "speech" stem that is really the mixture.

Synchronization copies and moves whole segments of the voice stem.  When the
extractor hands its input back unchanged, the detected "speech segments" cover
the entire soundtrack, and moving hundreds of them shreds a continuous
background into displaced fragments.  These tests pin the measurement and the
refusal, including the cases where the gate must stay out of the way.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from experiments.paired_reference_cancel import speech_synchronizer


ROOT = Path(__file__).parents[2]
RATE = 48000
DURATION = 240.0
# A feature film yields ~126 windows at the shipped 45 s step.  Test material is
# four minutes long, so the step is tightened to reach the same window count
# without writing hour-long fixtures; every threshold stays at its shipped value.
GUARD = {"step_sec": 12.0}


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260811)


def _speech_like(samples: int, rng: np.random.Generator) -> np.ndarray:
    """Amplitude-modulated tone burst train: loud, gappy, speech-shaped."""
    t = np.arange(samples, dtype=np.float64) / RATE
    envelope = (np.sin(2.0 * np.pi * 0.45 * t) > 0.15).astype(np.float64)
    carrier = np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(
        2.0 * np.pi * 900.0 * t
    )
    return 0.30 * envelope * carrier + 0.005 * rng.standard_normal(samples)


def _background_like(samples: int, rng: np.random.Generator) -> np.ndarray:
    """Continuous music/effects bed with no gaps."""
    t = np.arange(samples, dtype=np.float64) / RATE
    return (
        0.12 * np.sin(2.0 * np.pi * 65.0 * t)
        + 0.09 * np.sin(2.0 * np.pi * 147.0 * t)
        + 0.05 * rng.standard_normal(samples)
    )


def _write(path: Path, data: np.ndarray, channels: int = 1) -> Path:
    values = np.asarray(data, dtype=np.float32)
    if channels > 1:
        values = np.repeat(values[:, None], channels, axis=1)
    sf.write(str(path), values, RATE, format="FLAC", subtype="PCM_16")
    return path


class StemSeparationMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._folder = tempfile.TemporaryDirectory()
        root = Path(cls._folder.name)
        rng = _rng()
        samples = int(DURATION * RATE)
        speech = _speech_like(samples, rng)
        background = _background_like(samples, rng)
        cls.mixture = _write(root / "mixture.flac", speech + background)
        cls.separated = _write(root / "separated.flac", speech)
        cls.passthrough = _write(root / "passthrough.flac", speech + background)
        # A model that only applied a global gain has still separated nothing.
        cls.gain_only = _write(
            root / "gain_only.flac", 0.5 * (speech + background)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._folder.cleanup()

    def test_real_separation_is_allowed(self) -> None:
        report = speech_synchronizer.evaluate_stem_separation(
            self.mixture, self.separated, GUARD
        )

        self.assertTrue(report["separated"])
        self.assertEqual(report["reason"], "stem_carries_separated_speech")
        self.assertGreater(report["windows"], 0)

    def test_untouched_stem_is_refused(self) -> None:
        report = speech_synchronizer.evaluate_stem_separation(
            self.mixture, self.passthrough, GUARD
        )

        self.assertFalse(report["separated"])
        self.assertEqual(report["reason"], "stem_repeats_the_mixture")
        measured = report["measured"]
        self.assertGreaterEqual(measured["median_correlation"], 0.93)
        self.assertLessEqual(measured["median_residual_ratio_db"], -9.0)

    def test_a_global_gain_does_not_count_as_separation(self) -> None:
        # Correlation is scale invariant on purpose: rescaling the mixture must
        # not be able to disguise a passthrough as a separated stem.
        report = speech_synchronizer.evaluate_stem_separation(
            self.mixture, self.gain_only, GUARD
        )

        self.assertGreaterEqual(report["measured"]["median_correlation"], 0.93)

    def test_report_records_the_channel_layout_it_compared(self) -> None:
        report = speech_synchronizer.evaluate_stem_separation(
            self.mixture, self.separated, GUARD
        )

        self.assertEqual(report["mixture_channels"], 1)
        self.assertEqual(
            report["algorithm"], speech_synchronizer.SEPARATION_ALGORITHM
        )

    def test_missing_input_never_disables_synchronization(self) -> None:
        report = speech_synchronizer.evaluate_stem_separation(
            self.mixture, Path(self._folder.name) / "absent.flac", GUARD
        )

        self.assertTrue(report["separated"])
        self.assertEqual(report["reason"], "missing_input")

    def test_too_short_material_never_disables_synchronization(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = _rng()
            samples = int(12.0 * RATE)
            body = _speech_like(samples, rng) + _background_like(samples, rng)
            mixture = _write(root / "short_mixture.flac", body)
            stem = _write(root / "short_stem.flac", body)

            report = speech_synchronizer.evaluate_stem_separation(
                mixture, stem, GUARD
            )

            self.assertTrue(report["separated"])
            self.assertEqual(report["reason"], "not_enough_measurable_windows")

    def test_silence_never_disables_synchronization(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = int(DURATION * RATE)
            quiet = np.zeros(samples, dtype=np.float32)
            mixture = _write(root / "quiet_mixture.flac", quiet)
            stem = _write(root / "quiet_stem.flac", quiet)

            report = speech_synchronizer.evaluate_stem_separation(
                mixture, stem, GUARD
            )

            self.assertTrue(report["separated"])
            self.assertEqual(report["reason"], "not_enough_measurable_windows")

    def test_thresholds_are_configurable(self) -> None:
        # Demanding that a passthrough stem also be 10 dB louder than its own
        # input is unreachable, so the same material must now pass the gate.
        strict = speech_synchronizer.evaluate_stem_separation(
            self.mixture,
            self.passthrough,
            {**GUARD, "passthrough_level_drop_db": -10.0},
        )

        self.assertTrue(strict["separated"])
        self.assertEqual(strict["reason"], "stem_carries_separated_speech")

    def test_a_stem_that_empties_speech_free_windows_is_not_a_passthrough(
        self,
    ) -> None:
        # Dialogue-dense material drags all three medians towards passthrough
        # even when the extractor works: there is simply nothing to remove in
        # most windows.  What a passthrough can never do is hand back silence,
        # so windows the extractor emptied outweigh the medians.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = _rng()
            samples = int(DURATION * RATE)
            speech = _speech_like(samples, rng)
            background = _background_like(samples, rng)
            speaking = np.ones(samples, dtype=bool)
            for start, end in ((24.0, 34.0), (48.0, 58.0)):
                speaking[int(start * RATE) : int(end * RATE)] = False
            speech[~speaking] = 0.0
            # Everywhere the film speaks the stem hands the mixture straight
            # back; where it does not, the extractor returns silence.
            mixture = _write(root / "dense_mixture.flac", speech + background)
            stem = _write(
                root / "dense_stem.flac", (speech + background) * speaking
            )

            report = speech_synchronizer.evaluate_stem_separation(
                mixture, stem, GUARD
            )

            measured = report["measured"]
            self.assertGreaterEqual(measured["median_correlation"], 0.93)
            self.assertLessEqual(measured["median_level_drop_db"], 1.5)
            self.assertLessEqual(measured["median_residual_ratio_db"], -9.0)
            self.assertGreaterEqual(measured["emptied_windows"], 1)
            self.assertTrue(report["separated"])
            self.assertEqual(
                report["separation_evidence"],
                "extractor_emptied_speech_free_windows",
            )


class ShippedSeparationGuardConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (
                ROOT
                / "python_src"
                / "experiments"
                / "paired_reference_cancel"
                / "config.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_guard_ships_enabled_with_measured_thresholds(self) -> None:
        guard = self.config["speech_synchronization"]["separation_guard"]

        # Measured on the shipped projects: separated stems reached 0.70-0.81
        # correlation, passthrough stems 0.97-0.99.  The threshold sits in the
        # gap so neither group lands on the wrong side.
        self.assertEqual(guard["passthrough_correlation"], 0.93)
        self.assertEqual(guard["passthrough_level_drop_db"], 1.5)
        self.assertEqual(guard["passthrough_residual_ratio_db"], -9.0)
        self.assertEqual(guard["minimum_windows"], 8)

    def test_defaults_match_the_shipped_configuration(self) -> None:
        guard = self.config["speech_synchronization"]["separation_guard"]

        for key, value in guard.items():
            with self.subTest(key=key):
                self.assertEqual(
                    float(speech_synchronizer.DEFAULT_SEPARATION_GUARD[key]),
                    float(value),
                )


class PipelineWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            ROOT
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "application_pipeline.py"
        ).read_text(encoding="utf-8")

    def test_gate_runs_against_the_extractor_input_and_its_own_output(
        self,
    ) -> None:
        # The comparison is only meaningful against exactly what the extractor
        # was handed, so the mixture argument must be the dubbed source.
        self.assertIn(
            "speech_synchronizer.evaluate_stem_separation(\n"
            "                dubbed_path,\n"
            "                dubbed_speech,",
            self.source,
        )

    def test_a_refused_stem_turns_synchronization_off(self) -> None:
        self.assertIn(
            'if synchronize_speech and not separation_report.get("separated", True):',
            self.source,
        )
        self.assertIn("            synchronize_speech = False", self.source)

    def test_verdict_reaches_the_manifest(self) -> None:
        self.assertIn('"stem_separation": {', self.source)
        self.assertIn(
            '"separated": bool(separation_report.get("separated", True)),',
            self.source,
        )
        self.assertIn("speech_stem_separation_report", self.source)

if __name__ == "__main__":
    unittest.main()
