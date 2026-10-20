from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel.reference_adapter import (
    PROFILE_MODE,
    PROFILE_SCHEMA_VERSION,
    ReferenceAdapterProfile,
    adapt_reference_file_with_profile,
    apply_profile,
    fit_algorithmic_profile,
)


class ReferenceAdapterTests(unittest.TestCase):
    sample_rate = 16_000
    seconds = 8

    @classmethod
    def setUpClass(cls) -> None:
        frames = cls.sample_rate * cls.seconds

        def coloured(seed: int, level: float) -> np.ndarray:
            rng = np.random.default_rng(seed)
            values = signal.lfilter(
                [1.0, -0.25],
                [1.0, -0.92],
                rng.standard_normal(frames),
            ).astype(np.float32)
            envelope = (
                0.45
                + 0.55
                * np.sin(np.linspace(0.0, 13.0 * np.pi, frames, dtype=np.float32)) ** 2
            )
            values *= envelope
            values *= np.float32(level / max(float(np.std(values)), 1e-8))
            return values.astype(np.float32)

        cls.reference = coloured(101, 0.02)
        # Four times the reference RMS is deliberately a strong independent
        # translated speaker.  It must increase Syy but not bias E[Y*conj(X)].
        cls.independent_ru = coloured(202, 0.08)

    def fit_plus_six(self):
        mixture = (
            self.reference * np.float32(10.0 ** (6.0 / 20.0))
            + self.independent_ru
        )
        return fit_algorithmic_profile(
            self.reference,
            mixture,
            self.sample_rate,
        )

    def test_common_english_gain_survives_strong_independent_ru(self) -> None:
        profile = self.fit_plus_six()
        report = profile.to_json()
        self.assertEqual(report["schema_version"], PROFILE_SCHEMA_VERSION)
        self.assertEqual(report["mode"], PROFILE_MODE)
        self.assertTrue(report["summary"]["usable"], report["summary"])
        self.assertAlmostEqual(profile.common_gain_db, 6.0, delta=0.75)
        self.assertGreaterEqual(report["summary"]["support_bins"], 8)
        self.assertGreaterEqual(report["summary"]["median_coherence"], 0.18)
        self.assertIn("gain_mad_db", report["summary"])
        self.assertIn("support", report["summary"])

    def test_no_common_english_fails_closed_to_identity(self) -> None:
        unrelated = np.roll(self.independent_ru, 3_971)
        profile = fit_algorithmic_profile(
            self.reference,
            unrelated,
            self.sample_rate,
        )
        self.assertFalse(profile.summary["usable"])
        self.assertEqual(profile.common_gain_db, 0.0)
        np.testing.assert_array_equal(
            apply_profile(self.reference, profile),
            self.reference,
        )

    def test_additive_ru_does_not_manufacture_gain_at_unity(self) -> None:
        mixture = self.reference + self.independent_ru
        profile = fit_algorithmic_profile(
            self.reference,
            mixture,
            self.sample_rate,
        )
        self.assertFalse(profile.summary["usable"])
        self.assertEqual(profile.common_gain_db, 0.0)
        np.testing.assert_array_equal(
            apply_profile(self.reference, profile),
            self.reference,
        )

    def test_scalar_application_is_linear_and_chunk_independent(self) -> None:
        profile = self.fit_plus_six()
        self.assertTrue(profile.summary["usable"])
        source = self.reference[:31_337]
        whole = apply_profile(source, profile)
        chunks = np.concatenate(
            [
                apply_profile(source[:1_003], profile),
                apply_profile(source[1_003:17_111], profile),
                apply_profile(source[17_111:], profile),
            ]
        )
        expected = source * np.float32(10.0 ** (profile.common_gain_db / 20.0))
        np.testing.assert_array_equal(whole, chunks)
        np.testing.assert_array_equal(whole, expected)

    def test_legacy_algorithmic_profile_is_never_applied(self) -> None:
        legacy = {
            "schema_version": 1,
            "mode": "algorithmic_quantile_magnitude_channel",
            "sample_rate": self.sample_rate,
            "n_fft": 512,
            "hop": 256,
            "bands": 3,
            "band_centers_hz": [200.0, 1_000.0, 4_000.0],
            "band_gains_db": [-30.0, 12.0, 12.0],
            "summary": {"usable": True},
        }
        profile = ReferenceAdapterProfile.from_json(legacy)
        self.assertFalse(profile.valid)
        np.testing.assert_array_equal(
            apply_profile(self.reference, profile),
            self.reference,
        )

    def test_full_file_application_copies_raw_when_gain_would_clip(self) -> None:
        profile = self.fit_plus_six().to_json()
        self.assertTrue(profile["summary"]["usable"])
        tone = (
            np.sin(
                np.linspace(
                    0.0,
                    200.0 * np.pi,
                    self.sample_rate,
                    dtype=np.float32,
                )
            )
            * np.float32(0.75)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.flac"
            output = root / "output.flac"
            sf.write(source, tone, self.sample_rate, format="FLAC", subtype="PCM_16")
            report = adapt_reference_file_with_profile(
                source,
                output,
                profile,
                block_sec=0.1,
            )
            self.assertFalse(report["applied"])
            self.assertEqual(report["application_reason"], "would_clip")
            self.assertEqual(source.read_bytes(), output.read_bytes())

    def test_full_file_scalar_result_does_not_depend_on_block_size(self) -> None:
        profile = self.fit_plus_six().to_json()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.flac"
            short_blocks = root / "short.flac"
            long_blocks = root / "long.flac"
            sf.write(
                source,
                self.reference,
                self.sample_rate,
                format="FLAC",
                subtype="PCM_16",
            )
            short_report = adapt_reference_file_with_profile(
                source,
                short_blocks,
                profile,
                block_sec=0.1,
            )
            long_report = adapt_reference_file_with_profile(
                source,
                long_blocks,
                profile,
                block_sec=30.0,
            )
            self.assertTrue(short_report["applied"])
            self.assertTrue(long_report["applied"])
            short, _ = sf.read(short_blocks, dtype="float32")
            long, _ = sf.read(long_blocks, dtype="float32")
            np.testing.assert_array_equal(short, long)


if __name__ == "__main__":
    unittest.main()
