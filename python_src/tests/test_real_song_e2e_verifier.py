from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import audio_io


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = REPOSITORY_ROOT / "portable_tools" / "verify_real_song_e2e.py"
SPEC = importlib.util.spec_from_file_location("verify_real_song_e2e", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class RealSongE2EVerifierTests(unittest.TestCase):
    def test_expected_stereo_is_adapted_like_product_for_six_channel_result(
        self,
    ) -> None:
        rate = 8_000
        frames = rate * 2
        time = np.arange(frames, dtype=np.float32) / rate
        stereo = np.column_stack(
            (
                0.2 * np.sin(2.0 * np.pi * 220.0 * time),
                0.1 * np.sin(2.0 * np.pi * 330.0 * time),
            )
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_path = root / "aligned-original-stereo.flac"
            actual_path = root / "protected-result-5.1.flac"
            sf.write(expected_path, stereo, rate, subtype="PCM_24")
            expected_decoded, _ = sf.read(
                expected_path, dtype="float32", always_2d=True
            )
            six_channel = audio_io.match_channels(expected_decoded, 6)
            sf.write(actual_path, six_channel, rate, subtype="PCM_24")

            result = VERIFIER.compare_interval(
                actual_path,
                expected_path,
                0.25,
                1.75,
                label="protected song core",
            )

        self.assertTrue(result["pass"], result)
        self.assertEqual(
            result["expected_channel_adaptation"],
            {
                "source_channels": 2,
                "target_channels": 6,
                "method": "audio_io.match_channels",
            },
        )
        self.assertLessEqual(
            result["maximum_absolute_error"], result["allowed_error"]
        )

    def test_sample_rate_mismatch_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actual_path = root / "actual.flac"
            expected_path = root / "expected.flac"
            sf.write(actual_path, np.zeros((8_000, 2), np.float32), 8_000)
            sf.write(expected_path, np.zeros((16_000, 2), np.float32), 16_000)

            result = VERIFIER.compare_interval(
                actual_path,
                expected_path,
                0.0,
                1.0,
                label="different rates",
            )

        self.assertFalse(result["pass"])
        self.assertEqual(result["reason"], "sample rate differs")


if __name__ == "__main__":
    unittest.main()
