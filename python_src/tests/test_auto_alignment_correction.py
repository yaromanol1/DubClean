from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import pipeline


class AutoAlignmentCorrectionTests(unittest.TestCase):
    def test_uses_scenes_across_unequal_length_tracks_and_rejects_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample_rate = 100
            original = root / "original.wav"
            dubbed = root / "dubbed.wav"
            sf.write(original, np.zeros(sample_rate * 140, dtype=np.float32), sample_rate)
            sf.write(dubbed, np.zeros(sample_rate * 120, dtype=np.float32), sample_rate)
            alignment_map = {
                "global": {"offset_sec": 0.0, "speed_ratio": 1.0},
                "segments": [
                    {
                        "dubbed_start": 0.0,
                        "dubbed_end": 110.0,
                        "original_start": 0.0,
                        "original_end": 110.0,
                        "speed_ratio": 1.0,
                        "usable": True,
                    }
                ],
            }
            calls: list[float] = []

            def fake_verifier(_original, _dubbed, dubbed_start, expected, *_args):
                calls.append(float(dubbed_start))
                residual = -12.0 if len(calls) == 4 else 2.0
                return {
                    "original_start": expected + residual,
                    "confidence": 0.9,
                    "usable": True,
                }

            config = {
                "verification_window_sec": 10.0,
                "auto_correction_points": 9,
                "auto_correction_min_points": 3,
                "min_confidence": 0.3,
            }
            with patch.object(pipeline, "_verify_sparse_point", side_effect=fake_verifier):
                result = pipeline.estimate_sparse_manual_correction(
                    original, dubbed, alignment_map, config
                )

            self.assertAlmostEqual(result["correction_sec"], -2.0, places=3)
            self.assertGreaterEqual(result["accepted_points"], 7)
            self.assertGreater(max(calls) - min(calls), 70.0)
            self.assertEqual(result["method"], "sparse_multi_scene_residual_v1")


if __name__ == "__main__":
    unittest.main()
