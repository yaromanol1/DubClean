from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.application_pipeline import (
    _route_semantic_model_windows,
)


class ApplicationRoutingSampleRateTests(unittest.TestCase):
    def test_primary_model_is_resampled_by_timeline_before_blending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary_path = root / "primary_24k.flac"
            mixture_path = root / "mixture_48k.flac"
            output_path = root / "routed_48k.flac"

            duration_sec = 2.0
            primary_rate = 24_000
            output_rate = 48_000
            time = np.arange(
                int(primary_rate * duration_sec), dtype=np.float32
            ) / primary_rate
            primary = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)
            mixture = np.zeros(
                (int(output_rate * duration_sec), 1), dtype=np.float32
            )
            sf.write(primary_path, primary, primary_rate, subtype="PCM_16")
            sf.write(mixture_path, mixture, output_rate, subtype="PCM_16")

            _route_semantic_model_windows(
                primary_path,
                mixture_path,
                {
                    "windows": [
                        {
                            "dubbed_start_sec": 0.0,
                            "dubbed_end_sec": duration_sec,
                            "recommended_model": "dubclean_voice",
                        }
                    ]
                },
                output_path,
            )

            routed, routed_rate = sf.read(
                output_path, dtype="float32", always_2d=True
            )
            self.assertEqual(routed_rate, output_rate)
            self.assertEqual(len(routed), len(mixture))

            stable = routed[
                int(0.25 * output_rate) : int(1.75 * output_rate), 0
            ]
            spectrum = np.abs(np.fft.rfft(stable))
            frequencies = np.fft.rfftfreq(len(stable), 1.0 / output_rate)
            dominant = float(frequencies[int(np.argmax(spectrum))])
            self.assertAlmostEqual(dominant, 440.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
