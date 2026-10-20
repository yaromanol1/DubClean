from __future__ import annotations

import unittest

import numpy as np

from experiments.paired_reference_cancel import alignment


def _noise_with_envelope(rng: np.random.Generator, sr: int, seconds: float) -> np.ndarray:
    """Broadband noise with a slow loudness contour so that all three delay
    cues (PHAT, plain xcorr, energy envelope) have structure to lock onto."""
    n = int(sr * seconds)
    base = rng.normal(size=n).astype(np.float64)
    t = np.arange(n) / sr
    envelope = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.37 * t)) ** 2
    burst = (np.sin(2.0 * np.pi * 0.11 * t) > 0.55).astype(np.float64)
    return (base * envelope * (0.35 + burst)).astype(np.float32)


class GlobalOffsetTests(unittest.TestCase):
    def test_delay_convention_dubbed_is_silence_plus_original(self) -> None:
        # Module contract: dubbed = K samples of silence + original recovers
        # delay == K (original_index = dubbed_index - delay).
        sr = 8_000
        rng = np.random.default_rng(7)
        original = _noise_with_envelope(rng, sr, 20.0)
        delay_sec = 1.25
        delay_samples = int(round(delay_sec * sr))
        dubbed = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), original]
        )

        result = alignment.estimate_global_offset(original, dubbed, sr, 3.0)

        self.assertAlmostEqual(result.offset_sec, delay_sec, delta=0.05)
        self.assertGreater(result.confidence, 0.5)

    def test_combine_prefers_agreeing_pair_over_high_scoring_outlier(self) -> None:
        # GCC-PHAT is the outlier with the best score; the other two methods
        # agree with each other.  The agreeing pair must win with the full
        # agreement bonus — not the single loudest method with a penalty.
        offsets = {"gcc_phat": 5.0, "normalized_xcorr": 1.00, "energy_envelope": 1.10}
        scores = {"gcc_phat": 0.90, "normalized_xcorr": 0.50, "energy_envelope": 0.40}

        chosen, spread, bonus = alignment._combine_method_offsets(offsets, scores)

        self.assertAlmostEqual(chosen, 1.05, places=6)
        self.assertAlmostEqual(spread, 0.10, places=6)
        self.assertEqual(bonus, 1.0)

    def test_combine_falls_back_to_best_score_without_any_agreement(self) -> None:
        offsets = {"gcc_phat": 0.0, "normalized_xcorr": 1.0, "energy_envelope": 2.0}
        scores = {"gcc_phat": 0.20, "normalized_xcorr": 0.70, "energy_envelope": 0.10}

        chosen, spread, bonus = alignment._combine_method_offsets(offsets, scores)

        self.assertEqual(chosen, 1.0)
        self.assertAlmostEqual(spread, 2.0, places=6)
        self.assertEqual(bonus, 0.4)

    def test_combine_keeps_previous_behaviour_when_all_methods_agree(self) -> None:
        offsets = {"gcc_phat": 0.50, "normalized_xcorr": 0.51, "energy_envelope": 0.49}
        scores = {"gcc_phat": 0.9, "normalized_xcorr": 0.8, "energy_envelope": 0.6}

        chosen, _spread, bonus = alignment._combine_method_offsets(offsets, scores)

        self.assertAlmostEqual(chosen, 0.50, places=6)
        self.assertEqual(bonus, 1.0)


class ChunkedAlignmentTests(unittest.TestCase):
    def test_chunk_offsets_and_warp_recover_constant_delay(self) -> None:
        sr = 8_000
        rng = np.random.default_rng(21)
        original = _noise_with_envelope(rng, sr, 24.0)
        delay_sec = 0.5
        delay_samples = int(round(delay_sec * sr))
        dubbed = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), original]
        )

        global_alignment = alignment.estimate_global_offset(original, dubbed, sr, 2.0)
        chunks, speed_ratio = alignment.build_alignment_map(
            original,
            dubbed,
            sr,
            global_alignment,
            chunk_seconds=4.0,
            search_radius_sec=1.0,
            min_confidence=0.2,
            discontinuity_threshold_ms=80.0,
        )

        usable = [chunk for chunk in chunks if chunk["usable"]]
        self.assertGreaterEqual(len(usable), 4)
        for chunk in usable:
            self.assertAlmostEqual(chunk["offset"], delay_sec, delta=0.02)
        self.assertAlmostEqual(speed_ratio, 1.0, delta=0.01)

        aligned, confidence = alignment.warp_original_to_dubbed_timeline(
            original,
            chunks,
            sr,
            len(dubbed),
            global_alignment.offset_sec,
        )
        interior = slice(sr, len(dubbed) - sr)
        self.assertLess(
            float(np.max(np.abs(aligned[interior, 0] - dubbed[interior]))), 1e-3
        )
        self.assertGreater(float(np.min(confidence[interior])), 0.2)

    def test_warp_marks_unusable_regions_with_zero_confidence(self) -> None:
        sr = 8_000
        rng = np.random.default_rng(5)
        original = _noise_with_envelope(rng, sr, 4.0)
        total = len(original)
        chunks = [
            {
                "dubbed_start": 0.0,
                "duration": 2.0,
                "original_start": 0.0,
                "speed_ratio": 1.0,
                "confidence": 0.9,
                "usable": True,
            },
            {
                "dubbed_start": 2.0,
                "duration": 2.0,
                "original_start": 2.0,
                "speed_ratio": 1.0,
                "confidence": 0.0,
                "usable": False,
            },
        ]

        aligned, confidence = alignment.warp_original_to_dubbed_timeline(
            original, chunks, sr, total, 0.0
        )

        self.assertEqual(aligned.shape[0], total)
        margin = int(0.2 * sr)
        self.assertGreater(float(np.min(confidence[margin : 2 * sr - margin])), 0.5)
        # The adaptive canceller must never subtract aggressively where the
        # map is unusable: the confidence gate has to be exactly zero there.
        self.assertEqual(float(np.max(confidence[2 * sr + margin :])), 0.0)


if __name__ == "__main__":
    unittest.main()
