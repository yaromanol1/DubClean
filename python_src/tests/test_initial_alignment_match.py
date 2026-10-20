from __future__ import annotations

import unittest

from experiments.paired_reference_cancel.pipeline import (
    _initial_match_acceptance,
    _verification_point_acceptance,
)


class InitialAlignmentMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "min_confidence": 0.30,
            "initial_min_confidence": 0.60,
            "initial_distinctive_min_confidence": 0.55,
            "initial_distinctive_min_margin": 0.20,
        }

    def test_normal_match_uses_strict_acceptance(self) -> None:
        result = _initial_match_acceptance(
            {"confidence": 0.72, "second_score": 0.50},
            self.config,
        )

        self.assertEqual(result, "strict")

    def test_real_borderline_but_unique_match_is_accepted(self) -> None:
        result = _initial_match_acceptance(
            {
                "confidence": 0.5887669421380799,
                "second_score": 0.06661965908127628,
            },
            self.config,
        )

        self.assertEqual(result, "distinctive")

    def test_borderline_ambiguous_match_is_rejected(self) -> None:
        result = _initial_match_acceptance(
            {"confidence": 0.5888, "second_score": 0.47},
            self.config,
        )

        self.assertIsNone(result)

    def test_weak_match_is_rejected_even_when_unique(self) -> None:
        result = _initial_match_acceptance(
            {"confidence": 0.54, "second_score": 0.01},
            self.config,
        )

        self.assertIsNone(result)

    def test_verification_point_uses_unique_prediction_consensus(self) -> None:
        config = {
            "min_confidence": 0.30,
            "verification_distinctive_min_confidence": 0.20,
            "verification_distinctive_min_margin": 0.12,
            "verification_prediction_tolerance_sec": 2.0,
        }

        self.assertEqual(
            _verification_point_acceptance(
                3420.81,
                3420.80,
                0.22,
                0.06,
                config,
            ),
            "distinctive_prediction",
        )
        self.assertIsNone(
            _verification_point_acceptance(
                3425.81,
                3420.80,
                0.22,
                0.06,
                config,
            )
        )
        self.assertIsNone(
            _verification_point_acceptance(
                3420.81,
                3420.80,
                0.22,
                0.15,
                config,
            )
        )


if __name__ == "__main__":
    unittest.main()
