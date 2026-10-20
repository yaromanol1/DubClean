from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import application_pipeline, pipeline


class _Context:
    def update(self, **_values) -> None:
        return None


class GlobalAffineCollapseTests(unittest.TestCase):
    """Accepting a map must not erase the offsets it was built to carry."""

    @staticmethod
    def _store() -> SimpleNamespace:
        return SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                    "global_affine_max_residual_sec": 0.05,
                }
            }
        )

    @staticmethod
    def _reel_map(method: str) -> dict:
        # Three reels 100 ms apart: a line through their control points fits
        # within a fifth of a second, which is what the old bound allowed.
        delays = [(0.0, 1800.0, -0.005), (1800.0, 3000.0, 0.085), (3000.0, 5400.0, -0.104)]
        segments = []
        controls = []
        for index, (start, end, delay) in enumerate(delays):
            segments.append(
                {
                    "id": index,
                    "dubbed_start": start,
                    "dubbed_end": end,
                    "original_start": start - delay,
                    "original_end": end - delay,
                    "duration": end - start,
                    "delay": delay,
                    "offset": delay,
                    "speed_ratio": 1.0,
                    "confidence": 0.35,
                    "usable": True,
                    "reason": None,
                }
            )
            controls.append(
                {
                    "dubbed_start": start,
                    "original_start": start - delay,
                    "confidence": 0.35,
                    "usable": True,
                    "speed_ratio": 1.0,
                }
            )
        controls.append(
            {
                "dubbed_start": 5400.0,
                "original_start": 5400.104,
                "confidence": 0.35,
                "usable": True,
                "speed_ratio": 1.0,
            }
        )
        return {
            "method": method,
            "summary": {
                "dubbed_duration_sec": 5400.0,
                "original_duration_sec": 5400.0,
            },
            "segments": segments,
            "chunks": segments,
            "control_points": controls,
        }

    def test_measured_reels_survive_acceptance(self) -> None:
        alignment = self._reel_map("same_container_measured_offsets_v1")

        result = pipeline._processing_alignment_map(
            self._store(), {"alignment_review": {"accepted": True}}, alignment
        )

        self.assertNotIn("processing_mode", result["summary"])
        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual(sum(item["usable"] for item in result["segments"]), 3)
        self.assertAlmostEqual(result["segments"][-1]["delay"], -0.104, places=6)

    def test_a_hundred_millisecond_scatter_is_not_one_global_mapping(self) -> None:
        # The same shape from any other method: the scatter is far too large to
        # call the pair a single affine mapping, whatever produced it.
        alignment = self._reel_map("first_match_sparse_verification_v3")

        result = pipeline._processing_alignment_map(
            self._store(), {"alignment_review": {"accepted": True}}, alignment
        )

        self.assertNotIn("processing_mode", result["summary"])
        self.assertEqual(len(result["segments"]), 3)

    def test_a_genuinely_uniform_pair_is_still_collapsed(self) -> None:
        alignment = self._reel_map("first_match_sparse_verification_v3")
        for segment in alignment["segments"]:
            segment["original_start"] = segment["dubbed_start"] - 0.01
            segment["original_end"] = segment["dubbed_end"] - 0.01
        for control in alignment["control_points"]:
            control["original_start"] = control["dubbed_start"] - 0.01

        result = pipeline._processing_alignment_map(
            self._store(), {"alignment_review": {"accepted": True}}, alignment
        )

        self.assertEqual(result["summary"]["processing_mode"], "global_affine")


class AlignmentProcessingSafetyTests(unittest.TestCase):
    def test_manual_acceptance_never_promotes_an_unsafe_speed_ratio(self) -> None:
        store = SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                }
            }
        )
        pair = {"alignment_review": {"accepted": True}}
        alignment = {
            "summary": {},
            "segments": [
                {
                    "id": 1,
                    "dubbed_start": 0.0,
                    "dubbed_end": 30.0,
                    "original_start": 0.0,
                    "original_end": 30.6,
                    "speed_ratio": 1.02,
                    "confidence": 0.08,
                    "usable": False,
                    "reason": "низкая уверенность контрольной точки",
                },
                {
                    "id": 2,
                    "dubbed_start": 30.0,
                    "dubbed_end": 60.0,
                    "original_start": 30.6,
                    "original_end": 45.2469837,
                    "speed_ratio": 0.48823279,
                    "confidence": 0.08,
                    "usable": False,
                    "reason": "низкая уверенность контрольной точки",
                },
            ],
        }

        result = pipeline._processing_alignment_map(store, pair, alignment)

        self.assertTrue(result["segments"][0]["usable"])
        self.assertFalse(result["segments"][1]["usable"])
        self.assertEqual(
            result["segments"][1]["processing_override_rejected"],
            "unsafe_speed_ratio",
        )
        self.assertEqual(
            result["summary"]["approved_unsafe_speed_rejections"], 1
        )

    def test_preview_selection_rejects_an_explicitly_unusable_segment(self) -> None:
        alignment = {
            "segments": [
                {
                    "id": 12,
                    "dubbed_start": 4100.0,
                    "dubbed_end": 4507.0,
                    "original_start": 4308.0,
                    "speed_ratio": 0.48823279,
                    "confidence": 0.08,
                    "usable": False,
                }
            ]
        }
        selection = {
            "zone": "75_percent",
            "preview_start_sec": 4299.0,
            "preview_end_sec": 4329.0,
            "preview_duration_sec": 30.0,
        }

        self.assertIsNone(
            application_pipeline._candidate_from_selection(alignment, selection)
        )

    def test_stale_usable_segment_with_unsafe_speed_is_demoted(self) -> None:
        store = SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                }
            }
        )
        pair = {"alignment_review": {"accepted": True}}
        alignment = {
            "summary": {},
            "segments": [
                {
                    "id": 12,
                    "dubbed_start": 0.0,
                    "dubbed_end": 30.0,
                    "original_start": 0.0,
                    "original_end": 14.6469837,
                    "speed_ratio": 0.48823279,
                    "confidence": 0.8,
                    "usable": True,
                }
            ],
        }

        result = pipeline._processing_alignment_map(store, pair, alignment)

        self.assertFalse(result["segments"][0]["usable"])
        self.assertEqual(
            result["segments"][0]["processing_override_rejected"],
            "unsafe_speed_ratio",
        )
        self.assertEqual(
            result["summary"]["approved_unsafe_speed_rejections"], 1
        )

    def test_preview_rejects_stale_usable_segment_with_unsafe_speed(self) -> None:
        alignment = {
            "summary": {"processing_max_speed_deviation": 0.06},
            "segments": [
                {
                    "id": 12,
                    "dubbed_start": 4100.0,
                    "dubbed_end": 4507.0,
                    "original_start": 4308.0,
                    "speed_ratio": 0.48823279,
                    "confidence": 0.8,
                    # A stale map may incorrectly retain this flag.
                    "usable": True,
                }
            ],
        }
        selection = {
            "zone": "75_percent",
            "preview_start_sec": 4299.0,
            "preview_end_sec": 4329.0,
            "preview_duration_sec": 30.0,
        }

        self.assertIsNone(
            application_pipeline._candidate_from_selection(alignment, selection)
        )

    def test_complementary_speed_spikes_are_bridged_between_safe_anchors(self) -> None:
        store = SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                }
            }
        )
        pair = {"alignment_review": {"accepted": True}}
        alignment = {
            "summary": {},
            "segments": [
                {
                    "id": 10,
                    "dubbed_start": 0.0,
                    "dubbed_end": 30.0,
                    "original_start": 0.0,
                    "original_end": 30.0,
                    "speed_ratio": 1.0,
                    "confidence": 0.8,
                    "usable": True,
                },
                {
                    "id": 11,
                    "dubbed_start": 30.0,
                    "dubbed_end": 60.0,
                    "original_start": 30.0,
                    "original_end": 75.0,
                    "speed_ratio": 1.5,
                    "confidence": 0.08,
                    "usable": False,
                    "reason": "низкая уверенность контрольной точки",
                },
                {
                    "id": 12,
                    "dubbed_start": 60.0,
                    "dubbed_end": 90.0,
                    "original_start": 75.0,
                    "original_end": 90.0,
                    "speed_ratio": 0.5,
                    "confidence": 0.08,
                    "usable": False,
                    "reason": "низкая уверенность контрольной точки",
                },
                {
                    "id": 13,
                    "dubbed_start": 90.0,
                    "dubbed_end": 120.0,
                    "original_start": 90.0,
                    "original_end": 120.0,
                    "speed_ratio": 1.0,
                    "confidence": 0.7,
                    "usable": True,
                },
            ],
        }

        result = pipeline._processing_alignment_map(store, pair, alignment)
        bridges = [
            item
            for item in result["segments"]
            if item.get("status") == "approved_safe_bridge"
        ]

        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0]["dubbed_start"], 30.0)
        self.assertEqual(bridges[0]["dubbed_end"], 90.0)
        self.assertEqual(bridges[0]["original_start"], 30.0)
        self.assertEqual(bridges[0]["original_end"], 90.0)
        self.assertEqual(bridges[0]["speed_ratio"], 1.0)
        self.assertTrue(bridges[0]["usable"])
        self.assertEqual(result["summary"]["approved_safe_speed_bridges"], 1)

    def test_three_unsafe_segments_are_never_guessed_through(self) -> None:
        store = SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                }
            }
        )
        pair = {"alignment_review": {"accepted": True}}
        safe_left = {
            "id": 1,
            "dubbed_start": 0.0,
            "dubbed_end": 30.0,
            "original_start": 0.0,
            "original_end": 30.0,
            "speed_ratio": 1.0,
            "confidence": 0.8,
            "usable": True,
        }
        unsafe = []
        for index, (left, right, speed) in enumerate(
            ((30.0, 75.0, 1.5), (75.0, 90.0, 0.5), (90.0, 120.0, 1.5)),
            2,
        ):
            dubbed_start = 30.0 * (index - 1)
            unsafe.append(
                {
                    "id": index,
                    "dubbed_start": dubbed_start,
                    "dubbed_end": dubbed_start + 30.0,
                    "original_start": left,
                    "original_end": right,
                    "speed_ratio": speed,
                    "confidence": 0.08,
                    "usable": False,
                    "reason": "низкая уверенность контрольной точки",
                }
            )
        safe_right = {
            "id": 5,
            "dubbed_start": 120.0,
            "dubbed_end": 150.0,
            "original_start": 120.0,
            "original_end": 150.0,
            "speed_ratio": 1.0,
            "confidence": 0.8,
            "usable": True,
        }

        result = pipeline._processing_alignment_map(
            store,
            pair,
            {"summary": {}, "segments": [safe_left, *unsafe, safe_right]},
        )

        self.assertEqual(result["summary"]["approved_safe_speed_bridges"], 0)
        self.assertFalse(any(item.get("status") == "approved_safe_bridge" for item in result["segments"]))
        self.assertTrue(all(not item["usable"] for item in result["segments"][1:4]))

    def test_unsafe_pair_is_not_bridged_when_raw_geometry_is_inconsistent(self) -> None:
        store = SimpleNamespace(
            cfg={
                "alignment": {
                    "approved_map_min_measured_confidence": 0.05,
                    "approved_map_processing_confidence": 0.45,
                    "max_speed_deviation": 0.06,
                }
            }
        )
        pair = {"alignment_review": {"accepted": True}}
        base_segments = [
            {
                "id": 10,
                "dubbed_start": 0.0,
                "dubbed_end": 30.0,
                "original_start": 0.0,
                "original_end": 30.0,
                "speed_ratio": 1.0,
                "confidence": 0.8,
                "usable": True,
            },
            {
                "id": 11,
                "dubbed_start": 30.0,
                "dubbed_end": 60.0,
                "original_start": 30.0,
                "original_end": 75.0,
                "speed_ratio": 1.5,
                "confidence": 0.08,
                "usable": False,
                "reason": "низкая уверенность контрольной точки",
            },
            {
                "id": 12,
                "dubbed_start": 60.0,
                "dubbed_end": 90.0,
                "original_start": 75.0,
                "original_end": 90.0,
                "speed_ratio": 0.5,
                "confidence": 0.08,
                "usable": False,
                "reason": "низкая уверенность контрольной точки",
            },
            {
                "id": 13,
                "dubbed_start": 90.0,
                "dubbed_end": 120.0,
                "original_start": 90.0,
                "original_end": 120.0,
                "speed_ratio": 1.0,
                "confidence": 0.8,
                "usable": True,
            },
        ]
        variants = {}
        internal_gap = copy.deepcopy(base_segments)
        internal_gap[2]["original_start"] = 80.0
        variants["internal_original_gap"] = internal_gap
        same_direction = copy.deepcopy(base_segments)
        same_direction[2]["speed_ratio"] = 1.5
        variants["same_direction_spikes"] = same_direction
        shifted_outer_boundary = copy.deepcopy(base_segments)
        shifted_outer_boundary[1]["original_start"] = 50.0
        variants["shifted_outer_boundary"] = shifted_outer_boundary

        for name, segments in variants.items():
            with self.subTest(case=name):
                result = pipeline._processing_alignment_map(
                    store, pair, {"summary": {}, "segments": segments}
                )
                self.assertEqual(
                    result["summary"]["approved_safe_speed_bridges"], 0
                )
                self.assertFalse(
                    any(
                        item.get("status") == "approved_safe_bridge"
                        for item in result["segments"]
                    )
                )

    def test_full_alignment_defensively_refuses_extreme_time_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            destination = root / "aligned.flac"
            rate = 8_000
            t = np.arange(rate, dtype=np.float32) / rate
            sf.write(source, np.sin(2.0 * np.pi * 440.0 * t), rate)
            for unsafe_speed in (0.48823279, 0.0, float("inf"), "bad"):
                with self.subTest(speed_ratio=unsafe_speed):
                    alignment = {
                        "summary": {"processing_max_speed_deviation": 0.06},
                        "segments": [
                            {
                                "dubbed_start": 0.0,
                                "dubbed_end": 1.0,
                                "original_start": 0.0,
                                "speed_ratio": unsafe_speed,
                                # Defence in depth: even a stale map that still
                                # says usable must not stretch the programme.
                                "usable": True,
                            }
                        ],
                    }

                    application_pipeline._align_to_dubbed(
                        source,
                        destination,
                        alignment,
                        rate,
                        1.0,
                        1,
                        _Context(),
                        "fixture",
                        "fixture",
                    )

                    output, output_rate = sf.read(
                        destination, dtype="float32", always_2d=True
                    )
                    self.assertEqual(output_rate, rate)
                    self.assertEqual(len(output), rate)
                    self.assertEqual(float(np.max(np.abs(output))), 0.0)


if __name__ == "__main__":
    unittest.main()
