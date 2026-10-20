from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import pipeline
from experiments.paired_reference_cancel.alignment import (
    plateau_segments_from_offsets,
)

RATE = 8_000


class _Context:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def check_stop(self) -> None:
        return None

    def update(self, **_values) -> None:
        return None

    def log(self, message: str) -> None:
        self.messages.append(message)


class _Store:
    def __init__(self, root: Path, source: str) -> None:
        self.root = root
        self.cfg = {"alignment": {"max_speed_deviation": 0.06}}
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "name": "Тест",
            "stages": {"2": {"status": "completed"}},
            "sources": {
                "original": {"path": source},
                "dubbed": {"path": source},
            },
        }

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair


def _programme(seconds: float, seed: int) -> np.ndarray:
    """Broadband material with a slow loudness contour and sharp events.

    Both the correlation cues and the loudness-contour cue need structure to
    lock onto, and the contour is what carries the verification score.
    """
    rng = np.random.default_rng(seed)
    count = int(RATE * seconds)
    time = np.arange(count) / RATE
    base = rng.normal(size=count)
    envelope = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.07 * time)) ** 2
    hits = np.zeros(count)
    for position in range(2, int(seconds) - 2, 3):
        start = int(position * RATE)
        hits[start : start + RATE // 4] = np.linspace(1.0, 0.0, RATE // 4) ** 2
    return ((base * envelope) + 3.0 * hits * rng.normal(size=count)).astype(np.float32)


def _remastered(values: np.ndarray, seed: int) -> np.ndarray:
    """The same programme as another release would carry it."""
    rng = np.random.default_rng(seed)
    return (
        0.68 * values + 0.04 * rng.normal(size=len(values)).astype(np.float32)
    ).astype(np.float32)


def _delayed(values: np.ndarray, delay_sec: float) -> np.ndarray:
    """Return ``values`` shifted so that original_index = dubbed_index - delay."""
    shift = int(round(delay_sec * RATE))
    out = np.zeros_like(values)
    if shift > 0:
        out[shift:] = values[:-shift]
    elif shift < 0:
        out[:shift] = values[-shift:]
    else:
        out = values.copy()
    return out


def _write_pair(root: Path, original: np.ndarray, dubbed: np.ndarray) -> None:
    extracted = root / "pair" / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    (root / "pair" / "alignment").mkdir(parents=True, exist_ok=True)
    sf.write(extracted / "original_proxy.wav", original, RATE, subtype="FLOAT")
    sf.write(extracted / "dubbed_proxy.wav", dubbed, RATE, subtype="FLOAT")


class PlateauSegmentationTests(unittest.TestCase):
    @staticmethod
    def _entries(delays: list[float], duration: float = 30.0) -> list[dict]:
        return [
            {
                "dubbed_start": index * duration,
                "duration": duration,
                "delay": delay,
                "confidence": 0.4,
                "usable": True,
            }
            for index, delay in enumerate(delays)
        ]

    def test_flat_profile_stays_one_segment(self) -> None:
        segments = plateau_segments_from_offsets(
            self._entries([0.015] * 20),
            tolerance_sec=0.025,
            min_segment_sec=90.0,
            min_windows=3,
        )

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["delay"], 0.015, places=6)

    def test_step_becomes_two_segments_split_between_them(self) -> None:
        segments = plateau_segments_from_offsets(
            self._entries([0.015] * 10 + [-0.104] * 10),
            tolerance_sec=0.025,
            min_segment_sec=90.0,
            min_windows=3,
        )

        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0]["delay"], 0.015, places=6)
        self.assertAlmostEqual(segments[1]["delay"], -0.104, places=6)
        self.assertAlmostEqual(segments[0]["dubbed_end"], 300.0, places=3)
        self.assertAlmostEqual(segments[1]["dubbed_start"], 300.0, places=3)

    def test_single_outlier_is_noise_not_a_splice(self) -> None:
        segments = plateau_segments_from_offsets(
            self._entries([0.015] * 8 + [0.4] + [0.015] * 8),
            tolerance_sec=0.025,
            min_segment_sec=90.0,
            min_windows=3,
        )

        self.assertEqual(len(segments), 1)

    def test_run_too_short_to_be_a_reel_merges_into_its_neighbour(self) -> None:
        segments = plateau_segments_from_offsets(
            self._entries([0.0] * 8 + [0.06] * 8 + [0.3] * 2),
            tolerance_sec=0.025,
            min_segment_sec=90.0,
            min_windows=3,
        )

        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[-1]["delay"], 0.06, places=6)


class SameContainerMeasurementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = dict(pipeline.DEFAULT_SAME_CONTAINER_VERIFICATION)

    def test_measured_delay_follows_the_module_sign_convention(self) -> None:
        # original_index = dubbed_index - delay, so a dubbed track that lags the
        # original by 80 ms must be reported as delay = +0.08.
        original = _programme(420.0, seed=11)
        dubbed = _remastered(_delayed(original, 0.08), seed=12)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_pair(root, original, dubbed)
            entries = pipeline.measure_same_container_offsets(
                root / "pair" / "extracted" / "original_proxy.wav",
                root / "pair" / "extracted" / "dubbed_proxy.wav",
                self.settings,
                None,
                duration_sec=420.0,
            )

        measured = [item["delay"] for item in entries if item["usable"]]
        self.assertGreaterEqual(len(measured), 8)
        self.assertAlmostEqual(float(np.median(measured)), 0.08, delta=0.006)

    def test_a_splice_produces_two_mapped_reels(self) -> None:
        original = _programme(600.0, seed=21)
        half = int(300.0 * RATE)
        dubbed = _remastered(
            np.concatenate(
                [
                    _delayed(original, 0.02)[:half],
                    _delayed(original, -0.1)[half:],
                ]
            ),
            seed=22,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_pair(root, original, dubbed)
            original_proxy = root / "pair" / "extracted" / "original_proxy.wav"
            dubbed_proxy = root / "pair" / "extracted" / "dubbed_proxy.wav"
            entries = pipeline.measure_same_container_offsets(
                original_proxy, dubbed_proxy, self.settings, None, duration_sec=600.0
            )
            verification = pipeline.summarize_same_container_offsets(
                entries, self.settings, 600.0
            )
            scores = pipeline.score_offset_policies(
                original_proxy,
                dubbed_proxy,
                verification,
                self.settings,
                None,
                duration_sec=600.0,
            )
            policy = pipeline.choose_same_container_policy(
                verification, scores, self.settings
            )
            segments, checkpoints = pipeline.same_container_segments(
                verification, 600.0, 600.0
            )

        self.assertEqual(policy, "piecewise")
        self.assertEqual(len(verification["plateaus"]), 2)
        usable = [item for item in segments if item["usable"]]
        self.assertEqual(len(usable), 2)
        self.assertAlmostEqual(usable[0]["delay"], 0.02, delta=0.006)
        self.assertAlmostEqual(usable[1]["delay"], -0.1, delta=0.006)
        # A segment maps the dubbed clock onto the original one.
        for item in usable:
            self.assertAlmostEqual(
                item["original_start"],
                item["dubbed_start"] - item["delay"],
                places=5,
            )
        self.assertEqual(len(checkpoints), len(usable) + 1)
        self.assertGreater(
            scores["piecewise_agreement"], scores["global_agreement"]
        )

    def test_truly_identical_tracks_need_no_correction(self) -> None:
        original = _programme(420.0, seed=31)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_pair(root, original, _remastered(original, seed=32))
            original_proxy = root / "pair" / "extracted" / "original_proxy.wav"
            dubbed_proxy = root / "pair" / "extracted" / "dubbed_proxy.wav"
            entries = pipeline.measure_same_container_offsets(
                original_proxy, dubbed_proxy, self.settings, None, duration_sec=420.0
            )
            verification = pipeline.summarize_same_container_offsets(
                entries, self.settings, 420.0
            )
            scores = pipeline.score_offset_policies(
                original_proxy,
                dubbed_proxy,
                verification,
                self.settings,
                None,
                duration_sec=420.0,
            )
            policy = pipeline.choose_same_container_policy(
                verification, scores, self.settings
            )

        self.assertEqual(policy, "identity")


class AlignPairSameContainerTests(unittest.TestCase):
    def test_constant_offset_inside_one_file_is_measured_not_assumed(self) -> None:
        original = _programme(420.0, seed=41)
        dubbed = _remastered(_delayed(original, 0.034), seed=42)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_pair(root, original, dubbed)
            store = _Store(root, str(root / "movie.mkv"))
            context = _Context()

            result = pipeline.align_pair(store, "project", "pair", context)

            written = root / "pair" / "alignment" / "alignment_map.json"
            self.assertTrue(written.is_file())
            import json

            alignment_map = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(alignment_map["method"], "same_container_measured_offsets_v1")
        usable = [item for item in alignment_map["segments"] if item["usable"]]
        self.assertEqual(len(usable), 1)
        self.assertAlmostEqual(usable[0]["delay"], 0.034, delta=0.006)
        self.assertAlmostEqual(
            usable[0]["original_start"], -0.034, delta=0.006
        )
        self.assertLess(alignment_map["summary"]["matched_ratio"], 1.0000001)
        self.assertTrue(
            any(
                warning.startswith("Дубляж смещён относительно оригинала:")
                for warning in store.pair["warnings"]
            )
        )
        self.assertEqual(result["summary"], alignment_map["summary"])

    def test_identical_tracks_keep_the_one_to_one_map(self) -> None:
        original = _programme(420.0, seed=51)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_pair(root, original, _remastered(original, seed=52))
            store = _Store(root, str(root / "movie.mkv"))

            pipeline.align_pair(store, "project", "pair", _Context())

            import json

            alignment_map = json.loads(
                (root / "pair" / "alignment" / "alignment_map.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(alignment_map["method"], "same_container_identity_v1")
        self.assertEqual(alignment_map["summary"]["matched_ratio"], 1.0)
        # The verdict is now a measurement, and it says so.
        verification = alignment_map["summary"]["same_container_verification"]
        self.assertEqual(verification["policy"], "identity")
        self.assertGreaterEqual(verification["measured_windows"], 8)
        self.assertEqual(store.pair["warnings"], [])


if __name__ == "__main__":
    unittest.main()
