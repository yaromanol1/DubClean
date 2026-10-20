from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.application_pipeline import (
    SCENE_MEDIA_DELAY_GUARD,
    _measure_scene_media_delays,
    _scene_media_delay_guard,
)

RATE = 16_000


class _Store:
    def __init__(self, configured: dict | None = None) -> None:
        self.cfg = {"alignment": {"scene_media_delay_guard": configured or {}}}


def _programme(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(RATE * seconds)
    time = np.arange(count) / RATE
    envelope = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.5 * time)) ** 2
    hits = np.zeros(count)
    for position in np.arange(0.5, seconds - 0.5, 1.7):
        start = int(position * RATE)
        hits[start : start + RATE // 8] = np.linspace(1.0, 0.0, RATE // 8) ** 2
    values = (rng.normal(size=count) * envelope) + 3.0 * hits * rng.normal(size=count)
    # The batches this runs on are FLAC: material that clips on write would be
    # measuring the clipper, not the delay.
    return (0.5 * values / max(float(np.max(np.abs(values))), 1e-9)).astype(np.float32)


def _delayed(values: np.ndarray, delay_sec: float) -> np.ndarray:
    shift = int(round(delay_sec * RATE))
    out = np.zeros_like(values)
    if shift > 0:
        out[shift:] = values[:-shift]
    elif shift < 0:
        out[:shift] = values[-shift:]
    else:
        out = values.copy()
    return out


def _run(original: np.ndarray, dubbed: np.ndarray, settings: dict) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        sf.write(root / "original.flac", original, RATE)
        sf.write(root / "dubbed.flac", dubbed, RATE)
        selected = [{"_range": (0, len(original))}]
        _measure_scene_media_delays(
            root / "original.flac", root / "dubbed.flac", selected, RATE, settings
        )
    return selected[0]


class SceneMediaDelayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = dict(SCENE_MEDIA_DELAY_GUARD)

    def test_a_real_scene_delay_is_measured_and_applied(self) -> None:
        original = _programme(30.0, seed=3)
        # The dubbed mix carries the same programme 60 ms later, so the bed has
        # to be moved 60 ms later too before it goes under the Russian voice.
        dubbed = 0.7 * _delayed(original, 0.06) + 0.02 * np.random.default_rng(
            4
        ).normal(size=len(original)).astype(np.float32)

        scene = _run(original, dubbed.astype(np.float32), self.settings)

        self.assertTrue(scene["media_delay_report"]["applied"])
        self.assertAlmostEqual(scene["local_media_delay_sec"], 0.06, delta=0.008)

    def test_an_aligned_scene_is_left_alone(self) -> None:
        original = _programme(30.0, seed=5)
        dubbed = 0.7 * original + 0.02 * np.random.default_rng(6).normal(
            size=len(original)
        ).astype(np.float32)

        scene = _run(original, dubbed.astype(np.float32), self.settings)

        self.assertEqual(scene["local_media_delay_sec"], 0.0)
        self.assertFalse(scene["media_delay_report"]["applied"])

    def test_unrelated_material_never_moves_the_bed(self) -> None:
        # Two different programmes: whatever the estimator reports here is an
        # artefact, and shifting on it would be worse than doing nothing.
        original = _programme(30.0, seed=7)
        dubbed = _programme(30.0, seed=8)

        scene = _run(original, dubbed, self.settings)

        self.assertEqual(scene["local_media_delay_sec"], 0.0)
        self.assertFalse(scene["media_delay_report"]["applied"])
        self.assertIn(
            scene["media_delay_report"]["reason"],
            {"low_confidence", "no_measurable_gain", "below_floor"},
        )

    def test_configuration_overrides_the_defaults(self) -> None:
        settings = _scene_media_delay_guard(_Store({"min_confidence": 0.9}))

        self.assertEqual(settings["min_confidence"], 0.9)
        self.assertEqual(
            settings["max_delay_sec"], SCENE_MEDIA_DELAY_GUARD["max_delay_sec"]
        )

    def test_a_delay_below_the_floor_is_not_worth_moving_audio_for(self) -> None:
        original = _programme(30.0, seed=9)
        dubbed = 0.7 * _delayed(original, 0.002) + 0.02 * np.random.default_rng(
            10
        ).normal(size=len(original)).astype(np.float32)

        scene = _run(original, dubbed.astype(np.float32), self.settings)

        self.assertEqual(scene["local_media_delay_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
