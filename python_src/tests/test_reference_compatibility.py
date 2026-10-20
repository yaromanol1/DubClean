from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.application_pipeline import (
    _route_passthrough_windows,
    _route_semantic_model_windows,
)
from experiments.paired_reference_cancel.reference_compatibility import (
    _map_original_interval,
    _route,
    _timeline_action,
    analyze_reference_compatibility,
    apply_reference_compatibility_recommendations,
)
from experiments.paired_reference_cancel.storage import atomic_json


class _Context:
    def check_stop(self) -> None:
        return None

    def update(self, **_values) -> None:
        return None


class _Store:
    def __init__(self, root: Path):
        self.root = root
        self.models_root = root / "models"
        self.models_root.mkdir(parents=True)
        self.cfg = {
            "reference_compatibility": {
                "window_sec": 10.0,
                "step_sec": 5.0,
                "max_lag_sec": 0.5,
                "mel_bands": 20,
                "min_reference_activity": 0.05,
                "min_active_windows": 2,
                "low_confidence": 0.25,
                "thresholds": {"high": 0.68, "mid": 0.42},
                "weights": {
                    "logmel_envelope": 0.36,
                    "gcc_lag_stability": 0.22,
                    "vad_spectral": 0.32,
                    "reference_activity": 0.10,
                },
            }
        }
        self.pair = {
            "id": "pair",
            "project_id": "project",
            "name": "Тест",
            "stages": {"2": {"status": "completed"}, "3": {"status": "completed"}},
        }

    def pair_dir(self, _project_id: str, _pair_id: str) -> Path:
        return self.root / "pair"

    def load_pair(self, _project_id: str, _pair_id: str) -> dict:
        return self.pair

    def save_pair(self, pair: dict) -> None:
        self.pair = pair


def _write_inputs(store: _Store, original: np.ndarray, dubbed: np.ndarray, rate: int) -> None:
    extracted = store.pair_dir("project", "pair") / "extracted"
    alignment = store.pair_dir("project", "pair") / "alignment"
    extracted.mkdir(parents=True)
    alignment.mkdir(parents=True)
    sf.write(extracted / "original_proxy.wav", original, rate, subtype="FLOAT")
    sf.write(extracted / "dubbed_proxy.wav", dubbed, rate, subtype="FLOAT")
    duration = len(dubbed) / rate
    atomic_json(
        alignment / "alignment_map.json",
        {
            "global": {"offset_sec": 0.0, "speed_ratio": 1.0},
            "segments": [
                {
                    "original_start": 0.0,
                    "original_end": duration,
                    "dubbed_start": 0.0,
                    "dubbed_end": duration,
                    "usable": True,
                }
            ],
        },
    )


class ReferenceCompatibilityTests(unittest.TestCase):
    def test_applying_recommendation_preserves_previous_manual_second_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _Store(Path(temp_dir))
            routing_path = store.pair_dir("project", "pair") / "analysis" / "routing_map.json"
            routing_path.parent.mkdir(parents=True)
            atomic_json(routing_path, {"windows": []})
            store.pair["speech_extraction_second_pass"] = False
            analysis = {
                "band": "MID",
                "created_at": "now",
                "recommended_model": "dubclean_voice",
                "recommended_alignment": "algorithmic",
                "recommended_speech_extraction_second_pass": True,
                "routing_map": str(routing_path),
            }

            apply_reference_compatibility_recommendations(
                store, "project", "pair", analysis
            )

            self.assertTrue(store.pair["speech_extraction_second_pass"])
            self.assertFalse(
                store.pair[
                    "reference_compatibility_manual_speech_extraction_second_pass"
                ]
            )

    def test_similar_mastering_writes_window_routing_contract(self) -> None:
        rate = 8_000
        seconds = 25
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        envelope = 0.25 + 0.75 * (np.sin(2 * np.pi * 0.43 * time) ** 2)
        original = envelope * (
            0.22 * np.sin(2 * np.pi * 183 * time)
            + 0.12 * np.sin(2 * np.pi * 417 * time)
        )
        dubbed = 0.72 * original + 0.018 * np.sin(2 * np.pi * 621 * time)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _Store(Path(temp_dir))
            _write_inputs(store, original, dubbed, rate)
            store.pair["reference_compatibility_applied"] = {"band": "HIGH"}
            store.pair["reference_compatibility_decision"] = {
                "choice": "manual",
                "analysis_created_at": "old",
                "decided_at": "old",
            }

            result = analyze_reference_compatibility(store, "project", "pair", _Context())

            self.assertIn(result["band"], {"HIGH", "MID"})
            self.assertEqual(result["recommended_model"], "dubclean_voice")
            routing = Path(result["routing_map"])
            self.assertTrue(routing.is_file())
            self.assertGreater(result["features"]["analysed_windows"], 1)
            self.assertEqual(store.pair["reference_compatibility_analysis"]["created_at"], result["created_at"])
            self.assertNotIn("reference_compatibility_applied", store.pair)
            self.assertNotIn("reference_compatibility_decision", store.pair)

    def test_silent_reference_is_passthrough_not_fake_v3(self) -> None:
        rate = 8_000
        original = np.zeros(rate * 20, dtype=np.float32)
        dubbed = 0.1 * np.sin(2 * np.pi * 220 * np.arange(rate * 20) / rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = _Store(Path(temp_dir))
            _write_inputs(store, original, dubbed.astype(np.float32), rate)

            result = analyze_reference_compatibility(store, "project", "pair", _Context())

            self.assertEqual(result["band"], "PASSTHROUGH")
            self.assertEqual(result["recommended_model"], "passthrough")
            self.assertIn("сохранена без изменений", result["summary"])

    def test_a_drifting_timeline_is_reported_even_at_a_high_score(self) -> None:
        # Mastering similarity and timeline correctness are separate questions.
        # A dub that steps by a tenth of a second mid-film still matches the
        # original on every spectral cue, so the band stays HIGH — but the
        # timeline verdict has to say the map must be rebuilt.
        cfg = {
            "timeline_offset_tolerance_sec": 0.010,
            "timeline_drift_tolerance_sec": 0.025,
        }

        self.assertEqual(_timeline_action(-0.015, 0.187, cfg), "realign_local")

    def test_a_constant_offset_asks_for_a_rebuilt_map(self) -> None:
        cfg = {
            "timeline_offset_tolerance_sec": 0.010,
            "timeline_drift_tolerance_sec": 0.025,
        }

        self.assertEqual(_timeline_action(0.034, 0.004, cfg), "realign_global")

    def test_an_aligned_pair_needs_nothing_done_to_its_map(self) -> None:
        cfg = {
            "timeline_offset_tolerance_sec": 0.010,
            "timeline_drift_tolerance_sec": 0.025,
        }

        self.assertEqual(_timeline_action(0.002, 0.006, cfg), "none")

    def test_the_timeline_verdict_never_switches_the_mastering_adapter(
        self,
    ) -> None:
        # ``alignment`` drives automatic_assembly's reference_adapter, which
        # adapts level and EQ.  A pair that is merely late must not have that
        # turned on for it.
        cfg = {
            "thresholds": {"high": 0.7, "mid": 0.5},
            "low_confidence": 0.4,
            "timeline_offset_tolerance_sec": 0.010,
            "timeline_drift_tolerance_sec": 0.025,
        }

        route = _route(0.9, 0.9, 0.95, cfg)

        self.assertEqual(route["band"], "HIGH")
        self.assertEqual(route["model"], "dubclean_voice")
        self.assertEqual(route["alignment"], "none")

    def test_a_map_without_segments_subtracts_its_offset(self) -> None:
        # ``offset_sec`` is a delay: original_time = dubbed_time - offset.
        alignment = {"global": {"offset_sec": 0.5, "speed_ratio": 1.0}, "segments": []}

        start, duration = _map_original_interval(alignment, 20.0, 10.0)

        self.assertAlmostEqual(start, 19.5, places=6)
        self.assertAlmostEqual(duration, 10.0, places=6)

    def test_low_route_uses_passthrough(self) -> None:
        cfg = {
            "thresholds": {"high": 0.7, "mid": 0.5},
            "low_confidence": 0.4,
        }
        route = _route(0.2, 0.9, 0.8, cfg)
        self.assertEqual(route["band"], "LOW")
        self.assertEqual(route["model"], "passthrough")
        self.assertEqual(route["alignment"], "none")

    def test_passthrough_window_replaces_only_unsafe_interval(self) -> None:
        rate = 8_000
        processed = np.full(rate * 4, 0.1, dtype=np.float32)
        mixture = np.full(rate * 4, 0.7, dtype=np.float32)
        routing = {
            "windows": [
                {
                    "dubbed_start_sec": 1.0,
                    "dubbed_end_sec": 3.0,
                    "band": "LOW",
                    "recommended_model": "passthrough",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_path = root / "processed.wav"
            mixture_path = root / "mixture.wav"
            destination = root / "routed.flac"
            sf.write(processed_path, processed, rate, subtype="FLOAT")
            sf.write(mixture_path, mixture, rate, subtype="FLOAT")

            _route_passthrough_windows(processed_path, mixture_path, routing, destination)
            output, _ = sf.read(destination, dtype="float32")

            self.assertAlmostEqual(float(np.mean(output[: rate // 2])), 0.1, places=3)
            self.assertAlmostEqual(float(np.mean(output[rate * 3 // 2 : rate * 5 // 2])), 0.7, places=3)
            self.assertAlmostEqual(float(np.mean(output[-rate // 2 :])), 0.1, places=3)

    def test_window_router_uses_voice_and_passthrough_sources(self) -> None:
        rate = 8_000
        seconds = 6
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {}
            for name, value in (("primary", 0.1), ("mixture", 0.8)):
                paths[name] = root / f"{name}.wav"
                sf.write(
                    paths[name], np.full(rate * seconds, value, dtype=np.float32),
                    rate, subtype="FLOAT",
                )
            routing = {
                "windows": [
                    {"dubbed_start_sec": 0.0, "dubbed_end_sec": 2.0,
                     "recommended_model": "dubclean_voice"},
                    {"dubbed_start_sec": 2.0, "dubbed_end_sec": 4.0,
                     "recommended_model": "passthrough"},
                    {"dubbed_start_sec": 4.0, "dubbed_end_sec": 6.0,
                     "recommended_model": "dubclean_voice"},
                ]
            }
            destination = root / "routed.flac"

            _route_semantic_model_windows(
                paths["primary"], paths["mixture"], routing, destination
            )
            output, _ = sf.read(destination, dtype="float32")

            self.assertAlmostEqual(float(np.mean(output[rate // 2 : rate])), 0.1, places=3)
            self.assertAlmostEqual(float(np.mean(output[rate * 5 // 2 : rate * 3])), 0.8, places=3)
            self.assertAlmostEqual(float(np.mean(output[rate * 9 // 2 : rate * 5])), 0.1, places=3)

    def test_window_router_falls_back_to_passthrough_when_voice_is_missing(self) -> None:
        rate = 8_000
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixture = root / "mixture.wav"
            destination = root / "routed.flac"
            sf.write(mixture, np.full(rate * 2, 0.65, dtype=np.float32), rate, subtype="FLOAT")
            routing = {
                "windows": [{
                    "dubbed_start_sec": 0.0,
                    "dubbed_end_sec": 2.0,
                    "recommended_model": "dubclean_voice",
                }]
            }

            _route_semantic_model_windows(None, mixture, routing, destination)
            output, _ = sf.read(destination, dtype="float32")

            self.assertAlmostEqual(float(np.mean(output)), 0.65, places=3)


if __name__ == "__main__":
    unittest.main()
