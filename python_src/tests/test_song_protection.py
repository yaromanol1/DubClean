from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.song_protection import (
    _merge_dialogue_windows,
    _merge_windows,
    _build_restoration_plan,
    _window_decision,
    apply_song_protection,
    detect_song_intervals,
    dialogue_timeline_guard,
    resolved_config,
    scan_song_candidates,
)


def _music(time: np.ndarray) -> np.ndarray:
    return (
        0.12 * np.sin(2 * np.pi * 110.0 * time)
        + 0.08 * np.sin(2 * np.pi * 220.0 * time)
        + 0.06 * np.sin(2 * np.pi * 330.0 * time)
        + 0.04 * np.sin(2 * np.pi * 1760.0 * time)
    ).astype(np.float32)


def _vocal(time: np.ndarray) -> np.ndarray:
    fundamental = 205.0 + 18.0 * np.sin(2 * np.pi * 0.18 * time)
    phase = 2 * np.pi * np.cumsum(fundamental) / 8000.0
    return (
        0.16 * np.sin(phase)
        + 0.10 * np.sin(2.0 * phase)
        + 0.06 * np.sin(3.0 * phase)
    ).astype(np.float32)


def _spoken_phrase(time: np.ndarray, start: float, end: float) -> np.ndarray:
    result = np.zeros_like(time, dtype=np.float32)
    active = (time >= start) & (time < end)
    local = time[active] - start
    if not np.any(active):
        return result
    phase = 2 * np.pi * (145.0 * local + 38.0 * local * local)
    syllables = 0.18 + 0.82 * np.square(np.sin(2 * np.pi * 3.8 * local))
    envelope = np.hanning(max(2, int(np.sum(active))))
    result[active] = (
        (
            0.18 * np.sin(phase)
            + 0.07 * np.sin(2.3 * phase)
            + 0.04 * np.sin(4.7 * phase)
        )
        * syllables
        * envelope
    )
    return result


class _FixedClassifier:
    backend = "test_audioset_classifier"

    def __init__(self, scores: dict[str, float], sample_rate: int = 8000) -> None:
        self.scores = scores
        self.sample_rate = sample_rate
        self.received: list[np.ndarray] = []

    def predict_many(self, waveforms: list[np.ndarray]) -> list[dict[str, float]]:
        self.received.extend(np.asarray(value) for value in waveforms)
        return [dict(self.scores) for _ in waveforms]


SONG_SCORES = {
    "singing_score": 0.55,
    "direct_singing_score": 0.24,
    "music_score": 0.92,
    "speech_score": 0.12,
    "vocal_music_score": 0.31,
    "song_score": 0.28,
}

DIALOGUE_SCORES = {
    "singing_score": 0.03,
    "direct_singing_score": 0.01,
    "music_score": 0.90,
    "speech_score": 0.88,
    "vocal_music_score": 0.01,
    "song_score": 0.01,
}


def _real_damage_window(
    *,
    music_presence: float = 0.42,
    speech_share: float = 0.03,
    removed_vocal_share: float = 0.99,
    removed_program_share: float = 0.99,
    vocal_continuity: float = 0.50,
    singing_score: float = 0.14,
    direct_singing_score: float = 0.09,
    music_score: float = 0.87,
    speech_score: float = 0.01,
    background_tonality: float = 0.90,
    background_rms_db: float = -35.0,
    vocal_tonality: float = 0.98,
) -> dict:
    """Feature-level analogue of the real scene_08 damage windows."""
    background_level_share = max(
        0.0,
        (
            float(music_presence)
            - 0.45 * background_tonality
        )
        / 0.55,
    )
    processed_vocal = max(1e-6, 1.0 - removed_vocal_share)
    # Keeping the processed vocal/backing ratio above the source ratio makes
    # relative removal fail, as it did with the learned real M&E mastering.
    processed_backing = processed_vocal / 2.0
    return _window_decision(
        {
            "rms_db": -18.0,
            "vocal_energy": 1.0,
            "backing_energy": 1.0,
            "full_energy": 1.0,
        },
        {
            "vocal_energy": speech_share,
            "vocal_continuity": vocal_continuity,
            "vocal_tonality": vocal_tonality,
        },
        {
            "full_energy": background_level_share,
            "music_tonality": background_tonality,
            "rms_db": background_rms_db,
        },
        {
            "vocal_energy": processed_vocal,
            "backing_energy": processed_backing,
            "full_energy": max(1e-6, 1.0 - removed_program_share),
        },
        {
            "singing_score": singing_score,
            "direct_singing_score": direct_singing_score,
            "music_score": music_score,
            "speech_score": speech_score,
            "vocal_music_score": 0.02,
            "song_score": 0.04,
        },
        resolved_config({}),
    )


def _timed_real_damage_window(
    index: int,
    **parameters,
) -> dict:
    decision = _real_damage_window(**parameters)
    start = float(index) * 5.0
    return {
        "index": index,
        "start_sec": start,
        "end_sec": start + 10.0,
        **decision,
    }


class SongProtectionTests(unittest.TestCase):
    def test_full_track_preview_scan_uses_only_aligned_english_original(self) -> None:
        rate = 8000
        seconds = 24
        original = np.full(rate * seconds, 0.25, dtype=np.float32)
        classifier = _FixedClassifier(SONG_SCORES, rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "aligned_english.wav"
            report_path = root / "scan.json"
            sf.write(original_path, original, rate, subtype="FLOAT")
            report = scan_song_candidates(
                original_path,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                report_path=report_path,
                classifier=classifier,
            )

        self.assertTrue(report["candidate_intervals"])
        self.assertEqual(report["purpose"], "preview_scene_selection_only")
        self.assertTrue(classifier.received)
        self.assertAlmostEqual(
            float(np.mean(classifier.received[0])), 0.25, places=3
        )

    def test_dialogue_guard_covers_later_voice_synchronization(self) -> None:
        cfg = resolved_config(
            {
                "dialogue_boundary_margin_sec": 0.10,
                "dialogue_timeline_guard_sec": 0.85,
            }
        )
        intervals = _merge_dialogue_windows(
            [
                {
                    "start_sec": 5.0,
                    "end_sec": 6.0,
                    "block_restoration": True,
                    "dialogue_candidate": True,
                    "features": {
                        "dialogue_score": 0.8,
                        "nonmatching_share": 0.7,
                        "speech_likeness": 0.6,
                        "translation_voice_available": True,
                        "translation_voice_evidence_score": 0.8,
                        "translation_voice_share": 0.5,
                    },
                }
            ],
            0.0,
            10.0,
            cfg,
        )

        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0]["start_sec"], 4.05, places=3)
        self.assertAlmostEqual(intervals[0]["end_sec"], 6.95, places=3)
        self.assertAlmostEqual(intervals[0]["timeline_guard_sec"], 0.85, places=3)

    def test_guard_policy_matches_off_manual_and_conservative_modes(
        self,
    ) -> None:
        cfg = {"synchronization_guard_sec": 0.85}
        self.assertEqual(
            dialogue_timeline_guard(cfg),
            0.0,
        )
        self.assertAlmostEqual(
            dialogue_timeline_guard(
                cfg,
                manual_voice_delay_sec=-0.4,
            ),
            0.4,
        )
        self.assertAlmostEqual(
            dialogue_timeline_guard(
                cfg,
                synchronize_speech=True,
            ),
            0.85,
        )
        self.assertAlmostEqual(
            dialogue_timeline_guard(
                cfg,
                manual_voice_delay_sec=1.1,
                synchronize_speech=True,
            ),
            1.1,
        )

    def test_real_damage_features_use_strong_absolute_route(self) -> None:
        decision = _real_damage_window()

        self.assertTrue(decision["candidate"])
        self.assertEqual(
            decision["damage_evidence_route"],
            "strong_absolute_damage",
        )
        self.assertFalse(
            decision["checks"]["standard_relative_damage_route"]
        )
        self.assertTrue(
            decision["checks"]["strong_absolute_damage_route"]
        )
        self.assertAlmostEqual(
            decision["features"]["music_presence"],
            0.42,
            places=3,
        )
        self.assertAlmostEqual(
            decision["features"]["removed_vocal_share"],
            0.99,
            places=3,
        )
        self.assertLess(
            decision["features"]["relative_vocal_removal_db"],
            0.0,
        )
        self.assertGreaterEqual(decision["confidence"], 0.78)

    def test_strong_absolute_route_rejects_every_boundary_negative(
        self,
    ) -> None:
        cases = {
            "dialogue_dominates": (
                {"speech_score": 0.12},
                "strong_singing_over_speech",
            ),
            "sparse_vocal": (
                {"vocal_continuity": 0.349},
                "strong_vocal_continuity",
            ),
            "weak_music_layer": (
                {
                    "music_presence": 0.399,
                    "background_tonality": 0.80,
                },
                "strong_music_background",
            ),
            "inaudible_music_layer": (
                {"background_rms_db": -42.001},
                "strong_audible_music_background",
            ),
            "weak_music_tonality": (
                {"background_tonality": 0.799},
                "strong_music_tonality",
            ),
            "near_zero_stem": (
                {"speech_share": 0.019},
                "strong_speech_stem_evidence",
            ),
            "weak_vocal_tonality": (
                {"vocal_tonality": 0.949},
                "strong_vocal_tonality",
            ),
            "insufficient_absolute_damage": (
                {"removed_vocal_share": 0.949},
                "strong_absolute_vocal_removal",
            ),
        }
        for name, (parameters, failed_check) in cases.items():
            with self.subTest(name=name):
                decision = _real_damage_window(**parameters)
                self.assertFalse(decision["candidate"])
                self.assertFalse(decision["checks"][failed_check])
                self.assertFalse(
                    decision["checks"][
                        "standard_relative_damage_route"
                    ]
                )
                self.assertFalse(
                    decision["checks"]["strong_absolute_damage_route"]
                )

    def test_standard_relative_route_is_unchanged(self) -> None:
        cfg = resolved_config({})
        decision = _window_decision(
            {
                "rms_db": -18.0,
                "vocal_energy": 1.0,
                "backing_energy": 1.0,
                "full_energy": 1.0,
            },
            {
                "vocal_energy": 0.35,
                "vocal_continuity": 0.60,
                "vocal_tonality": 0.95,
            },
            {
                "full_energy": 0.40,
                "music_tonality": 0.90,
            },
            {
                "vocal_energy": 0.50,
                "backing_energy": 2.0,
            },
            {
                **SONG_SCORES,
                # Speech dominance is irrelevant to the retained standard
                # route; it is only a veto on the new strong fallback.
                "speech_score": 0.98,
            },
            cfg,
        )

        self.assertTrue(decision["candidate"])
        self.assertEqual(
            decision["damage_evidence_route"],
            "standard_relative_damage",
        )
        self.assertTrue(
            decision["checks"]["standard_relative_damage_route"]
        )
        self.assertFalse(
            decision["checks"]["strong_absolute_damage_route"]
        )

    def test_short_strong_absolute_group_is_still_rejected(self) -> None:
        first = {
            **_real_damage_window(),
            "start_sec": 0.0,
            "end_sec": 4.0,
        }
        second = {
            **_real_damage_window(),
            "start_sec": 2.0,
            "end_sec": 6.0,
        }

        confirmed, rejected = _merge_windows(
            [first, second],
            resolved_config(
                {
                    "window_sec": 4.0,
                    "minimum_interval_sec": 12.0,
                }
            ),
        )

        self.assertEqual(confirmed, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("слишком короткий", rejected[0]["rejection_reason"])
        self.assertIn(
            "strong_absolute_damage",
            rejected[0]["damage_evidence_routes"],
        )

    def test_anchor_backed_group_recovers_real_like_scene_08(self) -> None:
        windows = [
            _timed_real_damage_window(
                0,
                removed_vocal_share=0.70,
                removed_program_share=0.75,
                speech_share=0.005,
                vocal_continuity=0.10,
            ),
            _timed_real_damage_window(
                1,
                removed_vocal_share=0.949,
                removed_program_share=0.954,
            ),
            _timed_real_damage_window(
                2,
                removed_vocal_share=0.933,
                removed_program_share=0.941,
            ),
            _timed_real_damage_window(
                3,
                removed_vocal_share=0.908,
                removed_program_share=0.924,
            ),
            _timed_real_damage_window(
                4,
                removed_vocal_share=0.960,
                removed_program_share=0.964,
            ),
            _timed_real_damage_window(
                5,
                removed_vocal_share=0.98,
                removed_program_share=0.98,
                speech_share=0.001,
                vocal_continuity=0.18,
            ),
            _timed_real_damage_window(
                6,
                removed_vocal_share=0.98,
                removed_program_share=0.98,
                singing_score=0.35,
                speech_score=0.74,
            ),
        ]

        confirmed, rejected = _merge_windows(
            windows,
            resolved_config({}),
            translation_voice_available=True,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(confirmed), 1)
        interval = confirmed[0]
        self.assertEqual(interval["start_sec"], 0.0)
        self.assertEqual(interval["end_sec"], 35.0)
        self.assertIn(
            "programme_and_vocal_loss",
            interval["damage_evidence_routes"],
        )
        anchor_group = interval["anchor_backed_groups"][0]
        self.assertEqual(anchor_group["support_window_count"], 3)
        self.assertEqual(anchor_group["strict_anchor_window_count"], 1)
        self.assertEqual(anchor_group["missed_grid_windows"], 1)
        self.assertGreaterEqual(
            anchor_group["removed_vocal_share_median"], 0.94
        )
        self.assertGreaterEqual(
            anchor_group["removed_program_share_median"], 0.94
        )
        extension = interval["context_boundary_extension"]
        self.assertTrue(extension["applied"])
        self.assertIn(
            "потерю программы и вокала",
            " ".join(interval["reasons"]),
        )

    def test_anchor_group_recovers_song_when_preview_background_was_removed(self) -> None:
        windows = [
            _timed_real_damage_window(
                0,
                removed_vocal_share=0.997,
                removed_program_share=0.997,
                background_rms_db=-49.6,
            ),
            _timed_real_damage_window(
                1,
                removed_vocal_share=0.971,
                removed_program_share=0.972,
                background_rms_db=-48.4,
            ),
            _timed_real_damage_window(
                2,
                removed_vocal_share=0.947,
                removed_program_share=0.949,
                background_rms_db=-48.7,
            ),
            _timed_real_damage_window(
                3,
                removed_vocal_share=0.889,
                removed_program_share=0.900,
                background_rms_db=-50.0,
            ),
        ]

        confirmed, rejected = _merge_windows(
            windows,
            resolved_config({}),
            translation_voice_available=True,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["start_sec"], 0.0)
        self.assertEqual(confirmed[0]["end_sec"], 25.0)
        self.assertTrue(
            windows[0]["checks"][
                "programme_and_vocal_loss_support_route"
            ]
        )
        self.assertFalse(
            windows[0]["checks"]["strong_audible_music_background"]
        )
        self.assertTrue(
            confirmed[0]["context_boundary_extension"]["applied"]
        )

    def test_anchor_group_requires_strict_anchor_and_dense_support(self) -> None:
        no_anchor = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=value,
                removed_program_share=0.95,
            )
            for index, value in enumerate((0.949, 0.945, 0.949))
        ]
        confirmed, _ = _merge_windows(
            no_anchor,
            resolved_config({}),
            translation_voice_available=True,
        )
        self.assertEqual(confirmed, [])
        self.assertIn(
            "нет строгого окна-якоря",
            no_anchor[0]["anchor_backed_group_evaluation"]["failures"],
        )

        two_windows = [
            _timed_real_damage_window(
                0,
                removed_vocal_share=0.949,
                removed_program_share=0.95,
            ),
            _timed_real_damage_window(
                1,
                removed_vocal_share=0.96,
                removed_program_share=0.96,
            ),
        ]
        confirmed, rejected = _merge_windows(
            two_windows,
            resolved_config({}),
            translation_voice_available=True,
        )
        self.assertEqual(confirmed, [])
        self.assertTrue(rejected)
        self.assertIn(
            "недостаточно опорных окон",
            two_windows[0]["anchor_backed_group_evaluation"]["failures"],
        )

    def test_anchor_group_rejects_uniform_gain_and_dense_dialogue(self) -> None:
        uniform_gain = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=0.9375,
                removed_program_share=0.9375,
            )
            for index in range(4)
        ]
        confirmed, _ = _merge_windows(
            uniform_gain,
            resolved_config({}),
            translation_voice_available=True,
        )
        self.assertEqual(confirmed, [])
        evaluation = uniform_gain[0]["anchor_backed_group_evaluation"]
        self.assertIn("нет строгого окна-якоря", evaluation["failures"])
        self.assertIn(
            "медианная потеря вокала недостаточна",
            evaluation["failures"],
        )
        self.assertIn(
            "медианная потеря программы недостаточна",
            evaluation["failures"],
        )

        dense_dialogue = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=0.97,
                removed_program_share=0.97,
                singing_score=0.35,
                direct_singing_score=0.10,
                speech_score=0.80,
            )
            for index in range(4)
        ]
        confirmed, rejected = _merge_windows(
            dense_dialogue,
            resolved_config({}),
            translation_voice_available=True,
        )
        self.assertEqual(confirmed, [])
        self.assertEqual(rejected, [])
        self.assertTrue(
            all(
                not item["checks"][
                    "programme_and_vocal_loss_support_route"
                ]
                for item in dense_dialogue
            )
        )

    def test_moderate_group_requires_translation_voice(self) -> None:
        windows = [
            _timed_real_damage_window(
                0,
                removed_vocal_share=0.949,
                removed_program_share=0.954,
            ),
            _timed_real_damage_window(
                1,
                removed_vocal_share=0.943,
                removed_program_share=0.945,
            ),
            _timed_real_damage_window(
                2,
                removed_vocal_share=0.96,
                removed_program_share=0.964,
            ),
        ]

        confirmed, rejected = _merge_windows(
            windows,
            resolved_config({}),
            translation_voice_available=False,
        )

        self.assertEqual(confirmed, [])
        self.assertTrue(rejected)
        self.assertIn(
            "нет русской речевой дорожки",
            " ".join(
                windows[0]["anchor_backed_group_evaluation"]["failures"]
            ),
        )
        self.assertFalse(
            any(
                item.get("damage_evidence_route")
                == "programme_and_vocal_loss"
                for item in windows
            )
        )

    def test_context_extension_stops_at_speech_and_configured_limit(self) -> None:
        speech_neighbor = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=0.96,
                removed_program_share=0.96,
            )
            for index in (1, 2, 3)
        ]
        speech_neighbor.append(
            _timed_real_damage_window(
                4,
                removed_vocal_share=0.98,
                removed_program_share=0.98,
                singing_score=0.35,
                speech_score=0.80,
            )
        )
        confirmed, _ = _merge_windows(
            speech_neighbor,
            resolved_config({}),
            translation_voice_available=True,
        )
        self.assertEqual(confirmed[0]["end_sec"], 25.0)
        self.assertFalse(
            confirmed[0]["context_boundary_extension"]["applied"]
        )

        one_step = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=(
                    0.96 if index in {2, 3, 4} else 0.70
                ),
                removed_program_share=(
                    0.96 if index in {2, 3, 4} else 0.75
                ),
                speech_share=(0.03 if index in {2, 3, 4} else 0.005),
                vocal_continuity=(
                    0.50 if index in {2, 3, 4} else 0.10
                ),
            )
            for index in range(7)
        ]
        confirmed, _ = _merge_windows(
            one_step,
            resolved_config(
                {"maximum_anchor_context_extension_sec": 5.0}
            ),
            translation_voice_available=True,
        )
        interval = confirmed[0]
        self.assertEqual(interval["start_sec"], 5.0)
        self.assertEqual(interval["end_sec"], 35.0)
        context = interval["anchor_backed_groups"][0][
            "context_boundary_extension"
        ]
        self.assertEqual(context["left_extension_sec"], 5.0)
        self.assertEqual(context["right_extension_sec"], 5.0)

    def test_confirmed_damage_extends_through_sustained_song_context(self) -> None:
        windows = [
            _timed_real_damage_window(
                index,
                removed_vocal_share=(
                    0.98 if index in {0, 1} else 0.80
                ),
                removed_program_share=(
                    0.98 if index in {0, 1} else 0.82
                ),
                singing_score=(0.35 if index == 6 else 0.20),
                speech_score=(0.80 if index == 6 else 0.02),
            )
            for index in range(7)
        ]

        confirmed, rejected = _merge_windows(
            windows,
            resolved_config(
                {"maximum_anchor_context_extension_sec": 20.0}
            ),
            translation_voice_available=True,
        )

        self.assertEqual(rejected, [])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0]["start_sec"], 0.0)
        self.assertEqual(confirmed[0]["end_sec"], 35.0)
        extension = confirmed[0]["context_boundary_extension"]
        self.assertTrue(extension["applied"])
        self.assertEqual(extension["right_extension_sec"], 20.0)
        self.assertEqual(
            [item["index"] for item in extension["windows"]],
            [2, 3, 4, 5],
        )

    def _write_case(
        self,
        root: Path,
        source: np.ndarray,
        speech: np.ndarray,
        background: np.ndarray,
        processed: np.ndarray,
        rate: int,
    ) -> tuple[Path, Path, Path, Path]:
        paths = tuple(root / name for name in ("source.wav", "speech.wav", "background.wav", "processed.wav"))
        for path, values in zip(paths, (source, speech, background, processed), strict=True):
            sf.write(path, values, rate, subtype="FLOAT")
        return paths

    def test_sustained_song_with_removed_vocal_is_confirmed(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        source = music + vocal
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._write_case(root, source, vocal, music, music + 0.08 * vocal, rate)
            report = detect_song_intervals(
                paths[0],
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
            )

        self.assertTrue(report["confirmed_intervals"])
        interval = report["confirmed_intervals"][0]
        self.assertGreaterEqual(interval["duration_sec"], 8.0)
        self.assertIn("вокальная", " ".join(interval["reasons"]))

    def test_matching_song_without_translation_voice_is_not_replaced(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        original_mix = music + vocal
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            paths = self._write_case(
                root,
                original_mix,
                vocal,
                music,
                music + 0.08 * vocal,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
            )

        self.assertTrue(report["confirmed_intervals"])
        interval = report["confirmed_intervals"][0]
        self.assertGreaterEqual(
            float(interval["track_comparison"]["music_similarity"]),
            0.78,
        )
        self.assertEqual(interval["restoration_source"], "none")
        self.assertEqual(interval["restoration_segments"], [])
        self.assertEqual(report["restoration_segments"], [])
        self.assertIn(
            "не найдена очищенная русская речевая дорожка",
            interval["restoration_source_reason"],
        )

    def test_ordinary_dialogue_over_music_is_not_replaced(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        spoken = np.zeros_like(time)
        for start in (1.0, 4.2, 8.0, 12.4, 17.0, 21.0):
            active = (time >= start) & (time < start + 0.75)
            local = time[active] - start
            spoken[active] = (
                0.18 * np.sin(2 * np.pi * 170.0 * local)
                + 0.07 * np.sin(2 * np.pi * 510.0 * local)
            ) * np.hanning(max(2, int(np.sum(active))))
        source = music + spoken
        # Even if a separator removes the dialogue almost completely, sparse
        # spoken phrases must not be mistaken for a sustained song.
        processed = music
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._write_case(root, source, spoken, music, processed, rate)
            report = detect_song_intervals(
                paths[0],
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(DIALOGUE_SCORES, rate),
            )

        self.assertEqual(report["confirmed_intervals"], [])

    def test_false_song_scores_and_attenuation_do_not_replace_dialogue(
        self,
    ) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        spoken = np.zeros_like(time)
        for start in np.arange(0.4, 23.0, 1.2):
            spoken += _spoken_phrase(
                time,
                float(start),
                min(float(start + 0.9), float(seconds)),
            )
        source = music + spoken
        # Deliberately create the dangerous numeric shape from the real
        # failure: audible but low-level learned M&E and ~99% absolute loss.
        learned_me = 0.10 * music
        processed = 0.01 * music
        false_song_scores = {
            "singing_score": 0.18,
            "direct_singing_score": 0.09,
            "music_score": 0.90,
            "speech_score": 0.45,
            "vocal_music_score": 0.05,
            "song_score": 0.05,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            sf.write(original_path, source, rate, subtype="FLOAT")
            paths = self._write_case(
                root,
                source,
                spoken,
                learned_me,
                processed,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(false_song_scores, rate),
            )

        dangerous = [
            item
            for item in report["window_decisions"]
            if item["checks"]["learned_singing"]
            and item["checks"]["learned_direct_singing"]
            and item["checks"]["strong_absolute_vocal_removal"]
            and item["checks"]["strong_music_background"]
        ]
        self.assertTrue(dangerous)
        self.assertTrue(
            all(
                not item["checks"]["strong_singing_over_speech"]
                for item in dangerous
            )
        )
        self.assertTrue(
            all(not item["candidate"] for item in dangerous)
        )
        self.assertEqual(report["confirmed_intervals"], [])

    def test_classifier_reads_original_track_not_dubbed_mix(self) -> None:
        rate = 8000
        seconds = 12
        original = np.full(rate * seconds, 0.25, dtype=np.float32)
        dubbed = np.full(rate * seconds, 0.05, dtype=np.float32)
        classifier = _FixedClassifier(DIALOGUE_SCORES, rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            sf.write(original_path, original, rate, subtype="FLOAT")
            paths = self._write_case(root, dubbed, dubbed, dubbed, dubbed, rate)
            detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                },
                classifier=classifier,
            )

        self.assertTrue(classifier.received)
        self.assertAlmostEqual(
            float(np.mean(classifier.received[0])), 0.25, places=3
        )

    def test_confirmed_interval_restores_source_with_crossfades(self) -> None:
        rate = 8000
        seconds = 6
        processed = np.full(rate * seconds, 0.1, dtype=np.float32)
        source = np.full(rate * seconds, 0.7, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_path = root / "processed.wav"
            source_path = root / "source.wav"
            destination = root / "protected.flac"
            sf.write(processed_path, processed, rate, subtype="FLOAT")
            sf.write(source_path, source, rate, subtype="FLOAT")
            apply_song_protection(
                processed_path,
                source_path,
                destination,
                [{"start_sec": 2.0, "end_sec": 4.0}],
                crossfade_sec=0.25,
            )
            output, _ = sf.read(destination, dtype="float32")

        self.assertAlmostEqual(float(np.mean(output[:rate])), 0.1, places=3)
        self.assertAlmostEqual(float(np.mean(output[rate * 5 // 2 : rate * 7 // 2])), 0.7, places=3)
        core = output[rate * 5 // 2 : rate * 7 // 2]
        self.assertLess(float(np.max(np.abs(core - 0.7))), 2e-6)
        self.assertAlmostEqual(float(np.mean(output[-rate:])), 0.1, places=3)
        self.assertGreater(float(output[rate * 2 + rate // 8]), 0.1)
        self.assertLess(float(output[rate * 2 + rate // 8]), 0.7)

    def test_selected_source_is_applied_per_restoration_segment(self) -> None:
        rate = 8000
        seconds = 8
        processed = np.full(rate * seconds, 0.1, dtype=np.float32)
        dubbed = np.full(rate * seconds, 0.5, dtype=np.float32)
        original = np.full(rate * seconds, 0.8, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_path = root / "processed.wav"
            dubbed_path = root / "dubbed.wav"
            original_path = root / "original.wav"
            destination = root / "protected.flac"
            sf.write(processed_path, processed, rate, subtype="FLOAT")
            sf.write(dubbed_path, dubbed, rate, subtype="FLOAT")
            sf.write(original_path, original, rate, subtype="FLOAT")
            apply_song_protection(
                processed_path,
                dubbed_path,
                destination,
                [
                    {
                        "start_sec": 1.0,
                        "end_sec": 3.0,
                        "restore_source": "original_aligned",
                    },
                    {
                        "start_sec": 5.0,
                        "end_sec": 7.0,
                        "restore_source": "dubbed",
                    },
                ],
                original_mix=original_path,
                crossfade_sec=0.1,
            )
            output, _ = sf.read(destination, dtype="float32")

        self.assertAlmostEqual(float(np.mean(output[rate * 3 // 2 : rate * 5 // 2])), 0.8, places=3)
        self.assertAlmostEqual(float(np.mean(output[rate * 11 // 2 : rate * 13 // 2])), 0.5, places=3)
        self.assertAlmostEqual(float(np.mean(output[rate * 7 // 2 : rate * 9 // 2])), 0.1, places=3)

    def test_production_dialogue_thresholds_keep_translated_dialogue_island(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        # Keep the translated phrase at a realistic programme level: on the
        # real film it occupies at least 0.597 of full-mix RMS, while leaked
        # song vocals remain below 0.475.
        translated_dialogue = 3.0 * _spoken_phrase(time, 9.0, 11.2)
        original_mix = music + vocal
        dubbed_mix = original_mix + translated_dialogue
        dubbed_speech = vocal + translated_dialogue
        processed = music + translated_dialogue
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            translation_voice_path = root / "translation_voice.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            sf.write(
                translation_voice_path,
                translated_dialogue,
                rate,
                subtype="FLOAT",
            )
            paths = self._write_case(
                root,
                dubbed_mix,
                dubbed_speech,
                music,
                processed,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(
                    {**SONG_SCORES, "speech_score": 0.98}, rate
                ),
                original_speech_stem=original_speech_path,
                translation_voice_stem=translation_voice_path,
            )
            protected_path = root / "protected.flac"
            apply_song_protection(
                paths[3],
                paths[0],
                protected_path,
                report["restoration_segments"],
                original_mix=original_path,
                crossfade_sec=0.1,
            )
            protected, _ = sf.read(protected_path, dtype="float32")

        interval = report["confirmed_intervals"][0]
        self.assertEqual(interval["restoration_source"], "original_aligned")
        dialogue_intervals = interval["dialogue_detection"]["confirmed_intervals"]
        self.assertFalse(
            interval["dialogue_detection"]["full_interval_translation_block"]
        )
        self.assertTrue(
            any(
                float(item["start_sec"]) < 10.0 < float(item["end_sec"])
                for item in dialogue_intervals
            )
        )
        self.assertIn(
            "translation_voice_spectral_dynamics",
            dialogue_intervals[0]["evidence_sources"],
        )
        self.assertIn(
            "translation_voice_program_share",
            dialogue_intervals[0]["evidence_sources"],
        )
        self.assertGreaterEqual(
            float(
                dialogue_intervals[0][
                    "mean_translation_voice_program_share"
                ]
            ),
            0.55,
        )
        self.assertTrue(
            all(
                not (
                    float(item["start_sec"]) <= 10.0
                    <= float(item["end_sec"])
                )
                for item in report["restoration_segments"]
            )
        )
        clean_slice = slice(rate * 4, rate * 5)
        dialogue_slice = slice(rate * 19 // 2, rate * 21 // 2)
        self.assertLess(
            float(np.max(np.abs(protected[clean_slice] - original_mix[clean_slice]))),
            2e-5,
        )
        self.assertLess(
            float(np.max(np.abs(protected[dialogue_slice] - processed[dialogue_slice]))),
            2e-5,
        )

    def test_gain_only_stem_difference_does_not_create_dialogue_island(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        original_mix = music + vocal
        dubbed_speech = 0.62 * vocal
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            translation_voice_path = root / "translation_voice.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            sf.write(
                translation_voice_path,
                np.zeros_like(vocal),
                rate,
                subtype="FLOAT",
            )
            paths = self._write_case(
                root,
                original_mix,
                dubbed_speech,
                music,
                music + 0.05 * vocal,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                    "minimum_track_music_similarity": 0.60,
                    "minimum_track_matched_window_share": 0.50,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
                translation_voice_stem=translation_voice_path,
            )

        interval = report["confirmed_intervals"][0]
        self.assertEqual(
            interval["dialogue_detection"]["blocked_restoration_intervals"],
            [],
        )
        self.assertTrue(report["restoration_segments"])

    def test_leaked_song_vocal_below_program_share_does_not_block_restoration(
        self,
    ) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        original_mix = music + vocal
        # The leak dominates this synthetic cleaned-speech stem, but remains
        # a minority of the full programme. It must not be called dialogue.
        leaked_vocal = 0.50 * vocal
        dubbed_speech = 0.55 * vocal
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            translation_voice_path = root / "translation_voice.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            sf.write(
                translation_voice_path,
                leaked_vocal,
                rate,
                subtype="FLOAT",
            )
            paths = self._write_case(
                root,
                original_mix,
                dubbed_speech,
                music,
                music + 0.05 * vocal,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                    "minimum_track_music_similarity": 0.60,
                    "minimum_track_matched_window_share": 0.50,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
                translation_voice_stem=translation_voice_path,
            )

        interval = report["confirmed_intervals"][0]
        dialogue = interval["dialogue_detection"]
        program_shares = [
            float(item["features"]["translation_voice_program_share"])
            for item in dialogue["windows"]
        ]
        stem_shares = [
            float(item["features"]["translation_voice_share"])
            for item in dialogue["windows"]
        ]
        self.assertGreater(min(stem_shares), 0.80)
        self.assertLess(max(program_shares), 0.55)
        self.assertEqual(dialogue["blocked_restoration_intervals"], [])
        self.assertTrue(report["restoration_segments"])

    def test_loud_tonal_stem_leak_is_not_mistaken_for_translation(
        self,
    ) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        original_mix = music + vocal
        # Reproduce the dangerous real-film shape: separator leakage is loud
        # enough to dominate both its stem and the full programme, but its
        # vocal-band energy does not dominate the musical backing like a
        # translated spoken phrase does.
        tonal_leak = 0.60 * vocal + 1.20 * music
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            translation_voice_path = root / "translation_voice.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            sf.write(
                translation_voice_path,
                tonal_leak,
                rate,
                subtype="FLOAT",
            )
            paths = self._write_case(
                root,
                original_mix,
                vocal,
                music,
                music + 0.05 * vocal,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                    "minimum_track_music_similarity": 0.60,
                    "minimum_track_matched_window_share": 0.50,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
                translation_voice_stem=translation_voice_path,
            )

        interval = report["confirmed_intervals"][0]
        dialogue = interval["dialogue_detection"]
        program_shares = [
            float(item["features"]["translation_voice_program_share"])
            for item in dialogue["windows"]
        ]
        dominance = [
            float(
                item["features"][
                    "translation_voice_vocal_to_backing_db"
                ]
            )
            for item in dialogue["windows"]
        ]
        self.assertGreater(min(program_shares), 0.80)
        self.assertLess(max(dominance), 6.0)
        self.assertEqual(dialogue["blocked_restoration_intervals"], [])
        self.assertTrue(report["restoration_segments"])

    def test_translation_vocal_dominance_hysteresis_is_conservative(
        self,
    ) -> None:
        rate = 8000
        seconds = 4
        silence = np.zeros(rate * seconds, dtype=np.float32)
        cfg = resolved_config(
            {
                "analysis_sample_rate": rate,
                "dialogue_window_sec": 1.5,
                "dialogue_step_sec": 0.5,
                "minimum_translation_voice_vocal_to_backing_db": 6.0,
                "translation_voice_vocal_to_backing_ambiguity_margin_db": 0.5,
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            sf.write(audio, silence, rate, subtype="FLOAT")

            def feature_values(dominance_db: float) -> dict:
                return {
                    "dialogue_score": 0.81,
                    "residual_dialogue_score": 0.80,
                    "nonmatching_share": 0.997,
                    "spectral_similarity": 0.01,
                    "speech_likeness": 0.45,
                    "speech_band_modulation": 0.40,
                    "spectral_flux": 0.20,
                    "spectral_stability": 0.40,
                    "activity_variation": 0.50,
                    "matched_gain": 1.0,
                    "dubbed_stem_rms_db": -22.0,
                    "translation_voice_available": True,
                    "translation_voice_rms_db": -22.0,
                    "translation_voice_share": 1.0,
                    "dubbed_program_rms_db": -19.0,
                    "translation_voice_program_share": 0.72,
                    "translation_voice_vocal_to_backing_db": dominance_db,
                    "translation_voice_speech_likeness": 0.45,
                    "translation_voice_speech_band_modulation": 0.40,
                    "translation_voice_spectral_flux": 0.20,
                    "translation_voice_spectral_stability": 0.40,
                    "translation_voice_activity_variation": 0.50,
                    "translation_voice_evidence_score": 0.81,
                }

            outcomes: dict[float, tuple[dict, list[dict]]] = {}
            for dominance_db in (5.49, 5.86, 6.02):
                confirmed = [{"start_sec": 0.0, "end_sec": 4.0}]
                with (
                    patch(
                        "experiments.paired_reference_cancel.song_protection."
                        "_track_music_similarity",
                        return_value={
                            "music_similarity": 0.90,
                            "matched_window_share": 1.0,
                        },
                    ),
                    patch(
                        "experiments.paired_reference_cancel.song_protection."
                        "_stem_difference_features",
                        return_value=feature_values(dominance_db),
                    ),
                ):
                    segments = _build_restoration_plan(
                        confirmed,
                        audio,
                        audio,
                        audio,
                        audio,
                        audio,
                        audio,
                        cfg,
                    )
                outcomes[dominance_db] = (confirmed[0], segments)

        below, below_segments = outcomes[5.49]
        self.assertEqual(
            below["dialogue_detection"]["blocked_restoration_intervals"],
            [],
        )
        self.assertTrue(below_segments)

        ambiguous, ambiguous_segments = outcomes[5.86]
        ambiguous_blocks = ambiguous["dialogue_detection"][
            "blocked_restoration_intervals"
        ]
        self.assertEqual(ambiguous_segments, [])
        self.assertTrue(ambiguous_blocks)
        self.assertTrue(
            all(
                item["decision"] == "keep_processed_low_confidence"
                for item in ambiguous_blocks
            )
        )
        self.assertAlmostEqual(
            ambiguous["dialogue_detection"][
                "translation_voice_vocal_to_backing_db_ambiguity_floor"
            ],
            5.5,
        )
        self.assertTrue(
            all(
                item["evidence"][
                    "translation_voice_vocal_to_backing_ambiguous"
                ]
                for item in ambiguous["dialogue_detection"]["windows"]
            )
        )

        confirmed_dialogue, confirmed_segments = outcomes[6.02]
        self.assertEqual(confirmed_segments, [])
        self.assertTrue(
            confirmed_dialogue["dialogue_detection"][
                "full_interval_translation_block"
            ]
        )
        self.assertEqual(
            confirmed_dialogue["dialogue_detection"][
                "confirmed_intervals"
            ][0]["decision"],
            "keep_processed_translation",
        )

    def test_sustained_translation_blocks_entire_preconfirmed_song(
        self,
    ) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        music = _music(time)
        vocal = _vocal(time)
        translated_dialogue = np.zeros_like(time)
        for start in np.arange(0.3, 23.0, 1.4):
            translated_dialogue += 3.0 * _spoken_phrase(
                time,
                float(start),
                min(float(start + 1.0), float(seconds)),
            )
        original_mix = music + vocal
        dubbed_mix = original_mix + translated_dialogue
        dubbed_speech = vocal + translated_dialogue
        processed = 0.60 * music + translated_dialogue
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            translation_voice_path = root / "translation_voice.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            sf.write(
                translation_voice_path,
                translated_dialogue,
                rate,
                subtype="FLOAT",
            )
            paths = self._write_case(
                root,
                dubbed_mix,
                dubbed_speech,
                music,
                processed,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                    # This test isolates dialogue blocking after song
                    # confirmation. The synthetic sustained dialogue masks
                    # the earlier vocal-removal metric much more than the
                    # short dialogue islands found in production material.
                    "minimum_relative_vocal_removal_db": -2.0,
                    "minimum_removed_vocal_share": 0.0,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
                translation_voice_stem=translation_voice_path,
            )

        self.assertTrue(report["confirmed_intervals"])
        interval = report["confirmed_intervals"][0]
        dialogue = interval["dialogue_detection"]
        self.assertTrue(dialogue["full_interval_translation_block"])
        self.assertGreaterEqual(
            float(dialogue["translation_voice_window_share"]),
            float(dialogue["translation_voice_majority_threshold"]),
        )
        self.assertEqual(interval["restoration_segments"], [])
        self.assertEqual(report["restoration_segments"], [])
        self.assertEqual(
            dialogue["confirmed_intervals"][0]["decision"],
            "keep_processed_translation",
        )

    def test_different_music_never_falls_back_to_full_dubbed_mix(self) -> None:
        rate = 8000
        seconds = 24
        time = np.arange(rate * seconds, dtype=np.float32) / rate
        english_music = _music(time)
        vocal = _vocal(time)
        dubbed_music = (
            0.14 * np.sin(2 * np.pi * 73.0 * time)
            + 0.10 * np.sin(2 * np.pi * 1493.0 * time)
            + 0.08 * np.sin(2 * np.pi * 3700.0 * time)
        ).astype(np.float32)
        original_mix = english_music + vocal
        dubbed_mix = dubbed_music + vocal
        processed = dubbed_music + 0.03 * vocal
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_path = root / "original.wav"
            original_speech_path = root / "original_speech.wav"
            sf.write(original_path, original_mix, rate, subtype="FLOAT")
            sf.write(original_speech_path, vocal, rate, subtype="FLOAT")
            paths = self._write_case(
                root,
                dubbed_mix,
                vocal,
                english_music,
                processed,
                rate,
            )
            report = detect_song_intervals(
                original_path,
                *paths,
                {
                    "analysis_sample_rate": rate,
                    "window_sec": 4.0,
                    "step_sec": 2.0,
                    "minimum_interval_sec": 8.0,
                    "minimum_confidence": 0.74,
                },
                classifier=_FixedClassifier(SONG_SCORES, rate),
                original_speech_stem=original_speech_path,
            )

        self.assertTrue(report["confirmed_intervals"])
        interval = report["confirmed_intervals"][0]
        self.assertEqual(interval["restoration_source"], "none")
        self.assertEqual(interval["restoration_segments"], [])
        self.assertEqual(report["restoration_segments"], [])
        self.assertLess(
            float(interval["track_comparison"]["music_similarity"]),
            0.78,
        )


if __name__ == "__main__":
    unittest.main()
