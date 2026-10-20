from __future__ import annotations

import copy
import json
import tempfile
import unittest
import inspect
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.application_pipeline import (
    _coincident_speech_protection_identity,
    _compatible_preparation_phases,
    _load_valid_coincident_report,
    _split_demo_files,
    build_full_application,
    prepare_application,
)
from experiments.paired_reference_cancel.coincident_speech_protection import (
    ALGORITHM,
    SCHEMA_VERSION,
    _output_level_gain_cap,
    apply_coincident_speech_protection,
    detect_coincident_speech,
    protect_coincident_speech,
    resolved_config,
    validate_detection_report,
)


def _speech_triplet(
    *,
    rate: int = 16000,
    duration: float = 10.0,
    event_start: float = 4.1,
    event_end: float = 4.7,
    remove_russian: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(int(round(rate * duration)), dtype=np.float64) / rate
    reference_envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * time) ** 2
    reference = 0.12 * reference_envelope * (
        np.sin(2 * np.pi * 181.0 * time)
        + 0.35 * np.sin(2 * np.pi * 362.0 * time)
    )
    russian = 0.09 * (
        0.60 + 0.40 * np.sin(2 * np.pi * 3.3 * time + 0.4) ** 2
    ) * (
        np.sin(2 * np.pi * 237.0 * time + 0.7)
        + 0.28 * np.sin(2 * np.pi * 474.0 * time + 0.2)
    )
    active = (time > 0.4) & (time < duration - 0.4)
    reference *= active
    russian *= active
    mixture = reference + russian
    processed = russian.copy()
    event = (time >= event_start) & (time < event_end)
    if remove_russian:
        processed[event] = 0.0
    return (
        time,
        reference.astype(np.float32),
        russian.astype(np.float32),
        mixture.astype(np.float32),
        processed.astype(np.float32),
    )


def _write(path: Path, values: np.ndarray, rate: int) -> Path:
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype="FLOAT")
    return path


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


class CoincidentSpeechProtectionTests(unittest.TestCase):
    def _paths(
        self,
        root: Path,
        reference: np.ndarray,
        mixture: np.ndarray,
        processed: np.ndarray,
        rate: int,
    ) -> tuple[Path, Path, Path]:
        return (
            _write(root / "mixture.wav", mixture, rate),
            _write(root / "reference.wav", reference, rate),
            _write(root / "processed.wav", processed, rate),
        )

    def test_short_coincident_word_is_detected_and_error_decreases(self) -> None:
        rate = 16000
        time, reference, russian, mixture, processed = _speech_triplet(rate=rate)
        event = (time >= 4.1) & (time < 4.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixture_path, reference_path, processed_path = self._paths(
                root, reference, mixture, processed, rate
            )
            output = root / "protected.wav"
            report_path = root / "report.json"
            report = protect_coincident_speech(
                mixture_path,
                reference_path,
                processed_path,
                output,
                {},
                report_path=report_path,
            )
            protected, protected_rate = sf.read(
                output, dtype="float32", always_2d=False
            )

        before_error = float(
            np.sqrt(np.mean((processed[event] - russian[event]) ** 2))
        )
        after_error = float(
            np.sqrt(np.mean((protected[event] - russian[event]) ** 2))
        )
        self.assertEqual(protected_rate, rate)
        self.assertEqual(protected.shape, processed.shape)
        self.assertEqual(report["algorithm"], ALGORITHM)
        self.assertEqual(report["summary"]["confirmed_count"], 1)
        self.assertTrue(report["summary"]["modified"])
        self.assertLess(after_error, before_error * 0.65)
        restored_delta = protected[event] - processed[event]
        reference_similarity = abs(
            float(np.dot(restored_delta, reference[event]))
        ) / (
            float(np.linalg.norm(restored_delta))
            * float(np.linalg.norm(reference[event]))
            + 1e-12
        )
        russian_similarity = abs(
            float(np.dot(restored_delta, russian[event]))
        ) / (
            float(np.linalg.norm(restored_delta))
            * float(np.linalg.norm(russian[event]))
            + 1e-12
        )
        # The waveform must come from D-HR, not from a scaled EN+RU removed
        # signal.  Otherwise the repair would audibly return English speech.
        self.assertLess(reference_similarity, 0.20)
        self.assertGreater(russian_similarity, 0.90)
        outside = ~((time >= 4.0) & (time < 4.8))
        np.testing.assert_allclose(
            protected[outside], processed[outside], atol=1e-7
        )

    def test_correctly_preserved_russian_dialogue_is_a_noop(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(
            rate=rate, remove_russian=False
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})
            original_bytes = paths[2].read_bytes()
            output_bytes = output.read_bytes()

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        self.assertFalse(report["summary"]["modified"])
        self.assertEqual(output_bytes, original_bytes)

    def test_correct_english_only_removal_is_not_restored(self) -> None:
        rate = 16000
        _, reference, _, _, _ = _speech_triplet(rate=rate)
        mixture = reference.copy()
        processed = np.zeros_like(reference)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        self.assertFalse(report["summary"]["modified"])

    def test_gain_only_english_change_is_explicitly_rejected(self) -> None:
        rate = 16000
        time, reference, _, _, _ = _speech_triplet(rate=rate)
        mixture = reference.copy()
        mixture[(time >= 3.0) & (time < 3.5)] *= 1.5
        processed = np.zeros_like(reference)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        reasons = {
            reason
            for candidate in report["rejected_candidates"]
            for reason in candidate["reasons"]
        }
        self.assertIn("reference_only_explains_removed", reasons)

    def test_dialogue_with_background_music_is_not_a_false_positive(self) -> None:
        rate = 16000
        time, reference, _, mixture, processed = _speech_triplet(
            rate=rate, remove_russian=False
        )
        music = (
            0.025 * np.sin(2 * np.pi * 110.0 * time)
            + 0.018 * np.sin(2 * np.pi * 330.0 * time)
        ).astype(np.float32)
        mixture = mixture + music
        processed = processed + music
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        self.assertFalse(report["summary"]["modified"])

    def test_sustained_vocal_loss_is_left_for_song_protection(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(
            rate=rate, event_start=3.0, event_end=6.2
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        reasons = {
            reason
            for candidate in report["rejected_candidates"]
            for reason in candidate["reasons"]
        }
        self.assertTrue(
            {"too_long", "sustained_vocal_activity"} & reasons
        )

    def test_preview_scope_excludes_events_outside_selected_scenes(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(
                *paths,
                output,
                {},
                analysis_intervals=[(0.5, 3.0), (6.0, 9.0)],
            )

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        self.assertAlmostEqual(
            report["summary"]["analysed_duration_sec"], 5.5, places=3
        )

    def test_one_decision_can_protect_a_different_rate_second_pass(self) -> None:
        source_rate = 16000
        target_rate = 24000
        time, reference, _, mixture, processed = _speech_triplet(
            rate=source_rate
        )
        target = audio_io.resample(
            (processed * 0.8)[:, None], source_rate, target_rate
        )[:, 0]
        target_time = np.arange(target.size) / target_rate
        event = (target_time >= 4.1) & (target_time < 4.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixture_path, reference_path, processed_path = self._paths(
                root, reference, mixture, processed, source_rate
            )
            target_path = _write(
                root / "second_pass.wav", target, target_rate
            )
            output = root / "second_pass_protected.wav"
            report = detect_coincident_speech(
                mixture_path, reference_path, processed_path, {}
            )
            report = apply_coincident_speech_protection(
                mixture_path,
                reference_path,
                processed_path,
                target_path,
                output,
                report,
                {},
            )
            protected, output_rate = sf.read(
                output, dtype="float32", always_2d=False
            )

        self.assertEqual(output_rate, target_rate)
        self.assertEqual(protected.shape, target.shape)
        self.assertGreater(
            float(np.sqrt(np.mean(protected[event] ** 2))),
            float(np.sqrt(np.mean(target[event] ** 2))) + 0.01,
        )
        self.assertEqual(report["summary"]["applied_count"], 1)

    def test_event_on_a_calibration_seam_is_one_undivided_restoration(
        self,
    ) -> None:
        rate = 16000
        cfg = resolved_config({})
        seam = float(cfg["analysis_chunk_sec"])
        time, reference, russian, mixture, processed = _speech_triplet(
            rate=rate,
            duration=seam + 15.0,
            event_start=seam - 0.2,
            event_end=seam + 0.4,
        )
        event = (time >= seam - 0.2) & (time < seam + 0.4)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})
            protected = sf.read(output, dtype="float32", always_2d=False)[0]

        # One calibration block owns the word; the other one sees it inside its
        # overlap and steps aside, so it is neither halved nor applied twice.
        self.assertEqual(report["summary"]["confirmed_count"], 1)
        self.assertEqual(len(report["application"]["segments"]), 1)
        interval = report["confirmed_intervals"][0]
        self.assertLessEqual(interval["detected_start_sec"], seam - 0.15)
        self.assertGreaterEqual(interval["detected_end_sec"], seam + 0.35)
        blocks = {
            int(block["index"]): block
            for block in report["calibration_blocks"]
        }
        owner = blocks[int(interval["calibration_block_index"])]
        self.assertLess(owner["start_sec"], owner["core_start_sec"] + 1e-9)
        self.assertGreater(owner["end_sec"], owner["core_end_sec"] - 1e-9)
        before_error = _rms(processed[event] - russian[event])
        after_error = _rms(protected[event] - russian[event])
        self.assertLess(after_error, before_error * 0.65)

    def test_two_close_words_in_one_block_are_both_repaired(self) -> None:
        rate = 16000
        # Long enough that ``maximum_restored_fraction`` is not the limit.
        time, reference, russian, mixture, processed = _speech_triplet(
            rate=rate, duration=60.0
        )
        processed = russian.copy()
        first = (time >= 4.10) & (time < 4.45)
        second = (time >= 4.60) & (time < 4.95)
        processed[first] = 0.0
        processed[second] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})
            protected = sf.read(output, dtype="float32", always_2d=False)[0]

        reasons = {
            reason
            for candidate in report["rejected_candidates"]
            for reason in candidate["reasons"]
        }
        self.assertNotIn("overlapping_candidate", reasons)
        self.assertGreaterEqual(report["summary"]["confirmed_count"], 2)
        for window in (first, second):
            self.assertLess(
                _rms(protected[window] - russian[window]),
                _rms(processed[window] - russian[window]) * 0.65,
            )

    def test_preview_scopes_do_not_change_global_safety_decisions(self) -> None:
        rate = 16000
        time, reference, russian, mixture, processed = _speech_triplet(
            rate=rate,
            duration=120.0,
            event_start=10.0,
            event_end=10.6,
        )
        second = (time >= 80.0) & (time < 80.6)
        processed = russian.copy()
        processed[(time >= 10.0) & (time < 10.6)] = 0.0
        processed[second] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            whole = detect_coincident_speech(*paths, {})
            preview = detect_coincident_speech(
                *paths,
                {},
                analysis_intervals=[(5.0, 16.0), (75.0, 86.0)],
            )

        whole_starts = [
            round(item["detected_start_sec"], 1)
            for item in whole["confirmed_intervals"]
            if 5.0 <= item["detected_start_sec"] <= 86.0
        ]
        preview_starts = [
            round(item["detected_start_sec"], 1)
            for item in preview["confirmed_intervals"]
        ]
        self.assertEqual(whole_starts, [10.0, 80.0])
        self.assertEqual(preview_starts, whole_starts)
        self.assertTrue(
            preview["summary"]["safety_advisories"][
                "restoration_duration_threshold_exceeded"
            ]
        )
        self.assertNotIn(
            "restoration_duration_guard",
            preview["summary"]["rejection_reasons"],
        )

    def test_sustained_density_is_advisory_not_a_scope_dependent_veto(
        self,
    ) -> None:
        rate = 16000
        time, reference, russian, mixture, _ = _speech_triplet(
            rate=rate,
            duration=15.0,
            event_start=6.0,
            event_end=7.0,
        )
        processed = russian.copy()
        processed[(time >= 6.0) & (time < 7.0)] = 0.0
        processed[(time >= 7.2) & (time < 8.2)] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            whole = detect_coincident_speech(*paths, {})
            preview = detect_coincident_speech(
                *paths,
                {},
                analysis_intervals=[(0.5, 7.05)],
            )

        self.assertEqual(whole["summary"]["confirmed_count"], 2)
        self.assertEqual(preview["summary"]["confirmed_count"], 1)
        self.assertAlmostEqual(
            whole["confirmed_intervals"][0]["detected_start_sec"],
            preview["confirmed_intervals"][0]["detected_start_sec"],
            places=3,
        )
        self.assertTrue(
            whole["summary"]["safety_advisories"][
                "sustained_vocal_threshold_exceeded"
            ]
        )

    def test_reference_activity_over_the_event_is_measured_and_enforced(
        self,
    ) -> None:
        rate = 16000
        time, reference, russian, _, _ = _speech_triplet(rate=rate)
        # The English reference falls silent inside the removed region, so
        # that slice of the loss is not a double-talk collision at all.
        reference = reference.copy()
        reference[(time >= 4.38) & (time < 4.47)] = 0.0
        mixture = (reference + russian).astype(np.float32)
        processed = russian.copy()
        processed[(time >= 4.1) & (time < 4.7)] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            permissive = detect_coincident_speech(*paths, {})
            strict = detect_coincident_speech(
                *paths, {"minimum_reference_active_share": 0.95}
            )

        self.assertEqual(permissive["summary"]["confirmed_count"], 1)
        share = permissive["confirmed_intervals"][0]["features"][
            "reference_active_share"
        ]
        # The old report measured this over frames that were selected *because*
        # the reference was active, so it could only ever read 1.0 and the
        # configured threshold was dead weight.
        self.assertLess(share, 0.95)
        self.assertGreater(share, 0.75)
        self.assertEqual(strict["summary"]["confirmed_count"], 0)
        self.assertIn(
            "reference_not_active_during_event",
            strict["summary"]["rejection_reasons"],
        )

    def test_english_word_over_silent_russian_is_not_restored(self) -> None:
        rate = 16000
        time, reference, russian, _, _ = _speech_triplet(rate=rate)
        # Russian is present everywhere except the event: the separator did
        # exactly the right thing there and nothing may be handed back.
        event = (time >= 4.1) & (time < 4.7)
        russian = russian.copy()
        russian[event] = 0.0
        mixture = (reference + russian).astype(np.float32)
        processed = russian.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})
            original_bytes = paths[2].read_bytes()
            output_bytes = output.read_bytes()

        self.assertEqual(report["summary"]["confirmed_count"], 0)
        self.assertFalse(report["summary"]["modified"])
        self.assertEqual(output_bytes, original_bytes)

    def test_restoration_never_reaches_the_untouched_input_mixture(
        self,
    ) -> None:
        rate = 16000
        time, reference, russian, mixture, processed = _speech_triplet(rate=rate)
        event = (time >= 4.1) & (time < 4.7)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(*paths, output, {})
            protected = sf.read(output, dtype="float32", always_2d=False)[0]

        self.assertTrue(report["summary"]["modified"])
        # Closer to the Russian dub than the damaged input was, and still far
        # away from simply pasting the English+Russian mixture back in.
        self.assertLess(
            _rms(protected[event] - russian[event]),
            _rms(processed[event] - russian[event]),
        )
        self.assertGreater(
            _rms(protected[event] - mixture[event]),
            0.5 * _rms(reference[event]),
        )
        self.assertLessEqual(_rms(protected[event]), _rms(mixture[event]))

    def test_output_level_guard_can_veto_the_whole_correction(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            output = root / "protected.wav"
            report = protect_coincident_speech(
                *paths, output, {"maximum_output_to_input_rms": 0.05}
            )
            protected = sf.read(output, dtype="float32", always_2d=False)[0]

        self.assertEqual(report["summary"]["confirmed_count"], 1)
        self.assertFalse(report["summary"]["modified"])
        self.assertEqual(report["summary"]["applied_count"], 0)
        self.assertEqual(report["summary"]["level_limited_count"], 1)
        segment = report["application"]["segments"][0]
        self.assertTrue(segment["limited_by_output_level_guard"])
        self.assertEqual(segment["recovery_gain"], 0.0)
        np.testing.assert_allclose(protected, processed, atol=1e-7)

    def test_output_level_guard_never_returns_an_unsafe_upper_cap(self) -> None:
        mixture = np.ones(100, dtype=np.float32)
        target = np.full(100, 2.0, dtype=np.float32)
        correction = np.full(100, -2.0, dtype=np.float32)

        cap = _output_level_gain_cap(
            mixture,
            target,
            correction,
            0,
            correction.size,
            1.0,
        )

        # The quadratic has a safe interval only after gain 0.5.  Returning its
        # upper root as an ordinary cap let a smaller caller gain remain louder
        # than the input.  An already-over-limit target must therefore veto.
        self.assertEqual(cap, 0.0)

    def test_output_level_guard_checks_every_channel(self) -> None:
        mixture = np.tile(
            np.asarray([[1.0, 0.10]], dtype=np.float32), (100, 1)
        )
        target = np.tile(
            np.asarray([[0.50, 0.20]], dtype=np.float32), (100, 1)
        )
        correction = np.full((100, 2), 0.25, dtype=np.float32)

        cap = _output_level_gain_cap(
            mixture,
            target,
            correction,
            0,
            correction.shape[0],
            1.0,
        )

        # The quiet right channel is already above its own source level even
        # though the mono average looks safe.
        self.assertEqual(cap, 0.0)

    def test_stereo_target_at_a_higher_rate_keeps_shape_and_gains_energy(
        self,
    ) -> None:
        source_rate = 16000
        target_rate = 48000
        _, reference, _, mixture, processed = _speech_triplet(rate=source_rate)
        stereo_mixture = np.stack([mixture, mixture * 0.85], axis=1)
        target = audio_io.resample(
            np.stack([processed, processed * 0.85], axis=1),
            source_rate,
            target_rate,
        )
        target_time = np.arange(target.shape[0]) / target_rate
        event = (target_time >= 4.1) & (target_time < 4.7)
        quiet = (target_time >= 7.0) & (target_time < 8.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mixture_path = _write(
                root / "mixture.wav", stereo_mixture, source_rate
            )
            reference_path = _write(
                root / "reference.wav", reference, source_rate
            )
            processed_path = _write(
                root / "processed.wav", processed, source_rate
            )
            target_path = _write(root / "target.wav", target, target_rate)
            output = root / "protected.wav"
            report = detect_coincident_speech(
                mixture_path, reference_path, processed_path, {}
            )
            report = apply_coincident_speech_protection(
                mixture_path,
                reference_path,
                processed_path,
                target_path,
                output,
                report,
                {},
            )
            protected, output_rate = sf.read(
                output, dtype="float32", always_2d=True
            )

        self.assertEqual(output_rate, target_rate)
        self.assertEqual(protected.shape, target.shape)
        self.assertEqual(report["summary"]["applied_count"], 1)
        for channel in range(protected.shape[1]):
            self.assertGreater(
                _rms(protected[event, channel]),
                _rms(target[event, channel]) + 0.01,
            )
        np.testing.assert_allclose(
            protected[quiet], target[quiet], atol=1e-6
        )

    def test_unreadable_cached_plan_leaves_the_target_untouched(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            good = detect_coincident_speech(*paths, {})
            self.assertEqual(good["summary"]["confirmed_count"], 1)
            unusable_plans = [
                {},
                {"confirmed_intervals": good["confirmed_intervals"]},
                {"confirmed_intervals": ["not-an-interval", 7]},
                {
                    **good,
                    "calibration_blocks": [],
                    "summary": dict(good["summary"]),
                },
            ]
            for index, plan in enumerate(unusable_plans):
                output = root / f"protected_{index}.wav"
                report = apply_coincident_speech_protection(
                    *paths,
                    paths[2],
                    output,
                    copy.deepcopy(plan),
                    {},
                )
                self.assertFalse(
                    report["summary"]["modified"],
                    msg=f"unusable plan {index} modified the audio",
                )
                self.assertEqual(
                    report["summary"]["applied_count"],
                    0,
                    msg=f"unusable plan {index} reported a repair",
                )
                self.assertEqual(
                    output.read_bytes(),
                    paths[2].read_bytes(),
                    msg=f"unusable plan {index} rewrote the target",
                )
            # A complete plan that simply lost its summary block still works
            # and rebuilds the summary instead of raising KeyError.
            without_summary = {
                key: value
                for key, value in copy.deepcopy(good).items()
                if key != "summary"
            }
            output = root / "protected_no_summary.wav"
            recovered = apply_coincident_speech_protection(
                *paths, paths[2], output, without_summary, {}
            )
            self.assertTrue(recovered["summary"]["modified"])
            self.assertEqual(recovered["summary"]["applied_count"], 1)
            # A plan whose transfer curve no longer matches the reported
            # frequency axis must be refused, not interpolated blindly.
            mismatched = json.loads(json.dumps(good))
            for block in mismatched["calibration_blocks"]:
                if block.get("transfer_power"):
                    block["transfer_power"] = block["transfer_power"][:-3]
            output = root / "protected_mismatch.wav"
            report = apply_coincident_speech_protection(
                *paths, paths[2], output, mismatched, {}
            )

        self.assertFalse(report["summary"]["modified"])
        self.assertEqual(
            report["summary"]["application_reason"],
            "invalid_calibration_transfer",
        )

    def test_actual_corrupt_cached_json_is_rejected_before_application(
        self,
    ) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            report_path = root / "cached_report.json"
            report_path.write_text('{"summary":', encoding="utf-8")

            truncated = _load_valid_coincident_report(
                report_path, *paths, {}
            )
            report_path.write_text("[]", encoding="utf-8")
            wrong_top_level = _load_valid_coincident_report(
                report_path, *paths, {}
            )
            output = root / "protected.wav"
            applied = apply_coincident_speech_protection(
                *paths,
                paths[2],
                output,
                [],
                {},
            )

        self.assertIsNone(truncated)
        self.assertIsNone(wrong_top_level)
        self.assertFalse(applied["summary"]["modified"])
        self.assertEqual(
            applied["summary"]["application_reason"],
            "invalid_report_type",
        )

    def test_cached_plan_provenance_must_match_current_inputs(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            report = detect_coincident_speech(*paths, {})
            expected_config = copy.deepcopy(report["config"])
            self.assertEqual(
                validate_detection_report(report, *paths, expected_config),
                (True, "valid"),
            )
            mutations = [
                ("schema_version", SCHEMA_VERSION - 1),
                ("algorithm", "other_algorithm"),
                ("config", {**report["config"], "minimum_confidence": 0.99}),
                (
                    "sources",
                    {
                        **report["sources"],
                        "english_reference": {
                            **report["sources"]["english_reference"],
                            "size": report["sources"]["english_reference"][
                                "size"
                            ]
                            + 1,
                        },
                    },
                ),
            ]
            for key, value in mutations:
                stale = copy.deepcopy(report)
                stale[key] = value
                valid, reason = validate_detection_report(
                    stale, *paths, expected_config
                )
                self.assertFalse(valid, msg=f"{key} was accepted")
                self.assertNotEqual(reason, "valid")

            oversized = copy.deepcopy(report)
            oversized["confirmed_intervals"][0]["end_sec"] = 1e9
            valid, reason = validate_detection_report(
                oversized, *paths, expected_config
            )
            self.assertFalse(valid)
            self.assertEqual(reason, "invalid_confirmed_interval")

    def test_zero_effective_correction_is_reported_as_a_noop(self) -> None:
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            report = detect_coincident_speech(*paths, {})
            target_path = _write(root / "already_full.wav", mixture, rate)
            output = root / "protected.wav"
            applied = apply_coincident_speech_protection(
                *paths,
                target_path,
                output,
                report,
                {},
            )
            target_bytes = target_path.read_bytes()
            output_bytes = output.read_bytes()

        self.assertEqual(applied["summary"]["confirmed_count"], 1)
        self.assertFalse(applied["summary"]["modified"])
        self.assertEqual(applied["summary"]["applied_count"], 0)
        self.assertEqual(output_bytes, target_bytes)

    def test_whole_file_scope_matches_an_explicit_full_range_scope(
        self,
    ) -> None:
        """The preview passes scopes, the full build does not - same core."""
        rate = 16000
        _, reference, _, mixture, processed = _speech_triplet(rate=rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._paths(root, reference, mixture, processed, rate)
            implicit = detect_coincident_speech(*paths, {})
            duration = float(
                implicit["sources"]["first_pass_result"]["duration_sec"]
            )
            explicit = detect_coincident_speech(
                *paths, {}, analysis_intervals=[(0.0, duration)]
            )

        def decisions(report: dict) -> list[tuple]:
            return [
                (
                    round(item["start_sec"], 6),
                    round(item["end_sec"], 6),
                    round(item["confidence"], 6),
                )
                for item in report["confirmed_intervals"]
            ]

        self.assertEqual(decisions(implicit), decisions(explicit))
        self.assertEqual(
            implicit["summary"]["rejection_reasons"],
            explicit["summary"]["rejection_reasons"],
        )

    def test_configuration_is_bounded(self) -> None:
        cfg = resolved_config(
            {
                "analysis_sample_rate": 100,
                "hop_sec": 2.0,
                "minimum_confidence": -1.0,
                "maximum_recovery_gain": 4.0,
                "maximum_intervals_per_minute": 0,
                "analysis_chunk_sec": 20.0,
                "analysis_overlap_sec": 60.0,
                "maximum_output_to_input_rms": 9.0,
                "maximum_interval_sec": float("nan"),
            }
        )

        self.assertEqual(cfg["analysis_sample_rate"], 8000)
        self.assertLess(cfg["hop_sec"], cfg["frame_sec"])
        self.assertEqual(cfg["minimum_confidence"], 0.0)
        self.assertEqual(cfg["maximum_recovery_gain"], 1.0)
        self.assertEqual(cfg["maximum_intervals_per_minute"], 1)
        # The overlap can never grow past the chunk it belongs to, otherwise a
        # block would own audio that its neighbour already restored.
        self.assertEqual(cfg["analysis_overlap_sec"], 8.0)
        self.assertEqual(cfg["maximum_output_to_input_rms"], 1.0)
        self.assertTrue(np.isfinite(cfg["maximum_interval_sec"]))
        self.assertEqual(
            resolved_config({"analysis_overlap_sec": -5.0})[
                "analysis_overlap_sec"
            ],
            0.0,
        )


class CoincidentSpeechPipelineIntegrationTests(unittest.TestCase):
    def test_config_change_invalidates_every_phase_after_speech(self) -> None:
        phases = {
            "speech": {"status": "completed"},
            "subtraction": {"status": "completed"},
            "background": {"status": "completed"},
            "preview": {"status": "completed"},
        }
        previous = {
            "identity": {
                "source_signature": {"source": "same"},
                "checkpoint": {"path": "voice"},
                "background_checkpoint": {"path": "background"},
                "routing": {"band": "HIGH"},
                "speech_second_pass": {"enabled": False, "extractor": []},
                "song_protection": {"config": {"same": True}},
                "coincident_speech_protection": {
                    "schema_version": 1,
                    "config": {"minimum_confidence": 0.84},
                },
            },
            "phases": phases,
        }
        changed = {
            **previous["identity"],
            "coincident_speech_protection": {
                "schema_version": 1,
                "config": {"minimum_confidence": 0.90},
            },
        }

        compatible = _compatible_preparation_phases(previous, changed)

        self.assertEqual(set(compatible), {"speech"})

    def test_preview_is_scene_limited_and_full_uses_the_same_core(self) -> None:
        preview_source = inspect.getsource(prepare_application)
        full_source = inspect.getsource(build_full_application)

        self.assertIn(
            "coincident_speech_protection.protect_coincident_speech(",
            preview_source,
        )
        self.assertIn(
            "analysis_intervals=preview_analysis_ranges",
            preview_source,
        )
        self.assertIn(
            "coincident_speech_protection.protect_coincident_speech(",
            full_source,
        )
        self.assertGreaterEqual(
            preview_source.count(
                "coincident_speech_protection.protect_coincident_speech("
            ),
            2,
        )
        self.assertGreaterEqual(
            full_source.count(
                "coincident_speech_protection.protect_coincident_speech("
            ),
            2,
        )
        self.assertIn(
            '"reason": "direct_mix_has_no_separate_speech_stem"',
            full_source,
        )
        # The full build analyses the whole film; only the preview narrows the
        # scope down to the selected scenes.
        self.assertNotIn("analysis_intervals=", full_source)

    def test_second_pass_builds_a_fresh_plan_from_its_own_raw_audio(
        self,
    ) -> None:
        """R2 must be diagnosed from P2, never copied from the first pass."""
        preview = "".join(inspect.getsource(prepare_application).split())
        full = "".join(inspect.getsource(build_full_application).split())

        self.assertIn(
            "protect_coincident_speech("
            "selected_dubbed_speech,reference_voice,raw_output_voice,"
            "protected_output_voice,",
            preview,
        )
        self.assertIn(
            "protect_coincident_speech("
            "dubbed_speech,reference_for_subtraction,"
            "raw_second_pass_voice,second_pass_voice,",
            full,
        )
        for source in (preview, full):
            self.assertIn("coincident_cfg", source)
            self.assertIn("coincident_module_path", source)
        self.assertNotIn("first_report_path", preview)
        self.assertNotIn("coincident_preview_report_data", preview)
        self.assertNotIn("json.loads(json.dumps(first_report", preview)
        self.assertNotIn("json.loads(json.dumps(coincident_report", full)

    def test_every_threshold_is_part_of_the_cache_identity(self) -> None:
        identity = _coincident_speech_protection_identity(
            {"minimum_confidence": 0.9}
        )
        defaults = _coincident_speech_protection_identity({})

        self.assertEqual(identity["algorithm"], ALGORITHM)
        self.assertEqual(identity["config"], resolved_config({"minimum_confidence": 0.9}))
        self.assertNotEqual(identity, defaults)
        for key in (
            "analysis_overlap_sec",
            "maximum_output_to_input_rms",
            "minimum_reference_active_share",
        ):
            self.assertIn(key, identity["config"])
            changed = _coincident_speech_protection_identity(
                {key: defaults["config"][key] * 0.5}
            )
            self.assertNotEqual(
                changed,
                defaults,
                msg=f"{key} does not invalidate cached results",
            )

    def test_preview_quality_metrics_use_raw_model_output(self) -> None:
        source = inspect.getsource(_split_demo_files)

        self.assertIn(
            '"model_ru_voice_before_coincident_protection"',
            source,
        )
        self.assertIn(
            '"model_ru_voice_algorithmic_before_coincident_protection"',
            source,
        )

    def test_portable_config_names_the_dsp_algorithm_and_no_new_model(
        self,
    ) -> None:
        config_path = (
            Path(__file__).parents[1]
            / "experiments"
            / "paired_reference_cancel"
            / "config.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        protection = config["coincident_speech_protection"]

        self.assertTrue(protection["enabled"])
        self.assertEqual(protection["algorithm"], ALGORITHM)
        self.assertNotIn("model_path", protection)
        self.assertNotIn("checkpoint", protection)


if __name__ == "__main__":
    unittest.main()
