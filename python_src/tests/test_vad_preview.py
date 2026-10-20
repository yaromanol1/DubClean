from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import vad_preview


class VadPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "fixture.wav"
        # Deliberately use 8 kHz: the VAD path must resample different sources.
        sf.write(self.path, np.zeros(8_000 * 60, dtype=np.float32), 8_000)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def probabilities(audio: np.ndarray) -> np.ndarray:
        return np.full(max(1, int(np.ceil(len(audio) / 512))), 0.82, dtype=np.float32)

    def analyse(self, intervals, *, duration=20.0, allowed=None):
        zone = {item["zone"]: item for item in vad_preview.build_analysis_zones(duration)}["start"]
        return vad_preview.analyse_zone(
            self.path,
            zone,
            allowed,
            probability_reader=self.probabilities,
            timestamp_reader=lambda _audio, _threshold: intervals,
        )

    def test_requested_zone_names_and_ninety_percent_bound(self) -> None:
        zones = vad_preview.build_analysis_zones(900.0)
        self.assertEqual([item["zone"] for item in zones], ["start", "25_percent", "50_percent", "75_percent", "90_percent"])
        self.assertEqual(zones[0]["analysis_start_sec"], 0.0)
        self.assertEqual(zones[-1]["analysis_end_sec"], 900.0)
        self.assertGreaterEqual(zones[-1]["analysis_start_sec"], 0.0)

    def test_short_file_clamps_all_five_zones(self) -> None:
        for zone in vad_preview.build_analysis_zones(75.0):
            self.assertEqual(zone["analysis_start_sec"], 0.0)
            self.assertEqual(zone["analysis_end_sec"], 75.0)

    def test_dialogue_at_start_end_and_percent_zones(self) -> None:
        # Covers a start utterance, a late utterance, and the same deterministic
        # selector used for 25/50/75/90 windows.
        for label, intervals in {
            "start": [(0.2, 7.0)],
            "end_of_first_three_minutes": [(11.0, 19.5)],
            "25_percent": [(4.0, 15.0)],
            "50_percent": [(3.0, 17.0)],
            "75_percent": [(6.0, 19.0)],
            "90_percent": [(1.0, 12.0)],
        }.items():
            with self.subTest(label=label):
                result = self.analyse(intervals)
                self.assertIn(result["status"], {"speech_selected", "speech_selected_low_confidence"})
                self.assertGreater(result["speech_duration_sec"], 0.0)
                self.assertLessEqual(result["preview_end_sec"], result["analysis_end_sec"])

    def test_music_silence_and_missing_speech_use_no_vad_selection(self) -> None:
        for label, intervals in {"music": [], "silence": [], "short_artifact": [(5.0, 5.1)]}.items():
            with self.subTest(label=label):
                result = self.analyse(intervals)
                self.assertEqual(result["status"], "no_speech_found")
                self.assertNotIn("preview_start_sec", result)

    def test_connected_dialogue_is_merged_and_repeatable(self) -> None:
        intervals = [(2.0, 5.0), (5.4, 9.0), (11.0, 17.0)]
        first = self.analyse(intervals)
        second = self.analyse(intervals)
        self.assertEqual(first["preview_start_sec"], second["preview_start_sec"])
        self.assertEqual(first["preview_end_sec"], second["preview_end_sec"])
        self.assertLess(first["speech_interval_count"], len(intervals) + 1)

    def test_quiet_short_and_long_replies_are_kept_with_metadata(self) -> None:
        for label, intervals in {"quiet": [(6.0, 8.0)], "short": [(8.0, 10.0)], "long": [(1.0, 19.0)]}.items():
            with self.subTest(label=label):
                result = self.analyse(intervals)
                self.assertIsNotNone(result.get("speech_region_start_sec"))
                self.assertIn("average_speech_confidence", result)
                self.assertIn("vad_parameters", result)

    def test_allowed_alignment_interval_is_respected(self) -> None:
        result = self.analyse([(6.0, 48.0)], duration=60.0, allowed=[(5.0, 50.0)])
        self.assertGreaterEqual(result["preview_start_sec"], 5.0)
        self.assertLessEqual(result["preview_end_sec"], 50.0)

    def test_positional_fallback_is_explicit_and_per_zone(self) -> None:
        from experiments.paired_reference_cancel.application_pipeline import _legacy_zone_fallback

        zone = vad_preview.build_analysis_zones(60.0)[0]
        alignment = {"segments": [{"id": 1, "dubbed_start": 0.0, "dubbed_end": 60.0, "original_start": 0.0, "speed_ratio": 1.0, "confidence": 0.9}]}
        fallback = _legacy_zone_fallback(alignment, zone)
        self.assertEqual(fallback["status"], "fallback_used")
        self.assertTrue(fallback["fallback"]["used"])
        self.assertEqual(fallback["zone"], "start")
        self.assertGreaterEqual(fallback["preview_start_sec"], zone["analysis_start_sec"])
        self.assertLessEqual(fallback["preview_end_sec"], zone["analysis_end_sec"])

    def test_missing_alignment_never_aborts_a_zone(self) -> None:
        from experiments.paired_reference_cancel.application_pipeline import _emergency_zone_fallback

        zone = vad_preview.build_analysis_zones(900.0)[-1]
        fallback = _emergency_zone_fallback(zone)
        self.assertEqual(fallback["status"], "fallback_used")
        self.assertEqual(fallback["fallback"]["reason"], "alignment_unavailable")
        self.assertGreaterEqual(fallback["preview_start_sec"], zone["analysis_start_sec"])
        self.assertLessEqual(fallback["preview_end_sec"], zone["analysis_end_sec"])
        self.assertEqual(fallback["alignment_confidence"], 0.0)

    def test_local_silero_package_contains_the_model(self) -> None:
        import importlib.resources as resources
        import silero_vad.data

        model = resources.files(silero_vad.data).joinpath("silero_vad.jit")
        self.assertTrue(model.is_file())

    def test_real_silero_marks_one_second_of_silence_as_no_speech(self) -> None:
        silence = Path(self.temp.name) / "actual_silence.wav"
        sf.write(silence, np.zeros(16_000, dtype=np.float32), 16_000)
        result = vad_preview.analyse_zone(silence, vad_preview.build_analysis_zones(1.0)[0])
        self.assertEqual(result["status"], "no_speech_found")
        self.assertFalse(result["fallback"]["used"])

    # ---- window geometry invariants (requirement 3) ------------------------
    def test_percent_zones_are_center_plus_minus_90_and_ninety_stays_in_file(self) -> None:
        for duration in (400.0, 720.0, 800.0, 1215.02, 3600.0):
            zones = {item["zone"]: item for item in vad_preview.build_analysis_zones(duration)}
            self.assertEqual(zones["start"]["analysis_start_sec"], 0.0)
            self.assertAlmostEqual(zones["start"]["analysis_end_sec"], min(180.0, duration), places=3)
            for name, pct in (("25_percent", 0.25), ("50_percent", 0.50), ("75_percent", 0.75), ("90_percent", 0.90)):
                start = zones[name]["analysis_start_sec"]
                end = zones[name]["analysis_end_sec"]
                center = duration * pct
                self.assertGreaterEqual(start, 0.0, name)
                self.assertLessEqual(end, duration + 1e-6, name)
                self.assertLessEqual(end - start, 180.0 + 1e-6, name)
                if duration >= 180.0:
                    self.assertAlmostEqual(end - start, 180.0, places=2)  # full ±90 window
                self.assertLessEqual(start - 1e-3, center, name)          # center inside window
                self.assertGreaterEqual(end + 1e-3, center, name)
            self.assertLessEqual(zones["90_percent"]["analysis_end_sec"], duration + 1e-6)

    def test_selection_interval_is_internally_consistent(self) -> None:
        result = self.analyse([(4.0, 16.0)])
        self.assertAlmostEqual(
            result["preview_end_sec"] - result["preview_start_sec"],
            result["preview_duration_sec"],
            places=3,
        )
        self.assertGreaterEqual(result["preview_start_sec"], result["analysis_start_sec"])
        self.assertLessEqual(result["preview_end_sec"], result["analysis_end_sec"])

    def test_selection_is_sample_rate_independent_with_same_intervals(self) -> None:
        path44 = Path(self.temp.name) / "f44.wav"
        sf.write(path44, np.zeros(44_100 * 60, dtype=np.float32), 44_100)
        intervals = [(4.0, 16.0)]
        base = self.analyse(intervals)
        zone = {item["zone"]: item for item in vad_preview.build_analysis_zones(20.0)}["start"]
        other = vad_preview.analyse_zone(
            path44, zone, None,
            probability_reader=self.probabilities,
            timestamp_reader=lambda _a, _t: intervals,
        )
        self.assertEqual(base["preview_start_sec"], other["preview_start_sec"])
        self.assertEqual(base["preview_end_sec"], other["preview_end_sec"])

    # ---- selection -> alignment mapping (requirement 3: intervals match) ----
    def test_candidate_maps_selection_to_original_timeline(self) -> None:
        from experiments.paired_reference_cancel.application_pipeline import _candidate_from_selection

        alignment = {
            "manual_correction_sec": 0.5,
            "segments": [{"id": 7, "dubbed_start": 100.0, "dubbed_end": 200.0, "original_start": 40.0, "speed_ratio": 1.0, "confidence": 0.8}],
        }
        selection = {"preview_start_sec": 130.0, "preview_end_sec": 160.0, "preview_duration_sec": 30.0, "zone": "50_percent"}
        candidate = _candidate_from_selection(alignment, selection)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["dubbed_start_sec"], 130.0)      # clip start == VAD start
        self.assertAlmostEqual(candidate["original_start_sec"], 70.5, places=3)  # 40 + 30 + 0.5
        self.assertEqual(candidate["duration_sec"], 30.0)
        self.assertEqual(candidate["alignment_segment_id"], 7)
        # a window crossing a segment boundary has no single-segment coverage
        crossing = {**selection, "preview_start_sec": 190.0, "preview_end_sec": 220.0}
        self.assertIsNone(_candidate_from_selection(alignment, crossing))

    # ---- fallback reason integrity (requirement 5) -------------------------
    def test_legacy_fallback_records_the_specific_reason(self) -> None:
        from experiments.paired_reference_cancel.application_pipeline import _legacy_zone_fallback

        zone = vad_preview.build_analysis_zones(60.0)[0]
        alignment = {"segments": [{"id": 1, "dubbed_start": 0.0, "dubbed_end": 60.0, "original_start": 0.0, "speed_ratio": 1.0, "confidence": 0.9}]}
        for reason in ("no_speech_found", "vad_error", "vad_window_unmapped", "zone_unavailable"):
            fallback = _legacy_zone_fallback(alignment, zone, reason=reason)
            self.assertEqual(fallback["fallback"]["reason"], reason)
            self.assertEqual(fallback["status"], "fallback_used")
            self.assertTrue(fallback["fallback"]["used"])

    def test_vad_error_in_one_zone_falls_back_without_aborting(self) -> None:
        from experiments.paired_reference_cancel import application_pipeline

        alignment = {
            "manual_correction_sec": 0.0,
            "segments": [{"id": 1, "dubbed_start": 0.0, "dubbed_end": 900.0, "original_start": 0.0, "speed_ratio": 1.0, "confidence": 0.9}],
        }
        original = vad_preview.analyse_zone

        def boom(_path, _zone, _allowed=None, **_kw):
            raise RuntimeError("Silero недоступен")

        vad_preview.analyse_zone = boom
        try:
            candidates = application_pipeline._vad_candidates(self.path, 900.0, alignment)
        finally:
            vad_preview.analyse_zone = original
        self.assertEqual(len(candidates), 5)
        for candidate in candidates:
            self.assertTrue((candidate.get("fallback") or {}).get("used"))
            self.assertEqual((candidate.get("fallback") or {}).get("reason"), "vad_error")
            self.assertIn("dubbed_start_sec", candidate)      # a usable clip coordinate exists
            self.assertFalse(candidate.get("subtraction_feasible", False) is True)

    # ---- reproducibility with the real model (requirement 8) ----------------
    def test_real_model_output_is_reproducible(self) -> None:
        path = Path(self.temp.name) / "signal.wav"
        rng = np.random.default_rng(1234)
        sr = 16_000
        base = 0.03 * rng.standard_normal(sr * 20).astype(np.float32)
        t = np.arange(sr * 4) / sr
        burst = (0.25 * np.sin(2 * np.pi * 200.0 * t) * (1.0 + 0.5 * np.sin(2 * np.pi * 4.0 * t))).astype(np.float32)
        base[sr * 6: sr * 6 + burst.size] += burst
        sf.write(path, base, sr)
        zone = vad_preview.build_analysis_zones(20.0)[0]
        first = vad_preview.analyse_zone(path, zone)
        second = vad_preview.analyse_zone(path, zone)
        self.assertEqual(first["status"], second["status"])
        self.assertEqual(first.get("preview_start_sec"), second.get("preview_start_sec"))
        self.assertEqual(first["average_speech_confidence"], second["average_speech_confidence"])
        self.assertEqual(first["speech_interval_count"], second["speech_interval_count"])

    # ---- waveform rendering (requirement 8) --------------------------------
    def test_waveform_svg_renders_and_is_safe(self) -> None:
        svg = vad_preview.waveform_svg(
            self.path, 0.0, 12.0, highlights=[(2.0, 4.0)], marker_start_sec=1.0, marker_end_sec=6.0
        )
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("</svg>", svg)
        self.assertEqual(vad_preview.waveform_svg(self.path, 5.0, 5.0), "")            # empty window
        self.assertEqual(vad_preview.waveform_svg(Path(self.temp.name) / "nope.wav", 0.0, 5.0), "")


if __name__ == "__main__":
    unittest.main()
