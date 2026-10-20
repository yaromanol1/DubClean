from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from scipy import signal

from experiments.paired_reference_cancel.speech_integrity_guard import (
    ALGORITHM,
    REPORT_KIND,
    SCHEMA_VERSION,
    guard_primary_pass,
    guard_second_pass,
    resolved_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "paired_reference_cancel"
    / "config.yaml"
)


def _voice(rate: int, duration: float, *, frequency: float = 187.0) -> np.ndarray:
    time = np.arange(int(round(rate * duration)), dtype=np.float64) / rate
    envelope = 0.48 + 0.30 * np.sin(2.0 * np.pi * 2.9 * time + 0.2)
    envelope += 0.18 * np.sin(2.0 * np.pi * 5.1 * time + 0.8)
    envelope = np.clip(envelope, 0.08, 1.0)
    values = envelope * (
        np.sin(2.0 * np.pi * frequency * time)
        + 0.43 * np.sin(2.0 * np.pi * frequency * 2.0 * time + 0.3)
        + 0.22 * np.sin(2.0 * np.pi * frequency * 3.0 * time + 0.6)
        + 0.11 * np.sin(2.0 * np.pi * frequency * 5.0 * time + 0.1)
    )
    return (0.10 * values).astype(np.float32)


def _unvoiced_phrase(
    rate: int,
    duration: float,
    *,
    modulated: bool,
    resonance_scale: float = 0.30,
) -> np.ndarray:
    """Deterministic band-limited turbulence with optional syllabic envelope."""

    time = np.arange(int(round(rate * duration)), dtype=np.float64) / rate
    noise = np.random.default_rng(9041).normal(size=time.size)
    bandpass = signal.butter(
        4,
        [250.0, min(4200.0, rate * 0.44)],
        btype="bandpass",
        fs=rate,
        output="sos",
    )
    noise = signal.sosfilt(bandpass, noise)
    noise /= max(float(np.std(noise)), 1e-9)
    if modulated:
        envelope = 0.05 + 0.95 * np.maximum(
            0.0, np.sin(2.0 * np.pi * 2.3 * time)
        ) ** 1.7
        # Weak inharmonic resonances create speech-like spectral structure
        # without turning the signal into periodic/voiced speech.
        noise += resonance_scale * np.sin(2.0 * np.pi * 930.0 * time)
        noise += 0.70 * resonance_scale * np.sin(
            2.0 * np.pi * 1710.0 * time + 0.4
        )
    else:
        envelope = np.ones_like(time)
    return (0.08 * envelope * noise).astype(np.float32)


def _impact(rate: int, duration: float) -> np.ndarray:
    """Deterministic door-slam style effect: transient, broadband, non-tonal."""

    time = np.arange(int(round(rate * duration)), dtype=np.float64) / rate
    noise = np.random.default_rng(7717).normal(size=time.size)
    shaped = signal.sosfilt(
        signal.butter(
            2,
            [70.0, min(5200.0, rate * 0.44)],
            btype="bandpass",
            fs=rate,
            output="sos",
        ),
        noise,
    )
    shaped /= max(float(np.std(shaped)), 1e-9)
    envelope = np.zeros_like(time)
    for onset in (1.30, 1.62, 1.98):
        hit = time >= onset
        envelope[hit] += np.exp(-(time[hit] - onset) / 0.11)
    envelope = np.clip(envelope, 0.0, 1.4)
    thump = np.sin(2.0 * np.pi * 58.0 * time) * envelope
    return (0.34 * (envelope * shaped + 0.45 * thump)).astype(np.float32)


def _write(path: Path, values: np.ndarray, rate: int) -> Path:
    subtype = "PCM_24" if path.suffix.casefold() == ".flac" else "FLOAT"
    sf.write(path, np.asarray(values, dtype=np.float32), rate, subtype=subtype)
    return path


def _interval_core(
    interval: dict[str, float], rate: int, config: dict[str, float]
) -> slice:
    """Samples a confirmed interval must reproduce exactly, minus crossfades."""

    fade = float(config["crossfade_sec"])
    start = int(round((float(interval["start_sec"]) + fade) * rate))
    end = int(round((float(interval["end_sec"]) - fade) * rate))
    return slice(start, max(start + 1, end))


def _rms(values: np.ndarray) -> float:
    data = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data))) if data.size else 0.0


def _scale_peak_window_dbfs(
    values: np.ndarray, rate: int, target_dbfs: float
) -> np.ndarray:
    window = max(1, int(round(rate * 0.40)))
    hop = max(1, int(round(rate * 0.20)))
    peaks = [
        _rms(values[start : start + window])
        for start in range(0, max(1, values.size - window + 1), hop)
    ]
    current = max(peaks, default=_rms(values))
    target = 10.0 ** (float(target_dbfs) / 20.0)
    return (np.asarray(values, dtype=np.float32) * target / max(current, 1e-12)).astype(
        np.float32
    )


def _base_config() -> dict[str, float]:
    # Unit fixtures are deliberately short.  Production still uses the
    # bounded full-film budget from the module defaults.
    return {
        "analysis_sample_rate": 8000,
        "analysis_chunk_sec": 2.0,
        "window_sec": 0.40,
        "hop_sec": 0.20,
        "minimum_restoration_budget_sec": 20.0,
        "maximum_restored_fraction": 1.0,
        "maximum_auto_interval_sec": 10.0,
        "crossfade_sec": 0.06,
    }


class SpeechIntegrityGuardTests(unittest.TestCase):
    def test_production_thresholds_are_loaded_from_portable_config(self) -> None:
        portable = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        configured = portable["speech_integrity_guard"]
        resolved = resolved_config(configured)

        self.assertTrue(resolved["enabled"])
        self.assertEqual(configured["algorithm"], ALGORITHM)
        self.assertEqual(resolved["algorithm"], ALGORITHM)
        for key in (
            "second_pass_suspect_loss_db",
            "second_pass_quiet_voiced_minimum_rms_db",
            "second_pass_quiet_voiced_minimum_loss_db",
            "second_pass_quiet_voiced_minimum_speech_band_share",
            "second_pass_quiet_voiced_minimum_temporal_dynamic_db",
            "second_pass_confirmed_loss_db",
            "second_pass_severe_loss_db",
            "second_pass_unvoiced_minimum_loss_db",
            "second_pass_unvoiced_minimum_band_loss_db",
            "second_pass_unvoiced_minimum_speech_band_share",
            "second_pass_unvoiced_minimum_shape_similarity",
            "second_pass_unvoiced_minimum_temporal_similarity",
            "second_pass_unvoiced_minimum_temporal_dynamic_db",
            "second_pass_unvoiced_minimum_spectral_peak_share",
            "primary_minimum_loss_db",
            "primary_minimum_en_margin_db",
            "primary_raw_reference_minimum_removed_shape_similarity",
            "primary_raw_reference_minimum_removed_temporal_similarity",
            "primary_raw_reference_sustained_shape_similarity",
            "primary_raw_reference_maximum_sustained_peak_share",
            "primary_boundary_probe_sec",
            "primary_unvoiced_minimum_loss_db",
            "primary_unvoiced_minimum_band_loss_db",
            "primary_unvoiced_minimum_speech_band_share",
            "primary_unvoiced_minimum_removed_shape_similarity",
            "primary_unvoiced_minimum_removed_temporal_similarity",
            "primary_unvoiced_minimum_temporal_dynamic_db",
            "primary_unvoiced_minimum_spectral_peak_share",
            "primary_unvoiced_minimum_confidence",
            "minimum_interval_sec",
            "crossfade_sec",
            "maximum_restored_fraction",
        ):
            self.assertIn(key, configured)
            self.assertEqual(resolved[key], float(configured[key]))
        self.assertNotIn("model_path", configured)
        self.assertNotIn("checkpoint", configured)

    def test_config_is_finite_and_bounded(self) -> None:
        cfg = resolved_config(
            {
                "analysis_sample_rate": 999999,
                "window_sec": float("nan"),
                "hop_sec": -4,
                "maximum_restored_fraction": 7,
            }
        )
        self.assertEqual(cfg["analysis_sample_rate"], 24000)
        self.assertGreaterEqual(cfg["window_sec"], 0.20)
        self.assertGreaterEqual(cfg["hop_sec"], 0.05)
        self.assertEqual(cfg["maximum_restored_fraction"], 1.0)
        self.assertEqual(cfg["algorithm"], ALGORITHM)

    def test_second_pass_restores_confirmed_local_phrase_loss(self) -> None:
        rate = 16000
        duration = 5.0
        safe = _voice(rate, duration)
        candidate = safe.copy()
        # Cross the 2 s analysis-block seam to exercise overlapped streaming.
        left, right = int(1.85 * rate), int(2.55 * rate)
        candidate[left:right] *= 0.035
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            safe_path = _write(root / "safe.flac", safe, rate)
            candidate_path = _write(root / "candidate.flac", candidate, rate)
            output_path = root / "output.flac"
            report_path = root / "report.json"
            progress: list[float] = []
            report = guard_second_pass(
                safe_path,
                candidate_path,
                output_path,
                _base_config(),
                report_path=report_path,
                progress_callback=lambda value: progress.append(value),
            )
            output, output_rate = sf.read(output_path, dtype="float32")
            persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(output_rate, rate)
        self.assertEqual(output.shape, safe.shape)
        self.assertTrue(report["summary"]["modified"])
        self.assertGreaterEqual(report["summary"]["confirmed_interval_count"], 1)
        self.assertTrue(
            any(
                item["start_sec"] < 2.1 < item["end_sec"]
                for item in report["confirmed_intervals"]
            )
        )
        self.assertGreater(_rms(output[left:right]), _rms(candidate[left:right]) * 10.0)
        self.assertLess(_rms(output[left:right] - safe[left:right]), 2e-4)
        self.assertLess(_rms(output[: int(1.5 * rate)] - candidate[: int(1.5 * rate)]), 1e-6)
        self.assertEqual(persisted["schema_version"], SCHEMA_VERSION)
        self.assertEqual(persisted["report_kind"], REPORT_KIND)
        self.assertEqual(progress[-1], 100.0)

    def test_second_pass_conservative_branch_keeps_safe_on_plausible_loss(self) -> None:
        rate = 16000
        safe = _voice(rate, 4.0)
        candidate = safe.copy()
        event = slice(int(1.3 * rate), int(2.1 * rate))
        candidate[event] *= 0.64  # 3.88 dB: suspicious, below confirmed threshold.
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        self.assertTrue(
            any(
                item["decision"] == "safe_previous_conservative"
                for item in report["confirmed_intervals"]
            )
        )
        self.assertGreater(_rms(result[event]), _rms(candidate[event]) * 1.35)
        protected_core = slice(int(1.5 * rate), int(1.9 * rate))
        self.assertLess(_rms(result[protected_core] - safe[protected_core]), 2e-4)

    def test_second_pass_restores_confirmed_quiet_voiced_loss(self) -> None:
        rate = 16000
        safe = _scale_peak_window_dbfs(_voice(rate, 4.0), rate, -49.0)
        candidate = safe.copy()
        event = slice(int(1.1 * rate), int(2.7 * rate))
        candidate[event] *= 0.01
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        quiet = [
            item
            for item in report["confirmed_intervals"]
            if item["decision"] == "safe_previous_quiet_voiced"
        ]
        self.assertTrue(quiet)
        self.assertTrue(
            all(
                "quiet_voiced_second_pass_loss_consensus" in item["reasons"]
                for item in quiet
            )
        )
        protected_core = _interval_core(quiet[0], rate, report["config"])
        self.assertLess(_rms(result[protected_core] - safe[protected_core]), 2e-4)

    def test_second_pass_quiet_route_requires_confirmed_loss(self) -> None:
        rate = 16000
        safe = _scale_peak_window_dbfs(_voice(rate, 4.0), rate, -49.0)
        candidate = safe.copy()
        event = slice(int(1.1 * rate), int(2.7 * rate))
        candidate[event] *= 10.0 ** (-4.5 / 20.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_second_pass_does_not_lower_the_quiet_floor_below_minus_50(self) -> None:
        rate = 16000
        safe = _scale_peak_window_dbfs(_voice(rate, 4.0), rate, -50.25)
        candidate = safe.copy()
        event = slice(int(1.1 * rate), int(2.7 * rate))
        candidate[event] *= 0.01
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_second_pass_quiet_route_rejects_stationary_harmonic_audio(self) -> None:
        rate = 16000
        duration = 4.0
        time = np.arange(int(rate * duration), dtype=np.float64) / rate
        stationary = (
            np.sin(2.0 * np.pi * 220.0 * time)
            + 0.55 * np.sin(2.0 * np.pi * 440.0 * time)
            + 0.30 * np.sin(2.0 * np.pi * 660.0 * time)
        ).astype(np.float32)
        safe = _scale_peak_window_dbfs(stationary, rate, -49.0)
        candidate = safe.copy()
        event = slice(int(1.1 * rate), int(2.7 * rate))
        candidate[event] *= 0.01
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_unvoiced_phrase_survives_primary_then_second_pass_chain(self) -> None:
        rate = 16000
        mixture = _unvoiced_phrase(rate, 4.0, modulated=True)
        first_candidate = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        first_candidate[event] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            primary_output = root / "primary_safe.wav"
            primary = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "first_candidate.wav", first_candidate, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                primary_output,
                _base_config(),
            )
            safe, _ = sf.read(primary_output, dtype="float32")
            second_candidate = safe.copy()
            second_candidate[event] *= 0.025
            final_output = root / "second_safe.wav"
            second = guard_second_pass(
                primary_output,
                _write(root / "second_candidate.wav", second_candidate, rate),
                final_output,
                _base_config(),
            )
            result, _ = sf.read(final_output, dtype="float32")

        self.assertTrue(
            any(
                item["decision"] == "restore_ru_only_mixture_unvoiced_severe"
                for item in primary["confirmed_intervals"]
            )
        )
        self.assertTrue(
            any(
                item["decision"] == "safe_previous_unvoiced_severe"
                for item in second["confirmed_intervals"]
            )
        )
        protected_core = slice(int(1.55 * rate), int(1.95 * rate))
        self.assertLess(_rms(result[protected_core] - safe[protected_core]), 2e-4)

    def test_second_pass_unvoiced_route_rejects_stationary_noise(self) -> None:
        rate = 16000
        safe = _unvoiced_phrase(rate, 4.0, modulated=False)
        candidate = safe.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        candidate[event] *= 0.025
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_second_pass_does_not_restore_tonal_music_over_dialogue(self) -> None:
        rate = 16000
        duration = 4.0
        dialogue = _voice(rate, duration, frequency=193.0)
        time = np.arange(dialogue.size, dtype=np.float64) / rate
        music = 0.045 * (
            np.sin(2.0 * np.pi * 440.0 * time)
            + 0.55 * np.sin(2.0 * np.pi * 660.0 * time + 0.3)
            + 0.30 * np.sin(2.0 * np.pi * 990.0 * time + 0.7)
        )
        safe = (dialogue + music).astype(np.float32)
        candidate = dialogue.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_second_pass_resamples_safe_24k_to_candidate_48k(self) -> None:
        safe_rate = 24000
        candidate_rate = 48000
        duration = 3.2
        safe = _voice(safe_rate, duration, frequency=197.0)
        candidate = _voice(candidate_rate, duration, frequency=197.0)
        event = slice(int(1.15 * candidate_rate), int(1.85 * candidate_rate))
        candidate[event] *= 0.03
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.flac"
            report = guard_second_pass(
                _write(root / "safe.flac", safe, safe_rate),
                _write(root / "candidate.flac", candidate, candidate_rate),
                output,
                _base_config(),
            )
            result, output_rate = sf.read(output, dtype="float32")

        self.assertEqual(output_rate, candidate_rate)
        self.assertEqual(result.shape, candidate.shape)
        self.assertTrue(report["summary"]["modified"])
        self.assertGreater(_rms(result[event]), _rms(candidate[event]) * 10.0)

    def test_second_pass_does_not_undo_broadband_noise_removal(self) -> None:
        rate = 16000
        speech = _voice(rate, 4.0)
        rng = np.random.default_rng(287)
        # High-frequency leakage increases total energy, but does not produce
        # coherent multi-band speech loss when the enhancer removes it.
        noise = 0.028 * rng.normal(size=speech.size)
        safe = (speech + noise).astype(np.float32)
        candidate = speech
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - candidate), 1e-7)

    def test_second_pass_ignores_silence_and_scope_limits_changes(self) -> None:
        rate = 16000
        safe = np.zeros(int(rate * 5.0), dtype=np.float32)
        phrase = _voice(rate, 0.7)
        safe[int(1.0 * rate) : int(1.7 * rate)] = phrase
        safe[int(3.0 * rate) : int(3.7 * rate)] = phrase
        candidate = safe.copy()
        candidate[int(1.0 * rate) : int(1.7 * rate)] *= 0.02
        candidate[int(3.0 * rate) : int(3.7 * rate)] *= 0.02
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
                analysis_intervals=[(0.7, 2.0)],
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertEqual(len(report["analysis_scopes"]), 1)
        self.assertGreater(
            _rms(result[int(1.0 * rate) : int(1.7 * rate)]),
            _rms(candidate[int(1.0 * rate) : int(1.7 * rate)]) * 10.0,
        )
        self.assertLess(
            _rms(
                result[int(3.0 * rate) : int(3.7 * rate)]
                - candidate[int(3.0 * rate) : int(3.7 * rate)]
            ),
            1e-7,
        )

    def test_preview_crossfades_never_modify_audio_outside_scene(self) -> None:
        rate = 16000
        safe = _voice(rate, 3.0)
        candidate = safe.copy()
        candidate[int(0.5 * rate) : int(2.5 * rate)] *= 0.02
        scope_start, scope_end = 1.0, 2.0
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_second_pass(
                _write(root / "safe.wav", safe, rate),
                _write(root / "candidate.wav", candidate, rate),
                output,
                _base_config(),
                analysis_intervals=[(scope_start, scope_end)],
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        self.assertTrue(report["confirmed_intervals"])
        self.assertTrue(
            all(
                scope_start <= item["start_sec"] < item["end_sec"] <= scope_end
                for item in report["confirmed_intervals"]
            )
        )
        left = int(scope_start * rate)
        right = int(scope_end * rate)
        self.assertLess(_rms(result[:left] - candidate[:left]), 1e-7)
        self.assertLess(_rms(result[right:] - candidate[right:]), 1e-7)
        self.assertGreater(
            _rms(result[left:right]),
            _rms(candidate[left:right]) * 10.0,
        )

    def test_primary_restores_only_triple_consensus_ru_only_loss(self) -> None:
        rate = 16000
        duration = 5.0
        mixture = _voice(rate, duration, frequency=211.0)
        processed = mixture.copy()
        left, right = int(2.0 * rate), int(2.8 * rate)
        processed[left:right] *= 0.025
        silent_reference = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report_path = root / "primary.json"
            mixture_path = _write(root / "mixture.wav", mixture, rate)
            processed_path = _write(root / "processed.wav", processed, rate)
            raw = _write(root / "raw.wav", silent_reference, rate)
            aligned = _write(root / "aligned.wav", silent_reference, rate)
            adapted = _write(root / "adapted.wav", silent_reference, rate)
            report = guard_primary_pass(
                mixture_path,
                processed_path,
                raw,
                aligned,
                adapted,
                output,
                _base_config(),
                report_path=report_path,
            )
            result, _ = sf.read(output, dtype="float32")
            persisted = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["summary"]["modified"])
        self.assertGreater(_rms(result[left:right]), _rms(processed[left:right]) * 15.0)
        self.assertLess(_rms(result[left:right] - mixture[left:right]), 2e-4)
        self.assertTrue(
            all(
                item["decision"] == "restore_ru_only_mixture"
                for item in report["confirmed_intervals"]
            )
        )
        self.assertEqual(persisted["mode"], "post_primary_pass")

    def test_primary_restores_severely_lost_unvoiced_phrase(self) -> None:
        rate = 16000
        mixture = _unvoiced_phrase(rate, 4.0, modulated=True)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        unvoiced = [
            item
            for item in report["confirmed_intervals"]
            if item["decision"] == "restore_ru_only_mixture_unvoiced_severe"
        ]
        self.assertTrue(unvoiced)
        self.assertTrue(
            any(
                "severe_unvoiced_speech_loss_consensus" in item["reasons"]
                for item in unvoiced
            )
        )
        protected_core = slice(int(1.55 * rate), int(1.95 * rate))
        self.assertLess(
            _rms(result[protected_core] - mixture[protected_core]),
            2e-4,
        )

    def test_primary_unvoiced_route_rejects_stationary_nonperiodic_noise(self) -> None:
        rate = 16000
        mixture = _unvoiced_phrase(rate, 4.0, modulated=False)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertFalse(
            any(
                item.get("decision")
                == "restore_ru_only_mixture_unvoiced_severe"
                for item in report["confirmed_intervals"]
            )
        )
        self.assertLess(_rms(result - processed), 1e-7)

    def test_primary_vetoes_when_any_english_reference_is_active(self) -> None:
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=229.0)
        english = _voice(rate, duration, frequency=173.0) * 0.8
        mixture = russian + english
        processed = russian.copy()
        event = slice(int(1.4 * rate), int(2.2 * rate))
        processed[event] *= 0.02
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", english, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - processed), 1e-7)
        self.assertTrue(
            any(
                "english_reference_consensus_failed" in item["reasons"]
                for item in report["rejected_candidates"]
            )
        )

    def test_primary_ignores_loud_raw_program_effect_without_speech(self) -> None:
        rate = 16000
        duration = 4.0
        mixture = _voice(rate, duration, frequency=223.0)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        # A loud centred broadband effect: it beats the mixture by RMS, but
        # has neither normal speech periodicity nor strict unvoiced evidence.
        raw_effect = (
            0.20
            * np.random.default_rng(5531).normal(size=mixture.size)
        ).astype(np.float32)
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", raw_effect, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        self.assertTrue(
            all(
                not item["metrics"]["raw_reference_speech_evidence"]
                for item in report["confirmed_intervals"]
            )
        )
        self.assertTrue(
            all(
                "raw_reference_non_speech_ignored" in item["reasons"]
                for item in report["confirmed_intervals"]
            )
        )
        protected_core = slice(int(1.55 * rate), int(1.95 * rate))
        self.assertLess(
            _rms(result[protected_core] - mixture[protected_core]),
            2e-4,
        )

    def test_primary_raw_nonperiodic_energy_is_not_an_english_veto(self) -> None:
        # The raw reference is a complete programme mix, so band-limited
        # modulated turbulence is at least as likely to be an effect as an
        # unvoiced English line.  It must not veto a repair by loudness alone
        # while both separated speech stems report no English at all.
        rate = 16000
        duration = 4.0
        mixture = _voice(rate, duration, frequency=223.0)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        raw_nonperiodic = 3.0 * _unvoiced_phrase(
            rate,
            duration,
            modulated=True,
            resonance_scale=0.75,
        )
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", raw_nonperiodic, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        confirmed = report["confirmed_intervals"]
        self.assertTrue(confirmed)
        self.assertTrue(
            all(
                not item["metrics"]["raw_reference_speech_evidence"]
                for item in confirmed
            )
        )
        self.assertTrue(
            all(
                "raw_reference_non_speech_ignored" in item["reasons"]
                for item in confirmed
            )
        )
        # The measurement stays visible for diagnostics even though it no
        # longer decides anything.
        self.assertTrue(
            confirmed[0]["metrics"]["raw_reference_unvoiced_speech_evidence"]
        )
        self.assertIn(
            "maximum_raw_reference_periodicity", confirmed[0]["metrics"]
        )
        protected_core = _interval_core(confirmed[0], rate, report["config"])
        self.assertLess(
            _rms(result[protected_core] - mixture[protected_core]),
            2e-4,
        )

    def test_primary_raw_periodic_english_still_vetoes(self) -> None:
        # The protection the change above relies on: English that only the raw
        # programme reference carries is periodic, matches the audio actually
        # removed, and must still block the repair.  Use the production 16 kHz
        # analysis rate because the production spectral thresholds are tuned
        # at that rate (the shared compact fixtures otherwise use 8 kHz).
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=223.0)
        raw_english = 0.75 * _voice(rate, duration, frequency=131.0)
        mixture = (russian + raw_english).astype(np.float32)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", raw_english, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                {**_base_config(), "analysis_sample_rate": 16000},
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        speech_vetoes = [
            item
            for item in report["rejected_candidates"]
            if item["metrics"].get("raw_reference_speech_evidence")
        ]
        self.assertTrue(speech_vetoes)
        self.assertTrue(
            all(
                item["metrics"]["raw_reference_periodicity"]
                >= report["config"]["minimum_periodicity"]
                for item in speech_vetoes
            )
        )
        self.assertTrue(
            all(
                item["metrics"]["raw_reference_removed_spectral_similarity"]
                >= report["config"][
                    "primary_raw_reference_minimum_removed_shape_similarity"
                ]
                for item in speech_vetoes
            )
        )
        self.assertTrue(
            all(
                item["metrics"]["raw_reference_removed_temporal_similarity"]
                >= report["config"][
                    "primary_raw_reference_minimum_removed_temporal_similarity"
                ]
                for item in speech_vetoes
            )
        )
        self.assertLess(_rms(result - processed), 1e-7)

    def test_primary_repairs_russian_under_a_loud_broadband_impact(self) -> None:
        # Regression for the disputed Snatch interval at 57:48.8.  A door-slam
        # style impact covers the Russian phrase in the original programme:
        # loud, broadband, non-periodic, with low-frequency weight.  Both
        # separated English stems report no English, so the phrase must be
        # restored instead of vetoed by the impact.
        rate = 16000
        duration = 4.0
        mixture = _voice(rate, duration, frequency=237.0)
        processed = mixture.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.03
        raw_impact = _impact(rate, duration)
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", raw_impact, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertGreater(_rms(raw_impact), _rms(mixture))
        self.assertTrue(report["summary"]["modified"])
        confirmed = report["confirmed_intervals"]
        self.assertTrue(confirmed)
        self.assertTrue(
            all(
                item["metrics"]["maximum_raw_reference_periodicity"]
                < report["config"]["minimum_periodicity"]
                for item in confirmed
            )
        )
        protected_core = _interval_core(confirmed[0], rate, report["config"])
        self.assertLess(
            _rms(result[protected_core] - mixture[protected_core]),
            2e-4,
        )
        # Nothing from the impact may enter the deliverable: the replacement
        # source is the dubbed speech stem, never the raw programme.
        self.assertGreater(
            _rms(result[protected_core] - raw_impact[protected_core]),
            _rms(raw_impact[protected_core]) * 0.5,
        )

    def test_primary_raw_stationary_tones_do_not_veto_russian_repair(self) -> None:
        rate = 16000
        duration = 4.0
        time = np.arange(int(rate * duration), dtype=np.float64) / rate
        russian = _voice(rate, duration, frequency=223.0)
        processed = russian.copy()
        event = slice(int(1.25 * rate), int(2.25 * rate))
        processed[event] *= 0.025
        silence = np.zeros_like(russian)
        raw_programmes = {
            "single_tone": (0.30 * np.sin(2.0 * np.pi * 440.0 * time)).astype(
                np.float32
            ),
            "three_tone_chord": (
                0.16
                * (
                    np.sin(2.0 * np.pi * 350.0 * time)
                    + np.sin(2.0 * np.pi * 700.0 * time + 0.2)
                    + np.sin(2.0 * np.pi * 1050.0 * time + 0.5)
                )
            ).astype(np.float32),
        }
        for label, raw_programme in raw_programmes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                output = root / "output.wav"
                report = guard_primary_pass(
                    _write(root / "mixture.wav", russian, rate),
                    _write(root / "processed.wav", processed, rate),
                    _write(root / "raw.wav", raw_programme, rate),
                    _write(root / "aligned.wav", silence, rate),
                    _write(root / "adapted.wav", silence, rate),
                    output,
                    _base_config(),
                )
                result, _ = sf.read(output, dtype="float32")

            self.assertTrue(report["summary"]["modified"])
            self.assertTrue(
                all(
                    not item["metrics"]["raw_reference_speech_evidence"]
                    for item in report["confirmed_intervals"]
                )
            )
            protected_core = _interval_core(
                report["confirmed_intervals"][0], rate, report["config"]
            )
            self.assertLess(
                _rms(result[protected_core] - russian[protected_core]), 2e-4
            )

    def test_primary_trims_raw_only_english_from_right_fade(self) -> None:
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=231.0)
        english = np.zeros_like(russian)
        onset = slice(int(2.25 * rate), int(2.30 * rate))
        english[onset] = (_voice(rate, duration, frequency=127.0) * 1.8)[onset]
        mixture = (russian + english).astype(np.float32)
        processed = russian.copy()
        processed[int(1.25 * rate) : int(2.25 * rate)] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", english, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        fade = float(report["config"]["crossfade_sec"])
        self.assertLessEqual(
            max(float(item["end_sec"]) for item in report["confirmed_intervals"])
            + fade,
            2.25 + 1e-6,
        )
        self.assertEqual(_rms(result[onset] - processed[onset]), 0.0)
        restored = slice(int(1.5 * rate), int(1.9 * rate))
        self.assertLess(_rms(result[restored] - mixture[restored]), 2e-4)

    def test_primary_trims_raw_only_english_from_left_fade(self) -> None:
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=231.0)
        english = np.zeros_like(russian)
        onset = slice(int(1.20 * rate), int(1.25 * rate))
        english[onset] = (_voice(rate, duration, frequency=127.0) * 1.8)[onset]
        mixture = (russian + english).astype(np.float32)
        processed = russian.copy()
        processed[int(1.25 * rate) : int(2.25 * rate)] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", english, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        fade = float(report["config"]["crossfade_sec"])
        self.assertGreaterEqual(
            min(float(item["start_sec"]) for item in report["confirmed_intervals"])
            - fade,
            1.25 - 1e-6,
        )
        self.assertEqual(_rms(result[onset] - processed[onset]), 0.0)
        restored = slice(int(1.55 * rate), int(1.95 * rate))
        self.assertLess(_rms(result[restored] - mixture[restored]), 2e-4)

    def test_primary_trims_repair_before_an_english_onset_at_the_edge(self) -> None:
        # A decision window is long enough to hide a short English onset in its
        # final moments behind a comfortable window-average margin, and the
        # crossfade reaches further still.  The repaired span must stop short
        # of the English instead of swapping it into the deliverable.
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=231.0)
        english = np.zeros_like(russian)
        onset = slice(int(2.40 * rate), int(2.44 * rate))
        english[onset] = (_voice(rate, duration, frequency=127.0) * 1.8)[onset]
        mixture = (russian + english).astype(np.float32)
        processed = russian.copy()
        processed[int(1.25 * rate):int(2.25 * rate)] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", english, rate),
                _write(root / "adapted.wav", english, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertTrue(report["summary"]["modified"])
        confirmed = report["confirmed_intervals"]
        self.assertTrue(confirmed)
        self.assertTrue(
            any(item.get("boundary_trim_end_sec", 0.0) > 0.0 for item in confirmed)
        )
        fade = float(report["config"]["crossfade_sec"])
        self.assertLessEqual(
            max(float(item["end_sec"]) for item in confirmed) + fade,
            2.40 + 1e-6,
        )
        # The English onset must be delivered exactly as the model left it.
        self.assertEqual(_rms(result[onset] - processed[onset]), 0.0)
        # The Russian loss is still repaired.
        restored = slice(int(1.5 * rate), int(1.9 * rate))
        self.assertLess(_rms(result[restored] - mixture[restored]), 2e-4)

    def test_primary_drops_a_repair_that_cannot_keep_a_clean_boundary(self) -> None:
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=231.0)
        english = _voice(rate, duration, frequency=127.0) * 1.4
        mixture = (russian + english).astype(np.float32)
        processed = russian.copy()
        processed[int(1.25 * rate):int(2.25 * rate)] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", english, rate),
                _write(root / "adapted.wav", english, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - processed), 1e-7)

    def test_primary_resamples_48k_inputs_to_24k_processed_output(self) -> None:
        source_rate = 48000
        processed_rate = 24000
        duration = 3.5
        mixture = _voice(source_rate, duration, frequency=217.0)
        processed = _voice(processed_rate, duration, frequency=217.0)
        event = slice(int(1.30 * processed_rate), int(2.05 * processed_rate))
        processed[event] *= 0.025
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.flac"
            report = guard_primary_pass(
                _write(root / "mixture.flac", mixture, source_rate),
                _write(root / "processed.flac", processed, processed_rate),
                _write(root / "raw.flac", silence, source_rate),
                _write(root / "aligned.flac", silence, source_rate),
                _write(root / "adapted.flac", silence, source_rate),
                output,
                _base_config(),
            )
            result, output_rate = sf.read(output, dtype="float32")

        self.assertEqual(output_rate, processed_rate)
        self.assertEqual(result.shape, processed.shape)
        self.assertTrue(report["summary"]["modified"])
        self.assertGreater(_rms(result[event]), _rms(processed[event]) * 15.0)

    def test_primary_leaves_low_confidence_loss_processed(self) -> None:
        rate = 16000
        mixture = _voice(rate, 4.0, frequency=203.0)
        processed = mixture.copy()
        event = slice(int(1.2 * rate), int(2.0 * rate))
        processed[event] *= 0.63  # Below the conservative primary threshold.
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - processed), 1e-7)

    def test_primary_does_not_restore_removed_music_under_dialogue(self) -> None:
        rate = 16000
        duration = 4.0
        russian = _voice(rate, duration, frequency=203.0)
        time = np.arange(russian.size, dtype=np.float64) / rate
        music = 0.16 * (
            np.sin(2.0 * np.pi * 440.0 * time)
            + 0.50 * np.sin(2.0 * np.pi * 660.0 * time + 0.2)
            + 0.30 * np.sin(2.0 * np.pi * 880.0 * time + 0.4)
        )
        mixture = (russian + music).astype(np.float32)
        processed = russian.copy()
        silence = np.zeros_like(mixture)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output.wav"
            report = guard_primary_pass(
                _write(root / "mixture.wav", mixture, rate),
                _write(root / "processed.wav", processed, rate),
                _write(root / "raw.wav", silence, rate),
                _write(root / "aligned.wav", silence, rate),
                _write(root / "adapted.wav", silence, rate),
                output,
                _base_config(),
            )
            result, _ = sf.read(output, dtype="float32")

        self.assertFalse(report["summary"]["modified"])
        self.assertLess(_rms(result - processed), 1e-7)

    def test_duration_mismatch_fails_safe_for_each_stage(self) -> None:
        rate = 8000
        long_voice = _voice(rate, 2.0)
        short_voice = _voice(rate, 1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            safe_path = _write(root / "safe.wav", long_voice, rate)
            candidate_path = _write(root / "candidate.wav", short_voice, rate)
            second_output = root / "second.wav"
            second = guard_second_pass(
                safe_path,
                candidate_path,
                second_output,
                _base_config(),
            )
            second_values, _ = sf.read(second_output, dtype="float32")

            mixture_path = _write(root / "mixture.wav", long_voice, rate)
            processed_path = _write(root / "processed.wav", long_voice * 0.2, rate)
            raw = _write(root / "raw.wav", long_voice * 0.0, rate)
            aligned = _write(root / "aligned.wav", long_voice * 0.0, rate)
            adapted = _write(root / "adapted.wav", short_voice * 0.0, rate)
            primary_output = root / "primary.wav"
            primary = guard_primary_pass(
                mixture_path,
                processed_path,
                raw,
                aligned,
                adapted,
                primary_output,
                _base_config(),
            )
            primary_values, _ = sf.read(primary_output, dtype="float32")

        self.assertEqual(second["summary"]["fallback"], "safe_previous")
        self.assertEqual(second_values.shape, long_voice.shape)
        self.assertLess(_rms(second_values - long_voice), 1e-7)
        self.assertEqual(primary["summary"]["fallback"], "processed")
        self.assertLess(_rms(primary_values - long_voice * 0.2), 1e-7)


if __name__ == "__main__":
    unittest.main()
