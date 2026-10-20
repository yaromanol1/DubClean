from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import speech_stem_runner


PYTHON_SRC_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    PYTHON_SRC_ROOT
    / "experiments"
    / "paired_reference_cancel"
    / "speech_stem_runner.py"
)
SEMANTIC_RUNNER = RUNNER.with_name("semantic_separator_infer.py")


class SpeechStemRunnerEntrypointTests(unittest.TestCase):
    def _assert_help_works_from_unrelated_cwd(self, runner: Path) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)

        with tempfile.TemporaryDirectory() as unrelated_cwd:
            completed = subprocess.run(
                [sys.executable, str(runner), "--help"],
                cwd=unrelated_cwd,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

        combined_output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, combined_output)
        self.assertIn("usage:", completed.stdout.casefold())
        self.assertNotIn("ModuleNotFoundError", combined_output)

    def test_speech_runner_absolute_help_works_from_unrelated_cwd(self) -> None:
        self._assert_help_works_from_unrelated_cwd(RUNNER)

    def test_semantic_runner_absolute_help_works_from_unrelated_cwd(self) -> None:
        self._assert_help_works_from_unrelated_cwd(SEMANTIC_RUNNER)


class SpeechStemRunnerOverlapAddTests(unittest.TestCase):
    SAMPLE_RATE = 1000

    def _run_stream(
        self,
        data: np.ndarray,
        *,
        duration_sec: float,
        chunk_sec: float,
        context_sec: float,
        enhancer,
    ) -> tuple[np.ndarray, int, list[dict[str, float | int]]]:
        progress: list[dict[str, float | int]] = []
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "source.wav"
            destination = Path(temporary_dir) / "result.wav"
            sf.write(source, np.asarray(data, dtype=np.float32), self.SAMPLE_RATE, subtype="FLOAT")
            with sf.SoundFile(str(source), "r") as reader, sf.SoundFile(
                str(destination),
                "w",
                samplerate=self.SAMPLE_RATE,
                channels=1,
                format="WAV",
                subtype="FLOAT",
            ) as writer:
                blocks = speech_stem_runner.stream_enhance(
                    reader,
                    writer,
                    model=object(),
                    source_sr=self.SAMPLE_RATE,
                    start_sec=0.0,
                    duration_sec=duration_sec,
                    target_sr=self.SAMPLE_RATE,
                    chunk_sec=chunk_sec,
                    context_sec=context_sec,
                    enhancer=enhancer,
                    progress_callback=progress.append,
                )
            result, sample_rate = sf.read(destination, dtype="float32")
        self.assertEqual(sample_rate, self.SAMPLE_RATE)
        return np.asarray(result, dtype=np.float32), blocks, progress

    def test_overlap_add_smooths_independent_chunk_boundary(self) -> None:
        calls = 0

        def stepped_enhancer(_model, block: np.ndarray) -> np.ndarray:
            nonlocal calls
            value = float(calls) * 0.4
            calls += 1
            return np.full(len(block), value, dtype=np.float32)

        result, blocks, _progress = self._run_stream(
            np.zeros(2500, dtype=np.float32),
            duration_sec=2.5,
            chunk_sec=1.0,
            context_sec=0.2,
            enhancer=stepped_enhancer,
        )

        self.assertEqual(blocks, 3)
        self.assertEqual(len(result), 2500)
        # A hard concatenation would jump by exactly 0.4 at each boundary.
        self.assertLess(float(np.max(np.abs(np.diff(result)))), 0.01)
        self.assertAlmostEqual(float(result[799]), 0.0, places=6)
        self.assertAlmostEqual(float(result[999]), 0.4, places=5)
        self.assertAlmostEqual(float(result[-1]), 0.8, places=5)

    def test_single_block_keeps_samples_and_exact_length(self) -> None:
        source = np.linspace(-0.75, 0.75, 731, dtype=np.float32)
        result, blocks, progress = self._run_stream(
            source,
            duration_sec=0.731,
            chunk_sec=1.0,
            context_sec=0.2,
            enhancer=lambda _model, block: block.copy(),
        )

        self.assertEqual(blocks, 1)
        self.assertEqual(len(result), len(source))
        np.testing.assert_allclose(result, source, atol=1e-6)
        self.assertEqual(progress[-1]["block"], 1)
        self.assertEqual(progress[-1]["blocks"], 1)
        self.assertEqual(progress[-1]["processed_sec"], 0.731)

    def test_overlap_add_does_not_boost_identical_estimates(self) -> None:
        timeline = np.arange(2500, dtype=np.float32) / self.SAMPLE_RATE
        source = 0.3 * np.sin(2.0 * np.pi * 7.0 * timeline)
        result, blocks, _progress = self._run_stream(
            source,
            duration_sec=2.5,
            chunk_sec=1.0,
            context_sec=0.2,
            enhancer=lambda _model, block: block.copy(),
        )

        self.assertEqual(blocks, 3)
        np.testing.assert_allclose(result, source, atol=2e-6)

    def test_final_partial_block_keeps_exact_duration(self) -> None:
        call_values: list[float] = []

        def numbered_enhancer(_model, block: np.ndarray) -> np.ndarray:
            value = float(len(call_values) + 1) / 10.0
            call_values.append(value)
            return np.full(len(block), value, dtype=np.float32)

        result, blocks, progress = self._run_stream(
            np.zeros(2370, dtype=np.float32),
            duration_sec=2.37,
            chunk_sec=1.0,
            context_sec=0.2,
            enhancer=numbered_enhancer,
        )

        self.assertEqual(blocks, 3)
        self.assertEqual(len(call_values), 3)
        self.assertEqual(len(result), 2370)
        self.assertAlmostEqual(float(result[-1]), 0.3, places=5)
        self.assertEqual(progress[-1]["block"], 3)
        self.assertEqual(progress[-1]["blocks"], 3)
        self.assertEqual(progress[-1]["processed_sec"], 2.37)
        self.assertEqual(progress[-1]["total_sec"], 2.37)


if __name__ == "__main__":
    unittest.main()
