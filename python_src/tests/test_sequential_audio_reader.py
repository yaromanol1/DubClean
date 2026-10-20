"""Чтение кэшированных стемов вперёд, крупными блоками.

Блочные читатели раньше просили у декодера ровно тот кусок, который нужен, —
по десять секунд. На реальном фильме это заставляло libsndfile 1.2.0 падать на
одной и той же секунде при каждой попытке, хотя те же байты, прочитанные
тридцатисекундными блоками, декодировались целиком; пересжатие артефакта не
помогало, то есть дело в форме чтения, а не в данных.

Тесты закрепляют два обещания: срез из буфера совпадает с прямым чтением
байт в байт, и декодер при этом никогда не просят перематывать.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel.application_pipeline import (
    _SequentialAudioReader,
)


RATE = 24000


def _fixture(folder: Path, channels: int, seconds: float = 40.0) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(20260813)
    data = (
        rng.standard_normal((int(seconds * RATE), channels)).astype(np.float32) * 0.2
    )
    path = folder / f"stem_{channels}.flac"
    sf.write(str(path), data, RATE, format="FLAC", subtype="PCM_24")
    reference, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return path, reference


class SequentialAudioReaderTests(unittest.TestCase):
    def test_slices_match_a_direct_read_for_every_channel_count(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            for channels in (1, 2):
                path, reference = _fixture(Path(folder), channels)
                reader = _SequentialAudioReader(path, block_sec=3.0)
                try:
                    position = 0
                    # Uneven sizes on purpose: the buffer must not assume the
                    # caller asks in multiples of its own block.
                    for frames in (1000, RATE, 5, 240000, 7777, 100000):
                        with self.subTest(channels=channels, position=position):
                            expected = reference[position : position + frames]
                            if expected.shape[0] < frames:
                                expected = np.pad(
                                    expected,
                                    ((0, frames - expected.shape[0]), (0, 0)),
                                )
                            np.testing.assert_allclose(
                                reader.read_at(position, frames), expected, atol=1e-6
                            )
                        position += frames
                finally:
                    reader.close()

    def test_reading_past_the_end_pads_with_silence(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path, reference = _fixture(Path(folder), 1)
            reader = _SequentialAudioReader(path, block_sec=3.0)
            try:
                tail = reader.read_at(len(reference) + 1000, 500)
            finally:
                reader.close()

            self.assertEqual(tail.shape, (500, 1))
            self.assertTrue(bool(np.all(tail == 0.0)))

    def test_reads_are_uniform_and_large_whatever_the_caller_asks(self) -> None:
        """Именно форма чтения, а не её отсутствие.

        Убрать перемотку целиком нельзя: soundfile сам перематывает после
        каждого ``read``. Что можно — не дробить поток под запросы вызывающего:
        декодер всегда получает один и тот же крупный размер блока, а мелкие
        и неровные срезы выдаются из памяти.
        """
        with tempfile.TemporaryDirectory() as folder:
            path, _reference = _fixture(Path(folder), 1)
            reader = _SequentialAudioReader(path, block_sec=3.0)
            sizes: list[int] = []
            original = reader._reader.read

            def spy(frames, **kwargs):
                sizes.append(frames)
                return original(frames, **kwargs)

            try:
                with mock.patch.object(reader._reader, "read", side_effect=spy):
                    reader.read_at(0, 5)
                    reader.read_at(5, 1000)
                    reader.read_at(1005, 7 * RATE)
            finally:
                reader.close()

            self.assertTrue(sizes)
            self.assertEqual(set(sizes), {3 * RATE})

    def test_large_requests_are_served_across_many_internal_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path, reference = _fixture(Path(folder), 1)
            reader = _SequentialAudioReader(path, block_sec=0.5)
            try:
                got = reader.read_at(0, 20 * RATE)
            finally:
                reader.close()

            np.testing.assert_allclose(got, reference[: 20 * RATE], atol=1e-6)

    def test_a_backwards_request_yields_silence_rather_than_wrong_audio(self) -> None:
        # The routing pass only moves forward.  Returning stale or shifted
        # samples for a backwards ask would be worse than admitting nothing.
        with tempfile.TemporaryDirectory() as folder:
            path, _reference = _fixture(Path(folder), 1)
            reader = _SequentialAudioReader(path, block_sec=3.0)
            try:
                reader.read_at(10 * RATE, RATE)
                back = reader.read_at(0, RATE)
            finally:
                reader.close()

            self.assertEqual(back.shape, (RATE, 1))
            self.assertTrue(bool(np.all(back == 0.0)))

    def test_reader_exposes_the_stream_shape_the_router_relies_on(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path, reference = _fixture(Path(folder), 2)
            reader = _SequentialAudioReader(path)
            try:
                self.assertEqual(reader.samplerate, RATE)
                self.assertEqual(reader.channels, 2)
                self.assertEqual(len(reader), reference.shape[0])
            finally:
                reader.close()


class RoutingUsesTheBufferedReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (
            Path(__file__).parents[2]
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "application_pipeline.py"
        ).read_text(encoding="utf-8")

    def test_router_opens_buffered_readers_not_raw_handles(self) -> None:
        block = self.source.split("def _route_semantic_model_windows(", 1)[1].split(
            "\ndef ", 1
        )[0]

        self.assertIn("_SequentialAudioReader(path)", block)
        self.assertIn("reader.read_at(source_position, source_frames)", block)
        # The old shape asked the decoder for the exact slice and seeked to it.
        self.assertNotIn("reader.seek(", block)
        self.assertNotIn('reader.read(\n', block)


if __name__ == "__main__":
    unittest.main()
