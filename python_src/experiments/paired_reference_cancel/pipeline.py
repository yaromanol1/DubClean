"""Streaming media pipeline for full-length paired-reference processing."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel import adaptive_cancel, audio_io
from experiments.paired_reference_cancel.alignment import (
    _fft_xcorr_range,
    _normalized_score_at,
    estimate_global_offset,
    plateau_segments_from_offsets,
)
from experiments.paired_reference_cancel.real_controls import (
    build_real_evaluation,
)
from experiments.paired_reference_cancel.storage import (
    Store,
    atomic_json,
    disk_free_bytes,
    read_json,
    root_path,
    sha256_file,
    utc_now,
)

Progress = Callable[..., None]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix}")


def _check_stop(ctx: Any) -> None:
    ctx.check_stop()


class _ProgressRangeContext:
    """Map one operation's progress into a task-wide range."""

    def __init__(self, parent: Any, base: float, span: float):
        self.parent = parent
        self.base = float(base)
        self.span = float(span)
        self._high_water = float(base)

    @property
    def task(self):
        return self.parent.task

    @property
    def child_pid(self):
        return self.parent.child_pid

    @child_pid.setter
    def child_pid(self, value):
        self.parent.child_pid = value

    def check_stop(self):
        return self.parent.check_stop()

    def heartbeat(self):
        heartbeat = getattr(self.parent, "heartbeat", None)
        return heartbeat() if heartbeat is not None else None

    def log(self, message):
        return self.parent.log(message)

    def update(self, force: bool = False, **values):
        has_progress = "progress" in values
        local = max(
            0.0,
            min(
                100.0,
                float(
                    values.get(
                        "local_progress",
                        values.get(
                            "progress",
                            self.parent.task.get("local_progress", 0.0),
                        ),
                    )
                    or 0.0
                ),
            ),
        )
        if has_progress:
            operation_progress = max(
                0.0,
                min(100.0, float(values.get("progress", 0.0) or 0.0)),
            )
            self._high_water = max(
                self._high_water,
                self.base + operation_progress / 100.0 * self.span,
            )
        values["progress"] = self._high_water
        values["local_progress"] = local
        return self.parent.update(force=force, **values)


def _run_ffmpeg_progress(
    command: list[str],
    duration: float,
    ctx: Any,
    stage: str,
    current_file: str,
) -> None:
    command = [
        command[0],
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-progress",
        "pipe:1",
        "-nostats",
        *command[1:],
    ]
    started = time.monotonic()
    tail: list[str] = []
    # Publish the new stage before ffmpeg opens a large source container.  On
    # slow/external disks ffmpeg may need noticeable time before its first
    # ``out_time`` record; without this update the UI misleadingly keeps the
    # previous 96–99% stage on screen.
    ctx.update(
        force=True,
        stage=stage,
        substage="Подготовка операции",
        progress=0.0,
        local_progress=0.0,
        processed_seconds=0.0,
        total_seconds=duration,
        eta_seconds=None,
        current_file=current_file,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ctx.child_pid = process.pid
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            _check_stop(ctx)
            line = raw.strip()
            tail.append(line)
            tail = tail[-80:]
            if line.startswith("out_time_ms="):
                try:
                    processed = float(line.split("=", 1)[1]) / 1_000_000.0
                except ValueError:
                    continue
                elapsed = max(time.monotonic() - started, 0.001)
                eta = max(0.0, (duration - processed) / max(processed / elapsed, 1e-6))
                ctx.update(
                    stage=stage,
                    substage=f"Обработано {processed:.1f} из {duration:.1f} с",
                    progress=min(
                        99.0, processed / max(duration, 0.001) * 100.0
                    ),
                    local_progress=min(
                        99.0, processed / max(duration, 0.001) * 100.0
                    ),
                    processed_seconds=processed,
                    total_seconds=duration,
                    eta_seconds=eta,
                    current_file=current_file,
                )
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        ctx.child_pid = None
    if code != 0:
        message = "\n".join(item for item in tail if item)
        raise RuntimeError(f"ffmpeg завершился с кодом {code}.\n{message[-4000:]}")
    ctx.update(
        stage=stage,
        progress=100.0,
        local_progress=100.0,
        processed_seconds=duration,
        total_seconds=duration,
        eta_seconds=0.0,
        current_file=current_file,
    )


def _selected_stream(pair: dict, role: str) -> dict:
    source = pair["sources"][role]
    selected = int(source["stream_index"])
    streams = source.get("probe", {}).get("streams", [])
    for stream in streams:
        if int(stream.get("index", -1)) == selected:
            return stream
    raise ValueError(f"Выбранный аудиопоток {selected} для {role} больше не найден.")


def _duration(pair: dict, role: str) -> float:
    source = pair["sources"][role]
    stream = _selected_stream(pair, role)
    value = stream.get("duration_sec") or source.get("probe", {}).get("duration_sec")
    if not value:
        value = audio_io.probe_duration(source["path"])
    if not value or float(value) <= 0:
        raise RuntimeError("Не удалось определить длительность выбранной дорожки.")
    return float(value)


def _hash_with_progress(path: Path, ctx: Any, role_label: str, block_bytes: int) -> str:
    total = path.stat().st_size
    done = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            _check_stop(ctx)
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
            done += len(block)
            ctx.update(
                stage="Проверка исходных файлов",
                substage=f"Контрольная сумма: {role_label}",
                progress=min(100.0, done / max(total, 1) * 100.0),
                local_progress=min(100.0, done / max(total, 1) * 100.0),
                current_file=path.name,
            )
    return digest.hexdigest()


def _probe_payload(path: Path) -> dict:
    info = sf.info(str(path))
    return {
        "path": str(path.resolve()),
        "duration_sec": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": info.format,
        "subtype": info.subtype,
    }


def extract_pair(store: Store, project_id: str, pair_id: str, ctx: Any) -> dict:
    pair = store.load_pair(project_id, pair_id)
    root = store.pair_dir(project_id, pair_id)
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    cfg = store.cfg
    block_bytes = int(cfg["audio"]["checksum_block_mb"]) * 1024 * 1024

    sample_format = str(cfg["audio"]["extraction_sample_format"])
    bytes_per_sample = {
        "u8": 1,
        "s16": 2,
        "s32": 4,
        "flt": 4,
        "dbl": 8,
    }.get(sample_format, 4)
    previous_metadata = read_json(extracted / "metadata.json", {}) or {}
    previous_files = previous_metadata.get("files") or {}
    reusable_roles: dict[str, bool] = {}
    source_stats: dict[str, os.stat_result] = {}
    estimates = []
    for role in ("original", "dubbed"):
        source = Path(pair["sources"][role]["path"]).resolve()
        stream = _selected_stream(pair, role)
        source_stat = source.stat()
        source_stats[role] = source_stat
        previous = previous_files.get(role) or {}
        destination = extracted / f"{role}.flac"
        proxy = extracted / f"{role}_proxy.wav"
        reusable_roles[role] = (
            destination.is_file()
            and destination.stat().st_size > 0
            and proxy.is_file()
            and proxy.stat().st_size > 0
            and os.path.normcase(
                os.path.normpath(str(previous.get("source_path") or ""))
            )
            == os.path.normcase(os.path.normpath(str(source)))
            and int(previous.get("source_size") or -1) == int(source_stat.st_size)
            and int(previous.get("stream_index") or -1) == int(stream["index"])
            and (
                previous.get("source_mtime_ns") is None
                or int(previous.get("source_mtime_ns") or -1)
                == int(source_stat.st_mtime_ns)
            )
        )
        if not reusable_roles[role]:
            estimates.append(
                _duration(pair, role)
                * int(stream.get("sample_rate") or 48000)
                * int(stream.get("channels") or 2)
                * bytes_per_sample
            )
    required = sum(estimates) * float(cfg["audio"]["free_space_reserve_ratio"])
    minimum = float(cfg["audio"]["minimum_free_space_gb"]) * 1024**3
    free = disk_free_bytes(extracted)
    if free < max(required, minimum):
        raise RuntimeError(
            f"Недостаточно свободного места: доступно {free / 1024**3:.2f} ГБ, "
            f"требуется не менее {max(required, minimum) / 1024**3:.2f} ГБ."
        )

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "prepared_at": utc_now(),
        "files": {},
    }
    ffmpeg = audio_io.check_ffmpeg()
    for index, (role, label) in enumerate(
        (("original", "оригинала"), ("dubbed", "версии с переводом")), 1
    ):
        _check_stop(ctx)
        source = Path(pair["sources"][role]["path"]).resolve()
        stream = _selected_stream(pair, role)
        duration = _duration(pair, role)
        destination = extracted / f"{role}.flac"
        proxy = extracted / f"{role}_proxy.wav"
        previous = previous_files.get(role) or {}
        source_stat = source_stats[role]
        reusable = reusable_roles[role]
        role_base = (index - 1) * 50.0
        if reusable:
            ctx.log(
                f"Повторное извлечение {label} не требуется: "
                f"используется готовая дорожка {destination.name}."
            )
            metadata["files"][role] = {
                **previous,
                **_probe_payload(destination),
                "source_path": str(source),
                "source_size": int(source_stat.st_size),
                "source_mtime_ns": int(source_stat.st_mtime_ns),
                "stream_index": int(stream["index"]),
                "time_base": stream.get("time_base"),
                "codec_name": stream.get("codec_name"),
                "proxy_path": str(proxy.resolve()),
            }
            ctx.update(
                stage="Полное извлечение аудио",
                substage=f"Готово {index} из 2 дорожек",
                progress=index / 2 * 100.0,
                local_progress=100.0,
            )
            continue

        source_hash = _hash_with_progress(
            source,
            _ProgressRangeContext(ctx, role_base, 5.0),
            label,
            block_bytes,
        )
        temporary = _temporary(destination)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-map",
            f"0:{int(stream['index'])}",
            "-vn",
            "-sn",
            "-dn",
            # Keep the selected stream on the container timeline.  Without
            # first_pts=0 ffmpeg resets a delayed secondary audio stream to the
            # beginning, so a track starting at +2 s becomes two seconds early
            # in previews and in the final remux.
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            str(cfg["audio"]["extraction_codec"]),
            "-sample_fmt",
            sample_format,
            "-compression_level",
            str(int(cfg["audio"].get("extraction_compression_level", 8))),
            str(temporary),
        ]
        ctx.log(f"Полное извлечение {label}: {source.name}, поток {stream['index']}.")
        try:
            _run_ffmpeg_progress(
                command,
                duration,
                _ProgressRangeContext(ctx, role_base + 5.0, 35.0),
                f"Полное извлечение аудио: {label}",
                source.name,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg не создал выходной аудиофайл.")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        proxy_temp = _temporary(proxy)
        proxy_command = [
            ffmpeg,
            "-y",
            "-i",
            str(destination),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(int(cfg["audio"]["proxy_sample_rate"])),
            "-c:a",
            "pcm_s16le",
            str(proxy_temp),
        ]
        try:
            _run_ffmpeg_progress(
                proxy_command,
                duration,
                _ProgressRangeContext(ctx, role_base + 40.0, 10.0),
                f"Подготовка облегчённой копии: {label}",
                destination.name,
            )
            os.replace(proxy_temp, proxy)
        finally:
            proxy_temp.unlink(missing_ok=True)
        metadata["files"][role] = {
            **_probe_payload(destination),
            "source_path": str(source),
            "source_sha256": source_hash,
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "stream_index": int(stream["index"]),
            "time_base": stream.get("time_base"),
            "codec_name": stream.get("codec_name"),
            "proxy_path": str(proxy.resolve()),
        }
        ctx.update(
            stage="Полное извлечение аудио",
            substage=f"Готово {index} из 2 дорожек",
            progress=index / 2 * 100.0,
            local_progress=100.0,
        )

    atomic_json(extracted / "metadata.json", metadata)
    pair["extraction"] = metadata
    pair["current_stage"] = max(int(pair.get("current_stage", 1)), 2)
    pair["stages"]["2"] = {"status": "completed", "message": "Обе дорожки полностью извлечены."}
    pair["stages"]["3"] = {"status": "available", "message": "Можно построить карту сопоставления."}
    store.save_pair(pair)
    return {"metadata": str(extracted / "metadata.json")}


def _refine_anchor_on_proxy(
    original_proxy: Path,
    dubbed_proxy: Path,
    proxy_sr: int,
    dubbed_start_sec: float,
    coarse_original_start_sec: float,
    window_sec: float,
    radius_sec: float,
) -> tuple[float, float] | None:
    """Refine a sparse control point to sample precision.

    The initial/local search runs on a reduced proxy. The adaptive canceller's
    STFT window cannot absorb even a small residual timing error, so this final
    local correlation uses the full proxy rate inside ``radius_sec`` and returns
    ``(refined_original_start_sec, waveform_correlation)``; ``None`` when the
    windows are unusable (out of range / near-silence).
    """
    dubbed_window, _ = _read_interval(dubbed_proxy, dubbed_start_sec, window_sec)
    dubbed_mono = dubbed_window[:, 0]
    if len(dubbed_mono) < proxy_sr // 4 or float(np.max(np.abs(dubbed_mono))) < 1e-5:
        return None
    ref_start_sec = coarse_original_start_sec - radius_sec
    ref_window, _ = _read_interval(
        original_proxy, ref_start_sec, window_sec + 2.0 * radius_sec
    )
    ref_mono = ref_window[:, 0]
    if float(np.max(np.abs(ref_mono))) < 1e-5:
        return None
    # _read_interval zero-pads negative starts, so ref_mono[0] always maps to
    # ref_start_sec on the original timeline.
    expected_orig_start_sample = int(round(coarse_original_start_sec * proxy_sr))
    ref_origin_sample = int(round(ref_start_sec * proxy_sr))
    expected_shift = ref_origin_sample - expected_orig_start_sample
    radius_samples = max(1, int(round(radius_sec * proxy_sr)))
    # PHAT sharpens the peak on broadband/reverberant material but collapses
    # on tonal content (whitening levels the informative bins with the
    # ambiguous ones); plain energy-weighted correlation behaves the other
    # way around. Locate with both and keep whichever candidate actually
    # scores higher on plain normalized correlation.
    best_shift, best_score = 0, -1.0
    for phat in (False, True):
        shift, _peak = _fft_xcorr_range(
            dubbed_mono,
            ref_mono,
            expected_shift - radius_samples,
            expected_shift + radius_samples,
            phat=phat,
        )
        score = abs(_normalized_score_at(dubbed_mono, ref_mono, shift))
        if score > best_score:
            best_shift, best_score = shift, score
    refined_start_sec = (ref_origin_sample - best_shift) / proxy_sr
    return refined_start_sec, best_score


def _resample_mono_for_search(
    data: np.ndarray, source_sr: int, search_sr: int
) -> np.ndarray:
    mono = np.mean(data, axis=1) if data.ndim == 2 else np.asarray(data)
    mono = np.nan_to_num(mono, copy=False).astype(np.float32)
    if source_sr != search_sr:
        mono = np.asarray(audio_io.resample(mono, source_sr, search_sr)).reshape(-1)
    return mono


def _best_valid_correlation(reference: np.ndarray, search: np.ndarray) -> tuple[int, float, float]:
    """Locate ``reference`` inside ``search`` with normalized FFT correlation."""
    reference = np.asarray(reference, dtype=np.float64)
    search = np.asarray(search, dtype=np.float64)
    if len(reference) < 32 or len(search) < len(reference):
        return 0, 0.0, 0.0
    reference = reference - np.mean(reference)
    reference_energy = float(np.sum(reference * reference))
    if reference_energy < 1e-10:
        return 0, 0.0, 0.0
    numerator = signal.fftconvolve(search, reference[::-1], mode="valid")
    size = len(reference)
    prefix = np.concatenate(([0.0], np.cumsum(search)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(search * search)))
    sums = prefix[size:] - prefix[:-size]
    sums_sq = prefix_sq[size:] - prefix_sq[:-size]
    variance = np.maximum(sums_sq - (sums * sums) / size, 1e-12)
    scores = np.abs(numerator) / np.sqrt(reference_energy * variance)
    best_index = int(np.argmax(scores))
    best_score = float(np.clip(scores[best_index], 0.0, 1.0))
    # Ignore the same correlation lobe when measuring ambiguity.
    masked = scores.copy()
    radius = max(1, size // 2)
    masked[max(0, best_index - radius) : best_index + radius + 1] = 0.0
    second_score = float(np.max(masked)) if len(masked) else 0.0
    return best_index, best_score, second_score


def _normalized_window_score(reference: np.ndarray, search: np.ndarray, index: int) -> float:
    if index < 0 or index + len(reference) > len(search) or len(reference) < 2:
        return 0.0
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(search[index : index + len(reference)], dtype=np.float64)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-10:
        return 0.0
    return float(np.clip(abs(np.dot(left, right) / denominator), 0.0, 1.0))


def _spectral_novelty(
    data: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int, int]:
    """Flattened multiband spectral-change descriptor.

    Each band is normalized over time independently. A louder/quieter master,
    a different channel layout, or an added voice therefore cannot dominate
    every part of the descriptor.
    """
    values = np.nan_to_num(np.asarray(data, dtype=np.float64), copy=False)
    nperseg = max(64, min(256, 2 ** int(round(np.log2(max(64, sample_rate * 0.064))))))
    hop = max(16, int(round(sample_rate * 0.02)))
    hop = min(hop, nperseg // 2)
    if len(values) < nperseg * 2:
        return np.empty(0, dtype=np.float64), hop, 1
    frequencies, _times, spectrum = signal.stft(
        values,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=nperseg - hop,
        boundary=None,
        padded=False,
    )
    log_magnitude = np.log1p(np.abs(spectrum) * 10.0)
    usable_bins = max(2, len(frequencies))
    edges = np.unique(np.geomspace(1, usable_bins, 13).astype(int))
    bands = [
        np.mean(log_magnitude[left:right], axis=0)
        for left, right in zip(edges[:-1], edges[1:])
        if right > left
    ]
    if not bands:
        return np.empty(0, dtype=np.float64), hop, 1
    features = np.stack(bands, axis=1)
    features = np.diff(features, axis=0)
    features -= np.median(features, axis=0, keepdims=True)
    robust_scale = np.median(np.abs(features), axis=0, keepdims=True) * 1.4826
    features /= np.maximum(robust_scale, 1e-4)
    features = np.clip(features, -3.0, 3.0)
    return features.reshape(-1), hop, features.shape[1]


def _best_strided_correlation(
    reference: np.ndarray,
    search: np.ndarray,
    stride: int,
) -> tuple[int, float, float]:
    """Normalized correlation restricted to complete feature frames."""
    reference = np.asarray(reference, dtype=np.float64)
    search = np.asarray(search, dtype=np.float64)
    if len(reference) < stride * 8 or len(search) < len(reference):
        return 0, 0.0, 0.0
    reference -= np.mean(reference)
    reference_energy = float(np.sum(reference * reference))
    if reference_energy < 1e-10:
        return 0, 0.0, 0.0
    numerator = signal.fftconvolve(search, reference[::-1], mode="valid")
    size = len(reference)
    prefix = np.concatenate(([0.0], np.cumsum(search)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(search * search)))
    sums = prefix[size:] - prefix[:-size]
    sums_sq = prefix_sq[size:] - prefix_sq[:-size]
    variance = np.maximum(sums_sq - sums * sums / size, 1e-12)
    scores = np.abs(numerator) / np.sqrt(reference_energy * variance)
    framed = scores[::stride]
    best_frame = int(np.argmax(framed))
    best_score = float(np.clip(framed[best_frame], 0.0, 1.0))
    masked = framed.copy()
    radius = max(1, (len(reference) // stride) // 2)
    masked[max(0, best_frame - radius) : best_frame + radius + 1] = 0.0
    second_score = float(np.max(masked)) if len(masked) else 0.0
    return best_frame, best_score, second_score


def _best_gain_invariant_match(
    reference: np.ndarray,
    search: np.ndarray,
    sample_rate: int,
) -> tuple[int, float, float]:
    """Combine waveform and normalized spectral evidence.

    Waveform correlation supplies timing precision. Spectral novelty is more
    tolerant of different gains, channel layouts and an added translated voice.
    """
    waveform_index, waveform_score, waveform_second = _best_valid_correlation(
        reference, search
    )
    reference_feature, hop, bands = _spectral_novelty(reference, sample_rate)
    search_feature, _search_hop, search_bands = _spectral_novelty(search, sample_rate)
    if len(reference_feature) < 16 or len(search_feature) < len(reference_feature):
        return waveform_index, waveform_score, waveform_second
    if search_bands != bands:
        return waveform_index, waveform_score, waveform_second
    spectral_frame, spectral_score, spectral_second = _best_strided_correlation(
        reference_feature, search_feature, bands
    )
    spectral_sample_index = spectral_frame * hop
    candidates = {waveform_index, spectral_sample_index}
    scored: list[tuple[float, int]] = []
    for sample_index in candidates:
        wave = _normalized_window_score(reference, search, sample_index)
        feature_index = int(round(sample_index / hop)) * bands
        spectral = _normalized_window_score(reference_feature, search_feature, feature_index)
        distinctiveness = float(
            np.clip(
                (spectral - spectral_second) / max(spectral, 0.05),
                0.0,
                1.0,
            )
        )
        confidence = 0.25 * wave + 0.45 * spectral + 0.30 * distinctiveness
        scored.append((confidence, sample_index))
    best_score, best_index = max(scored)
    second_score = min(best_score, 0.25 * waveform_second + 0.45 * spectral_second)
    return int(best_index), float(np.clip(best_score, 0.0, 1.0)), float(second_score)


def _initial_match_acceptance(candidate: dict, cfg: dict) -> str | None:
    """Accept a slightly weaker first match only when it is unambiguous.

    Different releases can have a translated voice, another channel layout and
    PAL speed-up at the same time.  Their best combined score may land just
    below the normal threshold even though the next-best location is far worse.
    Lowering the main threshold globally would make repetitive music and logos
    unsafe, so the relaxed path also requires a large distinctiveness margin.
    """
    score = float(candidate.get("confidence") or 0.0)
    second = float(candidate.get("second_score") or 0.0)
    threshold = float(cfg.get("initial_min_confidence", cfg["min_confidence"]))
    if score >= threshold and score - second >= -0.02:
        return "strict"

    distinctive_threshold = min(
        threshold,
        float(cfg.get("initial_distinctive_min_confidence", threshold)),
    )
    distinctive_margin = float(cfg.get("initial_distinctive_min_margin", 0.20))
    if (
        score >= distinctive_threshold
        and score - second >= distinctive_margin
    ):
        return "distinctive"
    return None


def _find_first_sparse_match(
    original_proxy: Path,
    dubbed_proxy: Path,
    cfg: dict,
    ctx: Any,
) -> dict:
    original_info = sf.info(str(original_proxy))
    dubbed_info = sf.info(str(dubbed_proxy))
    window_sec = min(
        float(cfg["initial_window_sec"]),
        original_info.duration,
        dubbed_info.duration,
    )
    if window_sec < 1.0:
        raise RuntimeError("Дорожки слишком короткие для поиска первого совпадения.")
    search_duration = min(
        dubbed_info.duration,
        float(cfg["initial_search_max_sec"])
        + float(cfg["reference_scan_sec"])
        + window_sec,
    )
    dubbed_data, dubbed_sr = _read_interval(dubbed_proxy, 0.0, search_duration)
    search_sr = min(int(cfg["search_sample_rate"]), dubbed_sr)
    dubbed_search = _resample_mono_for_search(dubbed_data, dubbed_sr, search_sr)

    duration_ratio = original_info.duration / max(dubbed_info.duration, 1e-9)
    configured_ratios = [
        float(value) for value in cfg.get("initial_speed_ratios", [1.0])
    ]
    speed_ratios = sorted(
        {
            round(value, 8)
            for value in [*configured_ratios, duration_ratio]
            if 0.90 <= value <= 1.10
        },
        key=lambda value: abs(value - 1.0),
    )
    maximum_reference_start = max(
        0.0,
        min(
            float(cfg["reference_scan_sec"]),
            original_info.duration - window_sec * max(speed_ratios),
        ),
    )
    count = max(1, int(cfg["reference_candidates"]))
    reference_starts = np.linspace(0.0, maximum_reference_start, count)
    best: dict | None = None
    for index, reference_start in enumerate(reference_starts):
        _check_stop(ctx)
        ctx.update(
            stage="Поиск первого совпадения",
            substage=f"Проверяется начальный ориентир: {index + 1} из {len(reference_starts)}",
            progress=index / len(reference_starts) * 35.0,
            local_progress=index / len(reference_starts) * 100.0,
            processed_seconds=search_duration,
            total_seconds=dubbed_info.duration,
            current_file=dubbed_proxy.name,
        )
        candidate = None
        for speed_ratio in speed_ratios:
            _check_stop(ctx)
            reference_data, reference_sr = _read_interval(
                original_proxy,
                float(reference_start),
                window_sec * speed_ratio,
            )
            reference = _resample_mono_for_search(
                reference_data, reference_sr, search_sr
            )
            reference = signal.resample(
                reference, int(round(window_sec * search_sr))
            ).astype(np.float32)
            if float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))) < 1e-4:
                score = second = 0.0
                match_start = 0.0
            else:
                match_index, score, second = _best_gain_invariant_match(
                    reference, dubbed_search, search_sr
                )
                match_start = match_index / search_sr
            current = {
                "original_start": float(reference_start),
                "dubbed_start": float(match_start),
                "confidence": score,
                "second_score": second,
                "window_sec": window_sec,
                "speed_ratio": float(speed_ratio),
            }
            if candidate is None or current["confidence"] > candidate["confidence"]:
                candidate = current
        assert candidate is not None
        score = float(candidate["confidence"])
        second = float(candidate["second_score"])
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate
        ctx.update(
            stage="Поиск первого совпадения",
            substage=f"Проверено начальных ориентиров: {index + 1} из {len(reference_starts)}",
            progress=(index + 1) / len(reference_starts) * 35.0,
            local_progress=(index + 1) / len(reference_starts) * 100.0,
            processed_seconds=search_duration,
            total_seconds=dubbed_info.duration,
            current_file=dubbed_proxy.name,
        )
        # The earliest original reference that matches reliably is the first
        # useful occurrence; later candidates are unnecessary.
        if _initial_match_acceptance(candidate, cfg) == "strict":
            candidate["acceptance"] = "strict"
            best = candidate
            break
    acceptance = _initial_match_acceptance(best or {}, cfg)
    if best is None or acceptance is None:
        if best is None:
            detail = ""
        else:
            detail = (
                f" Лучший результат: {float(best['confidence']):.3f}, "
                f"следующий: {float(best['second_score']):.3f}."
            )
        raise RuntimeError(
            "Надёжное совпадение выбранных дорожек не найдено."
            f"{detail} Проверьте выбранные дорожки."
        )
    best["acceptance"] = acceptance
    return best


def _verify_sparse_point(
    original_proxy: Path,
    dubbed_proxy: Path,
    dubbed_start: float,
    expected_original_start: float,
    cfg: dict,
    proxy_sr: int,
    speed_ratio: float = 1.0,
) -> dict:
    window_sec = float(cfg["verification_window_sec"])
    search_sr = min(int(cfg["search_sample_rate"]), proxy_sr)

    def search(radius: float) -> tuple[float, float, float]:
        dubbed_data, dubbed_sr = _read_interval(
            dubbed_proxy, dubbed_start, window_sec
        )
        reference = _resample_mono_for_search(dubbed_data, dubbed_sr, search_sr)
        search_start = expected_original_start - radius * speed_ratio
        original_data, original_sr = _read_interval(
            original_proxy,
            search_start,
            (window_sec + 2.0 * radius) * speed_ratio,
        )
        original_search = _resample_mono_for_search(
            original_data, original_sr, search_sr
        )
        original_search = signal.resample(
            original_search,
            int(round(len(original_search) / max(speed_ratio, 1e-9))),
        ).astype(np.float32)
        match_index, score, second = _best_gain_invariant_match(
            reference, original_search, search_sr
        )
        return (
            search_start + match_index / search_sr * speed_ratio,
            score,
            second,
        )

    original_start, score, second = search(float(cfg["local_search_radius_sec"]))
    if score < float(cfg["min_confidence"]):
        recovered = search(float(cfg["recovery_search_radius_sec"]))
        if recovered[1] > score:
            original_start, score, second = recovered
    if abs(speed_ratio - 1.0) < 0.005:
        refined = _refine_anchor_on_proxy(
            original_proxy,
            dubbed_proxy,
            proxy_sr,
            dubbed_start,
            original_start,
            window_sec,
            float(cfg["verification_refine_radius_sec"]),
        )
        if refined is not None and refined[1] >= score:
            original_start, score = refined
    acceptance = _verification_point_acceptance(
        original_start,
        expected_original_start,
        score,
        second,
        cfg,
    )
    return {
        "dubbed_start": float(dubbed_start),
        "original_start": float(original_start),
        "confidence": float(np.clip(score, 0.0, 1.0)),
        "second_score": float(second),
        "prediction_error_sec": float(original_start - expected_original_start),
        "acceptance": acceptance,
        "usable": acceptance is not None,
        "reason": None if acceptance is not None else "низкая уверенность контрольной точки",
        "kind": "verification",
    }


def _verification_point_acceptance(
    original_start: float,
    expected_original_start: float,
    score: float,
    second_score: float,
    cfg: dict,
) -> str | None:
    """Keep a weak verification point only when timing and uniqueness agree."""
    score = float(np.clip(score, 0.0, 1.0))
    second_score = float(np.clip(second_score, 0.0, 1.0))
    threshold = float(cfg["min_confidence"])
    if score >= threshold:
        return "strict"

    distinctive_threshold = min(
        threshold,
        float(cfg.get("verification_distinctive_min_confidence", threshold)),
    )
    distinctive_margin = float(
        cfg.get("verification_distinctive_min_margin", 0.12)
    )
    prediction_tolerance = max(
        0.0,
        float(cfg.get("verification_prediction_tolerance_sec", 2.0)),
    )
    if (
        score >= distinctive_threshold
        and score - second_score >= distinctive_margin
        and abs(float(original_start) - float(expected_original_start))
        <= prediction_tolerance
    ):
        return "distinctive_prediction"
    return None


def _alignment_segment_at(alignment_map: dict, dubbed_sec: float) -> dict | None:
    """Return the affine map segment covering ``dubbed_sec``."""
    segments = alignment_map.get("segments") or alignment_map.get("chunks") or []
    usable = [item for item in segments if item.get("usable", True)]
    if not usable:
        return None
    for item in usable:
        if float(item.get("dubbed_start", 0.0)) <= dubbed_sec < float(
            item.get("dubbed_end", 0.0)
        ):
            return item
    return min(
        usable,
        key=lambda item: min(
            abs(dubbed_sec - float(item.get("dubbed_start", 0.0))),
            abs(dubbed_sec - float(item.get("dubbed_end", 0.0))),
        ),
    )


def _mapped_original_time(alignment_map: dict, dubbed_sec: float) -> tuple[float, float]:
    segment = _alignment_segment_at(alignment_map, dubbed_sec)
    if segment is None:
        global_map = alignment_map.get("global") or {}
        speed = float(global_map.get("speed_ratio") or 1.0)
        offset = float(global_map.get("offset_sec") or 0.0)
        return dubbed_sec * speed - offset, speed
    speed = float(segment.get("speed_ratio") or 1.0)
    original = float(segment.get("original_start") or 0.0) + (
        dubbed_sec - float(segment.get("dubbed_start") or 0.0)
    ) * speed
    return original, speed


def estimate_sparse_manual_correction(
    original_proxy: Path,
    dubbed_proxy: Path,
    alignment_map: dict,
    alignment_cfg: dict,
) -> dict:
    """Estimate a residual manual correction from points spread over the film.

    The existing piecewise map remains the baseline.  This routine checks that
    baseline at independent positions across usable scenes and robustly
    aggregates their residuals.  It deliberately never infers the correction
    from file starts or total durations.
    """
    original_info = sf.info(str(original_proxy))
    dubbed_info = sf.info(str(dubbed_proxy))
    window_sec = min(
        float(alignment_cfg.get("verification_window_sec", 10.0)),
        float(original_info.duration),
        float(dubbed_info.duration),
    )
    if window_sec < 1.0:
        raise RuntimeError("Дорожки слишком короткие для автоматического выравнивания.")

    segments = [
        item
        for item in (alignment_map.get("segments") or alignment_map.get("chunks") or [])
        if item.get("usable", True)
        and float(item.get("dubbed_end", 0.0))
        - float(item.get("dubbed_start", 0.0))
        >= max(1.0, window_sec * 0.6)
    ]
    if not segments:
        matched = min(float(original_info.duration), float(dubbed_info.duration))
        segments = [
            {
                "dubbed_start": 0.0,
                "dubbed_end": matched,
                "original_start": 0.0,
                "speed_ratio": 1.0,
                "usable": True,
            }
        ]

    point_count = max(3, int(alignment_cfg.get("auto_correction_points", 9)))
    first = min(float(item.get("dubbed_start", 0.0)) for item in segments)
    last = max(float(item.get("dubbed_end", 0.0)) for item in segments) - window_sec
    candidates: list[float] = []
    if last >= first:
        for index in range(point_count):
            ratio = (index + 0.5) / point_count
            wanted = first + (last - first) * ratio
            segment = _alignment_segment_at(alignment_map, wanted)
            if segment is None:
                continue
            start = max(
                float(segment.get("dubbed_start", 0.0)),
                min(
                    wanted,
                    float(segment.get("dubbed_end", 0.0)) - window_sec,
                ),
            )
            if start >= 0.0 and all(abs(start - value) > window_sec * 0.35 for value in candidates):
                candidates.append(start)
    if not candidates:
        candidates = [max(0.0, first)]

    verifier_cfg = dict(alignment_cfg)
    verifier_cfg["local_search_radius_sec"] = float(
        alignment_cfg.get("auto_correction_local_radius_sec", 45.0)
    )
    verifier_cfg["recovery_search_radius_sec"] = float(
        alignment_cfg.get("auto_correction_recovery_radius_sec", 300.0)
    )
    verifier_cfg["min_confidence"] = float(
        alignment_cfg.get("auto_correction_min_confidence", alignment_cfg.get("min_confidence", 0.3))
    )

    measurements: list[dict] = []
    for dubbed_start in candidates:
        expected_original, speed = _mapped_original_time(alignment_map, dubbed_start)
        checkpoint = _verify_sparse_point(
            original_proxy,
            dubbed_proxy,
            dubbed_start,
            expected_original,
            verifier_cfg,
            int(dubbed_info.samplerate),
            speed,
        )
        if not checkpoint.get("usable"):
            continue
        residual = float(checkpoint["original_start"]) - expected_original
        correction = -residual / max(abs(speed), 1e-9)
        measurements.append(
            {
                "dubbed_start": round(dubbed_start, 3),
                "expected_original": round(expected_original, 3),
                "measured_original": round(float(checkpoint["original_start"]), 3),
                "correction_sec": round(correction, 6),
                "confidence": round(float(checkpoint.get("confidence") or 0.0), 6),
            }
        )

    minimum = max(2, int(alignment_cfg.get("auto_correction_min_points", 3)))
    if len(measurements) < minimum:
        raise RuntimeError(
            "Не удалось надёжно определить автоматический сдвиг: недостаточно совпадающих участков."
        )
    values = np.asarray([item["correction_sec"] for item in measurements], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    tolerance = max(0.12, 3.0 * 1.4826 * mad)
    inliers = [
        item for item in measurements if abs(float(item["correction_sec"]) - median) <= tolerance
    ]
    if len(inliers) < minimum:
        raise RuntimeError(
            "Автоматический сдвиг различается между сценами; оставлен ручной режим."
        )
    correction = float(np.median([item["correction_sec"] for item in inliers]))
    maximum = float(alignment_cfg.get("auto_correction_max_abs_sec", 600.0))
    correction = float(np.clip(correction, -maximum, maximum))
    confidence = float(np.median([item["confidence"] for item in inliers]))
    confidence *= min(1.0, len(inliers) / max(5.0, float(point_count)))
    return {
        "correction_sec": round(correction, 3),
        "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
        "analysed_points": len(measurements),
        "accepted_points": len(inliers),
        "points": measurements,
        "method": "sparse_multi_scene_residual_v1",
    }


def _segments_from_sparse_points(
    checkpoints: list[dict],
    original_duration: float,
    dubbed_duration: float,
    cfg: dict,
) -> list[dict]:
    checkpoints = sorted(checkpoints, key=lambda item: item["dubbed_start"])
    segments: list[dict] = []
    max_speed_delta = float(cfg["max_speed_deviation"])
    interval_speeds = [
        (right["original_start"] - left["original_start"])
        / max(right["dubbed_start"] - left["dubbed_start"], 1e-9)
        for left, right in zip(checkpoints[:-1], checkpoints[1:])
        if left["usable"] and right["usable"]
    ]
    stable_speed = (
        float(np.median(interval_speeds)) if interval_speeds else 1.0
    )
    discontinuity_sec = float(cfg["discontinuity_threshold_ms"]) / 1000.0

    def append_segment(
        dubbed_start: float,
        dubbed_end: float,
        original_start: float,
        original_end: float,
        confidence: float,
        usable: bool,
        reason: str | None,
        checkpoint_start: int | None,
        checkpoint_end: int | None,
    ) -> None:
        duration = max(0.0, dubbed_end - dubbed_start)
        if duration <= 1e-6:
            return
        speed = (original_end - original_start) / duration
        if usable and abs(speed - 1.0) > max_speed_delta:
            usable = False
            reason = f"изменение скорости {speed:.6f} превышает безопасный предел"
        segments.append(
            {
                "id": len(segments),
                "original_start": round(original_start, 6),
                "original_end": round(original_end, 6),
                "dubbed_start": round(dubbed_start, 6),
                "dubbed_end": round(dubbed_end, 6),
                "duration": round(duration, 6),
                "delay": round(dubbed_start - original_start, 6),
                "offset": round(dubbed_start - original_start, 6),
                "speed_ratio": round(speed, 8),
                "confidence": round(float(confidence), 6),
                "correlation": round(float(confidence), 6),
                "status": "usable" if usable else "unusable",
                "usable": bool(usable),
                "reason": reason,
                "checkpoint_start": checkpoint_start,
                "checkpoint_end": checkpoint_end,
            }
        )

    first = checkpoints[0]
    if first["dubbed_start"] > 0.01:
        append_segment(
            0.0,
            first["dubbed_start"],
            first["original_start"] - first["dubbed_start"],
            first["original_start"],
            0.0,
            False,
            "участок до первого подтверждённого совпадения",
            None,
            0,
        )
    for index, (left, right) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        duration = right["dubbed_start"] - left["dubbed_start"]
        observed_original_delta = right["original_start"] - left["original_start"]
        mapping_residual = abs(observed_original_delta - stable_speed * duration)
        interval_usable = bool(left["usable"] and right["usable"])
        interval_reason = left.get("reason") or right.get("reason")
        if (
            interval_usable
            and len(interval_speeds) >= 3
            and mapping_residual > discontinuity_sec
        ):
            interval_usable = False
            interval_reason = (
                f"между контрольными точками найден скачок временной карты "
                f"{mapping_residual * 1000.0:.0f} мс — возможен монтаж"
            )
        append_segment(
            left["dubbed_start"],
            right["dubbed_start"],
            left["original_start"],
            right["original_start"],
            min(left["confidence"], right["confidence"]),
            interval_usable,
            interval_reason,
            index,
            index + 1,
        )
    last = checkpoints[-1]
    if last["dubbed_start"] < dubbed_duration:
        speed = stable_speed
        tail_duration = dubbed_duration - last["dubbed_start"]
        tail_end = last["original_start"] + tail_duration * speed
        tail_usable = bool(last["usable"] and tail_end <= original_duration + 0.5)
        append_segment(
            last["dubbed_start"],
            dubbed_duration,
            last["original_start"],
            tail_end,
            last["confidence"],
            tail_usable,
            None if tail_usable else "конец одной версии выходит за границы другой",
            len(checkpoints) - 1,
            None,
        )
    return segments


DEFAULT_SAME_CONTAINER_VERIFICATION: dict[str, float] = {
    "chunk_sec": 30.0,
    "search_radius_sec": 1.0,
    "min_window_confidence": 0.12,
    "min_measured_windows": 8.0,
    "plateau_tolerance_sec": 0.025,
    "min_segment_sec": 90.0,
    "min_plateau_windows": 3.0,
    "max_plateaus": 48.0,
    "verification_window_sec": 20.0,
    "verification_step_sec": 60.0,
    "verification_margin": 0.01,
    "min_verification_windows": 6.0,
    "silence_rms": 1.0e-4,
}


def _same_container_settings(cfg: dict) -> dict[str, float]:
    settings = dict(DEFAULT_SAME_CONTAINER_VERIFICATION)
    for key, value in (cfg.get("same_container_verification") or {}).items():
        if key not in settings:
            continue
        try:
            settings[key] = float(value)
        except (TypeError, ValueError):
            continue
    return settings


def _read_proxy_window(
    handle: sf.SoundFile, start_sec: float, duration_sec: float
) -> np.ndarray:
    rate = int(handle.samplerate)
    start = int(round(start_sec * rate))
    if start < 0 or start >= len(handle):
        return np.zeros(0, dtype=np.float32)
    handle.seek(start)
    block = handle.read(
        max(0, int(round(duration_sec * rate))), dtype="float32", always_2d=True
    )
    if block.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.nan_to_num(np.mean(block, axis=1)).astype(np.float32)


def measure_same_container_offsets(
    original_proxy: Path,
    dubbed_proxy: Path,
    settings: dict[str, float],
    ctx: Any = None,
    *,
    duration_sec: float,
) -> list[dict]:
    """Measure the dubbed track's local delay against the original.

    A shared container guarantees a common time base, not a common edit.  A
    dubbed track licensed from another release keeps that release's splices, so
    its delay against the original holds for a reel and then steps by tens of
    milliseconds.  Only measurement separates the two cases, so the shared
    container is used for nothing except bounding the search radius.
    """
    chunk_sec = max(5.0, float(settings["chunk_sec"]))
    radius_sec = max(0.05, float(settings["search_radius_sec"]))
    silence = max(0.0, float(settings["silence_rms"]))
    minimum_confidence = float(settings["min_window_confidence"])
    entries: list[dict] = []
    total = max(0.0, float(duration_sec))
    planned = max(1, int(total // chunk_sec))
    with sf.SoundFile(str(original_proxy)) as original_file, sf.SoundFile(
        str(dubbed_proxy)
    ) as dubbed_file:
        rate = int(dubbed_file.samplerate)
        position = 0.0
        index = 0
        while position + chunk_sec <= total:
            if ctx is not None:
                _check_stop(ctx)
            dubbed_window = _read_proxy_window(dubbed_file, position, chunk_sec)
            original_window = _read_proxy_window(original_file, position, chunk_sec)
            size = min(dubbed_window.size, original_window.size)
            entry: dict[str, Any] = {
                "dubbed_start": round(position, 4),
                "duration": round(chunk_sec, 4),
                "delay": 0.0,
                "confidence": 0.0,
                "usable": False,
            }
            if size > rate:
                dubbed_window = dubbed_window[:size]
                original_window = original_window[:size]
                quietest = min(
                    float(
                        np.sqrt(np.mean(np.square(dubbed_window.astype(np.float64))))
                    ),
                    float(
                        np.sqrt(np.mean(np.square(original_window.astype(np.float64))))
                    ),
                )
                if quietest >= silence:
                    estimate = estimate_global_offset(
                        original_window, dubbed_window, rate, radius_sec
                    )
                    entry["delay"] = round(float(estimate.offset_sec), 6)
                    entry["confidence"] = round(float(estimate.confidence), 4)
                    entry["usable"] = bool(estimate.confidence >= minimum_confidence)
            entries.append(entry)
            position += chunk_sec
            index += 1
            if ctx is not None and index % 10 == 0:
                ctx.update(
                    stage="Сопоставление дорожек",
                    substage="Проверка общей временной шкалы",
                    progress=min(90.0, 5.0 + 85.0 * index / planned),
                    local_progress=min(100.0, 100.0 * index / planned),
                    processed_seconds=position,
                    total_seconds=total,
                )
    return entries


def summarize_same_container_offsets(
    entries: list[dict], settings: dict[str, float], matched: float
) -> dict:
    """Reduce measured windows to plateaus and the statistics describing them.

    A dub assembled from another release holds one delay for a whole reel and
    then steps at a splice, so the profile is piecewise constant rather than a
    line.  Whether those plateaus are worth applying is decided by measurement
    in ``score_offset_policies``, not here.
    """
    measured = [item for item in entries if item.get("usable")]
    verification: dict[str, Any] = {
        "analysed_windows": len(entries),
        "measured_windows": len(measured),
        "reason": "not_enough_measurable_windows",
        "offset_median_sec": 0.0,
        "offset_spread_sec": 0.0,
        "offset_noise_sec": 0.0,
        "plateau_residual_sec": 0.0,
        "plateaus": [],
    }
    if len(measured) < int(settings["min_measured_windows"]):
        return verification
    delays = np.asarray([float(item["delay"]) for item in measured], dtype=np.float64)
    median = float(np.median(delays))
    spread = float(np.percentile(delays, 90.0) - np.percentile(delays, 10.0))
    # Neighbouring windows straddle no splice almost everywhere, so the typical
    # step between them measures the estimator's own noise without being fooled
    # by the real structure that a median absolute deviation would pick up.
    noise = float(np.median(np.abs(np.diff(delays)))) if delays.size > 1 else 0.0
    plateaus = plateau_segments_from_offsets(
        measured,
        tolerance_sec=float(settings["plateau_tolerance_sec"]),
        min_segment_sec=float(settings["min_segment_sec"]),
        min_windows=int(settings["min_plateau_windows"]),
    )
    if len(plateaus) > int(settings["max_plateaus"]):
        # More splices than any real assembly has: the profile is measurement
        # noise, and grouping it would only dress that noise up as structure.
        plateaus = []
    residuals: list[float] = []
    for plateau in plateaus:
        inside = [
            float(item["delay"])
            for item in measured
            if float(plateau["dubbed_start"]) - 1e-6
            <= float(item["dubbed_start"])
            < float(plateau["dubbed_end"]) + 1e-6
        ]
        if inside:
            residuals.extend(
                np.abs(np.asarray(inside) - float(plateau["delay"])).tolist()
            )
    if plateaus:
        # Stretch the outer plateaus over stretches that were too quiet to
        # measure: they belong to the nearest measured reel, and leaving them
        # unmapped would drop real programme from processing.
        plateaus[0]["dubbed_start"] = 0.0
        plateaus[-1]["dubbed_end"] = round(matched, 4)
    verification.update(
        {
            "reason": "measured",
            "offset_median_sec": round(median, 6),
            "offset_spread_sec": round(spread, 6),
            "offset_noise_sec": round(noise, 6),
            "plateau_residual_sec": round(
                float(np.median(residuals)) if residuals else 0.0, 6
            ),
            "plateaus": plateaus,
        }
    )
    return verification


def _log_envelope(values: np.ndarray, rate: int, hop_ms: float = 5.0) -> np.ndarray:
    hop = max(1, int(round(rate * hop_ms / 1000.0)))
    count = len(values) // hop
    if count < 2:
        return np.zeros(0, dtype=np.float64)
    frames = values[: count * hop].astype(np.float64).reshape(count, hop)
    return np.log(np.sqrt(np.mean(np.square(frames), axis=1)) + 1e-7)


def _envelope_agreement(left: np.ndarray, right: np.ndarray, rate: int) -> float:
    """Correlate loudness contours, which survive a different mastering.

    Waveform correlation between an original and its dub is near zero even when
    the two are perfectly aligned, so it cannot referee an alignment decision;
    the loudness contour of the shared music and effects can.
    """
    size = min(len(left), len(right))
    if size < rate:
        return float("nan")
    first = _log_envelope(left[:size], rate)
    second = _log_envelope(right[:size], rate)
    count = min(first.size, second.size)
    if count < 8:
        return float("nan")
    first = first[:count] - float(np.mean(first[:count]))
    second = second[:count] - float(np.mean(second[:count]))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator > 0.0 else float("nan")


def score_offset_policies(
    original_proxy: Path,
    dubbed_proxy: Path,
    verification: dict,
    settings: dict[str, float],
    ctx: Any = None,
    *,
    duration_sec: float,
) -> dict:
    """Score doing nothing, one global shift, and the measured plateaus.

    The windows deliberately sit off the measurement grid and are judged by a
    different statistic than the one that produced the offsets, so a profile
    made of noise cannot certify itself.
    """
    window_sec = max(4.0, float(settings["verification_window_sec"]))
    step_sec = max(window_sec, float(settings["verification_step_sec"]))
    silence = max(0.0, float(settings["silence_rms"]))
    plateaus = verification.get("plateaus") or []
    global_delay = float(verification.get("offset_median_sec") or 0.0)

    def plateau_delay(position: float) -> float:
        for plateau in plateaus:
            if (
                float(plateau["dubbed_start"])
                <= position
                < float(plateau["dubbed_end"])
            ):
                return float(plateau["delay"])
        return global_delay

    scores: dict[str, list[float]] = {"zero": [], "global": [], "piecewise": []}
    with sf.SoundFile(str(original_proxy)) as original_file, sf.SoundFile(
        str(dubbed_proxy)
    ) as dubbed_file:
        rate = int(dubbed_file.samplerate)
        original_duration = len(original_file) / rate
        # Half a step in: never reuse a measurement window's own start.
        position = step_sec * 0.5
        while position + window_sec <= duration_sec:
            if ctx is not None:
                _check_stop(ctx)
            dubbed_window = _read_proxy_window(dubbed_file, position, window_sec)
            if dubbed_window.size < rate or float(
                np.sqrt(np.mean(np.square(dubbed_window.astype(np.float64))))
            ) < max(silence, 1e-3):
                position += step_sec
                continue
            for name, delay in (
                ("zero", 0.0),
                ("global", global_delay),
                ("piecewise", plateau_delay(position)),
            ):
                source = position - delay
                if source < 0.0 or source + window_sec > original_duration:
                    continue
                original_window = _read_proxy_window(
                    original_file, source, window_sec
                )
                value = _envelope_agreement(original_window, dubbed_window, rate)
                if math.isfinite(value):
                    scores[name].append(value)
            position += step_sec
    result = {
        f"{name}_agreement": round(float(np.median(values)), 4) if values else None
        for name, values in scores.items()
    }
    result["verification_windows"] = min(len(values) for values in scores.values())
    return result


def choose_same_container_policy(
    verification: dict, scores: dict, settings: dict[str, float]
) -> str:
    """Pick the timeline the measurement actually supports.

    ``identity`` is not a default here but a verdict: it wins only when moving
    the original does not measurably improve the agreement between the two
    tracks.
    """
    margin = max(0.0, float(settings["verification_margin"]))
    minimum = int(settings["min_verification_windows"])
    if scores.get("verification_windows", 0) < minimum:
        return "identity"
    zero = scores.get("zero_agreement")
    whole = scores.get("global_agreement")
    piecewise = scores.get("piecewise_agreement")
    if zero is None:
        return "identity"
    best = "identity"
    reference = float(zero)
    if whole is not None and float(whole) >= reference + margin:
        best, reference = "global", float(whole)
    if (
        piecewise is not None
        and len(verification.get("plateaus") or []) > 1
        and float(piecewise) >= reference + margin
    ):
        best = "piecewise"
    return best


def same_container_segments(
    verification: dict,
    matched: float,
    original_duration: float,
) -> tuple[list[dict], list[dict]]:
    """Build map segments and control points from measured plateaus."""
    segments: list[dict] = []
    checkpoints: list[dict] = []
    for plateau in verification.get("plateaus") or []:
        dubbed_start = max(0.0, float(plateau["dubbed_start"]))
        dubbed_end = min(matched, float(plateau["dubbed_end"]))
        delay = float(plateau["delay"])
        original_start = dubbed_start - delay
        original_end = dubbed_end - delay
        if original_end > original_duration:
            # The reel runs past the end of the original: keep the part that
            # still has a source instead of mapping onto silence.
            overshoot = original_end - original_duration
            dubbed_end -= overshoot
            original_end -= overshoot
        if dubbed_end - dubbed_start <= 1e-6:
            continue
        confidence = float(plateau.get("confidence") or 0.0)
        checkpoints.append(
            {
                "original_start": round(original_start, 6),
                "dubbed_start": round(dubbed_start, 6),
                "confidence": round(confidence, 6),
                "second_score": 0.0,
                "window_sec": 0.0,
                "speed_ratio": 1.0,
                "usable": True,
                "reason": None,
                "kind": "same_container_measured",
            }
        )
        segments.append(
            {
                "id": len(segments),
                "original_start": round(original_start, 6),
                "original_end": round(original_end, 6),
                "dubbed_start": round(dubbed_start, 6),
                "dubbed_end": round(dubbed_end, 6),
                "duration": round(dubbed_end - dubbed_start, 6),
                "delay": round(delay, 6),
                "offset": round(delay, 6),
                # Every plateau is a pure shift: the container fixes the clock,
                # so a tempo change measured here would be an artefact.
                "speed_ratio": 1.0,
                "confidence": round(confidence, 6),
                "correlation": round(confidence, 6),
                "status": "usable",
                "usable": True,
                "reason": None,
                "checkpoint_start": len(checkpoints) - 1,
                "checkpoint_end": len(checkpoints),
                "measured_windows": int(plateau.get("windows") or 0),
            }
        )
    if segments:
        last = segments[-1]
        checkpoints.append(
            {
                "original_start": round(float(last["original_end"]), 6),
                "dubbed_start": round(float(last["dubbed_end"]), 6),
                "confidence": round(float(last["confidence"]), 6),
                "second_score": 0.0,
                "window_sec": 0.0,
                "speed_ratio": 1.0,
                "usable": True,
                "reason": None,
                "kind": "same_container_measured_end",
            }
        )
        tail = matched - float(last["dubbed_end"])
        # A sliver left over by trimming the last reel to the original's end is
        # not unmatched programme; recording it would only add a degenerate
        # interval to the map and a warning about a tenth of a second.
        if tail > 0.25:
            segments.append(
                {
                    "id": len(segments),
                    "original_start": round(float(last["original_end"]), 6),
                    "original_end": round(original_duration, 6),
                    "dubbed_start": round(float(last["dubbed_end"]), 6),
                    "dubbed_end": round(matched, 6),
                    "duration": round(tail, 6),
                    "delay": round(float(last["delay"]), 6),
                    "offset": round(float(last["delay"]), 6),
                    "speed_ratio": 1.0,
                    "confidence": 0.0,
                    "correlation": 0.0,
                    "status": "unusable",
                    "usable": False,
                    "reason": "участок за пределами оригинала",
                    "checkpoint_start": len(checkpoints) - 1,
                    "checkpoint_end": None,
                }
            )
    return segments, checkpoints


def align_pair(store: Store, project_id: str, pair_id: str, ctx: Any) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("2", {}).get("status") != "completed":
        raise RuntimeError("Сначала полностью извлеките обе дорожки.")
    root = store.pair_dir(project_id, pair_id)
    cfg = store.cfg["alignment"]
    original_proxy_path = root / "extracted" / "original_proxy.wav"
    dubbed_proxy_path = root / "extracted" / "dubbed_proxy.wav"
    original_info = sf.info(str(original_proxy_path))
    dubbed_info = sf.info(str(dubbed_proxy_path))
    original_duration = float(original_info.duration)
    dubbed_duration = float(dubbed_info.duration)
    original_source = os.path.normcase(
        str(Path(pair["sources"]["original"]["path"]).resolve())
    )
    dubbed_source = os.path.normcase(
        str(Path(pair["sources"]["dubbed"]["path"]).resolve())
    )
    if original_source == dubbed_source:
        matched = min(original_duration, dubbed_duration)
        unmatched = max(0.0, dubbed_duration - matched)
        settings = _same_container_settings(cfg)
        ctx.update(
            stage="Сопоставление дорожек",
            substage="Проверка общей временной шкалы",
            progress=5.0,
            local_progress=0.0,
            processed_seconds=0.0,
            total_seconds=matched,
            current_file=pair["name"],
        )
        # One container fixes the clock both tracks are stamped on and nothing
        # else.  A dubbed track licensed from another release carries that
        # release's splices, so its delay against the original holds for a reel
        # and then steps.  Asserting identity here published confidence 1.0 for
        # pairs that drift by a tenth of a second, and every later stage — the
        # music and effects bed above all — inherited the error in silence.
        profile = measure_same_container_offsets(
            original_proxy_path,
            dubbed_proxy_path,
            settings,
            ctx,
            duration_sec=matched,
        )
        verification = summarize_same_container_offsets(profile, settings, matched)
        ctx.update(
            stage="Сопоставление дорожек",
            substage="Проверка измеренных смещений",
            progress=92.0,
            current_file=pair["name"],
        )
        scores = score_offset_policies(
            original_proxy_path,
            dubbed_proxy_path,
            verification,
            settings,
            ctx,
            duration_sec=matched,
        )
        policy = choose_same_container_policy(verification, scores, settings)
        verification["scores"] = scores
        verification["policy"] = policy
        segments = []
        checkpoints = []
        if policy == "piecewise":
            segments, checkpoints = same_container_segments(
                verification, matched, original_duration
            )
        elif policy == "global":
            whole = dict(verification)
            whole["plateaus"] = [
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": round(matched, 4),
                    "delay": round(float(verification["offset_median_sec"]), 6),
                    "confidence": float(
                        scores.get("global_agreement") or 0.0
                    ),
                    "windows": int(verification["measured_windows"]),
                }
            ]
            segments, checkpoints = same_container_segments(
                whole, matched, original_duration
            )
        measured_map = bool(segments)
        if measured_map:
            method = "same_container_measured_offsets_v1"
            matched_duration = sum(
                float(item["duration"]) for item in segments if item["usable"]
            )
            unmatched = max(0.0, dubbed_duration - matched_duration)
            stage_message = (
                "Дорожки из одного файла; дубляж смещён, карта построена "
                "по измеренным смещениям."
            )
        else:
            method = "same_container_identity_v1"
            matched_duration = matched
            if matched > 0:
                segments.append(
                    {
                        "original_start": 0.0,
                        "original_end": round(matched, 6),
                        "dubbed_start": 0.0,
                        "dubbed_end": round(matched, 6),
                        "duration": round(matched, 6),
                        "delay": 0.0,
                        "offset": 0.0,
                        "speed_ratio": 1.0,
                        "confidence": 1.0,
                        "correlation": 1.0,
                        "status": "usable",
                        "usable": True,
                        "reason": None,
                        "checkpoint_start": 0,
                        "checkpoint_end": 1,
                    }
                )
            if unmatched > 0:
                segments.append(
                    {
                        "original_start": round(matched, 6),
                        "original_end": round(original_duration, 6),
                        "dubbed_start": round(matched, 6),
                        "dubbed_end": round(dubbed_duration, 6),
                        "duration": round(unmatched, 6),
                        "delay": 0.0,
                        "offset": 0.0,
                        "speed_ratio": 1.0,
                        "confidence": 0.0,
                        "correlation": 0.0,
                        "status": "unusable",
                        "usable": False,
                        "reason": "конец одной дорожки выходит за границы другой",
                        "checkpoint_start": 1,
                        "checkpoint_end": None,
                    }
                )
            checkpoints = [
                {
                    "original_start": 0.0,
                    "dubbed_start": 0.0,
                    "confidence": 1.0,
                    "second_score": 0.0,
                    "window_sec": 0.0,
                    "speed_ratio": 1.0,
                    "usable": True,
                    "reason": None,
                    "kind": "same_container_start",
                },
                {
                    "original_start": round(matched, 6),
                    "dubbed_start": round(matched, 6),
                    "confidence": 1.0,
                    "second_score": 0.0,
                    "window_sec": 0.0,
                    "speed_ratio": 1.0,
                    "usable": True,
                    "reason": None,
                    "kind": "same_container_end",
                },
            ]
            stage_message = (
                "Дорожки из одного файла сопоставлены по общей временной шкале."
            )
        alignment_map = {
            "schema_version": 2,
            "created_at": utc_now(),
            "method": method,
            "global": {
                "offset_sec": round(float(verification["offset_median_sec"]), 6),
                "confidence": 1.0 if not measured_map else round(
                    float(np.median([item["confidence"] for item in segments])), 6
                ),
                "speed_ratio": 1.0,
                "first_original_sec": 0.0,
                "first_dubbed_sec": 0.0,
            },
            "summary": {
                "original_duration_sec": round(original_duration, 6),
                "dubbed_duration_sec": round(dubbed_duration, 6),
                "matched_duration_sec": round(matched_duration, 6),
                "unmatched_duration_sec": round(unmatched, 6),
                "matched_ratio": round(
                    matched_duration / max(dubbed_duration, 1e-9), 6
                ),
                "usable_segments": sum(item["usable"] for item in segments),
                "unusable_segments": sum(not item["usable"] for item in segments),
                "control_point_count": len(checkpoints),
                "continuous_full_track_scan": False,
                "same_container_verification": {
                    key: value
                    for key, value in verification.items()
                    if key != "plateaus"
                },
            },
            "control_points": checkpoints,
            "segments": segments,
            "chunks": segments,
        }
        destination = root / "alignment" / "alignment_map.json"
        atomic_json(destination, alignment_map)
        pair["alignment"] = alignment_map["summary"]
        pair["current_stage"] = max(int(pair.get("current_stage", 2)), 3)
        pair["stages"]["3"] = {"status": "completed", "message": stage_message}
        pair["stages"]["4"] = {"status": "available", "message": "Проверьте карту и контрольные точки."}
        pair["stages"]["5"] = {"status": "blocked", "message": "Сначала подтвердите проверку сопоставления."}
        pair["warnings"] = [
            warning
            for warning in pair.get("warnings", [])
            if not warning.startswith("Дубляж смещён относительно оригинала:")
        ]
        if measured_map:
            delays = [
                float(item["delay"]) * 1000.0 for item in segments if item["usable"]
            ]
            pair["warnings"].append(
                "Дубляж смещён относительно оригинала: "
                f"{len(delays)} участк(ов), смещения от {min(delays):.0f} до "
                f"{max(delays):.0f} мс. Карта построена по измерению; общая "
                "ручная поправка такие ступени не исправляет."
            )
        store.save_pair(pair)
        agreement = (
            f"согласие дорожек {scores.get('zero_agreement')} → "
            f"{scores.get(('piecewise' if policy == 'piecewise' else 'global') + '_agreement')}"
            if measured_map
            else f"согласие дорожек {scores.get('zero_agreement')}"
        )
        if measured_map:
            ctx.log(
                "Оригинал и перевод из одного файла, но дубляж смещён: "
                f"измерено {verification['measured_windows']} окон из "
                f"{verification['analysed_windows']}, разброс смещений "
                f"{verification['offset_spread_sec'] * 1000.0:.0f} мс, "
                f"построено {len(segments)} участк(ов), {agreement}."
            )
        else:
            ctx.log(
                "Оригинал и перевод взяты из одного файла: общая временная "
                "шкала подтверждена измерением "
                f"({verification['measured_windows']} окон, медиана "
                f"{verification['offset_median_sec'] * 1000.0:.0f} мс, "
                f"{agreement})."
            )
        ctx.update(
            stage="Сопоставление дорожек",
            substage=(
                "Один файл: измеренные смещения"
                if measured_map
                else "Один файл: общая временная шкала"
            ),
            progress=100.0,
            local_progress=100.0,
            processed_seconds=dubbed_duration,
            total_seconds=dubbed_duration,
            current_file=pair["name"],
        )
        return {"alignment_map": str(destination), "summary": alignment_map["summary"]}
    try:
        first = _find_first_sparse_match(
            original_proxy_path, dubbed_proxy_path, cfg, ctx
        )
    except RuntimeError as error:
        message = str(error)
        duration_delta = abs(original_duration - dubbed_duration)
        close_limit = max(30.0, min(original_duration, dubbed_duration) * 0.02)
        speed_ratio = original_duration / max(dubbed_duration, 1e-9)
        can_fallback = (
            (
                "Надёжное совпадение выбранных дорожек не найдено" in message
                or "Первое надёжное совпадение не найдено" in message
            )
            and duration_delta <= close_limit
            and abs(speed_ratio - 1.0) <= float(cfg["max_speed_deviation"])
        )
        if not can_fallback:
            raise

        confidence = float(cfg.get("approved_map_min_measured_confidence", 0.05))
        matched = min(dubbed_duration, original_duration / max(speed_ratio, 1e-9))
        segments = [
            {
                "id": 0,
                "original_start": 0.0,
                "original_end": round(matched * speed_ratio, 6),
                "dubbed_start": 0.0,
                "dubbed_end": round(matched, 6),
                "duration": round(matched, 6),
                "delay": 0.0,
                "offset": 0.0,
                "speed_ratio": round(speed_ratio, 8),
                "confidence": round(confidence, 6),
                "correlation": round(confidence, 6),
                "status": "usable",
                "usable": True,
                "reason": None,
                "checkpoint_start": 0,
                "checkpoint_end": 1,
            }
        ]
        unmatched = max(0.0, dubbed_duration - matched)
        checkpoints = [
            {
                "original_start": 0.0,
                "dubbed_start": 0.0,
                "confidence": round(confidence, 6),
                "second_score": 0.0,
                "window_sec": 0.0,
                "speed_ratio": round(speed_ratio, 8),
                "usable": True,
                "reason": "приближённая карта по длительности после неудачного поиска первого совпадения",
                "kind": "duration_fallback_start",
            },
            {
                "original_start": round(matched * speed_ratio, 6),
                "dubbed_start": round(matched, 6),
                "confidence": round(confidence, 6),
                "second_score": 0.0,
                "window_sec": 0.0,
                "speed_ratio": round(speed_ratio, 8),
                "usable": True,
                "reason": "приближённая карта по длительности после неудачного поиска первого совпадения",
                "kind": "duration_fallback_end",
            },
        ]
        alignment_map = {
            "schema_version": 2,
            "created_at": utc_now(),
            "method": "duration_fallback_after_failed_first_match_v1",
            "fallback_reason": message,
            "global": {
                "offset_sec": 0.0,
                "confidence": round(confidence, 6),
                "speed_ratio": round(speed_ratio, 8),
                "first_original_sec": 0.0,
                "first_dubbed_sec": 0.0,
            },
            "summary": {
                "original_duration_sec": round(original_duration, 6),
                "dubbed_duration_sec": round(dubbed_duration, 6),
                "matched_duration_sec": round(matched, 6),
                "unmatched_duration_sec": round(unmatched, 6),
                "matched_ratio": round(matched / max(dubbed_duration, 1e-9), 6),
                "usable_segments": 1,
                "unusable_segments": 0 if unmatched <= 1e-6 else 1,
                "control_point_count": len(checkpoints),
                "continuous_full_track_scan": False,
                "low_confidence_fallback": True,
            },
            "control_points": checkpoints,
            "segments": segments,
            "chunks": segments,
        }
        destination = root / "alignment" / "alignment_map.json"
        atomic_json(destination, alignment_map)
        pair["alignment"] = alignment_map["summary"]
        pair["current_stage"] = max(int(pair.get("current_stage", 2)), 3)
        pair["stages"]["3"] = {
            "status": "completed",
            "message": "Надёжное совпадение не найдено; применена приближённая карта по длительности.",
        }
        pair["stages"]["4"] = {
            "status": "available",
            "message": "Проверьте карту и контрольные точки: сопоставление низкой уверенности.",
        }
        pair["stages"]["5"] = {"status": "blocked", "message": "Сначала подтвердите проверку сопоставления."}
        pair["warnings"] = [
            warning
            for warning in pair.get("warnings", [])
            if not warning.startswith("Сопоставление низкой уверенности:")
        ]
        pair["warnings"].append(
            "Сопоставление низкой уверенности: точное первое совпадение не найдено, "
            "но длительности почти одинаковые, поэтому применена приближённая карта 1:1. "
            "Проверьте превью и выбранные дорожки."
        )
        store.save_pair(pair)
        ctx.log(
            "Первое надёжное совпадение не найдено; применена приближённая карта "
            f"по длительности, speed_ratio={speed_ratio:.8f}, delta={duration_delta:.3f} с."
        )
        ctx.update(
            stage="Сопоставление дорожек",
            substage=(
                "Резервное сопоставление по длительности: "
                "требуется проверка в превью"
            ),
            progress=100.0,
            local_progress=100.0,
            processed_seconds=dubbed_duration,
            total_seconds=dubbed_duration,
            current_file=pair["name"],
        )
        return {"alignment_map": str(destination), "summary": alignment_map["summary"]}
    first.update({"usable": True, "reason": None, "kind": "initial"})
    global_offset = first["dubbed_start"] - first["original_start"]
    global_confidence = float(first["confidence"])
    initial_speed_ratio = float(first.get("speed_ratio") or 1.0)
    if first.get("acceptance") == "distinctive":
        ctx.log(
            "Первое совпадение принято по высокой уникальности: "
            f"оценка {global_confidence:.3f}, "
            f"следующий кандидат {float(first.get('second_score') or 0.0):.3f}."
        )
    ctx.log(
        f"Первое совпадение: оригинал {first['original_start']:.3f} с, "
        f"перевод {first['dubbed_start']:.3f} с, "
        f"смещение {global_offset:.3f} с, уверенность {global_confidence:.3f}."
    )
    proxy_sr = int(store.cfg["audio"]["proxy_sample_rate"])
    window_sec = float(cfg["verification_window_sec"])
    last_verification_start = min(
        max(first["dubbed_start"], dubbed_duration - window_sec),
        first["dubbed_start"]
        + max(0.0, original_duration - first["original_start"] - window_sec),
    )
    requested = max(0, int(cfg["verification_points"]))
    if requested:
        raw_points = np.linspace(
            min(first["dubbed_start"] + window_sec, last_verification_start),
            last_verification_start,
            requested,
        )
        verification_times = sorted(
            {
                round(float(value), 6)
                for value in raw_points
                if value > first["dubbed_start"] + 0.01
            }
        )
    else:
        verification_times = []
    checkpoints = [first]
    for point_index, dubbed_start in enumerate(verification_times):
        _check_stop(ctx)
        usable = [item for item in checkpoints if item["usable"]]
        if len(usable) >= 2:
            xs = np.asarray([item["dubbed_start"] for item in usable])
            ys = np.asarray([item["original_start"] for item in usable])
            slope, intercept = np.polyfit(xs, ys, 1)
            expected_original = float(intercept + slope * dubbed_start)
        else:
            expected_original = first["original_start"] + (
                dubbed_start - first["dubbed_start"]
            ) * initial_speed_ratio
        checkpoint = _verify_sparse_point(
            original_proxy_path,
            dubbed_proxy_path,
            dubbed_start,
            expected_original,
            cfg,
            proxy_sr,
            initial_speed_ratio,
        )
        if checkpoint["original_start"] <= checkpoints[-1]["original_start"] - 0.5:
            checkpoint["usable"] = False
            checkpoint["reason"] = "немонотонная контрольная точка"
        checkpoints.append(checkpoint)
        ctx.update(
            stage="Разреженная проверка сопоставления",
            substage=f"Контрольная точка {point_index + 1} из {len(verification_times)}",
            progress=35.0
            + (point_index + 1) / max(len(verification_times), 1) * 65.0,
            local_progress=(point_index + 1)
            / max(len(verification_times), 1)
            * 100.0,
            processed_seconds=dubbed_start,
            total_seconds=dubbed_duration,
            current_file=pair["name"],
        )
    segments = _segments_from_sparse_points(
        checkpoints, original_duration, dubbed_duration, cfg
    )
    matched = sum(item["duration"] for item in segments if item["usable"])
    unmatched = max(0.0, dubbed_duration - matched)
    speed_values = [item["speed_ratio"] for item in segments if item["usable"]]
    alignment_map = {
        "schema_version": 2,
        "created_at": utc_now(),
        "method": "first_match_sparse_verification_v3",
        "global": {
            "offset_sec": round(global_offset, 6),
            "confidence": round(global_confidence, 6),
            "speed_ratio": round(float(np.median(speed_values)) if speed_values else 1.0, 8),
            "first_original_sec": round(float(first["original_start"]), 6),
            "first_dubbed_sec": round(float(first["dubbed_start"]), 6),
        },
        "summary": {
            "original_duration_sec": round(original_duration, 6),
            "dubbed_duration_sec": round(dubbed_duration, 6),
            "matched_duration_sec": round(matched, 6),
            "unmatched_duration_sec": round(unmatched, 6),
            "matched_ratio": round(matched / max(dubbed_duration, 1e-9), 6),
            "usable_segments": sum(item["usable"] for item in segments),
            "unusable_segments": sum(not item["usable"] for item in segments),
            "control_point_count": len(checkpoints),
            "continuous_full_track_scan": False,
        },
        "control_points": checkpoints,
        "segments": segments,
        "chunks": segments,
    }
    destination = root / "alignment" / "alignment_map.json"
    atomic_json(destination, alignment_map)
    pair["alignment"] = alignment_map["summary"]
    pair["current_stage"] = max(int(pair.get("current_stage", 2)), 3)
    pair["stages"]["3"] = {"status": "completed", "message": "Кусочная карта сопоставления построена."}
    pair["stages"]["4"] = {"status": "available", "message": "Проверьте карту и контрольные точки."}
    pair["stages"]["5"] = {"status": "blocked", "message": "Сначала подтвердите проверку сопоставления."}
    if unmatched > 0:
        pair["warnings"] = [
            warning
            for warning in pair.get("warnings", [])
            if not warning.startswith("Несопоставленный материал:")
        ]
        pair["warnings"].append(
            f"Несопоставленный материал: {unmatched:.1f} с. На этих участках вычитание будет отключено."
        )
    store.save_pair(pair)
    ctx.update(
        stage="Сопоставление дорожек",
        substage="Карта готова",
        progress=100.0,
        local_progress=100.0,
        processed_seconds=dubbed_duration,
        total_seconds=dubbed_duration,
        current_file=pair["name"],
    )
    return {"alignment_map": str(destination), "summary": alignment_map["summary"]}


def _read_interval(
    path: Path, start_sec: float, duration_sec: float, channels: int | None = None
) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    start_frame = int(round(max(0.0, start_sec) * info.samplerate))
    frames = max(0, int(round(duration_sec * info.samplerate)))
    prefix = max(0, int(round(-min(0.0, start_sec) * info.samplerate)))
    with sf.SoundFile(str(path), "r") as handle:
        handle.seek(min(start_frame, handle.frames))
        data = handle.read(max(0, frames - prefix), dtype="float32", always_2d=True)
    if prefix:
        data = np.vstack([np.zeros((prefix, info.channels), dtype=np.float32), data])
    if len(data) < frames:
        data = np.vstack(
            [data, np.zeros((frames - len(data), info.channels), dtype=np.float32)]
        )
    data = data[:frames]
    if channels is not None:
        data = audio_io.match_channels(data, channels).astype(np.float32)
    return data, int(info.samplerate)


def _resample_exact(data: np.ndarray, source_sr: int, target_sr: int, target_len: int) -> np.ndarray:
    if source_sr != target_sr:
        data = audio_io.resample(data, source_sr, target_sr)
    if len(data) != target_len:
        data = signal.resample(data, target_len, axis=0)
    return np.nan_to_num(data, copy=False).astype(np.float32)


def _run_speech_extractor(
    store: Store,
    source: Path,
    destination: Path,
    ctx: Any,
    pair_name: str,
    stage: str,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
    progress_base: float = 0.0,
    progress_span: float = 99.0,
    model_role: str = "original",
) -> None:
    model_cfg = store.cfg["speech_extraction"]
    python = root_path(model_cfg["python"])
    runner = root_path(model_cfg["runner"])
    checkpoint_key = (
        "dubbed_checkpoint_dir" if model_role == "dubbed" else "checkpoint_dir"
    )
    checkpoint = root_path(
        model_cfg.get(checkpoint_key) or model_cfg["checkpoint_dir"]
    )
    for required in (python, runner, checkpoint / "last_best_checkpoint"):
        if not required.exists():
            raise RuntimeError(f"Компонент модели выделения речи не найден: {required}")
    command = [
        str(python),
        str(runner),
        "--input", str(source),
        "--output", str(destination),
        "--checkpoint-dir", str(checkpoint),
        "--start", f"{max(0.0, start_sec):.6f}",
        "--duration", f"{max(0.0, duration_sec):.6f}",
        "--chunk-sec", str(float(model_cfg.get("chunk_sec", 20.0))),
        "--context-sec", str(float(model_cfg.get("context_sec", 1.0))),
        "--min-cuda-free-gb",
        str(
            float(
                (store.cfg.get("compute") or {}).get(
                    "speech_extraction_min_cuda_free_gb", 2.5
                )
            )
        ),
        "--device",
        str((store.cfg.get("compute") or {}).get("device", "auto")),
    ]
    if model_cfg.get("force_cpu"):
        command.append("--cpu")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ctx.child_pid = process.pid
    tail: list[str] = []
    started = time.monotonic()
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            _check_stop(ctx)
            line = raw.strip()
            if not line:
                continue
            tail.append(line)
            tail = tail[-100:]
            if line.startswith("[status]"):
                ctx.update(stage=stage, substage=line.removeprefix("[status]").strip())
            elif line.startswith("[progress]"):
                try:
                    info = json.loads(line.removeprefix("[progress]").strip())
                    processed = float(info["processed_sec"])
                    total = float(info["total_sec"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                elapsed = max(time.monotonic() - started, 0.001)
                eta = max(0.0, (total - processed) / max(processed / elapsed, 1e-6))
                ctx.update(
                    stage=stage,
                    substage=f"Блок {info.get('block')} из {info.get('blocks')}",
                    progress=min(
                        progress_base + progress_span,
                        progress_base + processed / max(total, 0.001) * progress_span,
                    ),
                    local_progress=min(
                        100.0,
                        processed / max(total, 0.001) * 100.0,
                    ),
                    processed_seconds=processed,
                    total_seconds=total,
                    eta_seconds=eta,
                    current_file=pair_name,
                )
        code = process.wait()
    except BaseException:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        ctx.child_pid = None
    if code != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(
            "Модель выделения речи завершилась с ошибкой.\n" + "\n".join(tail)[-5000:]
        )


def _trial_segment(
    alignment_map: dict[str, Any],
    requested_start: Any,
    requested_duration: float,
) -> tuple[dict[str, Any], float, float, float]:
    segments = [item for item in alignment_map.get("segments", []) if item.get("usable")]
    if not segments:
        raise RuntimeError("В карте нет уверенного участка для проверки.")
    if requested_start is None or str(requested_start).strip() == "":
        eligible = [
            item for item in segments
            if float(item["dubbed_end"]) - float(item["dubbed_start"]) >= 1.0
        ]
        if not eligible:
            raise RuntimeError("Нет уверенного участка продолжительностью хотя бы 1 секунду.")
        segment = max(
            eligible,
            key=lambda item: (
                float(item["dubbed_end"]) - float(item["dubbed_start"]),
                float(item.get("confidence", 0.0)),
            ),
        )
        duration = min(
            requested_duration,
            float(segment["dubbed_end"]) - float(segment["dubbed_start"]),
        )
        dubbed_start = float(segment["dubbed_start"]) + max(
            0.0,
            (float(segment["dubbed_end"]) - float(segment["dubbed_start"]) - duration) / 2.0,
        )
    else:
        dubbed_start = max(0.0, float(requested_start))
        segment = next(
            (
                item for item in segments
                if float(item["dubbed_start"]) <= dubbed_start < float(item["dubbed_end"])
            ),
            None,
        )
        if segment is None:
            raise ValueError("В выбранном времени нет уверенного участка карты.")
        duration = min(requested_duration, float(segment["dubbed_end"]) - dubbed_start)
    if duration < 1.0:
        raise ValueError("Для проверки нужно не менее 1 секунды уверенного материала.")
    speed = float(segment.get("speed_ratio") or 1.0)
    correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    original_start = float(segment["original_start"]) + (
        dubbed_start - float(segment["dubbed_start"])
    ) * speed + correction
    return segment, dubbed_start, original_start, duration


def _processing_alignment_map(
    store: Store, pair: dict[str, Any], alignment_map: dict[str, Any]
) -> dict[str, Any]:
    """Use smooth low-score anchors only after the user accepted the map."""
    if not pair.get("alignment_review", {}).get("accepted"):
        return alignment_map
    result = json.loads(json.dumps(alignment_map))
    cfg = store.cfg["alignment"]
    minimum = float(cfg.get("approved_map_min_measured_confidence", 0.05))
    floor = float(cfg.get("approved_map_processing_confidence", 0.45))
    max_speed_delta = float(cfg.get("max_speed_deviation", 0.06))
    overrides = 0
    rejected_unsafe_speed = 0
    segments = result.get("segments", [])
    original_duration = float(
        (result.get("summary") or {}).get("original_duration_sec") or 0.0
    )
    previous_usable: dict[str, Any] | None = None
    for segment in segments:
        measured = float(segment.get("confidence", 0.0))
        reason = str(segment.get("reason") or "")
        raw_speed = segment.get("speed_ratio")
        try:
            speed = float(1.0 if raw_speed is None else raw_speed)
        except (TypeError, ValueError):
            speed = float("nan")
        safe_speed = bool(
            math.isfinite(speed)
            and speed > 0.0
            and abs(speed - 1.0) <= max_speed_delta
        )
        if segment.get("usable") and not safe_speed:
            # Old maps can already contain an unsafe interval marked usable.
            # Never trust that flag by itself: every consumer must receive a
            # map whose local tempo stays inside the configured safety bound.
            segment["usable"] = False
            segment["status"] = "unsafe_speed_ratio"
            segment["reason"] = (
                "аномальный коэффициент временного сопоставления; "
                "участок исключён из обработки"
            )
            segment["processing_override_rejected"] = "unsafe_speed_ratio"
            rejected_unsafe_speed += 1
        elif (
            not segment.get("usable")
            and measured >= minimum
            and reason == "низкая уверенность контрольной точки"
            and safe_speed
        ):
            segment["measured_confidence"] = measured
            segment["confidence"] = max(measured, floor)
            segment["usable"] = True
            segment["status"] = "approved_interpolation"
            segment["reason"] = (
                "низкая автоматическая уверенность; карта вручную подтверждена"
            )
            segment["processing_override"] = True
            overrides += 1
        elif (
            not segment.get("usable")
            and measured >= minimum
            and reason == "низкая уверенность контрольной точки"
            and not safe_speed
        ):
            # Accepting a map may relax only confidence.  It must never turn a
            # grossly implausible local tempo estimate into processing input:
            # doing so stretches a short source interval over a whole preview
            # (and previously produced clearly slowed or accelerated audio).
            segment["processing_override_rejected"] = "unsafe_speed_ratio"
            rejected_unsafe_speed += 1
        elif (
            not segment.get("usable")
            and previous_usable is not None
            and all(
                key in previous_usable
                for key in ("dubbed_start", "original_start", "speed_ratio")
            )
            and all(key in segment for key in ("dubbed_start", "dubbed_end"))
            and reason
            in {
                "низкая уверенность контрольной точки",
                "конец одной версии выходит за границы другой",
            }
        ):
            # A single weak checkpoint in end credits used to invalidate the
            # entire final verification interval (up to ~18% of a film).  For a
            # user-approved map, safely extend the last confirmed affine map
            # only while it still fits inside the original track.
            speed = float(previous_usable.get("speed_ratio") or 1.0)
            dubbed_start = float(segment["dubbed_start"])
            dubbed_end = float(segment["dubbed_end"])
            previous_dubbed_start = float(previous_usable["dubbed_start"])
            previous_original_start = float(previous_usable["original_start"])
            expected_original_start = previous_original_start + (
                dubbed_start - previous_dubbed_start
            ) * speed
            expected_original_end = expected_original_start + (
                dubbed_end - dubbed_start
            ) * speed
            if (
                dubbed_end > dubbed_start
                and expected_original_start >= 0.0
                and (
                    original_duration <= 0.0
                    or expected_original_end <= original_duration + 0.25
                )
            ):
                segment["measured_confidence"] = measured
                segment["original_start"] = expected_original_start
                segment["original_end"] = expected_original_end
                segment["speed_ratio"] = speed
                segment["confidence"] = floor
                segment["usable"] = True
                segment["status"] = "approved_tail_extrapolation"
                segment["reason"] = (
                    "хвост продолжен по последнему подтверждённому сдвигу и темпу"
                )
                segment["processing_override"] = True
                overrides += 1
        if segment.get("usable"):
            previous_usable = segment
    # A single false control point creates two complementary tempo spikes: one
    # segment is much too fast and the next much too slow.  When both are
    # bounded by trusted, continuous segments and the affine bridge between
    # those trusted boundaries is itself safe, discard only that bad interior
    # anchor.  This preserves the real programme instead of either stretching
    # it grotesquely or replacing a long part of the film with silence.
    repaired_segments: list[dict[str, Any]] = []
    repaired_unsafe_runs = 0
    index = 0
    while index < len(segments):
        current = segments[index]
        if current.get("processing_override_rejected") != "unsafe_speed_ratio":
            repaired_segments.append(current)
            index += 1
            continue
        run_end = index
        while (
            run_end < len(segments)
            and segments[run_end].get("processing_override_rejected")
            == "unsafe_speed_ratio"
        ):
            run_end += 1
        run = segments[index:run_end]
        previous = repaired_segments[-1] if repaired_segments else None
        following = segments[run_end] if run_end < len(segments) else None
        can_bridge = bool(
            # Exactly two complementary spikes are the signature of one bad
            # interior control point. Longer runs are ambiguous and remain
            # unusable rather than being guessed through.
            len(run) == 2
            and previous
            and previous.get("usable")
            and following
            and following.get("usable")
        )
        if can_bridge:
            dubbed_start = float(run[0]["dubbed_start"])
            dubbed_end = float(run[-1]["dubbed_end"])
            previous_dubbed_end = float(previous.get("dubbed_end") or 0.0)
            following_dubbed_start = float(
                following.get("dubbed_start") or 0.0
            )
            run_contiguous = all(
                abs(
                    float(left.get("dubbed_end") or 0.0)
                    - float(right.get("dubbed_start") or 0.0)
                )
                <= 0.01
                for left, right in zip(run[:-1], run[1:])
            )
            contiguous = bool(
                abs(previous_dubbed_end - dubbed_start) <= 0.01
                and abs(following_dubbed_start - dubbed_end) <= 0.01
                and run_contiguous
            )
            previous_speed = float(previous.get("speed_ratio") or 1.0)
            following_speed = float(following.get("speed_ratio") or 1.0)
            original_start = float(previous["original_start"]) + (
                dubbed_start - float(previous["dubbed_start"])
            ) * previous_speed
            original_end = float(following["original_start"]) + (
                dubbed_end - following_dubbed_start
            ) * following_speed
            duration = dubbed_end - dubbed_start
            bridge_speed = (
                (original_end - original_start) / duration
                if duration > 0.0
                else float("nan")
            )
            try:
                first_raw_speed = float(run[0]["speed_ratio"])
                second_raw_speed = float(run[1]["speed_ratio"])
                raw_original_contiguous = abs(
                    float(run[0]["original_end"])
                    - float(run[1]["original_start"])
                ) <= 0.01
                raw_outer_boundaries_match = bool(
                    abs(float(run[0]["original_start"]) - original_start)
                    <= 0.05
                    and abs(float(run[-1]["original_end"]) - original_end)
                    <= 0.05
                )
                complementary_spikes = bool(
                    math.isfinite(first_raw_speed)
                    and math.isfinite(second_raw_speed)
                    and abs(first_raw_speed - 1.0) > max_speed_delta
                    and abs(second_raw_speed - 1.0) > max_speed_delta
                    and (first_raw_speed - 1.0)
                    * (second_raw_speed - 1.0)
                    < 0.0
                )
            except (KeyError, TypeError, ValueError):
                raw_original_contiguous = False
                raw_outer_boundaries_match = False
                complementary_spikes = False
            safe_bridge = bool(
                contiguous
                and raw_original_contiguous
                and raw_outer_boundaries_match
                and complementary_spikes
                and duration > 0.0
                and math.isfinite(previous_speed)
                and abs(previous_speed - 1.0) <= max_speed_delta
                and math.isfinite(following_speed)
                and abs(following_speed - 1.0) <= max_speed_delta
                and math.isfinite(bridge_speed)
                and bridge_speed > 0.0
                and abs(bridge_speed - 1.0) <= max_speed_delta
                and original_start >= -0.25
                and original_end > original_start
                and (
                    original_duration <= 0.0
                    or original_end <= original_duration + 0.25
                )
            )
            if safe_bridge:
                repaired_segments.append(
                    {
                        "id": (
                            f"safe_bridge_{run[0].get('id', index)}_"
                            f"{run[-1].get('id', run_end - 1)}"
                        ),
                        "original_start": round(original_start, 6),
                        "original_end": round(original_end, 6),
                        "dubbed_start": round(dubbed_start, 6),
                        "dubbed_end": round(dubbed_end, 6),
                        "duration": round(duration, 6),
                        "delay": round(dubbed_start - original_start, 6),
                        "offset": round(dubbed_start - original_start, 6),
                        "speed_ratio": round(bridge_speed, 8),
                        "confidence": round(
                            min(
                                float(previous.get("confidence") or floor),
                                float(following.get("confidence") or floor),
                            ),
                            6,
                        ),
                        "correlation": round(
                            min(
                                float(previous.get("confidence") or floor),
                                float(following.get("confidence") or floor),
                            ),
                            6,
                        ),
                        "status": "approved_safe_bridge",
                        "usable": True,
                        "reason": (
                            "аномальная низкоуверенная контрольная точка "
                            "отброшена; использовано сопоставление между "
                            "соседними подтверждёнными границами"
                        ),
                        "processing_override": True,
                        "replaced_segment_ids": [
                            item.get("id") for item in run
                        ],
                    }
                )
                repaired_unsafe_runs += 1
                index = run_end
                continue
        repaired_segments.extend(run)
        index = run_end
    segments = repaired_segments
    result["segments"] = segments
    result.setdefault("summary", {})["approved_processing_overrides"] = overrides
    result["summary"]["processing_max_speed_deviation"] = max_speed_delta
    result["summary"]["approved_unsafe_speed_rejections"] = (
        rejected_unsafe_speed
    )
    result["summary"]["approved_safe_speed_bridges"] = repaired_unsafe_runs
    controls = [
        item
        for item in result.get("control_points", [])
        if item.get("usable")
        and item.get("dubbed_start") is not None
        and item.get("original_start") is not None
    ]
    # Segments that were measured reel by reel are the answer, not scatter to
    # be smoothed: collapsing them re-introduces exactly the offsets they exist
    # to remove.
    measured_offsets = str(result.get("method") or "").startswith(
        "same_container_measured_offsets"
    )
    affine_residual_limit = float(
        cfg.get("global_affine_max_residual_sec", 0.05)
    )
    if len(controls) >= 3 and not measured_offsets:
        dubbed_points = np.asarray(
            [float(item["dubbed_start"]) for item in controls], dtype=np.float64
        )
        original_points = np.asarray(
            [float(item["original_start"]) for item in controls], dtype=np.float64
        )
        speed, intercept = np.polyfit(dubbed_points, original_points, 1)
        residuals = np.abs(original_points - (intercept + speed * dubbed_points))
        maximum_residual = float(np.max(residuals))
        # A fifth of a second of scatter is not evidence of "no edit
        # discontinuities" — at that size the residual *is* the discontinuity,
        # and the collapse used to bless it as one clean mapping.
        if abs(float(speed) - 1.0) <= 0.0015 and maximum_residual <= affine_residual_limit:
            summary = result.setdefault("summary", {})
            dubbed_duration = float(summary.get("dubbed_duration_sec") or 0.0)
            original_duration = float(summary.get("original_duration_sec") or 0.0)
            start = max(0.0, -float(intercept) / max(float(speed), 1e-9))
            end = dubbed_duration
            if original_duration > 0.0:
                source_limited_end = (
                    (original_duration - float(intercept))
                    / max(float(speed), 1e-9)
                )
                if dubbed_duration - source_limited_end > 0.25:
                    end = min(end, source_limited_end)
            collapsed: list[dict[str, Any]] = []
            if start > 1e-6:
                collapsed.append(
                    {
                        "dubbed_start": 0.0,
                        "dubbed_end": start,
                        "original_start": 0.0,
                        "speed_ratio": float(speed),
                        "confidence": 0.0,
                        "usable": False,
                        "reason": "до начала глобально сопоставленного оригинала",
                    }
                )
            if end > start:
                collapsed.append(
                    {
                        "dubbed_start": start,
                        "dubbed_end": end,
                        "original_start": float(intercept + speed * start),
                        "original_end": float(intercept + speed * end),
                        "speed_ratio": float(speed),
                        "confidence": float(
                            np.median(
                                [
                                    float(item.get("confidence") or 0.0)
                                    for item in controls
                                ]
                            )
                        ),
                        "usable": True,
                        "status": "global_affine_from_verified_points",
                        "reason": (
                            "одно глобальное сопоставление; контрольные точки "
                            "не показали монтажных разрывов"
                        ),
                    }
                )
            if dubbed_duration - end > 1e-6:
                collapsed.append(
                    {
                        "dubbed_start": end,
                        "dubbed_end": dubbed_duration,
                        "original_start": original_duration,
                        "speed_ratio": float(speed),
                        "confidence": 0.0,
                        "usable": False,
                        "reason": "оригинал закончился раньше версии с переводом",
                    }
                )
            result["segments"] = collapsed
            matched = max(0.0, end - start)
            summary.update(
                {
                    "processing_mode": "global_affine",
                    "global_offset_sec": round(float(intercept), 6),
                    "global_speed_ratio": round(float(speed), 9),
                    "global_fit_max_residual_sec": round(maximum_residual, 6),
                    "matched_duration_sec": round(matched, 6),
                    "unmatched_duration_sec": round(
                        max(0.0, dubbed_duration - matched), 6
                    ),
                    "matched_ratio": round(
                        matched / max(dubbed_duration, 1e-9), 6
                    ),
                    "usable_segments": sum(
                        1 for item in collapsed if item.get("usable")
                    ),
                    "unusable_segments": sum(
                        1 for item in collapsed if not item.get("usable")
                    ),
                }
            )
    return result


def _speech_activity_score(data: np.ndarray, sample_rate: int) -> float:
    """Estimate whether a short film window contains foreground dialogue."""
    values = audio_io.ensure_2d(np.asarray(data, dtype=np.float32))
    if values.shape[1] >= 3:
        mono = values[:, 2]
    else:
        mono = audio_io.to_mono(values)
    if mono.size < max(64, sample_rate // 4):
        return -1_000.0
    target_rate = min(12_000, sample_rate)
    if target_rate != sample_rate:
        mono = signal.resample_poly(
            mono,
            target_rate,
            sample_rate,
        ).astype(np.float32)
        sample_rate = target_rate
    mono = mono - float(np.mean(mono))
    full_rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    if full_rms < 1e-5:
        return -1_000.0
    nyquist = sample_rate / 2.0
    low = 120.0 / nyquist
    high = min(4_500.0, nyquist * 0.95) / nyquist
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    speech_band = signal.sosfiltfilt(sos, mono).astype(np.float32)
    frame = max(1, int(sample_rate * 0.025))
    hop = max(1, int(sample_rate * 0.010))
    if len(speech_band) < frame:
        return -1_000.0
    frame_count = 1 + (len(speech_band) - frame) // hop
    windows = np.lib.stride_tricks.sliding_window_view(speech_band, frame)[::hop]
    windows = windows[:frame_count]
    frame_rms = np.sqrt(
        np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12
    )
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-9))
    p20, p50, p85 = np.percentile(frame_db, [20, 50, 85])
    active_ratio = float(np.mean(frame_db >= max(p20 + 6.0, -48.0)))
    band_rms = float(
        np.sqrt(np.mean(speech_band.astype(np.float64) ** 2) + 1e-12)
    )
    band_ratio_db = 20.0 * math.log10(max(band_rms / full_rms, 1e-6))
    dynamics = min(30.0, max(0.0, float(p85 - p20)))
    return (
        float(p50)
        + 0.35 * dynamics
        + 8.0 * active_ratio
        + 0.5 * band_ratio_db
    )


def _automatic_trial_start(
    alignment_map: dict[str, Any],
    original_path: Path,
    dubbed_path: Path,
    requested_duration: float,
    ctx: Any,
    pair_name: str,
) -> tuple[float, dict[str, Any]]:
    """Pick a bounded, speech-rich window instead of an arbitrary long segment."""
    usable = [
        item
        for item in alignment_map.get("segments", [])
        if item.get("usable")
        and float(item["dubbed_end"]) - float(item["dubbed_start"]) >= 1.0
    ]
    if not usable:
        raise RuntimeError("В карте нет уверенного участка для проверки.")
    total_usable = sum(
        float(item["dubbed_end"]) - float(item["dubbed_start"]) for item in usable
    )
    maximum_candidates = 24
    candidates: list[tuple[dict[str, Any], float, float]] = []
    for item in usable:
        segment_start = float(item["dubbed_start"])
        segment_end = float(item["dubbed_end"])
        segment_duration = segment_end - segment_start
        window_duration = min(requested_duration, segment_duration)
        available = max(0.0, segment_duration - window_duration)
        proportional = int(
            round(maximum_candidates * segment_duration / max(total_usable, 1e-6))
        )
        count = max(1, min(maximum_candidates, proportional))
        if available <= 0.001 or count == 1:
            starts = [segment_start + available / 2.0]
        else:
            starts = np.linspace(segment_start, segment_start + available, count)
        for start in starts:
            candidates.append((item, float(start), window_duration))
    if len(candidates) > maximum_candidates:
        indexes = np.linspace(0, len(candidates) - 1, maximum_candidates).round().astype(int)
        candidates = [candidates[int(index)] for index in indexes]
    correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    best: tuple[float, float, dict[str, Any]] | None = None
    for index, (item, dubbed_start, duration) in enumerate(candidates):
        _check_stop(ctx)
        speed = float(item.get("speed_ratio") or 1.0)
        original_start = float(item["original_start"]) + (
            dubbed_start - float(item["dubbed_start"])
        ) * speed + correction
        original_data, original_sr = _read_interval(
            original_path,
            max(0.0, original_start),
            max(1.0, duration * speed),
        )
        dubbed_data, dubbed_sr = _read_interval(
            dubbed_path,
            dubbed_start,
            max(1.0, duration),
        )
        original_score = _speech_activity_score(original_data, original_sr)
        dubbed_score = _speech_activity_score(dubbed_data, dubbed_sr)
        score = 0.55 * original_score + 0.45 * dubbed_score
        candidate = (
            score,
            float(item.get("confidence", 0.0)),
            {
                "dubbed_start_sec": dubbed_start,
                "original_start_sec": original_start,
                "duration_sec": duration,
                "speech_activity_score": score,
                "original_activity_score": original_score,
                "dubbed_activity_score": dubbed_score,
                "candidate_count": len(candidates),
            },
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        ctx.update(
            stage="Выбор разговорного фрагмента",
            substage=f"Проверен участок {index + 1} из {len(candidates)}",
            progress=min(20.0, (index + 1) / max(len(candidates), 1) * 20.0),
            current_file=pair_name,
        )
    assert best is not None
    return float(best[2]["dubbed_start_sec"]), best[2]


def _automatic_demo_windows(
    alignment_map: dict[str, Any],
    original_path: Path,
    dubbed_path: Path,
    duration: float,
    count: int,
    ctx: Any,
    pair_name: str,
) -> list[dict[str, Any]]:
    """Choose one speech-rich aligned window from each part of the film."""
    usable = [
        item
        for item in alignment_map.get("segments", [])
        if item.get("usable")
        and float(item["dubbed_end"]) - float(item["dubbed_start"]) >= duration
    ]
    if not usable:
        raise RuntimeError(
            f"Нет уверенных участков длительностью {duration:.0f} секунд для превью."
        )
    timeline_end = max(float(item["dubbed_end"]) for item in usable)
    correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    selected: list[dict[str, Any]] = []
    minimum_gap = min(
        max(duration * 2.0, 60.0),
        max(duration, timeline_end / max(count, 1) * 0.8),
    )
    candidates_per_zone = 6
    for zone in range(count):
        zone_start = timeline_end * zone / count
        zone_end = timeline_end * (zone + 1) / count
        candidates: list[tuple[dict[str, Any], float]] = []
        for item in usable:
            start = max(float(item["dubbed_start"]), zone_start)
            end = min(float(item["dubbed_end"]), zone_end)
            latest = end - duration
            if latest < start:
                continue
            for value in np.linspace(start, latest, candidates_per_zone):
                candidates.append((item, float(value)))
        if not candidates:
            for item in usable:
                latest = float(item["dubbed_end"]) - duration
                if latest >= float(item["dubbed_start"]):
                    candidates.append(
                        (
                            item,
                            float(item["dubbed_start"])
                            + (latest - float(item["dubbed_start"])) / 2.0,
                        )
                    )
        candidates = [
            candidate
            for candidate in candidates
            if all(
                abs(candidate[1] - float(previous["dubbed_start_sec"]))
                >= minimum_gap
                for previous in selected
            )
        ]
        if not candidates:
            continue
        best: tuple[float, float, dict[str, Any]] | None = None
        for candidate_index, (item, dubbed_start) in enumerate(candidates):
            _check_stop(ctx)
            speed = float(item.get("speed_ratio") or 1.0)
            original_start = float(item["original_start"]) + (
                dubbed_start - float(item["dubbed_start"])
            ) * speed + correction
            original_data, original_sr = _read_interval(
                original_path,
                max(0.0, original_start),
                duration * speed,
            )
            dubbed_data, dubbed_sr = _read_interval(
                dubbed_path,
                dubbed_start,
                duration,
            )
            original_score = _speech_activity_score(original_data, original_sr)
            dubbed_score = _speech_activity_score(dubbed_data, dubbed_sr)
            score = 0.55 * original_score + 0.45 * dubbed_score
            value = {
                "id": f"demo_{zone + 1:02d}",
                "zone": zone + 1,
                "dubbed_start_sec": dubbed_start,
                "original_start_sec": original_start,
                "duration_sec": duration,
                "speed_ratio": speed,
                "alignment_confidence": float(item.get("confidence", 0.0)),
                "speech_activity_score": score,
                "original_activity_score": original_score,
                "dubbed_activity_score": dubbed_score,
            }
            ranked = (score, value["alignment_confidence"], value)
            if best is None or ranked[:2] > best[:2]:
                best = ranked
            checked = zone * candidates_per_zone + candidate_index + 1
            total = count * candidates_per_zone
            ctx.update(
                stage="Выбор пяти разговорных сцен",
                substage=(
                    f"Часть {zone + 1}/{count}, "
                    f"вариант {candidate_index + 1}/{len(candidates)}"
                ),
                progress=min(12.0, checked / max(total, 1) * 12.0),
                current_file=pair_name,
            )
        if best is None:
            raise RuntimeError(
                f"Не удалось подобрать разговорный фрагмент в части {zone + 1}."
            )
        selected.append(best[2])
    if len(selected) < count:
        fallback_candidates: list[tuple[float, float, dict[str, Any]]] = []
        fallback_gap = duration if timeline_end < 300.0 else minimum_gap
        for item in usable:
            segment_start = float(item["dubbed_start"])
            latest = float(item["dubbed_end"]) - duration
            if latest < segment_start:
                continue
            for dubbed_start in np.arange(
                segment_start,
                latest + 1e-6,
                max(duration, fallback_gap),
            ):
                if any(
                    abs(float(dubbed_start) - float(previous["dubbed_start_sec"]))
                    < fallback_gap
                    for previous in selected
                ):
                    continue
                speed = float(item.get("speed_ratio") or 1.0)
                original_start = float(item["original_start"]) + (
                    float(dubbed_start) - float(item["dubbed_start"])
                ) * speed + correction
                original_data, original_sr = _read_interval(
                    original_path,
                    max(0.0, original_start),
                    duration * speed,
                )
                dubbed_data, dubbed_sr = _read_interval(
                    dubbed_path,
                    float(dubbed_start),
                    duration,
                )
                original_score = _speech_activity_score(original_data, original_sr)
                dubbed_score = _speech_activity_score(dubbed_data, dubbed_sr)
                score = 0.55 * original_score + 0.45 * dubbed_score
                value = {
                    "id": "",
                    "zone": None,
                    "dubbed_start_sec": float(dubbed_start),
                    "original_start_sec": original_start,
                    "duration_sec": duration,
                    "speed_ratio": speed,
                    "alignment_confidence": float(item.get("confidence", 0.0)),
                    "speech_activity_score": score,
                    "original_activity_score": original_score,
                    "dubbed_activity_score": dubbed_score,
                }
                fallback_candidates.append(
                    (score, value["alignment_confidence"], value)
                )
        for _score, _confidence, value in sorted(
            fallback_candidates, key=lambda item: item[:2], reverse=True
        ):
            if len(selected) >= count:
                break
            if any(
                abs(float(value["dubbed_start_sec"]) - float(previous["dubbed_start_sec"]))
                < fallback_gap
                for previous in selected
            ):
                continue
            selected.append(value)
    if len(selected) != count:
        raise RuntimeError(
            f"Удалось найти только {len(selected)} из {count} уникальных "
            f"разговорных сцен с интервалом не менее {minimum_gap:.0f} секунд."
        )
    selected.sort(key=lambda item: float(item["dubbed_start_sec"]))
    for index, item in enumerate(selected, 1):
        item["id"] = f"demo_{index:02d}"
        item["zone"] = index
    return selected


def extract_original_speech_test(
    store: Store, project_id: str, pair_id: str, ctx: Any, parameters: dict[str, Any]
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("4", {}).get("status") != "completed":
        raise RuntimeError("Сначала подтвердите карту сопоставления.")
    root = store.pair_dir(project_id, pair_id)
    amap = _processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )
    default_duration = float(store.cfg["speech_extraction"].get("trial_duration_sec", 15.0))
    duration = min(60.0, max(5.0, float(parameters.get("duration_sec", default_duration))))
    requested_start = parameters.get("start_sec")
    selection = {"mode": "manual"}
    if requested_start is None or str(requested_start).strip() == "":
        requested_start, selection = _automatic_trial_start(
            amap,
            root / "extracted" / "original.flac",
            root / "extracted" / "dubbed.flac",
            duration,
            ctx,
            pair["name"],
        )
        selection["mode"] = "automatic_speech_activity"
    segment, dubbed_start, original_start, duration = _trial_segment(
        amap, requested_start, duration
    )
    output_root = root / "previews" / "speech_test"
    output_root.mkdir(parents=True, exist_ok=True)
    speech = output_root / "original_en_speech.flac"
    _run_speech_extractor(
        store,
        root / "extracted" / "original.flac",
        speech,
        ctx,
        pair["name"],
        "Пробное выделение английской речи",
        original_start,
        duration * float(segment.get("speed_ratio") or 1.0),
    )
    original_mix = output_root / "original_mix.flac"
    _clip(root / "extracted" / "original.flac", original_start, duration, original_mix)
    original_data, original_sr = _read_interval(original_mix, 0.0, duration)
    speech_data, speech_sr = _read_interval(speech, 0.0, duration, 1)
    speech_data = _resample_exact(
        speech_data, speech_sr, original_sr, len(original_data)
    )
    original_me_data = _direct_dialogue_cancel(
        speech_data,
        original_data,
        original_sr,
        1.0,
        store.cfg["adaptive_cancel"],
    ).residual
    original_me = output_root / "original_me.flac"
    sf.write(
        str(original_me),
        np.clip(original_me_data, -1.0, 1.0),
        original_sr,
        format="FLAC",
        subtype="PCM_16",
    )
    manifest = {
        "schema_version": 5,
        "created_at": utc_now(),
        "backend": store.cfg["speech_extraction"]["backend"],
        "film_name": pair["name"],
        "dubbed_start_sec": dubbed_start,
        "original_start_sec": original_start,
        "duration_sec": duration,
        "alignment_confidence": float(segment.get("confidence", 0.0)),
        "selection": selection,
        "files": {
            "original_mix": str(original_mix.resolve()),
            "original_speech": str(speech.resolve()),
            "original_me": str(original_me.resolve()),
        },
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["speech_test"] = manifest
    pair["speech_test_review"] = {
        "accepted": False,
        "reviewed_at": None,
        "message": "Прослушайте обе дорожки и подтвердите пробу.",
    }
    for key in (
        "components_test",
        "components_test_review",
        "components",
        "previews",
        "model_review",
        "result",
    ):
        pair.pop(key, None)
    pair["pipeline_version"] = 5
    pair["stages"]["5"] = {
        "status": "available",
        "message": "Проба EN-речи готова. Прослушайте её перед обработкой всего оригинала.",
    }
    for number, message in (
        ("6", "Сначала подтвердите пробу этапа 5."),
        ("7", "Сначала выберите метод и запустите сборку."),
        ("8", "Сначала завершите предпросмотры."),
        ("9", "Сначала подготовьте аудиоматериалы."),
        ("10", "Сначала выполните сборку выбранным методом."),
    ):
        pair["stages"][number] = {"status": "blocked", "message": message}
    store.save_pair(pair)
    return manifest


def extract_original_speech_pair(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    progress_base: float = 0.0,
    progress_span: float = 99.0,
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("4", {}).get("status") != "completed":
        raise RuntimeError("Сначала подтвердите карту сопоставления.")
    if not pair.get("speech_test"):
        raise RuntimeError("Сначала выполните пробное выделение EN-речи.")
    if not pair.get("speech_test_review", {}).get("accepted"):
        raise RuntimeError("Сначала прослушайте и явно подтвердите результат пробы.")
    root = store.pair_dir(project_id, pair_id)
    source = root / "extracted" / "original.flac"
    output_root = root / "stems" / "original"
    output_root.mkdir(parents=True, exist_ok=True)
    speech = output_root / "en_speech.flac"
    speech_span = progress_span * 0.70
    me_span = max(0.0, progress_span - speech_span)
    existing_speech = Path(str(pair.get("original_speech", {}).get("speech", "")))
    if speech.is_file() and speech.stat().st_size > 0:
        existing_speech = speech
    if existing_speech.is_file() and existing_speech.stat().st_size > 0:
        if existing_speech.resolve() != speech.resolve():
            _atomic_copy(existing_speech, speech)
        ctx.update(
            stage="Разделение оригинала",
            substage="Используется уже выделенная полная EN-речь",
            progress=progress_base + speech_span,
            current_file=pair["name"],
        )
    else:
        _run_speech_extractor(
            store,
            source,
            speech,
            ctx,
            pair["name"],
            "Выделение EN-речи из оригинала",
            progress_base=progress_base,
            progress_span=speech_span,
        )
    project = store.load_project(project_id)
    speech_only_training = "чистый перевод" in str(project.get("name", "")).lower()
    original_me = output_root / "music_and_effects.flac"
    if speech_only_training:
        ctx.update(
            stage="Речевая основа оригинала готова",
            substage="Музыка и эффекты не строятся для речевого датасета",
            progress=progress_base + progress_span,
            current_file=pair["name"],
        )
    else:
        _build_original_me_file(
            store,
            source,
            speech,
            original_me,
            ctx,
            pair["name"],
            progress_base=progress_base + speech_span,
            progress_span=me_span,
        )
    manifest = {
        "schema_version": 5,
        "created_at": utc_now(),
        "backend": store.cfg["speech_extraction"]["backend"],
        "source": str(source.resolve()),
        "speech": str(speech.resolve()),
        "music_effects": str(original_me.resolve()) if original_me.is_file() else None,
        "duration_sec": audio_io.probe_duration(speech),
        "sample_rate": 48000,
        "channels": 1,
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["original_speech"] = manifest
    pair["pipeline_version"] = 5
    pair["current_stage"] = max(int(pair.get("current_stage", 4)), 5)
    pair["stages"]["5"] = {
        "status": "completed",
        "message": (
            "Английская речь готова для речевого датасета."
            if speech_only_training
            else "Оригинал разделён на EN-речь и M&E."
        ),
    }
    pair["stages"]["6"] = {
        "status": "available",
        "message": (
            "Метод выбран; можно запустить сборку."
            if pair.get("components_test_review", {}).get("selected_method")
            in {"direct", "stem"}
            else "Создайте пять разговорных превью и выберите метод A или B."
        ),
    }
    store.save_pair(pair)
    return manifest


def extract_clean_translation_speech_pair(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
) -> dict:
    """Extract the Russian-only speech target from a clean translated track."""
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("4", {}).get("status") != "completed":
        raise RuntimeError("Сначала подтвердите карту сопоставления.")
    root = store.pair_dir(project_id, pair_id)
    source = root / "extracted" / "dubbed.flac"
    output_root = root / "stems" / "clean_translation"
    output_root.mkdir(parents=True, exist_ok=True)
    speech = output_root / "ru_speech.flac"
    _run_speech_extractor(
        store,
        source,
        speech,
        ctx,
        pair["name"],
        "Выделение русской речи из чистой переводной дорожки",
        model_role="dubbed",
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "backend": store.cfg["speech_extraction"]["backend"],
        "source": str(source.resolve()),
        "speech": str(speech.resolve()),
        "duration_sec": audio_io.probe_duration(speech),
        "sample_rate": 48000,
        "channels": 1,
        "role": "clean_russian_training_target",
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["clean_translation_speech"] = manifest
    store.save_pair(pair)
    return manifest


def _adaptive_result(
    predictor: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    confidence: float,
    ac: dict[str, Any],
):
    curve = np.full(len(target), confidence, dtype=np.float32)
    return adaptive_cancel.segmented_adaptive_cancel(
        predictor,
        target,
        sample_rate,
        curve,
        stft_n_fft=int(ac["stft_n_fft"]),
        stft_hop=int(ac["stft_hop"]),
        block_frames=int(ac["block_frames"]),
        gain_smoothing=float(ac["gain_smoothing"]),
        min_coherence=float(ac["min_coherence"]),
        regularization=float(ac["regularization"]),
        max_gain_linear=float(ac["max_gain_linear"]),
    )


def _dialogue_to_channels(dialogue: np.ndarray, channels: int) -> np.ndarray:
    """Place mono dialogue in the conventional center channel of surround audio."""
    mono = audio_io.to_mono(audio_io.ensure_2d(dialogue)).astype(np.float32)
    if channels <= 1:
        return mono[:, None]
    if channels == 2:
        return np.repeat(mono[:, None], 2, axis=1)
    result = np.zeros((len(mono), channels), dtype=np.float32)
    result[:, 2] = mono
    return result


def _direct_dialogue_cancel(
    predictor: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    confidence: float,
    ac: dict[str, Any],
):
    """Cancel dialogue from the center channel for surround, normally for mono/stereo."""
    target = audio_io.ensure_2d(target).astype(np.float32)
    if target.shape[1] < 3:
        return _adaptive_result(
            audio_io.to_mono(audio_io.ensure_2d(predictor))[:, None],
            target,
            sample_rate,
            confidence,
            ac,
        )
    center = target[:, 2:3]
    center_result = _adaptive_result(
        audio_io.to_mono(audio_io.ensure_2d(predictor))[:, None],
        center,
        sample_rate,
        confidence,
        ac,
    )
    predicted = np.zeros_like(target)
    predicted[:, 2] = center_result.predicted_common[:, 0]
    residual = target - predicted
    return adaptive_cancel.CancelResult(
        predicted_common=predicted,
        residual=residual.astype(np.float32),
        mean_coherence=float(center_result.mean_coherence),
        gated_fraction=float(center_result.gated_fraction),
        clipping_ratio=float(np.mean(np.abs(residual) >= 0.999)),
        bypassed=bool(center_result.bypassed),
    )


def _build_original_me_file(
    store: Store,
    original_path: Path,
    speech_path: Path,
    destination: Path,
    ctx: Any,
    pair_name: str,
    progress_base: float,
    progress_span: float,
) -> None:
    info = sf.info(str(original_path))
    sample_rate, channels, total = (
        int(info.samplerate),
        int(info.channels),
        float(info.duration),
    )
    chunk_sec = max(
        5.0, float(store.cfg["adaptive_cancel"].get("streaming_chunk_sec", 20.0))
    )
    temporary = _temporary(destination)
    writer = sf.SoundFile(
        str(temporary),
        "w",
        samplerate=sample_rate,
        channels=channels,
        format="FLAC",
        subtype="PCM_16",
    )
    processed = 0.0
    try:
        blocks = max(1, int(math.ceil(total / chunk_sec)))
        for block in range(blocks):
            _check_stop(ctx)
            start = block * chunk_sec
            duration = min(chunk_sec, total - start)
            original, original_sr = _read_interval(
                original_path, start, duration, channels
            )
            target_len = int(round(duration * sample_rate))
            original = _resample_exact(
                original, original_sr, sample_rate, target_len
            )
            speech, speech_sr = _read_interval(speech_path, start, duration, 1)
            speech = _resample_exact(speech, speech_sr, sample_rate, target_len)
            result = _direct_dialogue_cancel(
                speech,
                original,
                sample_rate,
                1.0,
                store.cfg["adaptive_cancel"],
            )
            writer.write(np.clip(result.residual, -1.0, 1.0))
            processed += duration
            ctx.update(
                stage="Построение оригинального M&E",
                substage=f"Блок {block + 1} из {blocks}",
                progress=progress_base
                + min(
                    progress_span,
                    processed / max(total, 1e-6) * progress_span,
                ),
                processed_seconds=processed,
                total_seconds=total,
                current_file=pair_name,
            )
    finally:
        writer.close()
    os.replace(temporary, destination)


def _component_variants(
    original_me: np.ndarray,
    en_speech: np.ndarray,
    dubbed_speech: np.ndarray,
    sample_rate: int,
    confidence: float,
    ac: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Remove EN speech from the EN+RU speech stem and rebuild with original M&E."""
    channels = audio_io.ensure_2d(original_me).shape[1]
    en_mono = audio_io.to_mono(audio_io.ensure_2d(en_speech))[:, None]
    dubbed_mono = audio_io.to_mono(audio_io.ensure_2d(dubbed_speech))[:, None]
    en_on_dubbed_speech = _adaptive_result(
        en_mono, dubbed_mono, sample_rate, confidence, ac
    )
    ru_voice = _dialogue_to_channels(en_on_dubbed_speech.residual, channels)
    aligned_en = _dialogue_to_channels(
        en_on_dubbed_speech.predicted_common, channels
    )
    final = (original_me + ru_voice).astype(np.float32)
    return (
        {
            "original_me": original_me.astype(np.float32),
            "en_speech": _dialogue_to_channels(en_mono, channels),
            "dubbed_speech_en_ru": _dialogue_to_channels(dubbed_mono, channels),
            "aligned_en_speech": aligned_en,
            "ru_voice": ru_voice,
            "final_ru_plus_me": final,
        },
        {
            "en_voice_cancel_coherence": float(
                en_on_dubbed_speech.mean_coherence
            ),
            "combined_speech_rms_db": _rms_db(dubbed_speech),
            "ru_voice_rms_db": _rms_db(ru_voice),
            "final_rms_db": _rms_db(final),
        },
    )


def build_components_test(
    store: Store, project_id: str, pair_id: str, ctx: Any
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if not pair.get("speech_test") or not pair.get(
        "speech_test_review", {}
    ).get("accepted"):
        raise RuntimeError("Сначала подготовьте и подтвердите пробу EN-речи на этапе 5.")
    root = store.pair_dir(project_id, pair_id)
    amap = _processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )
    duration = max(
        1.0,
        float(
            store.cfg["speech_extraction"].get(
                "comparison_demo_duration_sec", 30.0
            )
        ),
    )
    demo_count = max(
        1,
        int(store.cfg["speech_extraction"].get("comparison_demo_count", 5)),
    )
    demos = _automatic_demo_windows(
        amap,
        root / "extracted" / "original.flac",
        root / "extracted" / "dubbed.flac",
        duration,
        demo_count,
        ctx,
        pair["name"],
    )
    output_root = root / "previews" / "components_test"
    output_root.mkdir(parents=True, exist_ok=True)
    original_info = sf.info(str(root / "extracted" / "original.flac"))
    dubbed_info = sf.info(str(root / "extracted" / "dubbed.flac"))
    sample_rate = int(dubbed_info.samplerate)
    original_channels = int(original_info.channels)
    dubbed_channels = int(dubbed_info.channels)
    gap_samples = sample_rate
    original_batch: list[np.ndarray] = []
    dubbed_batch: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for index, demo in enumerate(demos):
        target_len = int(round(duration * sample_rate))
        original_mix, original_sr = _read_interval(
            root / "extracted" / "original.flac",
            float(demo["original_start_sec"]),
            duration * float(demo["speed_ratio"]),
            original_channels,
        )
        original_mix = _resample_exact(
            original_mix, original_sr, sample_rate, target_len
        )
        dubbed_mix, dubbed_sr = _read_interval(
            root / "extracted" / "dubbed.flac",
            float(demo["dubbed_start_sec"]),
            duration,
            dubbed_channels,
        )
        dubbed_mix = _resample_exact(
            dubbed_mix, dubbed_sr, sample_rate, target_len
        )
        original_batch.append(original_mix)
        dubbed_batch.append(dubbed_mix)
        ranges.append((cursor, cursor + target_len))
        cursor += target_len
        if index < len(demos) - 1:
            original_batch.append(
                np.zeros((gap_samples, original_channels), dtype=np.float32)
            )
            dubbed_batch.append(
                np.zeros((gap_samples, dubbed_channels), dtype=np.float32)
            )
            cursor += gap_samples
    batch_workspace = tempfile.TemporaryDirectory(
        prefix="speech_demo_", dir=str(output_root)
    )
    batch_root = Path(batch_workspace.name)
    batch_original_mix = batch_root / "original_mix.flac"
    batch_dubbed_mix = batch_root / "dubbed_mix.flac"
    sf.write(
        str(batch_original_mix),
        np.concatenate(original_batch, axis=0),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    sf.write(
        str(batch_dubbed_mix),
        np.concatenate(dubbed_batch, axis=0),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    batch_en_speech = batch_root / "en_speech.flac"
    batch_dubbed_speech = batch_root / "speech_en_ru.flac"
    _run_speech_extractor(
        store,
        batch_original_mix,
        batch_en_speech,
        ctx,
        pair["name"],
        "Выделение английской речи для пяти превью",
        progress_base=12.0,
        progress_span=37.0,
    )
    _run_speech_extractor(
        store,
        batch_dubbed_mix,
        batch_dubbed_speech,
        ctx,
        pair["name"],
        "Выделение речи оригинала и перевода для пяти превью",
        progress_base=50.0,
        progress_span=37.0,
        model_role="dubbed",
    )
    en_batch, en_sr = _read_interval(
        batch_en_speech, 0.0, cursor / sample_rate, 1
    )
    en_batch = _resample_exact(en_batch, en_sr, sample_rate, cursor)
    dubbed_speech_batch, dubbed_speech_sr = _read_interval(
        batch_dubbed_speech, 0.0, cursor / sample_rate, 1
    )
    dubbed_speech_batch = _resample_exact(
        dubbed_speech_batch, dubbed_speech_sr, sample_rate, cursor
    )
    ac = store.cfg["adaptive_cancel"]
    demo_manifests: list[dict[str, Any]] = []
    for index, (demo, (start, end)) in enumerate(zip(demos, ranges)):
        _check_stop(ctx)
        target_len = end - start
        original_mix = original_batch[index * 2]
        dubbed_mix = dubbed_batch[index * 2]
        en_mono = en_batch[start:end]
        dubbed_speech_mono = dubbed_speech_batch[start:end]
        original_me_result = _direct_dialogue_cancel(
            en_mono,
            original_mix,
            sample_rate,
            float(demo["alignment_confidence"]),
            ac,
        )
        original_me = audio_io.match_channels(
            original_me_result.residual, dubbed_channels
        ).astype(np.float32)
        direct = _direct_dialogue_cancel(
            en_mono,
            dubbed_mix,
            sample_rate,
            float(demo["alignment_confidence"]),
            ac,
        )
        stem_files, stem_metrics = _component_variants(
            original_me,
            en_mono,
            dubbed_speech_mono,
            sample_rate,
            float(demo["alignment_confidence"]),
            ac,
        )
        files_data = {
            "original_mix": original_mix,
            "dubbed_mix": dubbed_mix,
            "en_speech": stem_files["en_speech"],
            "original_me": original_me,
            "dubbed_speech_en_ru": stem_files["dubbed_speech_en_ru"],
            "direct_aligned_en_speech": direct.predicted_common,
            "direct_result_ru_plus_me": direct.residual,
            "stem_aligned_en_speech": stem_files["aligned_en_speech"],
            "stem_ru_voice": stem_files["ru_voice"],
            "stem_result_ru_plus_me": stem_files["final_ru_plus_me"],
        }
        demo_root = output_root / str(demo["id"])
        demo_root.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {}
        for key, data in files_data.items():
            path = demo_root / f"{key}.flac"
            temporary = _temporary(path)
            sf.write(
                str(temporary),
                np.clip(data, -1.0, 1.0),
                sample_rate,
                format="FLAC",
                subtype="PCM_16",
            )
            os.replace(temporary, path)
            files[key] = str(path.resolve())
        demo_manifests.append(
            {
                **demo,
                "files": files,
                "metrics": {
                    "direct_cancel_coherence": float(direct.mean_coherence),
                    "direct_result_rms_db": _rms_db(direct.residual),
                    **stem_metrics,
                },
            }
        )
        ctx.update(
            stage="Сборка двух вариантов пяти превью",
            substage=f"Превью {index + 1} из {len(demos)}",
            progress=87.0 + (index + 1) / len(demos) * 12.0,
            current_file=pair["name"],
        )
    batch_workspace.cleanup()
    manifest = {
        "schema_version": 5,
        "created_at": utc_now(),
        "film_name": pair["name"],
        "duration_sec": duration,
        "demo_count": len(demo_manifests),
        "demos": demo_manifests,
        "methods": {
            "direct": (
                "Метод A: дорожка перевода − выровненная оригинальная речь → русская речь с музыкой и эффектами"
            ),
            "stem": (
                "Метод B: дорожка перевода − оригинальная речь → русская речь; "
                "русская речь + оригинальные музыка и эффекты"
            ),
        },
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["components_test"] = manifest
    pair["components_test_review"] = {
        "accepted": False,
        "selected_method": None,
        "reviewed_at": None,
        "message": "Прослушайте пять сцен и выберите метод A или B.",
    }
    pair["stages"]["6"] = {
        "status": "available",
        "message": "Пять превью готовы. Выберите метод сборки.",
    }
    store.save_pair(pair)
    ctx.update(
        progress=100.0,
        local_progress=100.0,
        substage="Пять сравнительных превью готовы",
    )
    return manifest


def build_components_pair(
    store: Store, project_id: str, pair_id: str, ctx: Any
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    review = pair.get("components_test_review", {})
    selected_method = str(review.get("selected_method") or "")
    if not review.get("accepted") or selected_method not in {"direct", "stem"}:
        raise RuntimeError(
            "Сначала прослушайте пять превью и выберите метод A или B."
        )
    component_progress_start = 0.0
    original_manifest = pair.get("original_speech") or {}
    original_stems_ready = all(
        Path(str(original_manifest.get(key) or "")).is_file()
        for key in ("speech", "music_effects")
    )
    if (
        pair.get("stages", {}).get("5", {}).get("status") != "completed"
        or not original_stems_ready
    ):
        ctx.update(
            stage="Подготовка полного оригинала",
            substage="Выделение EN-речи и оригинального M&E",
            progress=0.0,
            current_file=pair["name"],
        )
        extract_original_speech_pair(
            store,
            project_id,
            pair_id,
            ctx,
            progress_base=0.0,
            progress_span=45.0,
        )
        pair = store.load_pair(project_id, pair_id)
        component_progress_start = 45.0
    root = store.pair_dir(project_id, pair_id)
    amap = _processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )
    segments = amap.get("segments", [])
    dubbed_path = root / "extracted" / "dubbed.flac"
    en_speech_path = Path(pair["original_speech"]["speech"])
    original_me_path = Path(pair["original_speech"]["music_effects"])
    info = sf.info(str(dubbed_path))
    sample_rate, channels = int(info.samplerate), int(info.channels)
    total = float(info.duration)
    dubbed_speech_path: Path | None = None
    progress_base = component_progress_start
    if selected_method == "stem":
        dubbed_stems = root / "stems" / "dubbed"
        dubbed_stems.mkdir(parents=True, exist_ok=True)
        dubbed_speech_path = dubbed_stems / "speech_en_ru.flac"
        neural_span = 35.0 if component_progress_start else 60.0
        _run_speech_extractor(
            store,
            dubbed_path,
            dubbed_speech_path,
            ctx,
            pair["name"],
            "Выделение общей речи EN+RU из переводной дорожки",
            progress_base=component_progress_start,
            progress_span=neural_span,
            model_role="dubbed",
        )
        progress_base = component_progress_start + neural_span
    output_root = root / "components"
    output_root.mkdir(parents=True, exist_ok=True)
    names = {
        "aligned_en_speech": "aligned_en_speech.flac",
        "original_me": "original_me_aligned.flac",
        "usable_mask": "usable_alignment_mask.flac",
    }
    if selected_method == "direct":
        names["final"] = "direct_ru_plus_me.flac"
    else:
        names["ru_voice"] = "ru_voice.flac"
        names["final"] = "stem_ru_plus_me.flac"
    temps = {key: _temporary(output_root / name) for key, name in names.items()}
    for stale in output_root.glob(".*.partial.*"):
        stale.unlink(missing_ok=True)
    writers = {
        key: sf.SoundFile(
            str(path),
            "w",
            samplerate=sample_rate,
            channels=channels,
            format="FLAC",
            subtype="PCM_16",
        )
        for key, path in temps.items()
    }
    ac = store.cfg["adaptive_cancel"]
    chunk_sec = max(5.0, float(ac.get("streaming_chunk_sec", 20.0)))
    correction = float(amap.get("manual_correction_sec") or 0.0)
    processed = 0.0
    try:
        for segment_index, segment in enumerate(segments):
            segment_start = float(segment["dubbed_start"])
            segment_end = float(segment["dubbed_end"])
            duration = max(0.0, segment_end - segment_start)
            speed = float(segment.get("speed_ratio") or 1.0)
            usable = bool(segment.get("usable"))
            confidence = float(segment.get("confidence", 0.0)) if usable else 0.0
            blocks = max(1, int(math.ceil(duration / chunk_sec)))
            for block in range(blocks):
                _check_stop(ctx)
                start = segment_start + block * chunk_sec
                block_duration = min(chunk_sec, segment_end - start)
                if block_duration <= 0:
                    continue
                target, target_sr = _read_interval(
                    dubbed_path, start, block_duration, channels
                )
                target_len = int(round(block_duration * sample_rate))
                target = _resample_exact(target, target_sr, sample_rate, target_len)
                if usable:
                    original_start = float(segment["original_start"]) + (
                        start - segment_start
                    ) * speed + correction
                    original_me, original_me_sr = _read_interval(
                        original_me_path,
                        original_start,
                        block_duration * speed,
                        channels,
                    )
                    original_me = _resample_exact(
                        original_me, original_me_sr, sample_rate, target_len
                    )
                    en_speech, en_speech_sr = _read_interval(
                        en_speech_path,
                        original_start,
                        block_duration * speed,
                        1,
                    )
                    en_speech = _resample_exact(
                        en_speech, en_speech_sr, sample_rate, target_len
                    )
                    if selected_method == "direct":
                        direct = _direct_dialogue_cancel(
                            en_speech,
                            target,
                            sample_rate,
                            confidence,
                            ac,
                        )
                        data = {
                            "aligned_en_speech": direct.predicted_common,
                            "original_me": original_me,
                            "usable_mask": np.ones_like(target),
                            "final": direct.residual,
                        }
                    else:
                        assert dubbed_speech_path is not None
                        dubbed_speech, dubbed_speech_sr = _read_interval(
                            dubbed_speech_path,
                            start,
                            block_duration,
                            1,
                        )
                        dubbed_speech = _resample_exact(
                            dubbed_speech,
                            dubbed_speech_sr,
                            sample_rate,
                            target_len,
                        )
                        variants, _metrics = _component_variants(
                            original_me,
                            en_speech,
                            dubbed_speech,
                            sample_rate,
                            confidence,
                            ac,
                        )
                        data = {
                            "aligned_en_speech": variants["aligned_en_speech"],
                            "original_me": variants["original_me"],
                            "usable_mask": np.ones_like(target),
                            "ru_voice": variants["ru_voice"],
                            "final": variants["final_ru_plus_me"],
                        }
                else:
                    zeros = np.zeros_like(target)
                    data = {
                        "aligned_en_speech": zeros,
                        "original_me": zeros,
                        "usable_mask": zeros,
                        "final": target,
                    }
                    if selected_method == "stem":
                        data["ru_voice"] = zeros
                for key, values in data.items():
                    writers[key].write(np.clip(values, -1.0, 1.0))
                processed += block_duration
                ctx.update(
                    stage=(
                        "Прямое удаление EN-речи из RU+EN+M&E"
                        if selected_method == "direct"
                        else "Очистка RU-речи и сборка с оригинальным M&E"
                    ),
                    substage=(
                        f"Сегмент {segment_index + 1}/{len(segments)}, "
                        f"блок {block + 1}/{blocks}"
                    ),
                    progress=progress_base
                    + min(
                        99.0 - progress_base,
                        processed / max(total, 1e-6) * (99.0 - progress_base),
                    ),
                    processed_seconds=processed,
                    total_seconds=total,
                    current_file=pair["name"],
                )
    finally:
        for writer in writers.values():
            writer.close()
    for key, temporary in temps.items():
        destination = output_root / names[key]
        if not temporary.is_file():
            raise RuntimeError(f"Не создан компонент {names[key]}")
        os.replace(temporary, destination)
    files = {key: str((output_root / name).resolve()) for key, name in names.items()}
    files["full_original_en_speech"] = str(en_speech_path.resolve())
    files["full_original_me"] = str(original_me_path.resolve())
    if dubbed_speech_path is not None:
        files["dubbed_speech_en_ru"] = str(dubbed_speech_path.resolve())
    manifest = {
        "schema_version": 5,
        "created_at": utc_now(),
        "film_name": pair["name"],
        "selected_method": selected_method,
        "selected_method_label": (
            "Метод A — прямое вычитание EN-речи из RU+EN+M&E"
            if selected_method == "direct"
            else "Метод B — очистка общего речевого стема и сборка с M&E"
        ),
        "duration_sec": total,
        "sample_rate": sample_rate,
        "channels": channels,
        "files": files,
        "chain": (
            [
                "Оригинал → EN-речь + оригинальный M&E",
                "RU+EN+M&E − выровненная EN-речь → финальная RU+M&E",
            ]
            if selected_method == "direct"
            else [
                "Оригинал → EN-речь + оригинальный M&E",
                "RU+EN версия → общая речь EN+RU",
                "Общая речь EN+RU − EN-речь → RU-голос",
                "RU-голос + оригинальный M&E → финальная дорожка",
            ]
        ),
        "low_confidence_policy": (
            "На несопоставленных участках исходная RU+EN дорожка проходит без "
            "вычитания; M&E и речевые стемы там заполнены тишиной, а файл "
            "usable_alignment_mask.flac явно отмечает пригодные участки."
        ),
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["components"] = manifest
    pair["current_stage"] = max(int(pair.get("current_stage", 5)), 6)
    pair["stages"]["6"] = {
        "status": "completed",
        "message": (
            "Полный фильм обработан выбранным методом A."
            if selected_method == "direct"
            else "Полный фильм обработан выбранным методом B."
        ),
    }
    pair["stages"]["7"] = {
        "status": "available",
        "message": "Можно прослушать контрольные сцены речевых стемов и финала.",
    }
    store.save_pair(pair)
    return manifest


def cancel_test_fragment(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    parameters: dict[str, Any],
) -> dict:
    """Run the real cancellation algorithm on one short, reviewable scene."""
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("4", {}).get("status") != "completed":
        raise RuntimeError("Сначала проверьте и подтвердите карту сопоставления.")
    root = store.pair_dir(project_id, pair_id)
    alignment_map = read_json(root / "alignment" / "alignment_map.json")
    segments = [item for item in alignment_map.get("segments", []) if item.get("usable")]
    if not segments:
        raise RuntimeError("В карте нет уверенного участка для пробного вычитания.")
    requested_duration = min(120.0, max(5.0, float(parameters.get("duration_sec", 30.0))))
    raw_start = parameters.get("start_sec")
    if raw_start is None or str(raw_start).strip() == "":
        eligible = [
            item
            for item in segments
            if float(item["dubbed_end"]) - float(item["dubbed_start"]) >= 5.0
        ]
        if not eligible:
            raise RuntimeError("Нет уверенного участка продолжительностью хотя бы 5 секунд.")
        segment = max(
            eligible,
            key=lambda item: (
                min(requested_duration, float(item["dubbed_end"]) - float(item["dubbed_start"])),
                float(item.get("confidence", 0.0)),
            ),
        )
        segment_duration = float(segment["dubbed_end"]) - float(segment["dubbed_start"])
        duration = min(requested_duration, segment_duration)
        start_sec = float(segment["dubbed_start"]) + max(0.0, (segment_duration - duration) / 2.0)
    else:
        start_sec = max(0.0, float(raw_start))
        segment = next(
            (
                item
                for item in segments
                if float(item["dubbed_start"]) <= start_sec < float(item["dubbed_end"])
            ),
            None,
        )
        if segment is None:
            raise ValueError("В выбранном времени нет уверенного участка карты.")
        duration = min(requested_duration, float(segment["dubbed_end"]) - start_sec)
    if duration < 5.0:
        raise ValueError("До конца уверенного участка осталось меньше 5 секунд.")
    _check_stop(ctx)
    ctx.update(
        force=True,
        stage="Пробное вычитание",
        substage=f"Фрагмент {start_sec:.1f}–{start_sec + duration:.1f} с",
        progress=10.0,
        current_file=pair["name"],
    )
    original_path = root / "extracted" / "original.flac"
    dubbed_path = root / "extracted" / "dubbed.flac"
    target, sample_rate = _read_interval(dubbed_path, start_sec, duration)
    channels = target.shape[1]
    speed_ratio = float(segment.get("speed_ratio") or 1.0)
    manual_correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    original_start = float(segment["original_start"]) + (
        start_sec - float(segment["dubbed_start"])
    ) * speed_ratio + manual_correction
    predictor, original_sr = _read_interval(
        original_path, original_start, duration * speed_ratio, channels
    )
    aligned = _resample_exact(predictor, original_sr, sample_rate, len(target))
    confidence = float(segment.get("confidence", 0.0))
    confidence_curve = np.full(len(target), confidence, dtype=np.float32)
    ac = store.cfg["adaptive_cancel"]
    result = adaptive_cancel.segmented_adaptive_cancel(
        aligned,
        target,
        sample_rate,
        confidence_curve,
        stft_n_fft=int(ac["stft_n_fft"]),
        stft_hop=int(ac["stft_hop"]),
        block_frames=int(ac["block_frames"]),
        gain_smoothing=float(ac["gain_smoothing"]),
        min_coherence=float(ac["min_coherence"]),
        regularization=float(ac["regularization"]),
        max_gain_linear=float(ac["max_gain_linear"]),
    )
    _check_stop(ctx)
    output_root = root / "previews" / "cancel_test"
    output_root.mkdir(parents=True, exist_ok=True)
    tracks = {
        "mixture": ("mixture_ru_en.flac", target),
        "aligned_original": ("aligned_original_en.flac", aligned),
        "predicted_common": ("predicted_common.flac", result.predicted_common),
        "residual": ("ru_pseudo_residual.flac", result.residual),
    }
    files: dict[str, str] = {}
    for key, (name, data) in tracks.items():
        destination = output_root / name
        temporary = _temporary(destination)
        sf.write(
            str(temporary),
            np.clip(data, -1.0, 1.0),
            sample_rate,
            format="FLAC",
            subtype="PCM_16",
        )
        os.replace(temporary, destination)
        files[key] = str(destination.resolve())
    manifest = {
        "schema_version": 2,
        "created_at": utc_now(),
        "pair_id": pair_id,
        "film_name": pair["name"],
        "start_sec": round(start_sec, 6),
        "duration_sec": round(duration, 6),
        "alignment_confidence": confidence,
        "mean_coherence": float(result.mean_coherence),
        "gated_fraction": float(result.gated_fraction),
        "files": files,
    }
    atomic_json(output_root / "manifest.json", manifest)
    pair["cancel_test"] = {
        key: manifest[key]
        for key in ("created_at", "start_sec", "duration_sec", "alignment_confidence", "files")
    }
    pair["stages"]["5"] = {
        "status": "available",
        "message": "Пробное вычитание готово. Прослушайте его перед полным фильмом.",
    }
    store.save_pair(pair)
    ctx.update(progress=100.0, substage="Пробный фрагмент готов")
    return manifest


def cancel_pair(store: Store, project_id: str, pair_id: str, ctx: Any) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("4", {}).get("status") != "completed":
        raise RuntimeError("Сначала проверьте и подтвердите карту сопоставления.")
    root = store.pair_dir(project_id, pair_id)
    alignment_map = read_json(root / "alignment" / "alignment_map.json")
    segments = alignment_map.get("segments", [])
    manual_correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    original_path = root / "extracted" / "original.flac"
    dubbed_path = root / "extracted" / "dubbed.flac"
    dubbed_info = sf.info(str(dubbed_path))
    sample_rate = int(dubbed_info.samplerate)
    channels = int(dubbed_info.channels)
    output_root = root / "results" / "baseline"
    output_root.mkdir(parents=True, exist_ok=True)
    for stale in output_root.glob(".*.partial.*"):
        stale.unlink(missing_ok=True)
    names = {
        "aligned": "aligned_original.flac",
        "common": "predicted_common.flac",
        "residual": "ru_pseudo_residual.flac",
        "protective": "protective_result.flac",
        "full": "full_frequency_result.flac",
        "confidence": "confidence.wav",
    }
    temporaries = {key: _temporary(output_root / name) for key, name in names.items()}
    writers: dict[str, sf.SoundFile] = {}
    ac = store.cfg["adaptive_cancel"]
    chunk_sec = max(5.0, float(ac.get("streaming_chunk_sec", 20.0)))
    context_sec = max(0.0, min(2.0, float(ac.get("streaming_context_sec", 0.5))))
    total = float(dubbed_info.duration)
    processed = 0.0
    started = time.monotonic()
    metrics: list[dict] = []
    try:
        for key in ("aligned", "common", "residual", "protective", "full"):
            writers[key] = sf.SoundFile(
                str(temporaries[key]),
                "w",
                samplerate=sample_rate,
                channels=channels,
                format="FLAC",
                subtype="PCM_24",
            )
        writers["confidence"] = sf.SoundFile(
            str(temporaries["confidence"]),
            "w",
            samplerate=sample_rate,
            channels=1,
            format="WAV",
            subtype="FLOAT",
        )
        for segment_index, segment in enumerate(segments):
            segment_start = float(segment["dubbed_start"])
            segment_end = float(segment["dubbed_end"])
            duration = max(0.0, segment_end - segment_start)
            usable = bool(segment.get("usable"))
            confidence = float(segment.get("confidence", 0.0)) if usable else 0.0
            speed_ratio = float(segment.get("speed_ratio") or 1.0)
            block_count = max(1, int(math.ceil(duration / chunk_sec)))
            coherence_weighted = 0.0
            gated_weighted = 0.0
            metric_duration = 0.0
            for block_index in range(block_count):
                _check_stop(ctx)
                chunk_start = segment_start + block_index * chunk_sec
                chunk_end = min(segment_end, chunk_start + chunk_sec)
                actual_duration = max(0.0, chunk_end - chunk_start)
                if actual_duration <= 0:
                    continue
                process_start = max(segment_start, chunk_start - context_sec)
                process_end = min(segment_end, chunk_end + context_sec)
                process_duration = process_end - process_start
                target, target_sr = _read_interval(
                    dubbed_path, process_start, process_duration, channels
                )
                expected_len = int(round(process_duration * sample_rate))
                if target_sr != sample_rate or len(target) != expected_len:
                    target = _resample_exact(target, target_sr, sample_rate, expected_len)
                target_len = len(target)
                if usable and target_len:
                    original_start = float(segment["original_start"]) + (
                        process_start - segment_start
                    ) * speed_ratio + manual_correction
                    predictor, original_sr = _read_interval(
                        original_path,
                        original_start,
                        process_duration * speed_ratio,
                        channels,
                    )
                    aligned = _resample_exact(
                        predictor, original_sr, sample_rate, target_len
                    )
                    confidence_curve = np.full(target_len, confidence, dtype=np.float32)
                    result = adaptive_cancel.segmented_adaptive_cancel(
                        aligned,
                        target,
                        sample_rate,
                        confidence_curve,
                        stft_n_fft=int(ac["stft_n_fft"]),
                        stft_hop=int(ac["stft_hop"]),
                        block_frames=int(ac["block_frames"]),
                        gain_smoothing=float(ac["gain_smoothing"]),
                        min_coherence=float(ac["min_coherence"]),
                        regularization=float(ac["regularization"]),
                        max_gain_linear=float(ac["max_gain_linear"]),
                    )
                    common = result.predicted_common.astype(np.float32)
                    residual = result.residual.astype(np.float32)
                    coherence_weighted += float(result.mean_coherence) * actual_duration
                    gated_weighted += float(result.gated_fraction) * actual_duration
                else:
                    aligned = np.zeros_like(target)
                    common = np.zeros_like(target)
                    residual = target.copy()
                    confidence_curve = np.zeros(target_len, dtype=np.float32)
                    gated_weighted += actual_duration
                left = int(round((chunk_start - process_start) * sample_rate))
                write_len = int(round(actual_duration * sample_rate))
                right = left + write_len
                for key, data in (
                    ("aligned", aligned),
                    ("common", common),
                    ("residual", residual),
                    ("protective", residual),
                    ("full", residual),
                ):
                    writers[key].write(np.clip(data[left:right], -1.0, 1.0))
                writers["confidence"].write(confidence_curve[left:right, None])
                metric_duration += actual_duration
                processed += actual_duration
                elapsed = max(time.monotonic() - started, 0.001)
                eta = max(0.0, (total - processed) / max(processed / elapsed, 1e-6))
                ctx.update(
                    stage="Предварительное удаление английской речи",
                    substage=(
                        f"Сегмент {segment_index + 1} из {len(segments)} · "
                        f"блок {block_index + 1} из {block_count}"
                    ),
                    progress=min(100.0, processed / max(total, 0.001) * 100.0),
                    processed_seconds=processed,
                    total_seconds=total,
                    eta_seconds=eta,
                    current_file=pair["name"],
                )
            metrics.append(
                {
                    "segment_id": segment["id"],
                    "coherence": coherence_weighted / max(metric_duration, 1e-9),
                    "gated_fraction": gated_weighted / max(metric_duration, 1e-9),
                }
            )
    finally:
        for writer in writers.values():
            writer.close()
    for key, temporary in temporaries.items():
        destination = output_root / names[key]
        if not temporary.is_file():
            raise RuntimeError(f"Не создан промежуточный результат {names[key]}.")
        os.replace(temporary, destination)
    report = {
        "schema_version": 2,
        "created_at": utc_now(),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": total,
        "method": "adaptive_paired_reference",
        "pseudo_target_warning": (
            "RU-остаток является pseudo-target и не считается гарантированно чистым эталоном."
        ),
        "safe_strategy": "На участках низкой уверенности исходная RU+EN дорожка сохранена без вычитания.",
        "segment_metrics": metrics,
        "files": {key: str((output_root / name).resolve()) for key, name in names.items()},
    }
    atomic_json(output_root / "metadata.json", report)
    pair["baseline_result"] = report
    pair["current_stage"] = max(int(pair.get("current_stage", 3)), 5)
    pair["stages"]["4"] = {
        "status": "completed",
        "message": "Карта доступна для визуальной и слуховой проверки.",
    }
    pair["stages"]["5"] = {
        "status": "completed",
        "message": "Полночастотный предварительный RU-остаток создан потоково.",
    }
    pair["stages"]["6"] = {"status": "available", "message": "Можно создать контрольные фрагменты."}
    store.save_pair(pair)
    return report


def _clip(path: Path, start: float, duration: float, destination: Path) -> dict:
    data, sr = _read_interval(path, start, duration)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination)
    sf.write(str(temporary), data, sr, format="FLAC", subtype="PCM_24")
    os.replace(temporary, destination)
    return _probe_payload(destination)


def _segment_for_time(segments: list[dict], value: float) -> dict | None:
    for segment in segments:
        if float(segment["dubbed_start"]) <= value < float(segment["dubbed_end"]):
            return segment
    return segments[-1] if segments else None


def _preview_window_start(segment: dict, duration: float, total_duration: float) -> float:
    """Place a full preview window around a map segment without leaving the track."""
    maximum_start = max(0.0, total_duration - duration)
    segment_start = max(0.0, float(segment.get("dubbed_start") or 0.0))
    segment_duration = max(0.0, float(segment.get("duration") or 0.0))
    if segment_duration >= duration:
        proposed = segment_start
    else:
        proposed = segment_start - max(0.0, duration - segment_duration) / 2.0
    return min(maximum_start, max(0.0, proposed))


def _distinct_preview_choices(
    candidate_groups: dict[str, list[dict]],
    categories: list[str],
    duration: float,
    total_duration: float,
    maximum: int,
) -> list[tuple[str, float]]:
    """Choose semantically useful windows while preventing repeated control scenes."""
    choices: list[tuple[str, float]] = []
    minimum_gap = min(max(5.0, duration * 0.5), max(5.0, total_duration / max(maximum, 1)))
    for category in categories:
        for segment in candidate_groups.get(category, []):
            start = _preview_window_start(segment, duration, total_duration)
            if any(abs(start - existing) < minimum_gap for _, existing in choices):
                continue
            choices.append((category, start))
            break
        if len(choices) >= maximum:
            break
    return choices


def make_previews(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    parameters: dict | None = None,
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("6", {}).get("status") != "completed":
        raise RuntimeError("Сначала постройте EN, RU и M&E компоненты.")
    root = store.pair_dir(project_id, pair_id)
    amap = _processing_alignment_map(
        store, pair, read_json(root / "alignment" / "alignment_map.json")
    )
    segments = amap.get("segments", [])
    manual_correction = float(amap.get("manual_correction_sec") or 0.0)
    preview_cfg = store.cfg["previews"]
    total_duration = float(sf.info(str(root / "extracted" / "dubbed.flac")).duration)
    duration = min(float(store.cfg["audio"]["preview_duration_sec"]), total_duration)
    manual = parameters or {}
    component_manifest = pair.get("components") or read_json(
        root / "components" / "manifest.json", {}
    )
    component_files = {
        key: Path(value)
        for key, value in (component_manifest.get("files") or {}).items()
        if value
    }
    final_component = component_files.get("final")
    if final_component is None or not final_component.is_file():
        raise RuntimeError("Финальный аудиокомпонент выбранного метода не найден.")
    residual_component = component_files.get("ru_voice", final_component)
    choices: list[tuple[str, float]] = []
    if "start_sec" in manual:
        duration = min(
            float(manual.get("duration_sec", duration)),
            float(store.cfg["audio"]["manual_preview_max_sec"]),
            total_duration,
        )
        start = min(
            max(0.0, float(manual["start_sec"])),
            max(0.0, total_duration - duration),
        )
        choices = [("manual", start)]
    else:
        usable = [item for item in segments if item.get("usable")]
        unusable = [item for item in segments if not item.get("usable")]
        confidence_sorted = sorted(usable, key=lambda item: item.get("confidence", 0.0))
        scored: list[tuple[dict, float, float, float, float]] = []
        for segment in usable:
            start = float(segment["dubbed_start"])
            sample_duration = min(duration, float(segment["duration"]))
            mixture, _ = _read_interval(
                root / "extracted" / "dubbed.flac", start, sample_duration
            )
            common, _ = _read_interval(
                component_files["aligned_en_speech"],
                start,
                sample_duration,
            )
            residual, _ = _read_interval(
                residual_component,
                start,
                sample_duration,
            )
            scored.append(
                (
                    segment,
                    _rms_db(mixture),
                    _rms_db(common),
                    _rms_db(residual),
                    abs(_rms_db(common) - _rms_db(residual)),
                )
            )
        simultaneous = [
            item for item in scored if item[2] > -45.0 and item[3] > -45.0
        ]
        candidate_groups = {
            "simultaneous": [item[0] for item in simultaneous]
            or list(usable),
            "en_first": sorted(usable, key=lambda item: float(item["dubbed_start"])),
            "balanced_levels": [item[0] for item in sorted(scored, key=lambda item: item[4])],
            "loud_en": [item[0] for item in sorted(scored, key=lambda item: item[2], reverse=True)],
            "quiet_en": [item[0] for item in sorted(scored, key=lambda item: item[2])],
            "silence_or_music": [item[0] for item in sorted(scored, key=lambda item: item[1])],
            "edit_boundary": sorted(unusable, key=lambda item: float(item["dubbed_start"])),
            "low_confidence": confidence_sorted,
        }
        choices = _distinct_preview_choices(
            candidate_groups,
            list(preview_cfg["categories"]),
            duration,
            total_duration,
            int(preview_cfg["maximum_per_pair"]),
        )
    preview_root = root / "previews"
    preview_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "dubbed": root / "extracted" / "dubbed.flac",
        "en_speech": component_files["aligned_en_speech"],
        "original_me": component_files["original_me"],
        "final": final_component,
    }
    if component_files.get("dubbed_speech_en_ru", Path()).is_file():
        artifacts["combined_speech"] = component_files["dubbed_speech_en_ru"]
    if component_files.get("ru_voice", Path()).is_file():
        artifacts["ru_voice"] = component_files["ru_voice"]
    if component_files.get("usable_mask", Path()).is_file():
        artifacts["usable_mask"] = component_files["usable_mask"]
    previews: list[dict] = []
    for index, (category, start) in enumerate(choices):
        _check_stop(ctx)
        target_dir = preview_root / f"{index + 1:02d}_{category}"
        item = {
            "id": f"{index + 1:02d}_{category}",
            "category": category,
            "dubbed_start_sec": round(start, 3),
            "duration_sec": duration,
            "files": {},
        }
        segment = _segment_for_time(segments, start)
        original_start = (
            float(segment["original_start"])
            + (start - float(segment["dubbed_start"]))
            * float(segment.get("speed_ratio") or 1.0)
            + manual_correction
            if segment
            else start
        )
        original_file = target_dir / "original_en.flac"
        item["files"]["original"] = _clip(
            root / "extracted" / "original.flac",
            original_start,
            duration,
            original_file,
        )["path"]
        for key, source in artifacts.items():
            destination = target_dir / f"{key}.flac"
            item["files"][key] = _clip(source, start, duration, destination)["path"]
        dubbed_data, sr = _read_interval(artifacts["dubbed"], start, duration)
        aligned_data, aligned_sr = _read_interval(artifacts["en_speech"], start, duration)
        if aligned_sr != sr:
            aligned_data = _resample_exact(aligned_data, aligned_sr, sr, len(dubbed_data))
        left = np.mean(dubbed_data, axis=1)
        right = np.mean(aligned_data, axis=1)
        both_path = target_dir / "both_signals_stereo.flac"
        sf.write(str(both_path), np.column_stack([left, right]), sr, format="FLAC", subtype="PCM_24")
        item["files"]["both"] = str(both_path.resolve())
        previews.append(item)
        ctx.update(
            stage="Подготовка контрольных фрагментов",
            substage=f"Фрагмент {index + 1} из {len(choices)}",
            progress=(index + 1) / max(len(choices), 1) * 100.0,
            current_file=pair["name"],
        )
    manifest = {
        "schema_version": 5,
        "created_at": utc_now(),
        "automatic": "start_sec" not in manual,
        "selected_method": component_manifest.get("selected_method"),
        "items": previews,
        "labels": {
            "dubbed": "Исходная дорожка RU+EN",
            "original": "Оригинальная EN-дорожка",
            "en_speech": "Выделенная и выровненная EN-речь",
            "combined_speech": "Общая речь EN+RU из переводной дорожки",
            "original_me": "M&E из оригинала",
            "ru_voice": "Очищенный RU-голос (метод B)",
            "final": "Итог выбранного метода",
            "usable_mask": "Маска пригодных участков (звук = участок пригоден)",
            "both": "RU+EN слева, выделенная EN-речь справа",
        },
        "category_labels": {
            "manual": "Ручной фрагмент",
            "simultaneous": "Одновременная EN- и RU-речь",
            "en_first": "EN-речь раньше RU",
            "balanced_levels": "Приблизительно равные уровни",
            "loud_en": "Громкая EN-речь",
            "quiet_en": "Тихая EN-речь",
            "silence_or_music": "Тихий участок или музыка",
            "edit_boundary": "Монтажная граница",
            "low_confidence": "Низкая уверенность сопоставления",
        },
    }
    atomic_json(preview_root / "preview_manifest.json", manifest)
    pair["previews"] = {"count": len(previews), "manifest": str(preview_root / "preview_manifest.json")}
    pair["current_stage"] = max(int(pair.get("current_stage", 6)), 7)
    pair["stages"]["7"] = {"status": "completed", "message": f"Создано контрольных фрагментов: {len(previews)}."}
    pair["stages"]["8"] = {"status": "available", "message": "Компоненты готовы к включению в датасет."}
    pair["stages"]["10"] = {"status": "available", "message": "Финальная RU+M&E дорожка готова."}
    store.save_pair(pair)
    return manifest


def _max_lag_correlation(
    target: np.ndarray, reference: np.ndarray, sample_rate: int, max_lag_sec: float = 0.4
) -> float:
    """Max |normalised cross-correlation| between two clips over ±max_lag_sec."""
    a = target.astype(np.float64) - float(np.mean(target))
    b = reference.astype(np.float64) - float(np.mean(reference))
    denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denominator < 1e-12:
        return 0.0
    size = 1
    while size < len(a) * 2:
        size *= 2
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    correlation = np.fft.irfft(spectrum, size)
    max_lag = max(1, min(int(max_lag_sec * sample_rate), len(a) - 1))
    window = np.concatenate([correlation[-max_lag:], correlation[: max_lag + 1]])
    return float(np.max(np.abs(window)) / denominator)


def _rms_db(data: np.ndarray) -> float:
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(data.astype(np.float64)))) + 1e-9))


def _scenario(mixture: np.ndarray, common: np.ndarray, residual: np.ndarray) -> tuple[str, dict]:
    mix_db = _rms_db(mixture)
    en_db = _rms_db(common)
    ru_db = _rms_db(residual)
    delta = en_db - ru_db
    def onset_seconds(data: np.ndarray) -> float | None:
        mono = np.mean(data, axis=1) if data.ndim == 2 else data
        frame = max(1, len(mono) // 40)
        energies = np.asarray(
            [
                np.sqrt(np.mean(np.square(mono[index : index + frame], dtype=np.float64)) + 1e-12)
                for index in range(0, len(mono), frame)
            ]
        )
        if not len(energies) or float(np.max(energies)) < 1e-5:
            return None
        threshold = max(float(np.max(energies)) * 0.18, 1e-4)
        active = np.flatnonzero(energies >= threshold)
        return float(active[0] * frame) if len(active) else None

    common_onset = onset_seconds(common)
    residual_onset = onset_seconds(residual)
    onset_delta_samples = (
        residual_onset - common_onset
        if common_onset is not None and residual_onset is not None
        else 0.0
    )
    onset_delta_ratio = onset_delta_samples / max(len(mixture), 1)
    if mix_db < -48:
        kind = "silence_or_music"
    elif en_db < -48 and ru_db >= -48:
        kind = "mostly_ru"
    elif ru_db < -48 and en_db >= -48:
        kind = "mostly_en"
    elif onset_delta_ratio > 0.03:
        kind = "en_first"
    elif onset_delta_ratio < -0.03:
        kind = "ru_first"
    elif delta > 4:
        kind = "en_louder"
    elif delta < -4:
        kind = "ru_louder"
    else:
        kind = "balanced_levels"
    return kind, {
        "mixture_rms_db": round(mix_db, 3),
        "common_component_rms_db": round(en_db, 3),
        "pseudo_ru_rms_db": round(ru_db, 3),
        "common_to_pseudo_ru_db": round(delta, 3),
        "onset_delta_ratio": round(onset_delta_ratio, 4),
    }


def prepare_dataset(
    store: Store, project_id: str, pair_ids: list[str], ctx: Any
) -> dict:
    pairs = [store.load_pair(project_id, item) for item in pair_ids]
    if not pairs:
        raise ValueError("Не выбрана ни одна пара.")
    for pair in pairs:
        if pair.get("stages", {}).get("7", {}).get("status") != "completed":
            raise RuntimeError(f"Для «{pair['name']}» сначала создайте и проверьте предпросмотры.")
    cfg = store.cfg["dataset"]
    dataset_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    dataset_root = store.project_dir(project_id) / "training" / "datasets" / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=False)
    clip_duration = float(cfg["clip_duration_sec"])
    stride = float(cfg["clip_stride_sec"])
    min_conf = float(cfg["minimum_confidence"])
    minimum_rms = float(cfg["minimum_rms_db"])
    minimum_reference_speech_rms = float(
        cfg.get("minimum_reference_speech_rms_db", -48.0)
    )
    maximum = int(cfg["maximum_examples_per_pair"])
    rows: list[dict] = []
    rejected: dict[str, int] = {}
    sorted_ids = sorted(pair_ids)

    def split_for(pair_id: str, scene_group: int) -> str:
        if len(sorted_ids) >= 3:
            if pair_id == sorted_ids[-1]:
                return "test"
            if pair_id == sorted_ids[-2]:
                return "valid"
            return "train"
        digest = int(hashlib.sha256(f"{pair_id}:{scene_group}".encode()).hexdigest()[:8], 16) % 100
        train = int(cfg["split_percent"]["train"])
        valid = int(cfg["split_percent"]["valid"])
        return "train" if digest < train else ("valid" if digest < train + valid else "test")

    total_candidates = 0
    for pair in pairs:
        root = store.pair_dir(project_id, pair["id"])
        amap = _processing_alignment_map(
            store, pair, read_json(root / "alignment" / "alignment_map.json")
        )
        for segment in amap.get("segments", []):
            total_candidates += max(0, int(float(segment["duration"]) // stride) + 1)
    completed_candidates = 0
    for pair in pairs:
        root = store.pair_dir(project_id, pair["id"])
        amap = _processing_alignment_map(
            store, pair, read_json(root / "alignment" / "alignment_map.json")
        )
        component_manifest = pair.get("components") or read_json(
            root / "components" / "manifest.json", {}
        )
        selected_method = str(component_manifest.get("selected_method") or "")
        component_files = {
            key: Path(value)
            for key, value in (component_manifest.get("files") or {}).items()
            if value
        }
        if selected_method not in {"direct", "stem"}:
            raise RuntimeError(
                f"Для «{pair['name']}» не найден выбранный метод обработки."
            )
        accepted_pair = 0
        for segment in amap.get("segments", []):
            segment_start = float(segment["dubbed_start"])
            segment_end = float(segment["dubbed_end"])
            position = segment_start
            while position + clip_duration <= segment_end + 1e-6 and accepted_pair < maximum:
                _check_stop(ctx)
                completed_candidates += 1
                if not segment.get("usable") or float(segment.get("confidence", 0)) < min_conf:
                    rejected["низкая уверенность сопоставления"] = rejected.get("низкая уверенность сопоставления", 0) + 1
                    position += stride
                    continue
                mixture, sr = _read_interval(root / "extracted" / "dubbed.flac", position, clip_duration)
                en_speech, en_sr = _read_interval(
                    component_files["aligned_en_speech"], position, clip_duration
                )
                if _rms_db(mixture) < minimum_rms:
                    rejected["слишком тихий участок"] = rejected.get("слишком тихий участок", 0) + 1
                    position += stride
                    continue
                target_len = len(mixture)
                if en_sr != sr or len(en_speech) != target_len:
                    en_speech = _resample_exact(en_speech, en_sr, sr, target_len)
                final_mix, final_sr = _read_interval(
                    component_files["final"], position, clip_duration
                )
                if final_sr != sr or len(final_mix) != target_len:
                    final_mix = _resample_exact(
                        final_mix, final_sr, sr, target_len
                    )
                if selected_method == "stem":
                    ru_voice, ru_sr = _read_interval(
                        component_files["ru_voice"], position, clip_duration
                    )
                    combined_speech, combined_speech_sr = _read_interval(
                        component_files["dubbed_speech_en_ru"],
                        position,
                        clip_duration,
                    )
                    if ru_sr != sr or len(ru_voice) != target_len:
                        ru_voice = _resample_exact(
                            ru_voice, ru_sr, sr, target_len
                        )
                    if (
                        combined_speech_sr != sr
                        or len(combined_speech) != target_len
                    ):
                        combined_speech = _resample_exact(
                            combined_speech,
                            combined_speech_sr,
                            sr,
                            target_len,
                        )
                    scenario_signal = ru_voice
                else:
                    ru_voice = None
                    combined_speech = None
                    scenario_signal = final_mix
                scenario, levels = _scenario(
                    mixture, en_speech, scenario_signal
                )
                scene_group = int(position // float(cfg["scene_group_sec"]))
                split = split_for(pair["id"], scene_group)
                example_id = f"{pair['id'][:6]}_{accepted_pair:04d}"
                example_dir = dataset_root / split / example_id
                example_dir.mkdir(parents=True, exist_ok=True)
                files = {
                    "dubbed_mix": example_dir / "dubbed_mix_ru_en.flac",
                    "en_speech": example_dir / "aligned_en_speech.flac",
                    "original_me": example_dir / "original_music_and_effects.flac",
                    "final_mix": example_dir / f"{selected_method}_result_ru_plus_me.flac",
                }
                if selected_method == "stem":
                    files["combined_speech_en_ru"] = (
                        example_dir / "combined_speech_en_ru.flac"
                    )
                    files["ru_voice"] = example_dir / "ru_voice.flac"
                original_me, original_me_sr = _read_interval(
                    component_files["original_me"], position, clip_duration
                )
                if original_me_sr != sr or len(original_me) != target_len:
                    original_me = _resample_exact(original_me, original_me_sr, sr, target_len)
                payloads: list[tuple[str, np.ndarray]] = [
                    ("dubbed_mix", mixture),
                    ("en_speech", en_speech),
                    ("original_me", original_me),
                    ("final_mix", final_mix),
                ]
                if selected_method == "stem":
                    assert combined_speech is not None and ru_voice is not None
                    payloads.extend(
                        [
                            ("combined_speech_en_ru", combined_speech),
                            ("ru_voice", ru_voice),
                        ]
                    )
                for key, data in payloads:
                    sf.write(str(files[key]), data, sr, format="FLAC", subtype="PCM_24")
                original_start = float(segment["original_start"]) + (
                    position - segment_start
                ) * float(segment.get("speed_ratio", 1.0))
                row = {
                    "id": example_id,
                    "project_id": project_id,
                    "pair_id": pair["id"],
                    "film_name": pair["name"],
                    "split": split,
                    "scene_group": f"{pair['id']}:{scene_group}",
                    "dubbed_start_sec": round(position, 6),
                    "dubbed_end_sec": round(position + clip_duration, 6),
                    "original_start_sec": round(original_start, 6),
                    "original_end_sec": round(original_start + clip_duration * float(segment.get("speed_ratio", 1.0)), 6),
                    "alignment_confidence": float(segment["confidence"]),
                    "scenario": scenario,
                    "selected_method": selected_method,
                    "target_kind": (
                        "direct_ru_plus_me_from_raw_mix_minus_en_speech"
                        if selected_method == "direct"
                        else "ru_voice_from_en_ru_speech_minus_en_speech"
                    ),
                    "target_warning": (
                        "Итог RU+M&E получен прямым вычитанием EN-речи и требует ручной проверки."
                        if selected_method == "direct"
                        else "RU-голос получен вычитанием EN-речи из общего речевого стема EN+RU и требует ручной проверки."
                    ),
                    "sample_rate": sr,
                    "channels": int(mixture.shape[1]),
                    "levels": levels,
                    "files": {key: str(path.resolve()) for key, path in files.items()},
                }
                atomic_json(example_dir / "metadata.json", row)
                rows.append(row)
                accepted_pair += 1
                position += stride
                ctx.update(
                    stage="Подготовка обучающих материалов",
                    substage=f"{pair['name']}: пример {accepted_pair}",
                    progress=min(100.0, completed_candidates / max(total_candidates, 1) * 100.0),
                    current_file=pair["name"],
                )
    if not rows:
        raise RuntimeError("Не найдено ни одного пригодного фрагмента для датасета.")
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "valid", "test")}
    scenario_counts: dict[str, int] = {}
    for row in rows:
        scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
    manifest = {
        "schema_version": 5,
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "project_id": project_id,
        "pair_ids": pair_ids,
        "film_count": len(pairs),
        "example_count": len(rows),
        "duration_sec": round(len(rows) * clip_duration, 3),
        "split_counts": counts,
        "scenario_counts": scenario_counts,
        "rejected_count": sum(rejected.values()),
        "rejected_reasons": rejected,
        "target_kind": "selected_method_audio_outputs",
        "method_counts": {
            method: sum(row["selected_method"] == method for row in rows)
            for method in ("direct", "stem")
        },
        "leakage_policy": "Разделение по фильмам; при нехватке фильмов — по непересекающимся группам сцен.",
        "examples": rows,
    }
    atomic_json(dataset_root / "manifest.json", manifest)
    project = store.load_project(project_id)
    project["latest_dataset_id"] = dataset_id
    project["latest_dataset"] = {
        key: manifest[key]
        for key in (
            "dataset_id",
            "film_count",
            "example_count",
            "duration_sec",
            "split_counts",
            "scenario_counts",
            "rejected_count",
            "rejected_reasons",
            "target_kind",
        )
    }
    store.save_project(project)
    for pair in pairs:
        pair["stages"]["8"] = {"status": "completed", "message": f"Включена в датасет {dataset_id}."}
        pair["stages"]["9"] = {"status": "available", "message": "Доступна проверка готовой модели или дообучение."}
        store.save_pair(pair)
    return manifest


def _training_mono(
    data: np.ndarray, source_rate: int, target_rate: int, target_frames: int
) -> np.ndarray:
    mono = np.mean(data, axis=1, dtype=np.float64).astype(np.float32)
    if source_rate != target_rate or len(mono) != target_frames:
        mono = _resample_exact(mono[:, None], source_rate, target_rate, target_frames)[:, 0]
    if len(mono) < target_frames:
        mono = np.pad(mono, (0, target_frames - len(mono)))
    return mono[:target_frames].astype(np.float32)


def _synthetic_reference_component(
    reference: np.ndarray, sample_rate: int, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Simulate the mastering changes applied to the original inside a dub mix."""
    gain_db = float(rng.uniform(-5.0, 3.0))
    tilt_db = float(rng.uniform(-4.0, 4.0))
    delay_ms = float(rng.uniform(-18.0, 18.0))
    drive = float(rng.uniform(1.0, 1.7))
    spectrum = np.fft.rfft(reference.astype(np.float64))
    frequencies = np.fft.rfftfreq(len(reference), 1.0 / sample_rate)
    normalized = np.log2(np.maximum(frequencies, 80.0) / 1000.0)
    curve_db = np.clip(normalized * tilt_db / 4.0, -abs(tilt_db), abs(tilt_db))
    filtered = np.fft.irfft(
        spectrum * np.power(10.0, curve_db / 20.0), n=len(reference)
    ).astype(np.float32)
    filtered *= float(10.0 ** (gain_db / 20.0))
    if drive > 1.0001:
        filtered = (np.tanh(filtered * drive) / np.tanh(drive)).astype(np.float32)
    delay = int(round(delay_ms * sample_rate / 1000.0))
    shifted = np.zeros_like(filtered)
    if delay > 0:
        shifted[delay:] = filtered[:-delay]
    elif delay < 0:
        shifted[:delay] = filtered[-delay:]
    else:
        shifted = filtered
    return shifted, {
        "gain_db": round(gain_db, 4),
        "spectral_tilt_db": round(tilt_db, 4),
        "delay_ms": round(delay_ms, 4),
        "soft_compression_drive": round(drive, 4),
    }


def _write_training_audio(path: Path, data: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(data, -1.0, 1.0),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return str(path.resolve())


def _prepare_clean_translation_dataset(
    store: Store,
    project_id: str,
    pair_ids: list[str],
    ctx: Any,
    parameters: dict[str, Any],
) -> dict:
    """Create recipe-based supervised examples from clean translated tracks.

    The translated source is the exact desired residual.  English speech from
    the aligned original is transformed and mixed into it.  Recipes are kept
    instead of materialising tens of gigabytes of duplicate audio; the runner
    synthesises every example deterministically while loading it.
    """
    pairs = [store.load_pair(project_id, item) for item in pair_ids]
    if not pairs:
        raise ValueError("Не выбран ни один фильм.")
    cfg = store.cfg["dataset"]
    sample_rate = int(parameters.get("sample_rate", cfg.get("training_sample_rate", 24000)))
    clip_duration = float(parameters.get("clip_duration_sec", 6.0))
    target_frames = int(round(sample_rate * clip_duration))
    requested_examples = max(1000, min(50000, int(parameters.get("example_count", 10000))))
    scan_stride = max(1.0, float(parameters.get("speech_scan_stride_sec", 3.0)))
    minimum_confidence = float(parameters.get("minimum_confidence", cfg["minimum_confidence"]))
    minimum_speech_rms = float(parameters.get("minimum_speech_rms_db", -46.0))
    dataset_id = time.strftime("%Y%m%d_%H%M%S") + "_cleanru_" + uuid.uuid4().hex[:6]
    dataset_root = store.project_dir(project_id) / "training" / "datasets" / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=False)

    contexts: list[dict[str, Any]] = []
    extraction_span = 30.0 / max(len(pairs), 1)
    for pair_index, pair in enumerate(pairs):
        if pair.get("stages", {}).get("4", {}).get("status") != "completed":
            raise RuntimeError(f"Для «{pair['name']}» сначала подтвердите сопоставление.")
        root = store.pair_dir(project_id, pair["id"])
        original_speech = root / "stems" / "original" / "en_speech.flac"
        if not original_speech.is_file():
            raise RuntimeError(
                f"Для «{pair['name']}» сначала полностью выделите английскую речь."
            )
        target_full = root / "extracted" / "dubbed.flac"
        target_root = root / "stems" / "clean_translation"
        target_root.mkdir(parents=True, exist_ok=True)
        target_speech = target_root / "ru_speech.flac"
        progress_base = pair_index * extraction_span
        if not target_speech.is_file():
            _run_speech_extractor(
                store,
                target_full,
                target_speech,
                ctx,
                pair["name"],
                "Выделение русской речи из чистой переводной дорожки",
                progress_base=progress_base,
                progress_span=extraction_span,
                model_role="dubbed",
            )
        alignment = _processing_alignment_map(
            store, pair, read_json(root / "alignment" / "alignment_map.json")
        )
        contexts.append(
            {
                "pair": pair,
                "root": root,
                "alignment": alignment,
                "original_speech": original_speech,
                "target_speech": target_speech,
            }
        )

    max_target_reference_correlation = float(
        parameters.get(
            "max_reference_in_target_correlation",
            cfg.get("max_reference_in_target_correlation", 0.30),
        )
    )
    russian_shift_margin = 1.5
    scene_group_sec = float(cfg.get("scene_group_sec", 120.0))

    def split_of_group(pair_id: str, group: int) -> str:
        key = int(
            hashlib.sha256(f"{pair_id}:{group}".encode()).hexdigest()[:8], 16
        ) % 100
        return "train" if key < 80 else ("valid" if key < 90 else "test")

    candidates: list[dict[str, Any]] = []
    rejected = {
        "низкая уверенность сопоставления": 0,
        "нет речи": 0,
        "оригинальная речь в чистом переводе": 0,
        "граница сцен между выборками": 0,
    }
    for context_index, context in enumerate(contexts):
        pair = context["pair"]
        for segment in context["alignment"].get("segments", []):
            if not segment.get("usable") or float(segment.get("confidence", 0.0)) < minimum_confidence:
                rejected["низкая уверенность сопоставления"] += 1
                continue
            dubbed_start = max(0.0, float(segment["dubbed_start"]))
            dubbed_end = float(segment["dubbed_end"])
            position = dubbed_start
            while position + clip_duration <= dubbed_end + 1e-6:
                _check_stop(ctx)
                original_start = float(segment["original_start"]) + (
                    position - dubbed_start
                ) * float(segment.get("speed_ratio") or 1.0)
                original_duration = clip_duration * float(segment.get("speed_ratio") or 1.0)
                en_data, en_rate = _read_interval(
                    context["original_speech"], original_start, original_duration
                )
                ru_data, ru_rate = _read_interval(
                    context["target_speech"], position, clip_duration
                )
                en = _training_mono(en_data, en_rate, sample_rate, target_frames)
                ru = _training_mono(ru_data, ru_rate, sample_rate, target_frames)
                en_rms = _rms_db(en)
                ru_rms = _rms_db(ru)
                if max(en_rms, ru_rms) < minimum_speech_rms:
                    rejected["нет речи"] += 1
                    position += scan_stride
                    continue
                # Clean translated tracks sometimes keep the untranslated
                # original (songs, background lines).  Such windows would put
                # English speech into the training target and teach the model
                # to preserve it, so they are rejected outright.
                if en_rms > -45.0 and _max_lag_correlation(ru, en, sample_rate) > max_target_reference_correlation:
                    rejected["оригинальная речь в чистом переводе"] += 1
                    position += scan_stride
                    continue
                group = int(position // scene_group_sec)
                split = split_of_group(pair["id"], group)
                # A window (plus the Russian-shift augmentation margin) that
                # crosses into a scene group of another split would leak the
                # same audio across train/valid/test.
                span_start = max(0.0, position - russian_shift_margin)
                span_end = position + clip_duration + russian_shift_margin
                spanned = {
                    split_of_group(pair["id"], g)
                    for g in range(int(span_start // scene_group_sec), int(span_end // scene_group_sec) + 1)
                }
                if len(spanned) > 1:
                    rejected["граница сцен между выборками"] += 1
                    position += scan_stride
                    continue
                candidates.append(
                    {
                        "pair_id": pair["id"],
                        "film_name": pair["name"],
                        "split": split,
                        "dubbed_start_sec": position,
                        "original_start_sec": original_start,
                        "original_duration_sec": original_duration,
                        "alignment_confidence": float(segment.get("confidence", 0.0)),
                        "english_speech_rms_db": en_rms,
                        "russian_speech_rms_db": ru_rms,
                        "sources": {
                            "reference": str(context["original_speech"].resolve()),
                            "target_speech": str(context["target_speech"].resolve()),
                        },
                    }
                )
                position += scan_stride
        ctx.update(
            stage="Поиск речевых фрагментов",
            substage=f"{pair['name']}: найдено {sum(item['pair_id'] == pair['id'] for item in candidates)}",
            progress=30.0 + (context_index + 1) / len(contexts) * 18.0,
            current_file=pair["name"],
        )
    if not candidates:
        raise RuntimeError("В сопоставленных участках не найдено речевых фрагментов.")

    by_split = {
        split: [item for item in candidates if item["split"] == split]
        for split in ("train", "valid", "test")
    }
    if any(not values for values in by_split.values()):
        # The external four-film test remains fully independent.  This fallback
        # only guarantees that the runner can report internal diagnostics.
        ordered = sorted(candidates, key=lambda item: (item["pair_id"], item["dubbed_start_sec"]))
        for index, item in enumerate(ordered):
            bucket = index % 10
            item["split"] = "valid" if bucket == 8 else ("test" if bucket == 9 else "train")
        by_split = {
            split: [item for item in ordered if item["split"] == split]
            for split in ("train", "valid", "test")
        }

    rng = np.random.default_rng(
        int(hashlib.sha256(dataset_id.encode()).hexdigest()[:16], 16)
    )
    split_targets = {
        "train": int(round(requested_examples * 0.80)),
        "valid": int(round(requested_examples * 0.10)),
    }
    split_targets["test"] = requested_examples - split_targets["train"] - split_targets["valid"]
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        pool = by_split[split]
        order = rng.permutation(len(pool))
        for number in range(split_targets[split]):
            source = pool[int(order[number % len(order)])]
            variant = number // len(order)
            seed_text = (
                f"{dataset_id}:{split}:{source['pair_id']}:"
                f"{source['dubbed_start_sec']:.6f}:{variant}"
            )
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
            item_rng = np.random.default_rng(seed)
            use_exact_target = bool(item_rng.random() < 0.20)
            russian_shift_sec = (
                0.0
                if use_exact_target
                else float(item_rng.uniform(-1.5, 1.5))
            )
            row = {
                "id": f"{split}_{number:05d}",
                "pair_id": source["pair_id"],
                "film_name": source["film_name"],
                "split": split,
                "duration_sec": clip_duration,
                "sample_rate": sample_rate,
                "alignment_confidence": source["alignment_confidence"],
                "recipe": {
                    "schema_version": 1,
                    "seed": seed,
                    "sources": source["sources"],
                    "dubbed_start_sec": round(float(source["dubbed_start_sec"]), 6),
                    "original_start_sec": round(float(source["original_start_sec"]), 6),
                    "original_duration_sec": round(float(source["original_duration_sec"]), 6),
                    "english_speech_rms_db": round(float(source["english_speech_rms_db"]), 4),
                    "russian_speech_rms_db": round(float(source["russian_speech_rms_db"]), 4),
                    "use_exact_target": use_exact_target,
                    "russian_shift_sec": round(russian_shift_sec, 6),
                    "english_to_target_db": round(float(item_rng.uniform(-30.0, 6.0)), 4),
                    "english_gain_change_db": round(float(item_rng.uniform(-8.0, 8.0)), 4),
                    "english_delay_ms": round(float(item_rng.uniform(-120.0, 120.0)), 4),
                    "spectral_tilt_db": round(float(item_rng.uniform(-8.0, 8.0)), 4),
                    "soft_compression_drive": round(float(item_rng.uniform(1.0, 2.5)), 4),
                    "reverb_wet": round(float(item_rng.uniform(0.0, 0.25)), 4),
                    # Studios lower the original while the Russian narrator
                    # speaks; the loader shapes this dip from the target
                    # envelope.  Zero keeps a share of un-ducked examples.
                    "ducking_db": round(
                        float(item_rng.uniform(0.0, 12.0))
                        if item_rng.random() < 0.7
                        else 0.0,
                        4,
                    ),
                },
            }
            rows.append(row)
        ctx.update(
            stage="Подготовка 10 000 рецептов обучения",
            substage=f"{split}: {split_targets[split]} примеров",
            progress=48.0 + len(rows) / requested_examples * 50.0,
            current_file=split,
        )

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "valid", "test")
    }
    external_evaluation: list[dict[str, Any]] = []
    external_project_id = str(parameters.get("external_test_project_id") or "").strip()
    if external_project_id:
        external_evaluation = build_real_evaluation(
            store,
            external_project_id,
            dataset_root,
            sample_rate=sample_rate,
            duration_sec=8.0,
            scenes_per_film=5,
        )
    manifest = {
        "schema_version": 8,
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "project_id": project_id,
        "pair_ids": pair_ids,
        "film_count": len(pairs),
        "example_count": len(rows),
        "base_speech_fragment_count": len(candidates),
        "duration_sec": round(len(rows) * clip_duration, 3),
        "sample_rate": sample_rate,
        "clip_duration_sec": clip_duration,
        "split_counts": split_counts,
        "rejected_count": sum(rejected.values()),
        "rejected_reasons": rejected,
        "target_kind": "clean_translation_reference_subtraction_supervised_v3",
        "task": (
            "По оригинальной английской речи удалить её преобразованную составляющую "
            "из смеси голосов EN+RU и оставить только извлечённую русскую речь."
        ),
        "recipe_based": True,
        "augmentation": {
            "english_to_target_db": [-30.0, 6.0],
            "english_gain_change_db": [-8.0, 8.0],
            "english_delay_ms": [-120.0, 120.0],
            "russian_shift_sec": [-1.5, 1.5],
            "spectral_tilt_db": [-8.0, 8.0],
            "soft_compression_drive": [1.0, 2.5],
            "reverb_wet": [0.0, 0.25],
            "ducking_db": [0.0, 12.0],
            "ducking_probability": 0.7,
            "exact_target_fraction": 0.20,
        },
        "integrity": {
            "max_reference_in_target_correlation": max_target_reference_correlation,
            "scene_group_boundary_margin_sec": russian_shift_margin,
            "english_speech_gate_db": -48.0,
        },
        "input_files": ["mixture", "reference"],
        "target_files": ["target_reference", "target_voice"],
        "leakage_policy": (
            "Сцены разделены группами по 120 секунд; четыре старых фильма "
            "не входят в датасет и используются как внешний тест."
        ),
        "examples": rows,
        "real_evaluation": external_evaluation,
        "external_test_project_id": external_project_id or None,
    }
    atomic_json(dataset_root / "manifest.json", manifest)
    project = store.load_project(project_id)
    project["latest_dataset_id"] = dataset_id
    project["latest_dataset"] = {
        key: manifest[key]
        for key in (
            "dataset_id",
            "film_count",
            "example_count",
            "base_speech_fragment_count",
            "duration_sec",
            "split_counts",
            "rejected_count",
            "rejected_reasons",
            "target_kind",
        )
    }
    store.save_project(project)
    for pair in pairs:
        pair["stages"]["8"] = {
            "status": "completed",
            "message": f"Добавлен в датасет чистого перевода {dataset_id}.",
        }
        pair["stages"]["9"] = {
            "status": "available",
            "message": "Можно обучать модель удаления оригинальной речи.",
        }
        store.save_pair(pair)
    ctx.update(
        stage="Датасет чистого перевода готов",
        substage=f"{len(rows)} примеров из {len(candidates)} речевых фрагментов",
        progress=100.0,
    )
    return manifest


def prepare_reference_subtraction_dataset(
    store: Store,
    project_id: str,
    pair_ids: list[str],
    ctx: Any,
    parameters: dict[str, Any] | None = None,
) -> dict:
    """Build supervised examples for a reference-conditioned subtraction model.

    Real dubbed tracks do not contain a clean RU-only target.  Training examples
    therefore use a known construction: a mastering-altered original reference
    plus an independently sampled speech stem.  Both the embedded reference and
    the residual voice are exact targets.  Separate real-film scenes are kept for
    listening tests and leakage diagnostics only.
    """
    parameters = parameters or {}
    if parameters.get("mode") == "clean_translation":
        return _prepare_clean_translation_dataset(
            store, project_id, pair_ids, ctx, parameters
        )
    pairs = [store.load_pair(project_id, item) for item in pair_ids]
    if not pairs:
        raise ValueError("Не выбран ни один фильм.")
    cfg = store.cfg["dataset"]
    sample_rate = int(cfg.get("training_sample_rate", 24000))
    clip_duration = float(cfg["clip_duration_sec"])
    target_frames = int(round(sample_rate * clip_duration))
    stride = float(cfg["clip_stride_sec"])
    minimum_rms = float(cfg["minimum_rms_db"])
    minimum_reference_speech_rms = float(
        cfg.get("minimum_reference_speech_rms_db", -48.0)
    )
    min_confidence = float(cfg["minimum_confidence"])
    maximum = int(cfg["maximum_examples_per_pair"])
    real_examples_per_pair = int(cfg.get("real_evaluation_examples_per_pair", 5))
    dataset_id = time.strftime("%Y%m%d_%H%M%S") + "_refsub_" + uuid.uuid4().hex[:6]
    dataset_root = store.project_dir(project_id) / "training" / "datasets" / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=False)

    pair_context: dict[str, dict] = {}
    sorted_ids = sorted(pair_ids)

    def split_for(pair_id: str) -> str:
        if len(sorted_ids) >= 4:
            if pair_id == sorted_ids[-1]:
                return "test"
            if pair_id == sorted_ids[-2]:
                return "valid"
            return "train"
        digest = int(hashlib.sha256(pair_id.encode()).hexdigest()[:8], 16) % 100
        train_percent = int(cfg["split_percent"]["train"])
        valid_percent = int(cfg["split_percent"]["valid"])
        return "train" if digest < train_percent else ("valid" if digest < train_percent + valid_percent else "test")

    for pair in pairs:
        root = store.pair_dir(project_id, pair["id"])
        if pair.get("stages", {}).get("4", {}).get("status") != "completed":
            raise RuntimeError(f"Для «{pair['name']}» сначала подтвердите сопоставление.")
        speech_manifest = read_json(root / "stems" / "original" / "manifest.json", {})
        speech_path = Path(str(speech_manifest.get("speech") or ""))
        music_effects_path = Path(str(speech_manifest.get("music_effects") or ""))
        if not speech_path.is_file() or not music_effects_path.is_file():
            raise RuntimeError(
                f"Для «{pair['name']}» нет полной речевой составляющей оригинала. "
                "Сначала выполните полное разделение оригинала."
            )
        alignment = _processing_alignment_map(
            store, pair, read_json(root / "alignment" / "alignment_map.json")
        )
        pair_context[pair["id"]] = {
            "pair": pair,
            "root": root,
            "alignment": alignment,
            "speech_path": speech_path,
            "music_effects_path": music_effects_path,
            "speech_duration": float(sf.info(str(speech_path)).duration),
            "split": split_for(pair["id"]),
        }

    donor_groups: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for context in pair_context.values():
        donor_groups[context["split"]].append(context)

    rows: list[dict] = []
    rejected: dict[str, int] = {}
    total_target = max(1, len(pairs) * maximum)
    for context in pair_context.values():
        pair = context["pair"]
        root = context["root"]
        split = context["split"]
        donors = donor_groups.get(split) or [context]
        accepted = 0
        for segment in context["alignment"].get("segments", []):
            if accepted >= maximum:
                break
            segment_start = float(segment["dubbed_start"])
            segment_end = float(segment["dubbed_end"])
            position = segment_start
            while position + clip_duration <= segment_end + 1e-6 and accepted < maximum:
                _check_stop(ctx)
                if not segment.get("usable") or float(segment.get("confidence", 0.0)) < min_confidence:
                    rejected["недостаточная уверенность сопоставления"] = rejected.get("недостаточная уверенность сопоставления", 0) + 1
                    position += stride
                    continue
                original_start = float(segment["original_start"]) + (
                    position - segment_start
                ) * float(segment.get("speed_ratio") or 1.0)
                original_duration = clip_duration * float(segment.get("speed_ratio") or 1.0)
                reference_data, reference_rate = _read_interval(
                    context["speech_path"], original_start, original_duration
                )
                reference = _training_mono(
                    reference_data, reference_rate, sample_rate, target_frames
                )
                if _rms_db(reference) < minimum_reference_speech_rms:
                    rejected["нет достаточно громкой английской речи"] = rejected.get("нет достаточно громкой английской речи", 0) + 1
                    position += stride
                    continue
                background_data, background_rate = _read_interval(
                    context["music_effects_path"], original_start, original_duration
                )
                background = _training_mono(
                    background_data, background_rate, sample_rate, target_frames
                )
                if _rms_db(background) < minimum_rms:
                    background = np.zeros(target_frames, dtype=np.float32)
                seed_text = f"{dataset_id}:{pair['id']}:{position:.6f}"
                seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                rng = np.random.default_rng(seed)
                voice = None
                donor_meta = None
                for _attempt in range(20):
                    donor = donors[int(rng.integers(0, len(donors)))]
                    maximum_start = max(0.0, donor["speech_duration"] - clip_duration)
                    donor_start = float(rng.uniform(0.0, maximum_start)) if maximum_start else 0.0
                    donor_data, donor_rate = _read_interval(
                        donor["speech_path"], donor_start, clip_duration
                    )
                    candidate = _training_mono(
                        donor_data, donor_rate, sample_rate, target_frames
                    )
                    if _rms_db(candidate) > -44.0:
                        voice = candidate
                        donor_meta = {
                            "pair_id": donor["pair"]["id"],
                            "film_name": donor["pair"]["name"],
                            "start_sec": round(donor_start, 6),
                        }
                        break
                if voice is None:
                    rejected["не найден речевой донор"] = rejected.get("не найден речевой донор", 0) + 1
                    position += stride
                    continue
                reference_component, transform = _synthetic_reference_component(
                    reference, sample_rate, rng
                )
                reference_rms = max(float(np.sqrt(np.mean(reference_component.astype(np.float64) ** 2))), 1e-6)
                voice_rms = max(float(np.sqrt(np.mean(voice.astype(np.float64) ** 2))), 1e-6)
                voice_to_reference_db = float(rng.uniform(-8.0, 5.0))
                voice *= float(reference_rms / voice_rms * 10.0 ** (voice_to_reference_db / 20.0))
                background_gain_db = float(rng.uniform(-3.0, 3.0))
                background *= float(10.0 ** (background_gain_db / 20.0))
                residual = background + voice
                mixture = reference_component + residual
                peak = max(float(np.max(np.abs(mixture))), 1.0)
                scale = 0.96 / peak
                mixture *= scale
                reference_component *= scale
                residual *= scale
                voice *= scale
                background *= scale
                example_id = f"{pair['id'][:6]}_{accepted:04d}"
                example_dir = dataset_root / split / example_id
                files = {
                    "mixture": _write_training_audio(example_dir / "synthetic_dubbed_mix.flac", mixture, sample_rate),
                    "reference": _write_training_audio(example_dir / "clean_english_speech_reference.flac", reference, sample_rate),
                    "target_reference": _write_training_audio(example_dir / "target_english_speech_inside_mix.flac", reference_component, sample_rate),
                    "target_voice": _write_training_audio(example_dir / "target_residual_voice_and_background.flac", residual, sample_rate),
                    "target_translation_voice": _write_training_audio(example_dir / "target_translation_voice.flac", voice, sample_rate),
                    "target_music_effects": _write_training_audio(example_dir / "target_music_and_effects.flac", background, sample_rate),
                }
                row = {
                    "id": example_id,
                    "pair_id": pair["id"],
                    "film_name": pair["name"],
                    "split": split,
                    "dubbed_start_sec": round(position, 6),
                    "original_start_sec": round(original_start, 6),
                    "duration_sec": clip_duration,
                    "sample_rate": sample_rate,
                    "alignment_confidence": float(segment.get("confidence", 0.0)),
                    "voice_to_reference_db": round(voice_to_reference_db, 4),
                    "background_gain_db": round(background_gain_db, 4),
                    "reference_transform": transform,
                    "speech_donor": donor_meta,
                    "files": files,
                }
                atomic_json(example_dir / "metadata.json", row)
                rows.append(row)
                accepted += 1
                position += stride
                ctx.update(
                    stage="Подготовка датасета парного вычитания",
                    substage=f"{pair['name']}: пример {accepted} из {maximum}",
                    progress=min(92.0, len(rows) / total_target * 92.0),
                    current_file=pair["name"],
                )

    if not rows:
        raise RuntimeError("Не удалось создать ни одного обучающего примера.")

    real_evaluation: list[dict] = []
    for context in pair_context.values():
        pair_rows = [row for row in rows if row["pair_id"] == context["pair"]["id"]]
        if not pair_rows:
            continue
        indexes = np.linspace(0, len(pair_rows) - 1, min(real_examples_per_pair, len(pair_rows)), dtype=int)
        for number, index in enumerate(indexes, 1):
            source_row = pair_rows[int(index)]
            position = float(source_row["dubbed_start_sec"])
            segment = _segment_for_time(context["alignment"].get("segments", []), position)
            if not segment:
                continue
            original_start = float(segment["original_start"]) + (
                position - float(segment["dubbed_start"])
            ) * float(segment.get("speed_ratio") or 1.0)
            original_duration = clip_duration * float(segment.get("speed_ratio") or 1.0)
            dubbed_data, dubbed_rate = _read_interval(
                context["root"] / "extracted" / "dubbed.flac", position, clip_duration
            )
            reference_data, reference_rate = _read_interval(
                context["speech_path"], original_start, original_duration
            )
            dubbed = _training_mono(dubbed_data, dubbed_rate, sample_rate, target_frames)
            reference = _training_mono(reference_data, reference_rate, sample_rate, target_frames)
            example_dir = dataset_root / "real_evaluation" / context["pair"]["id"] / f"{number:02d}"
            real_evaluation.append(
                {
                    "id": f"{context['pair']['id'][:6]}_real_{number:02d}",
                    "pair_id": context["pair"]["id"],
                    "film_name": context["pair"]["name"],
                    "dubbed_start_sec": round(position, 6),
                    "duration_sec": clip_duration,
                    "sample_rate": sample_rate,
                    "files": {
                        "mixture": _write_training_audio(example_dir / "real_dubbed_mix.flac", dubbed, sample_rate),
                        "reference": _write_training_audio(example_dir / "aligned_english_speech_reference.flac", reference, sample_rate),
                    },
                }
            )

    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "valid", "test")
    }
    manifest = {
        "schema_version": 6,
        "dataset_id": dataset_id,
        "created_at": utc_now(),
        "project_id": project_id,
        "pair_ids": pair_ids,
        "film_count": len(pairs),
        "example_count": len(rows),
        "real_evaluation_count": len(real_evaluation),
        "duration_sec": round(len(rows) * clip_duration, 3),
        "sample_rate": sample_rate,
        "clip_duration_sec": clip_duration,
        "split_counts": split_counts,
        "rejected_count": sum(rejected.values()),
        "rejected_reasons": rejected,
        "target_kind": "paired_speech_reference_subtraction_supervised_v2",
        "task": "Восстановить английскую речь внутри смеси; остатком получить голос перевода с музыкой и эффектами.",
        "input_files": ["mixture", "reference"],
        "target_files": ["target_reference", "target_voice", "target_translation_voice", "target_music_effects"],
        "leakage_policy": "Обучение, проверка и тест разделены по фильмам; речевые доноры не пересекают разделы.",
        "examples": rows,
        "real_evaluation": real_evaluation,
    }
    atomic_json(dataset_root / "manifest.json", manifest)
    project = store.load_project(project_id)
    project["latest_dataset_id"] = dataset_id
    project["latest_dataset"] = {
        key: manifest[key]
        for key in (
            "dataset_id",
            "film_count",
            "example_count",
            "real_evaluation_count",
            "duration_sec",
            "split_counts",
            "rejected_count",
            "rejected_reasons",
            "target_kind",
        )
    }
    store.save_project(project)
    for pair in pairs:
        pair["stages"]["8"] = {
            "status": "completed",
            "message": f"Добавлен в датасет парного вычитания {dataset_id}.",
        }
        pair["stages"]["9"] = {
            "status": "available",
            "message": "Можно обучать модель и запускать контрольные прогоны.",
        }
        store.save_pair(pair)
    return manifest


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(destination)
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_pair(
    store: Store,
    project_id: str,
    pair_id: str,
    ctx: Any,
    parameters: dict,
) -> dict:
    pair = store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("6", {}).get("status") != "completed":
        raise RuntimeError("Сначала выполните сборку выбранным методом.")
    root = store.pair_dir(project_id, pair_id)
    component_manifest = pair.get("components") or read_json(
        root / "components" / "manifest.json", {}
    )
    selected_method = str(component_manifest.get("selected_method") or "")
    component_files = {
        key: Path(value)
        for key, value in (component_manifest.get("files") or {}).items()
        if value
    }
    if selected_method not in {"direct", "stem"} or not component_files.get(
        "final"
    ):
        raise RuntimeError("Не найден результат метода, выбранного на шаге 6.")
    final_root = root / "results" / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    source = component_files["final"]
    destination = final_root / f"{selected_method}_russian_result.flac"
    ctx.update(
        stage="Финализация полного результата",
        substage=f"Копирование результата метода {selected_method}",
        progress=20.0,
    )
    _atomic_copy(source, destination)
    copied_components: dict[str, str] = {}
    component_names = {
        "aligned_en_speech": "english_speech_aligned.flac",
        "ru_voice": "russian_speech.flac",
        "original_me": "music_and_effects_aligned.flac",
        "dubbed_speech_en_ru": "combined_speech_en_ru.flac",
        "full_original_en_speech": "english_speech_original_timeline.flac",
        "full_original_me": "music_and_effects_original_timeline.flac",
    }
    for key, destination_name in component_names.items():
        source_path = component_files.get(key)
        if source_path is None or not source_path.is_file():
            continue
        target_path = final_root / destination_name
        _atomic_copy(source_path, target_path)
        copied_components[key] = str(target_path.resolve())
    amap = read_json(root / "alignment" / "alignment_map.json")
    problem = [item for item in amap.get("segments", []) if not item.get("usable")]
    report = {
        "schema_version": 5,
        "created_at": utc_now(),
        "selected_method": selected_method,
        "selected_method_label": component_manifest.get("selected_method_label"),
        "result": str(destination.resolve()),
        "components": copied_components,
        "low_confidence_strategy": store.cfg["full_processing"]["low_confidence_strategy"],
        "problem_segments": problem,
        "warning": (
            "На несопоставленных участках сохранено исходное аудио RU+EN; "
            "итог необходимо проверить на слух."
        ),
    }
    remux = bool(parameters.get("remux", True))
    if remux:
        ffmpeg = audio_io.check_ffmpeg()
        video = Path(pair["sources"]["dubbed"]["path"]).resolve()
        remux_path = final_root / f"result.{store.cfg['full_processing']['remux_container']}"
        temporary = _temporary(remux_path)
        remux_cfg = dict(store.cfg.get("full_processing") or {})
        command = audio_io.build_remux_command(
            ffmpeg,
            video,
            destination,
            temporary,
            audio_codec=str(remux_cfg.get("remux_audio_codec") or "aac"),
            audio_bitrate=str(remux_cfg.get("remux_audio_bitrate") or ""),
            audio_channels=int(remux_cfg.get("remux_audio_channels") or 0),
            audio_sample_rate=int(remux_cfg.get("remux_audio_sample_rate") or 0),
            track_title="Очищенная русская дорожка",
        )
        try:
            _run_ffmpeg_progress(
                command,
                float(pair["extraction"]["files"]["dubbed"]["duration_sec"]),
                ctx,
                "Сборка видео без перекодирования видеопотока",
                video.name,
            )
            os.replace(temporary, remux_path)
        finally:
            temporary.unlink(missing_ok=True)
        report["remux"] = str(remux_path.resolve())
    atomic_json(final_root / "metadata.json", report)
    atomic_json(root / "reports" / "problem_segments.json", {"segments": problem})
    pair["result"] = report
    pair["current_stage"] = 10
    pair["stages"]["10"] = {
        "status": "completed",
        "message": (
            "Итоговое видео и аудиодорожка готовы."
            if remux
            else "Итоговая аудиодорожка готова."
        ),
    }
    store.save_pair(pair)
    ctx.update(stage="Полная обработка фильма", substage="Готово", progress=100.0)
    return report
