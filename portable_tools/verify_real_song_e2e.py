#!/usr/bin/env python3
"""Read-only verification of a completed full-film song-protection result.

The verifier never changes project files.  It reads the completed manifest and
audio artifacts, writes one JSON report, and can optionally create short FLAC
A/B clips outside the project directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

PYTHON_ROOT = Path(__file__).resolve().parents[1] / "python_src"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiments.paired_reference_cancel import audio_io


class VerificationError(RuntimeError):
    """A completed result is missing an artifact required for verification."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise VerificationError(f"JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise VerificationError(f"Invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object in {path}")
    return value


def require_file(value: Any, label: str) -> Path:
    path = Path(str(value or "")).expanduser()
    if not str(value or "").strip() or not path.is_file() or path.stat().st_size <= 0:
        raise VerificationError(f"{label} is missing or empty: {path}")
    return path.resolve()


def audio_info(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "path": str(path),
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 6),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def pcm_lsb(subtype: str) -> float:
    bits = {
        "PCM_S8": 7,
        "PCM_U8": 7,
        "PCM_16": 15,
        "PCM_24": 23,
        "PCM_32": 31,
    }.get(str(subtype).upper())
    return 2.0 ** (-bits) if bits is not None else 1.0e-7


def compare_interval(
    actual_path: Path,
    expected_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    label: str,
    block_frames: int = 262_144,
) -> dict[str, Any]:
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))
    actual_meta = audio_info(actual_path)
    expected_meta = audio_info(expected_path)
    result: dict[str, Any] = {
        "label": label,
        "start_sec": round(start_sec, 6),
        "end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
        "actual": str(actual_path),
        "expected": str(expected_path),
    }
    if actual_meta["samplerate"] != expected_meta["samplerate"]:
        result.update(
            {
                "pass": False,
                "reason": "sample rate differs",
                "actual_audio": actual_meta,
                "expected_audio": expected_meta,
            }
        )
        return result

    rate = int(actual_meta["samplerate"])
    channels = int(actual_meta["channels"])
    expected_channels = int(expected_meta["channels"])
    if expected_channels != channels:
        result["expected_channel_adaptation"] = {
            "source_channels": expected_channels,
            "target_channels": channels,
            "method": "audio_io.match_channels",
        }
    start_frame = int(math.ceil(start_sec * rate))
    requested_end_frame = int(math.floor(end_sec * rate))
    available_end_frame = min(
        requested_end_frame,
        int(actual_meta["frames"]),
        int(expected_meta["frames"]),
    )
    requested_frames = max(0, requested_end_frame - start_frame)
    frames_to_compare = max(0, available_end_frame - start_frame)
    if frames_to_compare <= 0:
        result.update({"pass": False, "reason": "comparison interval is empty"})
        return result

    element_count = 0
    exact_count = 0
    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    sum_squared_expected = 0.0
    maximum_absolute_error = 0.0
    with sf.SoundFile(str(actual_path)) as actual_file, sf.SoundFile(
        str(expected_path)
    ) as expected_file:
        actual_file.seek(start_frame)
        expected_file.seek(start_frame)
        remaining = frames_to_compare
        while remaining > 0:
            count = min(block_frames, remaining)
            actual = actual_file.read(count, dtype="float32", always_2d=True)
            expected = expected_file.read(count, dtype="float32", always_2d=True)
            count = min(len(actual), len(expected))
            if count <= 0:
                break
            actual = actual[:count]
            expected = audio_io.match_channels(expected[:count], channels)
            difference = actual.astype(np.float64) - expected.astype(np.float64)
            absolute = np.abs(difference)
            element_count += int(difference.size)
            exact_count += int(np.count_nonzero(difference == 0.0))
            sum_squared_error += float(np.sum(difference * difference))
            sum_absolute_error += float(np.sum(absolute))
            sum_squared_expected += float(
                np.sum(expected.astype(np.float64) ** 2)
            )
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(np.max(absolute, initial=0.0)),
            )
            remaining -= count

    compared_frames = element_count // max(channels, 1)
    rms_error = math.sqrt(sum_squared_error / max(element_count, 1))
    expected_rms = math.sqrt(sum_squared_expected / max(element_count, 1))
    tolerance = 1.1 * max(
        pcm_lsb(actual_meta["subtype"]),
        pcm_lsb(expected_meta["subtype"]),
    )
    complete = compared_frames == requested_frames
    consistent = complete and maximum_absolute_error <= tolerance
    result.update(
        {
            "pass": bool(consistent),
            "requested_frames": requested_frames,
            "compared_frames": compared_frames,
            "sample_values_compared": element_count,
            "exact_sample_fraction": round(
                exact_count / max(element_count, 1), 9
            ),
            "maximum_absolute_error": maximum_absolute_error,
            "mean_absolute_error": (
                sum_absolute_error / max(element_count, 1)
            ),
            "rms_error": rms_error,
            "expected_rms": expected_rms,
            "snr_db": (
                None
                if rms_error == 0.0
                else round(20.0 * math.log10(max(expected_rms, 1.0e-15) / rms_error), 3)
            ),
            "allowed_error": tolerance,
            "actual_subtype": actual_meta["subtype"],
            "expected_subtype": expected_meta["subtype"],
        }
    )
    if not complete:
        result["reason"] = "one of the files ended before the requested interval"
    elif not consistent:
        result["reason"] = "decoded samples differ beyond one PCM quantisation step"
    return result


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    ):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def overlap_duration(
    target_start: float,
    target_end: float,
    intervals: Iterable[tuple[float, float]],
) -> float:
    return sum(
        max(0.0, min(target_end, end) - max(target_start, start))
        for start, end in merge_intervals(intervals)
    )


def blocked_dialogue_intervals(
    song_report: dict[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for song_index, interval in enumerate(
        song_report.get("confirmed_intervals") or []
    ):
        dialogue = interval.get("dialogue_detection") or {}
        for item in dialogue.get("blocked_restoration_intervals") or []:
            value = dict(item)
            value["song_index"] = song_index
            result.append(value)
    return result


def scan_expectations(
    scan_report: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    songs: list[dict[str, Any]] = []
    dialogues: list[dict[str, Any]] = []
    if not scan_report:
        return songs, dialogues
    for zone_index, zone in enumerate(scan_report.get("detected_song_zones") or []):
        target = zone.get("acceptance_target") or {}
        for item in target.get("restore_english_full_mix") or []:
            songs.append(
                {
                    "zone_index": zone_index,
                    "start_sec": float(item["start_sec"]),
                    "end_sec": float(item["end_sec"]),
                }
            )
        for item in target.get("keep_processed") or []:
            dialogues.append(
                {
                    "zone_index": zone_index,
                    "start_sec": float(item["start_sec"]),
                    "end_sec": float(item["end_sec"]),
                    "reason": str(item.get("reason") or ""),
                }
            )
    return songs, dialogues


def decoded_pcm_hash(ffmpeg: str, source: Path) -> dict[str, Any]:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s24le",
        "-f",
        "hash",
        "-hash",
        "sha256",
        "-",
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"SHA256=([0-9a-fA-F]{64})", completed.stdout)
    if completed.returncode != 0 or match is None:
        raise VerificationError(
            f"ffmpeg could not hash {source}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return {
        "path": str(source),
        "decoded_pcm_format": "signed 24-bit little-endian",
        "sha256": match.group(1).lower(),
    }


def write_clip(
    source: Path,
    destination: Path,
    start_sec: float,
    end_sec: float,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(str(source)) as reader:
        rate = int(reader.samplerate)
        channels = int(reader.channels)
        start_frame = max(0, int(round(float(start_sec) * rate)))
        end_frame = min(
            int(reader.frames), int(round(float(end_sec) * rate))
        )
        reader.seek(start_frame)
        temporary = destination.with_name(f".{destination.name}.partial")
        try:
            with sf.SoundFile(
                str(temporary),
                mode="w",
                samplerate=rate,
                channels=channels,
                format="FLAC",
                subtype="PCM_24",
            ) as writer:
                remaining = max(0, end_frame - start_frame)
                while remaining > 0:
                    values = reader.read(
                        min(262_144, remaining),
                        dtype="float32",
                        always_2d=True,
                    )
                    if len(values) <= 0:
                        break
                    writer.write(values)
                    remaining -= len(values)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(destination.resolve()),
        "start_sec": round(start_frame / rate, 6),
        "end_sec": round(end_frame / rate, 6),
        "duration_sec": round((end_frame - start_frame) / rate, 6),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def clip_ranges(
    scan_report: dict[str, Any] | None,
    song_report: dict[str, Any],
    *,
    padding_sec: float,
    maximum_duration_sec: float,
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    scan_zones = (scan_report or {}).get("detected_song_zones") or []
    if scan_zones:
        for index, zone in enumerate(scan_zones):
            target = zone.get("acceptance_target") or {}
            values = [
                item
                for key in ("restore_english_full_mix", "keep_processed")
                for item in (target.get(key) or [])
            ]
            if values:
                start = min(float(item["start_sec"]) for item in values)
                end = max(float(item["end_sec"]) for item in values)
            else:
                start = float(zone["classifier_start_sec"])
                end = float(zone["classifier_end_sec"])
            start = max(0.0, start - padding_sec)
            end = end + padding_sec
            if end - start > maximum_duration_sec:
                end = start + maximum_duration_sec
            ranges.append(
                {
                    "zone_index": index,
                    "start_sec": start,
                    "end_sec": end,
                }
            )
        return ranges

    for index, interval in enumerate(song_report.get("confirmed_intervals") or []):
        start = max(0.0, float(interval["start_sec"]) - padding_sec)
        end = min(
            float(interval["end_sec"]) + padding_sec,
            start + maximum_duration_sec,
        )
        ranges.append(
            {"zone_index": index, "start_sec": start, "end_sec": end}
        )
    return ranges


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = require_file(args.manifest, "full result manifest")
    manifest = read_json(manifest_path)
    if str(manifest.get("method") or "") != "speech_rebuild":
        raise VerificationError(
            "Song protection verification requires method=speech_rebuild"
        )

    intermediates = manifest.get("intermediates") or {}
    report_value = intermediates.get("song_protection_report")
    if report_value:
        song_report_path = require_file(
            report_value, "song protection report"
        )
        song_report = read_json(song_report_path)
    else:
        song_report_path = manifest_path
        song_report = manifest.get("song_protection") or {}
    classifier = song_report.get("classifier") or {}
    protected = require_file(manifest.get("audio"), "protected final FLAC")
    aligned_original = require_file(
        intermediates.get("song_detection_original_mix")
        or ((song_report.get("restoration_policy") or {}).get("sources") or {}).get(
            "original_aligned"
        ),
        "aligned English original",
    )
    synchronization_enabled = bool(
        (manifest.get("speech_synchronization") or {}).get("enabled")
    )
    pre_key = (
        "final_before_song_protection"
        if synchronization_enabled
        else "result_before_song_protection"
    )
    pre_protection = require_file(
        intermediates.get(pre_key)
        or intermediates.get("result_before_song_protection"),
        "pre-protection processed FLAC",
    )
    movie = require_file(manifest.get("movie"), "result MKV")

    scan_report_path: Path | None = None
    scan_report: dict[str, Any] | None = None
    if args.scan_report:
        scan_report_path = require_file(args.scan_report, "real song scan report")
        scan_report = read_json(scan_report_path)

    restoration_segments = [
        dict(item) for item in (song_report.get("restoration_segments") or [])
    ]
    crossfade_sec = float(
        ((song_report.get("restoration_policy") or {}).get("crossfade_sec"))
        or ((song_report.get("config") or {}).get("crossfade_sec"))
        or 0.35
    )
    core_checks: list[dict[str, Any]] = []
    original_spans: list[tuple[float, float]] = []
    for index, segment in enumerate(restoration_segments):
        if str(segment.get("restore_source") or "") != "original_aligned":
            continue
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        original_spans.append((start, end))
        margin = crossfade_sec + float(args.core_guard_sec)
        core_start = start + margin
        core_end = end - margin
        if core_end <= core_start:
            core_checks.append(
                {
                    "label": f"restoration segment {index}",
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "pass": False,
                    "reason": "segment has no constant-gain core outside crossfades",
                }
            )
            continue
        check = compare_interval(
            protected,
            aligned_original,
            core_start,
            core_end,
            label=f"restoration segment {index} core",
        )
        check["segment_index"] = index
        check["song_index"] = segment.get("song_index")
        check["crossfade_sec"] = crossfade_sec
        core_checks.append(check)

    blocked_checks: list[dict[str, Any]] = []
    for index, interval in enumerate(blocked_dialogue_intervals(song_report)):
        start = float(interval["start_sec"]) + float(args.dialogue_guard_sec)
        end = float(interval["end_sec"]) - float(args.dialogue_guard_sec)
        check = compare_interval(
            protected,
            pre_protection,
            start,
            end,
            label=f"reported blocked dialogue {index}",
        )
        check["song_index"] = interval.get("song_index")
        check["decision"] = interval.get("decision")
        blocked_checks.append(check)

    expected_songs, expected_dialogues = scan_expectations(scan_report)
    coverage_checks: list[dict[str, Any]] = []
    for item in expected_songs:
        duration = float(item["end_sec"]) - float(item["start_sec"])
        covered = overlap_duration(
            float(item["start_sec"]),
            float(item["end_sec"]),
            original_spans,
        )
        ratio = covered / max(duration, 1.0e-12)
        coverage_checks.append(
            {
                **item,
                "duration_sec": round(duration, 6),
                "restored_duration_sec": round(covered, 6),
                "coverage_ratio": round(ratio, 6),
                "minimum_required_ratio": float(args.minimum_song_coverage),
                "pass": ratio >= float(args.minimum_song_coverage),
            }
        )

    acceptance_dialogue_checks: list[dict[str, Any]] = []
    for index, item in enumerate(expected_dialogues):
        start = float(item["start_sec"]) + float(args.dialogue_guard_sec)
        end = float(item["end_sec"]) - float(args.dialogue_guard_sec)
        check = compare_interval(
            protected,
            pre_protection,
            start,
            end,
            label=f"scan acceptance dialogue {index}",
        )
        check["zone_index"] = item.get("zone_index")
        check["reason"] = item.get("reason")
        acceptance_dialogue_checks.append(check)

    ffmpeg = str(args.ffmpeg)
    resolved_ffmpeg = shutil.which(ffmpeg) if not Path(ffmpeg).is_file() else ffmpeg
    if not resolved_ffmpeg:
        raise VerificationError(f"ffmpeg not found: {ffmpeg}")
    protected_hash = decoded_pcm_hash(str(resolved_ffmpeg), protected)
    movie_hash = decoded_pcm_hash(str(resolved_ffmpeg), movie)
    movie_check = {
        "pass": protected_hash["sha256"] == movie_hash["sha256"],
        "protected_final": protected_hash,
        "result_mkv_audio": movie_hash,
    }
    if not movie_check["pass"]:
        movie_check["reason"] = (
            "decoded MKV audio does not match the protected final FLAC"
        )

    clips: list[dict[str, Any]] = []
    if args.clips_dir:
        clips_dir = Path(args.clips_dir).expanduser().resolve()
        for item in clip_ranges(
            scan_report,
            song_report,
            padding_sec=float(args.clip_padding_sec),
            maximum_duration_sec=float(args.maximum_clip_sec),
        ):
            zone_number = int(item["zone_index"]) + 1
            before = clips_dir / f"song-zone-{zone_number:02d}-A-before.flac"
            after = clips_dir / f"song-zone-{zone_number:02d}-B-after.flac"
            clips.append(
                {
                    **item,
                    "before": write_clip(
                        pre_protection,
                        before,
                        float(item["start_sec"]),
                        float(item["end_sec"]),
                    ),
                    "after": write_clip(
                        protected,
                        after,
                        float(item["start_sec"]),
                        float(item["end_sec"]),
                    ),
                }
            )

    classifier_pass = classifier.get("available") is True
    restoration_pass = bool(original_spans) and bool(core_checks) and all(
        item.get("pass") is True for item in core_checks
    )
    blocked_pass = all(item.get("pass") is True for item in blocked_checks)
    acceptance_dialogue_pass = all(
        item.get("pass") is True for item in acceptance_dialogue_checks
    )
    coverage_pass = all(item.get("pass") is True for item in coverage_checks)
    overall_pass = all(
        (
            classifier_pass,
            restoration_pass,
            blocked_pass,
            acceptance_dialogue_pass,
            coverage_pass,
            movie_check["pass"],
        )
    )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "state": "passed" if overall_pass else "failed",
        "pass": overall_pass,
        "task_id": str(args.task_id or ""),
        "inputs": {
            "manifest": str(manifest_path),
            "song_protection_report": str(song_report_path),
            "scan_report": str(scan_report_path) if scan_report_path else "",
            "protected_final": audio_info(protected),
            "pre_protection": audio_info(pre_protection),
            "aligned_english_original": audio_info(aligned_original),
            "result_mkv": str(movie),
        },
        "classifier": {
            **classifier,
            "pass": classifier_pass,
        },
        "detector_summary": {
            "summary": song_report.get("summary"),
            "analysed_windows": song_report.get("analysed_windows"),
            "confirmed_interval_count": len(
                song_report.get("confirmed_intervals") or []
            ),
            "restoration_segment_count": len(restoration_segments),
            "original_restoration_segment_count": len(original_spans),
            "crossfade_sec": crossfade_sec,
        },
        "protected_core_checks": {
            "pass": restoration_pass,
            "checks": core_checks,
        },
        "reported_blocked_dialogue_checks": {
            "pass": blocked_pass,
            "checks": blocked_checks,
        },
        "scan_song_coverage_checks": {
            "pass": coverage_pass,
            "checks": coverage_checks,
        },
        "scan_dialogue_tail_checks": {
            "pass": acceptance_dialogue_pass,
            "checks": acceptance_dialogue_checks,
        },
        "mkv_audio_check": movie_check,
        "ab_clips": clips,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Completed application/full/speech_rebuild/manifest.json",
    )
    parser.add_argument(
        "--scan-report",
        type=Path,
        help="Optional real-song-scan-report.json with acceptance intervals",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON report outside the project artifacts",
    )
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--core-guard-sec", type=float, default=0.05)
    parser.add_argument("--dialogue-guard-sec", type=float, default=0.05)
    parser.add_argument("--minimum-song-coverage", type=float, default=0.90)
    parser.add_argument("--clip-padding-sec", type=float, default=1.0)
    parser.add_argument("--maximum-clip-sec", type=float, default=45.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        report = verify(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "created_at": utc_now(),
            "state": "error",
            "pass": False,
            "task_id": str(args.task_id or ""),
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs": {
                "manifest": str(Path(args.manifest).expanduser().resolve()),
                "scan_report": (
                    str(Path(args.scan_report).expanduser().resolve())
                    if args.scan_report
                    else ""
                ),
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("pass") is True else 2


if __name__ == "__main__":
    sys.exit(main())
