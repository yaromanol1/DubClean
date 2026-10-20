"""Build a reproducible, publication-safe A/B speech comparison.

The public comparison contains clips from the two *final mixes*, while level
metrics are measured on the corresponding voice-only tracks.  A comparison is
published only after all fourteen clips have been created and the supplied
mixes have passed the shared-background contract on every review window.

The command is intentionally specific to the seven audited ``Snatch`` windows.
This prevents a stale or partial ad-hoc manifest from silently becoming the
public result of a different experiment.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import soundfile as sf

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.storage import atomic_json, utc_now


COMPARISON_ID = "snatch_speech_guard_20260801"
TITLE = "Большой куш — сравнение качества речи"
DESCRIPTION = "Контрольные фрагменты до и после исправления обработки русской речи."
PUBLICATION_READY = "ready"
PUBLICATION_WAITING = "awaiting_after_rebuild"
CLIP_SAMPLE_RATE = 48_000
MIX_CLIP_LIMIT = 0.98
BACKGROUND_RESIDUAL_MAX_ABS = 2.0e-5
BACKGROUND_RESIDUAL_RMS = 4.0e-6


@dataclass(frozen=True)
class ComparisonWindow:
    case_id: str
    file_stem: str
    event_start_sec: float
    event_end_sec: float
    clip_start_sec: float
    clip_duration_sec: float


SNATCH_WINDOWS: tuple[ComparisonWindow, ...] = (
    ComparisonWindow("case-01", "case_01_002435", 1475.25, 1476.25, 1473.75, 5.0),
    ComparisonWindow("case-02", "case_02_004337", 2617.25, 2617.75, 2615.75, 4.0),
    ComparisonWindow("case-03", "case_03_005609", 3369.25, 3369.75, 3367.50, 4.0),
    ComparisonWindow("case-04", "case_04_005748", 3468.75, 3469.75, 3467.25, 4.5),
    ComparisonWindow("case-05", "case_05_011503", 4503.75, 4504.25, 4502.25, 4.0),
    ComparisonWindow("case-06", "case_06_012257", 4977.25, 4977.75, 4975.75, 4.0),
    ComparisonWindow("case-07", "case_07_014137", 6097.70, 6098.10, 6095.75, 4.5),
)


def _default_after_dependencies() -> tuple[Path, ...]:
    module_root = Path(__file__).resolve().parent
    return tuple(
        path
        for path in (
            module_root / "application_pipeline.py",
            module_root / "audio_mix.py",
            module_root / "speech_integrity_guard.py",
            module_root / "speech_stem_runner.py",
            module_root / "speech_synchronizer.py",
            module_root / "config.yaml",
        )
        if path.is_file()
    )


def _audio_identity(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_sec": round(float(info.duration), 6),
        "format": str(info.format),
        "subtype": str(info.subtype),
    }


def _require_audio(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label}: файл не найден: {value}")
    try:
        info = sf.info(str(value))
    except RuntimeError as error:
        raise ValueError(f"{label}: не удалось прочитать аудио: {value}") from error
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"{label}: пустой или некорректный аудиофайл: {value}")
    return value


def _same_file(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return first.resolve() == second.resolve()


def _validate_after_freshness(
    after_voice: Path,
    after_mix: Path,
    dependencies: Iterable[Path],
) -> dict[str, Any]:
    existing = [Path(item).resolve() for item in dependencies if Path(item).is_file()]
    if not existing:
        return {"checked": False, "dependencies": []}
    newest = max(existing, key=lambda item: item.stat().st_mtime_ns)
    newest_ns = int(newest.stat().st_mtime_ns)
    stale = [
        path
        for path in (after_voice, after_mix)
        if int(path.stat().st_mtime_ns) < newest_ns
    ]
    if stale:
        names = ", ".join(str(path) for path in stale)
        raise ValueError(
            "Вариант B старше действующего кода обработки. "
            f"Сначала выполните полный rebuild: {names}"
        )
    return {
        "checked": True,
        "newest_dependency": str(newest),
        "newest_dependency_mtime_ns": newest_ns,
        "dependencies": [str(item) for item in existing],
    }


def _validate_lengths(
    before_voice: Path,
    before_mix: Path,
    after_voice: Path,
    after_mix: Path,
    windows: Sequence[ComparisonWindow],
) -> dict[str, Any]:
    identities = {
        "before_voice": _audio_identity(before_voice),
        "before_mix": _audio_identity(before_mix),
        "after_voice": _audio_identity(after_voice),
        "after_mix": _audio_identity(after_mix),
    }
    before_mix_info = sf.info(str(before_mix))
    after_mix_info = sf.info(str(after_mix))
    if (
        int(before_mix_info.samplerate) != int(after_mix_info.samplerate)
        or int(before_mix_info.channels) != int(after_mix_info.channels)
        or int(before_mix_info.frames) != int(after_mix_info.frames)
    ):
        raise ValueError("Финальные миксы A и B имеют разные форматы или длительность.")
    for label, path in (("A voice", before_voice), ("B voice", after_voice)):
        info = sf.info(str(path))
        if int(info.samplerate) != int(before_mix_info.samplerate):
            raise ValueError(f"{label} и final mix имеют разную частоту дискретизации.")
        if abs(int(info.frames) - int(before_mix_info.frames)) > 1:
            raise ValueError(f"{label} и final mix имеют разную длительность.")
    required_end = max(
        window.clip_start_sec + window.clip_duration_sec for window in windows
    )
    if float(before_mix_info.duration) + 1e-6 < required_end:
        raise ValueError(
            "Входные дорожки короче последнего обязательного контрольного окна."
        )
    return identities


def _read_interval(
    path: Path,
    start_sec: float,
    duration_sec: float,
    *,
    dtype: str = "float64",
) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as reader:
        rate = int(reader.samplerate)
        start = int(round(float(start_sec) * rate))
        frames = int(round(float(duration_sec) * rate))
        reader.seek(start)
        values = reader.read(frames, dtype=dtype, always_2d=True)
    if len(values) != frames:
        raise ValueError(f"Недостаточно аудиоданных в {path.name} на {start_sec:.3f} с.")
    return np.asarray(values), rate


def _voice_rms_db(path: Path, start_sec: float, end_sec: float) -> float | None:
    values, _ = _read_interval(path, start_sec, end_sec - start_sec)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    if rms == 0.0:
        return None
    if not math.isfinite(rms):
        raise ValueError(f"В речевой дорожке есть NaN/Inf: {path}")
    return round(20.0 * math.log10(rms), 2)


def _public_db(value: float | None) -> float | str:
    return "цифровая тишина" if value is None else float(value)


def _objective_decision(
    before_db: float | None,
    after_db: float | None,
) -> tuple[str, str, float | None]:
    if before_db is None and after_db is not None:
        return (
            "Речь восстановлена",
            "В итоговой речевой дорожке восстановлен отсутствовавший фрагмент.",
            None,
        )
    if before_db is None or after_db is None:
        return (
            "Требуется проверка",
            "Один из вариантов содержит цифровую тишину.",
            None,
        )
    difference = round(after_db - before_db, 2)
    if difference >= 1.0:
        return (
            "Речь восстановлена",
            "Уровень русской речи восстановлен без изменения фоновой дорожки.",
            difference,
        )
    if difference <= -1.0:
        return (
            "Требуется проверка",
            "После обработки уровень речи снизился.",
            difference,
        )
    return (
        "Без заметного изменения",
        "Существенного изменения уровня речи не обнаружено.",
        difference,
    )


def _mix_contract(
    before_voice: Path,
    before_mix: Path,
    after_voice: Path,
    after_mix: Path,
    windows: Sequence[ComparisonWindow],
) -> dict[str, Any]:
    """Verify that A/B differ only by the supplied voice on review windows."""
    checks: list[dict[str, Any]] = []
    changed_peak = 0.0
    overall_max = 0.0
    overall_squared = 0.0
    overall_count = 0
    for window in windows:
        mix_a, rate_a = _read_interval(
            before_mix, window.clip_start_sec, window.clip_duration_sec
        )
        mix_b, rate_b = _read_interval(
            after_mix, window.clip_start_sec, window.clip_duration_sec
        )
        voice_a, voice_rate_a = _read_interval(
            before_voice, window.clip_start_sec, window.clip_duration_sec
        )
        voice_b, voice_rate_b = _read_interval(
            after_voice, window.clip_start_sec, window.clip_duration_sec
        )
        if len({rate_a, rate_b, voice_rate_a, voice_rate_b}) != 1:
            raise ValueError("Частоты дорожек различаются внутри контрольного окна.")
        channels = mix_a.shape[1]
        if mix_b.shape != mix_a.shape:
            raise ValueError("Формы final mix A и B различаются.")
        matched_a = np.asarray(audio_io.match_channels(voice_a, channels), dtype=np.float64)
        matched_b = np.asarray(audio_io.match_channels(voice_b, channels), dtype=np.float64)
        changed_peak = max(changed_peak, float(np.max(np.abs(matched_b - matched_a))))
        residual_delta = (mix_a - matched_a) - (mix_b - matched_b)
        # Hard limiting makes the residual algebra undefined only at clipped
        # samples. The audited windows are far below the limiter, nevertheless
        # exclude those samples explicitly so the contract remains correct.
        valid = np.maximum(np.abs(mix_a), np.abs(mix_b)) < MIX_CLIP_LIMIT - 1e-5
        values = residual_delta[valid]
        if values.size < int(residual_delta.size * 0.95):
            raise ValueError("Слишком большая часть контрольного окна ограничена лимитером.")
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        squared = float(np.sum(np.square(values, dtype=np.float64)))
        rms = math.sqrt(squared / max(int(values.size), 1))
        overall_max = max(overall_max, max_abs)
        overall_squared += squared
        overall_count += int(values.size)
        checks.append(
            {
                "case_id": window.case_id,
                "background_residual_max_abs": max_abs,
                "background_residual_rms": rms,
                "validated_samples": int(values.size),
            }
        )
    overall_rms = math.sqrt(overall_squared / max(overall_count, 1))
    if changed_peak <= 1e-7:
        raise ValueError("Варианты A и B не различаются в семи контрольных окнах.")
    if overall_max > BACKGROUND_RESIDUAL_MAX_ABS or overall_rms > BACKGROUND_RESIDUAL_RMS:
        raise ValueError(
            "A и B сведены с разным фоном, усилением или задержкой: "
            f"max={overall_max:.8f}, rms={overall_rms:.8f}."
        )
    return {
        "valid": True,
        "method": "mix_minus_zero_gain_voice_residual_v1",
        "background_gain_db": 0.0,
        "voice_gain_db": 0.0,
        "voice_delay_sec": 0.0,
        "clip_limit": MIX_CLIP_LIMIT,
        "background_residual_max_abs": overall_max,
        "background_residual_rms": overall_rms,
        "voice_difference_peak": changed_peak,
        "checks": checks,
    }


def _extract_clip(source: Path, destination: Path, window: ComparisonWindow) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        audio_io.check_ffmpeg(),
        "-y",
        "-v",
        "error",
        "-ss",
        f"{window.clip_start_sec:.6f}",
        "-i",
        str(source),
        "-t",
        f"{window.clip_duration_sec:.6f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(CLIP_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Не удалось подготовить {destination.name}: {completed.stderr[-2000:]}"
        )
    info = sf.info(str(destination))
    expected_frames = int(round(window.clip_duration_sec * CLIP_SAMPLE_RATE))
    if (
        int(info.samplerate) != CLIP_SAMPLE_RATE
        or int(info.channels) != 2
        or int(info.frames) != expected_frames
        or str(info.subtype) != "PCM_16"
    ):
        raise RuntimeError(f"Некорректный выходной клип: {destination}")


def _draft_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "published": False,
        "publication_status": PUBLICATION_WAITING,
        "title": TITLE,
        "description": DESCRIPTION,
        "generated_at": "",
        "summary": {},
        "intervals": [],
    }


def mark_comparison_waiting(output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    atomic_json(manifest_path, _draft_manifest())
    return manifest_path


def build_comparison(
    *,
    before_voice: str | Path,
    before_mix: str | Path,
    after_voice: str | Path,
    after_mix: str | Path,
    output_dir: str | Path,
    windows: Sequence[ComparisonWindow] = SNATCH_WINDOWS,
    after_dependencies: Iterable[Path] | None = None,
    clip_extractor: Callable[[Path, Path, ComparisonWindow], None] = _extract_clip,
) -> dict[str, Any]:
    if len(windows) != 7:
        raise ValueError("Сравнение должно содержать ровно семь контрольных окон.")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Manifest is hidden first and becomes public last. A failed rebuild can
    # therefore never leave old clips advertised as the new result.
    mark_comparison_waiting(destination)

    voice_a = _require_audio(before_voice, "A voice")
    mix_a = _require_audio(before_mix, "A final mix")
    voice_b = _require_audio(after_voice, "B voice")
    mix_b = _require_audio(after_mix, "B final mix")
    if _same_file(voice_a, voice_b) or _same_file(mix_a, mix_b):
        raise ValueError("A и B должны ссылаться на разные файлы.")
    identities = _validate_lengths(voice_a, mix_a, voice_b, mix_b, windows)
    freshness = _validate_after_freshness(
        voice_b,
        mix_b,
        _default_after_dependencies() if after_dependencies is None else after_dependencies,
    )
    contract = _mix_contract(voice_a, mix_a, voice_b, mix_b, windows)

    with tempfile.TemporaryDirectory(prefix=".comparison-build-", dir=destination) as raw:
        staging = Path(raw)
        staged_before = staging / "before"
        staged_after = staging / "after"
        intervals: list[dict[str, Any]] = []
        improved = unchanged = review = 0
        for window in windows:
            before_name = f"{window.file_stem}_before.wav"
            after_name = f"{window.file_stem}_after.wav"
            before_clip = staged_before / before_name
            after_clip = staged_after / after_name
            clip_extractor(mix_a, before_clip, window)
            clip_extractor(mix_b, after_clip, window)

            before_db = _voice_rms_db(
                voice_a, window.event_start_sec, window.event_end_sec
            )
            after_db = _voice_rms_db(
                voice_b, window.event_start_sec, window.event_end_sec
            )
            decision, reason, restored_db = _objective_decision(before_db, after_db)
            if decision == "Речь восстановлена":
                improved += 1
            elif decision == "Без заметного изменения":
                unchanged += 1
            else:
                review += 1
            metrics: dict[str, Any] = {
                "before_dbfs": _public_db(before_db),
                "after_dbfs": _public_db(after_db),
            }
            if restored_db is not None:
                metrics["restored_db"] = restored_db
            intervals.append(
                {
                    "id": window.case_id,
                    "start_sec": window.event_start_sec,
                    "end_sec": window.event_end_sec,
                    "decision": decision,
                    "reason": reason,
                    "metrics": metrics,
                    "before_audio": f"before/{before_name}",
                    "after_audio": f"after/{after_name}",
                }
            )

        generated_at = utc_now()
        manifest = {
            "schema_version": 1,
            "published": True,
            "publication_status": PUBLICATION_READY,
            "title": TITLE,
            "description": DESCRIPTION,
            "generated_at": generated_at,
            "summary": {
                "проверено": len(windows),
                "улучшено": improved,
                "без заметного изменения": unchanged,
                "требуют проверки": review,
            },
            "intervals": intervals,
        }
        build_report = {
            "schema_version": 1,
            "comparison_id": COMPARISON_ID,
            "created_at": generated_at,
            "sources": identities,
            "after_freshness": freshness,
            "mix_contract": contract,
            "windows": [asdict(item) for item in windows],
        }

        final_before = destination / "before"
        final_after = destination / "after"
        final_before.mkdir(parents=True, exist_ok=True)
        final_after.mkdir(parents=True, exist_ok=True)
        for source in sorted(staged_before.glob("*.wav")):
            os.replace(source, final_before / source.name)
        for source in sorted(staged_after.glob("*.wav")):
            os.replace(source, final_after / source.name)
        atomic_json(destination / "build_report.json", build_report)
        # Publish last: routes cannot expose the new assets before this write.
        atomic_json(destination / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовить семь воспроизводимых A/B-фрагментов «Большого куша»."
    )
    parser.add_argument("--before-voice", required=True, type=Path)
    parser.add_argument("--before-mix", required=True, type=Path)
    parser.add_argument("--after-voice", required=True, type=Path)
    parser.add_argument("--after-mix", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parents[3]
            / "data"
            / "quality_comparisons"
            / COMPARISON_ID
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_comparison(
        before_voice=args.before_voice,
        before_mix=args.before_mix,
        after_voice=args.after_voice,
        after_mix=args.after_mix,
        output_dir=args.output_dir,
    )
    print(
        f"Готово: {len(manifest['intervals'])} A/B-фрагментов, "
        f"manifest: {args.output_dir.resolve() / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
