"""Build trustworthy external controls from raw aligned speech sources."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from experiments.paired_reference_cancel.recipe_audio import read_interval_mono


def _rms_db(values: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _correlation_and_lag(
    mixture: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    max_lag_sec: float = 0.25,
) -> tuple[float, float]:
    left = mixture.astype(np.float64) - float(np.mean(mixture))
    right = reference.astype(np.float64) - float(np.mean(reference))
    denominator = math.sqrt(
        float(np.sum(left * left)) * float(np.sum(right * right))
    ) + 1e-12
    correlation = signal.correlate(left, right, mode="full", method="fft")
    lags = signal.correlation_lags(len(left), len(right), mode="full")
    maximum = int(round(max_lag_sec * sample_rate))
    allowed = np.abs(lags) <= maximum
    window = np.abs(correlation[allowed])
    if not window.size:
        return 0.0, 0.0
    index = int(np.argmax(window))
    return (
        float(window[index] / denominator),
        float(lags[allowed][index] / sample_rate),
    )


def _write_flac(path: Path, values: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(values, -1.0, 1.0),
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_real_evaluation(
    store,
    external_project_id: str,
    dataset_root: Path,
    sample_rate: int = 24_000,
    duration_sec: float = 8.0,
    scenes_per_film: int = 5,
    scan_stride_sec: float = 10.0,
) -> list[dict[str, Any]]:
    """Create controls from raw EN speech mapped to the dubbed speech timeline.

    The previous implementation reused the output of an old adaptive
    subtractor as the EN reference.  That output can be silent precisely when
    subtraction failed.  Here the reference is always cut from the original
    speech stem and mapped with the accepted alignment map.
    """

    score_rate = 6_000
    score_frames = int(round(duration_sec * score_rate))
    output_frames = int(round(duration_sec * sample_rate))
    controls: list[dict[str, Any]] = []
    for pair in store.list_pairs(external_project_id):
        if pair.get("deleted"):
            continue
        pair_root = store.pair_dir(external_project_id, pair["id"])
        mixture_path = pair_root / "stems" / "dubbed" / "speech_en_ru.flac"
        reference_path = pair_root / "stems" / "original" / "en_speech.flac"
        alignment_path = pair_root / "alignment" / "alignment_map.json"
        if not (
            mixture_path.is_file()
            and reference_path.is_file()
            and alignment_path.is_file()
        ):
            continue
        alignment = _load_json(alignment_path)
        correction = float(alignment.get("manual_correction_sec") or 0.0)
        candidates: list[dict[str, Any]] = []
        for segment in alignment.get("segments") or []:
            if not segment.get("usable"):
                continue
            dubbed_start = float(segment["dubbed_start"])
            dubbed_end = float(segment["dubbed_end"])
            speed = float(segment.get("speed_ratio") or 1.0)
            position = dubbed_start
            while position + duration_sec <= dubbed_end + 1e-6:
                original_start = float(segment["original_start"]) + (
                    position - dubbed_start
                ) * speed + correction
                mixture = read_interval_mono(
                    mixture_path,
                    position,
                    duration_sec,
                    score_rate,
                    score_frames,
                )
                reference = read_interval_mono(
                    reference_path,
                    original_start,
                    duration_sec * speed,
                    score_rate,
                    score_frames,
                )
                mixture_rms = _rms_db(mixture)
                reference_rms = _rms_db(reference)
                if mixture_rms > -50.0 and reference_rms > -50.0:
                    correlation, lag_sec = _correlation_and_lag(
                        mixture, reference, score_rate
                    )
                    candidates.append(
                        {
                            "dubbed_start_sec": position,
                            "original_start_sec": original_start,
                            "original_duration_sec": duration_sec * speed,
                            "alignment_confidence": float(
                                segment.get("confidence") or 0.0
                            ),
                            "mixture_rms_db": mixture_rms,
                            "reference_rms_db": reference_rms,
                            "reference_correlation": correlation,
                            "estimated_lag_sec": lag_sec,
                        }
                    )
                position += scan_stride_sec
        selected: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item["reference_correlation"],
                item["reference_rms_db"],
            ),
            reverse=True,
        ):
            if candidate["reference_correlation"] < 0.10:
                continue
            if any(
                abs(
                    candidate["dubbed_start_sec"]
                    - existing["dubbed_start_sec"]
                )
                < 90.0
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= scenes_per_film:
                break
        if len(selected) < scenes_per_film:
            for candidate in sorted(
                candidates,
                key=lambda item: (
                    item["reference_correlation"],
                    item["reference_rms_db"],
                ),
                reverse=True,
            ):
                if candidate in selected:
                    continue
                if any(
                    abs(
                        candidate["dubbed_start_sec"]
                        - existing["dubbed_start_sec"]
                    )
                    < 45.0
                    for existing in selected
                ):
                    continue
                selected.append(candidate)
                if len(selected) >= scenes_per_film:
                    break
        for index, candidate in enumerate(
            sorted(selected, key=lambda item: item["dubbed_start_sec"]), start=1
        ):
            mixture = read_interval_mono(
                mixture_path,
                candidate["dubbed_start_sec"],
                duration_sec,
                sample_rate,
                output_frames,
            )
            reference = read_interval_mono(
                reference_path,
                candidate["original_start_sec"],
                candidate["original_duration_sec"],
                sample_rate,
                output_frames,
            )
            reference_peak = float(np.max(np.abs(reference)))
            if reference_peak > 1e-7:
                reference = reference * (0.98 / max(reference_peak, 0.98))
            scene_id = f"scene_{index:02d}"
            destination = (
                dataset_root
                / "real_evaluation_v2"
                / pair["id"]
                / scene_id
            )
            controls.append(
                {
                    "id": f"{pair['id'][:8]}_{scene_id}",
                    "pair_id": pair["id"],
                    "film_name": pair.get("name") or pair["id"],
                    "dubbed_start_sec": round(
                        float(candidate["dubbed_start_sec"]), 6
                    ),
                    "original_start_sec": round(
                        float(candidate["original_start_sec"]), 6
                    ),
                    "duration_sec": duration_sec,
                    "source_kind": "raw_original_speech_mapped_v2",
                    "alignment_confidence": candidate[
                        "alignment_confidence"
                    ],
                    "reference_correlation": candidate[
                        "reference_correlation"
                    ],
                    "estimated_lag_sec": candidate["estimated_lag_sec"],
                    "mixture_rms_db": candidate["mixture_rms_db"],
                    "reference_rms_db": candidate["reference_rms_db"],
                    "files": {
                        "mixture": _write_flac(
                            destination / "mixture.flac",
                            mixture,
                            sample_rate,
                        ),
                        "reference": _write_flac(
                            destination / "reference.flac",
                            reference,
                            sample_rate,
                        ),
                    },
                }
            )
    return controls
