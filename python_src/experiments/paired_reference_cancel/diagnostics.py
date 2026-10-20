"""Metrics and plain-Russian warnings for one paired-reference-cancel run.

Nothing here decides whether a result is "good" -- these are numbers to
look at *alongside* listening to the WAVs, not a pass/fail gate. See the
project brief: "Метрики не должны объявлять результат хорошим без
прослушивания."
"""
from __future__ import annotations

import numpy as np


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    if length < 8:
        return 0.0
    a64 = a[:length].astype(np.float64) - np.mean(a[:length])
    b64 = b[:length].astype(np.float64) - np.mean(b[:length])
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a64, b64) / denom)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


def rms_db(x: np.ndarray, floor_db: float = -120.0) -> float:
    value = rms(x)
    return 20.0 * np.log10(value) if value > 1e-9 else floor_db


def clipping_ratio(x: np.ndarray, threshold: float = 0.999) -> float:
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= threshold))


def si_snr(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SNR in dB; used by tests to check that a residual
    moved measurably closer to a known synthetic RU target."""
    length = min(len(estimate), len(target))
    if length < 8:
        return -120.0
    est = estimate[:length].astype(np.float64)
    ref = target[:length].astype(np.float64)
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    ref_energy = float(np.dot(ref, ref)) + eps
    projection = (float(np.dot(est, ref)) / ref_energy) * ref
    noise = est - projection
    return float(10.0 * np.log10((float(np.dot(projection, projection)) + eps) / (float(np.dot(noise, noise)) + eps)))


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x.astype(np.float64))))


def residual_common_energy_ratio(predicted_common: np.ndarray, target: np.ndarray) -> float:
    """Fraction of the target's energy explained by the predicted common signal."""
    target_energy = energy(target)
    if target_energy < 1e-12:
        return 0.0
    return float(min(1.0, energy(predicted_common) / target_energy))


def reconstruction_error(target: np.ndarray, reconstructed: np.ndarray) -> dict:
    length = min(len(target), len(reconstructed))
    diff = target[:length].astype(np.float64) - reconstructed[:length].astype(np.float64)
    error_rms = float(np.sqrt(np.mean(np.square(diff)))) if length else 0.0
    target_rms = rms(target[:length])
    relative_db = 20.0 * np.log10(error_rms / target_rms) if error_rms > 1e-12 and target_rms > 1e-12 else -180.0
    return {"rmse": round(error_rms, 8), "relative_to_target_db": round(relative_db, 2)}


def segment_ratios(alignment_map: list[dict]) -> dict:
    if not alignment_map:
        return {
            "confident_count_ratio": 0.0,
            "confident_duration_ratio": 0.0,
            "unusable_count_ratio": 1.0,
            "unusable_duration_ratio": 1.0,
        }
    total_duration = sum(c["duration"] for c in alignment_map) or 1.0
    usable = [c for c in alignment_map if c["usable"]]
    usable_duration = sum(c["duration"] for c in usable)
    return {
        "confident_count_ratio": round(len(usable) / len(alignment_map), 4),
        "confident_duration_ratio": round(usable_duration / total_duration, 4),
        "unusable_count_ratio": round(1.0 - len(usable) / len(alignment_map), 4),
        "unusable_duration_ratio": round(1.0 - usable_duration / total_duration, 4),
    }


def delay_drift_summary(global_offset_sec: float, global_confidence: float, global_speed_ratio: float) -> dict:
    drift_ppm = (global_speed_ratio - 1.0) * 1_000_000.0
    return {
        "global_delay_ms": round(global_offset_sec * 1000.0, 2),
        "global_confidence": round(global_confidence, 4),
        "global_speed_ratio": round(global_speed_ratio, 6),
        "drift_ppm": round(drift_ppm, 1),
    }


def build_warnings(
    global_confidence: float,
    agreement_spread_sec: float,
    ratios: dict,
    global_speed_ratio: float,
    reconstruction: dict,
    clip_ratio_values: dict[str, float],
    min_confident_duration_ratio_ok: float,
    min_global_confidence_ok: float = 0.35,
    max_agreement_spread_sec_ok: float = 0.2,
    max_abs_drift_ppm_ok: float = 20000.0,
) -> list[str]:
    warnings: list[str] = []
    if global_confidence < min_global_confidence_ok:
        warnings.append(
            f"Низкая общая уверенность выравнивания ({global_confidence:.2f}). "
            "Возможно, это разные монтажные версии, разные фрагменты или один из "
            "файлов сильно повреждён/зашумлён."
        )
    if agreement_spread_sec > max_agreement_spread_sec_ok:
        warnings.append(
            f"Методы поиска задержки разошлись на {agreement_spread_sec * 1000.0:.0f} мс "
            "между собой -- глобальная задержка может быть определена неверно."
        )
    if ratios["confident_duration_ratio"] < min_confident_duration_ratio_ok:
        warnings.append(
            f"Только {ratios['confident_duration_ratio']:.0%} длительности участка "
            "выровнено с достаточной уверенностью. Остальное не обрабатывалось "
            "агрессивно и передано с минимальными изменениями."
        )
    if abs((global_speed_ratio - 1.0) * 1_000_000.0) > max_abs_drift_ppm_ok:
        warnings.append(
            f"Обнаружен значительный временной дрейф скорости ({(global_speed_ratio - 1.0) * 100:.2f}%). "
            "Проверьте, что это действительно один и тот же материал (а не PAL/NTSC "
            "рассинхрон или разные релизы)."
        )
    if reconstruction.get("relative_to_target_db", -180.0) > -20.0:
        warnings.append(
            "Реконструкция (predicted_common + residual) заметно отличается от "
            "переводной дорожки -- проверьте 5_reconstructed_check.wav на слух."
        )
    for label, value in clip_ratio_values.items():
        if value > 0.001:
            warnings.append(f"Клиппинг в {label}: {value:.2%} сэмплов на пределе шкалы.")
    if not warnings:
        warnings.append(
            "Автоматические метрики не выявили явных проблем, но результат всё "
            "равно необходимо прослушать -- метрики этого не заменяют."
        )
    return warnings
