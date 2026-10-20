"""STFT-based adaptive reference cancellation.

Given a ``predictor`` signal (already time-aligned onto the ``target``'s
timeline by :mod:`alignment`) this estimates a regularized, time-varying,
per-frequency complex transfer function ``H(t, f)`` that maps the predictor
into whatever fraction of the target it can explain, then subtracts that
prediction from the target.

``H`` is estimated once from a mono downmix of both signals (a single
shared filter), then applied identically to every channel of the predictor
before subtracting from each channel of the target. This is a deliberate
simplification for multichannel/5.1 material -- true per-channel adaptive
filtering (e.g. independent center/surrounds) is out of scope for this
proof of concept; see README "Ограничения".

Everything downstream of ``H`` is gated twice:

- per (freq, frame) by the magnitude-squared coherence between predictor
  and target in a local block (low coherence => this bin is not reliably
  "common" => H is faded toward 0, i.e. bypass/no cancellation there);
- per frame by the alignment confidence curve from :mod:`alignment` (an
  unusable/low-confidence time span is never aggressively subtracted).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.audio_io import ensure_2d, match_channels, to_mono  # noqa: E402

EPS = 1e-10


@dataclass
class CancelResult:
    predicted_common: np.ndarray  # (n_samples, target_channels)
    residual: np.ndarray          # target - predicted_common, same shape as target
    mean_coherence: float
    gated_fraction: float         # activity-weighted average suppression applied by the confidence gate
    clipping_ratio: float
    bypassed: bool                # True if the input was too short/silent to filter at all


def _fit_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    if x.ndim == 1:
        pad = np.zeros(target_len - len(x), dtype=x.dtype)
    else:
        pad = np.zeros((target_len - len(x), x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def _block_average(values: np.ndarray, block_frames: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    size = max(1, int(block_frames))
    return uniform_filter1d(values, size=size, axis=-1, mode="nearest")


def estimate_transfer_function(
    predictor_mono: np.ndarray,
    target_mono: np.ndarray,
    sr: int,
    n_fft: int,
    hop: int,
    block_frames: int,
    gain_smoothing: float,
    regularization: float,
    max_gain_linear: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(H, coherence, activity_weight, frame_center_samples)``.

    ``activity_weight`` prevents thousands of effectively silent STFT bins
    from dominating reader-facing coherence/gating diagnostics.
    """
    from scipy.signal import stft, lfilter

    noverlap = n_fft - hop
    _, _, X = stft(predictor_mono, fs=sr, window="hann", nperseg=n_fft, noverlap=noverlap)
    _, _, Y = stft(target_mono, fs=sr, window="hann", nperseg=n_fft, noverlap=noverlap)
    n_frames = min(X.shape[1], Y.shape[1])
    X, Y = X[:, :n_frames], Y[:, :n_frames]

    cross = np.conj(X) * Y
    pxx = (X.real**2 + X.imag**2)
    pyy = (Y.real**2 + Y.imag**2)

    cross_avg = _block_average(cross.real, block_frames) + 1j * _block_average(cross.imag, block_frames)
    pxx_avg = _block_average(pxx, block_frames)
    pyy_avg = _block_average(pyy, block_frames)

    with np.errstate(invalid="ignore", divide="ignore"):
        h_block = cross_avg / (pxx_avg + regularization)
        coherence = (np.abs(cross_avg) ** 2) / (pxx_avg * pyy_avg + EPS)
    h_block = np.nan_to_num(h_block, nan=0.0, posinf=0.0, neginf=0.0)
    coherence = np.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0)
    coherence = np.clip(coherence, 0.0, 1.0)
    activity_weight = np.sqrt(np.maximum(pxx_avg, 0.0) * np.maximum(pyy_avg, 0.0))
    activity_weight = np.nan_to_num(activity_weight, nan=0.0, posinf=0.0, neginf=0.0)

    alpha = float(np.clip(gain_smoothing, 0.0, 0.999))
    h_smooth = lfilter([1.0 - alpha], [1.0, -alpha], h_block, axis=1)

    magnitude = np.abs(h_smooth)
    scale = np.minimum(1.0, max_gain_linear / np.maximum(magnitude, EPS))
    h_limited = h_smooth * scale
    h_limited = np.nan_to_num(h_limited, nan=0.0, posinf=0.0, neginf=0.0)

    frame_centers = np.arange(n_frames) * hop
    return h_limited, coherence, activity_weight, frame_centers


def _apply_gate(
    h: np.ndarray,
    coherence: np.ndarray,
    activity_weight: np.ndarray,
    frame_centers: np.ndarray,
    confidence_curve: np.ndarray,
    min_coherence: float,
) -> tuple[np.ndarray, float]:
    coherence_gate = np.clip(
        (coherence - min_coherence) / max(1e-6, 1.0 - min_coherence), 0.0, 1.0
    )
    if confidence_curve.size:
        indices = np.clip(frame_centers, 0, confidence_curve.size - 1)
        frame_confidence = confidence_curve[indices]
    else:
        frame_confidence = np.ones(len(frame_centers), dtype=np.float32)
    gate = coherence_gate * frame_confidence[None, :]
    total_activity = float(np.sum(activity_weight))
    if total_activity > EPS:
        gated_fraction = float(
            np.sum(activity_weight * (1.0 - gate)) / total_activity
        )
    else:
        gated_fraction = 1.0
    return h * gate, gated_fraction


def segmented_adaptive_cancel(
    predictor: np.ndarray,
    target: np.ndarray,
    sr: int,
    confidence_curve: np.ndarray | None,
    stft_n_fft: int = 2048,
    stft_hop: int = 512,
    block_frames: int = 24,
    gain_smoothing: float = 0.65,
    min_coherence: float = 0.30,
    regularization: float = 1e-6,
    max_gain_linear: float = 8.0,
) -> CancelResult:
    """Cancel the predictor's estimated contribution out of the target.

    ``predictor`` and ``target`` may have different lengths/channel counts;
    both are trimmed/padded to the target's length and the predictor is
    matched to the target's channel count before subtraction.
    """
    from scipy.signal import istft

    target = ensure_2d(
        np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    )
    predictor = ensure_2d(
        np.nan_to_num(np.asarray(predictor, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    )
    n_samples = target.shape[0]
    target_channels = target.shape[1]

    predictor = _fit_length(predictor, n_samples)
    predictor_matched = match_channels(predictor, target_channels)

    if confidence_curve is None:
        confidence_curve = np.ones(n_samples, dtype=np.float32)
    else:
        confidence_curve = _fit_length(
            np.nan_to_num(
                np.asarray(confidence_curve, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            n_samples,
        )

    min_len_required = stft_n_fft
    if n_samples < min_len_required or np.max(np.abs(predictor)) < 1e-9:
        return CancelResult(
            predicted_common=np.zeros_like(target),
            residual=target.copy(),
            mean_coherence=0.0,
            gated_fraction=1.0,
            clipping_ratio=0.0,
            bypassed=True,
        )

    predictor_mono = to_mono(predictor)
    target_mono = to_mono(target)
    h, coherence, activity_weight, frame_centers = estimate_transfer_function(
        predictor_mono, target_mono, sr, stft_n_fft, stft_hop, block_frames,
        gain_smoothing, regularization, max_gain_linear,
    )
    h_gated, gated_fraction = _apply_gate(
        h,
        coherence,
        activity_weight,
        frame_centers,
        confidence_curve,
        min_coherence,
    )

    noverlap = stft_n_fft - stft_hop
    predicted_common = np.zeros((n_samples, target_channels), dtype=np.float32)
    for ch in range(target_channels):
        from scipy.signal import stft as _stft

        _, _, Xc = _stft(predictor_matched[:, ch], fs=sr, window="hann", nperseg=stft_n_fft, noverlap=noverlap)
        n_frames = min(Xc.shape[1], h_gated.shape[1])
        predicted_freq = Xc[:, :n_frames] * h_gated[:, :n_frames]
        _, predicted_time = istft(predicted_freq, fs=sr, window="hann", nperseg=stft_n_fft, noverlap=noverlap)
        predicted_common[:, ch] = _fit_length(np.nan_to_num(predicted_time), n_samples).astype(np.float32)

    residual = target - predicted_common
    clip_ratio = float(np.mean(np.abs(residual) >= 0.999)) if residual.size else 0.0

    total_activity = float(np.sum(activity_weight))
    mean_coherence = (
        float(np.sum(coherence * activity_weight) / total_activity)
        if total_activity > EPS
        else 0.0
    )
    return CancelResult(
        predicted_common=predicted_common,
        residual=residual.astype(np.float32),
        mean_coherence=mean_coherence,
        gated_fraction=gated_fraction,
        clipping_ratio=clip_ratio,
        bypassed=False,
    )
