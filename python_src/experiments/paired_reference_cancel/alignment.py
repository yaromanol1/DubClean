"""Two-level alignment between an original track and a dubbed/mixed track.

Sign convention used everywhere in this module: a scalar ``delay`` (in
samples or seconds) satisfies::

    original_index = dubbed_index - delay

i.e. to find the original-track sample that corresponds to a given
dubbed-track sample, subtract ``delay``. Equivalently, ``aligned[t] =
original[t - delay]`` is the version of the original track resampled onto
the dubbed track's timeline. This convention is verified directly by
``tests/test_alignment.py`` (a dubbed track built as ``K`` samples of
silence followed by the original recovers ``delay == K``).

Level 1 (global): a single delay/confidence estimate for the whole clip,
combining GCC-PHAT, plain normalized cross-correlation, and a coarse
energy-envelope correlation -- three independent, complementary cues, per
the project brief's "do not rely on raw waveform correlation alone".

Level 2 (chunked): the clip is cut into a grid of chunks; each chunk gets
its own local delay/confidence via the same two-stage search (PHAT to
locate the lag, plain normalized correlation at that lag to score it).
Consecutive usable chunks give a local speed_ratio; chunks whose offset
does not fit the fitted drift line are flagged as edit cuts / mismatched
scenes and marked unusable.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-10


@dataclass
class GlobalAlignment:
    offset_samples: int
    offset_sec: float
    confidence: float
    method_offsets_sec: dict = field(default_factory=dict)
    method_scores: dict = field(default_factory=dict)
    agreement_spread_sec: float = 0.0


def _fft_xcorr(sig: np.ndarray, ref: np.ndarray, max_shift: int, phat: bool) -> tuple[int, float]:
    """Cross-correlate ``sig`` against ``ref``; return (shift, raw_peak).

    ``shift`` maximizes correlation between ``sig[t]`` and ``ref[t - shift]``
    (so ``shift`` is a "delay of sig relative to ref", matching the module's
    ``delay`` convention when sig=dubbed, ref=original).
    """
    from scipy.fft import next_fast_len, rfft, irfft

    if sig.size == 0 or ref.size == 0:
        return 0, 0.0
    n = next_fast_len(sig.size + ref.size)
    SIG = rfft(sig, n=n)
    REF = rfft(ref, n=n)
    cross = SIG * np.conj(REF)
    if phat:
        cross = cross / (np.abs(cross) + EPS)
    cc = irfft(cross, n=n)
    max_shift = int(min(max_shift, n // 2))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1])) if max_shift > 0 else cc[:1]
    peak_index = int(np.argmax(np.abs(cc)))
    shift = peak_index - max_shift
    return shift, float(cc[peak_index])


def _fft_xcorr_range(
    sig: np.ndarray,
    ref: np.ndarray,
    min_shift: int,
    max_shift: int,
    phat: bool,
) -> tuple[int, float]:
    """Cross-correlate inside an explicit lag interval.

    A local search is centred around the expected lag, which is not always
    zero when the reference window has context before the expected match.
    Searching a symmetric interval around zero silently doubles the requested
    radius and is especially unreliable near the beginning of a file.
    """
    from scipy.fft import next_fast_len, rfft, irfft

    if sig.size == 0 or ref.size == 0:
        return 0, 0.0
    min_shift, max_shift = int(min_shift), int(max_shift)
    if min_shift > max_shift:
        min_shift, max_shift = max_shift, min_shift
    n = next_fast_len(sig.size + ref.size)
    SIG = rfft(sig, n=n)
    REF = rfft(ref, n=n)
    cross = SIG * np.conj(REF)
    if phat:
        cross = cross / (np.abs(cross) + EPS)
    cc = irfft(cross, n=n)
    shifts = np.arange(min_shift, max_shift + 1, dtype=np.int64)
    indices = np.where(shifts >= 0, shifts, n + shifts)
    valid = (indices >= 0) & (indices < n)
    shifts, indices = shifts[valid], indices[valid]
    if not len(shifts):
        return 0, 0.0
    values = cc[indices]
    peak_index = int(np.argmax(np.abs(values)))
    return int(shifts[peak_index]), float(values[peak_index])


def _normalized_score_at(sig: np.ndarray, ref: np.ndarray, shift: int) -> float:
    """Pearson-style normalized correlation of sig[t] vs ref[t-shift] in [-1, 1]."""
    if shift >= 0:
        a = sig[shift:]
        b = ref[: len(a)]
    else:
        b = ref[-shift:]
        a = sig[: len(b)]
    length = min(len(a), len(b))
    if length < 8:
        return 0.0
    a = a[:length].astype(np.float64)
    b = b[:length].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom < EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def gcc_phat(sig: np.ndarray, ref: np.ndarray, sr: int, max_offset_sec: float) -> tuple[float, float]:
    """PHAT-weighted delay search. Returns (delay_sec, confidence_0_1).

    Confidence is the plain normalized correlation coefficient evaluated at
    the PHAT-located lag (PHAT sharpens the peak for *localization* but its
    raw magnitude is not a bounded, comparable score).
    """
    max_shift = int(max_offset_sec * sr)
    shift, _ = _fft_xcorr(sig, ref, max_shift, phat=True)
    score = abs(_normalized_score_at(sig, ref, shift))
    return shift / sr, score


def normalized_xcorr(sig: np.ndarray, ref: np.ndarray, sr: int, max_offset_sec: float) -> tuple[float, float]:
    """Plain (non-PHAT) normalized cross-correlation delay search."""
    max_shift = int(max_offset_sec * sr)
    shift, _ = _fft_xcorr(sig, ref, max_shift, phat=False)
    score = abs(_normalized_score_at(sig, ref, shift))
    return shift / sr, score


def energy_envelope(mono: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    n_frames = max(1, len(mono) // frame_len)
    trimmed = mono[: n_frames * frame_len]
    if trimmed.size == 0:
        return np.zeros(1, dtype=np.float64)
    frames = trimmed.reshape(n_frames, frame_len).astype(np.float64)
    return np.sqrt(np.mean(frames**2, axis=1) + EPS)


def envelope_correlation(
    sig: np.ndarray, ref: np.ndarray, sr: int, max_offset_sec: float, frame_ms: float = 20.0
) -> tuple[float, float]:
    """Delay search on coarse loudness envelopes; robust to EQ/codec/timbre
    differences since it only tracks the loudness contour."""
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    env_sig = np.log1p(energy_envelope(sig, sr, frame_ms))
    env_ref = np.log1p(energy_envelope(ref, sr, frame_ms))
    max_shift_frames = max(1, int(max_offset_sec * 1000.0 / frame_ms))
    shift_frames, _ = _fft_xcorr(env_sig, env_ref, max_shift_frames, phat=False)
    score = abs(_normalized_score_at(env_sig, env_ref, shift_frames))
    return shift_frames * frame_len / sr, score


def _combine_method_offsets(
    offsets: dict[str, float],
    scores: dict[str, float],
    tolerance_sec: float = 0.15,
) -> tuple[float, float, float]:
    """Pick the offset supported by the largest mutually-agreeing cluster.

    Agreement must be checked between every pair of methods, not only against
    one designated "reference" method: when e.g. GCC-PHAT is the outlier while
    plain correlation and the energy envelope agree with each other, the
    agreeing pair wins.  Returns ``(offset_sec, spread_sec, agreement_bonus)``.
    """
    names = list(offsets)
    clusters = [
        [name for name in names if abs(offsets[name] - offsets[anchor]) <= tolerance_sec]
        for anchor in names
    ]
    best_cluster = max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            float(np.mean([scores[name] for name in cluster])),
        ),
    )
    if len(best_cluster) >= 2:
        chosen_offsets = [offsets[name] for name in best_cluster]
        return (
            float(np.median(chosen_offsets)),
            float(max(chosen_offsets) - min(chosen_offsets)),
            1.0,
        )
    # No agreement anywhere: fall back to the highest-scoring single method,
    # but confidence must reflect the disagreement.
    best = max(scores, key=lambda name: scores[name])
    return (
        offsets[best],
        float(max(offsets.values()) - min(offsets.values())),
        0.4,
    )


def estimate_global_offset(
    original_mono: np.ndarray,
    dubbed_mono: np.ndarray,
    sr: int,
    max_offset_sec: float,
) -> GlobalAlignment:
    """Combine three independent delay cues into one robust global estimate."""
    methods = {
        "gcc_phat": gcc_phat(dubbed_mono, original_mono, sr, max_offset_sec),
        "normalized_xcorr": normalized_xcorr(dubbed_mono, original_mono, sr, max_offset_sec),
        "energy_envelope": envelope_correlation(dubbed_mono, original_mono, sr, max_offset_sec),
    }
    offsets = {name: value[0] for name, value in methods.items()}
    scores = {name: value[1] for name, value in methods.items()}

    chosen, spread, agreement_bonus = _combine_method_offsets(offsets, scores)

    base_confidence = float(np.mean(list(scores.values())))
    confidence = max(0.0, min(1.0, base_confidence * agreement_bonus))
    return GlobalAlignment(
        offset_samples=int(round(chosen * sr)),
        offset_sec=chosen,
        confidence=confidence,
        method_offsets_sec={k: round(v, 4) for k, v in offsets.items()},
        method_scores={k: round(v, 4) for k, v in scores.items()},
        agreement_spread_sec=round(spread, 4),
    )


def build_chunk_grid(total_dubbed_sec: float, chunk_seconds: float) -> list[tuple[float, float]]:
    if total_dubbed_sec <= 0:
        return []
    grid = []
    t = 0.0
    while t < total_dubbed_sec - EPS:
        duration = min(chunk_seconds, total_dubbed_sec - t)
        if duration < chunk_seconds * 0.5 and grid:
            prev_start, prev_dur = grid[-1]
            grid[-1] = (prev_start, prev_dur + duration)
        else:
            grid.append((t, duration))
        t += chunk_seconds
    return grid


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


def estimate_chunk_offsets(
    original_mono: np.ndarray,
    dubbed_mono: np.ndarray,
    sr: int,
    global_delay_sec: float,
    chunk_seconds: float,
    search_radius_sec: float,
    silence_rms: float = 1e-4,
) -> list[dict]:
    """Local delay/confidence per chunk of the dubbed timeline."""
    total_dubbed_sec = len(dubbed_mono) / sr
    grid = build_chunk_grid(total_dubbed_sec, chunk_seconds)
    results = []
    for dubbed_start, duration in grid:
        d0 = int(round(dubbed_start * sr))
        d1 = int(round((dubbed_start + duration) * sr))
        dub_excerpt = dubbed_mono[d0:d1]

        expected_orig_start_sec = dubbed_start - global_delay_sec
        window_start_sec = expected_orig_start_sec - search_radius_sec
        w0 = int(round(window_start_sec * sr))
        w1 = int(round((expected_orig_start_sec + duration + search_radius_sec) * sr))
        w0_clamped = max(0, w0)
        w1_clamped = min(len(original_mono), w1)
        ref_window = original_mono[w0_clamped:w1_clamped]

        entry = {
            "dubbed_start": round(dubbed_start, 4),
            "duration": round(duration, 4),
        }
        if _rms(dub_excerpt) < silence_rms or _rms(ref_window) < silence_rms:
            entry.update(
                original_start=round(expected_orig_start_sec, 4),
                offset=round(dubbed_start - expected_orig_start_sec, 4),
                speed_ratio=1.0,
                correlation=0.0,
                confidence=0.0,
                usable=False,
                reason="тишина или отсутствие сигнала в этом участке",
            )
            results.append(entry)
            continue

        # The expected lag depends on where the clamped reference window
        # begins. Search exactly ±radius around that lag.
        search_radius_samples = max(1, int(round(search_radius_sec * sr)))
        expected_orig_start_sample = int(round(expected_orig_start_sec * sr))
        expected_shift = w0_clamped - expected_orig_start_sample
        local_shift, _ = _fft_xcorr_range(
            dub_excerpt,
            ref_window,
            expected_shift - search_radius_samples,
            expected_shift + search_radius_samples,
            phat=True,
        )
        correlation = abs(_normalized_score_at(dub_excerpt, ref_window, local_shift))
        # local_shift: dub_excerpt[t] ~= ref_window[t - local_shift]
        # ref_window[k] == original_mono[w0_clamped + k], so:
        # original_index(dubbed_index=d0+t) = w0_clamped - local_shift + t
        original_start_sample = w0_clamped - local_shift
        original_start_sec = original_start_sample / sr
        offset_sec = dubbed_start - original_start_sec
        entry.update(
            original_start=round(original_start_sec, 4),
            offset=round(offset_sec, 4),
            speed_ratio=1.0,  # filled in by estimate_drift_and_flag()
            correlation=round(correlation, 4),
            confidence=round(correlation, 4),
            usable=True,
            reason=None,
        )
        results.append(entry)
    return results


def _robust_linear_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Least-squares fit y = a + b*x with one round of outlier trimming."""
    if len(xs) == 0:
        return 0.0, 1.0
    if len(xs) == 1:
        return float(ys[0] - xs[0]), 1.0
    coeffs = np.polyfit(xs, ys, 1)
    b, a = float(coeffs[0]), float(coeffs[1])
    if len(xs) >= 5:
        residuals = np.abs(ys - (a + b * xs))
        keep = residuals <= max(np.median(residuals) * 3.0, 1e-6)
        if keep.sum() >= 3 and keep.sum() < len(xs):
            coeffs = np.polyfit(xs[keep], ys[keep], 1)
            b, a = float(coeffs[0]), float(coeffs[1])
    return a, b


def estimate_drift_and_flag(
    chunks: list[dict],
    min_confidence: float,
    discontinuity_threshold_ms: float,
) -> tuple[list[dict], float]:
    """Fit the drift line, tag discontinuities, fill per-chunk speed_ratio.

    Returns (updated_chunks, global_speed_ratio).
    """
    usable_idx = [i for i, c in enumerate(chunks) if c["usable"] and c["confidence"] >= min_confidence]
    if len(usable_idx) < 2:
        for c in chunks:
            if c["usable"] and c["confidence"] < min_confidence:
                c["usable"] = False
                c["reason"] = c["reason"] or "уверенность ниже порога"
        return chunks, 1.0

    xs = np.array([chunks[i]["dubbed_start"] for i in usable_idx], dtype=np.float64)
    ys = np.array([chunks[i]["original_start"] for i in usable_idx], dtype=np.float64)
    intercept, slope = _robust_linear_fit(xs, ys)

    threshold_sec = discontinuity_threshold_ms / 1000.0
    prev_usable: dict | None = None
    for c in chunks:
        if not c["usable"]:
            continue
        if c["confidence"] < min_confidence:
            c["usable"] = False
            c["reason"] = c["reason"] or "уверенность ниже порога"
            continue
        predicted = intercept + slope * c["dubbed_start"]
        deviation_sec = abs(c["original_start"] - predicted)
        if deviation_sec * 1000.0 > threshold_sec * 1000.0:
            c["usable"] = False
            c["reason"] = (
                f"отклонение от тренда выравнивания {deviation_sec * 1000.0:.0f} мс "
                f"(> {discontinuity_threshold_ms:.0f} мс) — вероятно монтажная вставка"
            )
            continue
        if prev_usable is not None:
            d_dub = c["dubbed_start"] - prev_usable["dubbed_start"]
            d_orig = c["original_start"] - prev_usable["original_start"]
            c["speed_ratio"] = round(d_orig / d_dub, 6) if abs(d_dub) > EPS else 1.0
        else:
            c["speed_ratio"] = round(slope, 6)
        prev_usable = c
    return chunks, float(slope)


def build_alignment_map(
    original_mono: np.ndarray,
    dubbed_mono: np.ndarray,
    sr: int,
    global_alignment: GlobalAlignment,
    chunk_seconds: float,
    search_radius_sec: float,
    min_confidence: float,
    discontinuity_threshold_ms: float,
) -> tuple[list[dict], float]:
    chunks = estimate_chunk_offsets(
        original_mono, dubbed_mono, sr, global_alignment.offset_sec, chunk_seconds, search_radius_sec
    )
    chunks, global_speed_ratio = estimate_drift_and_flag(chunks, min_confidence, discontinuity_threshold_ms)
    return chunks, global_speed_ratio


def _resolve_segment(chunk: dict, intercept: float, slope: float) -> tuple[float, float]:
    if chunk["usable"]:
        return chunk["original_start"], chunk.get("speed_ratio", slope) or slope
    predicted_start = intercept + slope * chunk["dubbed_start"]
    return predicted_start, slope


def warp_original_to_dubbed_timeline(
    original: np.ndarray,
    alignment_map: list[dict],
    sr: int,
    total_len: int,
    global_offset_sec: float,
    crossfade_sec: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample/shift ``original`` onto the dubbed track's sample timeline.

    Returns ``(aligned, confidence_curve)``. ``aligned`` has the same
    channel count as ``original`` and exactly ``total_len`` samples.
    ``confidence_curve`` is a per-sample value in [0, 1] usable to gate the
    adaptive canceller (low where alignment is unusable/uncertain).
    """
    original = original if original.ndim == 2 else original[:, None]
    n_channels = original.shape[1]
    out = np.zeros((total_len, n_channels), dtype=np.float64)
    weight = np.zeros(total_len, dtype=np.float64)
    conf_acc = np.zeros(total_len, dtype=np.float64)

    if not alignment_map:
        # No map at all (e.g. clip too short to chunk): fall back to a
        # single flat shift by the global offset.
        alignment_map = [
            {
                "dubbed_start": 0.0,
                "duration": total_len / sr,
                "original_start": -global_offset_sec,
                "speed_ratio": 1.0,
                "confidence": 0.0,
                "usable": False,
            }
        ]

    usable = [c for c in alignment_map if c["usable"]]
    if usable:
        xs = np.array([c["dubbed_start"] for c in usable])
        ys = np.array([c["original_start"] for c in usable])
        intercept, slope = _robust_linear_fit(xs, ys)
    else:
        intercept, slope = -global_offset_sec, 1.0

    margin = max(0, int(round(crossfade_sec * sr / 2)))

    for chunk in alignment_map:
        dub_start_sample = int(round(chunk["dubbed_start"] * sr))
        dub_len_sample = int(round(chunk["duration"] * sr))
        seg_out_start = max(0, dub_start_sample - margin)
        seg_out_end = min(total_len, dub_start_sample + dub_len_sample + margin)
        out_len = seg_out_end - seg_out_start
        if out_len <= 0:
            continue

        orig_start_sec, speed_ratio = _resolve_segment(chunk, intercept, slope)
        speed_ratio = speed_ratio if speed_ratio and math.isfinite(speed_ratio) and speed_ratio > 0 else 1.0
        pre_margin_sec = (seg_out_start - dub_start_sample) / sr  # <= 0
        orig_excerpt_start_sample = (orig_start_sec + pre_margin_sec * speed_ratio) * sr
        source_positions = orig_excerpt_start_sample + np.arange(out_len, dtype=np.float64) * speed_ratio
        left = np.floor(source_positions).astype(np.int64)
        fraction = source_positions - left
        resampled = np.zeros((out_len, n_channels), dtype=np.float64)
        valid = (left >= 0) & (left < len(original))
        if np.any(valid):
            valid_indices = np.flatnonzero(valid)
            left_valid = left[valid]
            right_valid = np.minimum(left_valid + 1, len(original) - 1)
            frac_valid = fraction[valid, None]
            resampled[valid_indices] = (
                original[left_valid] * (1.0 - frac_valid)
                + original[right_valid] * frac_valid
            )

        # Trapezoidal fade window: ramps only at the true edges of a
        # segment's overlap region, flat 1.0 across the core.
        window = np.ones(out_len, dtype=np.float64)
        fade_in = min(margin, dub_start_sample - seg_out_start) if seg_out_start < dub_start_sample else 0
        fade_out_avail = seg_out_end - (dub_start_sample + dub_len_sample)
        fade_out = min(margin, fade_out_avail) if fade_out_avail > 0 else 0
        if fade_in > 0:
            window[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False)
        if fade_out > 0:
            window[-fade_out:] = np.linspace(1.0, 0.0, fade_out, endpoint=False)

        out[seg_out_start:seg_out_end] += resampled[:out_len] * window[:, None]
        weight[seg_out_start:seg_out_end] += window
        chunk_confidence = float(chunk.get("confidence", 0.0)) if chunk.get("usable") else 0.0
        conf_acc[seg_out_start:seg_out_end] += chunk_confidence * window

    safe_weight = np.maximum(weight, EPS)
    aligned = (out / safe_weight[:, None]).astype(np.float32)
    confidence_curve = (conf_acc / safe_weight).astype(np.float32)
    confidence_curve = np.clip(confidence_curve, 0.0, 1.0)
    return aligned, confidence_curve


def plateau_segments_from_offsets(
    entries: list[dict],
    *,
    tolerance_sec: float,
    min_segment_sec: float,
    min_windows: int,
) -> list[dict]:
    """Group per-window delays into piecewise-constant plateaus.

    A dubbed track that was assembled from another source usually holds a
    constant delay for a long stretch and then jumps at a splice.  Fitting one
    line through such a profile hides both facts: the median looks fine and the
    local error stays at the size of the jump.  Grouping keeps every plateau
    exactly where it is and puts the boundary between two of them.

    A single deviating window is measurement noise, not a splice, so a new
    plateau opens only after ``min_windows`` consecutive windows agree with
    each other and disagree with the plateau in progress.  Entries must be
    sorted by ``dubbed_start`` and carry ``delay`` in seconds
    (``original_index = dubbed_index - delay``).
    """
    usable = [dict(item) for item in entries if item.get("usable", True)]
    if not usable:
        return []
    tolerance = max(0.0, float(tolerance_sec))
    wanted = max(1, int(min_windows))

    def _median(group: list[dict]) -> float:
        return float(np.median([float(item["delay"]) for item in group]))

    groups: list[list[dict]] = [[usable[0]]]
    pending: list[dict] = []
    for entry in usable[1:]:
        delay = float(entry["delay"])
        if abs(delay - _median(groups[-1])) <= tolerance:
            # The plateau reasserted itself, so the deviating run was noise.
            groups[-1].extend(pending)
            pending = []
            groups[-1].append(entry)
            continue
        if pending and abs(delay - _median(pending)) > tolerance:
            # The deviations disagree with each other too: nothing here is a
            # plateau, so give them back and start watching this entry.
            groups[-1].extend(pending)
            pending = []
        pending.append(entry)
        if len(pending) >= wanted:
            groups.append(pending)
            pending = []
    if pending:
        groups[-1].extend(pending)

    minimum = max(0.0, float(min_segment_sec))
    while len(groups) > 1:
        spans = [
            float(group[-1]["dubbed_start"]) + float(group[-1].get("duration") or 0.0)
            - float(group[0]["dubbed_start"])
            for group in groups
        ]
        short = [
            index
            for index, group in enumerate(groups)
            if spans[index] < minimum or len(group) < wanted
        ]
        if not short:
            break
        index = min(short, key=lambda position: (spans[position], len(groups[position])))
        neighbours = [position for position in (index - 1, index + 1) if 0 <= position < len(groups)]
        target = min(
            neighbours,
            key=lambda position: abs(_median(groups[position]) - _median(groups[index])),
        )
        groups[target] = sorted(
            groups[target] + groups[index], key=lambda item: float(item["dubbed_start"])
        )
        groups.pop(index)

    segments: list[dict] = []
    for index, group in enumerate(groups):
        start = float(group[0]["dubbed_start"])
        end = float(group[-1]["dubbed_start"]) + float(group[-1].get("duration") or 0.0)
        if index > 0:
            # Put the splice halfway between the last window that still agreed
            # with the previous plateau and the first one that did not.
            start = 0.5 * (segments[-1]["dubbed_end"] + start)
            segments[-1]["dubbed_end"] = start
        delays = np.asarray([float(item["delay"]) for item in group], dtype=np.float64)
        confidences = [float(item.get("confidence") or 0.0) for item in group]
        segments.append(
            {
                "dubbed_start": round(start, 4),
                "dubbed_end": round(end, 4),
                "delay": round(float(np.median(delays)), 6),
                "delay_spread_sec": round(
                    float(np.median(np.abs(delays - np.median(delays)))), 6
                ),
                "confidence": round(float(np.median(confidences)), 4),
                "windows": len(group),
            }
        )
    for segment in segments:
        segment["dubbed_start"] = round(float(segment["dubbed_start"]), 4)
        segment["dubbed_end"] = round(float(segment["dubbed_end"]), 4)
    return segments
