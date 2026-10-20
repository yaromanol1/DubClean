"""Audio extraction and low-level I/O for the paired-reference-cancel experiment.

Everything here works on float32 PCM. Multi-channel audio is kept as a 2-D
array shaped ``(n_samples, n_channels)`` throughout the pipeline so the
native channel count of a file (mono/stereo/5.1) survives as far as
possible; only the alignment search itself works on a mono downmix proxy.

This module shells out to ffmpeg/ffprobe (same tools the main project uses)
instead of depending on a Python container-demuxing library, so the
experiment's isolated venv only needs numpy/scipy/soundfile/flask.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.safe_paths import is_within, safe_component, safe_filename  # noqa: E402  (re-exported below)

__all__ = [
    "is_within",
    "safe_component",
    "safe_filename",
    "FFmpegNotFoundError",
    "check_ffmpeg",
    "check_ffprobe",
    "probe_audio_streams",
    "probe_duration",
    "extract_audio_window",
    "load_wav",
    "save_wav",
    "resample",
    "to_mono",
    "match_channels",
    "ensure_2d",
]


class FFmpegNotFoundError(RuntimeError):
    pass


def check_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FFmpegNotFoundError(
            "ffmpeg не найден в PATH. Установите ffmpeg (например "
            "winget install --id Gyan.FFmpeg) и перезапустите терминал."
        )
    return exe


def check_ffprobe() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise FFmpegNotFoundError(
            "ffprobe не найден в PATH. Он входит в тот же пакет, что и ffmpeg."
        )
    return exe


def probe_audio_streams(path: str | Path) -> list[dict]:
    """List audio streams via ffprobe, in the order ffmpeg addresses with -map.

    Each entry's ``index`` is the *absolute* stream index inside the
    container (what ``-map 0:<index>`` expects), not the audio-only index.
    """
    ffprobe = check_ffprobe()
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe не смог прочитать файл за 60 секунд: {path}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe не смог прочитать файл: {path}\nstderr:\n{result.stderr[-4000:]}"
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError as exc:
        raise RuntimeError(f"ffprobe вернул нечитаемый JSON для {path}: {exc}") from exc
    streams = []
    for stream in payload.get("streams", []):
        duration = stream.get("duration")
        try:
            duration = float(duration) if duration is not None else None
            if duration is not None and (not math.isfinite(duration) or duration <= 0):
                duration = None
        except (TypeError, ValueError):
            duration = None
        tags = stream.get("tags", {}) or {}
        streams.append(
            {
                "index": int(stream["index"]),
                "codec_name": stream.get("codec_name", "?"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "channels": int(stream["channels"]) if stream.get("channels") else None,
                "channel_layout": stream.get("channel_layout", "?"),
                "duration_sec": duration,
                "time_base": stream.get("time_base"),
                "start_time": stream.get("start_time"),
                "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
                "language": tags.get("language"),
                "title": tags.get("title"),
            }
        )
    return streams


def probe_video_stream(path: str | Path) -> dict | None:
    """Describe the one real picture stream, ignoring cover art.

    ``-select_streams V`` (capital) excludes attached pictures, which are the
    embedded posters Matroska stores as attachments.  Without that the answer
    for a film with cover art would describe a 425x700 JPEG.
    """
    ffprobe = check_ffprobe()
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "V:0",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except ValueError:
        return None
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]

    def _rate(value: str | None) -> float | None:
        if not value or "/" not in str(value):
            return None
        numerator, _, denominator = str(value).partition("/")
        try:
            top, bottom = float(numerator), float(denominator)
        except ValueError:
            return None
        return round(top / bottom, 4) if bottom else None

    width = int(stream["width"]) if stream.get("width") else None
    height = int(stream["height"]) if stream.get("height") else None
    return {
        "index": int(stream["index"]),
        "codec_name": stream.get("codec_name", "?"),
        "profile": stream.get("profile"),
        "width": width,
        "height": height,
        "pixels": (width * height) if width and height else None,
        "frame_rate": _rate(stream.get("r_frame_rate")),
        "bit_rate": int(stream["bit_rate"]) if stream.get("bit_rate") else None,
        "pix_fmt": stream.get("pix_fmt"),
    }


def probe_duration(path: str | Path) -> float | None:
    ffprobe = check_ffprobe()
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _run_ffmpeg(command: list[str], description: str) -> None:
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"{description}\nКоманда: {' '.join(command)}\nffmpeg stderr:\n{result.stderr[-4000:]}")


def extract_audio_window(
    input_path: str | Path,
    stream_index: int,
    output_path: str | Path,
    start_sec: float,
    duration_sec: float | None,
    sample_rate: int,
) -> Path:
    """Decode one audio stream into float32 PCM WAV, preserving its channel count.

    ``stream_index`` is the absolute ffprobe stream index (see
    :func:`probe_audio_streams`). Writes atomically via a temp file.
    """
    ffmpeg = check_ffmpeg()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp{output.suffix}")
    command = [
        ffmpeg, "-y",
        "-ss", f"{max(0.0, start_sec):.3f}",
        "-i", str(input_path),
    ]
    if duration_sec is not None:
        command += ["-t", f"{duration_sec:.3f}"]
    command += [
        "-map", f"0:{stream_index}",
        "-vn", "-sn",
        "-ar", str(sample_rate),
        "-acodec", "pcm_f32le",
        str(temporary),
    ]
    try:
        _run_ffmpeg(command, f"Не удалось извлечь аудиопоток {stream_index} из {input_path}.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


# Bumping this string forces every cached final film to be rebuilt, because the
# muxing recipe — not only the video and audio it is built from — decides
# whether ordinary players can open the result.
REMUX_RECIPE_VERSION = "compatible_single_video_v3"

# Appended to the cleaned track's title so it is unmistakable in a player's
# track menu, whichever descriptive name the pipeline chose for it.
PROCESSED_TRACK_SUFFIX = "обработано DubClean"


def build_remux_command(
    ffmpeg: str,
    video: str | Path,
    audio: str | Path,
    output: str | Path,
    *,
    audio_codec: str,
    audio_bitrate: str = "",
    audio_channels: int = 0,
    audio_sample_rate: int = 0,
    track_title: str = "",
    track_language: str = "rus",
    extra_audio: list[dict[str, str]] | None = None,
    keep_source_audio: bool = False,
) -> list[str]:
    """Build the ffmpeg call that packs the cleaned track into the film.

    The recipe is deliberately conservative, because a technically valid
    Matroska file is not automatically a *playable* one:

    * ``0:V:0`` (capital V) takes the single real video stream.  Lowercase
      ``0:v`` also matches cover art, which ffmpeg demuxes from Matroska
      attachments as ``attached_pic`` video streams; copying those produces
      extra video tracks that declare ~30000 fps and deliver one frame at
      t=0, after which players wait forever for samples that never come.
    * Every disposition is set explicitly.  Copied streams keep the source
      flags, so without this a source audio track or a subtitle track could
      still claim ``default`` next to the cleaned one.
    * The cleaned track is the first audio stream, so players that simply
      take audio stream 0 land on it.

    ``extra_audio`` carries separate files that were already brought onto the
    video's timeline; they are re-encoded like the cleaned track.
    ``keep_source_audio`` keeps the video file's own tracks, which need no
    retiming because they ship inside the very file supplying the picture.
    Tracks from the *other* source file must never be added this way: they sit
    on a different timeline and would play out of sync.
    """
    extras = list(extra_audio or [])
    command = [str(ffmpeg), "-y", "-fflags", "+genpts", "-i", str(video), "-i", str(audio)]
    for item in extras:
        command += ["-i", str(item["path"])]
    command += ["-map", "0:V:0?", "-map", "1:a:0"]
    for position in range(len(extras)):
        command += ["-map", f"{position + 2}:a:0"]
    if keep_source_audio:
        command += ["-map", "0:a?"]
    command += [
        "-map", "0:s?",
        "-map", "0:t?",
        "-map_metadata", "0", "-map_chapters", "0",
        "-c", "copy",
        "-c:a:0", str(audio_codec),
    ]
    if audio_bitrate:
        command += ["-b:a:0", str(audio_bitrate)]
    if audio_channels:
        command += ["-ac:a:0", str(int(audio_channels))]
    if audio_sample_rate:
        command += ["-ar:a:0", str(int(audio_sample_rate))]
    for position in range(len(extras)):
        command += [f"-c:a:{position + 1}", str(audio_codec)]
        if audio_bitrate:
            command += [f"-b:a:{position + 1}", str(audio_bitrate)]
        if audio_sample_rate:
            command += [f"-ar:a:{position + 1}", str(int(audio_sample_rate))]
    command += [
        "-disposition:v:0", "default",
        # Clear first, then grant: a copied source track arrives with the
        # source's own default flag and would otherwise fight the cleaned one.
        "-disposition:a", "0",
        "-disposition:a:0", "default",
        # Copied subtitles keep the source flags too.  A subtitle track that
        # arrives flagged "default" turns itself on over the cleaned dub.
        "-disposition:s", "0",
    ]
    if track_title:
        command += ["-metadata:s:a:0", f"title={track_title} · {PROCESSED_TRACK_SUFFIX}"]
    command += ["-metadata:s:a:0", f"language={track_language}"]
    for position, item in enumerate(extras):
        if item.get("title"):
            command += [f"-metadata:s:a:{position + 1}", f"title={item['title']}"]
        command += [
            f"-metadata:s:a:{position + 1}",
            f"language={item.get('language') or 'und'}",
        ]
    command += ["-avoid_negative_ts", "make_zero", str(output)]
    return command


def ensure_2d(data: np.ndarray) -> np.ndarray:
    """Force ``(n_samples,)`` -> ``(n_samples, 1)``; leave 2-D arrays as-is."""
    if data.ndim == 1:
        return data[:, None]
    return data


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as float32, shape (n_samples, n_channels), always 2-D."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data), int(sr)


def save_wav(path: str | Path, data: np.ndarray, sample_rate: int) -> None:
    """Save float32 PCM WAV. Accepts 1-D mono or 2-D (n_samples, n_channels).

    NaN/Inf are replaced with 0 and the signal is hard-clipped to [-1, 1] as
    a last-resort safety net; callers should already guard against this, see
    diagnostics.clipping_ratio for a metric that reports how often it fires.
    """
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, -1.0, 1.0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, int(sample_rate), subtype="FLOAT")


def resample(data: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase-resample a (n_samples, n_channels) or 1-D array along time."""
    if source_sr == target_sr:
        return data
    from scipy.signal import resample_poly

    divisor = math.gcd(int(source_sr), int(target_sr))
    up, down = target_sr // divisor, source_sr // divisor
    axis = 0
    return resample_poly(data, up, down, axis=axis).astype(np.float32, copy=False)


def to_mono(data: np.ndarray) -> np.ndarray:
    """Average channels down to a 1-D mono signal."""
    data = ensure_2d(data)
    return data.mean(axis=1).astype(np.float32, copy=False)


def match_channels(data: np.ndarray, target_channels: int) -> np.ndarray:
    """Broadcast/downmix a (n_samples, n_channels) array to exactly target_channels.

    Fewer channels than requested: the mono-downmix is duplicated to every
    output channel (crude but predictable; documented as a limitation for
    true 5.1 handling in README). More channels than requested: averaged
    down to target_channels via mono, then duplicated if target_channels>1.
    """
    data = ensure_2d(data)
    if data.shape[1] == target_channels:
        return data
    mono = to_mono(data)
    if target_channels <= 1:
        return mono[:, None]
    return np.tile(mono[:, None], (1, target_channels))
