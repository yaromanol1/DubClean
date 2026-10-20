"""Create a local five-zone Silero VAD listening report for any media file."""
from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import soundfile as sf

from experiments.paired_reference_cancel import vad_preview


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--stream", type=int, default=0, help="Absolute FFmpeg stream index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio = output / "source_audio.flac"
    _run([args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(args.source), "-map", f"0:{args.stream}", "-vn", "-ac", "2", "-c:a", "flac", str(audio)])
    duration = float(sf.info(str(audio)).duration)
    selections = vad_preview.analyse_file(audio, duration)
    rows: list[str] = []
    for item in selections:
        clip = None
        if item.get("preview_start_sec") is not None:
            clip = output / f"{item['zone']}.mp3"
            _run([args.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(item["preview_start_sec"]), "-i", str(audio), "-t", str(item["preview_duration_sec"]), "-vn", "-c:a", "libmp3lame", "-b:a", "160k", str(clip)])
            item["preview_file"] = clip.name
        meta = {
            "Окно анализа": f"{item['analysis_start_sec']}–{item['analysis_end_sec']} с",
            "Превью": f"{item.get('preview_start_sec')}–{item.get('preview_end_sec')} с",
            "Речевая область": f"{item.get('speech_region_start_sec')}–{item.get('speech_region_end_sec')} с",
            "VAD-доля речи": item.get("speech_ratio"),
            "Статус": item.get("status"),
            "Fallback": (item.get("fallback") or {}).get("used", False),
        }
        audio_tag = f'<audio controls src="{html.escape(clip.name)}"></audio>' if clip else "<p>Клип не создан.</p>"
        wave = vad_preview.waveform_svg(
            audio,
            item["analysis_start_sec"],
            item["analysis_end_sec"],
            highlights=[(iv["start_sec"], iv["end_sec"]) for iv in item.get("speech_intervals") or []],
            marker_start_sec=item.get("preview_start_sec"),
            marker_end_sec=item.get("preview_end_sec"),
        )
        rows.append("<section><h2>" + html.escape(item["zone"]) + "</h2>" + wave + audio_tag + "<dl>" + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>" for k, v in meta.items()) + "</dl><p>" + html.escape(str(item.get("selection_reason") or "")) + "</p></section>")
    (output / "vad_selection_manifest.json").write_text(json.dumps({"source": str(args.source), "stream": args.stream, "duration_sec": duration, "selections": selections}, ensure_ascii=False, indent=2), encoding="utf-8")
    page = """<!doctype html><meta charset=\"utf-8\"><title>Silero VAD preview check</title><style>body{font:16px system-ui;max-width:900px;margin:30px auto;padding:0 18px}section{border:1px solid #ddd;border-radius:12px;padding:15px;margin:14px 0}audio{width:100%}dl{display:grid;grid-template-columns:190px 1fr;gap:5px}dt{font-weight:bold}dd{margin:0}</style><h1>Проверка пяти зон Silero VAD</h1>""" + "\n".join(rows)
    (output / "vad_preview_report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"manifest": str(output / "vad_selection_manifest.json"), "report": str(output / "vad_preview_report.html"), "selections": selections}, ensure_ascii=False))


if __name__ == "__main__":
    main()
