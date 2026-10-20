#!/usr/bin/env python
"""Run the dedicated vocals/instrumental separator in an isolated process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_separator.separator import Separator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--chunk-sec", type=int, default=120)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        model_file_dir=str(Path(args.model_dir).resolve()),
        output_dir=str(output_dir),
        output_format="FLAC",
        sample_rate=48000,
        use_soundfile=True,
        use_autocast=True,
        chunk_duration=max(10, int(args.chunk_sec)),
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
        },
    )
    print("[status] Загрузка модели разделения голоса и сопровождения", flush=True)
    separator.load_model(args.model)
    print("[status] Разделение на голос и музыку с эффектами", flush=True)
    outputs = separator.separate(str(source))
    payload = {
        "outputs": [
            str(
                (
                    Path(item)
                    if Path(item).is_absolute()
                    else output_dir / Path(item).name
                ).resolve()
            )
            for item in outputs
        ]
    }
    print("[result] " + json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
