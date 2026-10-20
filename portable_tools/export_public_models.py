"""Export inference checkpoints without private training metadata.

Run with the original installation as --source and a clean release copy as
--destination. Original files are never modified. Tensor equality is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def export(source: Path, destination: Path, repository: str) -> None:
    import torch

    if source.resolve() == destination.resolve():
        raise ValueError("Export into a separate release copy, never the working installation")
    manifest = json.loads((source / "portable_manifest.json").read_text(encoding="utf-8"))
    tag = "v" + manifest["version"]
    for model_id, spec in manifest["models"].items():
        if not spec.get("required_for_default"):
            continue
        relative = spec.get("file") or spec["dir"] + "/" + spec["weight_file"]
        original, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if model_id in {"dubclean_voice", "dubclean_me", "speech_extractor_dubbed"}:
            state = torch.load(original, map_location="cpu", weights_only=True)
            if model_id == "speech_extractor_dubbed":
                published = {"model": state["model"]}
                tensor_key = "model"
            else:
                allowed = ("n_fft", "hop_length", "base_channels", "lag_bank_max_frames",
                           "lag_bank_step_frames", "bottleneck_time_dilations")
                published = {"format": state["format"], "model_state": state["model_state"],
                             "configuration": {k: v for k, v in state["configuration"].items() if k in allowed}}
                tensor_key = "model_state"
            torch.save(published, target)
            loaded = torch.load(target, map_location="cpu", weights_only=True)
            assert loaded[tensor_key].keys() == state[tensor_key].keys()
            for key, value in state[tensor_key].items():
                if not torch.equal(value, loaded[tensor_key][key]):
                    raise ValueError(f"Export changed tensor {model_id}:{key}")
            spec["export"] = "inference_only; tensors identical; optimizer and private training metadata omitted"
        else:
            import shutil
            shutil.copy2(original, target)
        digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest().upper()
        spec["size_bytes"] = target.stat().st_size
        spec["weight_sha256" if "weight_sha256" in spec else "sha256"] = digest
        asset = {"speech_extractor": "mossformer2-original-48k.pt",
                 "speech_extractor_dubbed": "mossformer-plus-48k.pt"}.get(model_id, target.name)
        spec["distribution"] = "github_release" if target.stat().st_size > 100 * 1024**2 else "git"
        spec["download_urls"] = [f"https://github.com/{repository}/releases/download/{tag}/{asset}"] if spec["distribution"] == "github_release" else [f"https://raw.githubusercontent.com/{repository}/{tag}/{relative}"]
        print(model_id, spec["size_bytes"], digest, flush=True)
    manifest["repository"] = f"https://github.com/{repository}"
    manifest["external_runtime"]["network"] = "required for first dependency, model and FFmpeg installation only"
    (destination / "portable_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    args = parser.parse_args()
    export(args.source, args.destination, args.repository)
