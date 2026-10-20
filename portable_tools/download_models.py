"""Install manifest-pinned model files with atomic SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def verified_download(url: str, target: Path, digest: str, size: int = 0) -> None:
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS downloads are accepted")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".partial")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DubClean-Installer/0.6.0"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            if not response.geturl().startswith("https://"):
                raise ValueError("Download redirected to an insecure URL")
            total, last_report = 0, time.monotonic()
            for block in iter(lambda: response.read(1024 * 1024), b""):
                total += len(block)
                if size and total > size:
                    raise ValueError("Downloaded file is larger than the manifest size")
                stream.write(block)
                if time.monotonic() - last_report > 5:
                    print(f"[DubClean] {target.name}: {total / 1024**2:.1f} MiB", flush=True)
                    last_report = time.monotonic()
        if size and temporary.stat().st_size != size:
            raise ValueError("Incomplete download")
        if sha256(temporary) != digest.upper():
            raise ValueError("Downloaded SHA-256 does not match the manifest")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def model_path(root: Path, spec: dict) -> Path:
    relative = spec.get("file") or spec["dir"] + "/" + spec["weight_file"]
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Model path escapes the installation directory")
    return target


def install(root: Path, *, verify_only: bool = False) -> None:
    manifest = json.loads((root / "portable_manifest.json").read_text(encoding="utf-8"))
    for name, spec in manifest["models"].items():
        if not spec.get("required_for_default"):
            continue
        target = model_path(root, spec)
        expected = spec.get("sha256") or spec["weight_sha256"]
        size = int(spec["size_bytes"])
        valid = target.is_file() and target.stat().st_size == size and sha256(target) == expected.upper()
        if not valid:
            if verify_only:
                raise ValueError(f"Missing or invalid model: {name}")
            if target.exists():
                raise ValueError(f"Model {name} was modified. Move it aside before restoring; it will not be overwritten.")
            errors = []
            for url in spec.get("download_urls", []):
                for attempt in range(2):
                    try:
                        print(f"[DubClean] Downloading {name}: {url}", flush=True)
                        verified_download(url, target, expected, size)
                        break
                    except (OSError, ValueError) as error:
                        errors.append(str(error))
                        if attempt == 0:
                            time.sleep(1)
                if target.is_file():
                    break
            else:
                raise RuntimeError(f"Cannot download {name}: {'; '.join(errors) or 'no download URL in manifest'}")
        if spec.get("weight_file"):
            pointer = target.parent / "last_best_checkpoint"
            wanted = spec["weight_file"] + "\n"
            if not pointer.exists() or pointer.read_text(encoding="utf-8") != wanted:
                pointer.write_text(wanted, encoding="utf-8")
        print(f"[OK] {name}: SHA-256 verified", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        install(args.root.resolve(), verify_only=args.verify_only)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"[DubClean] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
