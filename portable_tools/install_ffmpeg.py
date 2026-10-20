"""Download the Windows essentials FFmpeg build directly from its publisher."""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from download_models import verified_download

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip"
CHECKSUM = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"
SIZE = 114768076


def install(root: Path) -> None:
    directory = root / "tools" / "ffmpeg"
    if all((directory / "bin" / (name + ".exe")).is_file() or shutil.which(name)
           for name in ("ffmpeg", "ffprobe")):
        print("[OK] FFmpeg and ffprobe are already available")
        return
    checksum = CHECKSUM
    with tempfile.TemporaryDirectory(prefix="dubclean-ffmpeg-") as temp:
        archive = Path(temp) / "ffmpeg.zip"
        for attempt in range(3):
            try:
                verified_download(URL, archive, checksum, SIZE)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)
        with zipfile.ZipFile(archive) as z:
            entries = [info for info in z.infolist() if not info.is_dir()]
            roots = {Path(info.filename).parts[0] for info in entries}
            if len(roots) != 1:
                raise ValueError("Unexpected FFmpeg archive layout")
            for info in entries:
                relative = Path(*Path(info.filename).parts[1:])
                target = (directory / relative).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise ValueError("Unsafe path in FFmpeg archive")
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        if not all((directory / "bin" / (name + ".exe")).is_file() for name in ("ffmpeg", "ffprobe")):
            raise ValueError("FFmpeg archive did not contain both required tools")
        (directory / "DOWNLOAD_SOURCE.txt").write_text(
            f"Downloaded directly by the user installer from {URL}\nSHA-256: {checksum}\n"
            "Build license: GPLv3. Publisher notices and documentation retained.\n"
            "Build/source information: https://www.gyan.dev/ffmpeg/builds/\n", encoding="utf-8")
    print("[OK] FFmpeg installed in tools/ffmpeg/bin")


if __name__ == "__main__":
    try:
        install(ROOT)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"[DubClean] FFmpeg installation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
