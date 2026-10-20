from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_URL = "http://127.0.0.1:8768/api/config"
PRODUCT_URL = "http://127.0.0.1:8768/product"


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def main() -> int:
    try:
        with urllib.request.urlopen(CONFIG_URL, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeError, ValueError, urllib.error.URLError):
        return 1
    configured_models = str(((payload or {}).get("paths") or {}).get("models") or "")
    if not configured_models or not _same_path(configured_models, ROOT / "models"):
        return 1
    print("[DubClean] Сервис уже запущен; открывается рабочая страница.")
    if os.environ.get("PAIRED_REFERENCE_CANCEL_NO_BROWSER") != "1":
        webbrowser.open(PRODUCT_URL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
