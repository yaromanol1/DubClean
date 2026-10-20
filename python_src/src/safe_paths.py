from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_filename(name: str, *, fallback: str = "file") -> str:
    """Return a filesystem-safe filename while preserving readable Unicode."""
    value = Path(str(name or "")).name
    value = _UNSAFE_CHARS.sub("_", value)
    value = value.replace("..", "_")
    value = " ".join(value.split()).strip(" ._")
    return value or fallback


def safe_component(value: str, *, fallback: str = "item") -> str:
    """Return one safe path component, never a path with separators."""
    return safe_filename(value, fallback=fallback)


def is_within(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """True when ``path`` resolves inside one of the allowed roots."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for root in roots:
        try:
            root_resolved = Path(root).resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False
