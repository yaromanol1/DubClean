from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "python_src" / "experiments" / "paired_reference_cancel"
CONFIG = PKG / "config.yaml"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DATA = ROOT / "data"
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party_models"
RUNTIME = DATA / "runtime"
USER_SETTINGS = RUNTIME / "user_settings.json"
LOCATION_STATE = RUNTIME / "portable_location.json"


def p(value: Path) -> str:
    """Write a path into config.yaml.

    Everything that lives inside the package is stored relative to the package
    root, so config.yaml never records the machine it was configured on and the
    same file keeps working after the folder is moved.  Only a projects folder
    the user deliberately placed on another disk stays absolute.
    """
    # Normalise without following links: models/ or third_party_models/ may be
    # a junction to shared storage, and resolving it would write the link
    # target instead of the package-relative path the package must keep.
    absolute = Path(os.path.abspath(value))
    try:
        relative = absolute.relative_to(Path(os.path.abspath(ROOT)))
    except ValueError:
        return absolute.as_posix()
    return "./" + relative.as_posix() if relative.parts else "."


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
        return default


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_TOP_LEVEL = re.compile(r"^([A-Za-z_]\w*):")
_INDENTED_KEY = re.compile(r"^(\s+)([A-Za-z_]\w*):(.*)$")
_LIST_ITEM = re.compile(r"^\s+-\s")
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _existing_scalars(lines: list[str]) -> dict[tuple[str, str], str]:
    """Read simple scalar values without requiring PyYAML during bootstrap."""
    values: dict[tuple[str, str], str] = {}
    current_section: str | None = None
    for line in lines:
        top = _TOP_LEVEL.match(line)
        if top:
            current_section = top.group(1)
            continue
        key_match = _INDENTED_KEY.match(line)
        if not key_match or current_section is None:
            continue
        _indent, key, rest = key_match.groups()
        value = rest.strip()
        if not value or value.startswith(("[", "-", "{", "#")):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[(current_section, key)] = value
    return values


def _as_absolute(value: str | None) -> Path | None:
    if not value:
        return None
    try:
        candidate = Path(value).expanduser()
    except (OSError, ValueError):
        return None
    if not candidate.is_absolute():
        return None
    return candidate.resolve()


def _norm(value: Path | str) -> str:
    return os.path.normcase(os.path.normpath(str(value)))


def _same_path(left: Path, right: Path) -> bool:
    return _norm(left) == _norm(right)


def _old_portable_roots(
    existing: dict[tuple[str, str], str],
) -> list[Path]:
    roots: list[Path] = []

    # Packages configured by an older build stored paths.uploads absolutely.
    # Its parent structure still recovers the package root from before a move;
    # newer builds keep the value relative and rely on portable_location.json.
    old_uploads = _as_absolute(existing.get(("paths", "uploads")))
    if (
        old_uploads is not None
        and old_uploads.name.casefold() == "uploads"
        and old_uploads.parent.name.casefold() == "data"
    ):
        roots.append(old_uploads.parent.parent)

    state = _read_json(LOCATION_STATE, {})
    accepted_products = {
        "dubclean-portable",
        "DubClean Portable",
        "DubClean-RU-portable",  # legacy state written before the public rename
    }
    state_product = None
    if isinstance(state, dict):
        state_product = state.get("product_id") or state.get("product")
    if state_product in accepted_products:
        state_root = _as_absolute(str(state.get("root") or ""))
        if state_root is not None:
            roots.append(state_root)

    unique: list[Path] = []
    for candidate in roots:
        if _same_path(candidate, ROOT):
            continue
        if not any(_same_path(candidate, item) for item in unique):
            unique.append(candidate)
    return unique


def _rebase_path_string(
    value: str, mappings: list[tuple[Path, Path]]
) -> tuple[str, bool]:
    """Rebase one complete Windows path, never arbitrary prose or a URL."""
    if not value or not _WINDOWS_ABSOLUTE.match(value):
        return value, False
    raw = os.path.normpath(value)
    compared = os.path.normcase(raw)
    for old_root, new_root in mappings:
        old_raw = os.path.normpath(str(old_root))
        old_compared = os.path.normcase(old_raw)
        if compared == old_compared:
            return str(new_root), True
        prefix = old_compared + os.sep
        if compared.startswith(prefix):
            # normcase does not change string length on Windows. Slice the
            # original normalized value so readable filename casing is kept.
            relative = raw[len(old_raw) :].lstrip("\\/")
            return str(new_root / Path(relative)), True
    return value, False


def _rebase_node(
    value: Any, mappings: list[tuple[Path, Path]]
) -> tuple[Any, int]:
    if isinstance(value, str):
        replacement, changed = _rebase_path_string(value, mappings)
        return replacement, int(changed)
    if isinstance(value, list):
        result: list[Any] = []
        changes = 0
        for item in value:
            replacement, count = _rebase_node(item, mappings)
            result.append(replacement)
            changes += count
        return result, changes
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        changes = 0
        for key, item in value.items():
            replacement, count = _rebase_node(item, mappings)
            result[key] = replacement
            changes += count
        return result, changes
    return value, 0


def _rebase_json_file(
    path: Path, mappings: list[tuple[Path, Path]]
) -> tuple[int, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return 0, f"{path}: {error}"
    replacement, changes = _rebase_node(payload, mappings)
    if changes:
        try:
            _atomic_json(path, replacement)
        except OSError as error:
            return 0, f"{path}: {error}"
    return changes, None


def _configured_projects_root(
    existing: dict[tuple[str, str], str],
    mappings: list[tuple[Path, Path]],
) -> Path:
    settings = _read_json(USER_SETTINGS, {})
    raw_value = ""
    if isinstance(settings, dict):
        raw_value = str(settings.get("projects_root") or "").strip()
    if not raw_value:
        raw_value = str(existing.get(("paths", "projects")) or "").strip()
    if not raw_value:
        return DATA / "projects"
    if not _WINDOWS_ABSOLUTE.match(raw_value):
        # The published config.yaml ships package-relative placeholders so it
        # carries no path of the machine it was built on.  That is the normal
        # first-run state, not a broken user setting.
        return DATA / "projects"

    rebased, was_rebased = _rebase_path_string(raw_value, mappings)
    candidate = _as_absolute(rebased)
    if candidate is None:
        print(
            "[DubClean] Предупреждение: папка проектов в настройках некорректна; "
            "используется data\\projects."
        )
        return DATA / "projects"

    if was_rebased:
        return candidate
    if candidate.is_dir():
        return candidate

    # An explicitly selected external disk may be disconnected after moving
    # the package. Starting with the local folder keeps the UI available; the
    # saved choice is not erased, so reconnecting the disk restores it.
    print(
        f"[DubClean] Предупреждение: внешняя папка проектов недоступна: {candidate}. "
        "Временно используется data\\projects."
    )
    return DATA / "projects"


def _section_scalars(projects_root: Path) -> dict[tuple[str, str], str]:
    return {
        ("paths", "uploads"): p(DATA / "uploads"),
        ("paths", "projects"): p(projects_root),
        ("paths", "outputs"): p(DATA / "outputs"),
        ("paths", "legacy_outputs"): p(DATA / "outputs"),
        ("paths", "models"): p(MODELS),
        ("paths", "runtime"): p(RUNTIME),
        ("paths", "backups"): p(DATA / "backups"),
        ("speech_extraction", "python"): p(PYTHON),
        ("speech_extraction", "runner"): p(PKG / "speech_stem_runner.py"),
        (
            "speech_extraction",
            "checkpoint_dir",
        ): p(THIRD_PARTY / "MossFormer2_SE_48K_original"),
        (
            "speech_extraction",
            "dubbed_checkpoint_dir",
        ): p(THIRD_PARTY / "MossFormer2_SE_48K"),
        ("music_separation", "python"): p(PYTHON),
        ("music_separation", "runner"): p(PKG / "music_separator_runner.py"),
        ("music_separation", "model_dir"): p(THIRD_PARTY / "audio_separator"),
        ("training", "python"): p(PYTHON),
        ("training", "runner"): p(PKG / "semantic_separator_runner.py"),
    }


def _allowed_media_roots() -> list[str]:
    # Folders chosen through the native dialog are persisted separately in
    # data/runtime/authorized_media_roots.json by the backend.
    return [p(DATA / "uploads")]


def rewrite_config(lines: list[str], projects_root: Path) -> None:
    scalars = _section_scalars(projects_root)
    allowed_roots = _allowed_media_roots()
    out: list[str] = []
    current_section: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        top = _TOP_LEVEL.match(line)
        if top:
            current_section = top.group(1)
            out.append(line)
            index += 1
            continue
        if line and not line[0].isspace():
            out.append(line)
            index += 1
            continue
        if current_section == "paths" and re.match(
            r"^\s+allowed_media_roots:\s*$", line
        ):
            out.append("  allowed_media_roots:")
            for root in allowed_roots:
                out.append(f"    - {root}")
            index += 1
            while index < len(lines) and _LIST_ITEM.match(lines[index]):
                index += 1
            continue
        key_match = _INDENTED_KEY.match(line)
        if key_match and current_section is not None:
            indent, key, rest = key_match.groups()
            target = scalars.get((current_section, key))
            if target is not None and rest.strip() and not rest.lstrip().startswith(
                ("[", "-")
            ):
                out.append(f"{indent}{key}: {target}")
                index += 1
                continue
        out.append(line)
        index += 1
    replacement = "\n".join(out) + "\n"
    # START_DUBCLEAN runs this configurator on every launch.  Rewriting an
    # already-correct config changes its mtime and makes completed processing
    # results look stale even though no effective setting changed.
    try:
        current = CONFIG.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        current = None
    if current != replacement:
        CONFIG.write_text(replacement, encoding="utf-8")


def _metadata_roots(projects_root: Path) -> list[Path]:
    roots = [DATA]
    if not any(_same_path(projects_root, item) for item in roots):
        roots.append(projects_root)
    return roots


def rebase_metadata(
    mappings: list[tuple[Path, Path]], projects_root: Path
) -> tuple[int, int, list[str]]:
    if not mappings:
        return 0, 0, []
    files_changed = 0
    paths_changed = 0
    errors: list[str] = []
    seen: set[str] = set()
    for root in _metadata_roots(projects_root):
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            identity = _norm(path)
            if identity in seen or _same_path(path, LOCATION_STATE):
                continue
            seen.add(identity)
            changes, error = _rebase_json_file(path, mappings)
            if error:
                errors.append(error)
            if changes:
                files_changed += 1
                paths_changed += changes
    return files_changed, paths_changed, errors


def main() -> int:
    if not CONFIG.is_file():
        print(f"[DubClean] Не найден config.yaml: {CONFIG}")
        return 1

    lines = CONFIG.read_text(encoding="utf-8").splitlines()
    existing = _existing_scalars(lines)
    mappings = [(old_root, ROOT) for old_root in _old_portable_roots(existing)]
    projects_root = _configured_projects_root(existing, mappings)

    for directory in [
        DATA / "uploads",
        projects_root,
        DATA / "outputs",
        RUNTIME,
        DATA / "backups",
    ]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"[DubClean] Не удалось подготовить папку {directory}: {error}")
            return 1

    files_changed, paths_changed, errors = rebase_metadata(mappings, projects_root)
    if errors:
        print(
            f"[DubClean] Предупреждение: не удалось проверить JSON-файлов: "
            f"{len(errors)}. Первый: {errors[0]}"
        )
    if paths_changed:
        print(
            f"[DubClean] После переноса обновлено внутренних путей: "
            f"{paths_changed} в {files_changed} JSON-файлах."
        )

    rewrite_config(lines, projects_root)
    _atomic_json(
        LOCATION_STATE,
        {
            "schema_version": 1,
            "product": "DubClean Portable",
            "product_id": "dubclean-portable",
            "root": str(ROOT),
        },
    )
    print(f"[DubClean] Переносимые пути настроены: {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
