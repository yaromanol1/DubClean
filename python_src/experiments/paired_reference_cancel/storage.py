"""Persistent project, pair and task storage for the isolated experiment."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import yaml

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
PORTABLE_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG_PATH = MODULE_DIR / "config.yaml"
ID_RE = re.compile(r"^[a-f0-9]{12,32}$")


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if int(value.get("schema_version", 0)) != 2:
        raise RuntimeError("Неподдерживаемая версия config.yaml.")
    return value


def root_path(value: str | Path) -> Path:
    # A relative entry in config.yaml is the shipped placeholder written
    # against the package root, so it must never be read from python_src or
    # from whatever directory the process happens to be started in.
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PORTABLE_ROOT / path).resolve()


def configured_path(cfg: dict, key: str) -> Path:
    return root_path(cfg["paths"][key])


def is_within(path: Path, roots: Iterable[Path]) -> bool:
    candidate = path.resolve()
    for root in roots:
        try:
            candidate.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def validate_id(value: str, label: str = "идентификатор") -> str:
    if not ID_RE.fullmatch(str(value or "")):
        raise ValueError(f"Недопустимый {label}.")
    return str(value)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def safe_archive_label(value: str) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "_", str(value or "").casefold()).strip("_")
    return label[:48] or "derived_materials_changed"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        # The panel polls these files while workers rewrite them; on Windows
        # os.replace transiently fails with WinError 5 when a reader holds
        # the destination open. Retry briefly instead of failing the task.
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except PermissionError as exc:
            # Transient share violation while another process replaces the
            # file; see atomic_json above.
            last_error = exc
            time.sleep(0.05)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Повреждён или недоступен файл данных: {path.name}: {exc}"
            ) from exc
    raise RuntimeError(
        f"Повреждён или недоступен файл данных: {path.name}: {last_error}"
    ) from last_error


def sha256_file(path: Path, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class Store:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.projects_root = configured_path(self.cfg, "projects")
        self.uploads_root = configured_path(self.cfg, "uploads")
        self.runtime_root = configured_path(self.cfg, "runtime")
        self.models_root = configured_path(self.cfg, "models")
        self.legacy_outputs = configured_path(self.cfg, "legacy_outputs")
        self.allowed_media_roots = [
            root_path(item) for item in self.cfg["paths"].get("allowed_media_roots", [])
        ]
        # Folders the user explicitly opened through the native OS file dialog
        # are trusted for this and future sessions, so films can live anywhere
        # on disk without pre-listing every folder in config.yaml.
        self._authorized_roots_file = self.runtime_root / "authorized_media_roots.json"
        for item in read_json(self._authorized_roots_file, []) or []:
            try:
                candidate = Path(str(item)).resolve()
            except (OSError, ValueError):
                continue
            if candidate not in self.allowed_media_roots:
                self.allowed_media_roots.append(candidate)
        self._task_read_cache: dict[str, tuple[int, int, dict]] = {}
        for directory in (
            self.projects_root,
            self.uploads_root,
            self.runtime_root,
            self.models_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _rebase_persisted_path(
        self,
        value: str,
        *,
        project_id: str | None = None,
        pair_id: str | None = None,
    ) -> str:
        """Rebase an internal absolute path after the portable folder moved.

        Source films selected outside DubClean must keep their original
        absolute paths.  Only paths with an unambiguous service-owned suffix
        (a known project/pair id, data/uploads, models or third_party_models)
        are rewritten.  Existing paths are never touched.
        """
        try:
            original = Path(value)
        except (OSError, TypeError, ValueError):
            return value
        if not original.is_absolute():
            return value
        try:
            if original.exists():
                return value
        except OSError:
            pass

        parts = list(original.parts)
        folded = [part.casefold() for part in parts]

        def suffix_after(marker: str) -> tuple[str, ...] | None:
            marker_folded = marker.casefold()
            for index, part in enumerate(folded):
                if part == marker_folded:
                    return tuple(parts[index + 1 :])
            return None

        # Pair artefacts are the most common persisted paths.  Matching the
        # pair id makes this safe even when the projects root has a custom name.
        if pair_id and project_id:
            for index, part in enumerate(folded[:-1]):
                if part == "pairs" and parts[index + 1].casefold() == pair_id.casefold():
                    candidate = self.pair_dir(project_id or "", pair_id).joinpath(
                        *parts[index + 2 :]
                    )
                    return str(candidate)

        # Project-level checkpoints and reports can belong to a different
        # project than the pair that references them.  A known project id is an
        # unambiguous anchor under the currently configured projects root.
        for index, part in enumerate(parts):
            if ID_RE.fullmatch(part.casefold()):
                candidate_root = self.projects_root / part.casefold()
                if candidate_root.is_dir() or (project_id and part.casefold() == project_id):
                    return str(candidate_root.joinpath(*parts[index + 1 :]))

        uploads_suffix = suffix_after("uploads")
        if uploads_suffix is not None and "data" in folded:
            return str(self.uploads_root.joinpath(*uploads_suffix))

        third_party_suffix = suffix_after("third_party_models")
        if third_party_suffix is not None:
            return str((PORTABLE_ROOT / "third_party_models").joinpath(*third_party_suffix))

        models_suffix = suffix_after("models")
        if models_suffix is not None:
            candidate = self.models_root.joinpath(*models_suffix)
            # Avoid rebasing an arbitrary user folder also named "models".
            if candidate.exists() or original.suffix.casefold() == ".pt":
                return str(candidate)
        return value

    def _rebase_payload_paths(
        self,
        payload: Any,
        *,
        project_id: str | None = None,
        pair_id: str | None = None,
    ) -> tuple[Any, bool]:
        """Recursively normalize service-owned absolute paths in persisted JSON."""
        if isinstance(payload, dict):
            changed = False
            normalized: dict[Any, Any] = {}
            for key, item in payload.items():
                value, item_changed = self._rebase_payload_paths(
                    item, project_id=project_id, pair_id=pair_id
                )
                normalized[key] = value
                changed = changed or item_changed
            return normalized, changed
        if isinstance(payload, list):
            changed = False
            normalized_list = []
            for item in payload:
                value, item_changed = self._rebase_payload_paths(
                    item, project_id=project_id, pair_id=pair_id
                )
                normalized_list.append(value)
                changed = changed or item_changed
            return normalized_list, changed
        if isinstance(payload, str):
            normalized_path = self._rebase_persisted_path(
                payload, project_id=project_id, pair_id=pair_id
            )
            return normalized_path, normalized_path != payload
        return payload, False

    def authorize_media_dir(self, path: str | Path) -> None:
        """Trust the folder of a file the user picked via the native dialog."""
        source = Path(str(path))
        directory = (source.parent if source.suffix else source)
        try:
            directory = directory.resolve()
        except (OSError, ValueError):
            return
        if not directory.is_dir():
            return
        if directory not in self.allowed_media_roots:
            self.allowed_media_roots.append(directory)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        existing = read_json(self._authorized_roots_file, []) or []
        value = str(directory)
        if value not in existing:
            existing.append(value)
            atomic_json(self._authorized_roots_file, existing)

    def project_dir(self, project_id: str) -> Path:
        return self.projects_root / validate_id(project_id, "идентификатор проекта")

    def pair_dir(self, project_id: str, pair_id: str) -> Path:
        return (
            self.project_dir(project_id)
            / "pairs"
            / validate_id(pair_id, "идентификатор пары")
        )

    def task_path(self, project_id: str, task_id: str) -> Path:
        return (
            self.project_dir(project_id)
            / "tasks"
            / f"{validate_id(task_id, 'идентификатор задачи')}.json"
        )

    def create_project(self, name: str) -> dict:
        title = str(name or "").strip()
        if not title:
            raise ValueError("Введите название проекта.")
        project_id = new_id()
        root = self.project_dir(project_id)
        for relative in (
            "pairs",
            "tasks",
            "training/datasets",
            "training/runs",
            "checkpoints",
            "reports",
            "archive",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        project = {
            "schema_version": 2,
            "id": project_id,
            "name": title[:200],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "deleted": False,
        }
        atomic_json(root / "project.json", project)
        return project

    def load_project(self, project_id: str) -> dict:
        path = self.project_dir(project_id) / "project.json"
        project = read_json(path)
        if not isinstance(project, dict):
            raise FileNotFoundError("Проект не найден.")
        normalized, changed = self._rebase_payload_paths(
            project, project_id=project_id
        )
        if changed:
            atomic_json(path, normalized)
        return normalized

    def save_project(self, project: dict) -> None:
        project["updated_at"] = utc_now()
        atomic_json(self.project_dir(project["id"]) / "project.json", project)

    def list_projects(self) -> list[dict]:
        projects: list[dict] = []
        for path in self.projects_root.glob("*/project.json"):
            try:
                project = read_json(path)
            except RuntimeError as exc:
                projects.append(
                    {
                        "id": path.parent.name,
                        "name": "Повреждённый проект",
                        "error": str(exc),
                        "pair_count": 0,
                    }
                )
                continue
            if not project or project.get("deleted"):
                continue
            project, changed = self._rebase_payload_paths(
                project, project_id=path.parent.name
            )
            if changed:
                atomic_json(path, project)
            project = dict(project)
            project["pair_count"] = len(self.list_pairs(project["id"]))
            projects.append(project)
        return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)

    def validate_media_path(self, value: str | Path) -> Path:
        if not value:
            raise ValueError("Путь к медиафайлу не указан.")
        candidate = Path(value).expanduser().resolve()
        if not candidate.is_file() or not is_within(candidate, self.allowed_media_roots):
            roots = ", ".join(str(item) for item in self.allowed_media_roots)
            raise ValueError(f"Файл не найден или находится вне разрешённых папок: {roots}")
        return candidate

    def add_pair(
        self,
        project_id: str,
        name: str,
        original_path: str | Path,
        dubbed_path: str | Path,
        original_stream: int,
        dubbed_stream: int,
        original_probe: dict,
        dubbed_probe: dict,
    ) -> dict:
        self.load_project(project_id)
        pair_id = new_id()
        root = self.pair_dir(project_id, pair_id)
        for relative in (
            "source",
            "extracted",
            "alignment",
            "stems/original",
            "components",
            "previews",
            "dataset",
            "results/baseline",
            "results/final",
            "reports",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)
        now = utc_now()
        pair = {
            "schema_version": 3,
            "pipeline_version": 5,
            "id": pair_id,
            "project_id": project_id,
            "name": str(name or "").strip()[:200] or f"Пара {pair_id[:6]}",
            "created_at": now,
            "updated_at": now,
            "deleted": False,
            "sources": {
                "original": {
                    "path": str(self.validate_media_path(original_path)),
                    "stream_index": int(original_stream),
                    "probe": original_probe,
                },
                "dubbed": {
                    "path": str(self.validate_media_path(dubbed_path)),
                    "stream_index": int(dubbed_stream),
                    "probe": dubbed_probe,
                },
            },
            "current_stage": 1,
            "stages": {
                str(number): {
                    "status": "available" if number == 2 else "blocked",
                    "message": (
                        "Файлы проверены, можно извлекать аудио."
                        if number == 2
                        else "Сначала завершите предыдущий этап."
                    ),
                }
                for number in range(2, 11)
            },
            "warnings": [],
            "errors": [],
            "result": None,
        }
        atomic_json(root / "pair.json", pair)
        project = self.load_project(project_id)
        self.save_project(project)
        return pair

    def load_pair(self, project_id: str, pair_id: str) -> dict:
        path = self.pair_dir(project_id, pair_id) / "pair.json"
        pair = read_json(path)
        if not isinstance(pair, dict):
            raise FileNotFoundError("Пара фильмов не найдена.")
        normalized, changed = self._rebase_payload_paths(
            pair, project_id=project_id, pair_id=pair_id
        )
        if changed:
            atomic_json(path, normalized)
        return normalized

    def save_pair(self, pair: dict) -> None:
        pair["updated_at"] = utc_now()
        atomic_json(self.pair_dir(pair["project_id"], pair["id"]) / "pair.json", pair)
        project = self.load_project(pair["project_id"])
        self.save_project(project)

    def list_pairs(self, project_id: str, include_deleted: bool = False) -> list[dict]:
        self.load_project(project_id)
        pairs: list[dict] = []
        for path in (self.project_dir(project_id) / "pairs").glob("*/pair.json"):
            try:
                pair = read_json(path)
            except RuntimeError as exc:
                pairs.append(
                    {
                        "id": path.parent.name,
                        "project_id": project_id,
                        "name": "Повреждённая пара",
                        "error": str(exc),
                    }
                )
                continue
            if pair and (include_deleted or not pair.get("deleted")):
                pair, changed = self._rebase_payload_paths(
                    pair, project_id=project_id, pair_id=path.parent.name
                )
                if changed:
                    atomic_json(path, pair)
                pairs.append(pair)
        return sorted(pairs, key=lambda item: item.get("created_at", ""))

    def archive_pair(self, project_id: str, pair_id: str) -> dict:
        pair = self.load_pair(project_id, pair_id)
        pair["deleted"] = True
        pair["archived_at"] = utc_now()
        self.save_pair(pair)
        return pair

    @staticmethod
    def _source_identity(source: dict | None) -> tuple[Any, ...]:
        source = source or {}
        probe = source.get("probe") or {}
        path_value = str(source.get("path") or "")
        try:
            normalized_path = os.path.normcase(os.path.normpath(str(Path(path_value).resolve())))
        except (OSError, ValueError):
            normalized_path = os.path.normcase(os.path.normpath(path_value))
        stream_value = source.get("stream_index")
        try:
            stream_index = int(stream_value)
        except (TypeError, ValueError):
            stream_index = None
        # mtime_ns is deliberately part of the identity: replacing a movie in
        # place under the same name must invalidate every derived artefact.
        return (
            normalized_path,
            stream_index,
            probe.get("size"),
            probe.get("mtime_ns"),
        )

    def _archive_pair_directories(
        self,
        pair: dict,
        directory_names: Iterable[str],
        *,
        reason: str,
    ) -> list[str]:
        root = self.pair_dir(pair["project_id"], pair["id"])
        timestamp = utc_now().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        archive_dir = (
            root
            / "material_variants"
            / f"{timestamp}_{new_id()[:8]}_{safe_archive_label(reason)}"
        )
        archived_dirs: list[str] = []
        for name in directory_names:
            source_dir = root / name
            if not source_dir.is_dir():
                continue
            try:
                has_content = next(source_dir.iterdir(), None) is not None
            except OSError:
                has_content = True
            if not has_content:
                shutil.rmtree(source_dir, ignore_errors=True)
                continue
            archive_dir.mkdir(parents=True, exist_ok=True)
            destination = archive_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source_dir), str(destination))
            archived_dirs.append(name)
        if not archived_dirs:
            return []
        snapshot = copy.deepcopy(pair)
        snapshot.pop("material_variants", None)
        readiness = self.material_variant_readiness(snapshot, archive_dir)
        atomic_json(archive_dir / "pair_snapshot.json", snapshot)
        atomic_json(
            archive_dir / "archive_manifest.json",
            {
                "archived_at": utc_now(),
                "reason": reason,
                "sources": pair.get("sources") or {},
                "archived_dirs": archived_dirs,
                "readiness": readiness,
            },
        )
        variants = pair.get("material_variants")
        if not isinstance(variants, list):
            variants = []
            pair["material_variants"] = variants
        variants.append(
            {
                "id": archive_dir.name,
                "path": str(archive_dir.resolve()),
                "created_at": utc_now(),
                "reason": reason,
                "sources": pair.get("sources") or {},
                "dirs": archived_dirs,
                "readiness": readiness,
            }
        )
        return archived_dirs

    @staticmethod
    def material_variant_readiness(snapshot: dict, archive_dir: Path) -> dict:
        """Describe the five visible product sections backed by real artefacts."""

        def nonempty(relative: str) -> bool:
            path = archive_dir / relative
            try:
                return path.is_file() and path.stat().st_size > 0
            except OSError:
                return False

        def tree_has(relative: str, pattern: str) -> bool:
            root = archive_dir / relative
            if not root.is_dir():
                return False
            try:
                return any(path.is_file() and path.stat().st_size > 0 for path in root.glob(pattern))
            except OSError:
                return False

        sources = snapshot.get("sources") or {}
        original = sources.get("original") or {}
        dubbed = sources.get("dubbed") or {}
        files_ready = bool(original.get("path") and dubbed.get("path"))
        tracks_ready = all(
            nonempty(relative)
            for relative in (
                "extracted/original.flac",
                "extracted/dubbed.flac",
                "alignment/alignment_map.json",
            )
        )
        processing_ready = bool(
            nonempty("application/candidate_scan/selection.json")
            and tree_has("application/candidate_scan/model_results", "**/*.flac")
            and tree_has("application/candidate_scan/background_results", "**/*.flac")
        )
        preview_ready = bool(
            nonempty("application/preview_manifest.json")
            and tree_has("application/previews", "**/*_preview.mp4")
        )
        movie_ready = tree_has("application/full", "**/result.mkv")

        sections = [
            {"key": "files", "label": "Файлы", "ready": files_ready},
            {"key": "tracks", "label": "Дорожки", "ready": tracks_ready},
            {"key": "processing", "label": "Обработка", "ready": processing_ready},
            {"key": "preview", "label": "Превью", "ready": preview_ready},
            {"key": "movie", "label": "Сборка фильма", "ready": movie_ready},
        ]
        completed = sum(1 for section in sections if section["ready"])
        highest = next(
            (section["label"] for section in reversed(sections) if section["ready"]),
            "Нет готовых разделов",
        )
        return {
            "schema_version": 1,
            "completed_sections": completed,
            "total_sections": len(sections),
            "percent": int(round(completed * 100 / len(sections))),
            "highest_stage": highest,
            "sections": sections,
        }

    def delete_material_variant(
        self,
        project_id: str,
        pair_id: str,
        variant_id: str,
    ) -> dict:
        """Permanently delete one archived version and all files inside it."""
        pair = self.load_pair(project_id, pair_id)
        variants = pair.get("material_variants") or []
        variant = next(
            (item for item in variants if str(item.get("id") or "") == str(variant_id or "")),
            None,
        )
        if not variant:
            raise ValueError("Предыдущая версия не найдена.")

        root = self.pair_dir(project_id, pair_id)
        variants_root = (root / "material_variants").resolve()
        archive_dir = Path(str(variant.get("path") or "")).resolve()
        if (
            archive_dir.parent != variants_root
            or archive_dir.name != str(variant.get("id") or "")
            or not is_within(archive_dir, [variants_root])
        ):
            raise ValueError("Путь предыдущей версии находится вне архива проекта.")
        if archive_dir.exists():
            if not archive_dir.is_dir():
                raise ValueError("Путь предыдущей версии не является папкой.")
            shutil.rmtree(archive_dir)

        pair["material_variants"] = [
            item for item in variants
            if str(item.get("id") or "") != str(variant_id or "")
        ]
        pair["updated_at"] = utc_now()
        self.save_pair(pair)
        return pair

    def restore_material_variant(
        self,
        project_id: str,
        pair_id: str,
        variant_id: str,
    ) -> dict:
        """Restore a previously archived pair state without discarding the current one."""
        pair = self.load_pair(project_id, pair_id)
        variants = pair.get("material_variants") or []
        variant = next(
            (item for item in variants if str(item.get("id") or "") == str(variant_id or "")),
            None,
        )
        if not variant:
            raise ValueError("Предыдущая версия не найдена.")

        root = self.pair_dir(project_id, pair_id)
        variants_root = (root / "material_variants").resolve()
        archive_dir = Path(str(variant.get("path") or "")).resolve()
        if not is_within(archive_dir, [variants_root]):
            raise ValueError("Путь предыдущей версии находится вне проекта.")
        snapshot_path = archive_dir / "pair_snapshot.json"
        if snapshot_path.is_file():
            snapshot = read_json(snapshot_path)
        else:
            # Archives created by older portable builds did not contain a pair
            # snapshot.  Their manifests and standard directory layout still
            # contain enough information to reconstruct the visible project
            # state and make those versions reversible as well.
            snapshot = copy.deepcopy(pair)
            snapshot.pop("material_variants", None)
            snapshot["sources"] = copy.deepcopy(variant.get("sources") or {})
            extracted_metadata = archive_dir / "extracted" / "metadata.json"
            alignment_map_path = archive_dir / "alignment" / "alignment_map.json"
            preview_manifest = archive_dir / "application" / "preview_manifest.json"
            full_manifest = (
                archive_dir / "application" / "full" / "speech_rebuild" / "manifest.json"
            )
            stages = copy.deepcopy(snapshot.get("stages") or {})
            if extracted_metadata.is_file():
                snapshot["extraction"] = read_json(extracted_metadata, {})
                stages["2"] = {
                    "status": "completed",
                    "message": "Обе дорожки полностью извлечены.",
                }
            if alignment_map_path.is_file():
                alignment_map = read_json(alignment_map_path, {})
                snapshot["alignment_map"] = alignment_map
                snapshot["alignment"] = alignment_map
                snapshot["alignment_manual_correction_sec"] = float(
                    alignment_map.get("manual_correction_sec") or 0.0
                )
                snapshot["alignment_review"] = {
                    "accepted": True,
                    "note": "Восстановлено из предыдущей версии.",
                    "reviewed_at": utc_now(),
                }
                stages["3"] = {
                    "status": "completed",
                    "message": "Карта сопоставления восстановлена.",
                }
                stages["4"] = {
                    "status": "completed",
                    "message": "Сопоставление восстановлено из предыдущей версии.",
                }
                snapshot["current_stage"] = max(int(snapshot.get("current_stage", 1)), 3)
            if preview_manifest.is_file():
                preview = read_json(preview_manifest, {})
                snapshot["application_preview"] = preview
                snapshot["application_checkpoint"] = preview.get("checkpoint") or ""
                snapshot["application_background_checkpoint"] = (
                    preview.get("background_checkpoint") or ""
                )
            if full_manifest.is_file():
                snapshot["application_result"] = read_json(full_manifest, {})
            snapshot["stages"] = stages
            snapshot["warnings"] = []
            atomic_json(snapshot_path, snapshot)
        if not isinstance(snapshot, dict):
            raise RuntimeError("Снимок предыдущей версии повреждён.")

        allowed_directories = {
            "extracted", "alignment", "analysis", "application", "stems", "components",
            "previews", "dataset", "results", "reports",
        }
        selected_directories = [
            name for name in (variant.get("dirs") or [])
            if name in allowed_directories and (archive_dir / name).is_dir()
        ]
        if not selected_directories:
            raise RuntimeError("В предыдущей версии не найдены материалы для восстановления.")

        current_directories = (
            "extracted", "alignment", "analysis", "application", "stems", "components",
            "previews", "dataset", "results", "reports",
        )
        self._archive_pair_directories(
            pair,
            current_directories,
            reason="before_version_restore",
        )

        restored_directories: list[str] = []
        for name in selected_directories:
            source_dir = archive_dir / name
            destination = root / name
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            shutil.move(str(source_dir), str(destination))
            restored_directories.append(name)
        remaining_variants = [
            item for item in (pair.get("material_variants") or [])
            if str(item.get("id") or "") != str(variant_id)
        ]
        restored = copy.deepcopy(snapshot)
        restored["id"] = pair_id
        restored["project_id"] = project_id
        restored["material_variants"] = remaining_variants
        restored["restored_from_variant"] = {
            "id": variant_id,
            "restored_at": utc_now(),
        }
        restored["updated_at"] = utc_now()
        self.save_pair(restored)
        return restored

    def invalidate_pair_derivatives(
        self,
        project_id: str,
        pair_id: str,
        *,
        from_stage: int,
        reason: str,
    ) -> dict:
        """Archive and invalidate artefacts derived from a changed input stage."""
        if from_stage not in {2, 3, 4}:
            raise ValueError("Недопустимый этап инвалидирования материалов.")
        pair = self.load_pair(project_id, pair_id)
        if from_stage == 2:
            directories = (
                "extracted", "alignment", "analysis", "application", "stems", "components",
                "previews", "dataset", "results", "reports",
            )
        elif from_stage == 3:
            directories = (
                "alignment", "analysis", "application", "stems", "components", "previews",
                "dataset", "results", "reports",
            )
        else:
            directories = (
                "analysis", "application", "stems", "components", "previews", "dataset",
                "results", "reports",
            )
        self._archive_pair_directories(pair, directories, reason=reason)

        root = self.pair_dir(project_id, pair_id)
        for relative in (
            "extracted", "alignment", "analysis", "stems/original", "stems/dubbed",
            "components", "previews", "dataset", "results/baseline",
            "results/final", "reports", "application",
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

        if from_stage <= 2:
            pair.pop("extraction", None)
        if from_stage <= 3:
            pair.pop("alignment", None)
            pair.pop("alignment_manual_correction_sec", None)
        if from_stage <= 4:
            pair.pop("alignment_review", None)
        for key in (
            "speech_test", "speech_test_review", "original_speech",
            "target_speech", "components_test", "components_test_review",
            "components", "previews", "model_review", "application_preview",
            "application_result", "application_full_progress", "application_checkpoint",
            "application_background_checkpoint", "application_reused_preparation",
            "application_preparation",
            "reference_compatibility_analysis", "reference_compatibility_applied",
            "reference_compatibility_decision", "reference_compatibility_effective_model",
            "result",
        ):
            pair.pop(key, None)

        pair["current_stage"] = min(int(pair.get("current_stage", 1)), from_stage - 1)
        for number in range(from_stage, 11):
            pair.setdefault("stages", {})[str(number)] = {
                "status": "available" if number == from_stage else "blocked",
                "message": (
                    "Материалы изменились; этап нужно выполнить заново."
                    if number == from_stage
                    else "Сначала завершите предыдущий этап."
                ),
            }
        self.save_pair(pair)
        return pair

    def replace_pair_sources(
        self,
        project_id: str,
        pair_id: str,
        original_path: str | Path,
        dubbed_path: str | Path,
        original_stream: int,
        dubbed_stream: int,
        original_probe: dict,
        dubbed_probe: dict,
    ) -> dict:
        pair = self.load_pair(project_id, pair_id)
        next_sources = {
            "original": {
                "path": str(self.validate_media_path(original_path)),
                "stream_index": int(original_stream),
                "probe": original_probe,
            },
            "dubbed": {
                "path": str(self.validate_media_path(dubbed_path)),
                "stream_index": int(dubbed_stream),
                "probe": dubbed_probe,
            },
        }
        previous_sources = pair.get("sources") or {}
        changed_roles = [
            role
            for role in ("original", "dubbed")
            if
            self._source_identity(previous_sources.get(role))
            != self._source_identity(next_sources.get(role))
        ]
        if not changed_roles:
            pair["sources"] = next_sources
            self.save_pair(pair)
            return pair

        # Keep the already decoded side when only one selected source/stream
        # changed.  ``invalidate_pair_derivatives`` archives the whole
        # ``extracted`` directory to preserve the previous version, so move the
        # reusable files aside first and put them back afterwards.  These are
        # same-volume renames rather than multi-gigabyte copies.
        root = self.pair_dir(project_id, pair_id)
        reusable_roles = [
            role for role in ("original", "dubbed") if role not in changed_roles
        ]
        staging = root / f".extracted_reuse_{new_id()[:12]}"
        reusable_metadata: dict[str, Any] = {}
        extracted_metadata = read_json(root / "extracted" / "metadata.json", {}) or {}
        extracted_files = extracted_metadata.get("files") or {}
        for role in reusable_roles:
            audio = root / "extracted" / f"{role}.flac"
            proxy = root / "extracted" / f"{role}_proxy.wav"
            entry = extracted_files.get(role)
            if (
                not isinstance(entry, dict)
                or not audio.is_file()
                or audio.stat().st_size == 0
                or not proxy.is_file()
                or proxy.stat().st_size == 0
            ):
                continue
            staging.mkdir(parents=True, exist_ok=True)
            shutil.move(str(audio), str(staging / audio.name))
            shutil.move(str(proxy), str(staging / proxy.name))
            reusable_metadata[role] = copy.deepcopy(entry)

        # Archive *all* source-dependent outputs.  Previously components,
        # previews and final results survived a track change and the UI could
        # advertise files produced from a different stream.
        try:
            pair = self.invalidate_pair_derivatives(
                project_id,
                pair_id,
                from_stage=2,
                reason="track_source_changed",
            )
        finally:
            if staging.is_dir():
                restored_root = root / "extracted"
                restored_root.mkdir(parents=True, exist_ok=True)
                for role in tuple(reusable_metadata):
                    for name in (f"{role}.flac", f"{role}_proxy.wav"):
                        staged_file = staging / name
                        if staged_file.is_file():
                            shutil.move(str(staged_file), str(restored_root / name))
                shutil.rmtree(staging, ignore_errors=True)
                if reusable_metadata:
                    atomic_json(
                        restored_root / "metadata.json",
                        {
                            "schema_version": 2,
                            "prepared_at": utc_now(),
                            "partial": True,
                            "files": reusable_metadata,
                        },
                    )
        pair["sources"] = next_sources
        pair["stages"]["2"] = {
            "status": "available",
            "message": "Файлы заменены, аудио нужно извлечь заново.",
        }
        pair["warnings"] = [
            "Выбранные дорожки изменены; старые материалы перенесены в архив вариантов и больше не используются как текущие."
        ]
        self.save_pair(pair)
        return pair

    def create_task(
        self,
        project_id: str,
        operation: str,
        pair_id: str | None = None,
        parameters: dict | None = None,
    ) -> dict:
        self.load_project(project_id)
        if pair_id:
            self.load_pair(project_id, pair_id)
        task_id = new_id()
        task = {
            "schema_version": 2,
            "id": task_id,
            "project_id": project_id,
            "pair_id": pair_id,
            "operation": operation,
            "parameters": parameters or {},
            "state": "queued",
            "stage": "Ожидает запуска",
            "substage": "",
            "progress": 0.0,
            "local_progress": 0.0,
            "processed_seconds": 0.0,
            "total_seconds": 0.0,
            "eta_seconds": None,
            "current_file": "",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "heartbeat_at": None,
            "pid": None,
            "error": None,
            "result": None,
            "stop_file": str(
                self.project_dir(project_id) / "tasks" / f"{task_id}.stop"
            ),
            "log_file": str(
                self.project_dir(project_id) / "tasks" / f"{task_id}.log"
            ),
        }
        atomic_json(self.task_path(project_id, task_id), task)
        return task

    def load_task(self, project_id: str, task_id: str) -> dict:
        path = self.task_path(project_id, task_id)
        task = read_json(path)
        if not isinstance(task, dict):
            raise FileNotFoundError("Задача не найдена.")
        normalized, changed = self._rebase_payload_paths(
            task, project_id=project_id
        )
        expected_stop = str(path.with_suffix(".stop"))
        expected_log = str(path.with_suffix(".log"))
        if normalized.get("stop_file") != expected_stop:
            normalized["stop_file"] = expected_stop
            changed = True
        if normalized.get("log_file") != expected_log:
            normalized["log_file"] = expected_log
            changed = True
        if changed:
            atomic_json(path, normalized)
        return normalized

    def save_task(self, task: dict) -> None:
        atomic_json(self.task_path(task["project_id"], task["id"]), task)

    def list_tasks(
        self, project_id: str | None = None, states: set[str] | None = None
    ) -> list[dict]:
        paths = (
            (self.project_dir(project_id) / "tasks").glob("*.json")
            if project_id
            else self.projects_root.glob("*/tasks/*.json")
        )
        tasks: list[dict] = []
        for path in paths:
            try:
                stat = path.stat()
                cache_key = str(path)
                cached = self._task_read_cache.get(cache_key)
                if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                    task = cached[2]
                else:
                    task = read_json(path)
                    if isinstance(task, dict):
                        task, _changed = self._rebase_payload_paths(
                            task, project_id=task.get("project_id")
                        )
                        expected_stop = str(path.with_suffix(".stop"))
                        expected_log = str(path.with_suffix(".log"))
                        task["stop_file"] = expected_stop
                        task["log_file"] = expected_log
                        self._task_read_cache[cache_key] = (
                            stat.st_mtime_ns,
                            stat.st_size,
                            task,
                        )
            except (OSError, RuntimeError):
                continue
            if task and (not states or task.get("state") in states):
                tasks.append(dict(task))
        return sorted(tasks, key=lambda item: item.get("created_at", ""), reverse=True)

    def task_log_tail(self, task: dict) -> str:
        path = Path(task.get("log_file", ""))
        if not path.is_file():
            return ""
        maximum = int(self.cfg["tasks"].get("log_tail_lines", 300))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-maximum:])

    def pair_artifact(self, project_id: str, pair_id: str, relative: str) -> Path:
        root = self.pair_dir(project_id, pair_id).resolve()
        path = (root / relative).resolve()
        if not is_within(path, [root]):
            raise ValueError("Недопустимый путь артефакта.")
        return path

    def project_artifact(self, project_id: str, relative: str) -> Path:
        root = self.project_dir(project_id).resolve()
        path = (root / relative).resolve()
        if not is_within(path, [root]):
            raise ValueError("Недопустимый путь артефакта.")
        return path

    def migrate_component_pipeline_v3(self) -> dict:
        """Invalidate only post-alignment state from the obsolete pseudo-target flow."""
        changed = 0
        for path in self.projects_root.glob("*/pairs/*/pair.json"):
            pair = read_json(path)
            if not pair or int(pair.get("pipeline_version", 0)) >= 3:
                continue
            pair["schema_version"] = max(3, int(pair.get("schema_version", 2)))
            pair["pipeline_version"] = 3
            pair["legacy_post_alignment"] = {
                "baseline_result": pair.get("baseline_result"),
                "cancel_test": pair.get("cancel_test"),
                "note": "Сохранено только для истории; новая цепочка этапов 5–10 эти данные не использует.",
            }
            for key in (
                "baseline_result", "cancel_test", "speech_test", "original_speech",
                "components_test", "components", "preview_manifest", "model_review", "result",
            ):
                pair.pop(key, None)
            stage4_complete = pair.get("stages", {}).get("4", {}).get("status") == "completed"
            for number in range(5, 11):
                pair.setdefault("stages", {})[str(number)] = {
                    "status": "available" if number == 5 and stage4_complete else "blocked",
                    "message": (
                        "Карта подтверждена. Начните с пробного выделения EN-речи."
                        if number == 5 and stage4_complete
                        else "Сначала завершите предыдущий этап новой компонентной цепочки."
                    ),
                }
            pair["current_stage"] = min(int(pair.get("current_stage", 4)), 4)
            atomic_json(path, pair)
            changed += 1
        return {"changed": changed, "pipeline_version": 3}

    def migrate_speech_stem_pipeline_v4(self) -> dict:
        """Invalidate only post-alignment state from the superseded v3 chain."""
        changed = 0
        for project in self.list_projects():
            for listed in self.list_pairs(project["id"], include_deleted=True):
                pair = self.load_pair(project["id"], listed["id"])
                if int(pair.get("pipeline_version", 0)) >= 4:
                    continue
                legacy = pair.setdefault("legacy_post_alignment", {})
                for key in (
                    "speech_test",
                    "speech_test_review",
                    "components_test",
                    "components_test_review",
                    "components",
                    "previews",
                    "model_review",
                    "final",
                ):
                    if key in pair:
                        legacy[f"v3_{key}"] = pair.pop(key)
                pair["pipeline_version"] = 4
                stage4_ready = (
                    pair.get("stages", {}).get("4", {}).get("status") == "completed"
                )
                pair["stages"]["5"] = {
                    "status": "available" if stage4_ready else "blocked",
                    "message": (
                        "Запустите пробное разделение оригинала на EN-речь и M&E."
                        if stage4_ready
                        else "Сначала подтвердите сопоставление."
                    ),
                }
                for number, message in (
                    ("6", "Сначала завершите этап 5."),
                    ("7", "Сначала завершите этап 6."),
                    ("8", "Сначала завершите предпросмотры."),
                    ("9", "Сначала подготовьте аудиоматериалы."),
                    ("10", "Сначала завершите речевую цепочку."),
                ):
                    pair["stages"][number] = {"status": "blocked", "message": message}
                self.save_pair(pair)
                changed += 1
        return {"changed": changed, "pipeline_version": 4}

    def migrate_method_comparison_pipeline_v5(self) -> dict:
        """Reset stage 6+ for five-demo A/B selection without repeating stage 5."""
        changed = 0
        for project in self.list_projects():
            for listed in self.list_pairs(project["id"], include_deleted=True):
                pair = self.load_pair(project["id"], listed["id"])
                if int(pair.get("pipeline_version", 0)) >= 5:
                    continue
                legacy = pair.setdefault("legacy_post_alignment", {})
                for key in (
                    "components_test",
                    "components_test_review",
                    "components",
                    "previews",
                    "model_review",
                    "result",
                ):
                    if key in pair:
                        legacy[f"v4_{key}"] = pair.pop(key)
                pair["pipeline_version"] = 5
                stage4_ready = (
                    pair.get("stages", {}).get("4", {}).get("status")
                    == "completed"
                )
                stage5_trial_ready = bool(
                    pair.get("speech_test_review", {}).get("accepted")
                )
                pair["stages"]["6"] = {
                    "status": (
                        "available"
                        if stage4_ready and stage5_trial_ready
                        else "blocked"
                    ),
                    "message": (
                        "Создайте пять разговорных фрагментов и сравните методы A и B."
                        if stage4_ready and stage5_trial_ready
                        else "Сначала подтвердите пробное разделение оригинала на этапе 5."
                    ),
                }
                for number, message in (
                    ("7", "Сначала выберите метод и запустите сборку."),
                    ("8", "Сначала завершите предпросмотры."),
                    ("9", "Сначала подготовьте аудиоматериалы."),
                    ("10", "Сначала выполните сборку выбранным методом."),
                ):
                    pair["stages"][number] = {
                        "status": "blocked",
                        "message": message,
                    }
                pair["current_stage"] = min(int(pair.get("current_stage", 5)), 5)
                self.save_pair(pair)
                changed += 1
        return {"changed": changed, "pipeline_version": 5}

    def migrate_legacy_runs(self) -> dict:
        marker = self.runtime_root / "legacy_migration_v2.json"
        if marker.is_file():
            return read_json(marker, {}) or {}
        runs = sorted(self.legacy_outputs.glob("run_*/metadata.json"))
        report = {
            "created_at": utc_now(),
            "found": len(runs),
            "imported": 0,
            "note": "Старые результаты не перемещались и не удалялись.",
        }
        if runs:
            project = self.create_project("Архив прежних одиночных запусков")
            project_root = self.project_dir(project["id"])
            index: list[dict] = []
            for metadata_path in runs:
                try:
                    metadata = read_json(metadata_path)
                except RuntimeError as exc:
                    index.append({"path": str(metadata_path), "error": str(exc)})
                    continue
                index.append(
                    {
                        "path": str(metadata_path.parent.resolve()),
                        "metadata": metadata,
                    }
                )
                report["imported"] += 1
            atomic_json(project_root / "reports" / "legacy_runs.json", {"runs": index})
            report["project_id"] = project["id"]
        atomic_json(marker, report)
        return report


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)
