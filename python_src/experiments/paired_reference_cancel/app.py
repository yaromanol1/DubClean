"""Local DubClean web application and persistent processing API."""
from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from array import array
from pathlib import Path
from typing import Any

import yaml
import soundfile as sf
from flask import Flask, Response, jsonify, redirect, request, send_file

MODULE_DIR = Path(__file__).resolve().parent
PORTABLE_ROOT = MODULE_DIR.parents[2]
CONFIG_PATH = MODULE_DIR / "config.yaml"


def _same_local_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _ensure_portable_config() -> None:
    """Rebase paths if this backend was started directly after relocation."""
    try:
        raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        configured_models = str((raw_config.get("paths") or {}).get("models") or "")
    except (OSError, UnicodeError, yaml.YAMLError):
        configured_models = ""
    if configured_models:
        # Package-internal paths are stored relative to the package root, so a
        # relative value is already correct and must not trigger a rewrite.
        candidate = Path(configured_models)
        if not candidate.is_absolute():
            candidate = PORTABLE_ROOT / candidate
        if _same_local_path(candidate.resolve(), (PORTABLE_ROOT / "models").resolve()):
            return

    configure_script = PORTABLE_ROOT / "portable_tools" / "configure_portable.py"
    result = subprocess.run([sys.executable, str(configure_script)], cwd=str(PORTABLE_ROOT))
    if result.returncode:
        raise RuntimeError("Не удалось настроить переносимые пути DubClean.")


_ensure_portable_config()

PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paired_reference_cancel.audio_io import (  # noqa: E402
    check_ffmpeg,
    probe_audio_streams,
    probe_duration,
    probe_video_stream,
    safe_filename,
)
from experiments.paired_reference_cancel.automatic_assembly import (  # noqa: E402
    normalize_policy,
)
from experiments.paired_reference_cancel.service_control import (  # noqa: E402
    find_module_instances,
    kill_verified,
    pid_matches,
    stop_running_tasks,
)
from experiments.paired_reference_cancel.pipeline import (  # noqa: E402
    estimate_sparse_manual_correction,
)
from experiments.paired_reference_cancel.reference_compatibility import (  # noqa: E402
    apply_reference_compatibility_recommendations,
)
from experiments.paired_reference_cancel import audio_mix, song_protection  # noqa: E402
from experiments.paired_reference_cancel.storage import (  # noqa: E402
    Store,
    atomic_json,
    is_within,
    read_json,
    root_path,
    utc_now,
)
from src.audio_extract import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402

cfg_store = Store()
cfg = cfg_store.cfg
cfg_store.migrate_legacy_runs()
cfg_store.migrate_component_pipeline_v3()
cfg_store.migrate_speech_stem_pipeline_v4()
cfg_store.migrate_method_comparison_pipeline_v5()
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
RUNTIME_DIR = cfg_store.runtime_root
PID_FILE = RUNTIME_DIR / "web.pid"
USER_SETTINGS_FILE = RUNTIME_DIR / "user_settings.json"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = int(cfg["web"]["max_upload_mb"]) * 1024 * 1024
_scheduler_lock = threading.Lock()
_task_mutation_lock = threading.RLock()
_scheduler_stop = threading.Event()
_scheduler_thread: threading.Thread | None = None
_processes: dict[str, subprocess.Popen] = {}
_clip_lock = threading.Lock()
_waveform_lock = threading.Lock()
_waveform_cache: dict[str, dict[str, Any]] = {}
_dataset_preview_lock = threading.Lock()
_preview_mix_lock = threading.Lock()
_browser_video_lock = threading.Lock()
_application_catalog_lock = threading.Lock()
_application_catalog_cache: dict[str, Any] | None = None
_application_catalog_cached_at = 0.0
_APPLICATION_CATALOG_TTL_SEC = 5.0
_application_detail_lock = threading.Lock()
_application_detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_APPLICATION_DETAIL_TTL_SEC = 1.25
# Per-segment locks: two requests for the SAME segment dedup onto one transcode,
# but different segments (e.g. the shared picture and the swappable audio, or a
# seek target and its neighbours) transcode in parallel instead of queuing
# behind a single global lock — that queuing was what made seeks slow and let
# the picture arrive well before its audio.
_hls_segment_locks: dict[str, threading.Lock] = {}
_hls_segment_locks_guard = threading.Lock()


def _hls_key_lock(cache_key: str) -> threading.Lock:
    with _hls_segment_locks_guard:
        lock = _hls_segment_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _hls_segment_locks[cache_key] = lock
            # Keep the map from growing without bound over a long session.
            if len(_hls_segment_locks) > 4096:
                for stale_key in list(_hls_segment_locks)[:2048]:
                    if stale_key != cache_key:
                        _hls_segment_locks.pop(stale_key, None)
        return lock


def _is_loopback_hostname(hostname: str | None) -> bool:
    value = str(hostname or "").strip().strip("[]").casefold()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


@app.before_request
def enforce_local_request_origin():
    """Reject DNS-rebinding and cross-origin writes to the local API."""
    request_host = urllib.parse.urlsplit(f"//{request.host}").hostname
    if not _is_loopback_hostname(request_host):
        return jsonify({"error": "DubClean доступен только локально."}), 403
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = str(request.headers.get("Origin") or "").strip()
        if origin:
            origin_host = urllib.parse.urlsplit(origin).hostname
            if not _is_loopback_hostname(origin_host):
                return jsonify({"error": "Запрос с внешнего сайта отклонён."}), 403
    return None


@app.after_request
def disable_interface_cache(response):
    """Prevent an old HTML/JS pair from leaving the panel half-initialized."""
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        application_match = re.match(
            r"^/api/applications/([0-9a-f]+)(?:/|$)", request.path
        )
        if application_match:
            _invalidate_application_detail(application_match.group(1))
            _invalidate_application_catalog()
    no_cache_paths = {"/", "/training", "/apply", "/product", "/product/", "/support.js"}
    no_cache_extensions = (".html", ".js", ".css")
    if (
        request.path in no_cache_paths
        or request.path.startswith("/product/")
        or request.path.startswith("/static/")
        or request.path.endswith(no_cache_extensions)
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


@app.errorhandler(413)
def upload_too_large(_error):
    return (
        jsonify(
            {
                "error": (
                    f"Файл превышает ограничение {int(cfg['web']['max_upload_mb'])} МБ. "
                    "Для полнометражного файла укажите путь внутри разрешённой папки."
                )
            }
        ),
        413,
    )


@app.errorhandler(ValueError)
def bad_value(error):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(FileNotFoundError)
def not_found(error):
    return jsonify({"error": str(error)}), 404


@app.errorhandler(RuntimeError)
def runtime_error(error):
    return jsonify({"error": str(error)}), 409


def _kill_task_process(task: dict) -> bool:
    """Kill a task's worker tree without ever touching a recycled PID.

    A Popen handle owned by this panel keeps the PID reserved, so it is safe
    to kill directly; a PID recovered from a task file (e.g. after a panel
    restart) must first be verified against the worker's command line.
    """
    process = _processes.get(task["id"])
    if process is not None and process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        finally:
            _processes.pop(task["id"], None)
        return process.poll() is not None
    killed = kill_verified(task.get("pid"), "task_worker.py", str(task["id"]))
    verdict = pid_matches(task.get("pid"), "task_worker.py", str(task["id"]))
    return bool(killed or verdict is False)


def _stop_task_record(
    task: dict,
    *,
    state: str = "stopped",
    substage: str = "Остановлено пользователем.",
    error: str | None = None,
) -> dict:
    stop_file = Path(task["stop_file"])
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("stop", encoding="utf-8")
    # Give the worker a short cooperative window to close temporary writers
    # and persist a phase that completed immediately before Stop was pressed.
    # Long-running model subprocesses are still force-killed below if they do
    # not observe the marker promptly.
    process = _processes.get(task["id"])
    deadline = time.monotonic() + min(
        2.0, float(cfg.get("tasks", {}).get("stop_timeout_sec", 8.0))
    )
    while process is not None and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    stopped = _kill_task_process(task)
    if not stopped:
        raise RuntimeError(
            "Не удалось безопасно подтвердить остановку фонового процесса. "
            "Повторите остановку или перезапустите сервис."
        )
    task.update(
        {
            "state": state,
            "stage": "Остановлено",
            "substage": substage,
            "finished_at": utc_now(),
            "heartbeat_at": utc_now(),
            "eta_seconds": None,
            "pid": None,
            "error": error,
        }
    )
    cfg_store.save_task(task)
    return task


def _recover_tasks() -> None:
    for task in cfg_store.list_tasks(states={"running", "starting"}):
        process = _processes.get(task["id"])
        owned_alive = process is not None and process.poll() is None
        identity = (
            True
            if owned_alive
            else pid_matches(task.get("pid"), "task_worker.py", str(task["id"]))
        )
        # If Windows cannot read the command line, leave the task alone rather
        # than either killing an unrelated recycled PID or falsely declaring a
        # still-running worker interrupted.
        if identity is False:
            task.update(
                {
                    "state": "interrupted",
                    "stage": "Прервано",
                    "substage": "Процесс не найден после перезапуска сервиса. Задачу можно запустить повторно.",
                    "finished_at": utc_now(),
                    "error": "Фоновый процесс завершился без финального статуса.",
                }
            )
            cfg_store.save_task(task)


def _task_pair_ids(task: dict) -> set[str]:
    pair_ids: set[str] = set()
    if task.get("pair_id"):
        pair_ids.add(str(task["pair_id"]))
    parameters = task.get("parameters") or {}
    pair_ids.update(str(item) for item in (parameters.get("pair_ids") or []) if item)
    return pair_ids


def _active_tasks_for_pair(project_id: str, pair_id: str) -> list[dict]:
    return [
        task
        for task in cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
        if pair_id in _task_pair_ids(task)
    ]


def _task_conflicts(task: dict, active: list[dict]) -> bool:
    if task["operation"] == "train" and any(item["operation"] == "train" for item in active):
        return True
    # Full-film adaptive cancellation is deliberately serialized. Two workers
    # otherwise compete for RAM and FFT CPU, become dramatically slower and
    # look stalled even though both processes are still alive.
    if task["operation"] == "cancel" and any(
        item["operation"] == "cancel" for item in active
    ):
        return True
    speech_gpu_operations = {"speech_test", "speech_extract", "target_speech_extract"}
    if task["operation"] in speech_gpu_operations:
        if sum(item["operation"] in speech_gpu_operations for item in active) >= 3:
            return True
    else:
        gpu_operations = {
            "components_test",
            "components_build",
            "train",
            "application_tracks",
            "application_prepare",
            "application_full",
            "application_auto",
        }
        if task["operation"] in gpu_operations and any(
            item["operation"] in gpu_operations or item["operation"] in speech_gpu_operations
            for item in active
        ):
            return True
    pair_ids = _task_pair_ids(task)
    if pair_ids and any(
        item.get("project_id") == task["project_id"]
        and pair_ids.intersection(_task_pair_ids(item))
        for item in active
    ):
        return True
    return False


def _start_task(task: dict) -> None:
    command = [
        sys.executable,
        str(MODULE_DIR / "task_worker.py"),
        "--project-id",
        task["project_id"],
        "--task-id",
        task["id"],
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        creationflags=flags,
    )
    _processes[task["id"]] = process
    task.update(
        {
            "state": "starting",
            "pid": process.pid,
            "started_at": utc_now(),
            "heartbeat_at": utc_now(),
            "stage": "Запуск фонового процесса",
        }
    )
    cfg_store.save_task(task)


def scheduler_tick() -> None:
    if not _scheduler_lock.acquire(blocking=False):
        return
    try:
        with _task_mutation_lock:
            for task_id, process in list(_processes.items()):
                if process.poll() is not None:
                    process.wait()
                    _processes.pop(task_id, None)
            _recover_tasks()
            active = cfg_store.list_tasks(states={"running", "starting"})
            maximum = int(cfg["tasks"]["max_parallel_pair_tasks"])
            for task in reversed(cfg_store.list_tasks(states={"queued"})):
                if len(active) >= maximum:
                    break
                if _task_conflicts(task, active):
                    continue
                _start_task(task)
                active.append(task)
    finally:
        _scheduler_lock.release()


def _scheduler_loop() -> None:
    interval = float(cfg["tasks"]["poll_interval_sec"])
    while not _scheduler_stop.wait(interval):
        scheduler_tick()


def _start_scheduler() -> None:
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
    _scheduler_thread.start()


def _allow_existing_media_path(path_value: str) -> None:
    """Allow a user-picked local media file without copying it into uploads."""
    if not path_value:
        return
    candidate = Path(path_value).expanduser().resolve()
    if candidate.is_file() and not is_within(candidate, cfg_store.allowed_media_roots):
        # Keep the permission across a normal portable-service restart.  An
        # in-memory append made file previews disappear for an already saved
        # project after restarting the panel.
        cfg_store.authorize_media_dir(candidate)


def _restore_saved_source_roots() -> None:
    """Re-authorize existing project sources after migrating older metadata.

    Projects created before persistent source-root authorization may contain
    valid files outside data/uploads.  Only existing files recorded in project
    metadata are restored; arbitrary request paths are never trusted here.
    """
    for project in cfg_store.list_projects():
        project_id = str(project.get("id") or "")
        if not project_id:
            continue
        try:
            pairs = cfg_store.list_pairs(project_id)
        except (OSError, RuntimeError, ValueError):
            continue
        for pair in pairs:
            for source in (pair.get("sources") or {}).values():
                if isinstance(source, dict):
                    _allow_existing_media_path(str(source.get("path") or ""))


def _set_projects_root(path_value: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    settings = read_json(USER_SETTINGS_FILE, {}) or {}
    settings["projects_root"] = str(path)
    atomic_json(USER_SETTINGS_FILE, settings)

    cfg["paths"]["projects"] = path.as_posix()
    CONFIG_PATH.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    cfg_store.projects_root = path
    return path


def _probe(path_value: str) -> tuple[Path, dict]:
    _allow_existing_media_path(path_value)
    path = cfg_store.validate_media_path(path_value)
    streams = probe_audio_streams(path)
    if not streams:
        raise ValueError("В файле нет аудиопотоков.")
    stat = path.stat()
    return path, {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "duration_sec": probe_duration(path),
        "streams": streams,
        # The picture is what the two sources may differ in most: one release
        # can be a 1200x510 H.264 and the other a 720x304 XviD.  Without this
        # the operator has no way to see which file should provide the video.
        "video": probe_video_stream(path),
    }


def _validate_stream(probe: dict, value: Any, label: str) -> int:
    try:
        stream_index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Номер аудиопотока {label} должен быть целым.") from exc
    if stream_index not in {int(item["index"]) for item in probe["streams"]}:
        raise ValueError(f"Выбранный аудиопоток {label} отсутствует в файле.")
    return stream_index


def _public_config() -> dict:
    return {
        "port": int(cfg["web"]["port"]),
        "paths": {
            "projects": str(cfg_store.projects_root),
            "uploads": str(cfg_store.uploads_root),
            "models": str(cfg_store.models_root),
            "runtime": str(cfg_store.runtime_root),
        },
        "allowed_media_roots": [str(item) for item in cfg_store.allowed_media_roots],
        "training": {
            "backend": cfg["training"]["backend"],
            "short_epoch_choices": cfg["training"]["short_epoch_choices"],
            "full_epoch_choices": cfg["training"]["full_epoch_choices"],
        },
        "preview_duration_sec": cfg["audio"]["preview_duration_sec"],
        "stage_titles": {
            "1": "Добавление пар фильмов",
            "2": "Полное извлечение аудиодорожек",
            "3": "Автоматическое сопоставление",
            "4": "Проверка сопоставления",
            "5": "Разделение оригинала на EN-речь и M&E",
            "6": "Очистка RU-голоса и сборка с оригинальным M&E",
            "7": "Предпросмотр компонентов",
            "8": "Подготовка аудиоматериалов",
            "9": "Проверка готовой модели",
            "10": "Финальная сборка RU + M&E",
        },
    }


def _heartbeat_is_stale(task: dict) -> bool:
    if task.get("state") != "running":
        return False
    heartbeat = task.get("heartbeat_at")
    if not heartbeat:
        return False
    import calendar

    try:
        beat = calendar.timegm(time.strptime(heartbeat, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return False
    return time.time() - beat > float(cfg["tasks"].get("stale_heartbeat_sec", 45))


def _task_with_log(task: dict, *, include_result: bool = False) -> dict:
    value = dict(task)
    if not include_result:
        result = value.pop("result", None)
        if isinstance(result, dict):
            value["result_available"] = True
            value["result_keys"] = sorted(result)
    value["log"] = cfg_store.task_log_tail(task)
    value["stalled"] = _heartbeat_is_stale(task)
    return value


def _existing_file_in(path_value: Any, roots: list[Path]) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    try:
        path = Path(path_value).resolve()
        if not path.is_file() or path.stat().st_size <= 0 or not is_within(path, roots):
            return None
    except (OSError, ValueError):
        return None
    return path


def _sanitize_application_preview(
    pair: dict, root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = pair.get("application_preview")
    if not isinstance(payload, dict):
        return None, []
    missing: list[str] = []
    manifest_path = root / "application" / "preview_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        return None, [str(manifest_path)]
    value = copy.deepcopy(payload)
    valid_scenes: list[dict[str, Any]] = []
    for scene in value.get("scenes") or []:
        if not isinstance(scene, dict):
            continue
        files = scene.get("files") or {}
        if not isinstance(files, dict):
            continue
        valid_files: dict[str, str] = {}
        for key, path_value in files.items():
            path = _existing_file_in(path_value, [root])
            if path is None:
                if path_value:
                    missing.append(str(path_value))
                continue
            valid_files[str(key)] = str(path)
        playable_audio = any(
            valid_files.get(key)
            for key in (
                "rebuilt_result", "synchronized_rebuilt_result",
                "model_ru_voice", "direct_dirty_result", "dubbed_mix",
            )
        )
        if not valid_files.get("video_preview") or not playable_audio:
            continue
        scene["files"] = valid_files
        valid_scenes.append(scene)
    if not valid_scenes:
        return None, missing or ["Файлы сцен превью отсутствуют."]
    value["scenes"] = valid_scenes
    if (
        int(value.get("schema_version") or 0) >= 5
        or value.get("processing_scope") == "speech_and_song_scenes"
    ):
        song_payload = value.get("song_protection")
        if not isinstance(song_payload, dict):
            return None, missing or ["Отчёт защиты песен для превью отсутствует."]
        report_value = song_payload.get("report")
        report_path = _existing_file_in(report_value, [root])
        if report_path is None:
            if report_value:
                missing.append(str(report_value))
            return None, missing or ["Отчёт защиты песен для превью отсутствует."]
        song_payload["report"] = str(report_path)
        value["song_protection"] = song_payload
    return value, missing


def _sanitize_application_result(
    pair: dict, root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = pair.get("application_result")
    if not isinstance(payload, dict):
        return None, []
    value = copy.deepcopy(payload)
    missing: list[str] = []
    for key in (
        "audio",
        "internal_audio",
        "audio_without_speech_synchronization",
        "movie",
        "movie_without_speech_synchronization",
        "manifest",
    ):
        path_value = value.get(key)
        if not path_value:
            continue
        path = _existing_file_in(path_value, [root])
        if path is None:
            missing.append(str(path_value))
            value.pop(key, None)
        else:
            value[key] = str(path)
    intermediates = value.get("intermediates") or {}
    if isinstance(intermediates, dict):
        valid_intermediates: dict[str, Any] = {}
        for key, path_value in intermediates.items():
            path = _existing_file_in(
                path_value, [root, cfg_store.projects_root, cfg_store.models_root]
            )
            if path is None:
                if path_value:
                    missing.append(str(path_value))
                continue
            valid_intermediates[str(key)] = str(path)
        value["intermediates"] = valid_intermediates
    user_files = value.get("user_files") or {}
    if isinstance(user_files, dict):
        valid_user_files: dict[str, str] = {}
        for key, path_value in user_files.items():
            path = _existing_file_in(path_value, [root])
            if path is None:
                if path_value:
                    missing.append(str(path_value))
                continue
            valid_user_files[str(key)] = str(path)
        value["user_files"] = valid_user_files
    if not value.get("audio") and not value.get("movie"):
        return None, missing or ["Итоговые файлы отсутствуют."]
    return value, missing


def _sanitize_application_full_progress(
    pair: dict, root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = pair.get("application_full_progress")
    if not isinstance(payload, dict):
        return None, []
    value = copy.deepcopy(payload)
    missing: list[str] = []
    for key in ("audio", "movie"):
        path_value = value.get(key)
        if not path_value:
            continue
        path = _existing_file_in(path_value, [root])
        if path is None:
            missing.append(str(path_value))
            value.pop(key, None)
        else:
            value[key] = str(path)
    valid_intermediates: dict[str, str] = {}
    for key, path_value in (value.get("intermediates") or {}).items():
        path = _existing_file_in(path_value, [root])
        if path is None:
            if path_value:
                missing.append(str(path_value))
            continue
        valid_intermediates[str(key)] = str(path)
    value["intermediates"] = valid_intermediates
    valid_user_files: dict[str, str] = {}
    for key, path_value in (value.get("user_files") or {}).items():
        path = _existing_file_in(path_value, [root])
        if path is None:
            if path_value:
                missing.append(str(path_value))
            continue
        valid_user_files[str(key)] = str(path)
    value["user_files"] = valid_user_files
    if not valid_intermediates and not value.get("audio") and not value.get("movie"):
        # Keep the running descriptor so the UI can render disabled expected
        # rows before the first full-length artifact is ready.
        if value.get("state") != "running":
            return None, missing
    return value, missing


def _sanitize_application_preparation(
    pair: dict, root: Path
) -> tuple[dict[str, Any] | None, list[str]]:
    payload = pair.get("application_preparation")
    if not isinstance(payload, dict):
        return None, []
    value = copy.deepcopy(payload)
    missing: list[str] = []
    valid_phases: dict[str, Any] = {}
    for phase, phase_value in (value.get("phases") or {}).items():
        if not isinstance(phase_value, dict) or phase_value.get("status") != "completed":
            continue
        artifacts = phase_value.get("artifacts") or {}
        if not isinstance(artifacts, dict) or not artifacts:
            continue
        valid_artifacts: dict[str, str] = {}
        for key, path_value in artifacts.items():
            path = _existing_file_in(path_value, [root])
            if path is None:
                if path_value:
                    missing.append(str(path_value))
                continue
            valid_artifacts[str(key)] = str(path)
        # A phase is complete only while every artifact recorded at completion
        # is still present.  This prevents stale metadata from hiding a deleted
        # or interrupted output.
        if len(valid_artifacts) != len(artifacts):
            continue
        phase_copy = copy.deepcopy(phase_value)
        phase_copy["artifacts"] = valid_artifacts
        valid_phases[str(phase)] = phase_copy
    value["phases"] = valid_phases
    return value, missing


def _discover_application_preparation(root: Path) -> dict[str, Any] | None:
    """Recover phase state for projects stopped before phase metadata existed."""
    scan_root = root / "application" / "candidate_scan"
    phases: dict[str, Any] = {}

    def completed_phase(artifacts: dict[str, Path]) -> dict[str, Any] | None:
        resolved: dict[str, str] = {}
        for key, path in artifacts.items():
            valid = _existing_file_in(str(path), [root])
            if valid is None:
                return None
            resolved[key] = str(valid)
        return {
            "status": "completed",
            "completed_at": None,
            "artifacts": resolved,
            "recovered": True,
        }

    speech = completed_phase(
        {
            "original_en_speech": scan_root / "selected_original_speech.flac",
            "original_background_speech": scan_root / "selected_original_background_speech.flac",
            "dubbed_en_ru_speech": scan_root / "selected_dubbed_speech.flac",
            "original_mix": scan_root / "selected_original_mix.flac",
            "dubbed_mix": scan_root / "selected_dubbed_mix.flac",
        }
    )
    if speech:
        phases["speech"] = speech

    model_candidates = sorted(
        scan_root.glob("model_results/selection_v*/**/model_ru_voice.flac"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for voice in model_candidates:
        model_root = voice.parent
        subtraction = completed_phase(
            {
                "model_ru_voice": voice,
                "model_ru_voice_algorithmic": (
                    model_root / "model_ru_voice_algorithmic_reference.flac"
                ),
                "direct_dirty_result": model_root / "model_direct_dirty_mix.flac",
            }
        )
        if subtraction:
            phases["subtraction"] = subtraction
            break

    background_candidates = sorted(
        list(scan_root.glob("background_results/selection_v*/**/learned_music_effects.flac"))
        + list(scan_root.glob("specialized_separator/**/instrumental*.flac")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if background_candidates:
        background = completed_phase(
            {"specialized_me": background_candidates[0]}
        )
        if background:
            phases["background"] = background

    if not phases:
        return None
    return {
        "schema_version": 1,
        "identity": {"recovered": True},
        "phases": phases,
        "updated_at": utc_now(),
        "recovered": True,
    }


def _backfill_source_video_probe(pair: dict) -> bool:
    """Describe each source's picture once, for projects created before that.

    The choice of which file supplies the video cannot be offered until the
    operator can see what the two pictures are.  Probing is a local ffprobe
    call, so it is done once and persisted rather than on every request.
    """
    changed = False
    for role in ("original", "dubbed"):
        source = (pair.get("sources") or {}).get(role) or {}
        probe = source.get("probe")
        if not isinstance(probe, dict) or "video" in probe:
            continue
        path_value = str(source.get("path") or "")
        if not path_value or not Path(path_value).is_file():
            continue
        try:
            probe["video"] = probe_video_stream(path_value)
        except (OSError, RuntimeError):
            continue
        changed = True
    return changed


def _pair_detail(project_id: str, pair_id: str) -> dict:
    pair = cfg_store.load_pair(project_id, pair_id)
    root = cfg_store.pair_dir(project_id, pair_id)
    if _backfill_source_video_probe(pair):
        cfg_store.save_pair(pair)
    if not pair.get("application_preparation"):
        recovered_preparation = _discover_application_preparation(root)
        if recovered_preparation:
            pair["application_preparation"] = recovered_preparation
            active = cfg_store.list_tasks(
                project_id, states={"queued", "starting", "running"}
            )
            if not active:
                cfg_store.save_pair(pair)
    detail = copy.deepcopy(pair)
    stages = detail.setdefault("stages", {})
    required_extracted = [
        root / "extracted" / name
        for name in (
            "original.flac", "dubbed.flac", "original_proxy.wav",
            "dubbed_proxy.wav", "metadata.json",
        )
    ]
    stage2_ready = (
        stages.get("2", {}).get("status") == "completed"
        and all(path.is_file() and path.stat().st_size > 0 for path in required_extracted)
    )
    if stages.get("2", {}).get("status") == "completed" and not stage2_ready:
        stages["2"] = {
            "status": "available",
            "message": "Извлечённые дорожки отсутствуют; выполните извлечение заново.",
        }
        for number in range(3, 11):
            stages[str(number)] = {
                "status": "blocked",
                "message": "Сначала заново извлеките дорожки.",
            }
        detail.pop("extraction", None)
    alignment_map_path = root / "alignment" / "alignment_map.json"
    stage3_ready = (
        stage2_ready
        and stages.get("3", {}).get("status") == "completed"
        and alignment_map_path.is_file()
        and alignment_map_path.stat().st_size > 0
    )
    if stage2_ready and stages.get("3", {}).get("status") == "completed" and not stage3_ready:
        stages["3"] = {
            "status": "available",
            "message": "Карта сопоставления отсутствует; постройте её заново.",
        }
        for number in range(4, 11):
            stages[str(number)] = {
                "status": "blocked",
                "message": "Сначала заново постройте карту сопоставления.",
            }
        detail.pop("alignment", None)
    files: list[dict] = []
    for path in root.glob("**/*"):
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in {".flac", ".wav", ".json", ".mkv"}
        ):
            relative = str(path.relative_to(root)).replace("\\", "/")
            if relative.startswith("material_variants/"):
                continue
            if relative.startswith("extracted/") and not stage2_ready:
                continue
            if relative.startswith("alignment/") and not stage3_ready:
                continue
            files.append(
                {
                    "name": path.name,
                    "relative": relative,
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "media": path.suffix.lower() in {".flac", ".wav", ".mkv"},
                }
            )
    detail["files"] = files
    extracted_files: list[dict[str, Any]] = []
    if stage2_ready:
        for role, label in (
            ("original", "Оригинальная дорожка"),
            ("dubbed", "Дорожка RU+EN"),
        ):
            extracted = root / "extracted" / f"{role}.flac"
            if extracted.is_file():
                extracted_files.append(
                    {
                        "role": role,
                        "label": label,
                        "path": str(extracted.resolve()),
                        "name": extracted.name,
                        "size": extracted.stat().st_size,
                    }
                )
    detail["extracted_files"] = extracted_files
    artifact_warnings: list[str] = []
    preview, preview_missing = _sanitize_application_preview(detail, root)
    result, result_missing = _sanitize_application_result(detail, root)
    full_progress, full_progress_missing = _sanitize_application_full_progress(
        detail, root
    )
    preparation, preparation_missing = _sanitize_application_preparation(detail, root)
    if preview is None:
        detail.pop("application_preview", None)
    else:
        detail["application_preview"] = preview
    if result is None:
        detail.pop("application_result", None)
    else:
        detail["application_result"] = result
    if full_progress is None:
        detail.pop("application_full_progress", None)
    else:
        detail["application_full_progress"] = full_progress
    if preparation is None:
        detail.pop("application_preparation", None)
    else:
        detail["application_preparation"] = preparation
    artifact_warnings.extend(preview_missing)
    artifact_warnings.extend(result_missing)
    artifact_warnings.extend(full_progress_missing)
    artifact_warnings.extend(preparation_missing)
    if artifact_warnings:
        detail["missing_artifacts"] = artifact_warnings
    for key, relative, enabled in (
        ("alignment_map", "alignment/alignment_map.json", stage3_ready),
        ("speech_test_manifest", "previews/speech_test/manifest.json", bool(pair.get("speech_test"))),
        ("original_speech_manifest", "stems/original/manifest.json", bool(pair.get("original_speech"))),
        ("components_test_manifest", "previews/components_test/manifest.json", bool(pair.get("components_test"))),
        ("components_manifest", "components/manifest.json", bool(pair.get("components"))),
        ("preview_manifest", "previews/preview_manifest.json", bool(pair.get("previews"))),
    ):
        if not enabled:
            continue
        path = root / relative
        if path.is_file():
            try:
                detail[key] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                detail[key] = {"error": f"{path.name} повреждён."}
    detail["tasks"] = [
        _task_with_log(item)
        for item in cfg_store.list_tasks(project_id)
        if item.get("pair_id") == pair_id
    ][:20]
    material_versions: list[dict[str, Any]] = []
    variants_root = (root / "material_variants").resolve()
    raw_material_variants = detail.pop("material_variants", []) or []
    for variant in reversed(raw_material_variants):
        archive_dir = Path(str(variant.get("path") or "")).resolve()
        if not archive_dir.is_dir() or not is_within(archive_dir, [variants_root]):
            continue
        sources = variant.get("sources") or {}
        original = sources.get("original") or {}
        dubbed = sources.get("dubbed") or {}
        readiness = variant.get("readiness")
        if not isinstance(readiness, dict):
            manifest = read_json(archive_dir / "archive_manifest.json", {}) or {}
            readiness = manifest.get("readiness")
        if not isinstance(readiness, dict):
            snapshot = read_json(archive_dir / "pair_snapshot.json", {}) or {}
            if not snapshot.get("sources"):
                snapshot["sources"] = sources
            readiness = cfg_store.material_variant_readiness(snapshot, archive_dir)
        completed_sections = int(readiness.get("completed_sections") or 0)
        readiness_percent = int(readiness.get("percent") or 0)
        highest_stage = str(readiness.get("highest_stage") or "Нет готовых разделов")
        restorable = any(
            path.is_file()
            for name in (variant.get("dirs") or [])
            if (archive_dir / name).is_dir()
            for path in (archive_dir / name).rglob("*")
        )
        material_versions.append(
            {
                "id": str(variant.get("id") or archive_dir.name),
                "path": str(archive_dir),
                "created_at": variant.get("created_at") or "",
                "reason": variant.get("reason") or "",
                "original_stream": original.get("stream_index"),
                "dubbed_stream": dubbed.get("stream_index"),
                "materials_count": completed_sections,
                "readiness_percent": readiness_percent,
                "readiness": readiness,
                "kind": highest_stage,
                "restorable": restorable,
            }
        )
    detail["material_versions"] = material_versions
    return detail


@app.get("/")
@app.get("/training")
@app.get("/apply")
def legacy_product_redirect():
    """Keep stale entry points from exposing the retired experimental UI."""
    return redirect("/product", code=302)


def _portable_product_ui_dir() -> Path:
    # Portable layout:
    # DubClean Portable/
    #   python_src/experiments/paired_reference_cancel/app.py
    #   docs/концепт интерфейса/DubClean-RU.dc.html
    return MODULE_DIR.parents[2] / "docs" / "концепт интерфейса"


_QUALITY_COMPARISON_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_QUALITY_COMPARISON_AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm"}
)
_QUALITY_COMPARISON_MANIFEST_MAX_BYTES = 8 * 1024 * 1024


def _quality_comparisons_root() -> Path:
    """Return the private root used by the standalone quality-review page."""
    return PORTABLE_ROOT / "data" / "quality_comparisons"


def _quality_comparison_dir(comparison_id: str) -> Path:
    value = str(comparison_id or "").strip()
    if not _QUALITY_COMPARISON_ID_RE.fullmatch(value):
        raise ValueError("Недопустимый идентификатор сравнения.")
    root = _quality_comparisons_root().resolve()
    directory = (root / value).resolve()
    if not is_within(directory, [root]):
        raise ValueError("Недопустимый идентификатор сравнения.")
    return directory


def _quality_comparison_audio_path(
    comparison_id: str, relative_name: Any, *, require_file: bool = True
) -> Path | None:
    """Resolve a manifest audio reference without permitting path escape."""
    value = str(relative_name or "").strip()
    if not value or "\\" in value or "\x00" in value:
        return None
    relative = Path(value)
    if relative.is_absolute() or relative.suffix.casefold() not in _QUALITY_COMPARISON_AUDIO_SUFFIXES:
        return None
    directory = _quality_comparison_dir(comparison_id)
    candidate = (directory / relative).resolve()
    if not is_within(candidate, [directory]):
        return None
    if require_file and not candidate.is_file():
        return None
    return candidate


def _quality_comparison_public_metric(value: Any) -> Any:
    """Keep report metrics compact and JSON-safe for the review UI."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:240]
    return None


def _quality_comparison_payload(comparison_id: str) -> dict[str, Any]:
    """Read and sanitize one comparison manifest before exposing it locally."""
    directory = _quality_comparison_dir(comparison_id)
    manifest_path = (directory / "manifest.json").resolve()
    if not is_within(manifest_path, [directory]):
        raise ValueError("Манифест сравнения находится вне разрешённой папки.")
    if not manifest_path.is_file():
        raise FileNotFoundError("Сравнение качества не найдено.")
    if manifest_path.stat().st_size > _QUALITY_COMPARISON_MANIFEST_MAX_BYTES:
        raise ValueError("Манифест сравнения качества слишком большой.")
    try:
        payload = read_json(manifest_path)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise ValueError("Некорректный манифест сравнения качества.") from error
    if not isinstance(payload, dict):
        raise ValueError("Некорректный манифест сравнения качества.")
    try:
        schema_version = int(payload.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError("Некорректная версия манифеста сравнения качества.") from error
    if schema_version != 1:
        raise ValueError("Неподдерживаемая версия манифеста сравнения качества.")
    raw_published = payload.get("published", True)
    published = raw_published if isinstance(raw_published, bool) else False
    publication_status = str(
        payload.get("publication_status")
        or ("ready" if published else "awaiting_after_rebuild")
    )[:80]

    intervals: list[dict[str, Any]] = []
    raw_intervals = payload.get("intervals")
    if raw_intervals is None:
        raw_intervals = []
    if not isinstance(raw_intervals, list):
        raise ValueError("В манифесте некорректно задан список интервалов.")
    for index, raw_interval in enumerate(raw_intervals[:10000]):
        if not isinstance(raw_interval, dict):
            continue
        try:
            start_sec = float(raw_interval.get("start_sec", 0.0))
            end_sec = float(raw_interval.get("end_sec", start_sec))
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(start_sec)
            or not math.isfinite(end_sec)
            or start_sec < 0.0
            or end_sec < start_sec
        ):
            continue

        metrics: dict[str, Any] = {}
        raw_metrics = raw_interval.get("metrics")
        if isinstance(raw_metrics, dict):
            for key, value in list(raw_metrics.items())[:64]:
                public_value = _quality_comparison_public_metric(value)
                if public_value is not None:
                    metrics[str(key)[:80]] = public_value

        before_name = raw_interval.get("before_audio")
        after_name = raw_interval.get("after_audio")
        before_path = (
            _quality_comparison_audio_path(comparison_id, before_name)
            if published
            else None
        )
        after_path = (
            _quality_comparison_audio_path(comparison_id, after_name)
            if published
            else None
        )

        def public_audio_url(path: Path | None) -> str | None:
            if path is None:
                return None
            relative = path.relative_to(directory).as_posix()
            encoded = urllib.parse.quote(relative, safe="/")
            return f"/api/quality-comparisons/{comparison_id}/audio/{encoded}"

        intervals.append(
            {
                "id": str(raw_interval.get("id") or f"interval-{index + 1}")[:80],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "reason": str(raw_interval.get("reason") or "")[:500],
                "decision": str(raw_interval.get("decision") or "")[:120],
                "confidence": _quality_comparison_public_metric(
                    raw_interval.get("confidence")
                ),
                "metrics": metrics,
                "before_audio_url": public_audio_url(before_path),
                "after_audio_url": public_audio_url(after_path),
                "audio_ready": before_path is not None and after_path is not None,
            }
        )

    summary: dict[str, Any] = {}
    raw_summary = payload.get("summary")
    if isinstance(raw_summary, dict):
        for key, value in list(raw_summary.items())[:64]:
            public_value = _quality_comparison_public_metric(value)
            if public_value is not None:
                summary[str(key)[:80]] = public_value

    return {
        "schema_version": 1,
        "id": comparison_id,
        "published": published,
        "publication_status": publication_status,
        "title": str(payload.get("title") or comparison_id)[:180],
        "description": str(payload.get("description") or "")[:1000],
        "generated_at": str(payload.get("generated_at") or "")[:80],
        "summary": summary,
        "intervals": intervals,
    }


def _quality_comparison_catalog() -> list[dict[str, Any]]:
    root = _quality_comparisons_root().resolve()
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not directory.is_dir() or not _QUALITY_COMPARISON_ID_RE.fullmatch(directory.name):
            continue
        try:
            payload = _quality_comparison_payload(directory.name)
        except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
        if not payload["published"]:
            continue
        items.append(
            {
                "id": payload["id"],
                "title": payload["title"],
                "generated_at": payload["generated_at"],
                "interval_count": len(payload["intervals"]),
                "audio_ready_count": sum(
                    1 for interval in payload["intervals"] if interval["audio_ready"]
                ),
            }
        )
    return items


@app.get("/product")
@app.get("/product/")
def product_page():
    """Open the portable product UI."""
    path = _portable_product_ui_dir() / "DubClean-RU.dc.html"
    if not path.is_file():
        return Response("Файлы интерфейса DubClean не найдены.", status=404)
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html; charset=utf-8")


@app.get("/product/quality-comparison")
def quality_comparison_page():
    """Open the isolated before/after quality review UI."""
    path = _portable_product_ui_dir() / "QualityComparison.dc.html"
    if not path.is_file():
        return Response("Страница сравнения качества не найдена.", status=404)
    return Response(path.read_text(encoding="utf-8"), mimetype="text/html; charset=utf-8")


@app.get("/api/quality-comparisons")
def quality_comparisons_catalog():
    return jsonify({"comparisons": _quality_comparison_catalog()})


@app.get("/api/quality-comparisons/<comparison_id>")
def quality_comparison_manifest(comparison_id: str):
    payload = _quality_comparison_payload(comparison_id)
    if not payload["published"]:
        return jsonify({"error": "Сравнение ещё не подготовлено."}), 409
    return jsonify(payload)


@app.get("/api/quality-comparisons/<comparison_id>/audio/<path:asset_name>")
def quality_comparison_audio(comparison_id: str, asset_name: str):
    try:
        payload = _quality_comparison_payload(comparison_id)
    except FileNotFoundError:
        return Response("Not found", status=404)
    if not payload["published"]:
        return Response("Not found", status=404)
    directory = _quality_comparison_dir(comparison_id)
    candidate = _quality_comparison_audio_path(
        comparison_id, asset_name, require_file=False
    )
    if candidate is None:
        raw_candidate = (directory / asset_name).resolve()
        if not is_within(raw_candidate, [directory]):
            return Response("Forbidden", status=403)
        return Response("Not found", status=404)
    if not candidate.is_file():
        return Response("Not found", status=404)
    return send_file(candidate, conditional=True)


@app.get("/product/support.js")
@app.get("/support.js")
def product_concept_support():
    path = _portable_product_ui_dir() / "support.js"
    if not path.is_file():
        return Response("support.js is missing in docs/концепт интерфейса", status=404)
    return send_file(path)


@app.get("/product/<path:asset_name>")
@app.get("/<path:asset_name>")
def product_concept_asset(asset_name: str):
    """Serve sibling DC component files used by the product concept."""
    if "/" in asset_name or "\\" in asset_name:
        return Response("Forbidden", status=403)
    allowed_suffixes = (".dc.html", ".css", ".js")
    if not asset_name.endswith(allowed_suffixes):
        return Response("Not found", status=404)
    ui_dir = _portable_product_ui_dir().resolve()
    path = (ui_dir / asset_name).resolve()
    if not is_within(path, [ui_dir]) or not path.is_file():
        return Response("Not found", status=404)
    return send_file(path)


def _application_projects() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for project in cfg_store.list_projects():
        if not project.get("application_mode"):
            continue
        pairs = cfg_store.list_pairs(project["id"])
        pair = pairs[0] if pairs else None
        pair_root = cfg_store.pair_dir(project["id"], pair["id"]) if pair else None
        preview = (
            _sanitize_application_preview(pair, pair_root)[0]
            if pair and pair_root is not None
            else None
        )
        result = (
            _sanitize_application_result(pair, pair_root)[0]
            if pair and pair_root is not None
            else None
        )
        active_tasks = [
            item
            for item in cfg_store.list_tasks(project["id"], {"queued", "starting", "running"})
            if not pair or not item.get("pair_id") or item.get("pair_id") == pair.get("id")
        ]
        active_task = _task_with_log(active_tasks[0]) if active_tasks else None
        items.append(
            {
                "id": project["id"],
                "name": project["name"],
                "folder": project.get("folder") or "Без папки",
                "created_at": project.get("created_at"),
                "updated_at": project.get("updated_at"),
                "pair_id": pair.get("id") if pair else None,
                "preview_ready": preview is not None,
                "result_ready": result is not None,
                "active_task": active_task,
            }
        )
    return items


def _bundled_semantic_models() -> list[dict[str, Any]]:
    """The single production speech-cleaning model shipped with DubClean."""
    models: list[dict[str, Any]] = []
    root = cfg_store.models_root / "semantic_ru_separator"
    if not root.is_dir():
        return models
    portable_manifest = read_json(PORTABLE_ROOT / "portable_manifest.json", {}) or {}
    semantic_spec = (portable_manifest.get("models") or {}).get("dubclean_voice") or {}
    recommended_name = Path(str(semantic_spec.get("file") or "")).name
    for path in sorted(root.glob("*.pt")):
        if recommended_name and path.name != recommended_name:
            continue
        models.append(
            {
                "path": str(path.resolve()),
                "name": str(semantic_spec.get("public_name") or "DubClean Voice"),
                "project_name": "DubClean",
                "algorithm": "Очистка русской речи по иностранному референсу",
                "epoch": None,
                "recommended_for_real_films": path.name == recommended_name,
                "review_required": False,
                "size": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            }
        )
    return models


def _subtraction_model_options() -> list[dict[str, Any]]:
    """Return the only supported production subtraction model."""
    manifest = read_json(PORTABLE_ROOT / "portable_manifest.json", {}) or {}
    semantic_spec = (manifest.get("models") or {}).get("dubclean_voice") or {}
    model_path = (PORTABLE_ROOT / str(semantic_spec.get("file") or "")).resolve()
    return [
        {
            "id": "dubclean_voice",
            "label": str(semantic_spec.get("public_name") or "DubClean Voice"),
            "path": str(model_path) if model_path.is_file() else "",
            "available": model_path.is_file(),
        }
    ]


def _subtraction_checkpoint(choice: str) -> tuple[str, Path | None]:
    normalized = str(choice or "dubclean_voice").strip()
    option = next(
        (item for item in _subtraction_model_options() if item["id"] == normalized),
        None,
    )
    if option is None:
        raise ValueError("Неизвестная модель очистки перевода.")
    value = str(option.get("path") or "").strip()
    path = Path(value).resolve() if value else None
    if not option.get("available") or path is None or not path.is_file():
        raise ValueError("Модель DubClean Voice не найдена.")
    return normalized, path


def _bundled_background_models() -> list[dict[str, Any]]:
    """Background (M&E) restorer checkpoints shipped inside portable models/."""
    models: list[dict[str, Any]] = []
    root = cfg_store.models_root / "background_restorer"
    if not root.is_dir():
        return models
    portable_manifest = read_json(PORTABLE_ROOT / "portable_manifest.json", {}) or {}
    background_spec = (portable_manifest.get("models") or {}).get("dubclean_me") or {}
    recommended_name = Path(str(background_spec.get("file") or "")).name
    for path in sorted(root.glob("*.pt")):
        if recommended_name and path.name != recommended_name:
            continue
        models.append(
            {
                "path": str(path.resolve()),
                "name": str(background_spec.get("public_name") or "DubClean M&E"),
                "project_name": "DubClean",
                "recommended": path.name == recommended_name,
                "size": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            }
        )
    return models


def _application_models() -> list[dict[str, Any]]:
    return _bundled_semantic_models()


def _application_background_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = _bundled_background_models()
    for project in cfg_store.list_projects():
        root = cfg_store.project_dir(project["id"])
        selection = (
            read_json(root / "background_checkpoints" / "model_selection.json", {})
            or {}
        )
        selected = str(selection.get("selected_checkpoint") or "")
        for path in (root / "background_checkpoints").glob("*.pt"):
            if "background" not in path.name.casefold():
                continue
            models.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "project_name": project["name"],
                    "recommended": str(path.resolve()) == selected,
                    "size": path.stat().st_size,
                    "modified_at": path.stat().st_mtime,
                }
            )
    models.sort(key=lambda item: (not item["recommended"], -item["modified_at"]))
    return models


def _application_reference_adapter_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for project in cfg_store.list_projects():
        root = cfg_store.project_dir(project["id"])
        checkpoint_roots = [
            *sorted(root.glob("reference_adapter_checkpoints*")),
            *sorted(root.glob("global_mastering_checkpoints*")),
        ]
        for checkpoint_root in checkpoint_roots:
            if not checkpoint_root.is_dir():
                continue
            # Каталоги с маркером невалидности (например smoothmix_v3, где
            # config чекпойнта не совпадает с фактическим обучением) в веб-НЕ
            # предлагаем.
            if (checkpoint_root / "INVALID_CHECKPOINTS.md").is_file():
                continue
            for path in checkpoint_root.glob("*.pt"):
                name_lower = path.name.casefold()
                is_global = "global_mastering_estimator" in name_lower
                is_reference = "reference_adapter" in name_lower
                if not is_reference and not is_global:
                    continue
                is_best = path.name in {
                    "neural_reference_adapter_best.pt",
                    "global_mastering_estimator_best.pt",
                }
                epoch = None
                match = re.search(r"_epoch(\d+)\.pt$", path.name, re.I)
                if match:
                    epoch = int(match.group(1))
                models.append(
                    {
                        "path": str(path.resolve()),
                        "name": path.name,
                        "project_name": project["name"],
                        "kind": "global_mastering" if is_global else "reference_adapter",
                        "recommended": is_best,
                        "epoch": epoch,
                        "size": path.stat().st_size,
                        "modified_at": path.stat().st_mtime,
                    }
                )
    models.sort(key=lambda item: (not item["recommended"], -item["modified_at"]))
    return models


def _safe_checkpoint(value: str) -> Path:
    path = Path(str(value or "")).resolve()
    if (
        not path.is_file()
        or path.suffix.lower() != ".pt"
        or not is_within(path, [cfg_store.projects_root, cfg_store.models_root])
    ):
        raise ValueError("Выбранное сохранение модели не найдено.")
    return path


def _safe_background_checkpoint(value: str) -> Path:
    path = _safe_checkpoint(value)
    # Accept both trained project checkpoints (…/background_checkpoints/…) and
    # the model bundled with the portable package (models/background_restorer/…).
    parent_names = {parent.name.casefold() for parent in path.parents}
    bundled = is_within(path, [cfg_store.models_root]) and path.name == "dubclean_me.pt"
    if "background_checkpoints" not in parent_names and not bundled:
        raise ValueError("Выбранный файл не относится к модели музыки и эффектов.")
    return path


def _safe_reference_adapter_checkpoint(value: str) -> str:
    """Пустое значение допустимо: пайплайн возьмёт дефолтный чекпойнт."""
    if not str(value or "").strip():
        return ""
    path = _safe_checkpoint(value)
    if not any(
        parent.name.casefold().startswith("reference_adapter_checkpoints")
        or parent.name.casefold().startswith("global_mastering_checkpoints")
        for parent in path.parents
    ):
        raise ValueError("Выбранное сохранение не относится к адаптеру референса.")
    if (path.parent / "INVALID_CHECKPOINTS.md").is_file():
        raise ValueError(
            "Этот чекпойнт адаптера помечен как невалидный (INVALID_CHECKPOINTS.md)."
        )
    return str(path)


def _invalidate_application_catalog() -> None:
    global _application_catalog_cache, _application_catalog_cached_at
    with _application_catalog_lock:
        _application_catalog_cache = None
        _application_catalog_cached_at = 0.0


def _build_application_catalog() -> dict[str, Any]:
    speech_cfg = cfg["speech_extraction"]
    speech_checkpoint_dir = root_path(speech_cfg["checkpoint_dir"])
    speech_checkpoint = speech_checkpoint_dir / "last_best_checkpoint.pt"
    music_cfg = cfg["music_separation"]
    music_model_path = root_path(music_cfg["model_dir"]) / str(music_cfg["model_name"])
    music_expected_size = int(music_cfg.get("model_size_bytes") or 0)
    music_ready = (
        music_model_path.is_file()
        and not music_model_path.with_name(music_model_path.name + ".aria2").is_file()
        and (not music_expected_size or music_model_path.stat().st_size == music_expected_size)
    )
    silero_model_path = (
        PORTABLE_ROOT / ".venv" / "Lib" / "site-packages"
        / "silero_vad" / "data" / "silero_vad.jit"
    )
    silero_model_size = silero_model_path.stat().st_size if silero_model_path.is_file() else 0
    song_cfg = cfg.get("song_protection") or {}
    song_model_path = root_path(
        str(
            song_cfg.get("classifier_model_path")
            or "./third_party_models/efficientat/"
            "efficientat_mn04_audioset_waveform.pt"
        )
    )
    song_model_size = song_model_path.stat().st_size if song_model_path.is_file() else 0
    return {
            "applications": _application_projects(),
            "config": _public_config(),
            "models": _application_models(),
            "subtraction_models": _subtraction_model_options(),
            "background_models": _application_background_models(),
            "reference_adapter_models": _application_reference_adapter_models(),
            "speech_extractor": {
                "backend": speech_cfg["backend"],
                "model_name": speech_cfg["model_name"],
                "checkpoint_dir": str(speech_checkpoint_dir),
                "model_size_bytes": speech_checkpoint.stat().st_size if speech_checkpoint.is_file() else 0,
                "model_ready": speech_checkpoint.is_file(),
            },
            "vad_model": {
                "backend": "silero-vad",
                "model_name": "Silero VAD",
                "model_size_bytes": silero_model_size,
                "model_ready": bool(silero_model_size),
            },
            "song_classifier": {
                "backend": str(
                    song_cfg.get("classifier_backend")
                    or "efficientat_mn04_audioset"
                ),
                "model_name": "EfficientAT MN04 · поиск песен",
                "model_size_bytes": song_model_size,
                "model_ready": bool(song_model_size),
                "input_track": "original_english",
            },
            "music_separator": {
                "backend": music_cfg["backend"],
                "model_name": music_cfg["model_name"],
                "model_size_bytes": music_expected_size,
                "model_ready": music_ready,
            },
            "separator": {
                "backend": music_cfg["backend"],
                "model_name": music_cfg["model_name"],
                "model_size_bytes": music_expected_size,
                "model_ready": music_ready,
            },
        }


def _application_catalog_payload() -> dict[str, Any]:
    """Coalesce the expensive all-project scan across browser tabs.

    Old completed projects may live on a slow disk and contain many artifacts.
    Several open tabs used to start the same scan every three seconds, causing
    an ever-growing queue of Flask threads and making even button clicks stall.
    """
    global _application_catalog_cache, _application_catalog_cached_at
    with _application_catalog_lock:
        now = time.monotonic()
        if (
            _application_catalog_cache is not None
            and now - _application_catalog_cached_at < _APPLICATION_CATALOG_TTL_SEC
        ):
            return copy.deepcopy(_application_catalog_cache)
        payload = _build_application_catalog()
        _application_catalog_cache = copy.deepcopy(payload)
        _application_catalog_cached_at = time.monotonic()
        return payload


@app.get("/api/applications")
def api_applications():
    return jsonify(_application_catalog_payload())


@app.post("/api/applications")
def api_create_application():
    body = request.get_json(silent=True) or {}
    original_path, original_probe = _probe(str(body.get("original_path", "")))
    dubbed_path, dubbed_probe = _probe(str(body.get("dubbed_path", "")))
    original_stream = _validate_stream(
        original_probe, body.get("original_stream"), "оригинала"
    )
    dubbed_stream = _validate_stream(
        dubbed_probe, body.get("dubbed_stream"), "перевода"
    )
    title = str(body.get("name") or "").strip() or f"Применение модели — {dubbed_path.stem}"
    source_mode = str(body.get("source_mode") or "").strip().lower()
    if source_mode not in {"single", "pair"}:
        source_mode = "single" if original_path == dubbed_path else "pair"
    project = cfg_store.create_project(title)
    project["application_mode"] = True
    project["application_schema_version"] = 1
    project["application_source_mode"] = source_mode
    project["folder"] = str(body.get("folder") or "Без папки").strip() or "Без папки"
    cfg_store.save_project(project)
    try:
        pair = cfg_store.add_pair(
            project["id"],
            str(body.get("film_name") or dubbed_path.stem),
            original_path,
            dubbed_path,
            original_stream,
            dubbed_stream,
            original_probe,
            dubbed_probe,
        )
    except Exception:
        # Do not leave an empty project card when pair creation fails after the
        # project directory itself was already created.
        shutil.rmtree(cfg_store.project_dir(project["id"]), ignore_errors=True)
        raise
    _invalidate_application_catalog()
    return jsonify({"project": project, "pair": pair}), 201


def _invalidate_application_detail(project_id: str | None = None) -> None:
    with _application_detail_lock:
        if project_id is None:
            _application_detail_cache.clear()
        else:
            _application_detail_cache.pop(str(project_id), None)


def _build_application_detail(project_id: str) -> dict[str, Any]:
    project = cfg_store.load_project(project_id)
    if not project.get("application_mode"):
        raise ValueError("Это не проект DubClean.")
    pairs = cfg_store.list_pairs(project_id)
    pair = _pair_detail(project_id, pairs[0]["id"]) if pairs else None
    return {
        "project": project,
        "pair": pair,
        "tasks": [
            _task_with_log(item, include_result=True)
            for item in cfg_store.list_tasks(project_id)[:20]
        ],
    }


def _application_detail_payload(project_id: str) -> dict[str, Any]:
    """Share one short-lived project snapshot between simultaneous tabs."""
    key = str(project_id)
    with _application_detail_lock:
        now = time.monotonic()
        cached = _application_detail_cache.get(key)
        if cached and now - cached[0] < _APPLICATION_DETAIL_TTL_SEC:
            return copy.deepcopy(cached[1])
        payload = _build_application_detail(key)
        _application_detail_cache[key] = (time.monotonic(), copy.deepcopy(payload))
        return payload


@app.get("/api/applications/<project_id>")
def api_application(project_id: str):
    return jsonify(_application_detail_payload(project_id))


@app.patch("/api/applications/<project_id>")
def api_update_application(project_id: str):
    project = cfg_store.load_project(project_id)
    if not project.get("application_mode"):
        raise ValueError("Это не проект DubClean.")
    body = request.get_json(silent=True) or {}
    if "name" in body:
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("Название проекта не может быть пустым.")
        project["name"] = name[:200]
    if "folder" in body:
        project["folder"] = str(body.get("folder") or "Без папки").strip() or "Без папки"
    if "auto_assemble" in body:
        project["auto_assemble"] = bool(body.get("auto_assemble"))
    cfg_store.save_project(project)
    pairs = cfg_store.list_pairs(project_id)
    pair = pairs[0] if pairs else None
    if pair is not None:
        if "subtraction_model" in body:
            choice = str(body.get("subtraction_model") or "dubclean_voice")
            normalized, _checkpoint = _subtraction_checkpoint(choice)
            pair["reference_compatibility_manual_model"] = normalized
        if "reference_alignment" in body:
            alignment = str(body.get("reference_alignment") or "algorithmic")
            if alignment not in {"none", "algorithmic"}:
                raise ValueError("Неизвестный режим выравнивания референса.")
            pair["reference_compatibility_manual_alignment"] = alignment
        if "speech_extraction_second_pass" in body:
            second_pass = bool(body.get("speech_extraction_second_pass"))
            pair["speech_extraction_second_pass"] = second_pass
            # A direct toggle is the user's manual choice.  Keep it separately
            # so “Вернуть ручной выбор” can restore it after recommendations
            # have temporarily changed the effective setting.
            pair[
                "reference_compatibility_manual_speech_extraction_second_pass"
            ] = second_pass
            pair.pop("reference_compatibility_applied", None)
            analysis = pair.get("reference_compatibility_analysis") or {}
            if analysis.get("created_at"):
                pair["reference_compatibility_decision"] = {
                    "choice": "manual",
                    "manual_alignment": str(
                        pair.get("reference_compatibility_manual_alignment")
                        or "algorithmic"
                    ),
                    "manual_model": str(
                        pair.get("reference_compatibility_manual_model")
                        or "dubclean_voice"
                    ),
                    "speech_extraction_second_pass": second_pass,
                    "analysis_created_at": analysis.get("created_at"),
                    "decided_at": utc_now(),
                }
        if (
            "subtraction_model" in body
            or "reference_alignment" in body
            or "speech_extraction_second_pass" in body
        ):
            cfg_store.save_pair(pair)
    _invalidate_application_catalog()
    return jsonify({"project": project, "pair": pair})


@app.delete("/api/applications/<project_id>")
def api_delete_application(project_id: str):
    with _task_mutation_lock:
        project = cfg_store.load_project(project_id)
        if not project.get("application_mode"):
            raise ValueError("Это не проект DubClean.")
        project_dir = cfg_store.project_dir(project_id).resolve()
        projects_root = cfg_store.projects_root.resolve()
        if project_dir == projects_root or not is_within(project_dir, [projects_root]):
            raise ValueError("Небезопасный путь проекта: удаление остановлено.")
        for task in cfg_store.list_tasks(project_id, {"queued", "starting", "running"}):
            _stop_task_record(
                task,
                state="cancelled",
                substage="Проект удалён пользователем.",
                error="Проект удалён пользователем.",
            )
        last_error: OSError | None = None
        for attempt in range(6):
            try:
                if project_dir.exists():
                    shutil.rmtree(project_dir)
                last_error = None
                break
            except OSError as error:
                last_error = error
                if attempt < 5:
                    time.sleep(0.2 * (attempt + 1))
        if last_error is not None or project_dir.exists():
            raise RuntimeError(
                "Не удалось полностью удалить папку проекта. Закройте открытые "
                "файлы проекта и повторите удаление."
            ) from last_error
    _invalidate_application_catalog()
    return jsonify({"ok": True, "deleted_path": str(project_dir)})


def _start_application_task(
    project_id: str, operation: str, parameters: dict[str, Any]
):
    with _task_mutation_lock:
        project = cfg_store.load_project(project_id)
        if not project.get("application_mode"):
            raise ValueError("Это не проект DubClean.")
        pairs = cfg_store.list_pairs(project_id)
        if len(pairs) != 1:
            raise RuntimeError("Рабочий проект должен содержать ровно одну пару фильмов.")
        pair = pairs[0]
        _ensure_no_duplicate(project_id, pair["id"], operation)
        pair_root = cfg_store.pair_dir(project_id, pair["id"])
        if operation == "application_full":
            preview, _missing = _sanitize_application_preview(pair, pair_root)
            if preview is None:
                pair.pop("application_preview", None)
                cfg_store.save_pair(pair)
                raise RuntimeError(
                    "Файлы превью отсутствуют или повреждены. Сначала создайте превью заново."
                )
            pair["application_preview"] = preview
        if operation in {"application_full", "application_auto"}:
            # A new full run must not advertise the previous movie while its
            # replacement is still being built.  The files remain recoverable
            # on disk and the new task may reuse valid intermediates, but the
            # API/UI only sees artifacts published by this run.
            pair.pop("application_result", None)
            pair.pop("application_full_progress", None)
        elif operation == "application_prepare":
            for key in (
                "application_preview", "application_checkpoint",
                "application_background_checkpoint", "application_reused_preparation",
            ):
                pair.pop(key, None)
            subtraction_model = str(parameters.get("subtraction_model") or "")
            if subtraction_model == "dubclean_voice":
                pair["reference_compatibility_manual_model"] = subtraction_model
                if not pair.get("reference_compatibility_applied"):
                    pair["reference_compatibility_effective_model"] = subtraction_model
            pair["speech_extraction_second_pass"] = bool(
                parameters.get(
                    "speech_extraction_second_pass",
                    pair.get("speech_extraction_second_pass", False),
                )
            )
        cfg_store.save_pair(pair)
        task = cfg_store.create_task(project_id, operation, pair["id"], parameters)
    scheduler_tick()
    return jsonify({"task": task}), 202


@app.post("/api/applications/<project_id>/prepare")
def api_prepare_application(project_id: str):
    body = request.get_json(silent=True) or {}
    requested_model, checkpoint = _subtraction_checkpoint(
        str(body.get("subtraction_model") or "dubclean_voice")
    )
    assert checkpoint is not None
    background_checkpoint = _safe_background_checkpoint(
        str(body.get("background_checkpoint") or "")
    )
    return _start_application_task(
        project_id,
        "application_prepare",
        {
            "checkpoint": str(checkpoint),
            "subtraction_model": requested_model,
            "background_checkpoint": str(background_checkpoint),
            "reference_adapter_checkpoint": _safe_reference_adapter_checkpoint(
                str(body.get("reference_adapter_checkpoint") or "")
            ),
            "speech_extraction_second_pass": bool(
                body.get("speech_extraction_second_pass", False)
            ),
        },
    )


@app.post("/api/applications/<project_id>/tracks")
def api_prepare_application_tracks(project_id: str):
    return _start_application_task(project_id, "application_tracks", {})


@app.post("/api/applications/<project_id>/reference-analysis")
def api_analyze_application_reference(project_id: str):
    return _start_application_task(project_id, "application_reference_analysis", {})


@app.post("/api/applications/<project_id>/reference-analysis/apply")
def api_apply_application_reference_analysis(project_id: str):
    body = request.get_json(silent=True) or {}
    with _task_mutation_lock:
        project = cfg_store.load_project(project_id)
        if not project.get("application_mode"):
            raise ValueError("Это не проект DubClean.")
        pairs = cfg_store.list_pairs(project_id)
        if len(pairs) != 1:
            raise RuntimeError("Рабочий проект должен содержать ровно одну пару фильмов.")
        pair = pairs[0]
        if _active_tasks_for_pair(project_id, pair["id"]):
            raise RuntimeError("Дождитесь завершения текущей операции проекта.")
        analysis = copy.deepcopy(pair.get("reference_compatibility_analysis") or {})
        manual_alignment = str(body.get("manual_alignment") or "").strip()
        if manual_alignment in {"none", "algorithmic"}:
            pair["reference_compatibility_manual_alignment"] = manual_alignment
        manual_model = str(body.get("manual_model") or "").strip()
        if manual_model:
            manual_model, _manual_checkpoint = _subtraction_checkpoint(manual_model)
            pair["reference_compatibility_manual_model"] = manual_model
        if (
            "manual_speech_extraction_second_pass" in body
            and (
                "reference_compatibility_manual_speech_extraction_second_pass"
                not in pair
            )
        ):
            pair["reference_compatibility_manual_speech_extraction_second_pass"] = bool(
                body.get("manual_speech_extraction_second_pass")
            )
        cfg_store.save_pair(pair)
        applied = apply_reference_compatibility_recommendations(
            cfg_store, project_id, pair["id"], analysis
        )
    return jsonify({"ok": True, "applied": applied})


@app.post("/api/applications/<project_id>/reference-analysis/manual")
def api_keep_manual_application_reference_settings(project_id: str):
    """Keep the user's alignment choice and opt out of analysed routing."""
    body = request.get_json(silent=True) or {}
    with _task_mutation_lock:
        project = cfg_store.load_project(project_id)
        if not project.get("application_mode"):
            raise ValueError("Это не проект DubClean.")
        pairs = cfg_store.list_pairs(project_id)
        if len(pairs) != 1:
            raise RuntimeError("Рабочий проект должен содержать ровно одну пару фильмов.")
        pair = pairs[0]
        if _active_tasks_for_pair(project_id, pair["id"]):
            raise RuntimeError("Дождитесь завершения текущей операции проекта.")
        analysis = pair.get("reference_compatibility_analysis") or {}
        if analysis.get("band") not in {"HIGH", "MID", "LOW", "PASSTHROUGH"}:
            raise RuntimeError("Сначала выполните анализ соответствия дорожек.")
        requested_manual = str(body.get("manual_alignment") or "").strip()
        if requested_manual in {"none", "algorithmic"}:
            pair["reference_compatibility_manual_alignment"] = requested_manual
        manual_alignment = str(
            pair.get("reference_compatibility_manual_alignment") or "algorithmic"
        )
        if manual_alignment not in {"none", "algorithmic"}:
            manual_alignment = "algorithmic"
        requested_model = str(body.get("manual_model") or "").strip()
        if requested_model:
            manual_model, _manual_checkpoint = _subtraction_checkpoint(requested_model)
            pair["reference_compatibility_manual_model"] = manual_model
        manual_model = str(
            pair.get("reference_compatibility_manual_model")
            or "dubclean_voice"
        )
        if "manual_speech_extraction_second_pass" in body:
            pair["reference_compatibility_manual_speech_extraction_second_pass"] = bool(
                body.get("manual_speech_extraction_second_pass")
            )
        manual_second_pass = bool(
            pair.get(
                "reference_compatibility_manual_speech_extraction_second_pass",
                pair.get("speech_extraction_second_pass", False),
            )
        )
        pair.pop("reference_compatibility_applied", None)
        pair["reference_compatibility_effective_model"] = manual_model
        pair["speech_extraction_second_pass"] = manual_second_pass
        pair["reference_compatibility_decision"] = {
            "choice": "manual",
            "manual_alignment": manual_alignment,
            "manual_model": manual_model,
            "speech_extraction_second_pass": manual_second_pass,
            "analysis_created_at": analysis.get("created_at"),
            "decided_at": utc_now(),
        }
        cfg_store.save_pair(pair)
    return jsonify(
        {
            "ok": True,
            "decision": pair["reference_compatibility_decision"],
            "manual_alignment": manual_alignment,
            "manual_model": manual_model,
            "speech_extraction_second_pass": manual_second_pass,
        }
    )


@app.post("/api/applications/<project_id>/auto")
def api_auto_application(project_id: str):
    """Run the complete application workflow without browser-side chaining.

    Keeping the orchestration in one persisted worker makes the automatic mode
    independent from page navigation and avoids races between polling and the
    next UI click.  The individual pipeline stages remain idempotent and reuse
    valid extracted tracks where possible.
    """
    body = request.get_json(silent=True) or {}
    requested_model, checkpoint = _subtraction_checkpoint("dubclean_voice")
    assert checkpoint is not None
    background_checkpoint = _safe_background_checkpoint(
        str(body.get("background_checkpoint") or "")
    )
    reference_alignment_policy = normalize_policy(
        body.get("reference_alignment_policy"),
        setting_name="выравнивание качества",
    )
    speech_second_pass_policy = normalize_policy(
        body.get("speech_second_pass_policy"),
        setting_name="повторный проход MossFormer",
    )
    video_source = str(body.get("video_source") or "dubbed").strip().lower()
    if video_source not in {"dubbed", "original"}:
        raise ValueError("Выберите, из какого файла брать видео.")
    # The one-click film must obey the same answers as the step-by-step one.
    # Anything omitted here is decided by the worker from the analysis, which
    # is correct only for the two settings that offer «Автоматически».
    parameters: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "subtraction_model": requested_model,
        "background_checkpoint": str(background_checkpoint),
        "reference_adapter_checkpoint": _safe_reference_adapter_checkpoint(
            str(body.get("reference_adapter_checkpoint") or "")
        ),
        "method": "speech_rebuild",
        "remux": True,
        "force": False,
        "reference_alignment_policy": reference_alignment_policy,
        "speech_second_pass_policy": speech_second_pass_policy,
        "video_source": video_source,
        "include_all_tracks": bool(body.get("include_all_tracks", True)),
        "voice_gain_db": _clamp_float(body.get("voice_gain_db"), -12.0, 12.0),
        "background_gain_db": _clamp_float(
            body.get("background_gain_db"), -12.0, 12.0
        ),
        "voice_delay_sec": _clamp_float(body.get("voice_delay_sec"), -2.0, 2.0),
    }
    if "synchronize_speech" in body:
        parameters["synchronize_speech"] = bool(body.get("synchronize_speech"))
    if "balance_final_mix" in body:
        parameters["balance_final_mix"] = bool(body.get("balance_final_mix"))
    return _start_application_task(project_id, "application_auto", parameters)


def _clamp_float(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return max(low, min(high, number))


@app.post("/api/applications/<project_id>/full")
def api_full_application(project_id: str):
    body = request.get_json(silent=True) or {}
    method = str(body.get("method") or "")
    if method not in {"voice_only", "speech_rebuild", "direct_mix"}:
        raise ValueError("Выберите доступный способ сборки звука.")
    reference_selection_mode = str(
        body.get("reference_selection_mode") or ""
    ).strip().lower()
    if reference_selection_mode and reference_selection_mode not in {
        "raw", "algorithmic"
    }:
        raise ValueError("Выберите доступный способ подготовки референса.")
    reference_adapter = (
        reference_selection_mode
        or str(body.get("reference_adapter") or "auto")
    )
    video_source = str(body.get("video_source") or "dubbed").strip().lower()
    if video_source not in {"dubbed", "original"}:
        raise ValueError("Выберите, из какого файла брать видео.")
    return _start_application_task(
        project_id,
        "application_full",
        {
            "method": method,
            "remux": bool(body.get("remux", True)),
            "video_source": video_source,
            "include_all_tracks": bool(body.get("include_all_tracks", True)),
            "force": bool(body.get("force", False)),
            "synchronize_speech": bool(body.get("synchronize_speech", False)),
            "balance_final_mix": bool(body.get("balance_final_mix", True)),
            "reference_adapter": reference_adapter,
            "reference_selection_mode": reference_selection_mode,
            "reference_adapter_checkpoint": _safe_reference_adapter_checkpoint(
                str(body.get("reference_adapter_checkpoint") or "")
            ),
            # Manual mix trims from the editor faders. 0 keeps the automatic mix
            # unchanged, so existing runs are byte-for-byte identical.
            "voice_gain_db": _clamp_float(body.get("voice_gain_db"), -12.0, 12.0),
            "background_gain_db": _clamp_float(body.get("background_gain_db"), -12.0, 12.0),
            "voice_delay_sec": _clamp_float(body.get("voice_delay_sec"), -2.0, 2.0),
            "speech_extraction_second_pass": bool(
                body.get("speech_extraction_second_pass", False)
            ),
        },
    )


@app.get("/api/config")
def api_config():
    return jsonify(_public_config())


@app.get("/api/projects")
def api_projects():
    return jsonify({"projects": cfg_store.list_projects()})


@app.post("/api/projects")
def api_create_project():
    body = request.get_json(silent=True) or {}
    return jsonify({"project": cfg_store.create_project(body.get("name", ""))}), 201


@app.get("/api/projects/<project_id>")
def api_project(project_id: str):
    project = cfg_store.load_project(project_id)
    root = cfg_store.project_dir(project_id)
    pairs = cfg_store.list_pairs(project_id)
    for pair in pairs:
        pair_root = cfg_store.pair_dir(project_id, pair["id"])
        stage2_ready = (
            (pair.get("stages") or {}).get("2", {}).get("status") == "completed"
        )
        extracted_files = []
        if stage2_ready:
            for role, label in (
                ("original", "Оригинальная дорожка"),
                ("dubbed", "Дорожка RU+EN"),
            ):
                path = pair_root / "extracted" / f"{role}.flac"
                if path.is_file():
                    extracted_files.append(
                        {
                            "role": role,
                            "label": label,
                            "path": str(path.resolve()),
                            "name": path.name,
                            "size": path.stat().st_size,
                        }
                    )
        pair["extracted_files"] = extracted_files
    datasets = []
    for path in sorted((root / "training" / "datasets").glob("*/manifest.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            example_count = len(manifest.get("examples") or [])
            manifest["real_evaluation_count"] = len(manifest.get("real_evaluation") or [])
            manifest.pop("examples", None)
            manifest.pop("real_evaluation", None)
            manifest["examples_omitted_from_interface"] = example_count
            datasets.append(manifest)
        except (OSError, ValueError):
            continue
    histories = []
    for path in sorted((root / "training" / "runs").glob("*/history.json"), reverse=True):
        try:
            histories.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    checkpoints = [
        {"name": path.name, "path": str(path.resolve()), "size": path.stat().st_size}
        for path in sorted(
            [
                item
                for item in (root / "checkpoints").glob("*")
                if item.is_file() and item.suffix.lower() in {".pt", ".npz"}
            ],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    ]
    model_selection = None
    selection_path = root / "checkpoints" / "model_selection.json"
    if selection_path.is_file():
        try:
            model_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            model_selection = None
    return jsonify(
        {
            "project": project,
            "pairs": pairs,
            "tasks": [_task_with_log(item) for item in cfg_store.list_tasks(project_id)[:30]],
            "datasets": datasets,
            "training_histories": histories,
            "checkpoints": checkpoints,
            "model_selection": model_selection,
        }
    )


@app.get("/api/projects/<project_id>/datasets/<dataset_id>")
def api_dataset_detail(project_id: str, dataset_id: str):
    dataset_root = cfg_store.project_artifact(
        project_id, f"training/datasets/{dataset_id}"
    )
    expected_parent = (
        cfg_store.project_dir(project_id) / "training" / "datasets"
    ).resolve()
    if dataset_root.parent != expected_parent or dataset_root.name != dataset_id:
        raise ValueError("Недопустимый идентификатор датасета.")
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return jsonify({"error": "Датасет не найден."}), 404
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = manifest.get("examples") or []

    def level_group(item: dict[str, Any]) -> str:
        value = float((item.get("recipe") or {}).get("english_to_target_db", 0.0))
        if value < -20.0:
            return "very_quiet"
        if value < -10.0:
            return "quiet"
        if value < 0.0:
            return "balanced"
        return "loud"

    def delay_group(item: dict[str, Any]) -> str:
        value = float((item.get("recipe") or {}).get("english_delay_ms", 0.0))
        if value < -30.0:
            return "negative"
        if value > 30.0:
            return "positive"
        return "aligned"

    film = str(request.args.get("film") or "").strip()
    split = str(request.args.get("split") or "").strip()
    exact = str(request.args.get("exact") or "").strip()
    level = str(request.args.get("level") or "").strip()
    delay = str(request.args.get("delay") or "").strip()
    ducking = str(request.args.get("ducking") or "").strip()
    scenario = str(request.args.get("scenario") or "").strip()
    query = str(request.args.get("query") or "").strip().casefold()
    try:
        page = max(1, int(request.args.get("page", "1")))
        page_size = max(5, min(50, int(request.args.get("page_size", "12"))))
    except ValueError as error:
        raise ValueError("Страница и размер страницы должны быть числами.") from error

    filtered = []
    for item in examples:
        recipe = item.get("recipe") or {}
        if film and str(item.get("film_name") or "") != film:
            continue
        if split and str(item.get("split") or "") != split:
            continue
        if exact in {"yes", "no"} and bool(recipe.get("use_exact_target")) != (
            exact == "yes"
        ):
            continue
        if level and level_group(item) != level:
            continue
        if delay and delay_group(item) != delay:
            continue
        if ducking in {"yes", "no"} and (
            float(recipe.get("ducking_db", 0.0)) > 0.05
        ) != (ducking == "yes"):
            continue
        item_scenario = str(
            (recipe.get("short_utterance") or {}).get("scenario")
            or "long_speech"
        )
        if scenario and item_scenario != scenario:
            continue
        if query and query not in str(item.get("id") or "").casefold():
            continue
        filtered.append(item)

    total_filtered = len(filtered)
    page_count = max(1, (total_filtered + page_size - 1) // page_size)
    page = min(page, page_count)
    page_rows = filtered[(page - 1) * page_size : page * page_size]
    samples: list[dict[str, Any]] = []
    for item in page_rows:
        recipe = item.get("recipe") or {}
        samples.append(
            {
                "id": item.get("id"),
                "split": item.get("split"),
                "film_name": item.get("film_name"),
                "duration_sec": item.get("duration_sec"),
                "alignment_confidence": item.get("alignment_confidence"),
                "dubbed_start_sec": recipe.get("dubbed_start_sec"),
                "original_start_sec": recipe.get("original_start_sec"),
                "english_speech_rms_db": recipe.get("english_speech_rms_db"),
                "russian_speech_rms_db": recipe.get("russian_speech_rms_db"),
                "english_to_target_db": recipe.get("english_to_target_db"),
                "english_gain_change_db": recipe.get("english_gain_change_db"),
                "english_delay_ms": recipe.get("english_delay_ms"),
                "russian_shift_sec": recipe.get("russian_shift_sec"),
                "spectral_tilt_db": recipe.get("spectral_tilt_db"),
                "soft_compression_drive": recipe.get("soft_compression_drive"),
                "reverb_wet": recipe.get("reverb_wet"),
                "ducking_db": recipe.get("ducking_db"),
                "use_exact_target": recipe.get("use_exact_target"),
                "augmentation_kind": item.get("augmentation_kind") or "long_speech",
                "short_utterance": recipe.get("short_utterance"),
                "audio_url": (
                    f"/api/projects/{project_id}/datasets/{dataset_id}"
                    f"/examples/{item.get('id')}/audio"
                ),
            }
        )
    film_counts: dict[str, int] = {}
    for item in examples:
        name = str(item.get("film_name") or "Без названия")
        film_counts[name] = film_counts.get(name, 0) + 1

    def count_groups(grouping) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in examples:
            key = str(grouping(item))
            result[key] = result.get(key, 0) + 1
        return result

    return jsonify(
        {
            "dataset_id": manifest.get("dataset_id"),
            "target_kind": manifest.get("target_kind"),
            "created_at": manifest.get("created_at"),
            "example_count": manifest.get("example_count", len(examples)),
            "base_speech_fragment_count": manifest.get(
                "base_speech_fragment_count"
            ),
            "split_counts": manifest.get("split_counts") or {},
            "film_counts": film_counts,
            "augmentation": manifest.get("augmentation") or {},
            "integrity": manifest.get("integrity") or {},
            "rejected_count": manifest.get("rejected_count"),
            "rejected_reasons": manifest.get("rejected_reasons") or {},
            "facets": {
                "films": film_counts,
                "splits": count_groups(lambda item: item.get("split") or ""),
                "exact": count_groups(
                    lambda item: "yes"
                    if (item.get("recipe") or {}).get("use_exact_target")
                    else "no"
                ),
                "levels": count_groups(level_group),
                "delays": count_groups(delay_group),
                "ducking": count_groups(
                    lambda item: "yes"
                    if float((item.get("recipe") or {}).get("ducking_db", 0.0))
                    > 0.05
                    else "no"
                ),
                "scenarios": count_groups(
                    lambda item: (
                        ((item.get("recipe") or {}).get("short_utterance") or {}).get(
                            "scenario"
                        )
                        or "long_speech"
                    )
                ),
            },
            "page": page,
            "page_size": page_size,
            "page_count": page_count,
            "total_filtered": total_filtered,
            "samples": samples,
        }
    )


@app.get(
    "/api/projects/<project_id>/datasets/<dataset_id>"
    "/examples/<example_id>/audio/<kind>"
)
def api_dataset_example_audio(
    project_id: str, dataset_id: str, example_id: str, kind: str
):
    kinds = {
        "mixture": "mixture.flac",
        "reference": "reference.flac",
        "target_reference": "target_reference.flac",
        "target_voice": "target_voice.flac",
    }
    if kind not in kinds:
        raise ValueError("Неизвестная дорожка примера.")
    dataset_root = cfg_store.project_artifact(
        project_id, f"training/datasets/{dataset_id}"
    )
    expected_parent = (
        cfg_store.project_dir(project_id) / "training" / "datasets"
    ).resolve()
    if dataset_root.parent != expected_parent or dataset_root.name != dataset_id:
        raise ValueError("Недопустимый идентификатор датасета.")
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        return jsonify({"error": "Датасет не найден."}), 404
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = next(
        (
            item
            for item in (manifest.get("examples") or [])
            if str(item.get("id") or "") == example_id
        ),
        None,
    )
    if row is None:
        return jsonify({"error": "Пример датасета не найден."}), 404
    cache_key = hashlib.sha256(
        f"{dataset_id}:{example_id}".encode("utf-8")
    ).hexdigest()[:20]
    preview_root = dataset_root / "inspection_audio" / cache_key
    output = preview_root / kinds[kind]
    with _dataset_preview_lock:
        if not output.is_file():
            import numpy as np
            import soundfile as sf

            from experiments.paired_reference_cancel.recipe_audio import (
                load_recipe_example,
                place_short_utterances,
            )

            mixture, reference, target_reference, target_voice = (
                load_recipe_example(row)
            )
            sample_rate = int(row.get("sample_rate") or manifest.get("sample_rate") or 24000)
            short_profile = (row.get("recipe") or {}).get("short_utterance")
            if short_profile:
                reference, target_reference, target_voice = place_short_utterances(
                    reference,
                    target_reference,
                    target_voice,
                    sample_rate,
                    short_profile,
                )
                mixture = target_reference + target_voice
                peak = max(float(np.max(np.abs(mixture))), 0.98)
                scale = 0.98 / peak
                mixture *= scale
                target_reference *= scale
                target_voice *= scale
            values = {
                "mixture": mixture,
                "reference": reference,
                "target_reference": target_reference,
                "target_voice": target_voice,
            }
            preview_root.mkdir(parents=True, exist_ok=True)
            for name, value in values.items():
                sf.write(
                    str(preview_root / kinds[name]),
                    np.clip(value, -1.0, 1.0),
                    sample_rate,
                    format="FLAC",
                    subtype="PCM_16",
                )
    return send_file(output, conditional=True, mimetype="audio/flac")


@app.get("/api/probe")
def api_probe():
    _path, probe = _probe(request.args.get("path", ""))
    return jsonify(probe)


def _run_windows_dialog(script: str, failure_message: str) -> str:
    """Run a PowerShell dialog and preserve non-ASCII Windows paths.

    Windows Python normally decodes text subprocesses with the active ANSI
    code page (cp1251 on the Russian installation), while the dialog scripts
    deliberately write UTF-8.  Leaving the decoder implicit corrupts every
    newly selected Cyrillic path before validation.
    """
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            (result.stderr or result.stdout or failure_message).strip()
        )
    return (result.stdout or "").strip()


@app.post("/api/pick-file")
def api_pick_file():
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Выберите видео или аудио для DubClean'
$dialog.Filter = 'Видео и аудио|*.mkv;*.mp4;*.avi;*.mov;*.m4v;*.webm;*.wav;*.flac;*.mp3;*.aac;*.m4a|Все файлы|*.*'
$dialog.Multiselect = $false
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Write($dialog.FileName)
}
"""
    selected = _run_windows_dialog(script, "Не удалось открыть выбор файла.")
    if not selected:
        return jsonify({"cancelled": True})
    # The user picked this file through the native OS dialog, so trust its
    # folder for validation (portable config only pre-allows data/uploads).
    cfg_store.authorize_media_dir(selected)
    _path, probe = _probe(selected)
    return jsonify(probe)


@app.post("/api/projects-root")
def api_projects_root():
    if cfg_store.list_tasks(states={"queued", "starting", "running"}):
        raise RuntimeError(
            "Нельзя менять папку проектов, пока выполняются или ожидают запуска задачи."
        )
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Выберите папку хранения проектов DubClean'
$dialog.ShowNewFolderButton = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
  [Console]::Write($dialog.SelectedPath)
}
"""
    selected = _run_windows_dialog(script, "Не удалось открыть выбор папки.")
    if not selected:
        return jsonify({"cancelled": True, "config": _public_config()})
    with _task_mutation_lock:
        if cfg_store.list_tasks(states={"queued", "starting", "running"}):
            raise RuntimeError(
                "Нельзя менять папку проектов, пока выполняются или ожидают запуска задачи."
            )
        path = _set_projects_root(selected)
    return jsonify({"ok": True, "path": str(path), "config": _public_config()})


@app.get("/api/uploads")
def api_uploads():
    items = [
        {"name": path.name, "path": str(path.resolve()), "size": path.stat().st_size}
        for path in sorted(
            cfg_store.uploads_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True
        )
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return jsonify({"items": items})


@app.post("/api/upload")
def api_upload():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        raise ValueError("Файл не передан.")
    name = safe_filename(upload.filename)
    if Path(name).suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError("Допускаются только аудио- и видеофайлы.")
    destination = cfg_store.uploads_root / name
    counter = 1
    while destination.exists():
        destination = cfg_store.uploads_root / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    temporary = cfg_store.uploads_root / f".{destination.name}.{uuid.uuid4().hex}.upload"
    try:
        upload.save(temporary)
        os.replace(temporary, destination)
        _path, probe = _probe(str(destination))
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return jsonify(probe), 201


@app.post("/api/projects/<project_id>/pairs")
def api_add_pair(project_id: str):
    body = request.get_json(silent=True) or {}
    original_path, original_probe = _probe(str(body.get("original_path", "")))
    dubbed_path, dubbed_probe = _probe(str(body.get("dubbed_path", "")))
    original_stream = _validate_stream(original_probe, body.get("original_stream"), "оригинала")
    dubbed_stream = _validate_stream(dubbed_probe, body.get("dubbed_stream"), "перевода")
    pair = cfg_store.add_pair(
        project_id,
        body.get("name", ""),
        original_path,
        dubbed_path,
        original_stream,
        dubbed_stream,
        original_probe,
        dubbed_probe,
    )
    return jsonify({"pair": pair}), 201


@app.get("/api/projects/<project_id>/pairs/<pair_id>")
def api_pair(project_id: str, pair_id: str):
    return jsonify({"pair": _pair_detail(project_id, pair_id)})


@app.patch("/api/projects/<project_id>/pairs/<pair_id>")
def api_replace_pair(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    original_path, original_probe = _probe(str(body.get("original_path", "")))
    dubbed_path, dubbed_probe = _probe(str(body.get("dubbed_path", "")))
    original_stream = _validate_stream(
        original_probe, body.get("original_stream"), "оригинала"
    )
    dubbed_stream = _validate_stream(
        dubbed_probe, body.get("dubbed_stream"), "перевода"
    )
    with _task_mutation_lock:
        active = [
            item
            for item in cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
            if pair_id in _task_pair_ids(item)
        ]
        if active:
            raise RuntimeError(
                "Сначала остановите активную задачу проекта, затем замените файлы или дорожки."
            )
        pair = cfg_store.replace_pair_sources(
            project_id,
            pair_id,
            original_path,
            dubbed_path,
            original_stream,
            dubbed_stream,
            original_probe,
            dubbed_probe,
        )
        requested_source_mode = str(body.get("source_mode") or "").strip().lower()
        inferred_source_mode = (
            "single" if _same_local_path(original_path, dubbed_path) else "pair"
        )
        if requested_source_mode not in {"single", "pair"}:
            requested_source_mode = inferred_source_mode
        # A single-file project is defined by one concrete media file. Never
        # persist an inconsistent mode merely because a stale UI submitted it.
        project = cfg_store.load_project(project_id)
        project["application_source_mode"] = inferred_source_mode
        cfg_store.save_project(project)
    return jsonify({"pair": pair})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/material-variants/<variant_id>/restore")
def api_restore_material_variant(project_id: str, pair_id: str, variant_id: str):
    with _task_mutation_lock:
        active = [
            item
            for item in cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
            if pair_id in _task_pair_ids(item)
        ]
        if active:
            raise RuntimeError("Сначала дождитесь завершения или остановите активную задачу проекта.")
        pair = cfg_store.restore_material_variant(project_id, pair_id, variant_id)
    return jsonify({"pair": pair})


@app.delete("/api/projects/<project_id>/pairs/<pair_id>/material-variants/<variant_id>")
def api_delete_material_variant(project_id: str, pair_id: str, variant_id: str):
    with _task_mutation_lock:
        active = [
            item
            for item in cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
            if pair_id in _task_pair_ids(item)
        ]
        if active:
            raise RuntimeError("Сначала дождитесь завершения или остановите активную задачу проекта.")
        pair = cfg_store.delete_material_variant(project_id, pair_id, variant_id)
    return jsonify({"deleted": True, "pair": pair})


@app.delete("/api/projects/<project_id>/pairs/<pair_id>")
def api_archive_pair(project_id: str, pair_id: str):
    with _task_mutation_lock:
        active = [
            item
            for item in cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
            if pair_id in _task_pair_ids(item)
        ]
        if active:
            raise RuntimeError("Сначала остановите активную задачу этой пары.")
        pair = cfg_store.archive_pair(project_id, pair_id)
    return jsonify({"archived": True, "pair": pair})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/alignment-review")
def api_alignment_review(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    with _task_mutation_lock:
        if _active_tasks_for_pair(project_id, pair_id):
            raise RuntimeError("Сначала остановите активную задачу этой пары.")
        pair = cfg_store.load_pair(project_id, pair_id)
        if pair.get("stages", {}).get("3", {}).get("status") != "completed":
            raise RuntimeError("Карта сопоставления ещё не готова.")
        accepted = bool(body.get("accepted"))
        note = str(body.get("note", "")).strip()[:1000]
        if not accepted:
            pair = cfg_store.invalidate_pair_derivatives(
                project_id,
                pair_id,
                from_stage=4,
                reason="alignment_rejected",
            )
        pair["alignment_review"] = {
            "accepted": accepted,
            "note": note,
            "reviewed_at": utc_now(),
        }
        pair["stages"]["4"] = {
            "status": "completed" if accepted else "error",
            "message": "Сопоставление принято пользователем." if accepted else "Сопоставление отклонено; повторите этап 3.",
        }
        pair["stages"]["5"] = {
            "status": "available" if accepted else "blocked",
            "message": "Можно запустить адаптивное вычитание." if accepted else "Требуется исправить сопоставление.",
        }
        cfg_store.save_pair(pair)
    return jsonify({"pair": pair})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/alignment-correction")
def api_alignment_correction(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    correction_sec = float(body.get("correction_sec", 0.0))
    if not -600.0 <= correction_sec <= 600.0:
        raise ValueError("Ручная поправка должна быть от -600 до +600 секунд.")
    with _task_mutation_lock:
        if _active_tasks_for_pair(project_id, pair_id):
            raise RuntimeError("Сначала остановите активную задачу этой пары.")
        pair = cfg_store.load_pair(project_id, pair_id)
        root = cfg_store.pair_dir(project_id, pair_id)
        map_path = root / "alignment" / "alignment_map.json"
        if not map_path.is_file():
            raise RuntimeError("Карта сопоставления ещё не готова.")
        alignment_map = json.loads(map_path.read_text(encoding="utf-8"))
        correction_sec = round(correction_sec, 3)
        saved_correction_sec = round(
            float(
                alignment_map.get("manual_correction_sec")
                if alignment_map.get("manual_correction_sec") is not None
                else pair.get("alignment_manual_correction_sec", 0.0)
            ),
            3,
        )
        # Saving the value that is already active must be a true no-op.  In
        # particular it must not archive a completed preview/result merely
        # because the user moved the draft slider and then returned it to the
        # original position before continuing.
        if abs(correction_sec - saved_correction_sec) < 0.0005:
            metadata_changed = (
                pair.get("alignment_manual_correction_sec") != saved_correction_sec
                or pair.get("alignment_map") != alignment_map
            )
            if metadata_changed:
                pair["alignment_map"] = alignment_map
                pair["alignment_manual_correction_sec"] = saved_correction_sec
                cfg_store.save_pair(pair)
            return jsonify(
                {
                    "pair": pair,
                    "correction_sec": saved_correction_sec,
                    "changed": False,
                }
            )

        alignment_map["manual_correction_sec"] = correction_sec
        atomic_json(map_path, alignment_map)
        pair = cfg_store.invalidate_pair_derivatives(
            project_id,
            pair_id,
            from_stage=4,
            reason="alignment_correction_changed",
        )
        pair["alignment_map"] = alignment_map
        pair["alignment_manual_correction_sec"] = correction_sec
        pair["alignment_review"] = None
        pair["stages"]["4"] = {
            "status": "available",
            "message": "Ручная поправка сохранена; прослушайте и подтвердите сопоставление.",
        }
        pair["stages"]["5"] = {
            "status": "blocked",
            "message": "После ручной поправки требуется повторное подтверждение сопоставления.",
        }
        cfg_store.save_pair(pair)
    return jsonify({"pair": pair, "correction_sec": correction_sec, "changed": True})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/alignment-auto-correction")
def api_alignment_auto_correction(project_id: str, pair_id: str):
    """Calculate a draft residual shift from sparse scenes across the film."""
    with _task_mutation_lock:
        if _active_tasks_for_pair(project_id, pair_id):
            raise RuntimeError("Сначала остановите активную задачу этой пары.")
        pair = cfg_store.load_pair(project_id, pair_id)
        root = cfg_store.pair_dir(project_id, pair_id)
        map_path = root / "alignment" / "alignment_map.json"
        if not map_path.is_file():
            raise RuntimeError("Карта сопоставления ещё не готова.")
        original_proxy = root / "extracted" / "original_proxy.wav"
        dubbed_proxy = root / "extracted" / "dubbed_proxy.wav"
        if not original_proxy.is_file() or not dubbed_proxy.is_file():
            raise RuntimeError("Извлечённые дорожки для автоматического выравнивания не найдены.")
        alignment_map = json.loads(map_path.read_text(encoding="utf-8"))
        estimate = estimate_sparse_manual_correction(
            original_proxy,
            dubbed_proxy,
            alignment_map,
            dict(cfg_store.cfg.get("alignment") or {}),
        )
    return jsonify({"ok": True, **estimate})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/model-review")
def api_model_review(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    pair = cfg_store.load_pair(project_id, pair_id)
    if pair.get("stages", {}).get("9", {}).get("status") not in {"available", "completed"}:
        raise RuntimeError("Сначала подготовьте датасет и проверьте контрольные сцены.")
    pair["model_review"] = {
        "method": str(body.get("method", "baseline")),
        "note": str(body.get("note", ""))[:1000],
        "reviewed_at": utc_now(),
    }
    pair["stages"]["9"] = {"status": "completed", "message": "Готовая компонентная цепочка принята после проверки."}
    pair["stages"]["10"] = {"status": "available", "message": "Можно собрать итоговую дорожку RU + M&E."}
    cfg_store.save_pair(pair)
    return jsonify({"pair": pair})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/speech-test-review")
def api_speech_test_review(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    pair = cfg_store.load_pair(project_id, pair_id)
    if not pair.get("speech_test"):
        raise RuntimeError("Сначала выполните пробное выделение EN-речи.")
    accepted = bool(body.get("accepted"))
    pair["speech_test_review"] = {
        "accepted": accepted,
        "reviewed_at": utc_now(),
        "message": (
            "Проба одобрена. Теперь можно обработать полный оригинал."
            if accepted
            else "Проба отклонена. Выберите другой участок и повторите."
        ),
    }
    pair["stages"]["5"] = {
        "status": "available",
        "message": pair["speech_test_review"]["message"],
    }
    pair["stages"]["6"] = {
        "status": "available" if accepted else "blocked",
        "message": (
            "Создайте пять разговорных превью и сравните методы A и B."
            if accepted
            else "Сначала подтвердите пробное разделение оригинала на этапе 5."
        ),
    }
    cfg_store.save_pair(pair)
    return jsonify({"pair": pair})


@app.post("/api/projects/<project_id>/pairs/<pair_id>/components-test-review")
def api_components_test_review(project_id: str, pair_id: str):
    body = request.get_json(silent=True) or {}
    pair = cfg_store.load_pair(project_id, pair_id)
    if not pair.get("components_test"):
        raise RuntimeError("Сначала подготовьте пять сравнительных превью этапа 6.")
    selected_method = str(body.get("method") or "")
    if selected_method not in {"direct", "stem"}:
        raise ValueError("Выберите метод A или метод B.")
    pair["components_test_review"] = {
        "accepted": True,
        "selected_method": selected_method,
        "reviewed_at": utc_now(),
        "message": (
            "Выбран метод A: прямое вычитание из исходной дорожки."
            if selected_method == "direct"
            else "Выбран метод B: очистка речевого стема и сборка с M&E."
        ),
    }
    pair["stages"]["6"] = {
        "status": "available",
        "message": pair["components_test_review"]["message"],
    }
    cfg_store.save_pair(pair)
    return jsonify({"pair": pair})


_OP_REQUIREMENTS = {
    "extract": (2, {"available", "error", "completed"}),
    "align": (2, {"completed"}),
    "speech_test": (4, {"completed"}),
    "speech_extract": (4, {"completed"}),
    "target_speech_extract": (4, {"completed"}),
    "components_test": (4, {"completed"}),
    "components_build": (4, {"completed"}),
    "preview": (6, {"completed"}),
    "finalize": (6, {"completed"}),
}


def _ensure_pair_operation(pair: dict, operation: str) -> None:
    if operation not in _OP_REQUIREMENTS:
        raise ValueError("Неизвестная операция для пары.")
    stage, allowed = _OP_REQUIREMENTS[operation]
    status = pair.get("stages", {}).get(str(stage), {}).get("status")
    if status not in allowed:
        message = pair.get("stages", {}).get(str(stage), {}).get("message", "")
        raise RuntimeError(f"Действие пока недоступно. {message}")
    if operation == "speech_extract" and not pair.get("speech_test"):
        raise RuntimeError("Сначала выполните пробное выделение EN-речи.")
    if operation == "speech_extract" and not pair.get("speech_test_review", {}).get("accepted"):
        raise RuntimeError("Сначала прослушайте и явно подтвердите результат пробы.")
    if operation == "components_test" and not pair.get("speech_test"):
        raise RuntimeError("Сначала выполните пробное выделение EN-речи.")
    if operation == "components_build" and not pair.get("components_test"):
        raise RuntimeError("Сначала создайте и прослушайте пять сравнительных превью.")
    if operation == "components_build" and not pair.get(
        "components_test_review", {}
    ).get("accepted"):
        raise RuntimeError("Сначала прослушайте пять превью и выберите метод A или B.")
    if operation == "components_build" and pair.get(
        "components_test_review", {}
    ).get("selected_method") not in {"direct", "stem"}:
        raise RuntimeError("Для сборки не выбран метод A или B.")


def _ensure_no_duplicate(project_id: str, pair_id: str | None, operation: str) -> None:
    active = cfg_store.list_tasks(project_id, {"queued", "starting", "running"})
    for task in active:
        if pair_id and pair_id in _task_pair_ids(task):
            raise RuntimeError("Для этой пары уже выполняется или ожидает запуска другая операция.")
        if operation == "train" and task["operation"] == "train":
            raise RuntimeError("Обучение уже выполняется или ожидает запуска.")
        if pair_id is None and task["operation"] == operation:
            raise RuntimeError("Такая проектная операция уже выполняется или ожидает запуска.")


@app.post("/api/projects/<project_id>/operations")
def api_operations(project_id: str):
    with _task_mutation_lock:
        return _api_operations_locked(project_id)


def _api_operations_locked(project_id: str):
    body = request.get_json(silent=True) or {}
    operation = str(body.get("operation", ""))
    pair_ids = list(dict.fromkeys(str(item) for item in (body.get("pair_ids") or [])))
    parameters = body.get("parameters") or {}
    tasks = []
    pair_errors = []
    if operation in _OP_REQUIREMENTS:
        if not pair_ids:
            raise ValueError("Не выбрана ни одна пара.")
        for pair_id in pair_ids:
            pair_name = pair_id
            try:
                pair = cfg_store.load_pair(project_id, pair_id)
                pair_name = pair.get("name", pair_id)
                _ensure_pair_operation(pair, operation)
                _ensure_no_duplicate(project_id, pair_id, operation)
                if operation == "extract":
                    cfg_store.invalidate_pair_derivatives(
                        project_id,
                        pair_id,
                        from_stage=2,
                        reason="audio_extraction_restarted",
                    )
                elif operation == "align":
                    cfg_store.invalidate_pair_derivatives(
                        project_id,
                        pair_id,
                        from_stage=3,
                        reason="alignment_restarted",
                    )
                tasks.append(
                    cfg_store.create_task(project_id, operation, pair_id, parameters)
                )
            except (RuntimeError, ValueError) as error:
                pair_errors.append(
                    {
                        "pair_id": pair_id,
                        "pair_name": pair_name,
                        "error": str(error),
                    }
                )
    elif operation == "dataset":
        if not pair_ids:
            raise ValueError("Не выбрана ни одна пара для датасета.")
        for pair_id in pair_ids:
            pair = cfg_store.load_pair(project_id, pair_id)
            if pair.get("stages", {}).get("4", {}).get("status") != "completed":
                raise RuntimeError(f"Для «{pair['name']}» сначала подтвердите сопоставление.")
            _ensure_no_duplicate(project_id, pair_id, operation)
        _ensure_no_duplicate(project_id, None, operation)
        tasks.append(
            cfg_store.create_task(
                project_id, operation, None, {**parameters, "pair_ids": pair_ids}
            )
        )
    elif operation == "train":
        project = cfg_store.load_project(project_id)
        dataset_id = str(parameters.get("dataset_id") or project.get("latest_dataset_id") or "")
        if not dataset_id:
            raise RuntimeError("Сначала подготовьте датасет.")
        manifest_path = (
            cfg_store.project_dir(project_id)
            / "training" / "datasets" / dataset_id / "manifest.json"
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError("Манифест выбранного датасета не найден.")
        if manifest.get("target_kind") not in {
            "paired_speech_reference_subtraction_supervised_v2",
            "clean_translation_reference_subtraction_supervised_v3",
        }:
            raise RuntimeError(
                "Выбран старый датасет без точной цели. Подготовьте новый датасет "
                "парного вычитания."
            )
        _ensure_no_duplicate(project_id, None, operation)
        tasks.append(cfg_store.create_task(project_id, operation, None, parameters))
    else:
        raise ValueError("Неизвестная операция.")
    if not tasks and pair_errors:
        return jsonify(
            {"tasks": [], "pair_errors": pair_errors, "error": pair_errors[0]["error"]}
        ), 409
    scheduler_tick()
    return jsonify({"tasks": tasks, "pair_errors": pair_errors}), 207 if pair_errors else 202


@app.get("/api/projects/<project_id>/tasks")
def api_tasks(project_id: str):
    scheduler_tick()
    return jsonify({"tasks": [_task_with_log(item) for item in cfg_store.list_tasks(project_id)]})


@app.get("/api/projects/<project_id>/tasks/<task_id>")
def api_task(project_id: str, task_id: str):
    scheduler_tick()
    return jsonify(
        {"task": _task_with_log(cfg_store.load_task(project_id, task_id), include_result=True)}
    )


@app.post("/api/projects/<project_id>/tasks/<task_id>/stop")
def api_stop_task(project_id: str, task_id: str):
    with _task_mutation_lock:
        task = cfg_store.load_task(project_id, task_id)
        if task["state"] not in {"queued", "starting", "running"}:
            return jsonify({"stopped": False, "message": "Задача уже не выполняется."})
        _stop_task_record(task)
    return jsonify({"stopped": True})


@app.post("/api/stop-all")
def api_stop_all():
    stopped = []
    with _task_mutation_lock:
        for task in cfg_store.list_tasks(states={"queued", "starting", "running"}):
            _stop_task_record(task, substage="Сервис остановлен.")
            stopped.append(task["id"])
    return jsonify({"stopped": stopped})


def _safe_public_file(value: str) -> Path:
    if not value:
        raise ValueError("Путь не передан.")
    path = Path(value).resolve()
    roots = [
        cfg_store.projects_root,
        cfg_store.uploads_root,
        *cfg_store.allowed_media_roots,
    ]
    if not path.is_file() or not is_within(path, roots):
        raise ValueError("Файл не найден или путь запрещён.")
    return path


@app.get("/media")
def media():
    try:
        path = _safe_public_file(request.args.get("path", ""))
    except ValueError:
        return "Не найдено", 404
    return send_file(path, conditional=True)


@app.get("/api/browser-video")
def browser_video():
    """Stream a browser-compatible MP4 for source and final-film review.

    Source containers in real projects are commonly AVI/MKV with MPEG-4 video,
    AC3 or FLAC audio, which Chromium cannot play reliably.  A fragmented MP4
    starts playing while ffmpeg is still converting it, so a full film does not
    have to be duplicated on disk before the operator can review it.  Short
    source-file previews use the same endpoint with ``duration`` set.
    """
    try:
        video = _safe_public_file(request.args.get("path", ""))
        audio_value = str(request.args.get("audio", "") or "").strip()
        audio = _safe_public_file(audio_value) if audio_value else None
        stream_value = request.args.get("stream")
        stream_index = int(stream_value) if stream_value not in (None, "") else None
        if stream_index is not None:
            streams = probe_audio_streams(video)
            if stream_index not in {int(item["index"]) for item in streams}:
                raise ValueError("Выбранная аудиодорожка отсутствует в видео.")
        start = max(0.0, float(request.args.get("start", "0") or 0.0))
        duration_value = str(request.args.get("duration", "") or "").strip()
        duration = (
            min(180.0, max(3.0, float(duration_value)))
            if duration_value
            else None
        )
        ffmpeg = check_ffmpeg()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return str(error), 400

    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
    ]
    if start:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(video)]
    if audio is not None:
        if start:
            command += ["-ss", f"{start:.3f}"]
        command += ["-i", str(audio)]
    command += ["-map", "0:v:0"]
    if audio is not None:
        command += ["-map", "1:a:0"]
    elif stream_index is not None:
        command += ["-map", f"0:{stream_index}"]
    else:
        command += ["-map", "0:a:0"]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    command += [
        "-vf", "scale='min(1280,iw)':-2:flags=fast_bilinear",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-crf", "25", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000",
    ]

    # Native browser seeking requires a finalized MP4 with a stable duration and
    # byte ranges. This applies to full-film review too: fragmented stdout makes
    # Chromium expose only the currently buffered fragment as the whole movie.
    fingerprints = [
        str(video), video.stat().st_size, video.stat().st_mtime_ns,
        str(audio or ""),
        audio.stat().st_size if audio is not None else 0,
        audio.stat().st_mtime_ns if audio is not None else 0,
        stream_index, round(start, 3), round(duration, 3) if duration is not None else None,
        "seekable-v2",
    ]
    cache_key = hashlib.sha256(
        json.dumps(fingerprints, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_root = RUNTIME_DIR / "browser_video_cache"
    output = cache_root / f"{cache_key}.mp4"
    with _browser_video_lock:
        if not output.is_file() or output.stat().st_size < 1024:
            cache_root.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.partial.mp4")
            render_command = command + [
                "-movflags", "+faststart", "-f", "mp4", "-y", str(temporary)
            ]
            result = subprocess.run(
                render_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode or not temporary.is_file() or temporary.stat().st_size < 1024:
                temporary.unlink(missing_ok=True)
                return (result.stderr.strip() or "Не удалось подготовить видео для браузера."), 500
            os.replace(temporary, output)
    response = send_file(output, mimetype="video/mp4", conditional=True, max_age=3600)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# Segment length of the on-demand HLS ladder.  The whole film is exposed to the
# browser as a VOD playlist of fixed-length segments, but each segment is only
# transcoded (and cached) the first time hls.js actually requests it — the
# operator can seek anywhere and only the touched 45-second windows are built.
HLS_SEGMENT_SEC = 45.0


_HLS_AV_MODES = ("both", "video", "audio")


def _hls_media_arguments(validate_stream: bool = True):
    """Parse and validate the shared source/track parameters of an HLS request.

    ``av`` selects the elementary stream(s):
      * ``both``  — video + audio (default; full self-contained preview);
      * ``video`` — picture only (``-an``); faster, shared across every track;
      * ``audio`` — sound only (``-vn``); the swappable per-track element.
    Splitting picture and sound is what lets the assembly screen switch tracks
    instantly: the muted picture element never reloads, only the tiny audio
    playlist does.

    ``validate_stream`` runs an ffprobe to confirm the stream exists.  It is
    done once when the playlist is built, but MUST be skipped for every segment
    request: an ffprobe over a multi-hundred-MB source on a slow disk takes
    seconds, and doing it per 45-second segment was the real cause of sluggish
    seeking (the transcode itself is a fraction of a second).
    """
    video = _safe_public_file(request.args.get("path", ""))
    audio_value = str(request.args.get("audio", "") or "").strip()
    audio = _safe_public_file(audio_value) if audio_value else None
    stream_value = request.args.get("stream")
    stream_index = int(stream_value) if stream_value not in (None, "") else None
    if validate_stream and stream_index is not None and audio is None:
        streams = probe_audio_streams(video)
        if stream_index not in {int(item["index"]) for item in streams}:
            raise ValueError("Выбранная аудиодорожка отсутствует в видео.")
    av = str(request.args.get("av", "both") or "both").strip().lower()
    if av not in _HLS_AV_MODES:
        raise ValueError(f"Недопустимый режим av={av!r}.")
    return video, audio, audio_value, stream_index, av


@app.get("/api/hls-playlist")
def hls_playlist():
    """VOD playlist covering the whole film as fixed 45-second segments.

    ``#EXT-X-PLAYLIST-TYPE:VOD`` + ``#EXT-X-ENDLIST`` give hls.js the full
    duration up front, so the native seek bar spans the entire film and the
    operator can jump anywhere; only the requested segments are ever built.
    """
    try:
        video, audio, audio_value, stream_index, av = _hls_media_arguments()
        total = float(probe_duration(video) or 0.0)
        if total <= 0.0:
            raise ValueError("Не удалось определить длительность видео.")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return str(error), 400

    count = max(1, int(math.ceil(total / HLS_SEGMENT_SEC)))
    base_params: list[tuple[str, str]] = [("path", request.args.get("path", ""))]
    if audio is not None:
        base_params.append(("audio", audio_value))
    elif stream_index is not None:
        base_params.append(("stream", str(stream_index)))
    if av != "both":
        base_params.append(("av", av))
    revision = str(request.args.get("v", "") or "").strip()
    if revision:
        # Carry the UI's result revision into every segment URL.  The backend
        # cache already fingerprints file size and mtime, while this query
        # value prevents a browser from reusing a one-hour-old segment after
        # result.mkv or a FLAC track has been rebuilt at the same path.
        base_params.append(("v", revision[:256]))

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-TARGETDURATION:{int(math.ceil(HLS_SEGMENT_SEC))}",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]
    for index in range(count):
        if index < count - 1:
            seg_duration = HLS_SEGMENT_SEC
        else:
            seg_duration = max(0.1, total - HLS_SEGMENT_SEC * (count - 1))
        query = urllib.parse.urlencode(base_params + [("index", str(index))])
        lines.append(f"#EXTINF:{seg_duration:.3f},")
        lines.append(f"/api/hls-segment?{query}")
    lines.append("#EXT-X-ENDLIST")

    response = Response(
        "\n".join(lines) + "\n",
        mimetype="application/vnd.apple.mpegurl",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/hls-segment")
def hls_segment():
    """Transcode (and cache) one 45-second MPEG-TS segment on demand.

    Continuous output timestamps (``-output_ts_offset``) and a forced keyframe
    at the segment start keep the segments stitchable into one seamless VOD
    timeline while each stays independently decodable.
    """
    try:
        # Skip the per-segment ffprobe (validated once at playlist time): it is
        # the slow part, not the transcode.
        video, audio, _audio_value, stream_index, av = _hls_media_arguments(
            validate_stream=False
        )
        index = int(request.args.get("index", "0"))
        if index < 0:
            raise ValueError("Отрицательный индекс сегмента.")
        ffmpeg = check_ffmpeg()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return str(error), 400

    start = index * HLS_SEGMENT_SEC
    want_video = av in ("both", "video")
    want_audio = av in ("both", "audio")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video)]
    if want_audio and audio is not None:
        command += ["-ss", f"{start:.3f}", "-i", str(audio)]
    if want_video:
        command += ["-map", "0:v:0"]
    if want_audio:
        if audio is not None:
            command += ["-map", "1:a:0"]
        elif stream_index is not None:
            command += ["-map", f"0:{stream_index}"]
        else:
            command += ["-map", "0:a:0"]
    command += ["-t", f"{HLS_SEGMENT_SEC:.3f}"]
    if want_video:
        command += [
            "-vf", "scale='min(1280,iw)':-2:flags=fast_bilinear",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-crf", "25", "-pix_fmt", "yuv420p",
            "-force_key_frames", "expr:gte(t,0)",
        ]
    else:
        command += ["-vn"]
    if want_audio:
        command += ["-c:a", "aac", "-b:a", "160k", "-ac", "2", "-ar", "48000"]
    else:
        command += ["-an"]
    command += [
        "-output_ts_offset", f"{start:.3f}",
        "-muxdelay", "0", "-muxpreload", "0",
    ]

    fingerprints = [
        str(video), video.stat().st_size, video.stat().st_mtime_ns,
        str(audio or ""),
        audio.stat().st_size if audio is not None else 0,
        audio.stat().st_mtime_ns if audio is not None else 0,
        stream_index, index, round(HLS_SEGMENT_SEC, 3),
        av,
        "hls-ts-v2",
    ]
    cache_key = hashlib.sha256(
        json.dumps(fingerprints, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_root = RUNTIME_DIR / "hls_segment_cache"
    output = cache_root / f"{cache_key}.ts"
    with _hls_key_lock(cache_key):
        if not output.is_file() or output.stat().st_size < 512:
            cache_root.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.name}.{os.getpid()}.partial.ts")
            render_command = command + ["-f", "mpegts", "-y", str(temporary)]
            result = subprocess.run(
                render_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode or not temporary.is_file() or temporary.stat().st_size < 512:
                temporary.unlink(missing_ok=True)
                return (result.stderr.strip() or "Не удалось подготовить сегмент видео."), 500
            os.replace(temporary, output)
    response = send_file(output, mimetype="video/mp2t", conditional=True, max_age=3600)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/preview-mix")
def preview_mix():
    """Render the exact full-pipeline mix and, when requested, protect songs."""
    try:
        video = _safe_public_file(request.args.get("video", ""))
        voice = _safe_public_file(request.args.get("voice", ""))
        background = _safe_public_file(request.args.get("background", ""))
        voice_gain = max(-12.0, min(12.0, float(request.args.get("voice_gain", "0"))))
        background_gain = max(
            -12.0, min(12.0, float(request.args.get("background_gain", "0")))
        )
        delay = max(-2.0, min(2.0, float(request.args.get("delay", "0"))))
        protect_song = str(
            request.args.get("song_protection", "0") or "0"
        ).strip().casefold() in {"1", "true", "yes", "on"}
        song_report: Path | None = None
        song_original: Path | None = None
        song_dubbed: Path | None = None
        song_dubbed_speech: Path | None = None
        song_detection_reference: Path | None = None
        song_detection_voice: Path | None = None
        song_translation_voice: Path | None = None
        song_payload: dict[str, Any] = {}
        selected_variant_name = ""
        sync_mode = "off"
        selected_voice_synchronized = False
        song_cfg: dict[str, Any] = {}
        if protect_song:
            song_report = _safe_public_file(
                request.args.get("song_report", "")
            )
            song_payload_value = read_json(song_report, {})
            if not isinstance(song_payload_value, dict):
                raise ValueError("Некорректный отчёт защиты песни.")
            song_payload = song_payload_value
            report_schema_version = song_payload.get("schema_version")
            if (
                song_report.name
                != song_protection.PREVIEW_CONTRACT_NAME
                or not isinstance(report_schema_version, int)
                or isinstance(report_schema_version, bool)
                or report_schema_version
                != song_protection.PREVIEW_CONTRACT_SCHEMA_VERSION
                or song_payload.get("report_kind")
                != song_protection.PREVIEW_CONTRACT_KIND
                or song_payload.get("report_name")
                != song_protection.PREVIEW_CONTRACT_NAME
            ):
                raise ValueError(
                    "Некорректная версия или имя отчёта защиты песни."
                )
            scene_directory_value = song_payload.get("scene_directory")
            if not isinstance(scene_directory_value, str):
                raise ValueError("В отчёте не указана папка сцены.")
            scene_directory = Path(scene_directory_value).resolve()
            if not _same_local_path(scene_directory, song_report.parent):
                raise ValueError(
                    "Отчёт защиты песни не соответствует папке сцены."
                )
            sources = song_payload.get("sources")
            variants = song_payload.get("variants")
            if not isinstance(sources, dict) or not isinstance(variants, dict):
                raise ValueError("Некорректная структура отчёта защиты песни.")
            song_original = _safe_public_file(
                request.args.get("song_original", "")
            )
            song_dubbed = _safe_public_file(
                request.args.get("song_dubbed", "")
            )
            song_dubbed_speech = _safe_public_file(
                str(sources.get("dubbed_speech_stem") or "")
            )
            contract_original = _safe_public_file(
                str(sources.get("original_detection_mix") or "")
            )
            contract_dubbed = _safe_public_file(
                str(sources.get("dubbed_mix") or "")
            )
            contract_background = _safe_public_file(
                str(sources.get("music_background") or "")
            )
            if (
                not _same_local_path(song_original, contract_original)
                or not _same_local_path(song_dubbed, contract_dubbed)
                or not _same_local_path(background, contract_background)
            ):
                raise ValueError(
                    "Отчёт защиты песни не соответствует файлам этой сцены."
                )
            song_detection_voice = _safe_public_file(
                request.args.get("song_detection_voice", "")
            )
            song_translation_voice = _safe_public_file(
                request.args.get("song_translation_voice", "")
            )
            sync_mode = str(
                request.args.get("sync_mode", "off") or "off"
            ).strip().casefold()
            if sync_mode not in {"off", "manual", "conservative"}:
                raise ValueError("Недопустимый режим синхронизации.")
            if sync_mode != "manual" and abs(delay) > 1e-6:
                raise ValueError(
                    "Сдвиг речи допустим только в ручном режиме."
                )
            matches: list[
                tuple[str, dict[str, Any], dict[str, Any]]
            ] = []
            for variant_name, variant_value in variants.items():
                if not isinstance(variant_value, dict):
                    continue
                detection_value = str(
                    variant_value.get("detection_voice") or ""
                )
                translation_value = str(
                    variant_value.get("translation_voice") or ""
                )
                if (
                    not detection_value
                    or not translation_value
                    or not _same_local_path(
                        Path(detection_value).resolve(),
                        song_detection_voice,
                    )
                    or not _same_local_path(
                        Path(translation_value).resolve(),
                        song_translation_voice,
                    )
                ):
                    continue
                for playback_value in (
                    variant_value.get("playback_voices") or []
                ):
                    if not isinstance(playback_value, dict):
                        continue
                    playback_path = str(
                        playback_value.get("path") or ""
                    )
                    allowed_modes = playback_value.get(
                        "allowed_sync_modes"
                    )
                    if (
                        playback_path
                        and isinstance(allowed_modes, list)
                        and sync_mode in allowed_modes
                        and _same_local_path(
                            Path(playback_path).resolve(),
                            voice,
                        )
                    ):
                        matches.append(
                            (
                                str(variant_name),
                                variant_value,
                                playback_value,
                            )
                        )
            if len(matches) != 1:
                raise ValueError(
                    "Выбранные файлы речи не соответствуют одному "
                    "варианту этой сцены."
                )
            (
                selected_variant_name,
                selected_variant,
                selected_playback,
            ) = matches[0]
            song_detection_reference = _safe_public_file(
                str(selected_variant.get("detection_reference") or "")
            )
            # Revalidate contract paths themselves instead of trusting that a
            # matching query path happened to be valid.
            contract_detection_voice = _safe_public_file(
                str(selected_variant.get("detection_voice") or "")
            )
            contract_translation_voice = _safe_public_file(
                str(selected_variant.get("translation_voice") or "")
            )
            if (
                not _same_local_path(
                    contract_detection_voice,
                    song_detection_voice,
                )
                or not _same_local_path(
                    contract_translation_voice,
                    song_translation_voice,
                )
            ):
                raise ValueError(
                    "Выбранный вариант не соответствует отчёту сцены."
                )
            selected_voice_synchronized = bool(
                selected_playback.get("synchronized")
            )
            if selected_voice_synchronized != (
                sync_mode == "conservative"
            ):
                raise ValueError(
                    "Режим синхронизации не соответствует файлу речи."
                )
            protected_paths = [
                song_report,
                video,
                voice,
                background,
                song_original,
                song_dubbed,
                song_dubbed_speech,
                song_detection_reference,
                song_detection_voice,
                song_translation_voice,
            ]
            if any(
                not _same_local_path(path.parent, scene_directory)
                for path in protected_paths
            ):
                raise ValueError(
                    "Все файлы защиты песни должны находиться "
                    "в одной папке сцены."
                )
            raw_song_cfg = song_payload.get("config")
            if not isinstance(raw_song_cfg, dict):
                raise ValueError(
                    "В отчёте отсутствуют настройки защиты песни."
                )
            song_cfg = song_protection.resolved_config(raw_song_cfg)
            song_cfg["dialogue_timeline_guard_sec"] = (
                song_protection.dialogue_timeline_guard(
                    song_cfg,
                    manual_voice_delay_sec=(
                        delay if sync_mode == "manual" else 0.0
                    ),
                    synchronize_speech=(
                        sync_mode == "conservative"
                    ),
                )
            )
        ffmpeg = check_ffmpeg()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return str(error), 400

    def fingerprint(path: Path | None) -> list[Any] | None:
        if path is None:
            return None
        return [
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ]

    identity = {
        "contract_version": (
            "preview-mix-v7-neutral-song-decision-gains"
        ),
        "video": fingerprint(video),
        "voice": fingerprint(voice),
        "background": fingerprint(background),
        "voice_gain": round(voice_gain, 2),
        "background_gain": round(background_gain, 2),
        "delay": round(delay, 3),
        "song_protection": bool(protect_song),
        "song_report": fingerprint(song_report),
        "song_original": fingerprint(song_original),
        "song_dubbed": fingerprint(song_dubbed),
        "song_dubbed_speech": fingerprint(song_dubbed_speech),
        "song_detection_reference": fingerprint(
            song_detection_reference
        ),
        "song_detection_voice": fingerprint(song_detection_voice),
        "song_translation_voice": fingerprint(song_translation_voice),
        "song_variant": selected_variant_name,
        "sync_mode": sync_mode if protect_song else "",
    }
    key = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_dir = RUNTIME_DIR / "preview_mix_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{key}.mp4"
    dynamic_report = cache_dir / f"{key}.song_report.json"

    if not output.is_file():
        with _preview_mix_lock:
            if not output.is_file():
                partial = cache_dir / f"{key}.partial.mp4"
                partial.unlink(missing_ok=True)
                unprotected_audio = cache_dir / (
                    f".{key}.{os.getpid()}.unprotected.flac"
                )
                detection_audio = cache_dir / (
                    f".{key}.{os.getpid()}.detection.flac"
                )
                protected_audio = cache_dir / (
                    f".{key}.{os.getpid()}.protected.flac"
                )
                for temporary_audio in (
                    unprotected_audio,
                    detection_audio,
                    protected_audio,
                ):
                    temporary_audio.unlink(missing_ok=True)
                restoration_segments: list[dict[str, Any]] = []
                try:
                    audio_mix.mix_audio_files(
                        background,
                        voice,
                        unprotected_audio,
                        background_gain_db=background_gain,
                        voice_gain_db=voice_gain,
                        voice_delay_sec=delay,
                    )
                    audio_for_mux = unprotected_audio
                    if protect_song:
                        assert song_original is not None
                        assert song_dubbed is not None
                        assert song_dubbed_speech is not None
                        assert song_detection_reference is not None
                        assert song_detection_voice is not None
                        assert song_translation_voice is not None
                        audio_mix.mix_audio_files(
                            background,
                            song_detection_voice,
                            detection_audio,
                            # User volume choices affect both A/B playback
                            # variants, but must not manufacture apparent song
                            # damage in the detector itself.
                            background_gain_db=0.0,
                            voice_gain_db=0.0,
                            voice_delay_sec=(
                                delay
                                if sync_mode == "manual"
                                else 0.0
                            ),
                        )
                        detection_report = (
                            song_protection.detect_song_intervals(
                                song_original,
                                song_dubbed,
                                song_dubbed_speech,
                                background,
                                detection_audio,
                                song_cfg,
                                report_path=dynamic_report,
                                original_speech_stem=(
                                    song_detection_reference
                                ),
                                translation_voice_stem=(
                                    song_translation_voice
                                ),
                            )
                        )
                        restoration_segments = [
                            item
                            for item in (
                                detection_report.get(
                                    "restoration_segments"
                                )
                                or []
                            )
                            if isinstance(item, dict)
                            and float(item.get("end_sec") or 0.0)
                            > float(item.get("start_sec") or 0.0)
                            and str(
                                item.get("restore_source") or ""
                            )
                            in {"original_aligned", "dubbed"}
                        ]
                        detection_report["preview_dynamic"] = {
                            "variant": selected_variant_name,
                            "sync_mode": sync_mode,
                            "playback_voice": str(voice.resolve()),
                            "detection_voice": str(
                                song_detection_voice.resolve()
                            ),
                            "translation_voice": str(
                                song_translation_voice.resolve()
                            ),
                            "dialogue_timeline_guard_sec": float(
                                song_cfg[
                                    "dialogue_timeline_guard_sec"
                                ]
                            ),
                            "playback_background_gain_db": background_gain,
                            "playback_voice_gain_db": voice_gain,
                            "detection_background_gain_db": 0.0,
                            "detection_voice_gain_db": 0.0,
                            "voice_delay_sec": delay,
                        }
                        atomic_json(dynamic_report, detection_report)
                    if restoration_segments:
                        song_protection.apply_song_protection(
                            unprotected_audio,
                            song_dubbed,
                            protected_audio,
                            restoration_segments,
                            original_mix=song_original,
                            crossfade_sec=float(
                                song_cfg.get("crossfade_sec", 0.35)
                            ),
                        )
                        audio_for_mux = protected_audio
                    command = [
                        ffmpeg,
                        "-y",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(video),
                        "-i",
                        str(audio_for_mux),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-shortest",
                        "-movflags",
                        "+faststart",
                        str(partial),
                    ]
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        creationflags=getattr(
                            subprocess, "CREATE_NO_WINDOW", 0
                        ),
                    )
                except Exception as error:
                    partial.unlink(missing_ok=True)
                    return (
                        "Не удалось подготовить защищённый звук превью. "
                        + str(error),
                        500,
                    )
                finally:
                    for temporary_audio in (
                        unprotected_audio,
                        detection_audio,
                        protected_audio,
                    ):
                        temporary_audio.unlink(missing_ok=True)
                if result.returncode or not partial.is_file() or partial.stat().st_size == 0:
                    partial.unlink(missing_ok=True)
                    message = result.stderr.decode("utf-8", errors="replace")[-1200:]
                    return "Не удалось пересобрать превью. " + message, 500
                partial.replace(output)
    response = send_file(
        output,
        mimetype="video/mp4",
        conditional=True,
        max_age=3600,
    )
    cached_report = (
        read_json(dynamic_report, {})
        if protect_song and dynamic_report.is_file()
        else {}
    )
    restoration_segments = (
        cached_report.get("restoration_segments") or []
        if isinstance(cached_report, dict)
        else []
    )
    response.headers["X-DubClean-Song-Protection"] = (
        "applied"
        if restoration_segments
        else ("no-restoration-segments" if protect_song else "disabled")
    )
    return response


@app.get("/api/preview-track-video")
def preview_track_video():
    """Render a short preview video with one selected audio track.

    This is used by the preview screen when the user switches between
    "RU speech", "M&E", "translation before cleanup" and "EN reference".
    """
    try:
        video = _safe_public_file(request.args.get("video", ""))
        audio = _safe_public_file(request.args.get("audio", ""))
        gain = max(-12.0, min(12.0, float(request.args.get("gain", "0"))))
        delay = max(-2.0, min(2.0, float(request.args.get("delay", "0"))))
        ffmpeg = check_ffmpeg()
    except (OSError, RuntimeError, ValueError) as error:
        return str(error), 400

    identity = {
        "video": [str(video), video.stat().st_size, video.stat().st_mtime_ns],
        "audio": [str(audio), audio.stat().st_size, audio.stat().st_mtime_ns],
        "gain": round(gain, 2),
        "delay": round(delay, 3),
    }
    key = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_dir = RUNTIME_DIR / "preview_track_video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{key}.mp4"

    if not output.is_file():
        with _preview_mix_lock:
            if not output.is_file():
                partial = cache_dir / f"{key}.partial.mp4"
                partial.unlink(missing_ok=True)
                if delay >= 0:
                    timing = f"adelay={int(round(delay * 1000))}:all=1"
                else:
                    timing = f"atrim=start={abs(delay):.3f},asetpts=PTS-STARTPTS"
                filters = (
                    f"[1:a]volume={gain:.2f}dB,{timing},"
                    "apad,alimiter=limit=0.98[aout]"
                )
                command = [
                    ffmpeg,
                    "-y",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-i",
                    str(audio),
                    "-filter_complex",
                    filters,
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode or not partial.is_file() or partial.stat().st_size == 0:
                    partial.unlink(missing_ok=True)
                    message = result.stderr.decode("utf-8", errors="replace")[-1200:]
                    return "Не удалось пересобрать видео с выбранной дорожкой. " + message, 500
                partial.replace(output)
    return send_file(output, mimetype="video/mp4", conditional=True, max_age=3600)


@app.get("/api/source-preview")
def api_source_preview():
    """Return a seekable short MP3 of one selected source audio stream.

    Lets the upload screen confirm the language of a selected track before
    extraction. The path is validated against the allowed media roots (which
    include folders the user opened through the native file picker).
    """
    try:
        path = cfg_store.validate_media_path(request.args.get("path", ""))
        stream_index = int(request.args.get("stream", "0"))
        streams = probe_audio_streams(path)
        if stream_index not in {int(item["index"]) for item in streams}:
            raise ValueError("Выбранная аудиодорожка отсутствует в файле.")
        start = max(0.0, float(request.args.get("start", "0") or 0.0))
        duration = min(60.0, max(3.0, float(request.args.get("duration", "25") or 25.0)))
        source_duration = probe_duration(path)
        if source_duration is not None:
            if start >= source_duration:
                raise ValueError("Выбранная позиция находится за концом файла.")
            duration = min(duration, max(0.05, source_duration - start))
        clip = _cached_audio_clip(
            path,
            start,
            duration,
            stream_index=stream_index,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return str(error), 400
    response = send_file(
        clip,
        mimetype="audio/mpeg",
        conditional=True,
        max_age=0,
        download_name=f"{path.stem}-track-{stream_index}-preview.mp3",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/audio-preview")
def audio_preview():
    """Stream a browser-friendly stereo copy without altering the lossless file."""
    try:
        path = _safe_public_file(request.args.get("path", ""))
        ffmpeg = check_ffmpeg()
    except (RuntimeError, ValueError) as error:
        return str(error), 404

    def generate():
        process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "2",
                "-ar",
                "48000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "160k",
                "-f",
                "mp3",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            assert process.stdout is not None
            while chunk := process.stdout.read(256 * 1024):
                yield chunk
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
            if process.stdout is not None:
                process.stdout.close()

    return Response(
        generate(),
        mimetype="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            # Do not embed a possibly Cyrillic filename in a raw WSGI header;
            # Werkzeug's send_file handles RFC 5987 names where a download is
            # needed, while this streaming endpoint only needs inline playback.
            "Content-Disposition": "inline",
            "X-Accel-Buffering": "no",
        },
    )


def _cached_audio_clip(
    source: Path,
    start_sec: float,
    duration_sec: float,
    speed_ratio: float = 1.0,
    stream_index: int | None = None,
) -> Path:
    start_sec = max(0.0, float(start_sec))
    duration_sec = min(120.0, max(1.0, float(duration_sec)))
    speed_ratio = float(speed_ratio)
    if not 0.5 <= speed_ratio <= 2.0:
        speed_ratio = 1.0
    stat = source.stat()
    identity = json.dumps(
        {
            "path": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "start": round(start_sec, 3),
            "duration": round(duration_sec, 3),
            "speed": round(speed_ratio, 8),
            "stream_index": stream_index,
        },
        sort_keys=True,
    )
    cache_root = cfg_store.runtime_root / "audio_clips"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.mp3"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    with _clip_lock:
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp.mp3")
        source_duration = duration_sec * speed_ratio + 0.25
        audio_filter = (
            f"atempo={speed_ratio:.8f},"
            f"atrim=duration={duration_sec:.3f},asetpts=N/SR/TB"
        )
        command = [
            check_ffmpeg(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source),
            "-t",
            f"{source_duration:.3f}",
            "-map",
            f"0:{stream_index}" if stream_index is not None else "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            "48000",
            "-af",
            audio_filter,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "160k",
            "-f",
            "mp3",
            str(temporary),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(
                    "Не удалось подготовить фрагмент для браузера. "
                    + result.stderr[-1000:]
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


@app.get("/audio-clip")
def audio_clip():
    try:
        source = _safe_public_file(request.args.get("path", ""))
        start_sec = float(request.args.get("start", "0"))
        duration_sec = float(request.args.get("duration", "120"))
        clip = _cached_audio_clip(source, start_sec, duration_sec)
    except (RuntimeError, ValueError) as error:
        return str(error), 400
    return send_file(
        clip,
        mimetype="audio/mpeg",
        conditional=True,
        max_age=3600,
        download_name=f"{source.stem}-preview.mp3",
    )


def _alignment_preview_spec(
    project_id: str,
    pair_id: str,
    start_sec: float,
    requested_duration: float,
    manual_correction: float | None,
) -> tuple[Path, dict[str, Any], float, float]:
    root = cfg_store.pair_dir(project_id, pair_id)
    map_path = root / "alignment" / "alignment_map.json"
    if not map_path.is_file():
        raise FileNotFoundError("Карта сопоставления не найдена.")
    alignment_map = json.loads(map_path.read_text(encoding="utf-8"))
    segments = alignment_map.get("segments") or alignment_map.get("chunks") or []
    segment = next(
        (
            item
            for item in segments
            if float(item["dubbed_start"]) <= start_sec < float(item["dubbed_end"])
        ),
        None,
    )
    if segment is None:
        raise ValueError("Для выбранного времени нет участка карты.")
    duration_sec = min(requested_duration, float(segment["dubbed_end"]) - start_sec)
    if duration_sec < 1.0:
        raise ValueError("До конца участка осталось меньше секунды.")
    if manual_correction is None:
        manual_correction = float(alignment_map.get("manual_correction_sec") or 0.0)
    if not -600.0 <= manual_correction <= 600.0:
        raise ValueError("Ручная поправка должна быть от -600 до +600 секунд.")
    return root, segment, duration_sec, manual_correction


def _waveform_peaks(
    source: Path,
    start_sec: float,
    duration_sec: float,
    speed_ratio: float,
    point_count: int,
) -> dict[str, Any]:
    stat = source.stat()
    identity = json.dumps(
        {
            "path": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "start": round(start_sec, 3),
            "duration": round(duration_sec, 3),
            "speed": round(speed_ratio, 8),
            "points": point_count,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    with _waveform_lock:
        cached = _waveform_cache.get(key)
    if cached is not None:
        return cached
    source_duration = duration_sec * speed_ratio + 0.25
    command = [
        check_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start_sec):.3f}", "-i", str(source),
        "-t", f"{source_duration:.3f}", "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", "8000",
        "-af", f"atempo={speed_ratio:.8f},atrim=duration={duration_sec:.3f},asetpts=N/SR/TB",
        "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode or not result.stdout:
        message = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError("Не удалось построить осциллограмму. " + message)
    samples = array("f")
    samples.frombytes(result.stdout)
    absolute = sorted(abs(value) for value in samples)
    scale = absolute[min(len(absolute) - 1, int(len(absolute) * 0.995))] or 1.0
    mins: list[float] = []
    maxs: list[float] = []
    sample_count = len(samples)
    for index in range(point_count):
        left = index * sample_count // point_count
        right = max(left + 1, (index + 1) * sample_count // point_count)
        window = samples[left:right]
        mins.append(round(max(-1.0, min(window) / scale), 4))
        maxs.append(round(min(1.0, max(window) / scale), 4))
    payload = {"min": mins, "max": maxs, "normalization": round(scale, 6)}
    with _waveform_lock:
        if len(_waveform_cache) >= 64:
            _waveform_cache.pop(next(iter(_waveform_cache)))
        _waveform_cache[key] = payload
    return payload


@app.get("/api/projects/<project_id>/pairs/<pair_id>/alignment-audio/<role>")
def alignment_audio(project_id: str, pair_id: str, role: str):
    if role not in {"original", "dubbed"}:
        return "Неизвестная дорожка.", 404
    pair = cfg_store.load_pair(project_id, pair_id)
    if (pair.get("stages") or {}).get("2", {}).get("status") != "completed":
        return "Извлечённые дорожки ещё не готовы.", 404
    try:
        start_sec = max(0.0, float(request.args.get("start", "0")))
        requested_duration = min(60.0, max(2.0, float(request.args.get("duration", "20"))))
        correction_arg = request.args.get("correction")
        correction = float(correction_arg) if correction_arg is not None else None
        root, segment, duration_sec, manual_correction = _alignment_preview_spec(
            project_id, pair_id, start_sec, requested_duration, correction
        )
        if role == "dubbed":
            source_start = start_sec
            speed_ratio = 1.0
        else:
            speed_ratio = float(segment.get("speed_ratio") or 1.0)
            source_start = float(segment["original_start"]) + (
                start_sec - float(segment["dubbed_start"])
            ) * speed_ratio + manual_correction
            if source_start < 0:
                raise ValueError("Временная карта указывает за начало оригинала.")
        source = root / "extracted" / f"{role}.flac"
        if not source.is_file():
            raise ValueError("Извлечённая дорожка не найдена.")
        clip = _cached_audio_clip(source, source_start, duration_sec, speed_ratio)
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return str(error), 400
    return send_file(
        clip,
        mimetype="audio/mpeg",
        conditional=True,
        max_age=3600,
        download_name=f"{role}-aligned-preview.mp3",
    )


@app.get("/api/projects/<project_id>/pairs/<pair_id>/alignment-track/<role>")
def alignment_track(project_id: str, pair_id: str, role: str):
    """Stream a full extracted track (FLAC) with HTTP range support.

    The tracks screen plays the whole extracted ``original.flac`` /
    ``dubbed.flac`` directly and does synchronisation, seeking and the manual
    offset entirely in the browser. ``conditional=True`` enables ``Range``
    responses, so the browser can seek instantly without any transcoding — the
    fragile per-window mp3 clips are no longer needed for playback.
    """
    if role not in {"original", "dubbed"}:
        return "Неизвестная дорожка.", 404
    try:
        pair = cfg_store.load_pair(project_id, pair_id)
        if (pair.get("stages") or {}).get("2", {}).get("status") != "completed":
            return "Извлечённая дорожка ещё не готова.", 404
        root = cfg_store.pair_dir(project_id, pair_id)
    except (RuntimeError, ValueError) as error:
        return str(error), 404
    source = root / "extracted" / f"{role}.flac"
    if not source.is_file():
        return "Извлечённая дорожка ещё не готова.", 404
    return send_file(
        source,
        mimetype="audio/flac",
        conditional=True,
        max_age=3600,
        download_name=f"{role}.flac",
    )


# Full-track waveform overview: min/max peaks for the whole extracted track,
# in the track's own timeline, quantised to int8. The tracks screen fetches this
# once per role and does all panning / zooming / warping / offset on the client,
# so those interactions never touch the server again.
_PEAKS_TARGET_PPS = 120           # peaks per second at native resolution
_PEAKS_MAX = 720_000              # cap total peaks so multi-hour films stay small


def _full_track_peaks(source: Path) -> tuple[bytes, int, float, float]:
    """Return (payload, count, effective_pps, duration_sec) for the whole file.

    payload = int8 min[count] followed by int8 max[count]. Cached on disk keyed
    by the source file identity, so re-opening a project is instant.
    """
    import numpy as np

    stat = source.stat()
    cache_root = cfg_store.runtime_root / "waveform_peaks"
    cache_root.mkdir(parents=True, exist_ok=True)
    identity = (
        f"{source}|{stat.st_size}|{stat.st_mtime_ns}"
        f"|pps{_PEAKS_TARGET_PPS}|max{_PEAKS_MAX}|sr4000|v1"
    )
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    bin_path = cache_root / f"{key}.bin"
    meta_path = cache_root / f"{key}.json"
    if bin_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return bin_path.read_bytes(), int(meta["count"]), float(meta["pps"]), float(meta["duration"])

    with _waveform_lock:
        if bin_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return bin_path.read_bytes(), int(meta["count"]), float(meta["pps"]), float(meta["duration"])
        sample_rate = 4000
        command = [
            check_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode or not result.stdout:
            message = result.stderr.decode("utf-8", errors="replace")[-800:]
            raise RuntimeError("Не удалось построить осциллограмму. " + message)
        samples = np.frombuffer(result.stdout, dtype=np.float32)
        total_samples = int(samples.size)
        if total_samples == 0:
            raise RuntimeError("Дорожка пуста.")
        duration = total_samples / float(sample_rate)
        count = max(1, min(int(round(duration * _PEAKS_TARGET_PPS)), _PEAKS_MAX, total_samples))
        bounds = (np.arange(count, dtype=np.int64) * total_samples) // count
        mins = np.minimum.reduceat(samples, bounds)
        maxs = np.maximum.reduceat(samples, bounds)
        scale = float(np.quantile(np.abs(samples), 0.999)) or 1.0
        min_i8 = np.clip(np.round(mins / scale * 127.0), -127, 127).astype(np.int8)
        max_i8 = np.clip(np.round(maxs / scale * 127.0), -127, 127).astype(np.int8)
        payload = min_i8.tobytes() + max_i8.tobytes()
        pps = count / duration
        tmp_bin = bin_path.with_name(f".{bin_path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_bin.write_bytes(payload)
            os.replace(tmp_bin, bin_path)
        finally:
            tmp_bin.unlink(missing_ok=True)
        atomic_json(
            meta_path,
            {"count": count, "pps": pps, "duration": duration, "sample_rate": sample_rate},
        )
        return payload, count, pps, duration


@app.get("/api/projects/<project_id>/pairs/<pair_id>/alignment-peaks/<role>")
def alignment_peaks(project_id: str, pair_id: str, role: str):
    if role not in {"original", "dubbed"}:
        return "Неизвестная дорожка.", 404
    try:
        pair = cfg_store.load_pair(project_id, pair_id)
        if (pair.get("stages") or {}).get("2", {}).get("status") != "completed":
            return "Извлечённые дорожки ещё не готовы.", 404
        root = cfg_store.pair_dir(project_id, pair_id)
    except (RuntimeError, ValueError) as error:
        return str(error), 404
    source = root / "extracted" / f"{role}.flac"
    if not source.is_file():
        return "Извлечённая дорожка ещё не готова.", 404
    try:
        payload, count, pps, duration = _full_track_peaks(source)
    except (OSError, RuntimeError, ValueError) as error:
        return str(error), 400
    response = Response(payload, mimetype="application/octet-stream")
    response.headers["X-Peaks-Count"] = str(count)
    response.headers["X-Peaks-Pps"] = f"{pps:.6f}"
    response.headers["X-Peaks-Duration"] = f"{duration:.6f}"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/projects/<project_id>/pairs/<pair_id>/alignment-waveform")
def alignment_waveform(project_id: str, pair_id: str):
    try:
        pair = cfg_store.load_pair(project_id, pair_id)
        if (pair.get("stages") or {}).get("3", {}).get("status") != "completed":
            return jsonify({"error": "Карта сопоставления ещё не готова."}), 404
        start_sec = max(0.0, float(request.args.get("start", "0")))
        requested_duration = min(60.0, max(2.0, float(request.args.get("duration", "20"))))
        point_count = min(1800, max(300, int(request.args.get("points", "1200"))))
        correction_arg = request.args.get("correction")
        correction = float(correction_arg) if correction_arg is not None else None
        root, segment, duration_sec, manual_correction = _alignment_preview_spec(
            project_id, pair_id, start_sec, requested_duration, correction
        )
        speed_ratio = float(segment.get("speed_ratio") or 1.0)
        original_start = float(segment["original_start"]) + (
            start_sec - float(segment["dubbed_start"])
        ) * speed_ratio + manual_correction
        if original_start < 0:
            raise ValueError("Временная карта указывает за начало оригинала.")
        original = root / "extracted" / "original.flac"
        dubbed = root / "extracted" / "dubbed.flac"
        if not original.is_file() or not dubbed.is_file():
            raise ValueError("Извлечённые дорожки не найдены.")
        payload = {
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "correction_sec": manual_correction,
            "speed_ratio": speed_ratio,
            "points": point_count,
            "original": _waveform_peaks(original, original_start, duration_sec, speed_ratio, point_count),
            "dubbed": _waveform_peaks(dubbed, start_sec, duration_sec, 1.0, point_count),
        }
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(payload)


@app.get("/download")
def download():
    path = _safe_public_file(request.args.get("path", ""))
    return send_file(path, as_attachment=True, download_name=path.name, conditional=True)


@app.get("/api/system/health")
def system_health():
    """Cheap liveness probe behind the sidebar's service indicator."""
    return jsonify({"ok": True, "pid": os.getpid()})


def _shutdown_service() -> None:
    """Do what stop.bat does, then end this process.

    Runs off the request thread so the browser receives the reply before the
    socket dies; otherwise the tab reports a network error and the operator
    cannot tell a clean shutdown from a crash.
    """
    time.sleep(0.5)
    try:
        stop_running_tasks(cfg_store, "Сервис остановлен из панели.")
    except (OSError, RuntimeError, ValueError):
        # A task that refuses to stop must not keep the panel alive: the
        # operator asked for the service to close, and stop.bat would have
        # killed it anyway.
        pass
    finally:
        _scheduler_stop.set()
        PID_FILE.unlink(missing_ok=True)
        os._exit(0)


@app.post("/api/system/shutdown")
def system_shutdown():
    """Stop the service from the panel itself.

    The confirming header is required so that a stray link or an image tag
    pointing at this path cannot end the service; the browser only sends it
    from the panel's own script.
    """
    if request.headers.get("X-DubClean-Local") != "shutdown":
        return jsonify({"error": "Команда остановки не подтверждена."}), 403
    running = [
        task
        for task in cfg_store.list_tasks(
            states={"queued", "starting", "running"}
        )
    ]
    threading.Thread(
        target=_shutdown_service, name="dubclean-shutdown", daemon=False
    ).start()
    return jsonify({"ok": True, "stopped_tasks": len(running)})


@app.post("/api/reveal")
def reveal():
    value = str((request.get_json(silent=True) or {}).get("path", ""))
    path = Path(value).resolve()
    roots = [cfg_store.projects_root, *cfg_store.allowed_media_roots]
    if path.is_dir() and is_within(path, roots):
        subprocess.Popen(["explorer.exe", str(path)])
        return jsonify({"ok": True})
    if not path.is_file() or not is_within(path, roots):
        raise ValueError("Файл не найден или путь запрещён.")
    subprocess.Popen(["explorer.exe", "/select,", str(path)])
    return jsonify({"ok": True})


def main() -> None:
    configured_host = str(cfg["web"]["host"] or "").strip()
    if not _is_loopback_hostname(configured_host):
        raise RuntimeError(
            "DubClean разрешено запускать только на локальном адресе "
            "127.0.0.1, ::1 или localhost."
        )
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _restore_saved_source_roots()
    if PID_FILE.is_file():
        try:
            old_pid = int(PID_FILE.read_text(encoding="ascii").strip())
        except ValueError:
            old_pid = 0
        # pid_matches(None-safe): a recycled PID from a stale file (typical
        # after reboot) must not block startup; only a live process actually
        # running this module counts as "already running".
        if old_pid and old_pid != os.getpid() and pid_matches(
            old_pid, "paired_reference_cancel", "app.py"
        ) is not False:
            raise RuntimeError(f"Панель уже запущена, PID {old_pid}.")
    # A lost PID file must not allow a second instance: on Windows werkzeug
    # binds with SO_REUSEADDR, so two panels would silently share port 8768
    # and split requests between different code versions.
    others = find_module_instances("paired_reference_cancel", "app.py")
    if others:
        raise RuntimeError(
            f"Панель уже запущена (PID {', '.join(map(str, others))}), "
            "хотя PID-файл отсутствовал. Остановите её через stop.bat."
        )
    PID_FILE.write_text(str(os.getpid()), encoding="ascii")
    _recover_tasks()
    _start_scheduler()
    url = f"http://{configured_host}:{cfg['web']['port']}/product"
    if cfg["web"].get("open_browser", True) and os.environ.get("PAIRED_REFERENCE_CANCEL_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(
            host=configured_host,
            port=int(cfg["web"]["port"]),
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        _scheduler_stop.set()
        PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
