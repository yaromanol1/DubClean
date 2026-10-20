from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {
    ".bat", ".css", ".html", ".js", ".json", ".md", ".ps1", ".py",
    ".txt", ".yaml", ".yml",
}
MODEL_SUFFIXES = {".ckpt", ".jit", ".onnx", ".pt", ".pth"}
REQUIRED_ROOT_FILES = {
    "LICENSE",
    "NOTICE",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "portable_manifest.json",
    "requirements.txt",
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{40,})"),
    "API key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:\\(?:Users|zvuk|Новый набор|Zona Downloads)\\|"
    r"C:/Users/|/home/[^/]+/|/Users/[^/]+/)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def model_path(root: Path, spec: dict[str, Any]) -> Path | None:
    file_value = str(spec.get("file") or "").strip()
    if file_value:
        return root / file_value
    directory = str(spec.get("dir") or "").strip()
    weight = str(spec.get("weight_file") or "").strip()
    if directory and weight:
        return root / directory / weight
    return None


def audit(root: Path, *, require_models: bool) -> list[str]:
    errors: list[str] = []
    for name in sorted(REQUIRED_ROOT_FILES):
        if not (root / name).is_file():
            errors.append(f"нет обязательного файла: {name}")

    for forbidden in (".venv", "output", "dist"):
        if (root / forbidden).exists():
            errors.append(f"в комплект попал локальный каталог: {forbidden}")

    data_root = root / "data"
    if data_root.exists():
        data_files = [path for path in data_root.rglob("*") if path.is_file()]
        if data_files:
            errors.append("в комплект попали пользовательские файлы из data/")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix.casefold() in {".pyc", ".pyo"}:
            errors.append(f"скомпилированный Python-мусор: {relative}")
        if path.stat().st_size > 100 * 1024 * 1024:
            allowed_model = require_models and path.suffix.casefold() in MODEL_SUFFIXES
            if not allowed_model:
                errors.append(f"файл больше лимита GitHub 100 МБ: {relative}")
        if path.suffix.casefold() not in TEXT_SUFFIXES or path.name == "check_release.py":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if PRIVATE_PATH.search(content):
            errors.append(f"частный абсолютный путь: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"возможный секрет ({label}): {relative}")
        lowered = content.casefold()
        if "90_можно_точно_удалить" in lowered or "service_migration_package_ru" in lowered:
            errors.append(f"внутренняя рабочая заметка: {relative}")

    manifest_path = root / "portable_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            errors.append(f"не читается portable_manifest.json: {error}")
        else:
            for model_id, raw_spec in (manifest.get("models") or {}).items():
                if not isinstance(raw_spec, dict) or not raw_spec.get("required_for_default"):
                    continue
                expected = model_path(root, raw_spec)
                if expected is None:
                    errors.append(f"для модели {model_id} не указан файл")
                    continue
                if not expected.is_file():
                    if require_models or raw_spec.get("distribution") == "git":
                        errors.append(f"нет обязательной модели: {expected.relative_to(root)}")
                    continue
                expected_size = int(raw_spec.get("size_bytes") or 0)
                expected_hash = str(
                    raw_spec.get("sha256") or raw_spec.get("weight_sha256") or ""
                ).upper()
                if expected_size and expected.stat().st_size != expected_size:
                    errors.append(f"неверный размер модели: {expected.relative_to(root)}")
                if expected_hash and sha256(expected) != expected_hash:
                    errors.append(f"неверный SHA-256 модели: {expected.relative_to(root)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка чистоты релиза DubClean")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    errors = audit(root, require_models=bool(args.require_models))
    if errors:
        print("Релиз не готов:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Релиз проверен: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

