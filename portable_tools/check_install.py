from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
CONFIG = (
    ROOT
    / "python_src"
    / "experiments"
    / "paired_reference_cancel"
    / "config.yaml"
)
MANIFEST = ROOT / "portable_manifest.json"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[ОШИБКА] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[ПРЕДУПРЕЖДЕНИЕ] {message}")


def _same_path(left: Path | str, right: Path | str) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
        os.path.normpath(str(right))
    )


def _inside_package(value: str) -> Path:
    """Resolve a config path.

    The published config.yaml stores package-relative placeholders until
    configure_portable.py rewrites them, so a relative value must be read
    against the package root and never against the current directory.
    """
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _run_version(executable: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    first_line = (result.stdout or result.stderr or "").splitlines()
    return result.returncode == 0, first_line[0] if first_line else "без версии"


def check_python(report: Report) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 11):
        report.fail(
            f"нужен Python 3.11, сейчас {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        report.ok(f"Python {sys.version.split()[0]}")
    if not _same_path(Path(sys.prefix).resolve(), VENV.resolve()):
        report.fail(f"запущено не локальное окружение {VENV}: {sys.prefix}")
    else:
        report.ok("используется локальная .venv")

    modules = {
        "torch": "PyTorch",
        "torchaudio": "TorchAudio",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "soundfile": "SoundFile",
        "yaml": "PyYAML",
        "flask": "Flask",
        "silero_vad": "Silero VAD 6.2.0",
        "clearvoice": "ClearVoice",
    }
    imported: dict[str, Any] = {}
    for module_name, label in modules.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
        except Exception as error:  # import failures often include native DLL errors
            report.fail(f"не загружается {label}: {error}")
        else:
            report.ok(f"модуль {label}")
    torch = imported.get("torch")
    if torch is not None:
        report.ok(
            "CUDA доступна" if bool(torch.cuda.is_available()) else "режим CPU доступен"
        )
    silero = imported.get("silero_vad")
    torch_module = imported.get("torch")
    if silero is not None:
        model_path = None
        try:
            import importlib.resources as resources
            import silero_vad.data

            model_path = resources.files(silero_vad.data).joinpath("silero_vad.jit")
            if not model_path.is_file() or model_path.stat().st_size <= 0:
                raise FileNotFoundError(str(model_path))
        except Exception as error:
            report.fail(f"не найдена локальная JIT-модель Silero VAD: {error}")
            model_path = None
        else:
            report.ok("локальная JIT-модель Silero VAD найдена (без загрузки из сети)")
        # A present file is not enough: verify the model actually loads and runs.
        # We load from bytes exactly like the runtime does, so a Cyrillic portable
        # path never reaches torch.jit's ASCII-only fopen (errno 2 otherwise).
        if model_path is not None and torch_module is not None:
            try:
                from io import BytesIO

                model = torch_module.jit.load(
                    BytesIO(model_path.read_bytes()), map_location="cpu"
                )
                if hasattr(model, "reset_states"):
                    model.reset_states()
                block = torch_module.zeros(512, dtype=torch_module.float32)
                with torch_module.no_grad():
                    probability = float(model(block, 16000).item())
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(f"неожиданная вероятность {probability}")
            except Exception as error:
                report.fail(
                    "Silero VAD установлен, но локальная JIT-модель не запускается "
                    f"(несовместимость torch/JIT или недоступный путь): {error}"
                )
            else:
                report.ok(
                    "JIT-модель Silero VAD загружается из байтов и выполняет инференс "
                    "(путь с кириллицей безопасен)"
                )
    return imported


def check_config(report: Report, yaml_module: Any, *, full: bool) -> dict[str, Any]:
    if not CONFIG.is_file():
        report.fail(f"не найден {CONFIG}")
        return {}
    if yaml_module is None:
        return {}
    try:
        cfg = yaml_module.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except Exception as error:
        report.fail(f"не читается config.yaml: {error}")
        return {}

    expected = {
        ("paths", "uploads"): ROOT / "data" / "uploads",
        ("paths", "outputs"): ROOT / "data" / "outputs",
        ("paths", "legacy_outputs"): ROOT / "data" / "outputs",
        ("paths", "models"): ROOT / "models",
        ("paths", "runtime"): ROOT / "data" / "runtime",
        ("paths", "backups"): ROOT / "data" / "backups",
        ("speech_extraction", "python"): VENV / "Scripts" / "python.exe",
        (
            "speech_extraction",
            "runner",
        ): ROOT
        / "python_src"
        / "experiments"
        / "paired_reference_cancel"
        / "speech_stem_runner.py",
        (
            "speech_extraction",
            "checkpoint_dir",
        ): ROOT / "third_party_models" / "MossFormer2_SE_48K_original",
        (
            "speech_extraction",
            "dubbed_checkpoint_dir",
        ): ROOT / "third_party_models" / "MossFormer2_SE_48K",
        ("music_separation", "python"): VENV / "Scripts" / "python.exe",
        (
            "music_separation",
            "runner",
        ): ROOT
        / "python_src"
        / "experiments"
        / "paired_reference_cancel"
        / "music_separator_runner.py",
        (
            "music_separation",
            "model_dir",
        ): ROOT / "third_party_models" / "audio_separator",
        ("training", "python"): VENV / "Scripts" / "python.exe",
        (
            "training",
            "runner",
        ): ROOT
        / "python_src"
        / "experiments"
        / "paired_reference_cancel"
        / "semantic_separator_runner.py",
    }
    for (section, key), target in expected.items():
        try:
            configured = _inside_package(str(cfg[section][key]))
        except (KeyError, OSError, TypeError, ValueError) as error:
            report.fail(f"нет корректного {section}.{key}: {error}")
            continue
        if not _same_path(configured, target.resolve()):
            report.fail(f"{section}.{key} указывает вне пакета: {configured}")

    try:
        projects_root = _inside_package(str(cfg["paths"]["projects"]))
    except (KeyError, OSError, TypeError, ValueError) as error:
        report.fail(f"нет корректной папки проектов: {error}")
    else:
        if not projects_root.is_dir():
            report.fail(f"папка проектов недоступна: {projects_root}")
        else:
            probe = projects_root / f".dubclean-write-test-{uuid.uuid4().hex}.tmp"
            try:
                probe.write_bytes(b"")
            except OSError as error:
                report.fail(f"нет записи в папку проектов {projects_root}: {error}")
            else:
                report.ok(f"папка проектов доступна для записи: {projects_root}")
            finally:
                probe.unlink(missing_ok=True)
    return cfg


def _model_path(spec: dict[str, Any]) -> Path | None:
    if spec.get("file"):
        return ROOT / str(spec["file"])
    if spec.get("dir") and spec.get("weight_file"):
        return ROOT / str(spec["dir"]) / str(spec["weight_file"])
    return None


def check_models(report: Report, *, full: bool) -> None:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        report.fail(f"не читается portable_manifest.json: {error}")
        return
    models = manifest.get("models")
    if not isinstance(models, dict):
        report.fail("portable_manifest.json не содержит models")
        return
    for name, raw_spec in models.items():
        if not isinstance(raw_spec, dict) or raw_spec.get("required_for_default") is False:
            continue
        path = _model_path(raw_spec)
        if path is None:
            report.fail(f"для модели {name} не указан файл")
            continue
        if not path.is_file():
            report.fail(f"не найдена модель {name}: {path}")
            continue
        expected_size = int(raw_spec.get("size_bytes") or 0)
        if expected_size and path.stat().st_size != expected_size:
            report.fail(
                f"размер модели {name} неверен: {path.stat().st_size}, "
                f"ожидалось {expected_size}"
            )
            continue
        expected_hash = str(
            raw_spec.get("sha256") or raw_spec.get("weight_sha256") or ""
        ).upper()
        if full and expected_hash:
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                report.fail(f"контрольная сумма модели {name} не совпадает")
                continue
        report.ok(f"модель {name}: {path.name}")

        if name == "song_classifier":
            try:
                from io import BytesIO

                import torch

                classifier = torch.jit.load(
                    BytesIO(path.read_bytes()), map_location="cpu"
                ).eval()
                with torch.no_grad():
                    scores = classifier(
                        torch.zeros((1, 320000), dtype=torch.float32)
                    )
                if tuple(scores.shape) != (1, 527):
                    raise ValueError(f"неожиданная форма выхода {tuple(scores.shape)}")
            except Exception as error:
                report.fail(f"классификатор песен не запускается: {error}")
            else:
                report.ok(
                    "EfficientAT запускается локально и возвращает классы AudioSet"
                )

        if name in ("speech_extractor", "speech_extractor_dubbed") and raw_spec.get("dir"):
            pointer = ROOT / str(raw_spec["dir"]) / "last_best_checkpoint"
            try:
                pointed_name = pointer.read_text(encoding="utf-8").strip()
            except OSError as error:
                report.fail(f"не читается указатель MossFormer2 ({name}): {error}")
            else:
                if pointed_name != str(raw_spec.get("weight_file") or ""):
                    report.fail(
                        f"last_best_checkpoint ({name}) указывает не на зафиксированные веса"
                    )


def check_tools(report: Report) -> None:
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if not executable:
            report.fail(
                f"{name} не найден; установите FFmpeg или положите его в "
                "tools\\ffmpeg\\bin"
            )
            continue
        ok, version = _run_version(executable)
        if not ok:
            report.fail(f"{name} найден, но не запускается: {version}")
        else:
            report.ok(f"{version} ({executable})")


def check_pip(report: Report) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        report.fail(f"конфликт Python-зависимостей: {(result.stdout or result.stderr).strip()}")
    else:
        report.ok("конфликтов Python-зависимостей нет")


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка DubClean Portable")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="не считать SHA-256 и не выполнять расширенные проверки",
    )
    args = parser.parse_args()

    report = Report()
    imported = check_python(report)
    check_config(report, imported.get("yaml"), full=not args.quick)
    check_models(report, full=not args.quick)
    check_tools(report)
    if not args.quick:
        check_pip(report)

    if report.failures:
        print(f"\nПроверка не пройдена: ошибок {len(report.failures)}.")
        return 1
    print(
        f"\nDubClean готов к запуску. Предупреждений: {len(report.warnings)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
