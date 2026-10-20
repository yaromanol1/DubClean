from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

import yaml

from experiments.paired_reference_cancel.application_pipeline import (
    subtractor_runner_name,
)


PORTABLE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    PORTABLE_ROOT
    / "python_src"
    / "experiments"
    / "paired_reference_cancel"
    / "config.yaml"
)
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
SOURCE_ONLY = os.environ.get("DUBCLEAN_SOURCE_ONLY") == "1"


class PortableProductionModelTests(unittest.TestCase):
    def test_manifest_contains_only_the_accepted_internal_models(self) -> None:
        manifest = json.loads(
            (PORTABLE_ROOT / "portable_manifest.json").read_text(encoding="utf-8")
        )
        models = manifest["models"]

        self.assertIn("dubclean_voice", models)
        self.assertIn("dubclean_me", models)
        self.assertEqual(
            manifest["default_pipeline"]["semantic_model"], "dubclean_voice"
        )
        self.assertEqual(
            manifest["default_pipeline"]["background_model"], "dubclean_me"
        )

    def test_public_model_files_exist_under_stable_names(self) -> None:
        manifest = json.loads(
            (PORTABLE_ROOT / "portable_manifest.json").read_text(encoding="utf-8")
        )
        for model_id in ("dubclean_voice", "dubclean_me"):
            spec = manifest["models"][model_id]
            checkpoint = PORTABLE_ROOT / spec["file"]
            if SOURCE_ONLY:
                self.assertGreater(int(spec.get("size_bytes") or 0), 1_000_000)
                self.assertRegex(str(spec.get("sha256") or ""), r"^[A-F0-9]{64}$")
                continue
            self.assertTrue(checkpoint.is_file(), model_id)
            self.assertGreater(checkpoint.stat().st_size, 1_000_000)
            self.assertNotIn("epoch", checkpoint.name.casefold())

    def test_removed_model_names_are_absent_from_product_contract(self) -> None:
        manifest_text = (
            PORTABLE_ROOT / "portable_manifest.json"
        ).read_text(encoding="utf-8").casefold()

        removed_names = (
            "hard" + "master",
            "semantic_" + "v" + str(3),
            "hard_" + "v" + str(3),
        )
        for removed_name in removed_names:
            self.assertNotIn(removed_name, manifest_text)


class ProductionCheckpointRoutingTests(unittest.TestCase):
    """The public model name must still reach the program that can load it."""

    def test_published_voice_model_uses_the_semantic_program(self) -> None:
        manifest = json.loads(
            (PORTABLE_ROOT / "portable_manifest.json").read_text(encoding="utf-8")
        )
        checkpoint = PORTABLE_ROOT / manifest["models"]["dubclean_voice"]["file"]
        self.assertEqual(
            subtractor_runner_name(checkpoint), "semantic_separator_infer.py"
        )

    def test_public_name_alone_is_enough_to_choose_the_program(self) -> None:
        self.assertEqual(
            subtractor_runner_name(Path("anywhere/dubclean_voice.pt")),
            "semantic_separator_infer.py",
        )

    def test_legacy_paired_checkpoint_keeps_the_older_program(self) -> None:
        legacy = Path("project/checkpoints/reference_subtractor_best.pt")
        self.assertEqual(
            subtractor_runner_name(legacy), "reference_subtractor_infer.py"
        )


class PortableConfigPathTests(unittest.TestCase):
    """config.yaml is published, so it must not carry a machine path."""

    # paths.projects is the one value a user may deliberately point at another
    # disk through the interface, so it is allowed to become absolute.
    EXTERNAL_BY_DESIGN = {("paths", "projects")}

    def _config(self) -> dict:
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

    def test_package_paths_are_stored_relative_to_the_package(self) -> None:
        cfg = self._config()
        offenders: list[str] = []
        for section, values in cfg.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                candidates = value if isinstance(value, list) else [value]
                for item in candidates:
                    if not isinstance(item, str) or not _ABSOLUTE.match(item):
                        continue
                    if (section, key) in self.EXTERNAL_BY_DESIGN:
                        continue
                    offenders.append(f"{section}.{key}={item}")
        self.assertEqual(offenders, [], "абсолютные пути в опубликованном config.yaml")

    def test_relative_package_paths_point_at_files_that_exist(self) -> None:
        cfg = self._config()
        missing: list[str] = []
        optional = {("music_separation", "model_dir")}  # необязательный fallback
        for section, values in cfg.items():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if not isinstance(value, str) or not value.startswith("./"):
                    continue
                if (section, key) in optional:
                    continue
                if SOURCE_ONLY and (
                    value.startswith("./.venv/")
                    or Path(value).suffix.casefold()
                    in {".ckpt", ".jit", ".onnx", ".pt", ".pth"}
                ):
                    continue
                if not (PORTABLE_ROOT / value).exists():
                    missing.append(f"{section}.{key}={value}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
