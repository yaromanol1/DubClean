from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("model_downloads", ROOT / "portable_tools/download_models.py")
downloads = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloads)


class Response(io.BytesIO):
    def geturl(self):
        return "https://example.org/model.pt"


class ModelDownloadTests(unittest.TestCase):
    def test_corrupt_download_never_replaces_an_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.pt"
            target.write_bytes(b"existing model")
            with patch.object(downloads.urllib.request, "urlopen", return_value=Response(b"corrupt")):
                with self.assertRaises(ValueError):
                    downloads.verified_download("https://example.org/model.pt", target, "0" * 64)
            self.assertEqual(target.read_bytes(), b"existing model")
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_successful_download_is_verified_and_atomically_installed(self):
        data = b"verified model"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "model.pt"
            with patch.object(downloads.urllib.request, "urlopen", return_value=Response(data)):
                downloads.verified_download("https://example.org/model.pt", target, hashlib.sha256(data).hexdigest(), len(data))
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(list(Path(folder).glob("*.partial")), [])

    def test_manifest_cannot_write_outside_the_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                downloads.model_path(Path(folder), {"file": "../escaped.pt"})

    def test_offline_install_does_not_contact_network_and_repairs_pointer(self):
        data = b"verified model"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "weights/model.pt"
            model.parent.mkdir()
            model.write_bytes(data)
            manifest = {"models": {"speech": {"required_for_default": True, "dir": "weights", "weight_file": "model.pt", "size_bytes": len(data), "weight_sha256": hashlib.sha256(data).hexdigest()}}}
            (root / "portable_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(downloads.urllib.request, "urlopen", side_effect=AssertionError("network contacted")):
                downloads.install(root, verify_only=True)
            self.assertEqual((model.parent / "last_best_checkpoint").read_text(), "model.pt\n")


if __name__ == "__main__":
    unittest.main()
