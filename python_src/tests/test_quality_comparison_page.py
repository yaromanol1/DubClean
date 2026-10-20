from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.paired_reference_cancel import app as app_module


class QualityComparisonPageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "quality_comparisons"
        self.root.mkdir(parents=True)
        self.root_patch = mock.patch.object(
            app_module, "_quality_comparisons_root", return_value=self.root
        )
        self.root_patch.start()
        self.client = app_module.app.test_client()

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def _write_manifest(self, comparison_id: str = "snatch-check") -> Path:
        folder = self.root / comparison_id
        (folder / "audio").mkdir(parents=True)
        (folder / "audio" / "before.wav").write_bytes(b"RIFF-before")
        (folder / "audio" / "after.wav").write_bytes(b"RIFF-after")
        manifest = {
            "schema_version": 1,
            "id": "ignored-id",
            "title": "Большой куш — проверка речи",
            "description": "Контроль известных провалов.",
            "generated_at": "2026-08-01T10:00:00Z",
            "private_source_path": "H:/private/source.flac",
            "summary": {"confirmed": 1, "non_finite": float("inf")},
            "intervals": [
                {
                    "id": "case-1",
                    "start_sec": 1475.25,
                    "end_sec": 1476.25,
                    "reason": "Второй проход приглушил речь.",
                    "decision": "Восстановлен первый проход",
                    "confidence": 0.97,
                    "metrics": {"loss_db": 28.4, "nested": {"secret": True}},
                    "before_audio": "audio/before.wav",
                    "after_audio": "audio/after.wav",
                    "private_path": "H:/private/result.flac",
                }
            ],
        }
        (folder / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        return folder

    def test_page_and_assets_are_available_without_manifest(self) -> None:
        page = self.client.get("/product/quality-comparison")
        stylesheet = self.client.get("/product/quality-comparison.css")
        script = self.client.get("/product/quality-comparison.js")
        catalog = self.client.get("/api/quality-comparisons")

        self.assertEqual(page.status_code, 200)
        self.assertIn("Сравнение результата", page.get_data(as_text=True))
        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.get_json(), {"comparisons": []})
        page.close()
        stylesheet.close()
        script.close()
        catalog.close()

    def test_manifest_is_sanitized_and_catalogued(self) -> None:
        self._write_manifest()

        catalog = self.client.get("/api/quality-comparisons").get_json()
        self.assertEqual(len(catalog["comparisons"]), 1)
        self.assertEqual(catalog["comparisons"][0]["id"], "snatch-check")
        self.assertEqual(catalog["comparisons"][0]["audio_ready_count"], 1)

        response = self.client.get("/api/quality-comparisons/snatch-check")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["id"], "snatch-check")
        self.assertNotIn("private_source_path", payload)
        self.assertNotIn("non_finite", payload["summary"])
        interval = payload["intervals"][0]
        self.assertNotIn("private_path", interval)
        self.assertNotIn("nested", interval["metrics"])
        self.assertEqual(
            interval["before_audio_url"],
            "/api/quality-comparisons/snatch-check/audio/audio/before.wav",
        )
        self.assertTrue(interval["audio_ready"])

    def test_audio_is_served_inline_with_range_support(self) -> None:
        self._write_manifest()

        response = self.client.get(
            "/api/quality-comparisons/snatch-check/audio/audio/before.wav",
            headers={"Range": "bytes=0-3"},
        )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.data, b"RIFF")
        self.assertEqual(response.headers["Accept-Ranges"], "bytes")
        response.close()

    def test_audio_cannot_escape_comparison_folder(self) -> None:
        self._write_manifest()
        (self.root / "outside.wav").write_bytes(b"private")

        response = self.client.get(
            "/api/quality-comparisons/snatch-check/audio/%2E%2E/outside.wav"
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn(b"private", response.data)

    def test_manifest_cannot_reference_absolute_or_parent_audio(self) -> None:
        folder = self.root / "unsafe-check"
        folder.mkdir()
        (self.root / "outside.wav").write_bytes(b"private")
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "title": "Unsafe",
                    "intervals": [
                        {
                            "start_sec": 1,
                            "end_sec": 2,
                            "before_audio": "../outside.wav",
                            "after_audio": str((self.root / "outside.wav").resolve()),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        response = self.client.get("/api/quality-comparisons/unsafe-check")

        self.assertEqual(response.status_code, 200)
        interval = response.get_json()["intervals"][0]
        self.assertIsNone(interval["before_audio_url"])
        self.assertIsNone(interval["after_audio_url"])
        self.assertFalse(interval["audio_ready"])

    def test_invalid_identifier_and_non_audio_file_are_rejected(self) -> None:
        folder = self._write_manifest()
        (folder / "audio" / "note.txt").write_text("secret", encoding="utf-8")

        invalid_id = self.client.get("/api/quality-comparisons/..")
        non_audio = self.client.get(
            "/api/quality-comparisons/snatch-check/audio/audio/note.txt"
        )

        self.assertEqual(invalid_id.status_code, 400)
        self.assertEqual(non_audio.status_code, 404)
        self.assertNotIn(b"secret", non_audio.data)

    def test_corrupt_manifest_returns_controlled_client_error(self) -> None:
        folder = self.root / "broken"
        folder.mkdir()
        (folder / "manifest.json").write_text("{not-json", encoding="utf-8")

        response = self.client.get("/api/quality-comparisons/broken")

        self.assertEqual(response.status_code, 400)
        self.assertIn("Некорректный манифест", response.get_json()["error"])

    def test_unsupported_manifest_schema_is_rejected_and_hidden_from_catalog(self) -> None:
        folder = self.root / "future-schema"
        folder.mkdir()
        (folder / "manifest.json").write_text(
            json.dumps({"schema_version": 99, "title": "Future", "intervals": []}),
            encoding="utf-8",
        )

        detail = self.client.get("/api/quality-comparisons/future-schema")
        catalog = self.client.get("/api/quality-comparisons")

        self.assertEqual(detail.status_code, 400)
        self.assertIn("Неподдерживаемая версия", detail.get_json()["error"])
        self.assertEqual(catalog.get_json(), {"comparisons": []})

    def test_manifest_symlink_cannot_escape_quality_folder(self) -> None:
        folder = self.root / "linked-manifest"
        folder.mkdir()
        external = Path(self.temporary.name) / "external.json"
        external.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "title": "Private manifest",
                    "intervals": [],
                }
            ),
            encoding="utf-8",
        )
        try:
            os.symlink(external, folder / "manifest.json")
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Симлинки недоступны в тестовом окружении: {error}")

        response = self.client.get("/api/quality-comparisons/linked-manifest")

        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Private manifest", response.get_data(as_text=True))

    def test_audio_symlink_cannot_escape_comparison_folder(self) -> None:
        folder = self._write_manifest("linked-audio")
        external = Path(self.temporary.name) / "external.wav"
        external.write_bytes(b"private audio")
        link = folder / "audio" / "linked.wav"
        try:
            os.symlink(external, link)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Симлинки недоступны в тестовом окружении: {error}")

        response = self.client.get(
            "/api/quality-comparisons/linked-audio/audio/audio/linked.wav"
        )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn(b"private audio", response.data)


if __name__ == "__main__":
    unittest.main()
