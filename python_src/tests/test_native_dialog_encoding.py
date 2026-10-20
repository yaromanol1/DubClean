from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from experiments.paired_reference_cancel import app as app_module


class NativeDialogEncodingTests(unittest.TestCase):
    def test_cyrillic_picker_path_is_decoded_as_utf8(self) -> None:
        selected = r"E:\Фильмы\Большой куш\Большой куш.mkv"
        completed = subprocess.CompletedProcess(
            args=["powershell.exe"],
            returncode=0,
            stdout=selected,
            stderr="",
        )

        with mock.patch.object(
            app_module.subprocess,
            "run",
            return_value=completed,
        ) as run:
            actual = app_module._run_windows_dialog(
                "[Console]::Write('test')",
                "dialog failed",
            )

        self.assertEqual(actual, selected)
        kwargs = run.call_args.kwargs
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertEqual(kwargs["errors"], "replace")

    def test_dialog_error_is_also_decoded_without_crashing(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["powershell.exe"],
            returncode=1,
            stdout="",
            stderr="Ошибка выбора файла",
        )

        with mock.patch.object(
            app_module.subprocess,
            "run",
            return_value=completed,
        ):
            with self.assertRaisesRegex(RuntimeError, "Ошибка выбора файла"):
                app_module._run_windows_dialog(
                    "[Console]::Write('test')",
                    "dialog failed",
                )


if __name__ == "__main__":
    unittest.main()
