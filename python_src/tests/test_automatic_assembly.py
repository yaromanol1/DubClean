from __future__ import annotations

import unittest
from pathlib import Path

from experiments.paired_reference_cancel.automatic_assembly import (
    normalize_policy,
    resolve_policies,
)


class AutomaticAssemblyPolicyTests(unittest.TestCase):
    def test_auto_follows_both_recommendations(self) -> None:
        result = resolve_policies(
            {
                "reference_alignment_policy": "auto",
                "speech_second_pass_policy": "auto",
            },
            {
                "recommended_alignment": "algorithmic",
                "recommended_speech_extraction_second_pass": True,
            },
        )
        self.assertEqual(result["reference_adapter"], "algorithmic")
        self.assertTrue(result["speech_extraction_second_pass"])

    def test_explicit_on_overrides_negative_recommendation(self) -> None:
        result = resolve_policies(
            {
                "reference_alignment_policy": "on",
                "speech_second_pass_policy": "on",
            },
            {
                "recommended_alignment": "none",
                "recommended_speech_extraction_second_pass": False,
            },
        )
        self.assertEqual(result["reference_adapter"], "algorithmic")
        self.assertTrue(result["speech_extraction_second_pass"])

    def test_explicit_off_overrides_positive_recommendation(self) -> None:
        result = resolve_policies(
            {
                "reference_alignment_policy": "off",
                "speech_second_pass_policy": "off",
            },
            {
                "recommended_alignment": "local_auto",
                "recommended_speech_extraction_second_pass": True,
            },
        )
        self.assertEqual(result["reference_adapter"], "raw")
        self.assertFalse(result["speech_extraction_second_pass"])

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Неизвестный режим"):
            normalize_policy("sometimes", setting_name="проверка")

    def test_product_ui_persists_and_sends_both_policies(self) -> None:
        product_root = Path(__file__).parents[2]
        source = (
            product_root
            / "docs"
            / "концепт интерфейса"
            / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")
        self.assertIn("autoReferenceAlignment: 'auto'", source)
        self.assertIn("autoSpeechSecondPass: 'auto'", source)
        self.assertIn(
            "reference_alignment_policy:this.state.autoReferenceAlignment",
            source,
        )
        self.assertIn(
            "speech_second_pass_policy:this.state.autoSpeechSecondPass",
            source,
        )
        self.assertIn("Полная сборка: выравнивание качества", source)
        self.assertIn("Полная сборка: повторный проход MossFormer", source)


if __name__ == "__main__":
    unittest.main()
