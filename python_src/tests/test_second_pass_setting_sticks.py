"""Настройка «повторный проход» не должна сбрасываться сама.

Состояние экрана дорожек пересобирается каждым опросом раз в три секунды.
Пока переключатель читался только из пары, он возвращался к значению, которое
записал анализ соответствия, — и «Включить» в настройках приходилось нажимать
заново после каждого извлечения.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]


class SecondPassPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html_source = (
            ROOT / "docs" / "концепт интерфейса" / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")
        cls.patch = cls.html_source.split("applicationStatePatch(", 1)[1].split(
            "\n  async openBackendProject", 1
        )[0]

    def test_explicit_policy_survives_every_state_refresh(self) -> None:
        # Раньше здесь стояло безусловное !!pair.speech_extraction_second_pass,
        # и опрос затирал прямой ответ значением из пары.
        self.assertIn(
            "speechSecondPass: secondPassPolicy === 'auto'\n"
            "        ? !!pair.speech_extraction_second_pass\n"
            "        : secondPassPolicy === 'on',",
            self.patch,
        )
        self.assertNotIn(
            "speechSecondPass: !!pair.speech_extraction_second_pass,", self.patch
        )

    def test_automatic_policy_still_follows_the_analysis(self) -> None:
        self.assertIn("secondPassPolicy === 'auto'", self.patch)
        self.assertIn("!!pair.speech_extraction_second_pass", self.patch)

    def test_unknown_policy_value_falls_back_to_automatic(self) -> None:
        self.assertIn(
            "const secondPassPolicy = ['auto','on','off'].includes("
            "this.state.autoSpeechSecondPass)\n"
            "      ? this.state.autoSpeechSecondPass\n"
            "      : 'auto';",
            self.html_source,
        )

    def test_the_tracks_toggle_is_the_same_choice_as_the_setting(self) -> None:
        method = self.html_source.split("async setSpeechSecondPass(enabled) {", 1)[
            1
        ].split("\n  }", 1)[0]

        self.assertIn(
            "this.setPreference({ autoSpeechSecondPass: speechSecondPass "
            "? 'on' : 'off' });",
            method,
        )
        # Значение по-прежнему сохраняется и в паре: полная сборка читает его
        # оттуда, а не из настроек браузера.
        self.assertIn("speech_extraction_second_pass:speechSecondPass", method)

    def test_applying_recommendations_switches_the_policy_to_automatic(self) -> None:
        method = self.html_source.split("async applyReferenceAnalysis(", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("this.setPreference({ autoSpeechSecondPass: 'auto' });", method)

    def test_returning_to_the_manual_choice_restores_a_direct_answer(self) -> None:
        method = self.html_source.split("async keepReferenceAnalysisManual(", 1)[
            1
        ].split("\n  }", 1)[0]

        self.assertIn(
            "autoSpeechSecondPass: response.speech_extraction_second_pass "
            "? 'on' : 'off',",
            method,
        )

    def test_policy_is_persisted_between_sessions(self) -> None:
        self.assertIn(
            "if (['auto','on','off'].includes(saved.autoSpeechSecondPass))",
            self.html_source,
        )
        self.assertIn(
            "autoSpeechSecondPass: state.autoSpeechSecondPass,", self.html_source
        )


if __name__ == "__main__":
    unittest.main()
