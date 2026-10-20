"""Осциллограммы дорожек должны появляться сами после извлечения.

Наблюдавшееся поведение: запрос осциллограмм уходил, пока извлечение ещё шло,
висел без таймаута, а флаг «уже гружу» отсекал все последующие попытки. Экран
оставался в «строю осциллограммы» до тех пор, пока тот запрос сам не ответит,
и выглядело это так, будто помогает уход на другой экран и возврат.

Тесты закрепляют три обещания: во время извлечения запрос не уходит, по факту
завершения он делается принудительно, и зависший запрос не держит экран вечно.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
HTML = (
    ROOT / "docs" / "концепт интерфейса" / "DubClean-RU.dc.html"
).read_text(encoding="utf-8")


def _method(signature: str) -> str:
    """Тело метода по его определению, а не по первому попавшемуся вызову."""
    anchor = f"\n  {signature} {{"
    if anchor not in HTML:
        raise AssertionError(f"не найдено определение: {signature}")
    return HTML.split(anchor, 1)[1].split("\n  }", 1)[0]


class PeaksAreNotRequestedWhileExtractingTests(unittest.TestCase):
    def test_busy_check_covers_every_operation_that_makes_the_tracks(self) -> None:
        body = _method("matchTracksBusy(application)")

        for operation in ("application_tracks", "application_auto", "extract", "align"):
            with self.subTest(operation=operation):
                self.assertIn(f"'{operation}'", body)
        self.assertIn("'queued','starting','running'", body)

    def test_loading_is_skipped_while_the_tracks_are_still_being_made(self) -> None:
        body = _method("async loadMatchPeaks(force=false)")

        self.assertIn("if (this.matchTracksBusy()) return;", body)
        # Проверка обязана стоять до того, как выставляется флаг загрузки,
        # иначе экран снова залипнет в «строю осциллограммы».
        self.assertLess(
            body.index("this.matchTracksBusy()"),
            body.index("matchPeaksLoading:true"),
        )


class PeaksReloadOnCompletionTests(unittest.TestCase):
    def test_poll_detects_the_transition_and_forces_a_reload(self) -> None:
        body = _method("async refreshActiveApplication()")

        self.assertIn("const tracksWereBusy = !!this.__matchTracksBusy;", body)
        self.assertIn("const justFinished = tracksWereBusy && !tracksBusyNow;", body)
        self.assertIn("this.loadMatchPeaks(justFinished)", body)

    def test_transition_is_measured_on_the_fresh_answer_not_stale_state(self) -> None:
        # Сразу после setState в this.state ещё лежат прежние задачи: считать
        # переход по нему — значит пропустить его ровно один раз, а именно тот.
        body = _method("async refreshActiveApplication()")

        self.assertIn("this.matchTracksBusy(activeForState)", body)

    def test_busy_check_accepts_an_explicit_application(self) -> None:
        body = _method("matchTracksBusy(application)")

        self.assertIn("const source = application || this.state.activeApplication;", body)


class StuckRequestRecoveryTests(unittest.TestCase):
    def test_an_in_flight_request_cannot_block_retries_forever(self) -> None:
        body = _method("async loadMatchPeaks(force=false)")

        self.assertIn("this.__peaksLoadingAt", body)
        self.assertRegex(
            body,
            re.compile(
                r"if \(this\.__peaksLoadingKey === key\s*\n\s*"
                r"&& \(Date\.now\(\) - \(this\.__peaksLoadingAt \|\| 0\)\) < 60000\) return;"
            ),
        )


class RequestTimeoutTests(unittest.TestCase):
    """Здоровый ответ приходит за десятые доли секунды.

    Без обрыва по таймауту зависший запрос держал флаг загрузки, и снять его
    было некому: опрос каждые три секунды молча упирался в этот флаг.
    """

    def test_peaks_request_is_aborted_instead_of_hanging(self) -> None:
        body = _method("async fetchTrackPeaks(url)")

        self.assertIn("new AbortController()", body)
        self.assertIn("controller.abort()", body)
        self.assertIn("signal: controller.signal", body)
        self.assertIn("30000", body)

    def test_timeout_is_reported_rather_than_swallowed(self) -> None:
        body = _method("async fetchTrackPeaks(url)")

        self.assertIn("'Осциллограммы не пришли за 30 секунд.'", body)
        # Таймер снимается в любом случае, иначе успешный запрос оставит
        # висеть отложенный abort.
        self.assertIn("} finally {\n      clearTimeout(timer);", body)


class FailureIsReportedTests(unittest.TestCase):
    """Сбой со старой осциллограммой на экране раньше проглатывался.

    Плашка с ошибкой в этом случае не показывается — вместо неё остаётся
    прежняя картинка, — и повторяющийся отказ был бы совершенно незаметен.
    """

    def test_kept_waveform_still_reports_the_failure_once(self) -> None:
        body = _method("async loadMatchPeaks(force=false)")

        self.assertIn("const keptOldWaveform = !!(this.__peaks && this.__peaks.dubbed);", body)
        self.assertIn("if (keptOldWaveform && this.__lastPeaksError !== message)", body)
        self.assertIn("this.showNotice(message);", body)

    def test_a_successful_load_clears_the_reported_failure(self) -> None:
        body = _method("async loadMatchPeaks(force=false)")

        # Иначе одна и та же ошибка после починки и повторного сбоя смолчала бы.
        self.assertIn("this.__lastPeaksError = '';", body)


class StatusTextTests(unittest.TestCase):
    def test_the_screen_no_longer_asks_the_operator_to_refresh(self) -> None:
        self.assertNotIn("Осциллограммы ещё готовятся. Обновите проект.", HTML)
        self.assertIn(
            "Извлекаю дорожки. Осциллограммы появятся сами, как только они будут готовы.",
            HTML,
        )


if __name__ == "__main__":
    unittest.main()
