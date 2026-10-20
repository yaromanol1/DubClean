"""Stopping the service from the panel instead of leaving it running.

Closing the browser tab used to leave the panel and its workers alive in the
background.  The sidebar now carries the same stop the batch file performs, so
these tests pin both halves: the endpoint must be hard to trigger by accident,
and it must stop exactly what stop.bat stops.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from experiments.paired_reference_cancel import app as app_module
from experiments.paired_reference_cancel import service_control


ROOT = Path(__file__).parents[2]


class ShutdownEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_health_reports_the_live_process(self) -> None:
        response = self.client.get("/api/system/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIsInstance(payload["pid"], int)

    def test_shutdown_without_the_confirming_header_is_refused(self) -> None:
        with mock.patch.object(app_module.threading, "Thread") as thread:
            response = self.client.post("/api/system/shutdown")

        self.assertEqual(response.status_code, 403)
        thread.assert_not_called()

    def test_shutdown_with_a_wrong_header_value_is_refused(self) -> None:
        with mock.patch.object(app_module.threading, "Thread") as thread:
            response = self.client.post(
                "/api/system/shutdown",
                headers={"X-DubClean-Local": "please"},
            )

        self.assertEqual(response.status_code, 403)
        thread.assert_not_called()

    def test_cross_origin_shutdown_is_rejected_before_the_endpoint(self) -> None:
        with mock.patch.object(app_module.threading, "Thread") as thread:
            response = self.client.post(
                "/api/system/shutdown",
                headers={
                    "X-DubClean-Local": "shutdown",
                    "Origin": "https://example.com",
                },
            )

        self.assertEqual(response.status_code, 403)
        thread.assert_not_called()

    def test_confirmed_shutdown_answers_before_it_exits(self) -> None:
        # The worker must run off the request thread: a reply that never
        # reaches the browser is indistinguishable from a crash.
        with mock.patch.object(app_module.threading, "Thread") as thread:
            response = self.client.post(
                "/api/system/shutdown",
                headers={"X-DubClean-Local": "shutdown"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ok"])
        thread.assert_called_once()
        self.assertIs(
            thread.call_args.kwargs["target"], app_module._shutdown_service
        )
        self.assertFalse(thread.call_args.kwargs["daemon"])
        thread.return_value.start.assert_called_once()


class SharedStopImplementationTests(unittest.TestCase):
    def test_panel_and_batch_stop_tasks_through_one_function(self) -> None:
        source = (
            ROOT
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "service_control.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def stop_running_tasks(", source)
        self.assertIn(
            'stop_running_tasks(store, "Сервис остановлен через stop.bat.")',
            source,
        )

    def test_stop_running_tasks_marks_and_kills_every_live_task(self) -> None:
        task = {
            "id": "t1",
            "stop_file": "",
            "pid": 4242,
            "state": "running",
        }
        store = mock.Mock()
        store.list_tasks.return_value = [task]

        with mock.patch.object(service_control, "kill_verified") as kill, \
                mock.patch.object(service_control.Path, "write_text") as write:
            stopped = service_control.stop_running_tasks(store, "по кнопке")

        self.assertEqual(stopped, 1)
        write.assert_called_once_with("stop", encoding="utf-8")
        kill.assert_called_once_with(4242, "task_worker.py", "t1")
        self.assertEqual(task["state"], "stopped")
        self.assertEqual(task["substage"], "по кнопке")
        self.assertIsNone(task["pid"])
        store.save_task.assert_called_once_with(task)

    def test_no_live_tasks_is_not_an_error(self) -> None:
        store = mock.Mock()
        store.list_tasks.return_value = []

        self.assertEqual(service_control.stop_running_tasks(store, "x"), 0)
        store.save_task.assert_not_called()

    def test_shutdown_worker_exits_even_when_stopping_tasks_fails(self) -> None:
        # A worker that refuses to die must not keep the panel alive: the
        # operator asked for the service to close.
        pid_file = mock.Mock()
        with mock.patch.object(
            app_module, "stop_running_tasks", side_effect=RuntimeError("нет")
        ), mock.patch.object(app_module.time, "sleep"), \
                mock.patch.object(app_module.os, "_exit") as exit_call, \
                mock.patch.object(app_module, "PID_FILE", pid_file):
            app_module._shutdown_service()

        pid_file.unlink.assert_called_once_with(missing_ok=True)
        exit_call.assert_called_once_with(0)


class ServiceIndicatorUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html_source = (
            ROOT / "docs" / "концепт интерфейса" / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")

    def test_indicator_sits_above_settings_in_the_sidebar(self) -> None:
        system_block = self.html_source.split(
            ">Система</div>", 1
        )[1].split("</aside>", 1)[0]
        status = system_block.find('class="dc-service-status"')
        settings = system_block.find("{{ goSettings }}")

        self.assertGreater(status, 0)
        self.assertGreater(settings, status)

    def test_stop_button_is_revealed_by_hover_not_shown_outright(self) -> None:
        self.assertIn(".dc-service-stop{", self.html_source)
        self.assertIn("visibility:hidden", self.html_source)
        self.assertIn(
            ".dc-service-status:hover .dc-service-stop,"
            ".dc-service-status:focus-within .dc-service-stop"
            "{opacity:1;visibility:visible",
            self.html_source,
        )

    def test_stop_button_exists_only_while_the_service_answers(self) -> None:
        self.assertIn(
            '<sc-if value="{{ serviceOnline }}" hint-placeholder-val="{{ true }}">\n'
            '          <button class="dc-service-stop"',
            self.html_source,
        )

    def test_shutdown_sends_the_confirming_header(self) -> None:
        method = self.html_source.split("async shutdownService() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("'/api/system/shutdown'", method)
        self.assertIn("'X-DubClean-Local':'shutdown'", method)
        self.assertIn("window.close()", method)

    def test_running_tasks_are_confirmed_before_stopping(self) -> None:
        method = self.html_source.split("async shutdownService() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("window.confirm", method)
        self.assertIn("['queued','starting','running'].includes(task.state)", method)

    def test_health_poll_starts_and_is_cleared(self) -> None:
        self.assertIn(
            "this.__servicePoll = setInterval(() => this.pingService(), 5000);",
            self.html_source,
        )
        self.assertIn(
            "if (this.__servicePoll) clearInterval(this.__servicePoll);",
            self.html_source,
        )

    def test_poll_cannot_overwrite_the_stopping_state(self) -> None:
        method = self.html_source.split("async pingService() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("if (this.state.serviceStopping) return;", method)

    def test_liveness_is_decided_by_an_answer_not_by_its_status(self) -> None:
        # A reachable panel running an older backend answers 403 from its
        # catch-all route; reading that as "stopped" would be a plain lie.
        method = self.html_source.split("async pingService() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertNotIn("response.ok", method)
        self.assertIn("this.setState({ serviceOnline: true });", method)
        self.assertIn("catch (_) {", method)
        self.assertIn("this.setState({ serviceOnline: false });", method)

    def test_unknown_state_reads_as_connecting_not_broken(self) -> None:
        self.assertIn("serviceOnline: null,", self.html_source)
        self.assertIn(
            "serviceStatusLabel: s.serviceOnline === null\n"
            "        ? 'Подключение…'",
            self.html_source,
        )


if __name__ == "__main__":
    unittest.main()
