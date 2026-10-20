from __future__ import annotations

import unittest
from unittest import mock

from experiments.paired_reference_cancel import app as app_module


class LocalWebSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app_module.app.test_client()

    def test_only_loopback_hostnames_are_accepted(self) -> None:
        self.assertTrue(app_module._is_loopback_hostname("127.0.0.1"))
        self.assertTrue(app_module._is_loopback_hostname("::1"))
        self.assertTrue(app_module._is_loopback_hostname("localhost"))
        self.assertFalse(app_module._is_loopback_hostname("example.com"))
        self.assertFalse(app_module._is_loopback_hostname("0.0.0.0"))

    def test_dns_rebinding_host_is_rejected(self) -> None:
        response = self.client.get("/product", headers={"Host": "example.com"})

        self.assertEqual(response.status_code, 403)

    def test_cross_origin_write_is_rejected_before_endpoint(self) -> None:
        response = self.client.post(
            "/api/reveal",
            json={"path": "anything"},
            headers={"Origin": "https://example.com"},
        )

        self.assertEqual(response.status_code, 403)

    def test_product_response_has_security_headers(self) -> None:
        response = self.client.get("/product")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertIn("'unsafe-eval'", response.headers["Content-Security-Policy"])

    def test_application_catalog_reuses_the_expensive_scan(self) -> None:
        app_module._invalidate_application_catalog()
        payload = {"applications": [{"id": "project"}]}
        with mock.patch.object(
            app_module, "_build_application_catalog", return_value=payload
        ) as build:
            first = app_module._application_catalog_payload()
            second = app_module._application_catalog_payload()
            self.assertEqual(first, payload)
            self.assertEqual(second, payload)
            self.assertEqual(build.call_count, 1)

            app_module._invalidate_application_catalog()
            app_module._application_catalog_payload()
            self.assertEqual(build.call_count, 2)

    def test_application_detail_coalesces_simultaneous_tab_polls(self) -> None:
        app_module._invalidate_application_detail()
        payload = {"project": {"id": "project"}, "pair": None, "tasks": []}
        with mock.patch.object(
            app_module, "_build_application_detail", return_value=payload
        ) as build:
            first = app_module._application_detail_payload("project")
            second = app_module._application_detail_payload("project")
            self.assertEqual(first, payload)
            self.assertEqual(second, payload)
            self.assertEqual(build.call_count, 1)

            app_module._invalidate_application_detail("project")
            app_module._application_detail_payload("project")
            self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
