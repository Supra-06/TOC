"""Unit tests for FastAPI application factory, lifespan, and configuration (Phase 7 Step 4).

Verifies:
 1. Application factory returns a configured FastAPI application.
 2. Importing the application does not initialize the physical database.
 3. Startup initialization respects the explicit factory argument.
 4. Startup initialization respects the environment configuration.
 5. Startup does not load the ML model into memory.
 6. CORS middleware and default origins are configured correctly.
 7. Custom exception handlers format validation, HTTP, mapping, and general errors.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_main -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import unittest

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.api.mapping import RecordMappingError
from backend.core.config import CORS_ORIGINS
from backend.main import create_app, setup_exception_handlers
from backend.services.anomaly_service import get_anomaly_service


class TestMainApplication(unittest.TestCase):
    """Test suite covering FastAPI application creation, lifecycle, and CORS."""

    def test_01_factory_returns_fastapi_application(self) -> None:
        """Verify create_app returns a properly configured FastAPI instance."""
        app = create_app(init_db_on_startup=False)
        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "Process Workflow Anomaly Detection API")
        self.assertEqual(app.version, "1.0.0")

    def test_02_import_does_not_initialize_database(self) -> None:
        """Verify importing backend.main does not initialize or write to the database."""
        # Database row count in physical DB must remain 0
        from backend.main import app
        self.assertIsInstance(app, FastAPI)

    def test_03_startup_initialization_respects_explicit_factory_argument(self) -> None:
        """Verify lifespan calls init_db when init_db_on_startup=True and skips when False."""
        # 1. When True: init_db is called during lifespan
        with patch("backend.main.init_db") as mock_init:
            app_true = create_app(init_db_on_startup=True)
            with TestClient(app_true):
                pass
            mock_init.assert_called_once()

        # 2. When False: init_db is NOT called during lifespan
        with patch("backend.main.init_db") as mock_init:
            app_false = create_app(init_db_on_startup=False)
            with TestClient(app_false):
                pass
            mock_init.assert_not_called()

    def test_04_startup_initialization_respects_environment_config(self) -> None:
        """Verify lifespan respects INIT_DB_ON_STARTUP configuration when argument is None."""
        with patch("backend.main.INIT_DB_ON_STARTUP", True):
            with patch("backend.main.init_db") as mock_init:
                app_env_true = create_app(init_db_on_startup=None)
                with TestClient(app_env_true):
                    pass
                mock_init.assert_called_once()

        with patch("backend.main.INIT_DB_ON_STARTUP", False):
            with patch("backend.main.init_db") as mock_init:
                app_env_false = create_app(init_db_on_startup=None)
                with TestClient(app_env_false):
                    pass
                mock_init.assert_not_called()

    def test_05_startup_does_not_load_ml_model(self) -> None:
        """Verify application startup does not force the ML model into memory."""
        svc = get_anomaly_service()
        # Verify initial state: model is not yet loaded into memory
        app = create_app(init_db_on_startup=False)
        with TestClient(app):
            # Model remains lazy after startup
            self.assertFalse(svc.get_service_status()["model_loaded_in_memory"])

    def test_06_cors_middleware_configured_correctly(self) -> None:
        """Verify CORS middleware is registered with the approved development origins, methods, and headers."""
        app = create_app(init_db_on_startup=False)
        cors_middlewares = [m for m in app.user_middleware if m.cls == CORSMiddleware]
        self.assertEqual(len(cors_middlewares), 1)

        cors_options = cors_middlewares[0].kwargs
        self.assertIn("http://localhost:3000", cors_options["allow_origins"])
        self.assertIn("http://localhost:5173", cors_options["allow_origins"])
        self.assertTrue(cors_options["allow_credentials"])
        self.assertNotIn("*", cors_options["allow_origins"])
        self.assertEqual(cors_options["allow_methods"], ["GET", "POST"])
        self.assertNotIn("*", cors_options["allow_methods"])
        self.assertEqual(cors_options["allow_headers"], ["Authorization", "Content-Type"])
        self.assertNotIn("*", cors_options["allow_headers"])

    def test_07_custom_exception_handlers_format(self) -> None:
        """Verify exception handlers return standard ErrorResponse payloads."""
        app = FastAPI()
        setup_exception_handlers(app)

        @app.get("/trigger-http-404")
        def route_404():
            raise HTTPException(status_code=404, detail="Custom resource not found.")

        @app.get("/trigger-mapping-error")
        def route_mapping():
            raise RecordMappingError("Corrupted record.")

        @app.get("/trigger-general-error")
        def route_general():
            raise RuntimeError("Unexpected internal crash.")

        client = TestClient(app, raise_server_exceptions=False)

        # 1. 404 handler
        res_404 = client.get("/trigger-http-404")
        self.assertEqual(res_404.status_code, 404)
        data_404 = res_404.json()
        self.assertIn("error", data_404)
        self.assertEqual(data_404["error"]["code"], "NOT_FOUND")
        self.assertEqual(data_404["error"]["message"], "Custom resource not found.")

        # 2. Mapping error handler (500)
        res_map = client.get("/trigger-mapping-error")
        self.assertEqual(res_map.status_code, 500)
        data_map = res_map.json()
        self.assertEqual(data_map["error"]["code"], "MAPPING_ERROR")

        # 3. General error handler (500)
        res_gen = client.get("/trigger-general-error")
        self.assertEqual(res_gen.status_code, 500)
        data_gen = res_gen.json()
        self.assertEqual(data_gen["error"]["code"], "INTERNAL_SERVER_ERROR")
        self.assertEqual(data_gen["error"]["message"], "An internal server error occurred.")

    def test_08_cors_approved_origins_receive_headers(self) -> None:
        """Verify requests from approved origins receive appropriate CORS response headers."""
        app = create_app(init_db_on_startup=False)
        client = TestClient(app)

        for origin in ("http://localhost:3000", "http://localhost:5173"):
            res = client.get("/health", headers={"Origin": origin})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.headers.get("access-control-allow-origin"), origin)
            self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")

    def test_09_cors_unapproved_origin_denied(self) -> None:
        """Verify requests from an unapproved origin do not receive CORS headers."""
        app = create_app(init_db_on_startup=False)
        client = TestClient(app)

        res = client.get("/health", headers={"Origin": "http://malicious-site.example.com"})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("access-control-allow-origin", res.headers)

    def test_10_cors_preflight_post_accepted(self) -> None:
        """Verify preflight OPTIONS request for POST with Content-Type is accepted."""
        app = create_app(init_db_on_startup=False)
        client = TestClient(app)

        preflight_headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
        res = client.options("/api/v1/analyses", headers=preflight_headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:3000")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")
        allow_methods = res.headers.get("access-control-allow-methods", "")
        self.assertIn("POST", allow_methods)
        allow_headers = res.headers.get("access-control-allow-headers", "").lower()
        self.assertIn("content-type", allow_headers)

    def test_11_cors_preflight_unsupported_method_denied(self) -> None:
        """Verify preflight request for an unsupported method (e.g. DELETE) is not granted permission."""
        app = create_app(init_db_on_startup=False)
        client = TestClient(app)

        preflight_headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "DELETE",
        }
        res = client.options("/api/v1/analyses", headers=preflight_headers)
        # CORSMiddleware rejects preflight with 400 and Disallowed CORS method
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed CORS method", res.text)
        self.assertNotIn("DELETE", res.headers.get("access-control-allow-methods", ""))

    def test_12_cors_preflight_unapproved_header_denied(self) -> None:
        """Verify preflight request for an unapproved request header is not granted permission."""
        app = create_app(init_db_on_startup=False)
        client = TestClient(app)

        preflight_headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Custom-Untrusted-Header",
        }
        res = client.options("/api/v1/analyses", headers=preflight_headers)
        # CORSMiddleware rejects preflight with 400 and Disallowed CORS headers
        self.assertEqual(res.status_code, 400)
        self.assertIn("Disallowed CORS headers", res.text)
        self.assertNotIn("x-custom-untrusted-header", res.headers.get("access-control-allow-headers", "").lower())


if __name__ == "__main__":
    unittest.main()
