"""Unit tests for FastAPI dependencies (Phase 7 Step 3).

Verifies:
 1. get_db_session() yields an active, usable SQLAlchemy session.
 2. get_db_session() guarantees session closing upon completion.
 3. get_db_session() guarantees session closing even when an exception is raised.
 4. get_service() returns the shared AnomalyService singleton without forced model loading.
 5. get_db_session and get_service can be cleanly overridden using FastAPI's dependency override mechanism.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_api_dependencies -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import Depends, FastAPI
from fastapi.dependencies.utils import get_dependant
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.dependencies import get_db_session, get_service
import backend.database.connection as conn
from backend.services.anomaly_service import AnomalyService, get_anomaly_service


class TestApiDependencies(unittest.TestCase):
    """Test suite covering FastAPI dependencies for sessions and anomaly service."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create an isolated in-memory engine and sessionmaker to prevent touching physical DB."""
        cls.test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        cls.TestSessionLocal = sessionmaker(
            bind=cls.test_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    def test_01_get_db_session_yields_usable_session(self) -> None:
        """Verify get_db_session yields an active, usable SQLAlchemy session."""
        with patch.object(conn, "SessionLocal", self.TestSessionLocal):
            gen = get_db_session()
            session = next(gen)
            try:
                self.assertIsInstance(session, Session)
                self.assertTrue(session.is_active)
                result = session.execute(text("SELECT 1")).scalar()
                self.assertEqual(result, 1)
            finally:
                try:
                    next(gen)
                except StopIteration:
                    pass

    def test_02_get_db_session_closed_after_completion(self) -> None:
        """Verify the yielded session is closed after dependency generator exits."""
        with patch.object(conn, "SessionLocal", self.TestSessionLocal):
            gen = get_db_session()
            session = next(gen)
            close_spy = MagicMock(wraps=session.close)
            session.close = close_spy

            # Exhaust the generator to trigger the finally block
            try:
                next(gen)
            except StopIteration:
                pass

            close_spy.assert_called_once()

    def test_03_get_db_session_closed_on_exception(self) -> None:
        """Verify exceptions during request processing do not leave sessions open."""
        with patch.object(conn, "SessionLocal", self.TestSessionLocal):
            gen = get_db_session()
            session = next(gen)
            close_spy = MagicMock(wraps=session.close)
            session.close = close_spy

            # Simulate an unhandled exception thrown into the generator context
            with self.assertRaises(RuntimeError):
                gen.throw(RuntimeError("Simulated client error during request handling"))

            close_spy.assert_called_once()

    def test_04_get_service_returns_anomaly_service(self) -> None:
        """Verify get_service returns the existing AnomalyService singleton without forced model loading."""
        svc = get_service()
        self.assertIsInstance(svc, AnomalyService)
        # Calling get_service must match get_anomaly_service singleton
        self.assertIs(svc, get_anomaly_service())

        # Verify operational status is accessible without error
        status = svc.get_service_status()
        self.assertIn(status["service_status"], ("OPERATIONAL", "DEGRADED"))
        self.assertEqual(status["dataset_spec_version"], "1.4")

    def test_05_dependencies_overridable_via_fastapi(self) -> None:
        """Verify database and service dependencies can be overridden through FastAPI dependency_overrides."""
        test_app = FastAPI()

        def test_endpoint(
            session: Session = Depends(get_db_session),
            service: AnomalyService = Depends(get_service),
        ):
            return {"session": session, "service": service}

        # Verify dependency introspection recognizes both dependencies
        dependant = get_dependant(path="/test", call=test_endpoint)
        dep_calls = [d.call for d in dependant.dependencies]
        self.assertIn(get_db_session, dep_calls)
        self.assertIn(get_service, dep_calls)

        # Configure dependency overrides
        mock_session_obj = "MOCK_DB_SESSION"
        mock_service_obj = "MOCK_ANOMALY_SERVICE"

        def override_db():
            yield mock_session_obj

        def override_svc():
            return mock_service_obj

        test_app.dependency_overrides[get_db_session] = override_db
        test_app.dependency_overrides[get_service] = override_svc

        # Verify overrides are recorded and resolvable
        self.assertIs(test_app.dependency_overrides[get_db_session], override_db)
        self.assertIs(test_app.dependency_overrides[get_service], override_svc)

        resolved_db = test_app.dependency_overrides[get_db_session]()
        self.assertEqual(next(resolved_db), mock_session_obj)

        resolved_svc = test_app.dependency_overrides[get_service]()
        self.assertEqual(resolved_svc, mock_service_obj)

        # Cleanup overrides
        test_app.dependency_overrides.clear()
        self.assertEqual(len(test_app.dependency_overrides), 0)


if __name__ == "__main__":
    unittest.main()

