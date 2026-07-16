from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row

from email_thread_rag.rag.paradedb.repository import apply_migrations, verify_extensions

BASE_DATABASE_URL = os.getenv("DATABASE_URL")


def _require_admin_url() -> str:
    if not BASE_DATABASE_URL:
        pytest.skip("DATABASE_URL not set; start docker-compose.yml's `db` service to run integration tests")
    return BASE_DATABASE_URL


@pytest.fixture(scope="session")
def migrated_database_url():
    """Create a throwaway database, migrate it, yield its URL, then drop it.

    Isolates integration tests from any DB the developer already has data in,
    while still exercising "apply migrations to a clean database" for real.
    """
    admin_url = _require_admin_url()
    db_name = f"email_rag_test_{uuid.uuid4().hex[:8]}"
    admin_conn = psycopg.connect(admin_url, autocommit=True, row_factory=dict_row)
    try:
        admin_conn.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        admin_conn.close()

    test_url = _swap_database_name(admin_url, db_name)
    conn = psycopg.connect(test_url, row_factory=dict_row)
    # Extensions are per-database; the fresh throwaway database has neither yet.
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    verify_extensions(conn)
    apply_migrations(conn)
    conn.close()

    yield test_url

    admin_conn = psycopg.connect(admin_url, autocommit=True, row_factory=dict_row)
    try:
        admin_conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    finally:
        admin_conn.close()


def _swap_database_name(url: str, new_db_name: str) -> str:
    base, _, _old_db = url.rpartition("/")
    return f"{base}/{new_db_name}"


@pytest.fixture
def db_conn(migrated_database_url):
    conn = psycopg.connect(migrated_database_url, row_factory=dict_row)
    yield conn
    conn.rollback()
    conn.close()
