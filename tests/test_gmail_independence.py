"""RAG_BACKEND=memory must never depend on Gmail packages, credentials, or Postgres.

Stage 3 adds an optional dependency path; these tests are what stop it from
quietly becoming a required one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The memory retrieval path plus the Stage-2.5 backend selector: none of them
# may know Gmail exists.
MEMORY_PATH_MODULES = [
    "email_thread_rag/config.py",
    "email_thread_rag/app/main.py",
    "email_thread_rag/rag/backend.py",
    "email_thread_rag/rag/engine.py",
    "email_thread_rag/rag/retrieval.py",
    "email_thread_rag/rag/chunking.py",
]

# Modules the Gmail package may not import at module level. google-* and
# psycopg are lazy so an install without the extra still imports cleanly;
# cryptography is lazy for the same reason.
LAZY_ONLY_MODULES = ("psycopg", "cryptography", "google", "google.auth", "google.oauth2", "httpx")

GMAIL_PACKAGE_FILES = sorted(p for p in (REPO_ROOT / "email_thread_rag" / "gmail").glob("*.py"))


def top_level_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:  # module top level only; function-local imports are lazy by definition
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("relative_path", MEMORY_PATH_MODULES)
def test_memory_path_modules_never_import_gmail(relative_path):
    for name in top_level_imports((REPO_ROOT / relative_path).read_text(encoding="utf-8")):
        assert not name.startswith("email_thread_rag.gmail"), (
            f"{relative_path} must not import the gmail package at module level, found {name!r}"
        )


@pytest.mark.parametrize("path", GMAIL_PACKAGE_FILES, ids=lambda p: p.name)
def test_gmail_package_keeps_optional_dependencies_lazy(path):
    for name in top_level_imports(path.read_text(encoding="utf-8")):
        root = name.split(".")[0]
        assert root not in LAZY_ONLY_MODULES, (
            f"{path.name} imports {name!r} at module level; it must be a function-local import "
            "so installs without the 'gmail'/'postgres' extras still work"
        )


def test_memory_engine_imports_without_gmail_or_postgres_installed():
    """Import the memory path in a subprocess where psycopg/cryptography/google
    are un-importable. If anything in the memory path reaches for them, this
    fails the way a user's install would."""
    # find_spec, not the legacy find_module/load_module pair: Python 3.12
    # removed the latter, which would make this blocker silently do nothing.
    script = """
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("psycopg", "cryptography", "google", "googleapiclient")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{name} is blocked for this test")
        return None

for name in [m for m in sys.modules if m.split(".")[0] in Blocker.BLOCKED]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

try:
    import psycopg
except ImportError:
    pass
else:
    raise AssertionError("blocker is not working; this test would prove nothing")

import email_thread_rag.app.main  # noqa: F401
import email_thread_rag.rag.backend  # noqa: F401
from email_thread_rag.rag.engine import RAGEngine  # noqa: F401
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "RAG_BACKEND": "memory", "HOME": "/tmp", "EMAIL_RAG_SKIP_DOTENV": "1"},
    )
    assert result.returncode == 0, f"memory path failed without optional deps:\n{result.stderr}"
    assert "ok" in result.stdout


def test_memory_backend_needs_no_gmail_configuration(monkeypatch, tmp_path):
    for name in (
        "GMAIL_CLIENT_ID",
        "GMAIL_CLIENT_SECRET",
        "GMAIL_TOKEN_ENCRYPTION_KEY",
        "GMAIL_PUBSUB_TOPIC",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    from email_thread_rag.config import Settings

    settings = Settings(project_root=tmp_path, rag_backend="memory")
    assert settings.gmail_client_id is None
    assert settings.gmail_token_encryption_key is None
    assert settings.rag_backend == "memory"
