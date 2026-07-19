"""RAG_BACKEND=memory must never depend on psycopg, an LLM client, or a model.

Stage 4 adds an optional contextualization path; these tests are what stop it
from quietly becoming a required one. Same approach as the Stage-3 boundary
tests, extended to cover the context package.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The memory retrieval path plus the Stage-2.5 backend selector: none of them
# may know contextualization exists.
MEMORY_PATH_MODULES = [
    "email_thread_rag/config.py",
    "email_thread_rag/app/main.py",
    "email_thread_rag/rag/backend.py",
    "email_thread_rag/rag/engine.py",
    "email_thread_rag/rag/retrieval.py",
    "email_thread_rag/rag/chunking.py",
    "email_thread_rag/rag/email_segmentation.py",
]

# Modules the context package may not import at module level, so an install
# without the extras still imports cleanly. repository.py is exempt: it IS the
# Postgres store and is only ever reached through a function-local import.
LAZY_ONLY_MODULES = ("psycopg", "httpx", "openai", "torch", "sentence_transformers", "transformers")
LAZY_EXEMPT = {"repository.py"}

CONTEXT_PACKAGE_FILES = sorted(
    p for p in (REPO_ROOT / "email_thread_rag" / "context").glob("*.py") if p.name not in LAZY_EXEMPT
)


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
def test_memory_path_modules_never_import_the_context_package(relative_path):
    for name in top_level_imports((REPO_ROOT / relative_path).read_text(encoding="utf-8")):
        assert not name.startswith("email_thread_rag.context"), (
            f"{relative_path} must not import the context package at module level, found {name!r}"
        )


@pytest.mark.parametrize("path", CONTEXT_PACKAGE_FILES, ids=lambda p: p.name)
def test_context_package_keeps_optional_dependencies_lazy(path):
    for name in top_level_imports(path.read_text(encoding="utf-8")):
        root = name.split(".")[0]
        assert root not in LAZY_ONLY_MODULES, (
            f"{path.name} imports {name!r} at module level; it must be a function-local import "
            "so installs without the optional extras still work"
        )


def test_the_ingestion_seam_imports_the_context_package_lazily():
    """persist_corpus_to_paradedb and the Gmail sink must reach the context
    package from inside a function, so the disabled path costs nothing."""
    for relative_path in ("email_thread_rag/rag/paradedb/ingest.py", "email_thread_rag/gmail/sink.py"):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for name in top_level_imports(source):
            assert not name.startswith("email_thread_rag.context"), (
                f"{relative_path} imports {name!r} at module level; keep it function-local"
            )
        assert "from email_thread_rag.context.enqueue import" in source, (
            f"{relative_path} should enqueue context work (lazily)"
        )


def test_memory_engine_imports_without_psycopg_or_an_llm_client():
    """Import the memory path in a subprocess where psycopg/httpx/openai are
    un-importable. If anything in the memory path reaches for them, this fails
    the way a user's install would."""
    # find_spec, not the legacy find_module/load_module pair: Python 3.12
    # removed the latter, which would make this blocker silently do nothing.
    script = """
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("psycopg", "openai", "ollama")
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
from email_thread_rag.rag.chunking import chunk_email  # noqa: F401
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


def test_the_context_package_imports_without_psycopg_or_a_provider():
    """The package's pure logic (prompt, fingerprint, in-memory store) must be
    importable with no database driver and no HTTP client present."""
    script = """
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("psycopg", "httpx", "openai")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{name} is blocked for this test")
        return None

for name in [m for m in sys.modules if m.split(".")[0] in Blocker.BLOCKED]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

try:
    import httpx
except ImportError:
    pass
else:
    raise AssertionError("blocker is not working; this test would prove nothing")

from email_thread_rag.context.prompt import validate_output  # noqa: F401
from email_thread_rag.context.fingerprint import fingerprint_of  # noqa: F401
from email_thread_rag.context.store import InMemoryContextJobStore  # noqa: F401
from email_thread_rag.context.provider import MedhaContextualizer  # noqa: F401
from email_thread_rag.context.enqueue import enqueue_message_context  # noqa: F401
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "EMAIL_RAG_SKIP_DOTENV": "1"},
    )
    assert result.returncode == 0, f"context package needs optional deps to import:\n{result.stderr}"
    assert "ok" in result.stdout


def test_memory_backend_needs_no_context_configuration(monkeypatch, tmp_path):
    for name in (
        "CONTEXT_ENABLED",
        "CONTEXT_BASE_URL",
        "CONTEXT_MODEL",
        "CONTEXT_API_KEY",
        "MEDHA_BASE_URL",
        "MEDHA_MODEL",
        "MEDHA_API_KEY",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    from email_thread_rag.config import Settings

    settings = Settings(project_root=tmp_path, rag_backend="memory")
    assert settings.context_enabled is False
    assert settings.context_base_url is None
    assert settings.context_api_key is None
    assert settings.rag_backend == "memory"


def test_enqueue_is_inert_when_contextualization_is_disabled(tmp_path):
    """The enqueue seam must not touch the connection when disabled -- passing a
    connection that explodes on use proves it never gets there."""
    from email_thread_rag.config import Settings
    from email_thread_rag.context.enqueue import enqueue_message_context

    class ExplodingConnection:
        def execute(self, *args, **kwargs):
            raise AssertionError("the disabled path must not touch the database")

    settings = Settings(project_root=tmp_path, context_enabled=False)
    queued = enqueue_message_context(
        ExplodingConnection(),
        "<msg-1@example.com>",
        tenant_id="acme",
        mailbox_id="inbox",
        settings=settings,
    )
    assert queued == 0


def test_enqueue_is_inert_when_settings_are_absent():
    from email_thread_rag.context.enqueue import enqueue_message_context

    class ExplodingConnection:
        def execute(self, *args, **kwargs):
            raise AssertionError("the disabled path must not touch the database")

    # Callers that predate Stage 4 pass no settings and must be unaffected.
    assert (
        enqueue_message_context(
            ExplodingConnection(), "<msg-1@example.com>", tenant_id="acme", mailbox_id="inbox", settings=None
        )
        == 0
    )
