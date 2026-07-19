"""RAG_BACKEND=memory and disabled graph extraction must never depend on
psycopg, an LLM client, a model, or a socket.

Same approach as the Stage-3/Stage-4 boundary tests, extended to the graph
package. These are what stop the optional Stage-5 path from quietly becoming a
required one.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

MEMORY_PATH_MODULES = [
    "email_thread_rag/config.py",
    "email_thread_rag/app/main.py",
    "email_thread_rag/rag/backend.py",
    "email_thread_rag/rag/engine.py",
    "email_thread_rag/rag/retrieval.py",
    "email_thread_rag/rag/chunking.py",
]

LAZY_ONLY_MODULES = ("psycopg", "httpx", "openai", "torch", "sentence_transformers", "transformers")
LAZY_EXEMPT = {"repository.py"}  # IS the Postgres store; reached only via function-local import

GRAPH_PACKAGE_FILES = sorted(
    p for p in (REPO_ROOT / "email_thread_rag" / "graph").glob("*.py") if p.name not in LAZY_EXEMPT
)


def top_level_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("relative_path", MEMORY_PATH_MODULES)
def test_memory_path_modules_never_import_the_graph_package(relative_path):
    for name in top_level_imports((REPO_ROOT / relative_path).read_text(encoding="utf-8")):
        assert not name.startswith("email_thread_rag.graph"), (
            f"{relative_path} must not import the graph package at module level, found {name!r}"
        )


@pytest.mark.parametrize("path", GRAPH_PACKAGE_FILES, ids=lambda p: p.name)
def test_graph_package_keeps_optional_dependencies_lazy(path):
    for name in top_level_imports(path.read_text(encoding="utf-8")):
        root = name.split(".")[0]
        assert root not in LAZY_ONLY_MODULES, (
            f"{path.name} imports {name!r} at module level; it must be a function-local import"
        )


def test_the_ingestion_seam_imports_the_graph_package_lazily():
    for relative_path in ("email_thread_rag/rag/paradedb/ingest.py", "email_thread_rag/gmail/sink.py"):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for name in top_level_imports(source):
            assert not name.startswith("email_thread_rag.graph"), (
                f"{relative_path} imports {name!r} at module level; keep it function-local"
            )
        assert "from email_thread_rag.graph.enqueue import" in source


def test_the_graph_package_imports_without_psycopg_or_a_provider():
    """Pure logic (prompt, fingerprint, extract, in-memory store) must import
    with no database driver and no HTTP client present."""
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

from email_thread_rag.graph.prompt import validate_extraction  # noqa: F401
from email_thread_rag.graph.fingerprint import extraction_hash_of  # noqa: F401
from email_thread_rag.graph.extract import resolve_extraction  # noqa: F401
from email_thread_rag.graph.store import InMemoryGraphStore  # noqa: F401
from email_thread_rag.graph.provider import MedhaGraphExtractor  # noqa: F401
from email_thread_rag.graph.enqueue import enqueue_message_graph  # noqa: F401
from email_thread_rag.graph.retrieval import collect_graph_chunk_ids  # noqa: F401
from email_thread_rag.rag.planner import plan_query  # noqa: F401
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert result.returncode == 0, f"graph package needs optional deps to import:\n{result.stderr}"
    assert "ok" in result.stdout


def test_enqueue_is_inert_when_graph_extraction_is_disabled(tmp_path):
    from email_thread_rag.config import Settings
    from email_thread_rag.graph.enqueue import enqueue_message_graph

    class ExplodingConnection:
        def execute(self, *args, **kwargs):
            raise AssertionError("the disabled path must not touch the database")

    settings = Settings(project_root=tmp_path, graph_extraction_enabled=False)
    assert enqueue_message_graph(ExplodingConnection(), "<m@x>", tenant_id="acme", mailbox_id="inbox", settings=settings) == 0


def test_enqueue_is_inert_when_settings_are_absent():
    from email_thread_rag.graph.enqueue import enqueue_message_graph

    class ExplodingConnection:
        def execute(self, *args, **kwargs):
            raise AssertionError("the disabled path must not touch the database")

    assert enqueue_message_graph(ExplodingConnection(), "<m@x>", tenant_id="acme", mailbox_id="inbox", settings=None) == 0


def test_memory_backend_needs_no_graph_configuration(monkeypatch, tmp_path):
    for name in ("GRAPH_EXTRACTION_ENABLED", "GRAPH_BASE_URL", "GRAPH_MODEL", "GRAPH_API_KEY",
                 "MEDHA_BASE_URL", "MEDHA_MODEL", "MEDHA_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    from email_thread_rag.config import Settings

    settings = Settings(project_root=tmp_path, rag_backend="memory")
    assert settings.graph_extraction_enabled is False
    assert settings.graph_base_url is None and settings.graph_api_key is None
