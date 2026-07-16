from __future__ import annotations

import ast
from pathlib import Path

import pytest

from email_thread_rag.rag.paradedb.repository import ParadeDBConfigError, connect


# rag/backend.py is the intentional boundary: it references paradedb/psycopg,
# but only inside build_retriever()'s paradedb branch (a lazy, function-local
# import), never at module level. Everything else must have zero awareness.
MEMORY_PATH_MODULES = [
    "email_thread_rag/rag/bm25_index.py",
    "email_thread_rag/rag/vector_index.py",
    "email_thread_rag/rag/retrieval.py",
    "email_thread_rag/rag/chunking.py",
    "email_thread_rag/rag/engine.py",
    "email_thread_rag/rag/answer.py",
    "email_thread_rag/rag/fusion.py",
]
BOUNDARY_MODULES = MEMORY_PATH_MODULES + ["email_thread_rag/rag/backend.py"]


def _top_level_import_names(source: str) -> set[str]:
    """Module names imported by top-level (non-lazy) import statements only."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:  # tree.body is module top level only; skips FunctionDef bodies
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_explicit_paradedb_selection_fails_clearly_without_database_url():
    with pytest.raises(ParadeDBConfigError, match="DATABASE_URL"):
        connect(None)


def test_explicit_paradedb_selection_fails_clearly_on_unreachable_database_url():
    # Valid URL shape, nothing listening -> a clear ParadeDBConfigError, not a
    # silent fallback to the memory backend and not a raw psycopg traceback.
    with pytest.raises(ParadeDBConfigError, match="Could not connect"):
        connect("postgresql://user:pass@127.0.0.1:1/nonexistent_db")


def test_memory_path_modules_never_import_psycopg_at_module_level():
    # AST-based (not substring) so this is true regardless of whether the
    # postgres extra happens to be installed, and regardless of comments that
    # merely *mention* paradedb by name (Stage 2.5's engine.py duck-types over
    # either retriever and says so in a docstring/comment -- that's not an
    # import).
    repo_root = Path(__file__).resolve().parent.parent
    for relative_path in BOUNDARY_MODULES:
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        top_level_imports = _top_level_import_names(source)
        for name in top_level_imports:
            assert name != "psycopg" and not name.startswith("psycopg."), (
                f"{relative_path} must not import psycopg at module level, found {name!r}"
            )
            assert not name.startswith("email_thread_rag.rag.paradedb"), (
                f"{relative_path} must not import rag.paradedb at module level, found {name!r}"
            )
