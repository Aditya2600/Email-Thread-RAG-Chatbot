"""Stage-7 must not turn the optional answer layer into a required one.

The provider seam keeps httpx lazy, the grounded flow imports no provider SDK /
psycopg / torch, the engine only reaches the answer modules through a
function-local import behind the enabled flag, and a disabled provider is inert
with no network.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LAZY_ONLY = ("httpx", "psycopg", "openai", "torch", "sentence_transformers", "transformers")
ANSWER_MODULES = (
    "email_thread_rag/rag/answer_provider.py",
    "email_thread_rag/rag/grounded_answer.py",
)


def _top_level_imports(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_answer_modules_keep_optional_dependencies_lazy():
    for relative_path in ANSWER_MODULES:
        for name in _top_level_imports((REPO_ROOT / relative_path).read_text(encoding="utf-8")):
            root = name.split(".")[0]
            assert root not in LAZY_ONLY, f"{relative_path} imports {name!r} at module level; keep it function-local"


def test_engine_does_not_import_the_answer_modules_at_module_level():
    source = (REPO_ROOT / "email_thread_rag/rag/engine.py").read_text(encoding="utf-8")
    for name in _top_level_imports(source):
        assert name not in (
            "email_thread_rag.rag.answer_provider",
            "email_thread_rag.rag.grounded_answer",
        ), f"engine.py must reach {name!r} lazily, behind the answer_generation_enabled flag"


def test_answer_modules_import_without_a_provider_or_a_socket():
    script = """
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("httpx", "psycopg", "openai")
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

from email_thread_rag.rag.answer_provider import build_answer_provider  # noqa: F401
from email_thread_rag.rag.grounded_answer import GroundedAnswerer  # noqa: F401
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "EMAIL_RAG_SKIP_DOTENV": "1"},
    )
    assert result.returncode == 0, f"answer modules need optional deps to import:\n{result.stderr}"
    assert "ok" in result.stdout


def test_build_answer_provider_returns_none_when_disabled(tmp_path):
    from email_thread_rag.config import Settings
    from email_thread_rag.rag.answer_provider import build_answer_provider

    assert build_answer_provider(Settings(project_root=tmp_path, answer_generation_enabled=False)) is None


def test_disabled_engine_builds_no_grounded_answerer(tmp_path):
    from email_thread_rag.config import Settings
    from email_thread_rag.rag.engine import RAGEngine

    settings = Settings(project_root=tmp_path, answer_generation_enabled=False)

    class StubRetriever:
        def available_threads(self):
            return []

    engine = RAGEngine(settings, retriever=StubRetriever())
    assert engine.grounded_answerer is None
