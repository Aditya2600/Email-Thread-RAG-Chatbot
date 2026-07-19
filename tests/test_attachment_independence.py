"""Stage-8 must keep its heavy dependencies lazy.

fitz/pymupdf, pytesseract, PIL, and psycopg are all imported function-locally, so
importing the attachment package (types, store, models) never requires them. The
extraction worker pulls them in only when it actually runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

LAZY_ONLY = ("fitz", "pytesseract", "PIL", "psycopg")
ATTACHMENT_MODULES = (
    "email_thread_rag/rag/attachments/__init__.py",
    "email_thread_rag/rag/attachments/models.py",
    "email_thread_rag/rag/attachments/ocr.py",
    "email_thread_rag/rag/attachments/pdf.py",
    "email_thread_rag/rag/attachments/store.py",
    "email_thread_rag/rag/attachments/repository.py",
    "email_thread_rag/rag/attachments/worker.py",
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


@pytest.mark.parametrize("relative_path", ATTACHMENT_MODULES)
def test_attachment_modules_keep_heavy_deps_lazy(relative_path):
    for name in _top_level_imports((REPO_ROOT / relative_path).read_text(encoding="utf-8")):
        root = name.split(".")[0]
        assert root not in LAZY_ONLY, f"{relative_path} imports {name!r} at module level; keep it function-local"


def test_attachment_types_import_without_fitz_or_psycopg():
    import subprocess
    import sys

    script = """
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("fitz", "pytesseract", "PIL", "psycopg")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{name} is blocked for this test")
        return None

for name in [m for m in sys.modules if m.split(".")[0] in Blocker.BLOCKED]:
    del sys.modules[name]
sys.meta_path.insert(0, Blocker())

from email_thread_rag.rag.attachments.models import AttachmentMeta  # noqa: F401
from email_thread_rag.rag.attachments.store import InMemoryAttachmentJobStore  # noqa: F401
from email_thread_rag.rag.attachments.pdf import extract_pdf  # noqa: F401
from email_thread_rag.rag.attachments.worker import AttachmentExtractionWorker  # noqa: F401
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "EMAIL_RAG_SKIP_DOTENV": "1"},
    )
    assert result.returncode == 0, f"attachment types need a heavy dep to import:\n{result.stderr}"
    assert "ok" in result.stdout
