# Noncanonical — do not import

This `src/` tree is an **incomplete Stage-1 refactor scaffold** (Postgres/ParadeDB,
Gmail sync, HyDE, Self-RAG, etc.). It contains truncated files and empty stubs and is
**not** part of the runnable Stage-0 baseline.

- Canonical package: `email_thread_rag/` at the repo root.
- `src/` is excluded from packaging (`pyproject.toml` → `tool.setuptools.packages.find`).
- Nothing in the working baseline or tests imports from `src/`.

Kept in-tree only as reference for Stage 1+ work. Do not wire it into imports.
