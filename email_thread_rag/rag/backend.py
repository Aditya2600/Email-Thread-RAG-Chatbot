"""Stage-2.5 backend selection: memory (default) or paradedb.

Nothing here imports psycopg at module level -- the paradedb branch imports
it lazily, inside ``build_retriever``, only when actually selected. That
keeps ``import email_thread_rag.rag.engine`` (which imports this module)
free of any Postgres connection attempt for the memory-only path.
"""

from __future__ import annotations

from email_thread_rag.config import Settings


def build_retriever(settings: Settings):
    """Return a retriever exposing ``.available_threads()`` / ``.search(query, *, thread_id=None)``.

    memory: existing in-memory ``HybridRetriever``, unchanged default.
    paradedb: connects and verifies extensions immediately -- a misconfigured
    explicit paradedb selection (missing DATABASE_URL, unreachable database,
    missing pg_search/vector) raises ``ParadeDBConfigError`` here rather than
    silently continuing on the memory backend.
    """
    if settings.rag_backend == "memory":
        from email_thread_rag.rag.retrieval import HybridRetriever

        return HybridRetriever.from_chunk_store(settings)

    from email_thread_rag.rag.paradedb.repository import connect, verify_extensions
    from email_thread_rag.rag.paradedb.retrieval import ParadeDBEngineRetriever
    from email_thread_rag.rag.vector_index import SentenceTransformerEncoder

    conn = connect(settings.database_url)
    verify_extensions(conn)
    # Same encoder the memory backend defaults to: tries the real model,
    # falls back to the deterministic HashingEncoder if it can't load one.
    encoder = SentenceTransformerEncoder(settings)
    return ParadeDBEngineRetriever(conn, settings, encoder=encoder)
