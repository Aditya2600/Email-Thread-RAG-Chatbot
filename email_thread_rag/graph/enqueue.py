"""Enqueue-after-persistence, for both the local corpus and Gmail paths.

Both ingestion paths call ``enqueue_message_graph`` immediately after the
canonical persistence transaction commits. It is a no-op unless graph extraction
is explicitly enabled, so the default configuration queues nothing and imports
nothing (psycopg-backed modules are reached only from inside this function).

Enqueuing is all these paths do: the provider is never called here. Ingestion
must not block on an LLM, and a webhook must not call one at all.
"""

from __future__ import annotations

from email_thread_rag.graph.fingerprint import PROMPT_VERSION, SCHEMA_VERSION


def graph_identity(settings) -> tuple[str, str, str]:
    """The (schema_version, prompt_version, model_id) triple the fingerprint is
    bound to. Read from settings, not from a constructed provider: enqueuing must
    not build an HTTP client."""
    schema_version = getattr(settings, "graph_schema_version", None) or SCHEMA_VERSION
    prompt_version = getattr(settings, "graph_prompt_version", None) or PROMPT_VERSION
    model_id = getattr(settings, "graph_model", None) or "unconfigured"
    return schema_version, prompt_version, model_id


def enqueue_message_graph(conn, message_id, *, tenant_id, mailbox_id, settings) -> int:
    """Queue graph extraction for one persisted message's chunks.

    Returns the number of jobs created (0 when disabled, or when every chunk is
    already extracted for these exact inputs).
    """
    if settings is None or not getattr(settings, "graph_extraction_enabled", False):
        return 0

    from email_thread_rag.graph.repository import PostgresGraphStore

    schema_version, prompt_version, model_id = graph_identity(settings)
    store = PostgresGraphStore(conn)
    return store.enqueue_message(
        message_id,
        tenant_id=tenant_id, mailbox_id=mailbox_id,
        schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
    )
