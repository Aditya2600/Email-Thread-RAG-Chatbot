"""Enqueue-after-persistence, for both the local corpus and Gmail paths.

Both ingestion paths call ``enqueue_message_context`` immediately after the
canonical persistence transaction commits. It is a no-op unless
contextualization is explicitly enabled, so the default configuration queues
nothing and imports nothing.

Enqueuing is all these paths do: the provider is never called here. Ingestion
must not block on an LLM, and a webhook must not call one at all.
"""

from __future__ import annotations

from typing import Optional

from email_thread_rag.context.fingerprint import PROMPT_VERSION


def context_identity(settings) -> tuple[str, str]:
    """The (prompt_version, model_id) pair the fingerprint is bound to.

    Read from settings rather than from a constructed provider: enqueuing must
    not build an HTTP client, and these are the same values the provider would
    report.
    """
    prompt_version = getattr(settings, "context_prompt_version", None) or PROMPT_VERSION
    model_id = getattr(settings, "context_model", None) or "unconfigured"
    return prompt_version, model_id


def enqueue_message_context(
    conn,
    message_id: str,
    *,
    tenant_id: str,
    mailbox_id: str,
    settings,
    embedding_dim: int = 768,
) -> int:
    """Queue contextualization for one persisted message's chunks.

    Returns the number of jobs created (0 when disabled, or when every chunk is
    already contextualized for these exact inputs).
    """
    if settings is None or not getattr(settings, "context_enabled", False):
        return 0

    # Local import: the disabled path must not pull in psycopg-backed modules.
    from email_thread_rag.context.repository import PostgresContextJobStore

    prompt_version, model_id = context_identity(settings)
    store = PostgresContextJobStore(conn, embedding_dim=embedding_dim)
    return store.enqueue_message(
        message_id,
        tenant_id=tenant_id,
        mailbox_id=mailbox_id,
        prompt_version=prompt_version,
        model_id=model_id,
    )
