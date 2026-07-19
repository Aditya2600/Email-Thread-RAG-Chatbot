"""The stale-job guard, in one function.

A job carries the fingerprint of the inputs that existed when it was enqueued.
Before a worker writes a prefix it recomputes the fingerprint from the chunk's
*current* row: if the two differ, the chunk changed underneath a slow LLM call
and the result is discarded. That is the entire protection against a stale job
overwriting newer chunk state, so it lives here alone and is used by the
enqueue path and the commit path both -- two implementations could drift, and a
drifted fingerprint means either permanent no-ops or silent overwrites.
"""

from __future__ import annotations

import hashlib
import json

from email_thread_rag.context.models import ContextInput

# Bump when the prompt changes in a way that should invalidate existing
# prefixes: it is part of the fingerprint, so bumping it re-enqueues every
# chunk rather than leaving a mix of old and new prefixes in the index.
PROMPT_VERSION = "ctx-2026-07-v1"


def fingerprint_of(context_input: ContextInput, *, prompt_version: str, model_id: str) -> str:
    """Hash the exact inputs the prefix depends on.

    Canonical JSON (sorted keys, no whitespace drift) so the same inputs hash
    identically across processes and Python versions -- an unstable hash would
    re-enqueue the whole corpus on every ingest.
    """
    payload = {
        "chunk_id": context_input.chunk_id,
        "text": context_input.text,
        "subject": context_input.subject,
        "sender": context_input.sender,
        "thread_id": context_input.thread_id,
        "parent_message_id": context_input.parent_message_id,
        "parent_subject": context_input.parent_subject,
        "prompt_version": prompt_version,
        "model_id": model_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
