"""The stale-job guard, in one function -- mirrors context.fingerprint.

A job carries the fingerprint of the inputs that existed when it was enqueued.
Before a worker writes graph rows it recomputes the fingerprint from the chunk's
*current* row: if the two differ, the chunk changed underneath a slow LLM call
and the result is discarded. That is the entire protection against a stale job
overwriting newer chunk state, so it lives here alone and is used by the enqueue
path and the commit path both.

The hash folds in the extraction schema version *and* the prompt version *and*
the model id, exactly as the spec requires: bumping any of the three re-enqueues
the corpus rather than leaving a mix of old and new graph rows.
"""

from __future__ import annotations

import hashlib
import json

from email_thread_rag.graph.models import ExtractionInput

# The shape of what we extract (entity types, predicates, evidence rules). Bump
# when that shape changes in a way that should invalidate existing graph rows.
SCHEMA_VERSION = "graph-2026-07-v1"
# The extraction prompt. Bump when the prompt changes materially.
PROMPT_VERSION = "graph-prompt-2026-07-v1"


def extraction_hash_of(
    extraction_input: ExtractionInput, *, schema_version: str, prompt_version: str, model_id: str
) -> str:
    """Hash the exact inputs the extraction depends on.

    Canonical JSON (sorted keys, no whitespace drift) so the same inputs hash
    identically across processes and Python versions -- an unstable hash would
    re-enqueue the whole corpus on every ingest.
    """
    payload = {
        "chunk_id": extraction_input.chunk_id,
        "text": extraction_input.text,
        "subject": extraction_input.subject,
        "sender": extraction_input.sender,
        "thread_id": extraction_input.thread_id,
        "recipients": list(extraction_input.recipients),
        "cc": list(extraction_input.cc),
        "in_reply_to": extraction_input.in_reply_to,
        "parent_sender": extraction_input.parent_sender,
        "schema_version": schema_version,
        "prompt_version": prompt_version,
        "model_id": model_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
