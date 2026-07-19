"""Contract every ContextJobStore must satisfy.

Subclassed by tests/test_context_jobs.py (in-memory) and
tests/integration/test_context_paradedb.py (Postgres). One contract, two
implementations: the fake the fast suite relies on cannot drift from the real
store without this file failing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from email_thread_rag.context.fingerprint import fingerprint_of
from email_thread_rag.context.models import ChunkContextState

PROMPT_VERSION = "test-prompt-v1"
MODEL_ID = "fake-context-model"


class ContextStoreContract:
    """Subclasses provide `store` and `make_chunk` fixtures."""

    # --- enqueue ---------------------------------------------------------
    def test_enqueue_creates_a_pending_job(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="The budget draft is attached.")
        job = store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        assert job is not None
        assert job.status == "pending"
        assert job.chunk_id == "c-1"
        assert job.attempts == 0

    def test_enqueue_is_idempotent_for_unchanged_inputs(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="Unchanged text.")
        first = store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        second = store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        # Re-persisting an unchanged message must not create duplicate work.
        assert first is not None
        assert second is None

    def test_changed_text_enqueues_a_new_job(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="Original text.")
        first = store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        changed = make_chunk(chunk_id="c-1", text="Edited text.", replace=True)
        second = store.enqueue(changed, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        assert second is not None
        assert second.id != first.id
        assert second.context_input_hash != first.context_input_hash

    def test_a_new_prompt_version_enqueues_new_work(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="Same text.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        rerun = store.enqueue(state, prompt_version="test-prompt-v2", model_id=MODEL_ID)

        # Bumping the prompt version must re-contextualize, not silently no-op.
        assert rerun is not None

    def test_a_new_model_enqueues_new_work(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="Same text.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        rerun = store.enqueue(state, prompt_version=PROMPT_VERSION, model_id="other-model")

        assert rerun is not None

    # --- claiming --------------------------------------------------------
    def test_claim_leases_a_job_and_counts_the_attempt(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text="Claim me.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        claimed = store.claim_job(owner="worker-a")
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.attempts == 1
        assert claimed.lease_owner == "worker-a"
        assert claimed.leased_until is not None

    def test_a_leased_job_is_not_claimed_twice(self, store, make_chunk):
        store.enqueue(make_chunk(chunk_id="c-1", text="Only once."), prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        assert store.claim_job(owner="worker-a") is not None
        assert store.claim_job(owner="worker-b") is None

    def test_claim_returns_none_on_an_empty_queue(self, store):
        assert store.claim_job(owner="worker-a") is None

    # --- completion ------------------------------------------------------
    def test_commit_writes_prefix_and_retires_the_job(self, store, make_chunk, read_back):
        state = make_chunk(chunk_id="c-1", text="Budget approved.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        job = store.claim_job(owner="worker-a")

        written = store.commit_context(
            job,
            prefix="This chunk concerns the approved budget.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="Subject: Budget\n\nThis chunk concerns the approved budget.\n\nBudget approved.",
            embedding=None,
        )

        assert written is True
        assert store.get_job(job.id).status == "done"

        fresh = read_back(state)
        assert fresh.context_prefix == "This chunk concerns the approved budget."
        assert fresh.context_method == "llm"
        assert fresh.context_input_hash == job.context_input_hash
        # The evidence itself is untouched.
        assert fresh.text == "Budget approved."

    def test_a_committed_chunk_is_not_re_enqueued(self, store, make_chunk, read_back):
        state = make_chunk(chunk_id="c-1", text="Done already.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        job = store.claim_job(owner="worker-a")
        store.commit_context(
            job,
            prefix="A prefix.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="A prefix.\n\nDone already.",
            embedding=None,
        )

        again = store.enqueue(read_back(state), prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        assert again is None

    # --- the stale-job guard ---------------------------------------------
    def test_a_stale_job_cannot_overwrite_a_changed_chunk(self, store, make_chunk, read_back, mutate_chunk):
        state = make_chunk(chunk_id="c-1", text="Original text.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        job = store.claim_job(owner="worker-a")

        # The chunk is re-ingested while the (slow) LLM call is in flight.
        mutate_chunk(state, text="Completely different text.")

        written = store.commit_context(
            job,
            prefix="This chunk concerns the ORIGINAL text.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="This chunk concerns the ORIGINAL text.\n\nOriginal text.",
            embedding=None,
        )

        assert written is False
        fresh = read_back(state)
        # The stale prefix must not land, and the new text must survive intact.
        assert fresh.context_prefix is None
        assert fresh.text == "Completely different text."
        assert store.get_job(job.id).status == "done"

    def test_the_replacement_job_still_succeeds_after_a_stale_one(self, store, make_chunk, read_back, mutate_chunk):
        state = make_chunk(chunk_id="c-1", text="Original text.")
        store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        stale = store.claim_job(owner="worker-a")
        mutate_chunk(state, text="New text.")

        store.commit_context(
            stale,
            prefix="Stale prefix.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="Stale prefix.\n\nOriginal text.",
            embedding=None,
        )

        # Re-enqueue for the new text and process it.
        fresh_job = store.enqueue(read_back(state), prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        assert fresh_job is not None
        claimed = store.claim_job(owner="worker-b")
        written = store.commit_context(
            claimed,
            prefix="Correct prefix.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="Correct prefix.\n\nNew text.",
            embedding=None,
        )

        assert written is True
        assert read_back(state).context_prefix == "Correct prefix."

    # --- retries ---------------------------------------------------------
    def test_a_failed_job_returns_to_pending_for_retry(self, store, make_chunk):
        store.enqueue(make_chunk(chunk_id="c-1", text="Flaky."), prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        job = store.claim_job(owner="worker-a")

        store.fail_job(job.id, "context provider returned HTTP 503", max_attempts=3)

        retried = store.get_job(job.id)
        assert retried.status == "pending"
        assert retried.last_error == "context provider returned HTTP 503"
        # And it is claimable again.
        assert store.claim_job(owner="worker-b") is not None

    def test_a_job_fails_permanently_once_attempts_are_spent(self, store, make_chunk):
        store.enqueue(make_chunk(chunk_id="c-1", text="Doomed."), prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        for _ in range(3):
            job = store.claim_job(owner="worker-a")
            assert job is not None
            store.fail_job(job.id, "still down", max_attempts=3)

        assert store.get_job(job.id).status == "failed"
        assert store.claim_job(owner="worker-a") is None

    # --- backfill scan ---------------------------------------------------
    def test_chunks_needing_context_excludes_contextualized_chunks(self, store, make_chunk, read_back):
        a = make_chunk(chunk_id="c-1", text="First.")
        b = make_chunk(chunk_id="c-2", text="Second.")

        pending = store.chunks_needing_context(tenant_id=a.tenant_id, mailbox_id=a.mailbox_id)
        assert {s.chunk_id for s in pending} == {"c-1", "c-2"}

        store.enqueue(a, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        job = store.claim_job(owner="worker-a")
        store.commit_context(
            job,
            prefix="Prefix for the first.",
            method="llm",
            context_version=PROMPT_VERSION,
            model_id=MODEL_ID,
            prompt_version=PROMPT_VERSION,
            embed_text="Prefix for the first.\n\nFirst.",
            embedding=None,
        )

        remaining = store.chunks_needing_context(tenant_id=b.tenant_id, mailbox_id=b.mailbox_id)
        assert {s.chunk_id for s in remaining} == {"c-2"}

    def test_the_backfill_scan_pages_forward(self, store, make_chunk):
        first = make_chunk(chunk_id="c-1", text="First.")
        make_chunk(chunk_id="c-2", text="Second.")

        page = store.chunks_needing_context(
            tenant_id=first.tenant_id, mailbox_id=first.mailbox_id, limit=1
        )
        assert len(page) == 1
        nxt = store.chunks_needing_context(
            tenant_id=first.tenant_id, mailbox_id=first.mailbox_id, limit=1, after_id=page[0].chunk_db_id
        )
        assert len(nxt) == 1
        assert nxt[0].chunk_db_id > page[0].chunk_db_id

    # --- isolation -------------------------------------------------------
    def test_the_backfill_scan_is_tenant_scoped(self, store, make_chunk):
        make_chunk(chunk_id="c-1", text="Acme mail.", tenant_id="acme")
        make_chunk(chunk_id="c-2", text="Globex mail.", tenant_id="globex")

        acme = store.chunks_needing_context(tenant_id="acme", mailbox_id="inbox")
        assert {s.chunk_id for s in acme} == {"c-1"}

    def test_the_same_chunk_id_in_two_tenants_gets_separate_jobs(self, store, make_chunk):
        acme = make_chunk(chunk_id="shared-id", text="Acme text.", tenant_id="acme")
        globex = make_chunk(chunk_id="shared-id", text="Globex text.", tenant_id="globex")

        first = store.enqueue(acme, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        second = store.enqueue(globex, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)

        assert first is not None
        assert second is not None
        assert first.id != second.id
        assert first.tenant_id == "acme"
        assert second.tenant_id == "globex"
