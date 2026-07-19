"""The contextualization worker: fallback, retry, immutability, staleness.

All fakes. The autouse socket guard fails this module on any real connection.
"""

from __future__ import annotations

import json

import pytest

from email_thread_rag.context.fakes import FakeContextProvider, UnavailableContextProvider, context_json
from email_thread_rag.context.fingerprint import PROMPT_VERSION
from email_thread_rag.context.models import ChunkContextState
from email_thread_rag.context.store import InMemoryContextJobStore
from email_thread_rag.context.worker import ContextWorker
from email_thread_rag.rag.email_segmentation import build_embed_text
from email_thread_rag.rag.vector_index import HashingEncoder

TEXT = "The approved amount is $1200 for Acme Supplies."
PREFIX = "This chunk concerns the approved Acme Supplies budget."


@pytest.fixture
def store():
    return InMemoryContextJobStore()


@pytest.fixture
def chunk(store):
    state = ChunkContextState(
        chunk_db_id=1,
        chunk_id="msg-2-email-0",
        tenant_id="acme",
        mailbox_id="inbox",
        text=TEXT,
        subject="Re: Budget Review",
        sender="bob@corp.com",
        thread_id="thread-alpha",
        to=["alice@corp.com"],
    )
    store.add_chunk(state, embed_text=build_embed_text(TEXT, sender="bob@corp.com", subject="Re: Budget Review"))
    return state


def make_worker(store, provider, *, encoder=None):
    return ContextWorker(store, provider, encoder=encoder, prompt_version=PROMPT_VERSION)


def enqueue(store, state, provider):
    return store.enqueue(state, prompt_version=PROMPT_VERSION, model_id=provider.model_id)


# --- the happy path ------------------------------------------------------
def test_a_valid_prefix_is_written_and_the_job_retires(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    job = enqueue(store, chunk, provider)

    assert make_worker(store, provider).run_once() is True

    assert store.load_chunk_state(1).context_prefix == PREFIX
    assert store.load_chunk_state(1).context_method == "llm"
    assert store.get_job(job.id).status == "done"


def test_run_once_returns_false_on_an_empty_queue(store):
    assert make_worker(store, FakeContextProvider()).run_once() is False


def test_the_prefix_enters_embed_text_and_never_the_text(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    enqueue(store, chunk, provider)
    make_worker(store, provider).run_once()

    embed_text = store.embed_texts[1]
    assert PREFIX in embed_text
    assert TEXT in embed_text
    # The citable evidence is untouched by the model.
    assert store.load_chunk_state(1).text == TEXT
    assert PREFIX not in store.load_chunk_state(1).text


def test_embed_text_keeps_headers_prefix_and_text_in_canonical_order(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    enqueue(store, chunk, provider)
    make_worker(store, provider).run_once()

    embed_text = store.embed_texts[1]
    assert embed_text.index("From: bob@corp.com") < embed_text.index(PREFIX) < embed_text.index(TEXT)


def test_the_provider_receives_the_clean_text_and_metadata(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    enqueue(store, chunk, provider)
    make_worker(store, provider).run_once()

    sent = provider.calls[0]
    assert sent.text == TEXT
    assert sent.subject == "Re: Budget Review"
    assert sent.sender == "bob@corp.com"


def test_the_chunk_is_re_embedded_from_the_new_embed_text(store, chunk):
    encoder = HashingEncoder()
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    enqueue(store, chunk, provider)

    make_worker(store, provider, encoder=encoder).run_once()

    stored = store.embeddings[1]
    expected = list(encoder.encode([store.embed_texts[1]])[0])
    # The embedding must be of the *contextualized* embed_text, not the old one.
    assert stored == pytest.approx(expected)
    assert stored != pytest.approx(list(encoder.encode([TEXT])[0]))


# --- fallback ------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_output",
    [
        "not json",
        '{"summary": "wrong key"}',
        '{"context": ""}',
        '{"context": 42}',
        json.dumps({"context": "One. Two. Three. Four."}),
        json.dumps({"context": " ".join(["budget"] * 120)}),
        json.dumps({"context": "See [1] for details."}),
        None,
    ],
)
def test_invalid_output_falls_back_to_the_deterministic_embed_text(store, chunk, bad_output):
    provider = FakeContextProvider(responder=lambda ci: bad_output)
    job = enqueue(store, chunk, provider)

    make_worker(store, provider).run_once()

    state = store.load_chunk_state(1)
    assert state.context_prefix is None
    assert state.context_method == "deterministic"
    # Exactly the Stage-1 form: retrieval still works, there is just no prefix.
    assert store.embed_texts[1] == build_embed_text(
        TEXT, sender="bob@corp.com", to=["alice@corp.com"], subject="Re: Budget Review", thread_id="thread-alpha"
    )
    assert state.text == TEXT
    # Deterministic failure: retrying a temperature-0 call cannot help, so the
    # job is retired rather than spun.
    assert store.get_job(job.id).status == "done"


def test_a_fallback_is_not_retried_forever(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: "not json")
    enqueue(store, chunk, provider)
    worker = make_worker(store, provider)

    assert worker.drain(max_jobs=10) == 1
    assert len(provider.calls) == 1  # not a retry loop


# --- retries -------------------------------------------------------------
def test_a_provider_outage_leaves_the_job_pending_for_retry(store, chunk):
    provider = UnavailableContextProvider()
    job = enqueue(store, chunk, provider)

    make_worker(store, provider).run_once()

    retried = store.get_job(job.id)
    assert retried.status == "pending"
    assert "503" in retried.last_error
    # Nothing was written: an outage must not degrade the chunk.
    assert store.load_chunk_state(1).context_method is None


def test_a_retry_succeeds_once_the_provider_recovers(store, chunk):
    outage = UnavailableContextProvider()
    enqueue(store, chunk, outage)
    make_worker(store, outage).run_once()

    recovered = FakeContextProvider(responder=lambda ci: context_json(PREFIX), model_id=outage.model_id)
    make_worker(store, recovered).run_once()

    assert store.load_chunk_state(1).context_prefix == PREFIX


def test_a_persistently_failing_job_stops_retrying(store, chunk):
    provider = UnavailableContextProvider()
    job = enqueue(store, chunk, provider)
    worker = ContextWorker(store, provider, encoder=None, prompt_version=PROMPT_VERSION, max_attempts=3)

    worker.drain(max_jobs=20)

    assert store.get_job(job.id).status == "failed"


# --- staleness -----------------------------------------------------------
def test_a_stale_result_does_not_overwrite_a_changed_chunk(store, chunk):
    def slow_responder(ci):
        # The chunk is re-ingested while the model is "thinking".
        store.load_chunk_state(1).text = "Totally new authored text."
        return context_json(PREFIX)

    provider = FakeContextProvider(responder=slow_responder)
    enqueue(store, chunk, provider)

    make_worker(store, provider).run_once()

    state = store.load_chunk_state(1)
    assert state.context_prefix is None  # the stale prefix was discarded
    assert state.text == "Totally new authored text."


def test_no_duplicate_work_for_an_unchanged_chunk(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(PREFIX))
    enqueue(store, chunk, provider)
    make_worker(store, provider).run_once()

    # Re-persisting the same message enqueues nothing.
    assert enqueue(store, store.load_chunk_state(1), provider) is None
    assert make_worker(store, provider).run_once() is False
    assert len(provider.calls) == 1


def test_editing_a_chunk_re_contextualizes_it(store, chunk):
    provider = FakeContextProvider(responder=lambda ci: context_json(f"Concerns: {ci.text[:12]}"))
    enqueue(store, chunk, provider)
    make_worker(store, provider).run_once()
    first = store.load_chunk_state(1).context_prefix

    store.load_chunk_state(1).text = "An entirely rewritten body."
    assert enqueue(store, store.load_chunk_state(1), provider) is not None
    make_worker(store, provider).run_once()

    assert store.load_chunk_state(1).context_prefix != first


def test_a_deleted_chunk_does_not_crash_the_worker(store, chunk):
    provider = FakeContextProvider()
    job = enqueue(store, chunk, provider)
    del store.chunks[1]  # message deleted while the job waited

    assert make_worker(store, provider).run_once() is True
    assert store.get_job(job.id).status == "failed"
    assert provider.calls == []


# --- drain ---------------------------------------------------------------
def test_drain_processes_every_queued_job(store):
    provider = FakeContextProvider(responder=lambda ci: context_json(f"Concerns chunk {ci.chunk_id}."))
    for index in range(5):
        state = ChunkContextState(
            chunk_db_id=index + 1,
            chunk_id=f"c-{index}",
            tenant_id="acme",
            mailbox_id="inbox",
            text=f"Body number {index}.",
            subject="Budget",
        )
        store.add_chunk(state)
        enqueue(store, state, provider)

    assert make_worker(store, provider).drain() == 5
    assert all(store.load_chunk_state(i + 1).context_method == "llm" for i in range(5))
