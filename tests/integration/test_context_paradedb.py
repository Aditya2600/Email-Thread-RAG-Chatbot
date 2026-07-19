"""Stage-4 against the real ParadeDB container.

Three jobs here:
1. Run the *same* ContextStoreContract the in-memory store passes, against
   PostgresContextJobStore -- so the fast suite's store cannot drift.
2. Prove the schema enforces what the design claims (job uniqueness, the
   context_method check, cascade on chunk delete).
3. Smoke-test the whole path: persisted clean chunk -> fake LLM prefix ->
   rebuilt embed_text + new embedding -> hybrid retrieval -> original `text`
   citation.

The LLM is always a fake. No model is downloaded and no remote endpoint is
called.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg.rows import dict_row

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.context.backfill import backfill_context_jobs
from email_thread_rag.context.fakes import ExplodingContextProvider, FakeContextProvider, context_json
from email_thread_rag.context.fingerprint import PROMPT_VERSION
from email_thread_rag.context.models import ChunkContextState
from email_thread_rag.context.repository import PostgresContextJobStore
from email_thread_rag.context.worker import ContextWorker
from email_thread_rag.rag.chunking import chunk_email
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.paradedb.retrieval import HybridRetriever, LexicalRetriever, RetrievalFilters
from email_thread_rag.rag.vector_index import HashingEncoder
from context_store_contract import ContextStoreContract

from datetime import datetime, timezone

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=384)
TENANT = "acme"
MAILBOX = "inbox"
# A term that appears in NO authored text, so retrieving by it proves the hit
# came through the contextual prefix.
UNIQUE_PREFIX_TERM = "zorbulon"


@pytest.fixture
def autocommit_conn(migrated_database_url):
    """Autocommit + explicit transaction blocks: the same discipline the worker
    uses, so no transaction is held across an LLM call."""
    conn = psycopg.connect(migrated_database_url, row_factory=dict_row, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_tables(autocommit_conn):
    autocommit_conn.execute("TRUNCATE chunk_context_jobs, email_chunks, email_messages RESTART IDENTITY CASCADE")
    yield


def context_settings(**overrides) -> Settings:
    kwargs = dict(
        context_enabled=True,
        context_base_url="http://fake.invalid/v1",
        context_model="fake-context-model",
        embedding_dim=384,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def make_email(
    *,
    doc_id="msg-2",
    message_id="<msg-2@example.com>",
    subject="Re: Budget Review",
    body="Final budget attached. The approved amount is $1200 for Acme Supplies.",
    thread_id="thread-alpha",
    in_reply_to=None,
) -> EmailRecord:
    return EmailRecord(
        doc_id=doc_id,
        message_id=message_id,
        thread_id=thread_id,
        date=datetime(2024, 1, 7, tzinfo=timezone.utc),
        sender="bob@corp.com",
        to=["alice@corp.com"],
        subject=subject,
        body_text=body,
        in_reply_to=in_reply_to,
        source_path="/tmp/msg-2.json",
        source_type="fixture",
    )


def persist(conn, email: EmailRecord, *, settings=None, tenant_id=TENANT, mailbox_id=MAILBOX) -> dict:
    chunks = chunk_email(email)
    return persist_corpus_to_paradedb(
        conn,
        [email],
        chunks,
        tenant_id=tenant_id,
        mailbox_id=mailbox_id,
        encoder=ENCODER,
        embedding_dim=384,
        settings=settings,
    )


def read_chunk(conn, chunk_id, *, tenant_id=TENANT):
    return conn.execute(
        "SELECT * FROM email_chunks WHERE chunk_id = %s AND tenant_id = %s", (chunk_id, tenant_id)
    ).fetchone()


# =========================================================================
# 1. The shared contract, against Postgres
# =========================================================================
class TestPostgresContextJobStore(ContextStoreContract):
    @pytest.fixture
    def store(self, autocommit_conn):
        return PostgresContextJobStore(autocommit_conn, embedding_dim=384)

    @pytest.fixture
    def make_chunk(self, autocommit_conn):
        counter = {"index": 0}

        def _make(*, chunk_id, text, tenant_id=TENANT, mailbox_id=MAILBOX, replace=False):
            if replace:
                autocommit_conn.execute(
                    "UPDATE email_chunks SET text = %s WHERE chunk_id = %s AND tenant_id = %s",
                    (text, chunk_id, tenant_id),
                )
            else:
                message_id = f"<{chunk_id}@example.com>"
                autocommit_conn.execute(
                    """
                    INSERT INTO email_messages (tenant_id, mailbox_id, message_id, thread_id, authored_text)
                    VALUES (%s, %s, %s, 'thread-alpha', %s)
                    ON CONFLICT (tenant_id, mailbox_id, message_id) DO NOTHING
                    """,
                    (tenant_id, mailbox_id, message_id, text),
                )
                autocommit_conn.execute(
                    """
                    INSERT INTO email_chunks (
                        chunk_id, tenant_id, mailbox_id, message_id, thread_id, chunk_index,
                        chunk_kind, sender, subject, text, embed_text, content_hash, metadata
                    ) VALUES (%s, %s, %s, %s, 'thread-alpha', %s, 'email', 'alice@corp.com',
                              'Budget Review', %s, %s, 'hash', '{}'::jsonb)
                    """,
                    (chunk_id, tenant_id, mailbox_id, message_id, counter["index"], text, text),
                )
                counter["index"] += 1
            row = autocommit_conn.execute(
                "SELECT id FROM email_chunks WHERE chunk_id = %s AND tenant_id = %s", (chunk_id, tenant_id)
            ).fetchone()
            state = PostgresContextJobStore(autocommit_conn).load_chunk_state(row["id"])
            return state

        return _make

    @pytest.fixture
    def read_back(self, store):
        return lambda state: store.load_chunk_state(state.chunk_db_id)

    @pytest.fixture
    def mutate_chunk(self, autocommit_conn):
        def _mutate(state, *, text):
            autocommit_conn.execute(
                "UPDATE email_chunks SET text = %s WHERE id = %s", (text, state.chunk_db_id)
            )

        return _mutate


# =========================================================================
# 2. Schema
# =========================================================================
def test_the_context_columns_exist_with_the_expected_types(autocommit_conn):
    rows = autocommit_conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = 'email_chunks' AND column_name LIKE 'context%'"
    ).fetchall()
    columns = {row["column_name"]: row["data_type"] for row in rows}
    assert columns["context_prefix"] == "text"
    assert columns["context_method"] == "text"
    assert columns["context_version"] == "text"
    assert columns["context_input_hash"] == "text"
    assert columns["context_model"] == "text"
    assert columns["context_updated_at"] == "timestamp with time zone"


def test_context_method_is_constrained_at_the_database_level(autocommit_conn, db_chunk):
    with pytest.raises(psycopg.errors.CheckViolation):
        autocommit_conn.execute(
            "UPDATE email_chunks SET context_method = 'wishful-thinking' WHERE id = %s",
            (db_chunk.chunk_db_id,),
        )


@pytest.mark.parametrize("method", ["none", "deterministic", "llm"])
def test_the_three_valid_context_methods_are_accepted(autocommit_conn, db_chunk, method):
    autocommit_conn.execute(
        "UPDATE email_chunks SET context_method = %s WHERE id = %s", (method, db_chunk.chunk_db_id)
    )


def test_duplicate_jobs_are_rejected_by_the_database_not_just_python(autocommit_conn, db_chunk):
    for _ in range(2):
        autocommit_conn.execute(
            "INSERT INTO chunk_context_jobs (chunk_db_id, tenant_id, mailbox_id, chunk_id, context_input_hash) "
            "VALUES (%s, %s, %s, %s, 'same-hash') ON CONFLICT DO NOTHING",
            (db_chunk.chunk_db_id, TENANT, MAILBOX, db_chunk.chunk_id),
        )
    count = autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"]
    assert count == 1

    with pytest.raises(psycopg.errors.UniqueViolation):
        autocommit_conn.execute(
            "INSERT INTO chunk_context_jobs (chunk_db_id, tenant_id, mailbox_id, chunk_id, context_input_hash) "
            "VALUES (%s, %s, %s, %s, 'same-hash')",
            (db_chunk.chunk_db_id, TENANT, MAILBOX, db_chunk.chunk_id),
        )


def test_deleting_a_chunk_cascades_to_its_jobs(autocommit_conn, db_chunk):
    store = PostgresContextJobStore(autocommit_conn)
    store.enqueue(db_chunk, prompt_version=PROMPT_VERSION, model_id="m")
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 1

    autocommit_conn.execute("DELETE FROM email_chunks WHERE id = %s", (db_chunk.chunk_db_id,))

    # No orphan jobs pointing at chunks that no longer exist.
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 0


@pytest.fixture
def db_chunk(autocommit_conn):
    email = make_email()
    persist(autocommit_conn, email)
    row = autocommit_conn.execute("SELECT id FROM email_chunks ORDER BY id LIMIT 1").fetchone()
    return PostgresContextJobStore(autocommit_conn).load_chunk_state(row["id"])


# =========================================================================
# 3. Enqueue-on-ingest
# =========================================================================
def test_ingestion_queues_nothing_when_contextualization_is_disabled(autocommit_conn):
    stats = persist(autocommit_conn, make_email(), settings=None)

    assert stats["context_jobs"] == 0
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 0


def test_ingestion_queues_work_when_enabled(autocommit_conn):
    stats = persist(autocommit_conn, make_email(), settings=context_settings())

    assert stats["context_jobs"] == 1
    job = autocommit_conn.execute("SELECT * FROM chunk_context_jobs").fetchone()
    assert job["status"] == "pending"
    assert job["tenant_id"] == TENANT


def test_reprocessing_an_unchanged_message_creates_no_duplicate_work(autocommit_conn):
    settings = context_settings()
    email = make_email()
    first = persist(autocommit_conn, email, settings=settings)
    second = persist(autocommit_conn, make_email(), settings=settings)

    assert first["context_jobs"] == 1
    assert second["context_jobs"] == 0
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 1


def test_editing_a_message_queues_fresh_work(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    edited = persist(autocommit_conn, make_email(body="A completely rewritten body."), settings=settings)

    assert edited["context_jobs"] == 1
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 2


def test_the_parent_subject_is_supplied_only_when_the_parent_is_local(autocommit_conn):
    parent = make_email(doc_id="msg-1", message_id="<msg-1@example.com>", subject="Budget Review",
                        body="Please review the draft budget.")
    reply = make_email(in_reply_to="<msg-1@example.com>")

    persist(autocommit_conn, reply)
    store = PostgresContextJobStore(autocommit_conn)
    reply_chunk = store.load_chunk_state(
        autocommit_conn.execute(
            "SELECT id FROM email_chunks WHERE message_id = %s", (reply.message_id,)
        ).fetchone()["id"]
    )
    # Parent not yet ingested: no fabricated parent context.
    assert reply_chunk.parent_subject is None

    persist(autocommit_conn, parent)
    reply_chunk = store.load_chunk_state(reply_chunk.chunk_db_id)
    assert reply_chunk.parent_subject == "Budget Review"


# =========================================================================
# 4. The smoke test
# =========================================================================
def test_smoke_persisted_chunk_to_prefix_to_retrieval_to_clean_citation(autocommit_conn):
    settings = context_settings()

    # 1. Persist a clean canonical chunk.
    email = make_email()
    stats = persist(autocommit_conn, email, settings=settings)
    assert stats["chunks"] == 1 and stats["context_jobs"] == 1

    before = read_chunk(autocommit_conn, "msg-2-email-0")
    original_text = before["text"]
    original_start, original_end = before["source_start"], before["source_end"]
    original_embedding = before["embedding"]
    assert before["context_prefix"] is None

    # Baseline: BM25 matches the unique term against nothing at all yet. Without
    # this the post-contextualization assertion would be vacuous -- the dense
    # branch returns the only chunk in the table for any query whatsoever.
    lexical = LexicalRetriever(autocommit_conn)
    filters = RetrievalFilters(tenant_id=TENANT, mailbox_id=MAILBOX)
    assert lexical.search(UNIQUE_PREFIX_TERM, filters, 5) == []

    # 2. A fake LLM produces a prefix carrying a term found nowhere in the mail.
    prefix = f"This chunk concerns the {UNIQUE_PREFIX_TERM} procurement budget for Acme Supplies."
    provider = FakeContextProvider(responder=lambda ci: context_json(prefix), model_id="fake-context-model")
    store = PostgresContextJobStore(autocommit_conn, embedding_dim=384)
    worker = ContextWorker(store, provider, encoder=ENCODER, prompt_version=PROMPT_VERSION)

    assert worker.run_once() is True
    assert worker.run_once() is False  # queue drained

    # 3. embed_text was rebuilt and the chunk re-embedded.
    after = read_chunk(autocommit_conn, "msg-2-email-0")
    assert after["context_prefix"] == prefix
    assert after["context_method"] == "llm"
    assert after["context_version"] == PROMPT_VERSION
    assert after["context_model"] == "fake-context-model"
    assert after["context_updated_at"] is not None
    assert prefix in after["embed_text"]
    assert original_text in after["embed_text"]
    assert after["embedding"] != original_embedding  # re-embedded, not stale

    # 4. The evidence is immutable.
    assert after["text"] == original_text
    assert after["source_start"] == original_start
    assert after["source_end"] == original_end
    assert UNIQUE_PREFIX_TERM not in after["text"]

    # 5. The BM25 index now matches the term that exists ONLY in the prefix.
    #    Same query, same data, opposite result: the index really is searching
    #    the rebuilt embed_text.
    lexical_hits = lexical.search(UNIQUE_PREFIX_TERM, filters, 5)
    assert [hit.chunk_id for hit in lexical_hits] == ["msg-2-email-0"]

    retriever = HybridRetriever(autocommit_conn, settings, encoder=ENCODER)
    hits = retriever.search(f"{UNIQUE_PREFIX_TERM} procurement", filters, top_k=5)
    assert [hit.chunk_id for hit in hits] == ["msg-2-email-0"]

    # 6. The citation evidence handed back is clean authored text: no headers,
    #    no model words.
    evidence = hits[0].text
    assert evidence == original_text
    assert UNIQUE_PREFIX_TERM not in evidence
    assert "This chunk concerns" not in evidence
    assert not evidence.startswith("Subject:") and not evidence.startswith("From:")
    assert hits[0].source_start == original_start
    assert hits[0].source_end == original_end


def test_a_chunk_stays_retrievable_by_its_own_words_after_contextualization(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    provider = FakeContextProvider(
        responder=lambda ci: context_json(f"Concerns the {UNIQUE_PREFIX_TERM} budget."),
        model_id="fake-context-model",
    )
    ContextWorker(
        PostgresContextJobStore(autocommit_conn, embedding_dim=384),
        provider,
        encoder=ENCODER,
        prompt_version=PROMPT_VERSION,
    ).drain()

    retriever = HybridRetriever(autocommit_conn, settings, encoder=ENCODER)
    hits = retriever.search(
        "approved amount Acme Supplies", RetrievalFilters(tenant_id=TENANT, mailbox_id=MAILBOX), top_k=5
    )
    # Adding a prefix must not cost the chunk its original findability.
    assert "msg-2-email-0" in [hit.chunk_id for hit in hits]


def test_a_fallback_chunk_is_still_retrievable(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    # Malformed output every time.
    provider = FakeContextProvider(responder=lambda ci: "}} not json {{", model_id="fake-context-model")
    ContextWorker(
        PostgresContextJobStore(autocommit_conn, embedding_dim=384),
        provider,
        encoder=ENCODER,
        prompt_version=PROMPT_VERSION,
    ).drain()

    row = read_chunk(autocommit_conn, "msg-2-email-0")
    assert row["context_method"] == "deterministic"
    assert row["context_prefix"] is None

    retriever = HybridRetriever(autocommit_conn, settings, encoder=ENCODER)
    hits = retriever.search(
        "approved amount Acme Supplies", RetrievalFilters(tenant_id=TENANT, mailbox_id=MAILBOX), top_k=5
    )
    assert "msg-2-email-0" in [hit.chunk_id for hit in hits]


# =========================================================================
# 5. Staleness against a real re-ingest
# =========================================================================
def test_a_re_ingest_during_the_llm_call_defeats_the_stale_job(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    store = PostgresContextJobStore(autocommit_conn, embedding_dim=384)

    def responder(context_input):
        # The message is re-ingested with new text while the model "thinks".
        persist(autocommit_conn, make_email(body="An entirely new authored body."), settings=settings)
        return context_json("This chunk concerns the OLD body.")

    provider = FakeContextProvider(responder=responder, model_id="fake-context-model")
    ContextWorker(store, provider, encoder=ENCODER, prompt_version=PROMPT_VERSION).run_once()

    row = read_chunk(autocommit_conn, "msg-2-email-0")
    # The stale prefix must not land on the new text.
    assert row["context_prefix"] is None
    assert "entirely new authored body" in row["text"]

    # The replacement job (queued by the re-ingest) still does the right thing.
    good = FakeContextProvider(
        responder=lambda ci: context_json("This chunk concerns the new body."), model_id="fake-context-model"
    )
    ContextWorker(store, good, encoder=ENCODER, prompt_version=PROMPT_VERSION).drain()
    assert read_chunk(autocommit_conn, "msg-2-email-0")["context_prefix"] == (
        "This chunk concerns the new body."
    )


# =========================================================================
# 6. Backfill
# =========================================================================
def test_backfill_queues_pre_existing_chunks(autocommit_conn):
    # Ingested before Stage 4 was switched on: no jobs.
    for index in range(3):
        persist(
            autocommit_conn,
            make_email(doc_id=f"msg-{index}", message_id=f"<msg-{index}@example.com>", body=f"Body {index}."),
            settings=None,
        )
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 0

    queued = backfill_context_jobs(
        autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=context_settings()
    )
    assert queued == 3


def test_backfill_is_idempotent(autocommit_conn):
    persist(autocommit_conn, make_email(), settings=None)
    settings = context_settings()

    first = backfill_context_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings)
    second = backfill_context_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings)

    assert first == 1
    assert second == 0  # re-running queues nothing
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 1


def test_backfill_is_resumable(autocommit_conn):
    for index in range(5):
        persist(
            autocommit_conn,
            make_email(doc_id=f"msg-{index}", message_id=f"<msg-{index}@example.com>", body=f"Body {index}."),
            settings=None,
        )
    settings = context_settings()

    # An interrupted run: stop after two chunks.
    partial = backfill_context_jobs(
        autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings, batch_size=1, max_chunks=2
    )
    assert partial == 2

    # Resuming picks up the rest and does not redo the first two.
    rest = backfill_context_jobs(
        autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings, batch_size=2
    )
    assert rest == 3
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 5


def test_backfill_skips_already_contextualized_chunks(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    provider = FakeContextProvider(
        responder=lambda ci: context_json("Concerns the budget."), model_id="fake-context-model"
    )
    ContextWorker(
        PostgresContextJobStore(autocommit_conn, embedding_dim=384),
        provider,
        encoder=ENCODER,
        prompt_version=PROMPT_VERSION,
    ).drain()

    assert backfill_context_jobs(
        autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings
    ) == 0


def test_backfill_is_tenant_scoped(autocommit_conn):
    persist(autocommit_conn, make_email(), settings=None, tenant_id="acme")
    persist(autocommit_conn, make_email(), settings=None, tenant_id="globex")

    queued = backfill_context_jobs(
        autocommit_conn, tenant_id="acme", mailbox_id=MAILBOX, settings=context_settings()
    )

    assert queued == 1
    rows = autocommit_conn.execute("SELECT DISTINCT tenant_id FROM chunk_context_jobs").fetchall()
    assert [row["tenant_id"] for row in rows] == ["acme"]


# =========================================================================
# 7. Tenant isolation through retrieval
# =========================================================================
def test_one_tenants_prefix_never_surfaces_for_another(autocommit_conn):
    settings = context_settings()
    # Distinct bodies so a hit is attributable to a tenant by its text alone.
    persist(autocommit_conn, make_email(body="Acme authored body."), settings=settings, tenant_id="acme")
    persist(autocommit_conn, make_email(body="Globex authored body."), settings=settings, tenant_id="globex")

    # Only Acme's chunk gets the distinctive prefix: drop Globex's job first.
    autocommit_conn.execute("DELETE FROM chunk_context_jobs WHERE tenant_id <> 'acme'")
    prefix = f"This chunk concerns the {UNIQUE_PREFIX_TERM} programme."
    ContextWorker(
        PostgresContextJobStore(autocommit_conn, embedding_dim=384),
        FakeContextProvider(responder=lambda ci: context_json(prefix), model_id="fake-context-model"),
        encoder=ENCODER,
        prompt_version=PROMPT_VERSION,
    ).drain()

    assert read_chunk(autocommit_conn, "msg-2-email-0", tenant_id="acme")["context_prefix"] == prefix
    assert read_chunk(autocommit_conn, "msg-2-email-0", tenant_id="globex")["context_prefix"] is None

    retriever = HybridRetriever(autocommit_conn, settings, encoder=ENCODER)
    acme_hits = retriever.search(
        UNIQUE_PREFIX_TERM, RetrievalFilters(tenant_id="acme", mailbox_id=MAILBOX), top_k=5
    )
    globex_hits = retriever.search(
        UNIQUE_PREFIX_TERM, RetrievalFilters(tenant_id="globex", mailbox_id=MAILBOX), top_k=5
    )

    # Acme finds its chunk through the prefix term.
    assert [hit.chunk_id for hit in acme_hits] == ["msg-2-email-0"]
    assert acme_hits[0].text == "Acme authored body."

    # Globex's dense branch still returns its own nearest chunk -- that is
    # expected. What must never happen is Globex seeing Acme's row or Acme's
    # prefix, so assert on the evidence, not on emptiness.
    for hit in globex_hits:
        assert hit.text == "Globex authored body."
        assert UNIQUE_PREFIX_TERM not in hit.text


def test_a_worker_only_writes_the_tenant_its_job_names(autocommit_conn):
    settings = context_settings()
    persist(autocommit_conn, make_email(), settings=settings, tenant_id="acme")
    persist(autocommit_conn, make_email(), settings=settings, tenant_id="globex")

    ContextWorker(
        PostgresContextJobStore(autocommit_conn, embedding_dim=384),
        FakeContextProvider(
            responder=lambda ci: context_json("Concerns the budget."), model_id="fake-context-model"
        ),
        encoder=ENCODER,
        prompt_version=PROMPT_VERSION,
    ).drain()

    for tenant in ("acme", "globex"):
        row = read_chunk(autocommit_conn, "msg-2-email-0", tenant_id=tenant)
        # Each tenant's chunk is contextualized from its own job only.
        assert row["context_method"] == "llm"
        assert row["tenant_id"] == tenant


# =========================================================================
# 8. Gmail ingestion enqueues without calling the provider
# =========================================================================
def test_gmail_ingestion_enqueues_context_without_calling_the_provider(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    settings = context_settings()
    sink = ParadeDBChunkSink(
        autocommit_conn,
        tenant_id=TENANT,
        mailbox_id=MAILBOX,
        encoder=ENCODER,
        embedding_dim=384,
        settings=settings,
    )
    email = make_email()
    email.source_type = "gmail"

    # No provider is passed to the sink at all; if ingestion tried to call one,
    # there would be nothing there. The job must still be queued.
    persisted = sink.persist(email)

    assert persisted == 1
    job = autocommit_conn.execute("SELECT * FROM chunk_context_jobs").fetchone()
    assert job["status"] == "pending"
    assert job["tenant_id"] == TENANT

    # And the chunk is fully retrievable before any LLM has run.
    row = read_chunk(autocommit_conn, "msg-2-email-0")
    assert row["context_method"] is None
    assert row["embed_text"]


def test_the_gmail_sink_queues_nothing_when_contextualization_is_off(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    sink = ParadeDBChunkSink(
        autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, encoder=ENCODER, embedding_dim=384
    )
    sink.persist(make_email())

    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 0


def test_deleting_a_gmail_message_removes_its_context_jobs(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    sink = ParadeDBChunkSink(
        autocommit_conn,
        tenant_id=TENANT,
        mailbox_id=MAILBOX,
        encoder=ENCODER,
        embedding_dim=384,
        settings=context_settings(),
    )
    email = make_email()
    sink.persist(email)
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 1

    sink.delete_message(email.message_id)

    # Deleted mail leaves no queued work behind to contextualize a ghost.
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_context_jobs").fetchone()["n"] == 0
