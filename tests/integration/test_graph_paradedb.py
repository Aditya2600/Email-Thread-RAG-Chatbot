"""Stage-5 against the real ParadeDB container.

Four jobs here:
1. Run the *same* GraphStoreContract the in-memory store passes, against
   PostgresGraphStore -- so the fast suite's store cannot drift.
2. Prove the schema enforces what the design claims (job uniqueness, the
   evidence_kind CHECK, cascade on chunk delete).
3. Prove enqueue-on-ingest for both the local corpus and Gmail paths, staleness
   against a real re-ingest, and backfill idempotency/resumability.
4. Smoke-test the whole path: persisted clean chunk -> fake extraction ->
   entity/relation/fact rows -> graph evidence lookup -> exact authored text and
   offsets.

The extraction provider is always a fake. No model is downloaded and no remote
endpoint is called.
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg
import pytest
from psycopg.rows import dict_row

from email_thread_rag.app.schemas import EmailRecord
from email_thread_rag.config import Settings
from email_thread_rag.graph.backfill import backfill_graph_jobs
from email_thread_rag.graph.fakes import ExplodingGraphProvider, FakeGraphProvider, graph_json
from email_thread_rag.graph.fingerprint import PROMPT_VERSION, SCHEMA_VERSION
from email_thread_rag.graph.repository import PostgresGraphStore
from email_thread_rag.graph.worker import GraphWorker
from email_thread_rag.rag.chunking import chunk_email
from email_thread_rag.rag.paradedb.ingest import persist_corpus_to_paradedb
from email_thread_rag.rag.vector_index import HashingEncoder

from graph_store_contract import GraphStoreContract

pytestmark = pytest.mark.integration

ENCODER = HashingEncoder(dim=768)
TENANT = "acme"
MAILBOX = "inbox"
MODEL = "fake-graph-model"
BODY = "Final budget attached. The approved amount is $1200 for Acme Supplies."


@pytest.fixture
def autocommit_conn(migrated_database_url):
    conn = psycopg.connect(migrated_database_url, row_factory=dict_row, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def clean_tables(autocommit_conn):
    autocommit_conn.execute(
        "TRUNCATE graph_extraction_jobs, chunk_entity_mentions, relation_observations, "
        "fact_evidence, facts, graph_entities, email_chunks, email_messages "
        "RESTART IDENTITY CASCADE"
    )
    yield


def graph_settings(**overrides) -> Settings:
    kwargs = dict(
        graph_extraction_enabled=True,
        graph_base_url="http://fake.invalid/v1",
        graph_model=MODEL,
        graph_schema_version=SCHEMA_VERSION,
        graph_prompt_version=PROMPT_VERSION,
        embedding_dim=768,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def make_email(*, doc_id="msg-2", message_id="<msg-2@example.com>", subject="Re: Budget Review",
               body=BODY, thread_id="thread-alpha", in_reply_to=None) -> EmailRecord:
    return EmailRecord(
        doc_id=doc_id, message_id=message_id, thread_id=thread_id,
        date=datetime(2024, 1, 7, tzinfo=timezone.utc), sender="bob@corp.com",
        to=["alice@corp.com"], subject=subject, body_text=body, in_reply_to=in_reply_to,
        source_path="/tmp/msg-2.json", source_type="fixture",
    )


def persist(conn, email, *, settings=None, tenant_id=TENANT, mailbox_id=MAILBOX) -> dict:
    return persist_corpus_to_paradedb(
        conn, [email], chunk_email(email), tenant_id=tenant_id, mailbox_id=mailbox_id,
        encoder=ENCODER, embedding_dim=768, settings=settings,
    )


def read_chunk(conn, chunk_id, *, tenant_id=TENANT):
    return conn.execute(
        "SELECT * FROM email_chunks WHERE chunk_id = %s AND tenant_id = %s", (chunk_id, tenant_id)
    ).fetchone()


def drain(conn, responder):
    GraphWorker(
        PostgresGraphStore(conn),
        FakeGraphProvider(responder=responder, model_id=MODEL),
        schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
    ).drain()


# =========================================================================
# 1. The shared contract, against Postgres
# =========================================================================
class TestPostgresGraphStore(GraphStoreContract):
    @pytest.fixture
    def store(self, autocommit_conn):
        return PostgresGraphStore(autocommit_conn)

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
                    "INSERT INTO email_messages (tenant_id, mailbox_id, message_id, thread_id, sender, authored_text) "
                    "VALUES (%s,%s,%s,'thread-alpha','alice@corp.com',%s) "
                    "ON CONFLICT (tenant_id, mailbox_id, message_id) DO NOTHING",
                    (tenant_id, mailbox_id, message_id, text),
                )
                autocommit_conn.execute(
                    """
                    INSERT INTO email_chunks (
                        chunk_id, tenant_id, mailbox_id, message_id, thread_id, chunk_index,
                        chunk_kind, sender, subject, text, embed_text, source_start, content_hash, metadata
                    ) VALUES (%s,%s,%s,%s,'thread-alpha',%s,'email','alice@corp.com','Budget Review',
                              %s,%s,0,'hash',
                              '{"to":["bob@corp.com"],"cc":["carol@corp.com"]}'::jsonb)
                    """,
                    (chunk_id, tenant_id, mailbox_id, message_id, counter["index"], text, text),
                )
                counter["index"] += 1
            row = autocommit_conn.execute(
                "SELECT id FROM email_chunks WHERE chunk_id = %s AND tenant_id = %s", (chunk_id, tenant_id)
            ).fetchone()
            return PostgresGraphStore(autocommit_conn).load_chunk_state(row["id"])

        return _make

    @pytest.fixture
    def read_state(self, store):
        return lambda state: store.load_chunk_state(state.chunk_db_id)

    @pytest.fixture
    def mutate_chunk(self, autocommit_conn):
        def _mutate(state, *, text):
            autocommit_conn.execute("UPDATE email_chunks SET text = %s WHERE id = %s", (text, state.chunk_db_id))

        return _mutate


# =========================================================================
# 2. Schema
# =========================================================================
@pytest.fixture
def db_chunk(autocommit_conn):
    persist(autocommit_conn, make_email())
    row = autocommit_conn.execute("SELECT id FROM email_chunks ORDER BY id LIMIT 1").fetchone()
    return PostgresGraphStore(autocommit_conn).load_chunk_state(row["id"])


def test_graph_tables_and_columns_exist(autocommit_conn):
    for table in ("graph_entities", "chunk_entity_mentions", "relation_observations",
                  "facts", "fact_evidence", "graph_extraction_jobs"):
        assert autocommit_conn.execute(
            "SELECT to_regclass(%s) AS t", (f"public.{table}",)
        ).fetchone()["t"] is not None
    cols = {
        r["column_name"]
        for r in autocommit_conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'email_chunks'"
        ).fetchall()
    }
    assert {"graph_input_hash", "graph_extracted_at"} <= cols


def test_duplicate_jobs_are_rejected_by_the_database(autocommit_conn, db_chunk):
    for _ in range(2):
        autocommit_conn.execute(
            "INSERT INTO graph_extraction_jobs (chunk_db_id, tenant_id, mailbox_id, chunk_id, extraction_input_hash) "
            "VALUES (%s,%s,%s,%s,'h') ON CONFLICT DO NOTHING",
            (db_chunk.chunk_db_id, TENANT, MAILBOX, db_chunk.chunk_id),
        )
    n = autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"]
    assert n == 1
    with pytest.raises(psycopg.errors.UniqueViolation):
        autocommit_conn.execute(
            "INSERT INTO graph_extraction_jobs (chunk_db_id, tenant_id, mailbox_id, chunk_id, extraction_input_hash) "
            "VALUES (%s,%s,%s,%s,'h')",
            (db_chunk.chunk_db_id, TENANT, MAILBOX, db_chunk.chunk_id),
        )


def test_evidence_kind_check_forbids_metadata_with_offsets(autocommit_conn, db_chunk):
    # Two entities to satisfy the relation FKs.
    ids = [
        autocommit_conn.execute(
            "INSERT INTO graph_entities (tenant_id, mailbox_id, entity_type, canonical_name, normalized_name) "
            "VALUES (%s,%s,'PERSON',%s,%s) RETURNING entity_id",
            (TENANT, MAILBOX, name, name.lower()),
        ).fetchone()["entity_id"]
        for name in ("Alice", "Bob")
    ]
    base = (
        "INSERT INTO relation_observations (tenant_id, mailbox_id, subject_entity_id, predicate, "
        "object_entity_id, chunk_db_id, chunk_id, chunk_start, chunk_end, evidence_kind, "
        "extraction_method, extraction_version, extraction_model) VALUES "
        "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'m','v','model')"
    )
    # metadata with offsets -> rejected
    with pytest.raises(psycopg.errors.CheckViolation):
        autocommit_conn.execute(base, (TENANT, MAILBOX, ids[0], "SENT", ids[1],
                                       db_chunk.chunk_db_id, db_chunk.chunk_id, 0, 5, "metadata"))
    # text without offsets -> rejected
    with pytest.raises(psycopg.errors.CheckViolation):
        autocommit_conn.execute(base, (TENANT, MAILBOX, ids[0], "MENTIONS", ids[1],
                                       db_chunk.chunk_db_id, db_chunk.chunk_id, None, None, "text"))


def test_deleting_a_chunk_cascades_to_graph_rows(autocommit_conn, db_chunk):
    store = PostgresGraphStore(autocommit_conn)
    store.enqueue(db_chunk, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id=MODEL)
    job = store.claim_job(owner="w")
    resolved = _resolve_smoke(db_chunk)
    store.commit_graph(job, resolved=resolved, method="llm", extraction_version=SCHEMA_VERSION,
                       schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id=MODEL)
    assert autocommit_conn.execute("SELECT count(*) AS n FROM chunk_entity_mentions").fetchone()["n"] >= 1

    autocommit_conn.execute("DELETE FROM email_chunks WHERE id = %s", (db_chunk.chunk_db_id,))
    for table in ("graph_extraction_jobs", "chunk_entity_mentions", "relation_observations", "fact_evidence"):
        assert autocommit_conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"] == 0


def _resolve_smoke(state):
    from email_thread_rag.graph.extract import resolve_extraction
    from email_thread_rag.graph.prompt import validate_extraction

    return resolve_extraction(validate_extraction(_smoke_json()), state)


def _smoke_json():
    return graph_json(
        entities=[{"name": "Acme Supplies", "type": "ORG", "evidence": "Acme Supplies"}],
        facts=[{"subject": "approved amount", "predicate": "is", "object": "$1200",
                "evidence": "approved amount is $1200"}],
    )


# =========================================================================
# 3. Enqueue-on-ingest
# =========================================================================
def test_ingestion_queues_nothing_when_disabled(autocommit_conn):
    stats = persist(autocommit_conn, make_email(), settings=None)
    assert stats["graph_jobs"] == 0
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 0


def test_ingestion_queues_work_when_enabled(autocommit_conn):
    stats = persist(autocommit_conn, make_email(), settings=graph_settings())
    assert stats["graph_jobs"] == 1
    job = autocommit_conn.execute("SELECT * FROM graph_extraction_jobs").fetchone()
    assert job["status"] == "pending" and job["tenant_id"] == TENANT


def test_reprocessing_an_unchanged_message_creates_no_duplicate_work(autocommit_conn):
    settings = graph_settings()
    first = persist(autocommit_conn, make_email(), settings=settings)
    second = persist(autocommit_conn, make_email(), settings=settings)
    assert first["graph_jobs"] == 1 and second["graph_jobs"] == 0
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 1


# =========================================================================
# 4. The smoke test
# =========================================================================
def test_smoke_persisted_chunk_to_graph_to_evidence_lookup(autocommit_conn):
    settings = graph_settings()
    stats = persist(autocommit_conn, make_email(), settings=settings)
    assert stats["chunks"] == 1 and stats["graph_jobs"] == 1

    before = read_chunk(autocommit_conn, "msg-2-email-0")
    original_text = before["text"]
    assert before["graph_input_hash"] is None

    drain(autocommit_conn, lambda ei: _smoke_json())

    store = PostgresGraphStore(autocommit_conn)

    # Entity located and mention offsets map exactly to the clean authored text.
    entity = store.find_entity(tenant_id=TENANT, mailbox_id=MAILBOX, entity_type="ORG", name="Acme Supplies")
    assert entity is not None and entity["canonical_name"] == "Acme Supplies"
    mentions = store.list_mentions(tenant_id=TENANT, mailbox_id=MAILBOX, entity_id=entity["entity_id"])
    assert len(mentions) == 1
    m = mentions[0]
    assert m["clean_text"] == original_text
    assert m["clean_text"][m["chunk_start"]:m["chunk_end"]] == m["mention_text"] == "Acme Supplies"

    # Fact with exact evidence span + hash.
    facts = store.list_facts(tenant_id=TENANT, mailbox_id=MAILBOX, subject="approved amount")
    assert facts[0]["object_value"] == "$1200" and facts[0]["status"] == "active"
    ev = facts[0]["evidence"][0]
    assert ev["clean_text"][ev["chunk_start"]:ev["chunk_end"]] == ev["evidence_text"] == "approved amount is $1200"

    # Evidence lookup returns the clean authored text -- no headers, no model words.
    chunks = store.evidence_chunks(tenant_id=TENANT, mailbox_id=MAILBOX, chunk_ids=["msg-2-email-0"])
    assert chunks["msg-2-email-0"] == original_text
    assert not chunks["msg-2-email-0"].startswith("Subject:")

    # The chunk was marked extracted; re-ingest queues nothing new.
    assert read_chunk(autocommit_conn, "msg-2-email-0")["graph_input_hash"] is not None
    assert persist(autocommit_conn, make_email(), settings=settings)["graph_jobs"] == 0


# =========================================================================
# 5. Staleness against a real re-ingest
# =========================================================================
def test_a_re_ingest_during_the_llm_call_defeats_the_stale_job(autocommit_conn):
    settings = graph_settings()
    persist(autocommit_conn, make_email(), settings=settings)
    store = PostgresGraphStore(autocommit_conn)

    def responder(extraction_input):
        persist(autocommit_conn, make_email(body="An entirely new authored body about Zephyr."), settings=settings)
        return _smoke_json()  # describes the OLD body

    GraphWorker(store, FakeGraphProvider(responder=responder, model_id=MODEL),
                schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION).run_once()

    # The stale extraction must not land: Acme Supplies is gone from the new text.
    assert store.find_entity(tenant_id=TENANT, mailbox_id=MAILBOX, entity_type="ORG", name="Acme Supplies") is None
    row = read_chunk(autocommit_conn, "msg-2-email-0")
    assert "Zephyr" in row["text"] and row["graph_input_hash"] is None  # replacement job still pending


# =========================================================================
# 6. Backfill
# =========================================================================
def test_backfill_queues_pre_existing_chunks(autocommit_conn):
    for i in range(3):
        persist(autocommit_conn, make_email(doc_id=f"m-{i}", message_id=f"<m-{i}@x.com>", body=f"Body {i}."), settings=None)
    assert backfill_graph_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=graph_settings()) == 3


def test_backfill_is_idempotent(autocommit_conn):
    persist(autocommit_conn, make_email(), settings=None)
    settings = graph_settings()
    assert backfill_graph_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings) == 1
    assert backfill_graph_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings) == 0


def test_backfill_is_resumable(autocommit_conn):
    for i in range(5):
        persist(autocommit_conn, make_email(doc_id=f"m-{i}", message_id=f"<m-{i}@x.com>", body=f"Body {i}."), settings=None)
    settings = graph_settings()
    partial = backfill_graph_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings,
                                  batch_size=1, max_chunks=2)
    assert partial == 2
    rest = backfill_graph_jobs(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX, settings=settings, batch_size=2)
    assert rest == 3


def test_backfill_is_tenant_scoped(autocommit_conn):
    persist(autocommit_conn, make_email(), settings=None, tenant_id="acme")
    persist(autocommit_conn, make_email(), settings=None, tenant_id="globex")
    assert backfill_graph_jobs(autocommit_conn, tenant_id="acme", mailbox_id=MAILBOX, settings=graph_settings()) == 1
    rows = autocommit_conn.execute("SELECT DISTINCT tenant_id FROM graph_extraction_jobs").fetchall()
    assert [r["tenant_id"] for r in rows] == ["acme"]


# =========================================================================
# 7. Tenant isolation through the graph
# =========================================================================
def test_entities_and_evidence_are_tenant_isolated(autocommit_conn):
    settings = graph_settings()
    persist(autocommit_conn, make_email(body="Acme authored body about Acme Supplies."), settings=settings, tenant_id="acme")
    persist(autocommit_conn, make_email(body="Globex authored body."), settings=settings, tenant_id="globex")
    # Only Acme's chunk is extracted.
    autocommit_conn.execute("DELETE FROM graph_extraction_jobs WHERE tenant_id <> 'acme'")
    drain(autocommit_conn, lambda ei: graph_json(
        entities=[{"name": "Acme Supplies", "type": "ORG", "evidence": "Acme Supplies"}]))

    store = PostgresGraphStore(autocommit_conn)
    assert store.find_entity(tenant_id="acme", mailbox_id=MAILBOX, entity_type="ORG", name="Acme Supplies") is not None
    assert store.find_entity(tenant_id="globex", mailbox_id=MAILBOX, entity_type="ORG", name="Acme Supplies") is None
    # Globex's evidence lookup never returns Acme's chunk.
    assert store.evidence_chunks(tenant_id="globex", mailbox_id=MAILBOX, chunk_ids=["msg-2-email-0"]) != \
        store.evidence_chunks(tenant_id="acme", mailbox_id=MAILBOX, chunk_ids=["msg-2-email-0"])


# =========================================================================
# 8. Gmail ingestion enqueues without calling the provider
# =========================================================================
def test_gmail_ingestion_enqueues_graph_without_calling_the_provider(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    sink = ParadeDBChunkSink(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX,
                             encoder=ENCODER, embedding_dim=768, settings=graph_settings())
    email = make_email()
    email.source_type = "gmail"
    # No provider reaches the sink; if ingestion tried to call one it would fail.
    assert sink.persist(email) == 1

    job = autocommit_conn.execute("SELECT * FROM graph_extraction_jobs").fetchone()
    assert job["status"] == "pending" and job["tenant_id"] == TENANT
    # Chunk fully persisted, no extraction yet.
    assert read_chunk(autocommit_conn, "msg-2-email-0")["graph_input_hash"] is None


def test_the_gmail_sink_queues_nothing_when_graph_is_off(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    ParadeDBChunkSink(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX,
                      encoder=ENCODER, embedding_dim=768).persist(make_email())
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 0


def test_deleting_a_gmail_message_removes_its_graph_jobs(autocommit_conn):
    from email_thread_rag.gmail.sink import ParadeDBChunkSink

    sink = ParadeDBChunkSink(autocommit_conn, tenant_id=TENANT, mailbox_id=MAILBOX,
                             encoder=ENCODER, embedding_dim=768, settings=graph_settings())
    email = make_email()
    sink.persist(email)
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 1
    sink.delete_message(email.message_id)
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 0


def test_the_provider_is_never_constructed_on_the_ingestion_path(autocommit_conn):
    # Belt-and-braces: enqueue must not build or call an extractor.
    persist(autocommit_conn, make_email(), settings=graph_settings())
    exploding = ExplodingGraphProvider()
    # The job exists but nothing has called the provider.
    assert exploding.model_id == "exploding-graph-model"
    assert autocommit_conn.execute("SELECT count(*) AS n FROM graph_extraction_jobs").fetchone()["n"] == 1
