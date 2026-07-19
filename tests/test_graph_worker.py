"""The graph worker end to end against fakes: claim -> call -> validate ->
ground-in-evidence -> commit. No DB, no network, no model."""

from __future__ import annotations

from email_thread_rag.graph.fakes import FakeGraphProvider, UnavailableGraphProvider, graph_json
from email_thread_rag.graph.fingerprint import PROMPT_VERSION, SCHEMA_VERSION
from email_thread_rag.graph.models import ChunkGraphState
from email_thread_rag.graph.store import InMemoryGraphStore
from email_thread_rag.graph.worker import GraphWorker

TEXT = "Alice approved the Q3 budget of $1200 for Project Atlas."
MODEL = "fake-graph-model"


def _seed(store, *, chunk_id="c-1", text=TEXT, tenant_id="acme"):
    state = ChunkGraphState(
        chunk_db_id=len(store.chunks) + 1, chunk_id=chunk_id, tenant_id=tenant_id, mailbox_id="inbox",
        text=text, sender="alice@corp.com", recipients=["bob@corp.com"], source_start=0,
    )
    store.add_chunk(state)
    store.enqueue(state, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id=MODEL)
    return state


def _worker(store, provider):
    return GraphWorker(store, provider)


def test_worker_persists_a_full_graph_from_valid_output():
    store = InMemoryGraphStore()
    state = _seed(store)
    provider = FakeGraphProvider(responder=lambda ei: graph_json(
        entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"},
                  {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}],
        relations=[{"subject": "Alice", "predicate": "WORKS_ON", "object": "Project Atlas", "evidence": "Project Atlas"}],
        facts=[{"subject": "Q3 budget", "predicate": "amount", "object": "$1200", "evidence": "Q3 budget of $1200"}],
    ))
    assert _worker(store, provider).run_once() is True
    assert _worker(store, provider).run_once() is False  # queue drained

    alice = store.find_entity(tenant_id="acme", mailbox_id="inbox", entity_type="PERSON", name="Alice")
    assert alice is not None
    facts = store.list_facts(tenant_id="acme", mailbox_id="inbox", subject="Q3 budget")
    assert facts[0]["object_value"] == "$1200"
    # Evidence maps exactly to clean text.
    ev = facts[0]["evidence"][0]
    assert ev["clean_text"][ev["chunk_start"]:ev["chunk_end"]] == "Q3 budget of $1200"


def test_hallucinated_output_is_discarded_but_the_chunk_is_still_marked():
    store = InMemoryGraphStore()
    state = _seed(store)
    provider = FakeGraphProvider(responder=lambda ei: graph_json(
        entities=[{"name": "Bob", "type": "PERSON", "evidence": "Bob signed the deal"}],  # not in text
    ))
    _worker(store, provider).drain()

    assert store.find_entity(tenant_id="acme", mailbox_id="inbox", entity_type="PERSON", name="Bob") is None
    # Chunk marked extracted; not re-enqueued.
    assert store.enqueue(store.load_chunk_state(state.chunk_db_id),
                         schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id=MODEL) is None


def test_malformed_json_falls_back_to_metadata_only_graph():
    store = InMemoryGraphStore()
    state = _seed(store)
    provider = FakeGraphProvider(responder=lambda ei: "}} not json {{")
    _worker(store, provider).drain()

    # No text entities, but the deterministic header relations (SENT) still land.
    relations = store.list_relations(tenant_id="acme", mailbox_id="inbox")
    assert {r["predicate"] for r in relations} == {"SENT"}
    assert all(r["evidence_kind"] == "metadata" for r in relations)
    assert store.get_job(1).status == "done"


def test_provider_error_returns_the_job_to_pending():
    store = InMemoryGraphStore()
    _seed(store)
    _worker(store, UnavailableGraphProvider()).run_once()

    job = store.get_job(1)
    assert job.status == "pending" and job.error_rule == "provider"


def test_a_re_ingest_during_the_llm_call_defeats_the_stale_job():
    store = InMemoryGraphStore()
    state = _seed(store)

    def responder(extraction_input):
        # The chunk is re-ingested with new text while the model "thinks".
        store.load_chunk_state(state.chunk_db_id).text = "An entirely new body with no Alice."
        return graph_json(entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}])

    _worker(store, FakeGraphProvider(responder=responder)).run_once()

    # The stale entity must not land on the new text.
    assert store.find_entity(tenant_id="acme", mailbox_id="inbox", entity_type="PERSON", name="Alice") is None
    assert store.load_chunk_state(state.chunk_db_id).graph_input_hash is None  # still needs (re-)extraction
