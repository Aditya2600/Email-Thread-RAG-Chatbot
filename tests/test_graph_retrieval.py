"""Stage-6 graph-evidence resolution over the in-memory store.

``collect_graph_chunk_ids`` turns a plan into ordered evidence chunk ids using
only the narrow store reads. Proven here without ParadeDB: entity routing,
active-only temporal exclusion, conservative as-of dating, tenant isolation, and
"no match -> empty (caller falls back to hybrid)". The engine-level wiring and
exact-offset citation guarantees are covered in the ParadeDB integration suite.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from email_thread_rag.graph.models import ChunkGraphState
from email_thread_rag.graph.retrieval import collect_graph_chunk_ids
from email_thread_rag.graph.store import InMemoryGraphStore
from email_thread_rag.rag.planner import RetrievalPlan, RetrievalRoute

from graph_store_contract import PROMPT_VERSION, SCHEMA_VERSION, build_resolved, commit, enqueue


def _seed(store, *, chunk_id, text, tenant_id="acme", mailbox_id="inbox",
          recipients=None, entities=None, relations=None, facts=None):
    state = ChunkGraphState(
        chunk_db_id=len(store.chunks) + 1, chunk_id=chunk_id, tenant_id=tenant_id,
        mailbox_id=mailbox_id, text=text, sender="alice@corp.com",
        recipients=recipients or [], source_start=0,
    )
    store.add_chunk(state)
    enqueue(store, state)
    job = store.claim_job(owner="w")
    commit(store, job, build_resolved(state, entities=entities, relations=relations, facts=facts))
    return state


def _plan(routes, **kw):
    return RetrievalPlan(routes=tuple(routes), tenant_id="acme", mailbox_id="inbox", **kw)


def _set_effective_date(store, object_value, when):
    for fact in store._facts.values():
        if fact["object_value"] == object_value:
            fact["effective_date"] = when


# entity route -> the chunk that mentions the entity
def test_entity_route_returns_the_mentioning_chunk():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="c-1", text="Alice approved the Q3 budget of $1200 for Project Atlas.",
          entities=[{"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}])
    plan = _plan([RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_ENTITY], entity_terms=("Project Atlas",))
    assert collect_graph_chunk_ids(store, plan) == ["c-1"]


# 7. No graph match -> empty list (caller falls back to hybrid).
def test_no_entity_match_returns_empty():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="c-1", text="Alice approved the Q3 budget.",
          entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}])
    plan = _plan([RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_ENTITY], entity_terms=("Nonexistent Corp",))
    assert collect_graph_chunk_ids(store, plan) == []


# 5. Current route returns only active-fact evidence; superseded excluded.
def test_current_route_excludes_superseded_facts():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="c-old", text="The budget is $1000.",
          facts=[{"subject": "budget", "predicate": "amount", "object": "$1000", "evidence": "The budget is $1000"}])
    # Explicit update cue supersedes the prior fact (Stage-5 rule).
    _seed(store, chunk_id="c-new", text="The budget is now $1500.",
          facts=[{"subject": "budget", "predicate": "amount", "object": "$1500", "evidence": "The budget is now $1500"}])

    plan = _plan([RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_CURRENT], subject_terms=("budget",))
    got = collect_graph_chunk_ids(store, plan)
    assert got == ["c-new"]  # superseded $1000 chunk excluded


# 6. as-of uses dated facts only; undated facts are never "historical".
def test_as_of_uses_dated_facts_and_never_undated_ones():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="c-jan", text="The budget is $1000.",
          facts=[{"subject": "budget", "predicate": "amount", "object": "$1000", "evidence": "The budget is $1000"}])
    _seed(store, chunk_id="c-mar", text="The budget is $1500.",
          facts=[{"subject": "budget", "predicate": "amount", "object": "$1500", "evidence": "The budget is $1500"}])
    _set_effective_date(store, "$1000", datetime(2026, 1, 1, tzinfo=timezone.utc))
    # $1500 stays undated on purpose.

    plan = _plan([RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_AS_OF],
                 subject_terms=("budget",), as_of=date(2026, 2, 1))
    got = collect_graph_chunk_ids(store, plan)
    assert got == ["c-jan"]  # dated & <= as_of

    # A cutoff before the dated fact yields nothing; the undated one never counts.
    plan_before = _plan([RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_AS_OF],
                        subject_terms=("budget",), as_of=date(2025, 12, 1))
    assert collect_graph_chunk_ids(store, plan_before) == []


# 8. Tenant isolation: another tenant's evidence never leaks.
def test_graph_evidence_is_tenant_isolated():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="a", text="Alice approved Project Atlas.", tenant_id="acme",
          entities=[{"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}])
    _seed(store, chunk_id="b", text="Bob approved Project Atlas.", tenant_id="globex",
          entities=[{"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}])
    plan = RetrievalPlan(routes=(RetrievalRoute.HYBRID, RetrievalRoute.GRAPH_ENTITY),
                         tenant_id="acme", mailbox_id="inbox", entity_terms=("Project Atlas",))
    assert collect_graph_chunk_ids(store, plan) == ["a"]


# 9 (store level). Metadata relations carry no text offsets, so they can only
# help *retrieve* a chunk -- never fabricate an authored-text citation span.
def test_metadata_relation_has_no_text_offsets():
    store = InMemoryGraphStore()
    state = _seed(store, chunk_id="c-1", text="Please review the plan.",
                  recipients=["bob@corp.com"],
                  entities=[{"name": "Alice", "type": "PERSON", "evidence": "review"}])  # any text mention
    relations = store.list_relations(tenant_id="acme", mailbox_id="inbox")
    meta = [r for r in relations if r["evidence_kind"] == "metadata"]
    assert meta, "sender->recipient SENT edge should exist"
    for r in meta:
        assert r["chunk_start"] is None and r["chunk_end"] is None


# A hybrid-only plan asks the graph for nothing.
def test_hybrid_only_plan_collects_nothing():
    store = InMemoryGraphStore()
    _seed(store, chunk_id="c-1", text="Alice approved Project Atlas.",
          entities=[{"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}])
    plan = _plan([RetrievalRoute.HYBRID], entity_terms=("Project Atlas",))
    assert collect_graph_chunk_ids(store, plan) == []
