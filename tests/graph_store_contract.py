"""Contract every GraphStore must satisfy.

Subclassed by tests/test_graph_jobs.py (in-memory) and
tests/integration/test_graph_paradedb.py (Postgres). One contract, two
implementations: the fake the fast suite relies on cannot drift from the real
store without this file failing.

The resolved graph passed to ``commit_graph`` is always built by the *shared*
``resolve_extraction`` from evidence strings that really occur in the chunk text,
so the contract also proves the offsets both stores persist are the real ones.
"""

from __future__ import annotations

from email_thread_rag.graph.extract import resolve_extraction
from email_thread_rag.graph.models import ChunkGraphState
from email_thread_rag.graph.prompt import LLMEntity, LLMExtraction, LLMFact, LLMRelation

SCHEMA_VERSION = "test-graph-schema-v1"
PROMPT_VERSION = "test-graph-prompt-v1"
MODEL_ID = "fake-graph-model"

# Every evidence string below is a verbatim substring of this text.
CHUNK_TEXT = "Alice approved the Q3 budget of $1200 for Project Atlas."


def build_resolved(state: ChunkGraphState, *, entities=None, relations=None, facts=None):
    extraction = LLMExtraction(
        entities=[LLMEntity(**e) for e in (entities or [])],
        relations=[LLMRelation(**r) for r in (relations or [])],
        facts=[LLMFact(**f) for f in (facts or [])],
    )
    return resolve_extraction(extraction, state)


def commit(store, job, resolved):
    return store.commit_graph(
        job, resolved=resolved, method="llm",
        extraction_version=SCHEMA_VERSION, schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION, model_id=MODEL_ID,
    )


def enqueue(store, state):
    return store.enqueue(state, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id=MODEL_ID)


class GraphStoreContract:
    """Subclasses provide `store`, `make_chunk`, `read_state`, `mutate_chunk`."""

    # --- enqueue ---------------------------------------------------------
    def test_enqueue_creates_a_pending_job(self, store, make_chunk):
        job = enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        assert job is not None and job.status == "pending" and job.attempts == 0

    def test_enqueue_is_idempotent_for_unchanged_inputs(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        assert enqueue(store, state) is not None
        assert enqueue(store, state) is None

    def test_changed_text_enqueues_a_new_job(self, store, make_chunk):
        first = enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        second = enqueue(store, make_chunk(chunk_id="c-1", text="Different body.", replace=True))
        assert second is not None and second.id != first.id

    def test_a_new_schema_version_enqueues_new_work(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        enqueue(store, state)
        rerun = store.enqueue(state, schema_version="v2", prompt_version=PROMPT_VERSION, model_id=MODEL_ID)
        assert rerun is not None

    def test_a_new_model_enqueues_new_work(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        enqueue(store, state)
        rerun = store.enqueue(state, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, model_id="other")
        assert rerun is not None

    # --- claiming --------------------------------------------------------
    def test_claim_leases_a_job_and_counts_the_attempt(self, store, make_chunk):
        enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        claimed = store.claim_job(owner="w-a")
        assert claimed is not None and claimed.status == "running" and claimed.attempts == 1

    def test_a_leased_job_is_not_claimed_twice(self, store, make_chunk):
        enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        assert store.claim_job(owner="w-a") is not None
        assert store.claim_job(owner="w-b") is None

    def test_claim_returns_none_on_an_empty_queue(self, store):
        assert store.claim_job(owner="w-a") is None

    # --- completion: entities, mentions, relations, facts ----------------
    def test_commit_writes_the_full_graph_with_exact_evidence(self, store, make_chunk, read_state):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        enqueue(store, state)
        job = store.claim_job(owner="w-a")

        resolved = build_resolved(
            state,
            entities=[
                {"name": "Alice", "type": "PERSON", "evidence": "Alice"},
                {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"},
            ],
            relations=[{"subject": "Alice", "predicate": "WORKS_ON", "object": "Project Atlas",
                        "evidence": "Project Atlas"}],
            facts=[{"subject": "Q3 budget", "predicate": "amount", "object": "$1200",
                    "evidence": "Q3 budget of $1200"}],
        )
        assert commit(store, job, resolved) is True
        assert store.get_job(job.id).status == "done"

        alice = store.find_entity(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                  entity_type="PERSON", name="alice")
        assert alice is not None and alice["canonical_name"] == "Alice"

        mentions = store.list_mentions(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                       entity_id=alice["entity_id"])
        assert len(mentions) == 1
        m = mentions[0]
        # Every span maps EXACTLY to clean chunk text.
        assert m["clean_text"][m["chunk_start"]:m["chunk_end"]] == m["mention_text"] == "Alice"
        assert m["clean_text"] == CHUNK_TEXT

        relations = store.list_relations(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                         subject_entity_id=alice["entity_id"])
        text_rels = [r for r in relations if r["evidence_kind"] == "text"]
        assert [r["predicate"] for r in text_rels] == ["WORKS_ON"]
        assert text_rels[0]["object_name"] == "Project Atlas"

        facts = store.list_facts(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id, subject="Q3 budget")
        assert len(facts) == 1 and facts[0]["status"] == "active"
        ev = facts[0]["evidence"][0]
        assert ev["clean_text"][ev["chunk_start"]:ev["chunk_end"]] == ev["evidence_text"] == "Q3 budget of $1200"

    def test_evidence_chunks_returns_clean_text_scoped(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        got = store.evidence_chunks(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id, chunk_ids=["c-1"])
        assert got == {"c-1": CHUNK_TEXT}

    def test_a_committed_chunk_is_not_re_enqueued(self, store, make_chunk, read_state):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        enqueue(store, state)
        job = store.claim_job(owner="w-a")
        commit(store, job, build_resolved(state, entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]))
        assert enqueue(store, read_state(state)) is None

    # --- the stale-job guard ---------------------------------------------
    def test_a_stale_job_cannot_write_a_changed_chunk(self, store, make_chunk, read_state, mutate_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        enqueue(store, state)
        job = store.claim_job(owner="w-a")
        resolved = build_resolved(state, entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}])

        mutate_chunk(state, text="A completely different body with no Alice.")

        assert commit(store, job, resolved) is False
        fresh = read_state(state)
        assert fresh.text == "A completely different body with no Alice."
        assert store.find_entity(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                 entity_type="PERSON", name="Alice") is None
        assert store.get_job(job.id).status == "done"

    # --- temporal supersession -------------------------------------------
    def test_explicit_update_cue_supersedes_but_a_later_fact_alone_does_not(self, store, make_chunk):
        # Fact 1: no cue.
        s1 = make_chunk(chunk_id="c-1", text="The budget is $1000.")
        enqueue(store, s1)
        j1 = store.claim_job(owner="w-a")
        commit(store, j1, build_resolved(s1, facts=[{"subject": "budget", "predicate": "amount",
                                                     "object": "$1000", "evidence": "The budget is $1000"}]))

        # Fact 2: no cue -> both retained, none superseded.
        s2 = make_chunk(chunk_id="c-2", text="The budget is $1100.")
        enqueue(store, s2)
        j2 = store.claim_job(owner="w-a")
        commit(store, j2, build_resolved(s2, facts=[{"subject": "budget", "predicate": "amount",
                                                     "object": "$1100", "evidence": "The budget is $1100"}]))
        active = store.list_facts(tenant_id=s1.tenant_id, mailbox_id=s1.mailbox_id, subject="budget", status="active")
        assert {f["object_value"] for f in active} == {"$1000", "$1100"}

        # Fact 3: explicit cue -> supersedes the most recent active fact.
        s3 = make_chunk(chunk_id="c-3", text="The budget is now $1500.")
        enqueue(store, s3)
        j3 = store.claim_job(owner="w-a")
        commit(store, j3, build_resolved(s3, facts=[{"subject": "budget", "predicate": "amount",
                                                     "object": "$1500", "evidence": "The budget is now $1500"}]))
        active = store.list_facts(tenant_id=s1.tenant_id, mailbox_id=s1.mailbox_id, subject="budget", status="active")
        superseded = store.list_facts(tenant_id=s1.tenant_id, mailbox_id=s1.mailbox_id, subject="budget",
                                      status="superseded")
        assert "$1500" in {f["object_value"] for f in active}
        assert len(superseded) == 1  # exactly one prior fact was retired

    # --- retries ---------------------------------------------------------
    def test_a_failed_job_returns_to_pending_for_retry(self, store, make_chunk):
        enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        job = store.claim_job(owner="w-a")
        store.fail_job(job.id, "graph provider returned HTTP 503", error_rule="provider", max_attempts=3)
        retried = store.get_job(job.id)
        assert retried.status == "pending" and retried.last_error == "graph provider returned HTTP 503"
        assert store.claim_job(owner="w-b") is not None

    def test_a_job_fails_permanently_once_attempts_are_spent(self, store, make_chunk):
        enqueue(store, make_chunk(chunk_id="c-1", text=CHUNK_TEXT))
        for _ in range(3):
            job = store.claim_job(owner="w-a")
            store.fail_job(job.id, "still down", max_attempts=3)
        assert store.get_job(job.id).status == "failed"
        assert store.claim_job(owner="w-a") is None

    # --- backfill scan + isolation ---------------------------------------
    def test_chunks_needing_graph_excludes_extracted_chunks(self, store, make_chunk):
        a = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        make_chunk(chunk_id="c-2", text="Second body.")
        pending = store.chunks_needing_graph(tenant_id=a.tenant_id, mailbox_id=a.mailbox_id)
        assert {s.chunk_id for s in pending} == {"c-1", "c-2"}

        enqueue(store, a)
        job = store.claim_job(owner="w-a")
        commit(store, job, build_resolved(a, entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]))
        remaining = store.chunks_needing_graph(tenant_id=a.tenant_id, mailbox_id=a.mailbox_id)
        assert {s.chunk_id for s in remaining} == {"c-2"}

    def test_the_backfill_scan_is_tenant_scoped(self, store, make_chunk):
        make_chunk(chunk_id="c-1", text=CHUNK_TEXT, tenant_id="acme")
        make_chunk(chunk_id="c-2", text=CHUNK_TEXT, tenant_id="globex")
        acme = store.chunks_needing_graph(tenant_id="acme", mailbox_id="inbox")
        assert {s.chunk_id for s in acme} == {"c-1"}

    # --- Stage-6 planner reads ------------------------------------------
    def _extract(self, store, state, **kw):
        enqueue(store, state)
        job = store.claim_job(owner="w")
        commit(store, job, build_resolved(state, **kw))

    def test_entities_matching_is_exact_and_scoped(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        self._extract(store, state, entities=[
            {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"},
        ])
        hits = store.entities_matching(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                       names=["project atlas"])  # normalized match, any case
        assert [h["canonical_name"] for h in hits] == ["Project Atlas"]
        # No fuzzy merge: a different name does not match.
        assert store.entities_matching(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                       names=["Atlas"]) == []
        # Other tenant sees nothing.
        assert store.entities_matching(tenant_id="globex", mailbox_id="inbox",
                                       names=["Project Atlas"]) == []

    def test_entity_evidence_chunk_ids_reaches_real_chunks(self, store, make_chunk):
        state = make_chunk(chunk_id="c-1", text=CHUNK_TEXT)
        self._extract(store, state, entities=[
            {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"},
        ])
        entity = store.find_entity(tenant_id=state.tenant_id, mailbox_id=state.mailbox_id,
                                   entity_type="PROJECT", name="Project Atlas")
        chunk_ids = store.entity_evidence_chunk_ids(
            tenant_id=state.tenant_id, mailbox_id=state.mailbox_id, entity_ids=[entity["entity_id"]]
        )
        assert chunk_ids == ["c-1"]

    def test_fact_evidence_chunk_ids_filters_active_and_scope(self, store, make_chunk):
        s1 = make_chunk(chunk_id="c-1", text="The budget is $1000.")
        self._extract(store, s1, facts=[{"subject": "budget", "predicate": "amount",
                                         "object": "$1000", "evidence": "The budget is $1000"}])
        s2 = make_chunk(chunk_id="c-2", text="The budget is now $1500.")
        self._extract(store, s2, facts=[{"subject": "budget", "predicate": "amount",
                                        "object": "$1500", "evidence": "The budget is now $1500"}])

        active = store.fact_evidence_chunk_ids(
            tenant_id=s1.tenant_id, mailbox_id=s1.mailbox_id, subjects=["budget"], status="active"
        )
        assert active == ["c-2"]  # the $1000 fact was superseded by the explicit cue
        # No subject filter still respects tenant scope.
        assert store.fact_evidence_chunk_ids(tenant_id="globex", mailbox_id="inbox",
                                             subjects=["budget"], status="active") == []

    def test_entities_are_isolated_per_tenant(self, store, make_chunk):
        acme = make_chunk(chunk_id="a", text=CHUNK_TEXT, tenant_id="acme")
        globex = make_chunk(chunk_id="b", text=CHUNK_TEXT, tenant_id="globex")
        for state in (acme, globex):
            enqueue(store, state)
            job = store.claim_job(owner="w")
            commit(store, job, build_resolved(state, entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]))

        acme_alice = store.find_entity(tenant_id="acme", mailbox_id="inbox", entity_type="PERSON", name="Alice")
        globex_alice = store.find_entity(tenant_id="globex", mailbox_id="inbox", entity_type="PERSON", name="Alice")
        # Same name, two mailboxes: never fuzzy-merged into one entity.
        assert acme_alice["entity_id"] != globex_alice["entity_id"]
        # And Acme's evidence never surfaces for Globex.
        assert store.evidence_chunks(tenant_id="globex", mailbox_id="inbox", chunk_ids=["a"]) == {}
