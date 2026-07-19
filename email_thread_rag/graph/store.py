"""The graph store: an interface plus an in-memory implementation.

Same shape as Stage 4's ContextJobStore: a Protocol that both a dict-backed fake
and a Postgres store satisfy, exercised by one shared contract test so the fake
the fast suite relies on cannot drift from the real one.

Two responsibilities: the extraction *queue* (enqueue/claim/commit/fail, the
stale-job guard) and the narrow *read* methods over the resulting graph. Read
results are plain dicts with documented keys so the fake and Postgres return
byte-identical shapes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Protocol

from email_thread_rag.graph.fingerprint import extraction_hash_of
from email_thread_rag.graph.models import ChunkGraphState, GraphJob, ResolvedGraph


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GraphStore(Protocol):
    # --- queue -----------------------------------------------------------
    def enqueue(
        self, state: ChunkGraphState, *, schema_version: str, prompt_version: str, model_id: str
    ) -> Optional[GraphJob]:
        """Queue extraction for a chunk. None if already queued (idempotent by
        fingerprint)."""

    def claim_job(self, *, owner: str, lease_seconds: int = 300) -> Optional[GraphJob]: ...

    def get_job(self, job_id: int) -> Optional[GraphJob]: ...

    def load_chunk_state(self, chunk_db_id: int) -> Optional[ChunkGraphState]: ...

    def commit_graph(
        self,
        job: GraphJob,
        *,
        resolved: ResolvedGraph,
        method: str,
        extraction_version: str,
        schema_version: str,
        prompt_version: str,
        model_id: str,
    ) -> bool:
        """Atomically write the graph rows, or refuse if the chunk changed since
        the job was enqueued. Returns False when stale."""

    def fail_job(self, job_id: int, error: str, *, error_rule: str | None = None, max_attempts: int = 3) -> None: ...

    def chunks_needing_graph(
        self, *, tenant_id: str, mailbox_id: str, limit: int = 100, after_id: int = 0
    ) -> list[ChunkGraphState]: ...

    # --- reads -----------------------------------------------------------
    def find_entity(
        self, *, tenant_id: str, mailbox_id: str, entity_type: str, name: str
    ) -> Optional[dict]: ...

    def list_mentions(self, *, tenant_id: str, mailbox_id: str, entity_id: int) -> list[dict]:
        """Mentions of an entity, each with the clean source chunk text + offsets."""

    def list_relations(
        self, *, tenant_id: str, mailbox_id: str, subject_entity_id: int | None = None
    ) -> list[dict]: ...

    def list_facts(
        self, *, tenant_id: str, mailbox_id: str, subject: str | None = None, status: str | None = None
    ) -> list[dict]:
        """Facts with their exact evidence spans attached."""

    def evidence_chunks(self, *, tenant_id: str, mailbox_id: str, chunk_ids: list[str]) -> dict[str, str]:
        """Clean ``text`` for a set of chunk ids -- the authored evidence behind a
        graph result. Tenant/mailbox scoped; never returns another mailbox's text."""

    # --- Stage-6 planner reads (narrow, tenant/mailbox scoped) ------------
    def entities_matching(self, *, tenant_id: str, mailbox_id: str, names: list[str]) -> list[dict]:
        """Entities whose normalized_name exactly equals a normalized query term.
        Conservative match only -- no fuzzy merging."""

    def entity_evidence_chunk_ids(
        self, *, tenant_id: str, mailbox_id: str, entity_ids: list[int], limit: int = 20
    ) -> list[str]:
        """Chunk ids that mention or relate the given entities, sorted for
        determinism. Includes metadata-edge chunks (they help *retrieve* related
        emails) but the caller always cites the chunk's own clean text."""

    def fact_evidence_chunk_ids(
        self,
        *,
        tenant_id: str,
        mailbox_id: str,
        subjects: list[str] | None = None,
        status: str | None = None,
        as_of=None,
        limit: int = 20,
    ) -> list[str]:
        """Chunk ids backing facts in scope. ``status='active'`` excludes
        superseded facts; ``as_of`` keeps only facts with a real effective_date
        at or before it (undated facts are never treated as historical)."""


class InMemoryGraphStore:
    """Dict-backed store with the same semantics as Postgres. Test/demo only."""

    def __init__(self):
        self.chunks: dict[int, ChunkGraphState] = {}
        self._jobs: dict[int, GraphJob] = {}
        self._next_job_id = 1
        self._entities: dict[int, dict] = {}
        self._entity_index: dict[tuple, int] = {}  # (tenant, mailbox, type, normkey) -> entity_id
        self._next_entity_id = 1
        self._mentions: list[dict] = []
        self._facts: dict[int, dict] = {}
        self._fact_evidence: list[dict] = []
        self._next_fact_id = 1
        self._relations: list[dict] = []

    # --- test/demo seam --------------------------------------------------
    def add_chunk(self, state: ChunkGraphState) -> ChunkGraphState:
        self.chunks[state.chunk_db_id] = state
        return state

    # --- queue -----------------------------------------------------------
    def enqueue(self, state, *, schema_version, prompt_version, model_id) -> Optional[GraphJob]:
        digest = extraction_hash_of(
            state.as_extraction_input(),
            schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
        )
        if state.graph_input_hash == digest:
            return None
        for job in self._jobs.values():
            if (
                job.tenant_id == state.tenant_id
                and job.mailbox_id == state.mailbox_id
                and job.chunk_id == state.chunk_id
                and job.extraction_input_hash == digest
            ):
                return None
        job = GraphJob(
            id=self._next_job_id,
            chunk_db_id=state.chunk_db_id,
            tenant_id=state.tenant_id,
            mailbox_id=state.mailbox_id,
            chunk_id=state.chunk_id,
            extraction_input_hash=digest,
        )
        self._jobs[job.id] = job
        self._next_job_id += 1
        return job

    def claim_job(self, *, owner, lease_seconds=300) -> Optional[GraphJob]:
        now = utcnow()
        for job in sorted(self._jobs.values(), key=lambda j: j.id):
            expired = job.status == "running" and job.leased_until is not None and job.leased_until <= now
            if job.status == "pending" or expired:
                job.status = "running"
                job.attempts += 1
                job.lease_owner = owner
                job.leased_until = now + timedelta(seconds=lease_seconds)
                return job
        return None

    def get_job(self, job_id) -> Optional[GraphJob]:
        return self._jobs.get(job_id)

    def load_chunk_state(self, chunk_db_id) -> Optional[ChunkGraphState]:
        return self.chunks.get(chunk_db_id)

    def commit_graph(
        self, job, *, resolved, method, extraction_version, schema_version, prompt_version, model_id
    ) -> bool:
        state = self.chunks.get(job.chunk_db_id)
        if state is None:
            self._retire(job)
            return False
        current = extraction_hash_of(
            state.as_extraction_input(),
            schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
        )
        if current != job.extraction_input_hash:
            # Chunk changed under a slow LLM call: this graph describes text that
            # no longer exists. Drop it; a newer job covers the new text.
            self._retire(job)
            return False

        self._delete_chunk_graph(job.chunk_db_id)
        prov = dict(method=method, version=extraction_version, model=model_id)

        for m in resolved.mentions:
            entity_id = self._upsert_entity(state, m.entity_type, m.normalized_name, m.canonical_name)
            self._mentions.append({
                "tenant_id": state.tenant_id, "mailbox_id": state.mailbox_id,
                "chunk_db_id": state.chunk_db_id, "chunk_id": state.chunk_id, "entity_id": entity_id,
                "mention_text": m.mention_text, "chunk_start": m.chunk_start, "chunk_end": m.chunk_end,
                "source_start": m.source_start, "source_end": m.source_end, **prov,
            })

        for r in resolved.relations:
            subj_id = self._upsert_entity(state, r.subject_key[0], r.subject_key[1])
            obj_id = self._upsert_entity(state, r.object_key[0], r.object_key[1])
            self._relations.append({
                "tenant_id": state.tenant_id, "mailbox_id": state.mailbox_id,
                "subject_entity_id": subj_id, "predicate": r.predicate, "object_entity_id": obj_id,
                "chunk_db_id": state.chunk_db_id, "chunk_id": state.chunk_id,
                "chunk_start": r.chunk_start, "chunk_end": r.chunk_end, "mention_text": r.mention_text,
                "evidence_kind": r.evidence_kind, **prov,
            })

        for f in resolved.facts:
            fact_id = self._next_fact_id
            self._next_fact_id += 1
            supersedes = None
            if f.has_update_cue:
                supersedes = self._supersede_prior(state, f.normalized_subject, f.normalized_predicate)
            self._facts[fact_id] = {
                "fact_id": fact_id, "tenant_id": state.tenant_id, "mailbox_id": state.mailbox_id,
                "subject": f.subject, "predicate": f.predicate, "object_value": f.object_value,
                "normalized_subject": f.normalized_subject, "normalized_predicate": f.normalized_predicate,
                "status": "active", "effective_date": None, "supersedes_fact_id": supersedes, **prov,
            }
            self._fact_evidence.append({
                "fact_id": fact_id, "tenant_id": state.tenant_id, "mailbox_id": state.mailbox_id,
                "chunk_db_id": state.chunk_db_id, "chunk_id": state.chunk_id,
                "chunk_start": f.chunk_start, "chunk_end": f.chunk_end,
                "source_start": f.source_start, "source_end": f.source_end,
                "evidence_text": f.evidence_text, "evidence_hash": f.evidence_hash,
            })

        state.graph_input_hash = job.extraction_input_hash
        self._retire(job)
        return True

    def _delete_chunk_graph(self, chunk_db_id: int) -> None:
        self._mentions = [m for m in self._mentions if m["chunk_db_id"] != chunk_db_id]
        self._relations = [r for r in self._relations if r["chunk_db_id"] != chunk_db_id]
        stale_fact_ids = {e["fact_id"] for e in self._fact_evidence if e["chunk_db_id"] == chunk_db_id}
        self._fact_evidence = [e for e in self._fact_evidence if e["chunk_db_id"] != chunk_db_id]
        for fid in stale_fact_ids:
            self._facts.pop(fid, None)
        for fact in self._facts.values():
            if fact["supersedes_fact_id"] in stale_fact_ids:
                fact["supersedes_fact_id"] = None  # mirrors ON DELETE SET NULL

    def _upsert_entity(self, state, entity_type, normalized_name, canonical_name=None) -> int:
        key = (state.tenant_id, state.mailbox_id, entity_type, normalized_name)
        if key in self._entity_index:
            return self._entity_index[key]
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self._entities[entity_id] = {
            "entity_id": entity_id, "tenant_id": state.tenant_id, "mailbox_id": state.mailbox_id,
            "entity_type": entity_type, "canonical_name": canonical_name or normalized_name,
            "normalized_name": normalized_name,
        }
        self._entity_index[key] = entity_id
        return entity_id

    def _supersede_prior(self, state, normalized_subject, normalized_predicate) -> Optional[int]:
        for fact in sorted(self._facts.values(), key=lambda f: f["fact_id"], reverse=True):
            if (
                fact["tenant_id"] == state.tenant_id
                and fact["mailbox_id"] == state.mailbox_id
                and fact["normalized_subject"] == normalized_subject
                and fact["normalized_predicate"] == normalized_predicate
                and fact["status"] == "active"
            ):
                fact["status"] = "superseded"
                return fact["fact_id"]
        return None

    def _retire(self, job) -> None:
        job.status = "done"
        job.completed_at = utcnow()
        job.leased_until = None
        job.lease_owner = None
        job.last_error = None
        job.error_rule = None

    def fail_job(self, job_id, error, *, error_rule=None, max_attempts=3) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.last_error = error
        job.error_rule = error_rule
        job.leased_until = None
        job.lease_owner = None
        job.status = "failed" if job.attempts >= max_attempts else "pending"

    def chunks_needing_graph(self, *, tenant_id, mailbox_id, limit=100, after_id=0) -> list[ChunkGraphState]:
        matches = [
            s for s in self.chunks.values()
            if s.tenant_id == tenant_id and s.mailbox_id == mailbox_id
            and s.graph_input_hash is None and s.chunk_db_id > after_id
        ]
        return sorted(matches, key=lambda s: s.chunk_db_id)[:limit]

    # --- reads -----------------------------------------------------------
    def find_entity(self, *, tenant_id, mailbox_id, entity_type, name) -> Optional[dict]:
        from email_thread_rag.graph.extract import normalized_key

        key = (tenant_id, mailbox_id, entity_type, normalized_key(name))
        entity_id = self._entity_index.get(key)
        return dict(self._entities[entity_id]) if entity_id is not None else None

    def list_mentions(self, *, tenant_id, mailbox_id, entity_id) -> list[dict]:
        out = []
        for m in self._mentions:
            if m["tenant_id"] == tenant_id and m["mailbox_id"] == mailbox_id and m["entity_id"] == entity_id:
                chunk = self.chunks.get(m["chunk_db_id"])
                out.append({**m, "clean_text": chunk.text if chunk else None})
        return sorted(out, key=lambda x: (x["chunk_id"], x["chunk_start"]))

    def list_relations(self, *, tenant_id, mailbox_id, subject_entity_id=None) -> list[dict]:
        out = []
        for r in self._relations:
            if r["tenant_id"] != tenant_id or r["mailbox_id"] != mailbox_id:
                continue
            if subject_entity_id is not None and r["subject_entity_id"] != subject_entity_id:
                continue
            out.append({
                **r,
                "subject_name": self._entities[r["subject_entity_id"]]["canonical_name"],
                "object_name": self._entities[r["object_entity_id"]]["canonical_name"],
            })
        return out

    def list_facts(self, *, tenant_id, mailbox_id, subject=None, status=None) -> list[dict]:
        from email_thread_rag.graph.extract import normalized_key

        want_subject = normalized_key(subject) if subject is not None else None
        out = []
        for fact in sorted(self._facts.values(), key=lambda f: f["fact_id"]):
            if fact["tenant_id"] != tenant_id or fact["mailbox_id"] != mailbox_id:
                continue
            if want_subject is not None and fact["normalized_subject"] != want_subject:
                continue
            if status is not None and fact["status"] != status:
                continue
            evidence = [
                {**e, "clean_text": (self.chunks.get(e["chunk_db_id"]).text if self.chunks.get(e["chunk_db_id"]) else None)}
                for e in self._fact_evidence if e["fact_id"] == fact["fact_id"]
            ]
            out.append({**fact, "evidence": evidence})
        return out

    def evidence_chunks(self, *, tenant_id, mailbox_id, chunk_ids) -> dict[str, str]:
        wanted = set(chunk_ids)
        return {
            s.chunk_id: s.text
            for s in self.chunks.values()
            if s.tenant_id == tenant_id and s.mailbox_id == mailbox_id and s.chunk_id in wanted
        }

    # --- Stage-6 planner reads -------------------------------------------
    def entities_matching(self, *, tenant_id, mailbox_id, names) -> list[dict]:
        from email_thread_rag.graph.extract import normalized_key

        keys = {normalized_key(n) for n in names if n}
        out = [
            dict(e)
            for e in self._entities.values()
            if e["tenant_id"] == tenant_id and e["mailbox_id"] == mailbox_id and e["normalized_name"] in keys
        ]
        return sorted(out, key=lambda e: e["entity_id"])

    def entity_evidence_chunk_ids(self, *, tenant_id, mailbox_id, entity_ids, limit=20) -> list[str]:
        ids = set(entity_ids)
        found: set[str] = set()
        for m in self._mentions:
            if m["tenant_id"] == tenant_id and m["mailbox_id"] == mailbox_id and m["entity_id"] in ids:
                found.add(m["chunk_id"])
        for r in self._relations:
            if r["tenant_id"] != tenant_id or r["mailbox_id"] != mailbox_id:
                continue
            if r["subject_entity_id"] in ids or r["object_entity_id"] in ids:
                found.add(r["chunk_id"])
        return sorted(found)[:limit]

    def fact_evidence_chunk_ids(
        self, *, tenant_id, mailbox_id, subjects=None, status=None, as_of=None, limit=20
    ) -> list[str]:
        from email_thread_rag.graph.extract import normalized_key

        want = {normalized_key(s) for s in subjects if s} if subjects else None
        found: set[str] = set()
        for fact in self._facts.values():
            if fact["tenant_id"] != tenant_id or fact["mailbox_id"] != mailbox_id:
                continue
            if want is not None and fact["normalized_subject"] not in want:
                continue
            if status is not None and fact["status"] != status:
                continue
            if as_of is not None:
                effective = fact.get("effective_date")
                if effective is None or _as_date(effective) > as_of:
                    continue  # undated facts are never historically valid
            for e in self._fact_evidence:
                if e["fact_id"] == fact["fact_id"] and e["tenant_id"] == tenant_id and e["mailbox_id"] == mailbox_id:
                    found.add(e["chunk_id"])
        return sorted(found)[:limit]


def _as_date(value):
    return value.date() if isinstance(value, datetime) else value
