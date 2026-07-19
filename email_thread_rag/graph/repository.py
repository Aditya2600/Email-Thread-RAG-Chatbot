"""Postgres-backed graph store. Satisfies GraphStore.

psycopg is imported at module level here (this module IS the Postgres one and is
never imported by the memory path), but every caller reaches it lazily.

Connections are expected in autocommit mode with explicit ``transaction()``
blocks -- the same discipline Stages 3/4 use -- so no transaction is ever held
open across an LLM call. The stale-job guard (re-read + re-hash under FOR UPDATE)
lives in ``commit_graph``.
"""

from __future__ import annotations

from typing import Optional

from email_thread_rag.graph.fingerprint import extraction_hash_of
from email_thread_rag.graph.models import ChunkGraphState, GraphJob, ResolvedGraph

_JOB_COLUMNS = (
    "id, chunk_db_id, tenant_id, mailbox_id, chunk_id, extraction_input_hash, status, "
    "attempts, leased_until, lease_owner, last_error, error_rule, completed_at"
)

# Only clean fields: c.text (immutable evidence), safe headers, and the parent's
# sender via a deterministic thread-link join. embed_text / context_prefix are
# deliberately NOT selected -- they are retrieval-only and never graph evidence.
_CHUNK_STATE_SELECT = """
    SELECT c.id AS chunk_db_id, c.chunk_id, c.tenant_id, c.mailbox_id, c.text,
           c.sender, c.subject, c.thread_id, c.sent_at, c.source_start,
           c.metadata, c.graph_input_hash,
           parent.sender AS parent_sender
    FROM email_chunks c
    LEFT JOIN email_messages parent
      ON parent.tenant_id = c.tenant_id
     AND parent.mailbox_id = c.mailbox_id
     AND parent.message_id = c.metadata->>'in_reply_to'
"""


def _row_to_job(row) -> Optional[GraphJob]:
    return GraphJob(**row) if row is not None else None


def _row_to_state(row) -> Optional[ChunkGraphState]:
    if row is None:
        return None
    metadata = row.get("metadata") or {}
    return ChunkGraphState(
        chunk_db_id=row["chunk_db_id"],
        chunk_id=row["chunk_id"],
        tenant_id=row["tenant_id"],
        mailbox_id=row["mailbox_id"],
        text=row["text"],
        subject=row["subject"],
        sender=row["sender"],
        thread_id=row["thread_id"],
        date=row["sent_at"],
        source_start=row.get("source_start"),
        recipients=list(metadata.get("to") or []),
        cc=list(metadata.get("cc") or []),
        in_reply_to=metadata.get("in_reply_to"),
        parent_sender=row.get("parent_sender"),
        graph_input_hash=row["graph_input_hash"],
    )


class PostgresGraphStore:
    def __init__(self, conn):
        self.conn = conn

    # --- queue -----------------------------------------------------------
    def enqueue(self, state, *, schema_version, prompt_version, model_id) -> Optional[GraphJob]:
        digest = extraction_hash_of(
            state.as_extraction_input(),
            schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
        )
        if state.graph_input_hash == digest:
            return None
        row = self.conn.execute(
            f"""
            INSERT INTO graph_extraction_jobs (
                chunk_db_id, tenant_id, mailbox_id, chunk_id, extraction_input_hash
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, mailbox_id, chunk_id, extraction_input_hash) DO NOTHING
            RETURNING {_JOB_COLUMNS}
            """,
            (state.chunk_db_id, state.tenant_id, state.mailbox_id, state.chunk_id, digest),
        ).fetchone()
        return _row_to_job(row)

    def enqueue_message(self, message_id, *, tenant_id, mailbox_id, schema_version, prompt_version, model_id) -> int:
        rows = self.conn.execute(
            _CHUNK_STATE_SELECT + " WHERE c.message_id = %s AND c.tenant_id = %s AND c.mailbox_id = %s",
            (message_id, tenant_id, mailbox_id),
        ).fetchall()
        queued = 0
        for row in rows:
            if self.enqueue(
                _row_to_state(row),
                schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
            ):
                queued += 1
        return queued

    def claim_job(self, *, owner, lease_seconds=300) -> Optional[GraphJob]:
        with self.conn.transaction():
            candidate = self.conn.execute(
                "SELECT id FROM graph_extraction_jobs "
                "WHERE status = 'pending' OR (status = 'running' AND leased_until <= now()) "
                "ORDER BY id ASC FOR UPDATE SKIP LOCKED LIMIT 1"
            ).fetchone()
            if candidate is None:
                return None
            row = self.conn.execute(
                f"""
                UPDATE graph_extraction_jobs SET
                    status = 'running', attempts = attempts + 1, lease_owner = %s,
                    leased_until = now() + make_interval(secs => %s), updated_at = now()
                WHERE id = %s RETURNING {_JOB_COLUMNS}
                """,
                (owner, lease_seconds, candidate["id"]),
            ).fetchone()
        return _row_to_job(row)

    def get_job(self, job_id) -> Optional[GraphJob]:
        return _row_to_job(
            self.conn.execute(
                f"SELECT {_JOB_COLUMNS} FROM graph_extraction_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        )

    def load_chunk_state(self, chunk_db_id) -> Optional[ChunkGraphState]:
        return _row_to_state(
            self.conn.execute(_CHUNK_STATE_SELECT + " WHERE c.id = %s", (chunk_db_id,)).fetchone()
        )

    def commit_graph(
        self, job, *, resolved, method, extraction_version, schema_version, prompt_version, model_id
    ) -> bool:
        with self.conn.transaction():
            # FOR UPDATE: serialize against a concurrent re-ingest of this chunk
            # so the hash we check is the one we write against.
            row = self.conn.execute(
                _CHUNK_STATE_SELECT + " WHERE c.id = %s FOR UPDATE OF c", (job.chunk_db_id,)
            ).fetchone()
            state = _row_to_state(row)
            if state is None:
                self._retire(job.id)
                return False
            current = extraction_hash_of(
                state.as_extraction_input(),
                schema_version=schema_version, prompt_version=prompt_version, model_id=model_id,
            )
            if current != job.extraction_input_hash:
                self._retire(job.id)
                return False

            self._delete_chunk_graph(job.chunk_db_id)
            prov = (method, extraction_version, model_id)

            for m in resolved.mentions:
                entity_id = self._upsert_entity(state, m.entity_type, m.normalized_name, m.canonical_name)
                self.conn.execute(
                    """
                    INSERT INTO chunk_entity_mentions (
                        tenant_id, mailbox_id, chunk_db_id, chunk_id, entity_id, mention_text,
                        chunk_start, chunk_end, source_start, source_end,
                        extraction_method, extraction_version, extraction_model
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (tenant_id, mailbox_id, chunk_db_id, entity_id, chunk_start, chunk_end)
                    DO NOTHING
                    """,
                    (state.tenant_id, state.mailbox_id, state.chunk_db_id, state.chunk_id, entity_id,
                     m.mention_text, m.chunk_start, m.chunk_end, m.source_start, m.source_end, *prov),
                )

            for r in resolved.relations:
                subj_id = self._upsert_entity(state, r.subject_key[0], r.subject_key[1])
                obj_id = self._upsert_entity(state, r.object_key[0], r.object_key[1])
                self.conn.execute(
                    """
                    INSERT INTO relation_observations (
                        tenant_id, mailbox_id, subject_entity_id, predicate, object_entity_id,
                        chunk_db_id, chunk_id, chunk_start, chunk_end, mention_text, evidence_kind,
                        extraction_method, extraction_version, extraction_model
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (state.tenant_id, state.mailbox_id, subj_id, r.predicate, obj_id,
                     state.chunk_db_id, state.chunk_id, r.chunk_start, r.chunk_end, r.mention_text,
                     r.evidence_kind, *prov),
                )

            for f in resolved.facts:
                supersedes = None
                if f.has_update_cue:
                    supersedes = self._supersede_prior(state, f.normalized_subject, f.normalized_predicate)
                fact_id = self.conn.execute(
                    """
                    INSERT INTO facts (
                        tenant_id, mailbox_id, subject, predicate, object_value,
                        normalized_subject, normalized_predicate, status, supersedes_fact_id,
                        extraction_method, extraction_version, extraction_model
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,'active',%s,%s,%s,%s)
                    RETURNING fact_id
                    """,
                    (state.tenant_id, state.mailbox_id, f.subject, f.predicate, f.object_value,
                     f.normalized_subject, f.normalized_predicate, supersedes, *prov),
                ).fetchone()["fact_id"]
                self.conn.execute(
                    """
                    INSERT INTO fact_evidence (
                        fact_id, tenant_id, mailbox_id, chunk_db_id, chunk_id,
                        chunk_start, chunk_end, source_start, source_end, evidence_text, evidence_hash
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (fact_id, state.tenant_id, state.mailbox_id, state.chunk_db_id, state.chunk_id,
                     f.chunk_start, f.chunk_end, f.source_start, f.source_end, f.evidence_text, f.evidence_hash),
                )

            self.conn.execute(
                "UPDATE email_chunks SET graph_input_hash = %s, graph_extracted_at = now(), updated_at = now() "
                "WHERE id = %s",
                (job.extraction_input_hash, job.chunk_db_id),
            )
            self._retire(job.id)
        return True

    def _delete_chunk_graph(self, chunk_db_id: int) -> None:
        self.conn.execute("DELETE FROM chunk_entity_mentions WHERE chunk_db_id = %s", (chunk_db_id,))
        self.conn.execute("DELETE FROM relation_observations WHERE chunk_db_id = %s", (chunk_db_id,))
        # Facts whose evidence is this chunk. fact_evidence cascades; the
        # supersedes_fact_id self-FK is ON DELETE SET NULL, so no dangling links.
        self.conn.execute(
            "DELETE FROM facts WHERE fact_id IN (SELECT fact_id FROM fact_evidence WHERE chunk_db_id = %s)",
            (chunk_db_id,),
        )

    def _upsert_entity(self, state, entity_type, normalized_name, canonical_name=None) -> int:
        row = self.conn.execute(
            """
            INSERT INTO graph_entities (tenant_id, mailbox_id, entity_type, canonical_name, normalized_name)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, mailbox_id, entity_type, normalized_name)
            DO UPDATE SET updated_at = now()
            RETURNING entity_id
            """,
            (state.tenant_id, state.mailbox_id, entity_type, canonical_name or normalized_name, normalized_name),
        ).fetchone()
        return row["entity_id"]

    def _supersede_prior(self, state, normalized_subject, normalized_predicate) -> Optional[int]:
        row = self.conn.execute(
            """
            UPDATE facts SET status = 'superseded', updated_at = now()
            WHERE fact_id = (
                SELECT fact_id FROM facts
                WHERE tenant_id = %s AND mailbox_id = %s
                  AND normalized_subject = %s AND normalized_predicate = %s AND status = 'active'
                ORDER BY fact_id DESC LIMIT 1
            )
            RETURNING fact_id
            """,
            (state.tenant_id, state.mailbox_id, normalized_subject, normalized_predicate),
        ).fetchone()
        return row["fact_id"] if row else None

    def _retire(self, job_id) -> None:
        self.conn.execute(
            "UPDATE graph_extraction_jobs SET status = 'done', completed_at = now(), "
            "leased_until = NULL, lease_owner = NULL, last_error = NULL, error_rule = NULL, updated_at = now() "
            "WHERE id = %s",
            (job_id,),
        )

    def fail_job(self, job_id, error, *, error_rule=None, max_attempts=3) -> None:
        with self.conn.transaction():
            self.conn.execute(
                """
                UPDATE graph_extraction_jobs SET
                    status = CASE WHEN attempts >= %s THEN 'failed' ELSE 'pending' END,
                    last_error = %s, error_rule = %s, leased_until = NULL, lease_owner = NULL, updated_at = now()
                WHERE id = %s
                """,
                (max_attempts, error, error_rule, job_id),
            )

    def chunks_needing_graph(self, *, tenant_id, mailbox_id, limit=100, after_id=0) -> list[ChunkGraphState]:
        rows = self.conn.execute(
            _CHUNK_STATE_SELECT
            + """
            WHERE c.tenant_id = %s AND c.mailbox_id = %s
              AND c.graph_input_hash IS NULL AND c.id > %s
            ORDER BY c.id ASC LIMIT %s
            """,
            (tenant_id, mailbox_id, after_id, limit),
        ).fetchall()
        return [_row_to_state(row) for row in rows]

    # --- reads -----------------------------------------------------------
    def find_entity(self, *, tenant_id, mailbox_id, entity_type, name) -> Optional[dict]:
        from email_thread_rag.graph.extract import normalized_key

        return self.conn.execute(
            "SELECT entity_id, tenant_id, mailbox_id, entity_type, canonical_name, normalized_name "
            "FROM graph_entities WHERE tenant_id = %s AND mailbox_id = %s "
            "AND entity_type = %s AND normalized_name = %s",
            (tenant_id, mailbox_id, entity_type, normalized_key(name)),
        ).fetchone()

    def list_mentions(self, *, tenant_id, mailbox_id, entity_id) -> list[dict]:
        return self.conn.execute(
            """
            SELECT m.*, c.text AS clean_text
            FROM chunk_entity_mentions m JOIN email_chunks c ON c.id = m.chunk_db_id
            WHERE m.tenant_id = %s AND m.mailbox_id = %s AND m.entity_id = %s
            ORDER BY m.chunk_id, m.chunk_start
            """,
            (tenant_id, mailbox_id, entity_id),
        ).fetchall()

    def list_relations(self, *, tenant_id, mailbox_id, subject_entity_id=None) -> list[dict]:
        sql = (
            "SELECT r.*, s.canonical_name AS subject_name, o.canonical_name AS object_name "
            "FROM relation_observations r "
            "JOIN graph_entities s ON s.entity_id = r.subject_entity_id "
            "JOIN graph_entities o ON o.entity_id = r.object_entity_id "
            "WHERE r.tenant_id = %s AND r.mailbox_id = %s"
        )
        params = [tenant_id, mailbox_id]
        if subject_entity_id is not None:
            sql += " AND r.subject_entity_id = %s"
            params.append(subject_entity_id)
        sql += " ORDER BY r.id"
        return self.conn.execute(sql, tuple(params)).fetchall()

    def list_facts(self, *, tenant_id, mailbox_id, subject=None, status=None) -> list[dict]:
        from email_thread_rag.graph.extract import normalized_key

        sql = "SELECT * FROM facts WHERE tenant_id = %s AND mailbox_id = %s"
        params = [tenant_id, mailbox_id]
        if subject is not None:
            sql += " AND normalized_subject = %s"
            params.append(normalized_key(subject))
        if status is not None:
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY fact_id"
        facts = self.conn.execute(sql, tuple(params)).fetchall()
        for fact in facts:
            fact["evidence"] = self.conn.execute(
                """
                SELECT e.*, c.text AS clean_text
                FROM fact_evidence e JOIN email_chunks c ON c.id = e.chunk_db_id
                WHERE e.fact_id = %s ORDER BY e.id
                """,
                (fact["fact_id"],),
            ).fetchall()
        return facts

    def evidence_chunks(self, *, tenant_id, mailbox_id, chunk_ids) -> dict[str, str]:
        if not chunk_ids:
            return {}
        rows = self.conn.execute(
            "SELECT chunk_id, text FROM email_chunks "
            "WHERE tenant_id = %s AND mailbox_id = %s AND chunk_id = ANY(%s)",
            (tenant_id, mailbox_id, list(chunk_ids)),
        ).fetchall()
        return {row["chunk_id"]: row["text"] for row in rows}

    # --- Stage-6 planner reads (narrow, always tenant/mailbox scoped) -----
    def entities_matching(self, *, tenant_id, mailbox_id, names) -> list[dict]:
        from email_thread_rag.graph.extract import normalized_key

        keys = list({normalized_key(n) for n in names if n})
        if not keys:
            return []
        return self.conn.execute(
            "SELECT entity_id, tenant_id, mailbox_id, entity_type, canonical_name, normalized_name "
            "FROM graph_entities WHERE tenant_id = %s AND mailbox_id = %s "
            "AND normalized_name = ANY(%s) ORDER BY entity_id",
            (tenant_id, mailbox_id, keys),
        ).fetchall()

    def entity_evidence_chunk_ids(self, *, tenant_id, mailbox_id, entity_ids, limit=20) -> list[str]:
        if not entity_ids:
            return []
        # Mentions + relation endpoints (metadata edges included: they retrieve
        # related emails; the caller still cites the chunk's own clean text).
        rows = self.conn.execute(
            """
            SELECT chunk_id FROM (
                SELECT chunk_id FROM chunk_entity_mentions
                WHERE tenant_id = %(t)s AND mailbox_id = %(m)s AND entity_id = ANY(%(e)s)
                UNION
                SELECT chunk_id FROM relation_observations
                WHERE tenant_id = %(t)s AND mailbox_id = %(m)s
                  AND (subject_entity_id = ANY(%(e)s) OR object_entity_id = ANY(%(e)s))
            ) u
            ORDER BY chunk_id LIMIT %(l)s
            """,
            {"t": tenant_id, "m": mailbox_id, "e": list(entity_ids), "l": limit},
        ).fetchall()
        return [row["chunk_id"] for row in rows]

    def fact_evidence_chunk_ids(
        self, *, tenant_id, mailbox_id, subjects=None, status=None, as_of=None, limit=20
    ) -> list[str]:
        from email_thread_rag.graph.extract import normalized_key

        sql = [
            "SELECT DISTINCT e.chunk_id FROM facts f "
            "JOIN fact_evidence e ON e.fact_id = f.fact_id "
            "WHERE f.tenant_id = %s AND f.mailbox_id = %s"
        ]
        params: list = [tenant_id, mailbox_id]
        if subjects:
            keys = list({normalized_key(s) for s in subjects if s})
            if not keys:
                return []
            sql.append(" AND f.normalized_subject = ANY(%s)")
            params.append(keys)
        if status is not None:
            sql.append(" AND f.status = %s")
            params.append(status)
        if as_of is not None:
            # Undated facts are never historically valid.
            sql.append(" AND f.effective_date IS NOT NULL AND f.effective_date <= %s")
            params.append(as_of)
        sql.append(" ORDER BY e.chunk_id LIMIT %s")
        params.append(limit)
        rows = self.conn.execute("".join(sql), tuple(params)).fetchall()
        return [row["chunk_id"] for row in rows]


def build_store(conn) -> PostgresGraphStore:
    return PostgresGraphStore(conn)
