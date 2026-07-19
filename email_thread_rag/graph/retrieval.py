"""Stage-6 graph-evidence resolution: a ``RetrievalPlan`` -> ordered evidence
chunk ids, using only the narrow GraphStore read methods.

Pure and store-driven (no psycopg here, no SQL), so it runs against the
in-memory store in the fast suite and the Postgres store in integration with the
same code. Every id it returns is a real email chunk reached through a mention,
a relation, or fact evidence -- never a synthesized fact string. Tenant/mailbox
scope is threaded into every store call. If it yields nothing, the caller falls
back to hybrid retrieval.
"""

from __future__ import annotations

from email_thread_rag.rag.planner import RetrievalPlan, RetrievalRoute


def collect_graph_chunk_ids(store, plan: RetrievalPlan) -> list[str]:
    """Ordered, de-duplicated evidence chunk ids for the plan's graph routes.

    Order is deterministic (each store method returns chunk ids sorted by id);
    first-seen wins across routes so a chunk found by several routes appears
    once, preserving which route surfaced it first.
    """
    if not plan.uses_graph:
        return []

    scope = {"tenant_id": plan.tenant_id, "mailbox_id": plan.mailbox_id}
    ordered: list[str] = []
    seen: set[str] = set()

    def extend(chunk_ids) -> None:
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                ordered.append(chunk_id)

    for route in plan.routes:
        if route is RetrievalRoute.GRAPH_ENTITY and plan.entity_terms:
            entities = store.entities_matching(names=list(plan.entity_terms), **scope)
            entity_ids = [entity["entity_id"] for entity in entities]
            if entity_ids:
                extend(
                    store.entity_evidence_chunk_ids(
                        entity_ids=entity_ids, limit=plan.graph_candidate_limit, **scope
                    )
                )
            # Facts asserted *about* the named entities (subject match).
            extend(
                store.fact_evidence_chunk_ids(
                    subjects=list(plan.entity_terms), limit=plan.graph_candidate_limit, **scope
                )
            )
        elif route is RetrievalRoute.GRAPH_CURRENT:
            extend(
                store.fact_evidence_chunk_ids(
                    subjects=list(plan.subject_terms) or None,
                    status="active",
                    limit=plan.temporal_candidate_limit,
                    **scope,
                )
            )
        elif route is RetrievalRoute.GRAPH_AS_OF and plan.as_of is not None:
            extend(
                store.fact_evidence_chunk_ids(
                    subjects=list(plan.subject_terms) or None,
                    as_of=plan.as_of,
                    limit=plan.temporal_candidate_limit,
                    **scope,
                )
            )
    return ordered
