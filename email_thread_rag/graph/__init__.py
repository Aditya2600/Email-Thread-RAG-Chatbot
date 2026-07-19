"""Stage 5: evidence-backed entity, relation, and temporal-fact extraction.

Same shape as Stage 4's ``context`` package -- a job queue, a provider seam, a
deterministic validator, an in-memory store and a Postgres store held to one
contract, a worker, an enqueue seam, and a backfill. The difference is what is
produced: not one retrieval prefix, but graph rows, each traced to an exact span
of a chunk's immutable ``text``.

Nothing here is wired into the answer path or the query router. Graph results
are built and queryable; using them to answer is a later stage.
"""
