"""The deterministic query planner: pure routing rules, no LLM/embeddings/network.

Named ``test_graph_*`` so conftest's socket guard fails this test if planning
ever opens a network connection -- concrete proof of "the planner makes no
LLM/provider/network call" beyond the AST import check below.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

from email_thread_rag.config import Settings
from email_thread_rag.rag.planner import RetrievalRoute, plan_query

SCOPE = {"tenant_id": "acme", "mailbox_id": "inbox"}


def _routes(query, **kw):
    return set(plan_query(query, **SCOPE, **kw).routes)


# 1. A generic query is hybrid-only.
def test_generic_query_is_hybrid_only():
    plan = plan_query("summarize the discussion about pricing", **SCOPE)
    assert plan.routes == (RetrievalRoute.HYBRID,)
    assert "no_graph_signal_hybrid_only" in plan.rules
    assert plan.uses_graph is False


# 2. A named-entity / relationship query selects the graph entity route plus hybrid.
def test_entity_query_selects_graph_entity_and_hybrid():
    plan = plan_query("who works on Project Atlas?", **SCOPE)
    assert RetrievalRoute.HYBRID in plan.routes
    assert RetrievalRoute.GRAPH_ENTITY in plan.routes
    assert "Project Atlas" in plan.entity_terms
    # Sentence-initial "Who" is never treated as an entity.
    assert not any(term.casefold() == "who" for term in plan.entity_terms)


def test_quoted_phrase_becomes_an_entity_term():
    plan = plan_query('find the "acme supplies" invoice', **SCOPE)
    assert "acme supplies" in plan.entity_terms
    assert RetrievalRoute.GRAPH_ENTITY in plan.routes


# current/latest route
def test_current_cue_selects_temporal_current_route():
    plan = plan_query("what is the current budget?", **SCOPE)
    assert RetrievalRoute.GRAPH_CURRENT in plan.routes
    assert "temporal_current_cue" in plan.rules
    assert "budget" in plan.subject_terms


def test_current_cue_is_word_bounded():
    # "known" must not trip the "now" cue.
    plan = plan_query("what is the known budget", **SCOPE)
    assert RetrievalRoute.GRAPH_CURRENT not in plan.routes


# 6. Explicit, unambiguous "as of <date>" only.
def test_as_of_parses_unambiguous_dates():
    for query, expected in [
        ("what was the budget as of 2026-03-01?", date(2026, 3, 1)),
        ("the amount as of January 5, 2026", date(2026, 1, 5)),
        ("the amount as of 5 January 2026", date(2026, 1, 5)),
    ]:
        plan = plan_query(query, **SCOPE)
        assert plan.as_of == expected
        assert RetrievalRoute.GRAPH_AS_OF in plan.routes


def test_ambiguous_or_absent_date_does_not_select_as_of():
    for query in ["the budget as of last quarter", "the budget as of Q3", "the current budget"]:
        plan = plan_query(query, **SCOPE)
        assert plan.as_of is None
        assert RetrievalRoute.GRAPH_AS_OF not in plan.routes


def test_as_of_takes_priority_over_current_cue():
    # An explicit historical date is historical, not "current".
    plan = plan_query("what was the budget as of 2026-03-01 now?", **SCOPE)
    assert RetrievalRoute.GRAPH_AS_OF in plan.routes
    assert RetrievalRoute.GRAPH_CURRENT not in plan.routes


# planner can be disabled -> always hybrid (preserves existing deployments).
def test_disabled_planner_is_always_hybrid():
    settings = Settings(graph_planner_enabled=False)
    plan = plan_query("who works on Project Atlas?", **SCOPE, settings=settings)
    assert plan.routes == (RetrievalRoute.HYBRID,)
    assert "planner_disabled" in plan.rules


def test_limits_come_from_settings():
    settings = Settings(graph_candidate_limit=7, graph_temporal_candidate_limit=3)
    plan = plan_query("current Project Atlas budget", **SCOPE, settings=settings)
    assert plan.graph_candidate_limit == 7
    assert plan.temporal_candidate_limit == 3


# 12. The planner module imports nothing that could reach a model or the network.
def test_planner_module_has_no_model_or_network_imports():
    source = (Path(__file__).resolve().parent.parent / "email_thread_rag" / "rag" / "planner.py").read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"httpx", "openai", "psycopg", "torch", "sentence_transformers", "transformers", "spacy", "socket"}
    assert forbidden.isdisjoint(imported), f"planner must stay pure, found {forbidden & imported}"
