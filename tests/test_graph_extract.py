"""The evidence-grounding core: pure, no DB, no network, no model.

These are the tests that prove the spec's central promise -- every span maps
exactly to clean chunk.text, and anything the model says that cannot be located
verbatim is dropped.
"""

from __future__ import annotations

from email_thread_rag.graph.extract import (
    has_update_cue,
    locate_span,
    metadata_relations,
    normalize_name,
    normalized_key,
    resolve_extraction,
)
from email_thread_rag.graph.models import ChunkGraphState
from email_thread_rag.graph.prompt import LLMEntity, LLMExtraction, LLMFact, LLMRelation

TEXT = "Alice approved the Q3 budget of $1200 for Project Atlas."


def _state(text=TEXT, *, source_start=0, **kw):
    return ChunkGraphState(
        chunk_db_id=1, chunk_id="c-1", tenant_id="acme", mailbox_id="inbox",
        text=text, sender="alice@corp.com", source_start=source_start, **kw
    )


def _extraction(entities=None, relations=None, facts=None):
    return LLMExtraction(
        entities=[LLMEntity(**e) for e in (entities or [])],
        relations=[LLMRelation(**r) for r in (relations or [])],
        facts=[LLMFact(**f) for f in (facts or [])],
    )


# --- locate_span -------------------------------------------------------------
def test_locate_span_finds_exact_substring():
    start, end = locate_span(TEXT, "Q3 budget")
    assert TEXT[start:end] == "Q3 budget"


def test_locate_span_rejects_absent_or_altered_evidence():
    assert locate_span(TEXT, "Q4 budget") is None
    assert locate_span(TEXT, "") is None
    assert locate_span(TEXT, "alice") is None  # case-sensitive: not evidence


# --- entity resolution -------------------------------------------------------
def test_every_mention_span_maps_exactly_to_clean_text():
    resolved = resolve_extraction(
        _extraction(entities=[
            {"name": "Alice", "type": "PERSON", "evidence": "Alice"},
            {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"},
        ]),
        _state(),
    )
    assert len(resolved.mentions) == 2
    for m in resolved.mentions:
        assert TEXT[m.chunk_start:m.chunk_end] == m.mention_text


def test_hallucinated_entity_without_exact_evidence_is_dropped():
    resolved = resolve_extraction(
        _extraction(entities=[{"name": "Bob", "type": "PERSON", "evidence": "Bob signed off"}]),
        _state(),
    )
    assert resolved.mentions == ()


def test_authored_body_offsets_shift_by_chunk_source_start():
    # chunk.text is a slice of the authored body starting at offset 100.
    resolved = resolve_extraction(
        _extraction(entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]),
        _state(source_start=100),
    )
    m = resolved.mentions[0]
    assert (m.chunk_start, m.chunk_end) == (0, 5)
    assert (m.source_start, m.source_end) == (100, 105)


def test_no_source_offsets_when_chunk_has_no_body_offset():
    state = ChunkGraphState(chunk_db_id=1, chunk_id="c", tenant_id="a", mailbox_id="i", text=TEXT, source_start=None)
    m = resolve_extraction(_extraction(entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}]), state).mentions[0]
    assert m.source_start is None and m.source_end is None


# --- relation resolution -----------------------------------------------------
def test_relation_without_direct_evidence_is_discarded():
    resolved = resolve_extraction(
        _extraction(
            entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"},
                      {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}],
            relations=[{"subject": "Alice", "predicate": "WORKS_ON", "object": "Project Atlas",
                        "evidence": "Alice manages Atlas"}],  # not in text
        ),
        _state(),
    )
    assert all(r.evidence_kind == "metadata" for r in resolved.relations)  # only header edges survive


def test_relation_endpoint_must_be_a_grounded_entity():
    resolved = resolve_extraction(
        _extraction(
            entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"}],
            relations=[{"subject": "Alice", "predicate": "WORKS_ON", "object": "Project Atlas",
                        "evidence": "Project Atlas"}],  # object never extracted as an entity
        ),
        _state(),
    )
    assert [r for r in resolved.relations if r.evidence_kind == "text"] == []


def test_grounded_relation_survives_with_exact_span():
    resolved = resolve_extraction(
        _extraction(
            entities=[{"name": "Alice", "type": "PERSON", "evidence": "Alice"},
                      {"name": "Project Atlas", "type": "PROJECT", "evidence": "Project Atlas"}],
            relations=[{"subject": "Alice", "predicate": "WORKS_ON", "object": "Project Atlas",
                        "evidence": "Project Atlas"}],
        ),
        _state(),
    )
    text_rels = [r for r in resolved.relations if r.evidence_kind == "text"]
    assert len(text_rels) == 1
    r = text_rels[0]
    assert TEXT[r.chunk_start:r.chunk_end] == r.mention_text == "Project Atlas"


# --- metadata relations ------------------------------------------------------
def test_metadata_relations_come_from_headers_with_no_text_offsets():
    state = _state(recipients=["bob@corp.com"], cc=["carol@corp.com"],
                   in_reply_to="<parent@x>", parent_sender="dan@corp.com")
    _, rels = metadata_relations(state)
    kinds = {r.predicate for r in rels}
    assert kinds == {"SENT", "CC", "REPLY_TO"}
    for r in rels:
        assert r.evidence_kind == "metadata"
        assert r.chunk_start is None and r.chunk_end is None  # never a text span


def test_reply_to_only_when_parent_is_local():
    state = _state(in_reply_to="<parent@x>", parent_sender=None)  # parent not ingested
    _, rels = metadata_relations(state)
    assert all(r.predicate != "REPLY_TO" for r in rels)


# --- normalization -----------------------------------------------------------
def test_normalization_is_conservative():
    assert normalize_name("  Project   Atlas  ") == "Project Atlas"
    assert normalized_key("Project Atlas") == normalized_key("PROJECT  atlas")
    # NFKC folds compatibility forms (e.g. fullwidth) but does not fuzzy-merge.
    assert normalized_key("ACME") == "acme"
    assert normalized_key("Acme Corp") != normalized_key("Acme")


# --- update cue --------------------------------------------------------------
def test_update_cue_fires_only_on_explicit_wording():
    assert has_update_cue("The amount is now $1500")
    assert has_update_cue("This replaces the earlier figure")
    assert has_update_cue("updated from $1000")
    assert has_update_cue("$1500 instead of $1000")
    assert not has_update_cue("The known budget is $1000")   # 'now' inside 'known' must not fire
    assert not has_update_cue("The budget is $1200")
