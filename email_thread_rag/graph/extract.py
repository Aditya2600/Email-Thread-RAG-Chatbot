"""The evidence-grounding core. No DB, no network, no model -- pure functions.

This is where the spec's hardest rule lives: the model supplies evidence
*strings*; code locates each string in the chunk's own ``text`` and derives the
offsets itself, dropping anything it cannot locate exactly. Everything a test
needs to prove "every span maps exactly to clean chunk.text" is here.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

from email_thread_rag.graph.models import (
    ChunkGraphState,
    ResolvedFact,
    ResolvedGraph,
    ResolvedMention,
    ResolvedRelation,
)
from email_thread_rag.graph.prompt import LLMExtraction

# An explicit update/correction cue in the evidence text is the ONLY thing that
# may supersede a prior fact. A later email date alone never does. Word-bounded
# so "now" does not fire inside "known"/"nowhere".
UPDATE_CUE = re.compile(r"\b(replaces|replacing|updated from|instead of|now)\b", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """Conservative display normalization: NFKC + whitespace collapse. Preserves
    case -- this is the canonical form shown to a human."""
    collapsed = " ".join(unicodedata.normalize("NFKC", name).split())
    return collapsed


def normalized_key(name: str) -> str:
    """The casefolded uniqueness key. Conservative: no fuzzy merging, no token
    reordering, no stemming -- only Unicode normalization + case folding."""
    return normalize_name(name).casefold()


def evidence_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def locate_span(text: str, evidence: str) -> Optional[tuple[int, int]]:
    """Return (start, end) of the first exact occurrence of ``evidence`` in
    ``text``, or None. Case-sensitive and contiguous on purpose: an approximate
    match is not evidence."""
    if not evidence:
        return None
    idx = text.find(evidence)
    if idx < 0:
        return None
    return idx, idx + len(evidence)


def _source_offsets(state: ChunkGraphState, chunk_start: int, chunk_end: int) -> tuple[Optional[int], Optional[int]]:
    """Map chunk-relative offsets to authored-body offsets when the chunk carries
    one. chunk.text is a contiguous slice of the authored body starting at
    ``source_start``, so the body offset is a simple shift."""
    if state.source_start is None:
        return None, None
    return state.source_start + chunk_start, state.source_start + chunk_end


def has_update_cue(evidence_text: str) -> bool:
    return bool(UPDATE_CUE.search(evidence_text))


def metadata_relations(state: ChunkGraphState) -> tuple[list[ResolvedMention], list[ResolvedRelation]]:
    """Deterministic PERSON entities + SENT/CC/REPLY_TO relations from safe
    headers. No text evidence: these are metadata, offsets stay None, and the
    schema CHECK forbids them ever being read as authored-text proof.

    Returns metadata person entities as mentions-without-spans is *not* allowed
    (a mention requires a span), so these entities are surfaced only through the
    relations that reference them; the store upserts an entity per relation
    endpoint. We therefore return an empty mention list and the relations.
    """
    relations: list[ResolvedRelation] = []
    if not state.sender:
        return [], []
    sender_key = ("PERSON", normalized_key(state.sender))

    def person_key(addr: str) -> tuple[str, str]:
        return ("PERSON", normalized_key(addr))

    for recipient in state.recipients:
        if recipient:
            relations.append(
                ResolvedRelation(sender_key, "SENT", person_key(recipient), evidence_kind="metadata")
            )
    for cc in state.cc:
        if cc:
            relations.append(
                ResolvedRelation(sender_key, "CC", person_key(cc), evidence_kind="metadata")
            )
    # Deterministic thread link: this sender replied to the parent's sender, but
    # only when the parent is locally available -- never fetched.
    if state.in_reply_to and state.parent_sender:
        relations.append(
            ResolvedRelation(sender_key, "REPLY_TO", person_key(state.parent_sender), evidence_kind="metadata")
        )
    return [], relations


def _entity_canonicals(mentions: list[ResolvedMention]) -> dict[tuple[str, str], str]:
    """First-seen display name per entity key; used to name relation endpoints
    that came from the semantic extraction."""
    canonicals: dict[tuple[str, str], str] = {}
    for m in mentions:
        canonicals.setdefault(m.entity_key, m.canonical_name)
    return canonicals


def resolve_extraction(extraction: LLMExtraction, state: ChunkGraphState) -> ResolvedGraph:
    """Turn validated-but-untrusted LLM output into evidence-checked graph rows.

    Drops any entity/relation/fact whose evidence string is not an exact span of
    ``state.text``. Drops any relation whose subject or object is not itself an
    extracted, located entity (a relation with no grounded endpoints is a
    hallucinated edge). Metadata relations are added deterministically.
    """
    text = state.text

    mentions: list[ResolvedMention] = []
    entity_keys: set[tuple[str, str]] = set()
    for ent in extraction.entities:
        span = locate_span(text, ent.evidence)
        if span is None:
            continue  # no exact evidence -> not a mention
        start, end = span
        src_start, src_end = _source_offsets(state, start, end)
        canonical = normalize_name(ent.name)
        key = (ent.type, normalized_key(ent.name))
        mentions.append(
            ResolvedMention(
                entity_type=ent.type,
                canonical_name=canonical,
                normalized_name=key[1],
                mention_text=text[start:end],
                chunk_start=start,
                chunk_end=end,
                source_start=src_start,
                source_end=src_end,
            )
        )
        entity_keys.add(key)

    canonicals = _entity_canonicals(mentions)

    relations: list[ResolvedRelation] = []
    for rel in extraction.relations:
        span = locate_span(text, rel.evidence)
        if span is None:
            continue  # relation without direct evidence is discarded
        subj_key = _match_entity_key(rel.subject, entity_keys, canonicals)
        obj_key = _match_entity_key(rel.object, entity_keys, canonicals)
        if subj_key is None or obj_key is None:
            continue  # endpoint is not a grounded entity -> hallucinated edge
        start, end = span
        relations.append(
            ResolvedRelation(
                subject_key=subj_key,
                predicate=rel.predicate,
                object_key=obj_key,
                evidence_kind="text",
                mention_text=text[start:end],
                chunk_start=start,
                chunk_end=end,
            )
        )

    # Metadata relations reference PERSON entities that may not have text
    # mentions; the store upserts them. They are appended after the text edges.
    _, meta_rels = metadata_relations(state)
    relations.extend(meta_rels)

    facts: list[ResolvedFact] = []
    for fact in extraction.facts:
        span = locate_span(text, fact.evidence)
        if span is None:
            continue  # a fact must trace to an exact span
        start, end = span
        src_start, src_end = _source_offsets(state, start, end)
        ev_text = text[start:end]
        facts.append(
            ResolvedFact(
                subject=normalize_name(fact.subject),
                predicate=fact.predicate,
                object_value=fact.object,
                normalized_subject=normalized_key(fact.subject),
                normalized_predicate=fact.predicate.casefold(),
                evidence_text=ev_text,
                chunk_start=start,
                chunk_end=end,
                source_start=src_start,
                source_end=src_end,
                evidence_hash=evidence_hash(ev_text),
                has_update_cue=has_update_cue(ev_text),
            )
        )

    return ResolvedGraph(mentions=tuple(mentions), relations=tuple(relations), facts=tuple(facts))


def _match_entity_key(
    name: str, entity_keys: set[tuple[str, str]], canonicals: dict[tuple[str, str], str]
) -> Optional[tuple[str, str]]:
    """A relation endpoint names an entity by name only (no type). Match it to an
    extracted entity by normalized name, whatever its type."""
    target = normalized_key(name)
    for key in entity_keys:
        if key[1] == target:
            return key
    return None
