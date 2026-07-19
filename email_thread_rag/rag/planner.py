"""Stage-6 deterministic query planner.

Decides *how* to retrieve evidence for a query -- purely from the query string
and tenant/mailbox scope. No LLM, no embeddings, no spaCy, no network: routing
is regex + literal token rules so it is trivially reproducible and inspectable.
This module never touches the database or the graph package; it only produces a
typed ``RetrievalPlan`` that the ParadeDB retriever executes.

Intents (a plan may select several; HYBRID is always present as the fallback):
- HYBRID        -- existing BM25 + dense hybrid retrieval.
- GRAPH_ENTITY  -- the query names entities; pull their mentions/relations/facts.
- GRAPH_CURRENT -- "current/latest/now/updated"; pull only *active* facts.
- GRAPH_AS_OF   -- an explicit, unambiguous "as of <date>"; pull dated facts at
                   or before that date. A date alone never implies supersession
                   (Stage-5's explicit-cue rule is preserved) and an undated
                   fact is never treated as historically valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional


class RetrievalRoute(str, Enum):
    HYBRID = "hybrid"
    GRAPH_ENTITY = "graph_entity"
    GRAPH_CURRENT = "graph_current"
    GRAPH_AS_OF = "graph_as_of"


@dataclass(frozen=True)
class RetrievalPlan:
    """The typed, inspectable output of planning. Every field is data, not code:
    debug/trace surfaces read ``routes``/``rules``/``fallback`` directly."""

    routes: tuple[RetrievalRoute, ...]
    tenant_id: str
    mailbox_id: str
    thread_id: Optional[str] = None
    entity_terms: tuple[str, ...] = ()
    subject_terms: tuple[str, ...] = ()
    as_of: Optional[date] = None
    graph_candidate_limit: int = 20
    temporal_candidate_limit: int = 10
    # Deterministic reason labels for why each route was (not) chosen.
    rules: tuple[str, ...] = ()
    # The branch used when a graph route resolves to zero citable chunks.
    fallback: str = "hybrid"

    @property
    def uses_graph(self) -> bool:
        return any(route is not RetrievalRoute.HYBRID for route in self.routes)


# --- deterministic lexical signals -------------------------------------------

# Explicit "current state" cues. Word-bounded so "now" never fires inside
# "known"/"nowhere" (same discipline as Stage-5's UPDATE_CUE).
_CURRENT_CUE = re.compile(
    r"\b(current|currently|latest|now|newest|present|replaced|replacing|updated|up[\s-]?to[\s-]?date)\b",
    re.IGNORECASE,
)

# "as of <date>" up to the next sentence break. Only an unambiguous, named-month
# or ISO date is accepted below; anything fuzzy ("as of last quarter") parses to
# None and the AS_OF route is not selected.
_AS_OF = re.compile(r"\bas of\s+(.+?)\s*(?:[?!]|$)", re.IGNORECASE)
_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y")

_QUOTED = re.compile(r'["“]([^"”]+)["”]|\'([^\']+)\'')
# A capitalized token run: proper nouns, "Project Atlas", "Q3 Budget", "Acme".
_CAP_SEQ = re.compile(r"\b([A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*)*)\b")
_TOKEN = re.compile(r"[A-Za-z0-9$][A-Za-z0-9$&.\-]*")

# Question words / verbs / mail nouns that are never entities or fact subjects.
_STOPWORDS = frozenset(
    """a an the of for to in on at by is are was were be been do does did done have has had
    what who whom whose when where why how which that this these those it its and or but with
    as from about into over under between than then so if not no yes
    list show tell give get find me my our your their his her please can could should would
    will shall may might must need want know say said
    email emails message messages mail thread threads""".split()
)


def _dedup_ci(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _parse_as_of(query: str) -> Optional[date]:
    match = _AS_OF.search(query)
    if not match:
        return None
    raw = match.group(1).strip().rstrip(".")
    tokens = raw.split()
    # Try the whole phrase, then a 3-token window ("January 5, 2026" pulled out
    # of a longer tail), then the first token (an ISO date). Ambiguous numeric
    # forms like 03/05/2026 are deliberately unsupported.
    candidates = [raw]
    if len(tokens) >= 3:
        candidates.append(" ".join(tokens[:3]))
    if tokens:
        candidates.append(tokens[0])
    for candidate in candidates:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    return None


def _entity_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    for match in _QUOTED.finditer(query):
        phrase = (match.group(1) or match.group(2) or "").strip()
        if phrase:
            terms.append(phrase)
    for match in _CAP_SEQ.finditer(query):
        words = match.group(1).split()
        # Drop a leading sentence-initial stopword ("Who approved Atlas" -> "Atlas").
        while words and words[0].casefold() in _STOPWORDS:
            words.pop(0)
        if not words:
            continue
        if len(words) == 1 and words[0].casefold() in _STOPWORDS:
            continue
        terms.append(" ".join(words))
    return tuple(_dedup_ci(terms))


def _subject_terms(query: str, *, n_max: int = 3, cap: int = 16) -> tuple[str, ...]:
    tokens = [m.group(0) for m in _TOKEN.finditer(query.casefold())]
    content = [t for t in tokens if t not in _STOPWORDS]
    grams: list[str] = []
    for n in range(1, n_max + 1):
        for i in range(len(content) - n + 1):
            grams.append(" ".join(content[i : i + n]))
    return tuple(list(dict.fromkeys(grams))[:cap])


def plan_query(
    query: str,
    *,
    tenant_id: str,
    mailbox_id: str,
    thread_id: Optional[str] = None,
    settings=None,
) -> RetrievalPlan:
    """Deterministically route a query. Duck-typed on ``settings`` (only reads
    a few graph_* attributes) so it stays trivially importable and testable."""
    enabled = getattr(settings, "graph_planner_enabled", True) if settings is not None else True
    graph_limit = getattr(settings, "graph_candidate_limit", 20) if settings is not None else 20
    temporal_limit = getattr(settings, "graph_temporal_candidate_limit", 10) if settings is not None else 10

    if not enabled:
        return RetrievalPlan(
            routes=(RetrievalRoute.HYBRID,),
            tenant_id=tenant_id,
            mailbox_id=mailbox_id,
            thread_id=thread_id,
            rules=("planner_disabled",),
        )

    entity_terms = _entity_terms(query)
    subject_terms = _subject_terms(query)

    routes: list[RetrievalRoute] = [RetrievalRoute.HYBRID]
    rules: list[str] = []

    as_of = _parse_as_of(query)
    if as_of is not None:
        # An explicit historical date takes priority over a "current" cue.
        routes.append(RetrievalRoute.GRAPH_AS_OF)
        rules.append(f"as_of_date_parsed:{as_of.isoformat()}")
    elif _CURRENT_CUE.search(query):
        routes.append(RetrievalRoute.GRAPH_CURRENT)
        rules.append("temporal_current_cue")

    if entity_terms:
        routes.append(RetrievalRoute.GRAPH_ENTITY)
        rules.append("entity_terms_present")

    if len(routes) == 1:
        rules.append("no_graph_signal_hybrid_only")

    return RetrievalPlan(
        routes=tuple(routes),
        tenant_id=tenant_id,
        mailbox_id=mailbox_id,
        thread_id=thread_id,
        entity_terms=entity_terms,
        subject_terms=subject_terms,
        as_of=as_of,
        graph_candidate_limit=graph_limit,
        temporal_candidate_limit=temporal_limit,
        rules=tuple(rules),
        fallback="hybrid",
    )
