# Architecture: Gmail-Native Email RAG (Level 4 — Graph-Augmented)

> Companion to [Plan_Upgrad.md](Plan_Upgrad.md). That plan takes the Enron-slice
> system from Level 2 → Level 3 (pgvector + HyDE + Medha generation + Self-RAG).
> This document does two things:
>
> 1. **Reviews** the 8-layer Gmail architecture and records what is solid vs. missing.
> 2. **Adds the advanced RAG methods** that email specifically needs — chiefly a
>    **Knowledge Graph (GraphRAG)** layer, plus Contextual Retrieval, hierarchical
>    (RAPTOR) summaries, temporal/state reasoning, and an agentic retrieval router.
>
> Design constraint carried over from the plan: **one store** (ParadeDB Postgres).
> Every addition below is designed to live in that same Postgres, not a new service.

---

## 0. Why email needs more than vector + BM25

Plain hybrid retrieval (dense + lexical + rerank) answers *"find me the passage that
looks like the question."* That is necessary but insufficient for a mailbox, because
the hardest and most common email questions are **relational, temporal, and
aggregative**, not semantic-similarity:

| Question type | Example | What plain RAG misses |
|---|---|---|
| **Relational / multi-hop** | "Who did Sarah loop in about the Q3 budget?" | The answer is an *edge* (CC/forwarded-to), not a similar passage. |
| **Aggregative** | "What are all the open action items assigned to me?" | Spans dozens of threads; no single chunk contains the answer. |
| **Temporal / state** | "What's the **current** ship date?" | Three emails give three dates; only the latest is valid. Cosine similarity treats them equally. |
| **Thematic / global** | "Summarize everything about the Acme acquisition." | Topic is spread across 40 threads; top-k of any single retriever truncates it. |
| **Entity-centric** | "How is Bob connected to the vendor deal?" | The answer is a *path* through people/orgs/projects. |

These five classes are exactly what a **knowledge graph + hierarchical summaries +
temporal edges** solve, and why this document treats GraphRAG as a required layer
rather than a nice-to-have.

---

## 1. Review of the 8-layer Gmail architecture

The layered design (Ingestion → Chunking → Dual-index → Retrieval → Query-understanding
→ Generation → Self-RAG → API/UI) is **sound and stays as the backbone.** Verdict per layer:

| Layer | Status | Note |
|---|---|---|
| 1 — Gmail ingestion | ✅ Keep | OAuth + History API delta sync is correct. Add the `users.watch` push channel for near-real-time. |
| 2 — Chunking | ⚠️ Upgrade | Header injection is good but not enough. Add **Contextual Retrieval** (§4.2) and **quote-stripping** as first-class. |
| 3 — Dual index | ✅ Keep | pgvector + pg_search in ParadeDB. **Add a 3rd index: the graph** (§3) and a 4th: thread/topic summaries (§4.3). |
| 4 — Retrieval | ⚠️ Upgrade | Coarse→fine is right. Promote it to an **agentic router** (§4.5) that also queries the graph and summary tiers, not just vector+BM25. |
| 5 — Query understanding | ⚠️ Build | The `query_planner.py` stub becomes the router brain (intent → which retrievers → fuse). |
| 6 — Medha generation | ✅ Keep | Grounded generation with inline `[msg:id]` citations. Extend citations to graph facts (§3.5). |
| 7 — Self-RAG | ✅ Keep | ISREL/ISSUP/ISUSE + correction loop. Graph gives a **second grounding source** for ISSUP. |
| 8 — API/UI | ✅ Keep | Add a graph/trace panel so relational answers are explainable. |

**Gaps the original diagram did not cover** (now addressed below):

- No **relational/multi-hop** capability → §3 Knowledge Graph.
- No **global/thematic** capability (top-k truncates broad questions) → §4.3 RAPTOR + §3.6 communities.
- No **temporal supersession** (latest decision wins) → §4.4.
- Chunking loses cross-chunk context → §4.2 Contextual Retrieval.
- Retrieval is a fixed pipeline, not query-adaptive → §4.5 agentic router.

---

## 2. Target architecture (Level 4)

```
                              ┌───────────────────────────────────────────────┐
                              │                  GMAIL                         │
                              │   OAuth2 · History delta sync · users.watch    │
                              └───────────────────────┬───────────────────────┘
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ INGEST & PARSE                                                                        │
│  thread tree (In-Reply-To/References) · quote strip · attachment OCR · MIME normalize │
└───────────┬───────────────────────────────────────────────────────────┬──────────────┘
            ▼                                                             ▼
┌───────────────────────────────┐                       ┌───────────────────────────────┐
│ CHUNK + ENRICH                 │                       │ EXTRACT (LLM + spaCy)          │
│  header injection              │                       │  entities: Person/Org/Project/ │
│  + CONTEXTUAL RETRIEVAL (4.2)  │                       │    Topic/Doc/Meeting/Commitment│
│  raw_text  +  embed_text       │                       │  relations: SENT/CC/MENTIONS/  │
└───────────┬───────────────────┘                       │    COMMITTED/DECIDED/SUPERSEDES│
            ▼                                             └───────────────┬───────────────┘
   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  ▼
   │ embed (BGE 768/1024)│  │ summarize tiers    │  │                  │ ┌───────────────┐
   └────────┬───────────┘  │  msg→thread→topic  │  │                  │ │ resolve+dedupe│
            ▼              │  (RAPTOR, 4.3)     │  │                  │ │ entities      │
            ▼              └─────────┬──────────┘  │                  │ └───────┬───────┘
┌─────────────────────────────────────────────────────────────────────────────▼─────────┐
│ ONE STORE — ParadeDB Postgres                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ ┌──────────────────────────┐    │
│  │ pgvector     │ │ pg_search    │ │ summaries       │ │ GRAPH (nodes + edges)    │    │
│  │ HNSW dense   │ │ BM25 lexical │ │ thread / topic  │ │ recursive-CTE  or  AGE   │    │
│  │ chunks.embed │ │ chunks.text  │ │ embeddings too  │ │ temporal · weighted      │    │
│  └──────┬───────┘ └──────┬───────┘ └────────┬────────┘ └────────────┬─────────────┘    │
└─────────┼────────────────┼──────────────────┼───────────────────────┼──────────────────┘
          │                │                  │                       │
          └────────┬───────┴─────────┬────────┴───────────┬───────────┘
                   ▼                 ▼                     ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ AGENTIC RETRIEVAL ROUTER (4.5)  ← query_planner.py             │
        │  classify intent → pick retrievers → run → RRF fuse → rerank   │
        │   semantic→vector · keyword→BM25 · relational→graph            │
        │   thematic→topic-summaries · metadata→SQL · temporal→recency   │
        └───────────────────────────────┬───────────────────────────────┘
                                         ▼
                    ┌──────────────────────────────────────┐
                    │ ISREL filter → MEDHA grounded answer  │
                    │  inline [msg:id] + [graph:fact] cites │
                    │  → ISSUP / ISUSE → correction loop     │
                    └──────────────────┬───────────────────┘
                                       ▼
                        ┌────────────────────────────────┐
                        │ API (SSE stream) · UI · traces │
                        └────────────────────────────────┘
```

The backbone is unchanged from the 8-layer design. The **new spine** is the right-hand
*Extract → resolve → Graph* path and the two new index tiers (summaries, graph) inside
the single Postgres, all funnelling into an **agentic router** instead of a fixed pipeline.

---

## 3. Knowledge Graph / GraphRAG (the primary addition)

### 3.1 Why a graph, and why it fits email

Email is a social/relational medium: every message *is already* a small graph
(`sender —SENT→ message —TO/CC→ recipients`, `message —MENTIONS→ project`). Building
that graph explicitly turns the four question classes plain RAG can't answer
(relational, aggregative, temporal, thematic) into cheap graph traversals. Microsoft's
GraphRAG showed graph + community summaries beats vector RAG on *global* questions; the
relational structure of email makes the win even larger than on generic prose.

You already have the hooks: [entity_score.py](../src/memory/entity_score.py) (empty),
[thread_summary.py](../src/memory/thread_summary.py), and `spaCy` in requirements.

### 3.2 Node & edge schema

**Node types**

| Node | Source | Key attributes |
|---|---|---|
| `Person` | From/To/Cc, signatures | canonical_email, display_names[], org |
| `Org` | email domains, NER | domain, name |
| `Message` | every email | message_id, thread_id, date, subject |
| `Thread` | thread tree | thread_id, subject, participant_count |
| `Project`/`Topic` | LLM+NER on body | name, aliases[] |
| `Document` | attachments | filename, mime, sha |
| `Meeting`/`Event` | dates+calendar cues | when, attendees |
| `Commitment`/`ActionItem` | LLM extraction | text, owner→Person, due, status |
| `Decision` | LLM extraction | text, made_by, date, supersedes? |

**Edge types** (directed, time-stamped, weighted)

```
Person  -SENT->        Message          Message -MENTIONS->     Project|Person|Org
Message -TO|CC|BCC->   Person           Message -ABOUT->        Topic
Message -REPLY_TO->    Message          Person  -COMMITTED_TO-> ActionItem
Message -ATTACHES->    Document         ActionItem -ASSIGNED_TO-> Person
Person  -WORKS_AT->    Org              Decision -SUPERSEDES->   Decision   (temporal!)
Message -DECIDES->     Decision         Thread  -ABOUT->        Topic
```

Every edge carries `{ts, source_message_id, confidence}` so the graph is **auditable**
(every fact traces back to a citable message) and **temporal** (§4.4).

### 3.3 Extraction pipeline (where graph facts come from)

Run at ingest, per message, fused from three cheap-to-expensive sources:

1. **Structural (free, 100% precision):** headers already give
   `SENT / TO / CC / REPLY_TO / ATTACHES`. No model needed.
2. **spaCy NER (fast):** `PERSON / ORG / DATE / GPE` → candidate Person/Org/Date nodes
   and `MENTIONS` edges.
3. **Medha extraction (rich):** one prompt per message (or per thread) returns JSON:
   `{commitments[], decisions[], action_items[], topics[], superseded_by}`. Use
   **prompt caching** on the system/instructions block so this stays cheap at corpus scale.

Then **entity resolution / dedupe** (this is what [entity_score.py](../src/memory/entity_score.py)
should hold): merge `bob@acme.com` / `Bob Smith` / `Robert Smith` into one `Person`
node via email-canonicalization + name similarity + co-occurrence. Bad resolution is the
#1 way KGs rot, so gate merges with a confidence threshold and keep an alias list.

### 3.4 Storage — keep it in the one Postgres

You committed to a single ParadeDB store. Two ways to hold the graph there; **start with A.**

**A. Relational edge tables + recursive CTEs (default — zero new infra)**

```sql
CREATE TABLE kg_nodes (
  node_id     uuid PRIMARY KEY,
  user_id     uuid,                 -- multi-tenant isolation
  kind        text,                 -- 'person'|'org'|'project'|'commitment'|...
  canonical   text,                 -- canonical_email / normalized name
  attrs       jsonb,
  embedding   vector(768)           -- OPTIONAL: lets the graph be vector-searched too
);
CREATE TABLE kg_edges (
  src         uuid REFERENCES kg_nodes,
  dst         uuid REFERENCES kg_nodes,
  rel         text,                 -- 'sent'|'cc'|'mentions'|'committed_to'|'supersedes'
  ts          timestamptz,
  message_id  text,                 -- provenance → citation
  weight      real DEFAULT 1.0,
  confidence  real DEFAULT 1.0
);
CREATE INDEX ON kg_edges (src, rel);
CREATE INDEX ON kg_edges (dst, rel);
```

Multi-hop traversal is a `WITH RECURSIVE` query bounded to 2–3 hops. This covers ~90% of
email graph questions and needs **nothing beyond the Postgres you already run.**

**B. Apache AGE (openCypher in Postgres) — only if traversals get complex**

AGE gives real `MATCH (a)-[:CC*1..3]->(b)` Cypher. Caveat to verify before committing:
ParadeDB ships pg_search+pgvector; adding the AGE extension may need a **custom image**
(stacking three C extensions). Don't take that on until recursive CTEs prove insufficient.
Neo4j as a separate service is the last resort — it breaks the one-store rule and adds a
sync problem, so avoid unless graph workloads dominate.

> **Recommendation:** ship **A** in Phase 1, keep the schema AGE-compatible (node/edge with
> `kind`/`rel`) so a later swap to **B** is mechanical, not a rewrite.

### 3.5 How the graph participates in an answer

The graph is **both a retriever and a grounding source**:

- **As retriever:** the router (§4.5) issues a graph query, gets back a set of
  `message_id`s / nodes, and those messages' chunks enter the fusion pool alongside
  vector/BM25 hits. *Graph finds the right messages; vector/BM25 find the right passages.*
- **As facts:** structured results (e.g., a list of action items with owners/dates) are
  packed into the Medha context as a compact table and cited as `[graph: action_item #7,
  src msg:abc]`. Every graph fact is provenance-linked, so Self-RAG's **ISSUP** can verify
  it against the source message exactly like a text citation.

### 3.6 Community summaries (global questions)

Cluster the graph (Leiden/Louvain on the people-topic subgraph, or just group by
`Topic`/`Thread`) and have Medha write a **community summary** per cluster. "Summarize
everything about the Acme acquisition" is then answered from a handful of pre-computed
summaries instead of trying to cram 40 threads into 14k tokens. This is the GraphRAG
*global search* mode and overlaps cleanly with the RAPTOR topic tier (§4.3) — build them
as the same summary table.

---

## 4. Other advanced methods (priority-ordered)

### 4.1 Priority matrix

| Method | Solves | Effort | Priority |
|---|---|---|---|
| **Knowledge Graph (§3)** | relational, aggregative, entity-centric | M–L | **P0 — must** |
| **Contextual Retrieval (§4.2)** | chunks lose surrounding meaning | S | **P0 — must** |
| **Agentic router (§4.5)** | one pipeline can't serve all intents | M | **P1 — high** |
| **RAPTOR hierarchy (§4.3)** | thematic/global; long threads | M | **P1 — high** |
| **Temporal / state (§4.4)** | "current" vs stale facts | S–M | **P1 — high** |
| **ColBERT late-interaction (§4.6)** | precise term/entity matching | L | P2 — optional |

### 4.2 Contextual Retrieval (Anthropic) — cheapest big win

Before embedding each chunk, prepend a 1–2 sentence, LLM-generated description situating
it in its thread: *"This is from the May 3 reply by Bob in the 'Q3 budget' thread; it
revises the number first proposed on Apr 28."* Anthropic reported **~35–49% fewer
retrieval failures** from this + contextual BM25. For email it's natural because the
thread already supplies the context — generate it once per chunk with Medha (cache the
thread summary as the cache prefix). This *extends* the header-injection already in the
plan: header gives metadata; contextual prefix gives **discourse position**. Store it in
`embed_text`, keep `text` clean for display.

### 4.3 RAPTOR — hierarchical thread/topic summaries

Build a summary tree and **index every tier as its own embeddings** (reuse the summaries
table from §3.6):

```
chunk  ──►  message summary  ──►  thread summary  ──►  topic/community summary
(detail)                                              (global)
```

Retrieval can then match at the altitude the question needs: a specific question hits
chunks; "what was decided in this thread" hits the thread summary; "what's the state of
the merger" hits a topic summary. Long Gmail threads (50+ messages) especially benefit —
you retrieve one thread summary instead of 50 competing chunks.

### 4.4 Temporal & state-aware retrieval

Email facts **expire**. Two mechanisms:

1. **Recency-aware fusion:** add a mild recency prior to the fused score
   (`score *= exp(-λ·age)`) so newer evidence wins ties — tunable, off for archival queries.
2. **Supersession edges:** when Medha extraction detects a `Decision`/`Commitment` that
   replaces an earlier one, write a `SUPERSEDES` edge (§3.2). "Current ship date" then =
   the `Decision` node with no outgoing `SUPERSEDES`. This is the difference between
   "here are 3 dates people mentioned" and "the date is **June 14** (was May 30, moved
   on Jun 2 by Bob)."

### 4.5 Agentic retrieval router — give `query_planner.py` a brain

Replace the fixed coarse→fine pipeline with a classifier that **picks retrievers per query**:

```
classify(query) ──► intent
   metadata_lookup  ("emails from Bob in Jan")      → SQL WHERE only (no LLM)
   semantic         ("concerns about the vendor")    → vector (+HyDE) + BM25
   relational       ("who approved the budget")      → GRAPH traverse → msgs → vector
   aggregative      ("all my open action items")     → GRAPH aggregate (no generation guesswork)
   thematic/global  ("summarize the Acme deal")      → topic/community summaries
   temporal         ("current plan")                 → semantic + recency prior + supersession
  ──► run chosen retrievers in parallel ──► RRF fuse ──► cross-encoder rerank ──► ISREL
```

For hard questions, allow **decomposition**: "Did the vendor agree to the terms Sarah
proposed?" → (1) graph: find Sarah's proposal message → (2) semantic: find vendor's reply
→ (3) compare. Bound it to 1–2 steps to stay within latency/cost. This router is also the
natural place to enforce the **abstain** path you already have.

### 4.6 ColBERT / late interaction (optional)

Token-level (multi-vector) matching beats single-vector embeddings on exact entity/term
queries ("the *Henderson* contract") — common in email. Cost: bigger index, more compute,
and it doesn't live natively in pgvector. Treat as a P2 experiment *after* the graph and
contextual retrieval land; measure on the eval set (§ evals in the plan) before adopting.

---

## 5. Unified retrieval & fusion (how it all comes together)

```
                         query
                           │
                  ┌────────▼────────┐
                  │  ROUTER (4.5)   │ intent + filters (sender/date) + decomposition
                  └───┬───┬───┬───┬─┘
        ┌─────────────┘   │   │   └──────────────┐
        ▼                 ▼   ▼                  ▼
   vector (+HyDE)     BM25   GRAPH traverse   topic/thread summaries
   pgvector HNSW    pg_search  (CTE/AGE)       (RAPTOR + communities)
        │                 │   │                  │
        │                 │   └─► message_ids ───┤  (graph feeds messages into the pool)
        └────────┬────────┴──────────┬───────────┘
                 ▼                                 recency prior (4.4) ─┐
            RRF fusion  (reuse rag/fusion.py)  ◄──────────────────────┘
                 ▼
         cross-encoder rerank  (reuse rag/reranker.py)
                 ▼
            ISREL filter ──► Medha grounded answer ──► ISSUP/ISUSE ──► correct/abstain
```

Key principle: **the graph routes to the right *messages*; vector/BM25 find the right
*passages* inside them; summaries answer when no single passage suffices.** All paths
converge on the same RRF→rerank→Self-RAG tail you already built — the additions are new
*sources*, not a new pipeline.

---

## 6. Data-model additions (on top of the plan's `chunks` table)

```
chunks         (from Plan §Phase1) + context_prefix text   -- Contextual Retrieval (4.2)
summaries      summary_id, user_id, tier('message'|'thread'|'topic'),
               ref_id, text, embedding vector, member_ids[], created_from_msgs[]
kg_nodes       node_id, user_id, kind, canonical, attrs jsonb, embedding vector  -- §3.4
kg_edges       src, dst, rel, ts, message_id, weight, confidence                 -- §3.4
```

All four carry `user_id` for the multi-tenant Gmail case, and all reference real
`message_id`s so **every answer — text, summary, or graph fact — is citable.**

---

## 7. Implementation phasing (slots into Plan_Upgrad's Phase 0–5)

| Plan phase | Add from this doc |
|---|---|
| Phase 0 — package & green tests | *(no change — get baseline runnable first)* |
| Phase 1 — pgvector + pg_search | also create `kg_nodes`/`kg_edges`/`summaries` tables + structural-edge ingest (free graph: SENT/TO/CC/REPLY). |
| Phase 2 — header-injected chunks | **+ Contextual Retrieval prefix (4.2)** in the same `embed_text` build. |
| Phase 3 — Medha client | **+ extraction prompt (3.3)** reusing the client; **+ summary generation (4.3)**. |
| Phase 4 — Self-RAG | **+ agentic router (4.5)** in `query_planner.py`; **+ graph as ISSUP source (3.5)**; **+ temporal/supersession (4.4)**. |
| Phase 5 — evals | add **relational, aggregative, temporal, thematic** question sets — these are exactly what the new methods target, so they prove the upgrade. |
| (new) Phase 6 | optional: community detection (3.6), ColBERT experiment (4.6), AGE migration if CTEs strain. |

**Gmail-specific work** (replaces the Enron loader, orthogonal to the above): OAuth2,
`users.history.list` delta sync with stored `historyId`, `users.watch` Pub/Sub push for
real-time, per-user rate-limit handling, and `email-reply-parser` for quote stripping.

---

## 8. Review findings & risks (honest list)

- **Entity resolution is the make-or-break.** A graph with `Bob Smith` ≠ `bob@acme.com`
  as two nodes is worse than no graph. Budget real effort for [entity_score.py](../src/memory/entity_score.py)
  and keep it gated by confidence + alias lists.
- **Extraction cost at scale.** One Medha call per message is expensive over a full
  mailbox. Mitigate: prompt-cache the instruction block, batch per thread, and run
  extraction **async/incremental** (it's not on the query hot path).
- **AGE + ParadeDB stacking is unverified.** Don't assume it; ship recursive-CTE graph
  first (§3.4-A), keep schema portable.
- **ColBERT/late-interaction is the lowest ROI here** — list it, gate it behind eval
  numbers, don't build it speculatively.
- **Graph can hallucinate via bad edges.** Because every edge stores `message_id` +
  `confidence`, surface low-confidence facts as "likely" and let Self-RAG's ISSUP verify
  against the source — never present an unverified graph fact as ground truth.
- **Don't let the graph silently cap coverage.** If a traversal is bounded (top-N edges,
  2-hop limit), say so in the trace; a truncated graph answer that looks complete is the
  dangerous failure mode.

---

### TL;DR

Keep the 8-layer hybrid backbone. Add, in priority order: **(1) a Knowledge Graph in the
same Postgres** (structural edges free at ingest, LLM/NER edges async) to unlock
relational/aggregative/temporal/thematic questions; **(2) Contextual Retrieval** for a
cheap retrieval-quality jump; **(3) an agentic router** so each query hits the right
source; **(4) RAPTOR summaries + (5) temporal supersession** for global and "current-state"
answers. ColBERT stays optional behind eval numbers. Everything stays in one store and
stays citable.

---

# Upgrade Plan

Plan: Upgrade Inbox-Copilot to a Level-3 Email-Thread RAG
Context
inbox-copilot is an email-thread RAG chatbot (Enron slice + attachments). Today it is a Level-2 system: hybrid BM25 + dense retrieval, RRF fusion, cross-encoder reranking, and a deterministic citation validator — but answers are template/rule-based (no LLM), vectors live in FAISS pickle files, there is no HyDE and no self-correction.

The goal is Level 3: persistent pgvector storage, HyDE, header-injected chunks for sender/date-aware search, a Self-RAG–inspired verification layer, and real LLM generation via the remote Medha vLLM service (Google Gemma 26B-A4B AWQ-4bit, OpenAI-compatible, on an A100). Endpoint verified working: http://164.52.192.196:8001/v1 (model Medha, 14336 ctx, streaming).

Current repo is mid-refactor and not runnable
The working tree has two parallel trees. src/ is a half-done restructure that broke the package: the email_thread_rag import root has no packaging (empty pyproject.toml), src/ingest/chunker.py is truncated, src/retrieval/hyde.py + src/rag/grounding.py + src/rag/citations.py + src/db/* are empty stubs, and src/app/schemas.py was rewritten into names that diverge from what the engine and tests import.

Key recovery insight: HEAD (13b1d95) still contains a complete, coherent, test-backed generation at the repo root — app/, rag/, scripts/, ui/, config.py — whose module names (rag.chunking, rag.answer, rag.vector_index, rag.bm25_index, rag.fusion, rag.reranker, rag.retrieval, rag.rewrite, rag.engine) and schema names (ChunkRecord, RetrievalHit, RetrievalMetrics, TraceRecord, ClauseValidation, AskResponse) exactly match the imports in tests/ and rag/engine.py. The 19 tests in tests/ are the executable spec.

So we rebuild the baseline from HEAD's coherent modules (not from the broken src/ stubs), package it properly, get tests green, then layer the five features on top.

Decisions (confirmed with user)
Scope: full repair to runnable + all features.
Vector + BM25 store: single ParadeDB Postgres container (paradedb/paradedb) — bundles pgvector (dense) and pg_search (real BM25). One store, fused with RRF.
Postgres: runs as a docker-compose service.
Answers: Medha grounded generation with enforced inline citations; deterministic builder kept as abstain fallback.
Phase 0 — Consolidate to one installable package (make it run)
Package layout. Create email_thread_rag/ as the single canonical package and add a real pyproject.toml (setuptools; deps from requirements.txt + sqlalchemy[asyncio], asyncpg, pgvector, httpx, sentence-transformers, spacy). Install editable: pip install -e ..
Recover coherent modules from HEAD, not the broken src/ stubs: app/ (schemas.py, main.py, sessions.py, streaming.py), rag/ (chunking.py, answer.py, vector_index.py, bm25_index.py, fusion.py, reranker.py, retrieval.py, rewrite.py, engine.py, citation_validator.py, corpus.py, memory.py, parse_*.py, utils.py, enron_archive.py, threading.py), config.py, scripts/ (build_dataset_slice.py, ingest_corpus.py, run_demo.py), ui/app.py. Source via git show HEAD:<path>.
Salvage selectively from src/ only where clearly better and working (e.g. UI polish in src/ui/gradio_app.py); otherwise prefer HEAD. Delete the broken duplicate src/ and the stale root dupes once recovered, ending the two-tree confusion.
Green the baseline: pytest passes against the recovered Level-2 contract before adding features. The tests/conftest.py fixtures (memory backend: HashingEncoder + in-memory HybridRetriever) must keep working without a live DB — this constrains every later change.
Gate: do not start Phase 1+ until pip install -e . && pytest is green.

Phase 1 — pgvector + pg_search store (ParadeDB)
docker-compose: add a db service paradedb/paradedb:latest with POSTGRES_*, a named volume, and a healthcheck; wire DATABASE_URL into api/ui services and into Settings. Update infra/docker-compose.prod.yml similarly. (Also fix the existing compose, which still references email_thread_rag.scripts... — now correct.)
email_thread_rag/db/ (new): engine.py (async SQLAlchemy engine / asyncpg pool from DATABASE_URL), schema.sql + a migration that runs CREATE EXTENSION IF NOT EXISTS vector; and CREATE EXTENSION IF NOT EXISTS pg_search;, creates the chunks table, an HNSW index (vector_cosine_ops) on embedding, and a bm25 index over embed_text (+ sender, subject) via pg_search with key_field='chunk_id'.
chunks table: chunk_id PK, doc_id, thread_id, message_id, kind, sender, recipients, date timestamptz, subject, attachment_name, page_no, text, embed_text, embedding vector(768), token_count, ocr_used, source_path, source_type, metadata jsonb.
Embeddings: computed locally with sentence-transformers BAAI/bge-base-en-v1.5 (768-dim, matches config.embed_model). Medha is chat-only — no embeddings endpoint.
Pluggable retrieval backend. Extend rag/retrieval.py:HybridRetriever with backend="memory"|"postgres":
memory (unchanged): bm25_index (rank-bm25) + VectorIndex/HashingEncoder — keeps conftest.py and unit tests DB-free.
postgres (new, production): dense via pgvector (embedding <=> :qvec cosine, thread-filtered in SQL) + lexical via pg_search (embed_text @@@ :q, paradedb.score(chunk_id), thread-filtered) → RRF-fuse in app (reuse rag/fusion.py) → cross-encoder rerank (reuse rag/reranker.py).
Ingest: scripts/ingest_corpus.py / rag/corpus.py:ingest_corpus upsert rows (with embedding + embed_text) into Postgres by chunk_id, in addition to the existing chunks.jsonl (kept for the memory/test path).
Phase 2 — Header-injected chunks (sender/date-aware)
In rag/chunking.py: build embed_text = header + "\n\n" + body and store both text (raw body, for display/citation) and embed_text (embedded and BM25-indexed). Header = From / To / Cc / Date / Subject / Thread-ID. Fix the cc=to bug seen in the src draft (src/ingest/chunker.py:207 copies email.to into Cc).
Attachments: header = parent email's From/Date/Subject + Attachment: <filename> (page N) then page text. Token-window chunking with overlap (reuse rag/utils.sliding_text_chunks, count_tokens).
Add embed_text field to ChunkRecord (additive, defaults to text so existing tests/fixtures stay valid).
Optional precision path: the query planner extracts structured filters (sender, date range) → SQL WHERE predicates; header injection covers the soft/semantic path.
Phase 3 — Medha LLM client + grounded answers
email_thread_rag/llm/medha.py (new): async OpenAI-compatible client (httpx) for /v1/chat/completions with complete() (non-stream) and stream() (SSE). Config: medha_base_url (http://164.52.192.196:8001/v1), medha_api_key (env MEDHA_API_KEY, default to the provided token), medha_model="Medha", max_tokens, temperature (0.2), timeout, ctx cap 14336. Centralized retry/timeout/error handling.
rag/rewrite.py: route query rewrite through Medha (drop T5/Gemini); keep RewriteResult shape and the rule-based fallback (tests use RuleOnlyRewriter).
rag/answer.py: add a grounded LLM path — prompt Medha with the packed, citation-tagged context to produce an answer with inline [msg:<id>] / [msg:<id>, page:N] citations; stream tokens through existing app/streaming.py. Keep the deterministic AnswerBuilder as fallback/abstain. Respect token_budget against the 14336 ctx (header injection inflates context).
Config: replace ollama_* knobs with medha_*; update .env.example and compose env.
Phase 4 — Self-RAG–inspired verification layer
Implement email_thread_rag/rag/grounding.py (currently empty) as reflection on top of the existing CitationValidator. Engine flow becomes: plan → retrieve → ISREL filter → generate → ISSUP/validate → ISUSE → (bounded) correct → finalize.

ISREL (relevance): after retrieval, a cheap batched Medha call judges each top chunk's relevance; drop irrelevant chunks before generation and feed the signal into the existing outside-thread fallback decision.
ISSUP (support): keep the deterministic CitationValidator (token-F1 + entity match, 0.70 coverage gate) as the fast gate, and add a Medha verifier labeling each answer sentence Supported/Partial/Unsupported against its cited chunks; strip or flag unsupported sentences.
ISUSE (usefulness): Medha rates 1–5 whether the answer addresses the question.
Correction loop (max 1–2 iters): if coverage/support below threshold → (a) trigger the existing outside-thread re-retrieval (engine._outside_thread_reason thresholds in config.RetrievalThresholds), or (b) regenerate with stricter "state only what citations support" instruction, else abstain with the existing message.
Surface reflection scores in RetrievalMetrics/TraceRecord and the Gradio debug panel.
Phase 5 — Eval harness (substantiates "Level 3")
evals/ is empty. Add a small labeled question set over the Enron slice + scripts/run_eval.py reporting: retrieval hit@k, citation coverage, support/faithfulness (Medha-as-judge), and abstain correctness. Lets us show before/after numbers for the upgrade.
Critical files
Area	Files
Packaging	pyproject.toml (new), recover email_thread_rag/** from HEAD, delete broken src/
pgvector/pg_search	email_thread_rag/db/{engine,schema.sql,migrations} (new), docker-compose.yml, infra/docker-compose.prod.yml
Retrieval	rag/retrieval.py (pluggable backend), reuse rag/{fusion,reranker,bm25_index,vector_index}.py
HyDE	rag/hyde.py (new), wired into rag/retrieval.py
Chunking	rag/chunking.py, app/schemas.py (ChunkRecord.embed_text)
Medha LLM	email_thread_rag/llm/medha.py (new), rag/rewrite.py, rag/answer.py, config.py, .env.example
Self-RAG	rag/grounding.py (new), rag/citation_validator.py, rag/engine.py, ui/app.py
Evals	evals/*, scripts/run_eval.py (new)
Reused (do not rewrite)
RRF: rag/fusion.py · Cross-encoder rerank: rag/reranker.py · BM25 (memory path): rag/bm25_index.py · Deterministic grounding: rag/citation_validator.py · Chunk tokenization: rag/utils.py · Memory: rag/memory.py · Streaming SSE: app/streaming.py · Outside-thread thresholds: config.RetrievalThresholds + engine._outside_thread_reason.
HyDE detail
rag/hyde.py:generate_hypothetical(query, *, llm, thread_context) -> str: Medha drafts a short hypothetical email/answer passage → embed with bge → use as (or blend 0.5/0.5 with) the raw query embedding for the dense branch. Config enable_hyde, hyde_max_tokens; skip for metadata_lookup intent; cache per query.

Verification
Baseline gate: pip install -e . && pytest green on the recovered Level-2 contract (memory backend, no DB).
DB up: docker compose up -d db; confirm vector + pg_search extensions and chunks indexes created.
Ingest: run scripts/ingest_corpus.py → rows present in Postgres with non-null embedding and header-injected embed_text.
Retrieval (postgres backend): a sender/date query (e.g. "what did Bob send in January about the budget") returns thread-correct chunks; verify both BM25 (paradedb.score) and pgvector contribute via the trace's fused_ranking.
HyDE on/off: toggle enable_hyde; confirm dense hits and final answer change and HyDE text appears in the trace.
Medha generation: /ask returns an LLM answer with inline [msg:...] citations; streaming works through app/streaming.py; curl health on /v1/models as a smoke check.
Self-RAG: craft an unsupported-claim query → verify ISSUP strips/flags it, the correction loop fires (re-retrieve or stricter regen), and ISREL/ISSUP/ISUSE scores show in the Gradio debug panel and TraceRecord.
Evals: scripts/run_eval.py prints hit@k, citation coverage, faithfulness, abstain-correctness over the labeled set.
Full pytest green again after features (memory path unaffected; add DB-backed tests guarded by DATABASE_URL).