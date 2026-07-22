# Inbox Copilot — Architecture

Deep technical companion to the [project README](../README.md). The README is the
overview, demo, and quick start; this document is the *why* behind the design — the
contracts, failure-mode reasoning, and the full-system diagrams. It doubles as an
**interview walkthrough** (see §14).

---

## 1. The one design rule

> **Index enriched retrieval text, but cite only exact authored email evidence.**

Every indexed chunk carries two representations:

- **`text`** — exact newly authored email (or PDF-page) content. The *only* thing
  shown to a user and the *only* thing allowed to satisfy a citation.
- **`embed_text`** — compact headers + safe deterministic/LLM context + that same
  exact text. The *only* thing sent to the vector and lexical indexes.

This separation lets headers, summaries, quoted history, and LLM-generated context
improve *recall* without ever masquerading as *evidence*. A prefix can help find an
email; it can never become the citation.

---

## 2. Full system architecture

### 2.1 Background ingestion & indexing

```
┌──────────────────────────────┐
│ Gmail / Enron Email Source   │
│ OAuth2 · watch · Pub/Sub     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ SYNC + INGEST                                      Stage 3    │
│ webhook → durable sync job → immediate ACK → worker           │
│ history.list → messages.get → idempotent upsert               │
│ retry · historyId cursor · full-sync fallback · thread links  │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ MIME + EMAIL NORMALIZATION                         Stage 1    │
│ MIME/HTML → normalized plain text                             │
│ authored_text | quoted_text | signature | disclaimer          │
│ sender | To | Cc | subject | date | message/thread IDs        │
└──────────────┬───────────────────────────────────────────────┘
               │
       ┌───────┴─────────────────────────┐
       ▼                                 ▼
┌────────────────────────────┐  ┌──────────────────────────────┐
│ EMAIL-AWARE CHUNKER        │  │ PDF ATTACHMENT WORKER         │
│ Stage 1                    │  │ Stage 8                       │
│ Paragraph/list boundaries  │  │ Fetch PDF bytes async         │
│ 300–450 tokens             │  │ Native text per page          │
│ 40–60 token overlap        │  │ OCR only when required        │
│ Only authored text indexed │  │ Page-level chunks             │
└──────────────┬─────────────┘  └──────────────┬───────────────┘
               └───────────────┬───────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ CITATION-SAFE CHUNK RECORD                                    │
│ text       = exact email body or PDF-page evidence            │
│ embed_text = compact headers + context prefix + text          │
│ source_start/source_end = provenance offsets                  │
│ attachment chunks = filename + page + OCR/native label        │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ INITIAL INDEXING                                   Stage 2    │
│ GTE ModernBERT embeds embed_text → 768-dim vector             │
│ pgvector HNSW indexes vectors                                 │
│ pg_search BM25 indexes embed_text                             │
│ Original text remains unchanged for citations                 │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ PARADEDB / POSTGRES  — one store                              │
│ mailboxes · messages · threads · chunks · vector(768)         │
│ attachments · sync/extraction jobs · context jobs             │
│ graph entities/relations/facts · tenant/mailbox ownership     │
└──────────────┬───────────────────────────────────────────────┘
               │ asynchronous enrichment
       ┌───────┴──────────────────────────┐
       ▼                                  ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ LLM CONTEXTUALIZER          │  │ GRAPH EXTRACTOR             │
│ Stage 4                     │  │ Stage 5                     │
│ subject/sender/thread/text  │  │ entities · relations        │
│ → short factual prefix      │  │ evidence-backed facts       │
│ retrieval metadata only     │  │ exact source quote required │
│ never changes chunk.text    │  │ no unsupported graph edge   │
└──────────────┬──────────────┘  └──────────────┬──────────────┘
               ▼                                ▼
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ RE-EMBED + REINDEX          │  │ GRAPH TABLES                │
│ headers + context + text    │  │ facts point back to the     │
│ → GTE 768-dim embedding     │  │ original evidence chunks    │
│ update HNSW + BM25 state    │  │ graph is a retrieval cue    │
└──────────────┬──────────────┘  └──────────────┬──────────────┘
               └────────────────┬───────────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Search-ready mailbox │
                     └──────────────────────┘
```

Reading order that matters:

1. Embedding happens *outside* the database (it computes vectors); HNSW/BM25 live
   *inside* Postgres because they index persisted rows.
2. Contextualization, facts, and attachments are **asynchronous enrichments** — they
   never block Gmail sync and are retryable/backfillable.
3. The answer path always returns to the original leaf `text` before citing.

### 2.2 Query-time retrieval

```
┌──────────────────────────────┐
│ Question · "What was Q3?"     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ AUTHORIZATION + SCOPE                                         │
│ tenant_id · mailbox_id · session — every branch same filters  │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ DETERMINISTIC QUERY PLANNER                        Stage 6    │
│ detect: keyword | semantic | metadata | relationship          │
│         current/latest | historical "as of"                   │
│ select bounded retrieval branches; no planner LLM             │
└───────┬───────────────────────┬───────────────────────┬──────┘
        ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ STRUCTURED       │   │ DENSE SEARCH     │   │ LEXICAL SEARCH   │
│ sender/date/     │   │ GTE 768-dim      │   │ BM25 over        │
│ subject filters  │   │ → HNSW search    │   │ embed_text       │
│ graph + temporal │   │ semantic matches │   │ names/amounts/   │
│ facts → chunks   │   │                  │   │ IDs/filenames    │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
         └──────────────────────┼──────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────┐
│ CANDIDATE FUSION                                              │
│ Reciprocal Rank Fusion → branch weighting → deduplication     │
│ graph facts resolved back to original email/PDF chunks        │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│ CROSS-ENCODER RERANKER → BOUNDED EVIDENCE PACK                │
│ exact chunk.text + chunk/message ID + source offsets          │
│ sender/subject/date = metadata only, never proof              │
│ attachments = filename + page + OCR label                     │
└──────────────┬───────────────────────────────────────────────┘
               ▼
      ┌────────────────────────────────────────────┐
      │ GROUNDED ANSWER INFERENCE          Stage 7  │
      │ evidence wrapped as untrusted email data     │
      │ strict JSON claims + verbatim quotes         │
      └──────────────┬───────────────────────────────┘
                     ▼
      ┌────────────────────────────────────────────┐
      │ LOCAL CITATION VALIDATOR                     │
      │ chunk ID in evidence · quote is substring    │
      │ offset matches · every factual claim cited   │
      │ metadata is never evidence                   │
      └───────┬──────────────────────┬───────────────┘
           Valid                  Invalid
              ▼                      ▼
    ┌──────────────────┐   ┌──────────────────────┐
    │ GROUNDED ANSWER  │   │ ONE WIDER RETRIEVAL   │
    │ exact citations  │   │ retry once, bounded   │
    └──────────────────┘   └──────────┬───────────┘
                                      ▼  still invalid
                              ┌──────────────────┐
                              │ SAFE ABSTENTION  │
                              └──────────────────┘
```

---

## 3. Core contracts

### 3.1 Normalized message

| Field | Purpose |
|---|---|
| `message_id`, `thread_id` | Stable Gmail identity + thread grouping. |
| `in_reply_to`, `references` | Reply-chain reconstruction. |
| `sender`, `recipients`, `cc` | Searchable metadata. **Cc stored independently from To.** |
| `subject`, `sent_at` | Thread + temporal context. |
| `authored_text` | Newly authored normalized body. |
| `quoted_text`, `signature_text`, `disclaimer_text` | Audit only; excluded from retrieval. |
| `attachments` | Metadata; PDF content extracted in Stage 8. |

Source offsets index into normalized `authored_text`, never raw MIME/HTML — so spans
stay stable after decoding and cleanup.

### 3.2 Chunk contract

| Field | Meaning | Citable? |
|---|---|---|
| `text` | Exact authored evidence, no injected headers. | **Yes** |
| `embed_text` | Headers + safe context + exact text; search input. | No |
| `source_start`, `source_end` | Span within `authored_text`. | Validates citation |
| `content_hash` | Hash of exact text + inputs. | Idempotency |
| `context_prefix` | Deterministic or LLM retrieval help. | No |
| `context_method`, `context_version` | none / deterministic / llm; version. | Audit |
| `embedding_model`, `embedding_version` | Vector identity. | Audit / reindex |
| `enrichment_status` | pending / ready / failed / stale / deleted. | Ops |

### 3.3 What `embed_text` actually looks like

```
From: Bob <bob@acme.com>
To: Sarah <sarah@acme.com>
Cc: Finance <finance@acme.com>
Date: 2026-05-03
Subject: Q3 Budget
Thread-ID: thread_123

The approved budget is now $120,000.
```

The header block lifts recall for people/date/subject/thread queries. The **visible
citation stays only**: `The approved budget is now $120,000.`

---

## 4. Why the EmailAwareChunker exists

Fixed-window splitting treats email as an ordinary document and fails: reply chains
repeat old messages, signatures add stray names/phones, and one thread holds
superseded decisions. Indexing a quoted old budget lets a query "hit" text the
current sender never wrote.

Order of operations:

1. Normalize MIME/HTML → stable plain text.
2. Conservatively segment quoted history, disclaimers, signatures away from authored body.
3. Short authored content → one independently citable chunk.
4. Long → pack complete paragraphs/lists toward ~300–450 tokens, ~40–60 overlap only when needed.
5. Store exact authored text + its source span.
6. Build `embed_text` from compact metadata + that exact text.
7. Send **only** `embed_text` to indexing.

Segmentation is deliberately conservative — it can miss unmarked bottom-posted quotes,
delimiter-less signatures, novel disclaimers. Measure quote/signature leakage in eval;
don't claim perfect stripping. A quote-only email (pure forward) legitimately produces
zero chunks rather than resurrecting quoted history.

---

## 5. Gmail sync — correct delivery semantics

The webhook must be fast, durable, idempotent:

```
1. Verify Pub/Sub auth + mailbox ownership.
2. In one transaction, create or coalesce a durable sync_job.
3. Commit the job.
4. Return HTTP 200 (ACK) — only now.
5. Worker calls history.list from the persisted cursor.
6. Per affected message: messages.get → idempotent upsert.
7. Handle messageAdded / messageDeleted / labelAdded / labelRemoved.
8. Advance historyId only after every page + derived write commits.
```

Why commit before ACK: if you ACK first and crash before storing, the notification is
lost. Duplicate jobs are safe — DB uniqueness + content hashes make processing idempotent.

**Cursor rules that keep sync correct:**

- **A failure never advances it.** Gmail/DB error or crash leaves the old cursor; the
  retry replays the same window — safe because every message upserts by ID.
- **It never rewinds.** Commits use numeric `GREATEST`; history IDs live in
  `numeric(20,0)` and compare numerically. As strings, `'10' < '9'` would silently
  rewind and skip mail.
- **Watch renewal doesn't touch it.** A watch expires after 7 days and fails silently;
  renew daily. Re-watch seeds the cursor only when NULL.
- **Stale-cursor 404 → bounded full sync.** Mark `needs_full_sync` *first*, take the
  history checkpoint *before* scanning (after would lose changes made during the scan),
  then replay.
- **Deletes are real deletes.** A `messageDeleted` record removes the message + its
  chunks outright — a tombstone would linger in HNSW/BM25 and every query would have to
  remember to exclude it.

The `sync_jobs` table **is** the queue — no Celery/Redis/Kafka. Workers claim with
`FOR UPDATE SKIP LOCKED` + a lease, so many run at once and a dead worker's job is
reclaimed when its lease expires. At most one pending job per mailbox (partial unique
index); a second notification raises the pending job's `requested_history_id` instead of
queueing duplicate work.

---

## 6. ParadeDB — one store, hybrid retrieval

Postgres/ParadeDB is the system of record. No separate vector DB until real scale
justifies it. It bundles `pgvector` (HNSW dense) and `pg_search` (real BM25) in one
transactional database.

| Area | Rows |
|---|---|
| Mailbox core | mailboxes, threads, messages, attachments, sync_cursors, sync_jobs |
| Search | chunks, embeddings, chunk metadata, lexical + vector indexes |
| Enrichment | context jobs, extraction runs, summary runs |
| Evidence | entities, relations, facts, fact_evidence |
| Safety | authorization/audit, deletion state, retention |

**Why BM25 *and* dense.** BM25 rewards exact terms — names, dollar amounts, filenames,
reference numbers — a bag-of-embeddings model blurs. Dense rewards paraphrase and
topical closeness lexical matching misses. Email needs both ("find this exact number"
*and* "find the email about this").

**Why RRF, not raw scores.** A BM25 score and a cosine distance live on unrelated
scales; adding them is meaningless. Reciprocal Rank Fusion converts each branch to a
rank and combines `weight / (k + rank)` — comparable regardless of score distribution.

**Embedding dimension is pinned at 768** (GTE ModernBERT + the deterministic
`HashingEncoder` test fallback share it). Changing the *model* at fixed width still
needs a full re-embed: equal dimension is not a compatible vector space, and mixing two
models' vectors in one column yields cosine scores that look valid and mean nothing.

**Multi-tenant guardrail.** Every write and every retrieval query requires an explicit
`tenant_id` + `mailbox_id`; there is no unscoped search. Approximate HNSW filtering runs
after an approximate scan, so at scale, partition by tenant/mailbox or tune candidate
budgets so filters don't silently destroy recall.

---

## 7. Asynchronous LLM contextualization (Stage 4)

Terse chunks like `Approved. Please proceed.` are hard to retrieve. A background worker
asks a small LLM what the chunk *concerns* and prepends one factual sentence to
`embed_text`, then re-embeds:

```
chunk → context job → context_prefix + version → rebuild embed_text
      → re-embed → update vector + BM25 state
```

**Invariant:** `text`, `source_start`, `source_end` never change. The model's words only
ever enter `embed_text`. `build_embed_text` with no prefix returns byte-identical Stage-1
output — that is what lets a chunk be contextualized without its deterministic form drifting.

Email content is **untrusted data**, delimited and labeled as such; the model's output is
*also* untrusted and independently re-validated (strict JSON, token budget, sentence
count, no citation markers). Anything failing becomes a deterministic fallback: the chunk
keeps its Stage-1 `embed_text`, stays retrievable, gains no prefix.

**Retry vs. fallback is deliberate.** A provider outage (timeout, 5xx) is transient →
job returns to `pending`, writes nothing. Invalid output is deterministic — at
`temperature=0` a retry produces the identical bad output, so retrying is a
guaranteed-useless loop → fall back instead.

---

## 8. Knowledge graph / GraphRAG (Stages 5–6)

### 8.1 Why a graph, and why email specifically

Plain hybrid retrieval answers *"find the passage that looks like the question."*
Necessary but insufficient for a mailbox — the hardest email questions are relational,
temporal, and aggregative:

| Question type | Example | What plain RAG misses |
|---|---|---|
| Relational / multi-hop | "Who did Sarah loop in about the Q3 budget?" | The answer is an *edge* (CC/forwarded-to), not a similar passage. |
| Aggregative | "All open action items assigned to me?" | Spans dozens of threads; no single chunk holds it. |
| Temporal / state | "What's the **current** ship date?" | Three emails, three dates; only the latest is valid — cosine treats them equally. |
| Entity-centric | "How is Bob connected to the vendor deal?" | The answer is a *path* through people/orgs/projects. |

Email *is already* a small graph (`sender —SENT→ message —CC→ recipient`,
`message —MENTIONS→ project`). Making it explicit turns those question classes into cheap
traversals.

### 8.2 Node & edge schema (in the same Postgres)

Relational edge tables + recursive CTEs — zero new infra, kept AGE-compatible so a later
openCypher swap is mechanical:

```sql
kg_nodes (node_id, tenant_id, mailbox_id, kind, canonical, attrs jsonb, embedding vector(768))
kg_edges (src, dst, rel, ts, message_id, weight, confidence)
```

Node kinds: `Person / Org / Message / Thread / Project / Document / Meeting /
Commitment / Decision`. Edge kinds: `SENT / TO / CC / REPLY_TO / MENTIONS / COMMITTED_TO
/ DECIDES / SUPERSEDES`. Every edge carries `{ts, message_id, confidence}` so the graph
is **auditable** (traces to a citable message) and **temporal**.

### 8.3 The trust rule for graph facts

> Graph facts are evidence-backed retrieval **cues**, not independently verified truth.

The LLM supplies only evidence *strings*; code locates each string verbatim in the
chunk's own immutable `text` and derives offsets itself, dropping anything it cannot
locate. Metadata edges (`SENT/CC/REPLY_TO`) are stored with `evidence_kind=metadata` and
no offsets, and a DB `CHECK` forbids them ever carrying an authored-text span.

**`active` is the latest retained assertion under an explicit supersession rule.** A fact
moves to `superseded` only when a *later* fact's evidence text carries an explicit update
cue (`replaces`, `updated from`, `now`, `instead of`) — never merely because a newer
email exists. A date alone never supersedes. This is the difference between "here are 3
dates people mentioned" and "the date is **June 14** (was May 30, moved on Jun 2 by Bob)."

### 8.4 Deterministic query planner (Stage 6)

`rag/planner.py` classifies a query with regex + literal-token rules only — no LLM,
embeddings, spaCy, or network — and returns a typed `RetrievalPlan`:

- *Generic* → BM25 + dense hybrid.
- *Entity/relationship* → matching graph entities/relations/facts, whose evidence chunks
  fuse with hybrid candidates.
- *Current/latest* → only `active` facts for the query's subject scope.
- *`as of <date>`* → facts with a real `effective_date` at/before an explicit date;
  undated facts are never treated as historically valid.

**Graph retrieval always returns source email evidence.** Every branch resolves through
mentions / relation evidence / `fact_evidence` to real chunks — never synthetic fact
strings. If a graph route yields no citable chunks, it falls back to hybrid and records
the reason in the trace. Fusion reuses the same weighted-RRF; graph weight and candidate
limits are config. Safe by default: enabled but inert without graph data (every route
falls back to hybrid, so existing deployments retrieve identically).

---

## 9. Grounded answering + bounded Self-RAG (Stage 7)

Disabled by default; when off, a deterministic answer path is used and no LLM client is
imported. When enabled, the LLM drafts *on top of* Stage-6 retrieval and **local
validation — not the model — decides whether it ships**:

```
query → Stage-6 retrieval → clean evidence pack → structured LLM draft
      → local validation → accept | one retry | abstain
```

**Citation contract.** Every factual claim carries ≥1 citation; each resolves to a chunk
in the *current* result and quotes that chunk's clean authored `text` verbatim (with
exact offsets). Citations never resolve to `embed_text`, headers, graph prose, quoted
history, signatures, or metadata.

**Self-RAG is advisory.** The provider returns `is_relevant` / `is_supported` /
`is_useful` / `needs_more_evidence`, but local validation is authoritative. Malformed
JSON, invented ids/quotes, wrong offsets, uncited claims, or metadata-only "evidence"
reject the draft — as does any instruction injected inside an email body.

**Bounded retry, then abstain.** On rejection, retrieval re-runs once with a wider,
still-bounded budget and the model drafts again. The ceiling is fixed at two attempts in
code. If the second also fails, there's no evidence, or the provider is off/failing, the
result is an explicit `abstained` — never an unsupported answer. Traces stay body-free:
route, candidate counts, the validation rule, attempt count only.

---

## 10. PDF attachments (Stage 8)

Deliberately narrow: **PDF only.** Non-PDF attachments never enter the pipeline.

```
Gmail attachment metadata → extraction job → fetch/decode bytes
  → native PDF text per page → OCR fallback only for image-only pages
  → page chunks → the existing embedding/retrieval/graph/context pipelines
```

- **Sync never blocks.** Sync persists attachment metadata and enqueues idempotent
  extraction work; the **bytes** are fetched only in the extraction worker, off the sync
  path — a slow parse/OCR can't stall sync.
- **Native first, OCR fallback.** Each page is extracted natively; a page with no usable
  native text is OCR'd *only if* a local Tesseract backend is available (no cloud OCR, no
  API key, no model download). When OCR is off/unavailable the page is recorded
  **unavailable** and produces no chunk — text is never invented.
- **Citation format.** One page → one or more `chunk_kind='attachment'` chunks. `text` =
  exact page text; `embed_text` = parent metadata + filename + `Page: N`. An OCR-derived
  citation is labeled (`[budget.pdf, page: 1 (OCR)]`) and never presented as byte-perfect
  original text.
- **Reuse, not reinvention.** Attachment chunks ride the existing chunk/embed helpers, the
  one hybrid/graph retriever, and the Stage-7 answer path — no second vector store or
  citation system.

---

## 11. Security, privacy, operations

Architecture requirements, not later polish:

- `tenant_id` + `mailbox_id` on every content, chunk, fact, job, and query row.
- Encrypt OAuth refresh tokens (AES-256); never log tokens or raw email bodies.
- Redact PII from traces; restrict support access.
- Retention, mailbox disconnect, deletion/tombstone propagation, derived-data invalidation.
- Outbox/job pattern, retries with backoff, dead-letter visibility, idempotency keys for
  every external/enrichment operation.
- Trace sync cursor, content hashes, retrieval candidates, fusion scores, model/prompt
  versions, citation checks, latency, cost.

---

## 12. Evaluation

Measure each layer separately — every case names the expected message/chunk evidence,
not just a desired natural-language answer:

| Layer | Measures |
|---|---|
| Parsing/chunking | Quote/signature leakage, false removal, chunk coverage, source-span validity. |
| Retrieval | Recall@K, MRR, nDCG, metadata-filter correctness, hybrid-vs-single lift. |
| Answering | Citation precision/recall, support rate, abstention quality, temporal correctness. |
| Sync | Lag, duplicate delivery, stale-cursor recovery, retry/dead-letter rate, delete propagation. |
| Operations | p50/p95 latency, embedding/context cost, job freshness, error rate, index health. |
| Security | Cross-tenant access, RLS, disconnect/delete, log-redaction tests. |

Prove hybrid retrieval on a hand-labeled golden set *before* adding a complex router or
more LLM steps.

---

## 13. Design decisions in one line each

- **Email-aware parsing** keeps only newly authored content in normal retrieval.
- **The dual `text`/`embed_text` contract** improves retrieval while preserving exact citations.
- **One ParadeDB store** gives transactional metadata + lexical + vector search in a single
  database — simpler correctness and operations than a separate vector DB.
- **The DB is the queue** — leased `FOR UPDATE SKIP LOCKED` jobs, deduped by input hash;
  no Celery/Redis/Kafka.
- **Advanced LLM context, facts, and graph are asynchronous, versioned enrichments** that
  always trace back to original leaf evidence.
- **Deterministic routing + local citation validation** — the model drafts, code decides.

---

## 14. Interview walkthrough

Start with the problem: *"Email is not clean documentation. Replies duplicate old
content, signatures add noise, and the latest decision may conflict with earlier emails."*

Then the four choices in §13.

For **reliability**: *"The webhook is acknowledged only after a durable sync job commits.
History cursors advance only after work commits, are compared numerically so they can't
rewind, and a stale cursor triggers a full sync."*

For **trust**: *"The system never cites a summary, LLM prefix, or graph edge. It cites
exact authored email text with a stable message and source span — and if it can't, it
abstains."*

For **roadmap judgment**: *"I'd prove hybrid retrieval on a golden set before adding
autonomous routing or complex graph features. Those are optional improvements, not
foundations."*

---

## 15. References

- Gmail push notifications — https://developers.google.com/workspace/gmail/api/guides/push
- Gmail History API — https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.history/list
- ParadeDB indexing — https://docs.paradedb.com/documentation/indexing/create-index
- ParadeDB RRF scoring — https://docs.paradedb.com/documentation/sorting/score
- pgvector filtering / iterative scans — https://github.com/pgvector/pgvector
- Contextual Retrieval (Anthropic) — https://www.anthropic.com/news/contextual-retrieval
- RAPTOR paper — https://arxiv.org/abs/2401.18059
