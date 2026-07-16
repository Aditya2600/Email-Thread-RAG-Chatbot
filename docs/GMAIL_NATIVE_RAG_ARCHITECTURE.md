# Validated Gmail-Native Email RAG Architecture

## Architecture, delivery plan, and interview guide

> This is the target architecture, not a claim that every component is already in production.
> The current reported codebase has a Stage 0 canonical package and Stage 1 email-aware
> parsing/chunking work. Before Stage 2 starts, close the remaining full-suite test gate.

## 1. Executive summary

This system turns a Gmail mailbox into a trustworthy retrieval system for questions such as:

- What did Alice decide about the Q3 budget?
- What is the latest approved number?
- Which email supports that answer?

The central design rule is:

    Index enriched retrieval text, but cite exact email evidence.

Every indexed chunk has two distinct representations:

- text: exact newly authored email content. It is the only content shown to users and
  used to validate a citation.
- embed_text: compact email metadata, safe deterministic or LLM context, and the exact
  text. It is the only content used for vector and lexical retrieval.

This separation improves retrieval without allowing headers, summaries, quoted history,
or LLM-generated context to masquerade as source evidence.

## 2. Current implementation status

| Area | Status | Notes |
|---|---|---|
| Stage 0 package baseline | Verified | Canonical package is email_thread_rag; the old src tree remains excluded. |
| Stage 1 parse and email-aware chunking | Feature work reported complete | Quote, signature, disclaimer segmentation; citation-safe chunk fields; embed_text indexing. |
| Stage 1 quality gate | Must close before Stage 2 | Reported run was 33 passed, 1 failed in a correction retrieval test. Treat Stage 1 as not fully signed off until this is reproduced and fixed or correctly baselined. |
| Stages 2 through 7 | Target only | No production Gmail, ParadeDB, LLM, graph, RAPTOR, or routing system should be assumed present. |

Two Stage 1 acceptance details require explicit verification:

1. A quote-only email must produce zero normal retrieval chunks. It may remain in an
   audit record, but must not fall back to indexing its full quoted body.
2. If signatures are promised to be excluded, a sign-off such as Regards, Bob must be
   removed consistently or documented as intentionally preserved authored text.

## 3. Corrected target architecture

    ┌──────────────────────────────────────────────────────────────────────┐
    │ GMAIL                                                        Stage 3 │
    │ OAuth2 · users.watch · Cloud Pub/Sub · History API                    │
    │ Watch renewal daily; periodic polling fallback                        │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ SYNC + INGEST                                                Stage 3 │
    │ Verify Pub/Sub → transactionally create/coalesce sync_job → commit   │
    │ → HTTP 200 ACK → worker → history.list → messages.get                │
    │ Idempotent upsert · stored historyId · delete/tombstone handling      │
    │ 404 stale cursor → full-sync fallback · thread/reply links            │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ PARSE + NORMALIZE                                            Stage 1 │
    │ MIME/HTML → normalized body · authored | quote | signature | footer   │
    │ Attachment metadata · exact offsets in normalized authored_text       │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ EMAIL-AWARE CHUNKER                                          Stage 1 │
    │ Short authored mail → one chunk                              │
    │ Long authored mail → paragraph/list/section-aware chunks     │
    │ Quote, signature, disclaimer → audit only; never normal index │
    │ source_start/source_end → offsets into normalized authored_text │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ CHUNK REPRESENTATION                                          Stage 1 │
    │ text = exact authored evidence                                 │
    │ context_prefix = deterministic now; LLM-produced later         │
    │ embed_text = compact headers + context_prefix + exact text      │
    │ provenance and version fields protect citations and reindexing  │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ EMBEDDING WORKER                                             Stage 2 │
    │ BGE encoder embeds only embed_text; batch, retry, version results     │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ ONE STORE: PARADEDB POSTGRES                                 Stage 2 │
    │ Core: mailbox · thread · message · attachment · sync_cursor · jobs    │
    │ Search: chunks + embedding · pgvector HNSW · pg_search BM25           │
    │ Security: tenant_id/mailbox_id everywhere · RLS · audit fields        │
    └───────────────┬──────────────────────────────┬───────────────────────┘
                    │                              │
                    │                              └── query-time read path
                    ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ ASYNCHRONOUS ENRICHMENT JOBS                               Stages 4–5 │
    │ LLM contextualizer → rebuild embed_text → re-embed → reindex          │
    │ Fact worker → entities, relations, facts, exact fact_evidence spans   │
    │ RAPTOR worker → summary_nodes/edges with descendant leaf provenance   │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Updated rows return to PARADEDB; original leaf text never changes     │
    └──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────────────┐
    │ QUERY PLANNER / ADAPTIVE RETRIEVAL ROUTER                    Stage 6 │
    │ metadata → SQL · semantic → vector + BM25 · relational → facts/CTE    │
    │ temporal → active facts · thematic → RAPTOR summary then leaf chunks  │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ RRF FUSION → optional rerank → EVIDENCE PACK                          │
    │ Candidate chunks retain message IDs, exact text, and source spans     │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │ GROUNDED ANSWER                                                       │
    │ [msg:id/chunk:id] citations → support check → answer or abstain       │
    └──────────────────────────────────────────────────────────────────────┘

Important reading order:

1. The embedding worker is outside the database because it computes vectors.
2. The HNSW and BM25 indexes are inside ParadeDB/Postgres because they index persisted
   chunk rows.
3. Contextualization, facts, and RAPTOR are asynchronous enrichments. They do not block
   Gmail synchronization and can be retried or backfilled.
4. The answer path always returns to original leaf chunks before citing an email.

## 4. Core contracts

### 4.1 Normalized message

The canonical normalized message should retain:

| Field | Purpose |
|---|---|
| message_id, thread_id | Stable Gmail identity and thread grouping. |
| in_reply_to, references | Reply-chain reconstruction. |
| sender, recipients, cc | Searchable email metadata. Cc must be stored independently from To. |
| subject, sent_at | Lightweight thread and temporal context. |
| authored_text | The newly authored normalized body. |
| quoted_text, signature_text, disclaimer_text | Audit/debug fields; excluded from normal retrieval. |
| attachments | Metadata now; extracted attachment content only when a later boundary is implemented. |

Source offsets are offsets into normalized authored_text, never raw MIME bytes or original
HTML. That makes source spans stable after decoding and HTML cleanup.

### 4.2 Chunk contract

| Field | Meaning | Can be cited? |
|---|---|---|
| text | Exact authored chunk evidence, no injected headers. | Yes |
| embed_text | Headers + safe context + exact text; search input. | No |
| source_start, source_end | Span within normalized authored_text. | Supports citation validation |
| content_hash | Hash of the exact text and metadata inputs. | Supports idempotency |
| context_prefix | Deterministic or LLM-generated retrieval help. | No |
| context_method, context_version | none, deterministic, or llm; generator/prompt version. | Audit only |
| embedding_model, embedding_version | Model identity and version used for current vector. | Audit/reindex |
| enrichment_status | pending, ready, failed, stale, or deleted. | Operations only |

For compatibility with existing fixtures, old records may default embed_text to text. New
ingestion should always build the richer value deliberately.

### 4.3 Deterministic Stage 1 embed_text

    From: Bob <bob@acme.com>
    To: Sarah <sarah@acme.com>
    Cc: Finance <finance@acme.com>
    Date: 2026-05-03
    Subject: Q3 Budget
    Thread-ID: thread_123
    In-Reply-To: <parent@acme.com>

    The approved budget is now $120,000.

The header block improves retrieval for people, date, subject, and thread queries. The
visible citation remains only:

    The approved budget is now $120,000.

## 5. Why the EmailAwareChunker exists

Generic fixed-window splitting treats an email as an ordinary document. That fails because
reply chains repeat old messages, signatures add irrelevant names and phone numbers, and a
single thread can contain superseded decisions.

The EmailAwareChunker works in this order:

1. Normalize the MIME or HTML body into stable plain text.
2. Conservatively segment quoted history, disclaimers, and signatures away from the newly
   authored body.
3. If authored content is short, emit exactly one independently citable chunk.
4. If it is long, pack complete paragraphs, lists, and sections toward roughly 300 to
   450 tokens, with about 40 to 60 tokens of overlap only when needed.
5. Store exact authored text and its source span as the chunk.
6. Build embed_text from compact message metadata plus the exact chunk text.
7. Send only embed_text to BM25/vector indexing.

Quoted reply text is excluded because it is historical copied content, often duplicated
many times. Indexing it can retrieve an old budget as if the current sender wrote it. It
remains available in the audit representation when needed.

Deterministic segmentation is intentionally conservative. It can miss unusual
bottom-posted quotes, unmarked signatures, and novel legal notices. Measure quote leakage
and false removal in evaluation; do not claim perfect stripping.

## 6. Gmail synchronization: correct delivery semantics

The ingestion webhook must be fast, durable, and idempotent:

    1. Verify Pub/Sub authentication and mailbox ownership.
    2. In one transaction, create or coalesce a durable sync_job.
    3. Commit the job transaction.
    4. Return HTTP 200 to acknowledge the Pub/Sub delivery.
    5. A worker calls history.list from the persisted cursor.
    6. For each affected message identity, call messages.get and upsert idempotently.
    7. Process messageAdded, messageDeleted, labelAdded, and labelRemoved events.
    8. Advance historyId only after every page and derived write commits.

Why commit before ACK? If the service acknowledges first and crashes before storing the job,
the notification may be lost. Duplicate jobs are safe because the database uniqueness and
content hashes make processing idempotent.

Operational safeguards:

- Renew users.watch daily. Gmail requires renewal at least every seven days.
- Run periodic reconciliation/history polling because Gmail notifications can be delayed
  or dropped.
- If history.list returns a stale-cursor 404, enqueue a bounded full synchronization.
- Preserve delete/tombstone events and invalidate derived chunks, embeddings, facts, and
  summaries.
- Scope every sync cursor and job by tenant_id and mailbox_id.

## 7. Stage 2: ParadeDB and hybrid retrieval

ParadeDB/Postgres is the system of record. Avoid a separate vector database until a real
scale or operational requirement justifies it.

### Recommended persisted areas

| Area | Main rows |
|---|---|
| Mailbox core | mailboxes, threads, messages, attachments, sync_cursors, sync_jobs |
| Search | chunks, embedding values, chunk metadata, lexical and vector indexes |
| Enrichment | enrichment_jobs, contextualization runs, extraction runs, summary runs |
| Evidence | entities, relations, facts, fact_evidence, summary_nodes, summary_edges |
| Safety | authorization/audit records, deletion state, retention metadata |

### Hybrid query path

    user question
       │
       ├── metadata filter → SQL
       ├── semantic intent → vector search
       ├── exact terms/people/subjects → BM25
       ├── relationship question → fact table or recursive CTE
       ├── latest/current question → active fact plus supersession rules
       └── broad thematic question → RAPTOR summary node, then descendant leaves
                                      │
                                      ▼
                       Reciprocal Rank Fusion with deterministic tie-break
                                      │
                                      ▼
                       optional cross-encoder rerank of a small candidate set
                                      │
                                      ▼
                      original leaf chunks, source spans, and message identifiers

Begin Stage 2 with deterministic routing rules, vector plus BM25 retrieval, metadata
filters, and RRF. Do not make an LLM planner a dependency of baseline recall.

Multi-tenant guardrail: approximate HNSW filtering happens after an initial approximate
scan. Always apply tenant_id and mailbox_id filters plus RLS. At higher scale, partition
by tenant/mailbox or tune iterative scans and candidate budgets so filters do not silently
destroy recall.

## 8. Stage 4: LLM contextualization

Contextualization improves retrieval for terse email chunks such as:

    Approved. Please proceed.

The contextualizer receives only safe local inputs: selected mail headers, parent-message
identifier, and the exact chunk. It writes a one- or two-sentence prefix such as:

    This is Bob's approval of the Q3 Finance budget request in reply to Sarah.

It must never modify text or generate a citation. The update flow is:

    chunk → context job → context_prefix + version → rebuild embed_text
          → re-embed → update vector and BM25 search state

The job must be idempotent and versioned by content hash, prompt version, context model,
and embedding model. A stale result should be safely replaced by a new job.

Use no LLM in Stage 1. A local Gemma deployment is an optional Stage 4 implementation:
Gemma 4 E4B IT in W4A16 compressed-tensors format is a pragmatic NVIDIA/vLLM option.
Gemma 3 4B AWQ is acceptable only when an existing AWQ serving path makes that operationally
cheaper. The important design property is the evidence boundary, not the model brand.

## 9. Stage 5: facts and RAPTOR

### 9.1 Fact extraction

Facts need a producer, confidence, time validity, and exact evidence:

    leaf chunks → extraction worker → entity resolution → facts
                  │
                  └── fact_evidence: message_id, chunk_id, source span, extractor version

Useful fact fields are subject, predicate, object/value, valid_from, valid_to, confidence,
status, supersedes_fact_id, and evidence reference. This enables answers such as the latest
approved budget without treating an older value as current.

### 9.2 Where RAPTOR belongs

RAPTOR is not a chunker and it is not Stage 1. It is a recursive
Retrieve-Abstract-Process hierarchy builder:

    persisted chunk embeddings
       → cluster related leaf chunks
       → summarize each cluster
       → embed summary nodes
       → repeat at a higher level

For email, preserve the natural message-to-thread hierarchy deterministically. Use RAPTOR
only above that level: topics, projects, and cross-thread themes. Store:

    summary_nodes: text, embedding, level, version
    summary_edges: parent node, child node
    provenance: all descendant source chunk identifiers

RAPTOR helps broad questions. It must never be cited directly. A retrieved summary expands
to original descendant leaf chunks, which are then reranked and cited.

## 10. Grounded answer path

    evidence pack
       → answer model constrained to supplied evidence
       → inline citations such as [msg:abc/chunk:2]
       → support check
       → answer, qualified uncertainty, or abstention

The support check verifies that every material claim has supporting original text and that
the cited source belongs to the authorized mailbox. A generated context prefix, a fact
record, or a RAPTOR summary may guide retrieval but cannot independently support a claim.

## 11. Security, privacy, and operations

These are architecture requirements, not later polish:

- Put tenant_id and mailbox_id on every content, chunk, fact, job, and query row.
- Enforce authorization with Postgres Row Level Security and service-level checks.
- Encrypt OAuth refresh tokens; never place tokens or raw email bodies in application logs.
- Redact PII from traces where possible and restrict support access.
- Implement retention, mailbox disconnect, deletion/tombstone propagation, and derived-data
  invalidation.
- Use an outbox/job pattern, retries with backoff, dead-letter visibility, and idempotency
  keys for every external or enrichment operation.
- Trace sync cursor, chunk/content hashes, retrieval candidates, fusion scores, model and
  prompt versions, citation checks, latency, and cost.

## 12. Delivery plan

| Stage | Outcome | Exit criteria |
|---|---|---|
| 0 | Package and baseline repair | Installable canonical package; baseline tests pass. |
| 1 | Email-native parse/chunk contract | Clean citable text, correct headers in embed_text, quote/signature protections, all tests green. |
| 2 | ParadeDB hybrid retrieval | Schema/migrations, idempotent repository, BGE batch worker, BM25 + HNSW + RRF, local test stack. |
| 3 | Gmail sync | OAuth, watch renewal, durable jobs, History API cursor semantics, full-sync fallback, deletion handling. |
| 4 | Contextual retrieval and answer | Async LLM context jobs, reindex flow, grounded generation, citations, support/abstain check. |
| 5 | Facts and hierarchical retrieval | Entity/fact evidence, temporal supersession, RAPTOR summary tree with leaf provenance. |
| 6 | Adaptive query planner | Deterministic intent routing first; SQL/vector/BM25/fact/RAPTOR selection and rerank policy. |
| 7 | Evaluation and hardening | Golden set, observability, permission tests, load/failure testing, tuning and rollout gates. |

Do not skip from Stage 1 to Stage 4 or 5 as an implementation shortcut. Contextualization,
facts, and RAPTOR need durable canonical chunks, embeddings, metadata filters, and
reindexing semantics from Stage 2. They can be designed early but should be implemented
after that substrate exists.

## 13. Quality and evaluation

Measure each layer separately:

| Layer | Measures |
|---|---|
| Parsing/chunking | Quote leakage, signature leakage, false content removal, chunk coverage, source-span validity. |
| Retrieval | Recall at K, MRR, nDCG, metadata-filter correctness, hybrid-vs-single-retriever lift. |
| Answering | Citation precision, citation recall, support rate, abstention quality, temporal correctness. |
| Sync | Lag, duplicate delivery rate, stale cursor recovery, retry/dead-letter rate, delete propagation time. |
| Operations | p50/p95 latency, embedding/context cost, job freshness, error rate, index health. |
| Security | Cross-tenant access tests, RLS tests, disconnect/delete tests, log-redaction tests. |

Use a hand-labeled email question set before adding a complex router or more LLM steps.
Every evaluation case should identify the expected message/chunk evidence, not just a desired
natural-language answer.

## 14. Interview walkthrough

Start with the problem:

    Email is not clean documentation. Replies duplicate old content, signatures add noise,
    and the latest decision may conflict with earlier emails.

Then explain the four design choices:

1. Email-aware parsing keeps only newly authored content in normal retrieval.
2. The dual text/embed_text contract improves retrieval while preserving exact citations.
3. ParadeDB provides transactional metadata, lexical search, and vector search in one
   database, simplifying correctness and operations.
4. Advanced LLM context, facts, and RAPTOR are asynchronous, versioned enrichments that
   always trace back to original leaf evidence.

For reliability, say:

    The webhook is acknowledged only after a durable sync job commits. Gmail History API
    cursors advance only after work commits, and stale cursors trigger a full sync.

For trust, say:

    The system never cites a summary, LLM prefix, or graph edge. It cites exact authored
    email text with a stable message and source span.

For roadmap judgment, say:

    I would prove hybrid retrieval on a golden set before adding autonomous routing or
    complex graph features. Those advanced features are optional improvements, not
    foundations.

## 15. Recommended implementation ownership

| Work | Primary model / mode | Review |
|---|---|---|
| Stage 1 to 3 repository implementation | Claude Sonnet, high effort, SuperClaude task execution | Codex high-effort independent code review and test diagnosis |
| Database schema, migrations, and retrieval design | Claude Opus, extra-high effort, SuperClaude design mode | Codex review focused on migrations, idempotency, and security |
| Gmail API semantics | Claude Sonnet, high effort, official Gmail docs available in context | Codex review against official API behavior |
| Tests, failure injection, and refactoring | Codex high effort | Claude focused review if the change is large |
| Documentation and interview narrative | Claude Opus or Codex, high effort | Human technical review |

Recommended helper capabilities when available:

- Serena for repository navigation and symbol-aware changes.
- Context7 for current framework/library documentation.
- Sequential reasoning for migrations, sync failure modes, and staged plans.
- Docker/database tooling only for the Stage 2 local ParadeDB environment.

Do not have Claude and Codex edit the same worktree concurrently. Let one implement, then
give the other a clean diff and test output for review.

## 16. Primary references

- Gmail push notifications: https://developers.google.com/workspace/gmail/api/guides/push
- Gmail History API: https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.history/list
- ParadeDB indexes: https://docs.paradedb.com/documentation/indexing/create-index
- ParadeDB RRF scoring: https://docs.paradedb.com/documentation/sorting/score
- pgvector filtering and iterative scans: https://github.com/pgvector/pgvector
- RAPTOR paper: https://arxiv.org/abs/2401.18059
- Gemma documentation: https://ai.google.dev/gemma/docs/core