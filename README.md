# Inbox Copilot — Grounded Email RAG

**Ask your inbox a question and get an answer that only ever cites the exact email or PDF page it came from — or an honest "I don't know."**

Inbox Copilot turns a Gmail mailbox (or the public Enron email corpus) into a
trustworthy question-answering system. It combines lexical + semantic search, a
knowledge graph, and a strict citation validator so that every factual claim is
backed by a verbatim quote from real email evidence. If the evidence isn't there,
it abstains instead of guessing.

> Built as an end-to-end applied-AI / backend systems project: retrieval, LLM
> orchestration, a durable Gmail sync pipeline, and a Postgres-based vector +
> search store — all in one codebase.

---

## The problem

Email is the worst case for naive RAG. Replies quote old messages, signatures and
legal disclaimers add noise, and the *latest* decision often contradicts three
earlier ones in the same thread. A generic chatbot on top of email will happily
quote a superseded budget number as if it were current, or invent a citation.

Inbox Copilot is designed around one rule:

> **Index enriched text for recall, but cite only exact, authored email evidence.**

Every chunk stores two things — the exact authored text (what gets cited) and an
enriched `embed_text` with headers/context (what gets searched). Headers,
summaries, quoted history, and LLM-generated context help *find* an email but can
never *become* the citation.

---

## What it does

- 🔎 **Hybrid retrieval** — BM25 lexical search + dense vector search, fused with
  Reciprocal Rank Fusion and reordered by a cross-encoder reranker.
- 🧠 **Knowledge graph** — evidence-backed entities, relationships, and temporal
  facts, so "who was looped in?" and "what's the *current* ship date?" become
  graph lookups, not similarity guesses.
- 🧾 **Verifiable answers** — a deterministic validator checks that every claim
  quotes real evidence verbatim; unsupported claims are dropped and the bot
  abstains rather than hallucinate.
- 📎 **Attachments** — PDFs are parsed per page with OCR fallback, and answers
  cite the exact page (`[budget.pdf, page: 2]`).
- 📬 **Real Gmail sync** — OAuth, Pub/Sub push, and a durable, idempotent,
  crash-safe delta-sync worker on top of the Gmail History API.
- 💬 **Conversational** — pronoun/ellipsis follow-ups ("and when's the deadline?")
  and mid-conversation corrections ("no, I meant the attachment") are resolved
  before retrieval.

---

## Architecture at a glance

**Background ingestion & indexing**

```
Gmail / Enron ──▶ sync + ingest ──▶ email-aware chunker ──▶ ParadeDB (Postgres)
  OAuth · watch      durable jobs      authored text only     vectors + BM25 + graph
  Pub/Sub            idempotent        text vs embed_text
                                                │
                              asynchronous enrichment
                          ┌─────────────────────┴─────────────────────┐
                   LLM contextualizer                          graph extractor
                   (re-embed embed_text)                 (entities · relations · facts)
```

**Query-time retrieval**

```
question
   ▼
deterministic query planner ─┬─ structured (sender/date/graph/temporal)
                             ├─ dense (vector / HNSW)
                             └─ lexical (BM25)
   ▼
RRF fusion ──▶ cross-encoder rerank ──▶ bounded evidence pack
   ▼
grounded answer (LLM) ──▶ local citation validator ──▶ answer OR one wider retry OR safe abstention
```

Full diagrams and design rationale live in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language / API | Python, FastAPI (SSE streaming) |
| Store | ParadeDB / Postgres — `pgvector` (HNSW) + `pg_search` (BM25) in one database |
| Embeddings | GTE ModernBERT (768-dim) via `sentence-transformers` |
| Reranking | Cross-encoder (MS-MARCO MiniLM) |
| LLM | Any OpenAI-compatible endpoint (contextualization, grounded answers) |
| Ingestion | Gmail API (OAuth + History delta sync), Google Cloud Pub/Sub |
| Attachments | Per-page PDF extraction, Tesseract OCR fallback |
| Frontend | React (chat UI with sources/debug panel) |
| Infra | Docker Compose |

---

## Engineering highlights

Things I'd walk an interviewer through:

- **Correctness under failure.** The Gmail sync cursor advances in exactly one
  place, only after every message commits. Crashes, retries, and out-of-order
  Pub/Sub redeliveries are all safe — replaying a window is idempotent, and
  history IDs are compared numerically so a cursor can never silently rewind.
- **The queue *is* the database.** No Celery/Redis/Kafka — sync, contextualization,
  graph, and attachment jobs all use one Postgres pattern: `FOR UPDATE SKIP LOCKED`
  leased claims, `pending → running → done|failed`, deduped by input hash.
- **Prompt-injection boundary.** Email/PDF text is untrusted data wrapped in
  explicit delimiters; the model's output is *also* untrusted and independently
  re-validated. A body saying "ignore previous instructions" is data, not a command.
- **Tested without dependencies.** The whole suite runs on an in-memory backend
  with zero Postgres, Docker, Gmail, LLM, or network — enforced by tests, not
  convention. Fakes for Gmail/Pub/Sub/LLM are held to the same contract as the
  real implementations, so the fast tests can't drift from production.
- **Graph facts stay honest.** The LLM only supplies evidence *strings*; code
  locates each verbatim in the immutable chunk text and derives offsets itself,
  dropping anything it can't find. Every fact traces to an exact span of real text.

---

## Quick start

The baseline runs entirely in-memory — no database, Docker, Gmail account, LLM, or
network needed.

```bash
pip install -e '.[dev]'                                     # install + test deps
pytest -q                                                    # full suite, no services
python -m email_thread_rag.scripts.ingest_corpus             # build the demo corpus
uvicorn email_thread_rag.app.main:app --reload               # start the API
cd frontend && npm install && npm run dev                    # start the UI
```

Ask a question:

```bash
curl -sX POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"session_id":"...","text":"What does the email say is due and when?"}'
```

**Optional production path:** Postgres/ParadeDB, real Gmail sync, LLM
contextualization, and attachment OCR are all opt-in via environment variables and
Docker. See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** and the detailed setup notes below.

---

## Demo

[▶ Screen recording](https://github.com/Aditya2600/Email-Thread-RAG-Chatbot-Nexux-Ocean/raw/main/Screen_Recording/Screen%20Recording%202026-03-15%20at%2010.34.27%E2%80%AFPM.mov)

Shows thread selection, streaming chat, pronoun/ellipsis follow-ups, attachment
citations, correction override, and graceful abstention on out-of-scope questions.

Try these against the default Enron slice:

- *"What does the email say is due and when?"* → cites the source message
- *"What company is requesting bids in the attachment?"* → cites `[msg: …, page: 1]`
- *"And when is the deadline?"* → follow-up resolves the prior context
- *"What hotel was booked for the meeting?"* → abstains; that thread has no such evidence

---

## Project status

Built in stages, each independently tested:

| Stage | What shipped |
|---|---|
| 1 | Email-aware parsing & chunking (authored text vs. quoted/signature/disclaimer) |
| 2 | ParadeDB persistence + hybrid BM25 + dense retrieval |
| 3 | Gmail OAuth, Pub/Sub push, durable idempotent delta sync |
| 4 | Asynchronous LLM contextualization |
| 5 | Evidence-backed knowledge-graph extraction |
| 6 | Deterministic query planner + graph retrieval |
| 7 | Grounded answering + bounded Self-RAG with citation validation |
| 8 | PDF attachment extraction, OCR, page-level citations |

For the deep technical narrative — design decisions, failure-mode reasoning, the
knowledge-graph/GraphRAG design, and the full system diagrams — see
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

<details>
<summary><b>Detailed setup, environment variables, and internals</b></summary>

### Install extras

- `.[serve]` — `uvicorn` for the HTTP API.
- `.[models]` — `torch`, `sentence-transformers`, `faiss-cpu` for the model-backed
  encoder/reranker (baseline falls back without them).
- `.[gmail]` — `cryptography` + `google-auth` for Gmail sync.
- `.[ocr]` — local Tesseract-based PDF OCR.

Install Tesseract (`brew install tesseract`) and, for legacy `.doc` attachments,
`antiword` (`brew install antiword`) if not using Docker.

### Environment variables

| Variable | Purpose |
|---|---|
| `RAG_BACKEND` | `memory` (default, no DB) or `paradedb` |
| `DATABASE_URL` | Postgres/ParadeDB connection (paradedb backend) |
| `TENANT_ID` / `MAILBOX_ID` | Tenant + mailbox scope for every query |
| `EMAIL_RAG_EMBEDDING_MODEL_NAME` | Embedding model (`Alibaba-NLP/gte-modernbert-base`) |
| `EMAIL_RAG_RERANKER_MODEL_NAME` | Cross-encoder reranker |
| `EMAIL_RAG_ENABLE_CLOUD_REWRITE` / `EMAIL_RAG_CLOUD_REWRITE_*` | Optional Gemini query-rewrite enhancement |
| `GMAIL_CLIENT_ID` / `GMAIL_CLIENT_SECRET` / `GMAIL_REDIRECT_URI` | Gmail OAuth |
| `GMAIL_PUBSUB_TOPIC` / `GMAIL_PUBSUB_SUBSCRIPTION` | Pub/Sub push channel |
| `GMAIL_TOKEN_ENCRYPTION_KEY` | AES-256 key for refresh tokens at rest |
| `CONTEXT_ENABLED` / `CONTEXT_BASE_URL` / `CONTEXT_MODEL` / `CONTEXT_API_KEY` | LLM contextualization |
| `GRAPH_EXTRACTION_ENABLED` / `GRAPH_PLANNER_ENABLED` | Knowledge-graph extraction + retrieval |
| `ANSWER_GENERATION_ENABLED` | LLM grounded answering (deterministic path when off) |
| `ATTACHMENT_OCR_ENABLED` | PDF OCR fallback |

### Running with the real store

```bash
docker compose -f infra/docker-compose.yml up -d db
python -m email_thread_rag.scripts.apply_paradedb_migrations
RAG_BACKEND=paradedb python -m email_thread_rag.scripts.ingest_corpus
```

Background workers (each optional, each drains a Postgres-backed queue):

```bash
python -m email_thread_rag.gmail.worker                  # Gmail delta sync
python -m email_thread_rag.context.worker --once         # LLM contextualization
python -m email_thread_rag.graph.worker                  # graph extraction
python -m email_thread_rag.rag.attachments.worker --once # PDF/OCR extraction
```

### API

- `POST /start_session` `{"thread_id": "..."}`
- `POST /switch_thread` `{"session_id": "...", "thread_id": "..."}`
- `POST /reset_session` `{"session_id": "..."}`
- `POST /ask` `{"session_id": "...", "text": "...", "search_outside_thread": false}`
  — returns JSON, or SSE (`delta` events + one `final`) when `Accept: text/event-stream`.

`/ask` response fields: `answer`, `citations`, `rewrite`, `rewrite_mode`,
`retrieved`, `trace_id`, `outside_thread_used`, `metrics`.

### How retrieval works

Retrieval starts inside the active thread: BM25 top-15 + dense top-15 → RRF fusion
→ cross-encoder reranks the fused top-10 → top-5 as evidence. With
`search_outside_thread=true`, the engine retries globally only when the in-thread
result fails explicit support thresholds.

### How citation validation works

Answering is evidence-bound. Each factual clause gets a
`clause_support_score = 0.6·token_overlap_f1 + 0.4·entity_value_match`; unsupported
clauses are dropped, and if fewer than 70% survive, the bot abstains or asks one
short follow-up.

### Dataset

Uses `enronarchive/enron-mail`. The checked-in manifest auto-fetches a laptop-sized
slice (mailbox `allen-p`, Dec 2000–May 2001, ~20 threads) on first ingest. Rebuild
explicitly:

```bash
python -m email_thread_rag.scripts.build_dataset_slice --force
python -m email_thread_rag.scripts.ingest_corpus --build-slice
```

### Tests

```bash
pytest -q -m "not integration"    # fast suite, no services
```

Covers thread scoping, attachment page citations, correction override, SSE shape,
citation-validator filtering, OCR fallback, outside-thread fallback, rewrite
fallback, Gmail sync/store contracts, and the end-to-end ingest path. Integration
tests run against a real ParadeDB container (Gmail/LLM still faked).

### Known limitations

- The manifest targets the live Enron Archive GitHub repo; network fetch isn't
  verified in a sandbox. Pin a specific `revision` in the manifest for stricter builds.
- Without local model downloads, the project falls back to deterministic rewrite and
  lightweight test encoders/rerankers.
- Conservative quote/signature stripping can miss unmarked bottom-posted quotes or
  novel disclaimers (biased toward keeping authored content).

</details>
