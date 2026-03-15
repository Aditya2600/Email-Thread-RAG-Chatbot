# Email Thread RAG Chatbot

This project builds a local-first chatbot for a single selected email thread and its attachments. It uses mandatory hybrid retrieval with BM25, dense vectors, RRF, a reranker, deterministic evidence-bound answering, clause-level citation validation, short session memory, OCR fallback for low-text PDFs, and streaming responses in both the API and UI.

## Setup
All commands below assume you are running from the parent folder that contains the `email_thread_rag/` directory. If you are already inside `email_thread_rag/`, run `cd ..` first.

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r email_thread_rag/requirements.txt
```

3. Install Tesseract locally if you are not using Docker.
   On macOS:

```bash
brew install tesseract
```

4. Install `antiword` locally if you want native `.doc` extraction outside Docker.

```bash
brew install antiword
```

5. Copy the environment template:

```bash
cp email_thread_rag/.env.example email_thread_rag/.env
```

6. The checked-in manifest already points at the Enron Archive mailbox dataset and auto-fetches a laptop-sized slice on first ingest.
   `.eml` support remains available for tests and secondary/manual ingestion only.

## Run commands
Build the dataset slice and indexes:

```bash
python -m email_thread_rag.scripts.ingest_corpus
```

Run the API:

```bash
uvicorn email_thread_rag.app.main:app --reload
```

Run the UI:

```bash
python -m email_thread_rag.ui.app
```

Run everything with Docker:

```bash
docker compose up
```

## Environment variables
- `EMAIL_RAG_DATASET_MANIFEST_PATH`: raw pinned manifest path.
- `EMAIL_RAG_RESOLVED_MANIFEST_PATH`: resolved copied/downloaded manifest path.
- `EMAIL_RAG_CHUNK_STORE_PATH`: chunk output file path.
- `EMAIL_RAG_STATS_PATH`: ingest statistics output file path.
- `EMAIL_RAG_API_BASE_URL`: UI target for API calls.
- `EMAIL_RAG_EMBEDDING_MODEL_NAME`: embedding model override.
- `EMAIL_RAG_RERANKER_MODEL_NAME`: reranker override.
- `EMAIL_RAG_REWRITE_MODEL_NAME`: rewrite model override.
- `EMAIL_RAG_ENABLE_CLOUD_REWRITE`: optional cloud rewrite flag.
- `EMAIL_RAG_CLOUD_REWRITE_PROVIDER`: set to `gemini` for Gemini rewrite enhancement.
- `EMAIL_RAG_CLOUD_REWRITE_MODEL`: Gemini model name, such as `gemini-2.5-flash`.
- `EMAIL_RAG_API_PORT`: API port.
- `EMAIL_RAG_UI_PORT`: UI port.
- `HF_TOKEN`: optional Hugging Face token for higher download limits.
- `GEMINI_API_KEY`: optional Gemini API key used only when cloud rewrite is enabled.

Example `.env`:

```dotenv
EMAIL_RAG_DATASET_MANIFEST_PATH=/absolute/path/to/email_thread_rag/data/raw/dataset_manifest.json
EMAIL_RAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMAIL_RAG_RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
EMAIL_RAG_REWRITE_MODEL_NAME=t5-small
EMAIL_RAG_ENABLE_CLOUD_REWRITE=false
EMAIL_RAG_CLOUD_REWRITE_PROVIDER=gemini
EMAIL_RAG_CLOUD_REWRITE_MODEL=gemini-2.5-flash
EMAIL_RAG_API_BASE_URL=http://localhost:8000
HF_TOKEN=
GEMINI_API_KEY=
```

## Which dataset to use
- Use `enronarchive/enron-mail` as the main public dataset.
- Do not use the old Enron WebMail login site.
- The default checked-in manifest fetches mailbox JSON from the Enron Archive GitHub repo and selects a deterministic laptop-sized slice on first ingest.
- The default slice is currently configured for:
  - mailbox: `allen-p`
  - date window: `2000-12-01` to `2001-05-31`
  - target size: `20` threads, about `100+` messages, and about `20-50` attachments
- The main corpus path is mailbox JSON plus selected attachments. `.eml` is only a secondary/manual ingestion path.
- To rebuild the slice explicitly, run:

```bash
python -m email_thread_rag.scripts.build_dataset_slice --force
python -m email_thread_rag.scripts.ingest_corpus --build-slice
```

The ingest script auto-builds the slice if no resolved manifest exists yet.

## How ingestion works
- `build_dataset_slice.py` reads `data/raw/dataset_manifest.json`, downloads the selected mailbox JSON files from `enronarchive/enron-mail`, deterministically selects a small thread slice, downloads only the needed attachments, and writes `data/processed/resolved_dataset_manifest.json`.
- `ingest_corpus.py` auto-builds that slice on first run, then normalizes it into email records, attachment records, and chunk records.
- Enron mailbox JSON is the primary corpus path.
- `.eml` parsing is supported through `rag/parse_eml.py` for tests and secondary/manual ingestion only.
- Attachments are parsed page-aware. PDFs stay per page, and low-text pages trigger OCR.
- Legacy `.doc`, `.xls`, and `.rtf` attachments in the Enron slice are also extracted in the default path.

## How retrieval works
- Retrieval always starts inside the active thread.
- BM25 returns top 15 lexical hits.
- Dense retrieval returns top 15 vector hits.
- RRF fuses the two lists.
- A cross-encoder reranks the fused top 10.
- The final evidence set is the top 5 reranked hits.
- If `search_outside_thread=true`, the engine retries globally only when the in-thread result fails the explicit support thresholds.

## How rewrite works
- Every `/ask` call rewrites the query before retrieval.
- Primary path: local `t5-small`.
- Fallback path: deterministic rule-based rewrite for pronouns, ellipsis, temporal references, and corrections.
- Optional cloud enhancement path: Gemini, behind `EMAIL_RAG_ENABLE_CLOUD_REWRITE=true`.
- When Gemini is enabled, the app still produces a local rewrite first, then optionally refines that query with Gemini using only the session context and local draft rewrite.
- `rewrite_mode` is returned and logged as `t5`, `rules`, `t5+gemini`, or `rules+gemini`.

## How correction handling works
- Corrections such as `no, I meant the PDF` update `correction_override`.
- The previous target interpretation is invalidated.
- The query is rewritten against the corrected target.
- Retrieval reruns from scratch.
- The final answer uses only corrected evidence.

## How citation validation works
- Answer generation is deterministic and evidence-bound.
- Every factual clause is validated against retrieved evidence.
- Each clause gets a `clause_support_score = 0.6 * token_overlap_f1 + 0.4 * entity_value_match`.
- Unsupported clauses are dropped.
- If fewer than 70% of factual clauses survive validation, the bot abstains or asks one short follow-up.

## API
- `POST /start_session` with `{"thread_id": "..."}`.
- `POST /switch_thread` with `{"session_id": "...", "thread_id": "..."}`.
- `POST /reset_session` with `{"session_id": "..."}`.
- `POST /ask` with `{"session_id": "...", "text": "...", "search_outside_thread": false}`.

`/ask` content negotiation:
- `application/json`: normal structured response.
- `text/event-stream`: SSE with `delta` events followed by one `final` event.

Final `/ask` payload fields:
- `answer`
- `citations`
- `rewrite`
- `rewrite_mode`
- `retrieved`
- `trace_id`
- `outside_thread_used`
- `metrics`

## UI
- Gradio provides:
  - thread selector
  - streaming chat area
  - outside-thread toggle
  - debug panel for rewrite, hits, scores, reranked results, citations, trace ID, and fallback usage

## Tests
Run:

```bash
pytest email_thread_rag/tests
```

Included coverage:
- thread scoping
- attachment page citations
- correction override behavior
- comparison path
- SSE response shape
- citation validator filtering
- OCR fallback
- outside-thread fallback
- rewrite fallback
- end-to-end manifest build + ingest path

## Demo video
- Screen recording file: `Screen_Recording/Screen Recording 2026-03-15 at 10.34.27 PM.mov`
- Recommended submission note: include this recording alongside the repository so reviewers can verify thread focus, pronoun/ellipsis handling, correction handling, attachment citations, and one graceful failure.

## Sample questions
For one clean attachment-bearing thread in the default Allen slice, use:
- Chosen thread: `allen-p:bid solicitation`
- Email message IDs in the selected slice: `036c4137cd99dc2c513e96916a32f624` and `c90ecfb887b787e72b8ce2131cab5fab`
- Attachment filename: `mobidltr.501.doc`

Sample questions with expected citation shapes:
- `What does the email say is due and when?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab]`
- `What company is requesting bids in the attachment?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624, page: 1]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab, page: 1]`
- `Which territories are mentioned in the attachment?`
  Expected citation: `[msg: 036c4137cd99dc2c513e96916a32f624, page: 1]` or `[msg: c90ecfb887b787e72b8ce2131cab5fab, page: 1]`
- `And when is the deadline?`
  Expected behavior: follow-up rewrite resolves the prior attachment/email context and returns a cited date
- `No, I meant the attachment. What month are the supply requests for?`
  Expected behavior: correction override forces new retrieval and returns an attachment-grounded answer with `[msg: ..., page: 1]`
- `What hotel was booked for the meeting?`
  Expected behavior: graceful abstention or a concise missing-evidence answer because that thread does not discuss hotel bookings

## Known limitations
- The checked-in manifest points at the real Enron Archive GitHub repo, but this sandbox cannot verify a live network fetch during implementation.
- If you want a stricter pin than `revision=main`, set a specific dataset revision in `data/raw/dataset_manifest.json`.
- If local model downloads are unavailable, the project falls back to deterministic rewrite and lightweight test encoders/rerankers; the production path still targets the configured open-source models.
- The real Enron slice contains many legacy Office attachments; native `.doc` extraction outside Docker depends on `antiword`.
