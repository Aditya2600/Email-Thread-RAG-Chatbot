# DATASET.md

## Source
- Primary intended source: Enron Archive / `enronarchive/enron-mail`
- Link: [https://github.com/enronarchive/enron-mail](https://github.com/enronarchive/enron-mail)

## Current repository state
- The checked-in manifest targets the real Enron Archive mailbox dataset.
- First ingest auto-fetches a deterministic laptop-sized slice from mailbox JSON plus the selected attachments.
- `.eml` remains supported for tests and secondary/manual ingestion, but it is not the main corpus path.

## Selection method
- Manifest-driven selection by:
  - mailbox
  - mailbox JSON source files
  - deterministic thread/message limits
  - selected attachment paths discovered from the mailbox JSON

## Current default slice
- Mailbox: `allen-p`
- Date range: `2000-12-01` to `2001-05-31`
- Slice rule:
  - choose `20` total threads
  - prefer `12` attachment-bearing threads with `1` attachment-bearing message per selected thread
  - fill the rest with longer non-attachment threads from the same window
  - take up to `10` messages per selected thread
- Target counts:
  - threads: `20`
  - messages: about `100+`
  - attachments: about `20-50`
- Actual counts from the current rebuilt slice:
  - threads: `20`
  - messages: `107`
  - attachments: `23`
  - chunks: `388`
  - approximate indexed text size: `480898` characters
  - OCR-triggered pages: `4`
- These counts are also written to `data/processed/ingest_stats.json`

## Preprocessing notes
- Messages are normalized into a common schema shared with `.eml` ingestion.
- Enron mailbox HTML bodies are converted to visible text during normalization.
- Thread reconstruction uses headers first, then subject + participants + time fallback.
- Attachments are parsed page-aware.
- The default Enron slice includes legacy Office formats, so `.doc`, `.xls`, `.rtf`, plain text, HTML, and PDF parsing are all exercised.
- OCR triggers for PDF pages with fewer than 20 alphanumeric characters or text density below 0.05.
- OCR usage is recorded per page and per chunk.

## License notes
- The project code is local project code only.
- The intended live corpus source is public archive material; confirm the exact downstream reuse terms of the chosen EnronArchive snapshot before redistribution.
- The Enron Archive repository is the referenced source for the mailbox JSON and attachment slice used here.

## OCR note
- The offline placeholder slice does not include scanned PDFs.
- OCR acceptance is implemented and covered by dedicated fixtures/tests.
