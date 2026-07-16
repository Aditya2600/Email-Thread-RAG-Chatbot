import re

def make_box_line(text, total_len=85):
    return "│" + text + " " * (total_len - len(text)) + "│"

lines = []
# GMAIL Box
lines.append(" " * 44 + "┌───────────────────────────────────────┐")
lines.append(" " * 44 + "│                 GMAIL                 │")
lines.append(" " * 44 + "│ OAuth2 · users.watch · History API    │")
lines.append(" " * 44 + "│ Pub/Sub notifications                 │")
lines.append(" " * 44 + "└───────────────────┬───────────────────┘")
lines.append(" " * 64 + "▼")

# SYNC + INGEST
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" SYNC + INGEST"))
lines.append(make_box_line(" webhook → durable sync_job → ACK → worker → history.list → messages.get"))
lines.append(make_box_line(" idempotent upsert · stored historyId · full-sync fallback · thread/reply links"))
lines.append("└" + "─"*38 + "┬" + "─"*46 + "┘")
lines.append(" " * 39 + "▼")

# PARSE + NORMALIZE
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" PARSE + NORMALIZE"))
lines.append(make_box_line(" MIME/HTML → clean body · thread/reply links"))
lines.append(make_box_line(" split: authored body | quoted reply | signature | disclaimer"))
lines.append(make_box_line(" attachment metadata · exact source offsets"))
lines.append("└" + "─"*38 + "┬" + "─"*46 + "┘")
lines.append(" " * 39 + "▼")

# STAGE 1
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" STAGE 1: EMAIL-AWARE CHUNKER"))
lines.append(make_box_line(" short email → one chunk · long email → section-aware chunks"))
lines.append(make_box_line(" authored body only · quote/signature/disclaimer excluded"))
lines.append(make_box_line(" source_start / source_end · each message remains independently citable"))
lines.append("└" + "─"*38 + "┬" + "─"*46 + "┘")
lines.append(" " * 39 + "▼")

# CHUNK REPRESENTATION
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" CHUNK REPRESENTATION"))
lines.append(make_box_line(" text            = exact authored evidence for citation"))
lines.append(make_box_line(" context_prefix  = deterministic now; LLM-generated later"))
lines.append(make_box_line(" embed_text      = compact header + context_prefix + text"))
lines.append(make_box_line(" context_method  = none | deterministic | llm"))
lines.append(make_box_line(" context_version = generator/prompt version"))
lines.append("└" + "─"*15 + "┬" + "─"*55 + "┬" + "─"*13 + "┘")
lines.append(" " * 16 + "│" + " " * 55 + "│")
lines.append(" " * 16 + "│ deterministic index" + " " * 36 + "│ async backfill")
lines.append(" " * 16 + "▼" + " " * 55 + "▼")

# STAGE 2 and 4
lines.append("┌" + "─"*41 + "┐" + " "*7 + "┌" + "─"*35 + "┐")
lines.append("│ STAGE 2: BGE EMBEDDING + BM25 INDEXING  │       │ STAGE 4: LLM CONTEXTUALIZER       │")
lines.append("│ embed/index only embed_text             │       │ Gemma · 1–2 factual sentences     │")
lines.append("│ display/cite only original text         │       │ never changes text or citations   │")
lines.append("└" + "─"*22 + "┬" + "─"*18 + "┘" + " "*7 + "└" + "─"*15 + "┬" + "─"*19 + "┘")
lines.append(" " * 23 + "│" + " "*42 + "│")
lines.append(" " * 23 + "└" + "─"*19 + "┬" + "─"*22 + "┘")
lines.append(" " * 43 + "▼")

# ONE STORE
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" ONE STORE — PARADEDB POSTGRES"))
lines.append(make_box_line(" Core: mailbox · thread · message · attachment · sync cursor · sync_jobs"))
lines.append(make_box_line(" Search: chunks · pgvector HNSW · pg_search BM25"))
lines.append(make_box_line(" Facts: entities · relations · facts · fact_evidence                    [Stage 5]"))
lines.append(make_box_line(" Summary store: summary_nodes · summary_edges · source_chunk_ids        [Stage 5+]"))
lines.append("└" + "─"*30 + "┬" + "─"*39 + "┬" + "─"*14 + "┘")
lines.append(" " * 31 + "│" + " "*39 + "│")
lines.append(" " * 31 + "│ normal chunk retrieval" + " "*15 + "│ async hierarchy build")
lines.append(" " * 31 + "▼" + " "*39 + "▼")

# STAGE 5+ RAPTOR BUILDER
lines.append(" " * 17 + "┌" + "─"*30 + "┐" + " "*2 + "┌" + "─"*34 + "┐")
lines.append(" " * 17 + "│ Chunk retrieval              │  │ STAGE 5+: RAPTOR BUILDER         │")
lines.append(" " * 17 + "│ vector + BM25                │  │ chunks → cluster → summarize     │")
lines.append(" " * 17 + "│ exact evidence leaves        │  │ message → thread → topic nodes   │")
lines.append(" " * 17 + "└" + "─"*30 + "┘  │ summary embeddings + child links │")
lines.append(" " * 51 + "└" + "─"*15 + "┬" + "─"*18 + "┘")
lines.append(" " * 67 + "│ writes summary nodes")
lines.append(" " * 67 + "└" + "─"*11 + "┐")
lines.append(" " * 79 + "▼")

# QUERY PLANNER
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" QUERY PLANNER / ADAPTIVE RETRIEVAL ROUTER                             [Stage 6]"))
lines.append(make_box_line(" metadata → SQL · semantic → vector + BM25 · relational → facts/recursive CTE"))
lines.append(make_box_line(" temporal → active fact + supersession"))
lines.append(make_box_line(" thematic/broad → RAPTOR summary nodes → descend to child chunks"))
lines.append("└" + "─"*38 + "┬" + "─"*46 + "┘")
lines.append(" " * 39 + "▼")

# RRF FUSION
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" RRF FUSION → OPTIONAL CROSS-ENCODER RERANK → EVIDENCE PACK"))
lines.append(make_box_line(" summary nodes locate the area; original chunks provide evidence and citations"))
lines.append("└" + "─"*38 + "┬" + "─"*46 + "┘")
lines.append(" " * 39 + "▼")

# MEDHA GROUNDED ANSWER
lines.append("┌" + "─" * 85 + "┐")
lines.append(make_box_line(" STAGE 4: MEDHA GROUNDED ANSWER → [msg:id] CITATIONS → SUPPORT CHECK → ANSWER/ABSTAIN"))
lines.append("└" + "─" * 85 + "┘")

diagram = "\n".join(lines)
replacement = "```text\n" + diagram + "\n```"

import sys
path = "/Users/aditya/inbox-copilot/docs/GMAIL_NATIVE_RAG_ARCHITECTURE.md"
with open(path, "r") as f:
    content = f.read()

pattern = r"```text.*?```"
new_content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

with open(path, "w") as f:
    f.write(new_content)

print("Replaced ASCII diagram.")
