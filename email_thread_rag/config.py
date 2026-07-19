from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Repo root is the parent of the email_thread_rag package directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Auto-load the developer .env for real runs, but let a caller opt out. A test
# (or any process that has deliberately curated its environment) sets
# EMAIL_RAG_SKIP_DOTENV=1 so an on-disk .env cannot silently switch it onto the
# Gmail/Postgres integrations. Without this, a .env with DATABASE_URL + Gmail
# Pub/Sub settings would activate those paths even inside unit tests.
if os.getenv("EMAIL_RAG_SKIP_DOTENV", "").lower() not in ("1", "true", "yes"):
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class RetrievalThresholds(BaseModel):
    min_supported_chunks: int = 2
    min_chunk_support: float = 0.55
    min_top_rerank: float = 0.45
    min_citation_coverage: float = 0.70


class OCRThresholds(BaseModel):
    min_alnum_chars: int = 20
    min_text_density: float = 0.05


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMAIL_RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    index_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "indexes")
    runs_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "runs")

    dataset_manifest_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "dataset_manifest.json"
    )
    resolved_manifest_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "resolved_dataset_manifest.json"
    )
    chunk_store_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    )
    stats_path: Path = Field(
        default_factory=lambda: PROJECT_ROOT / "data" / "processed" / "ingest_stats.json"
    )

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_cloud_rewrite: bool = False
    cloud_rewrite_provider: Optional[str] = None
    cloud_rewrite_model: Optional[str] = None
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))

    bm25_top_k: int = 15
    dense_top_k: int = 15
    fused_top_k: int = 10
    evidence_top_k: int = 5
    max_recent_turns: int = 8
    rewrite_turn_window: int = 6
    answer_stream_chunk_size: int = 24
    rewrite_timeout_seconds: int = 15

    retrieval_thresholds: RetrievalThresholds = Field(default_factory=RetrievalThresholds)
    ocr_thresholds: OCRThresholds = Field(default_factory=OCRThresholds)

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    api_base_url: str = "http://localhost:8000"

    # Stage-2 ParadeDB/Postgres backend selection. memory stays the default and
    # needs no database dependency. An explicit paradedb selection must fail
    # loudly at startup rather than silently falling back to memory. These
    # read the plain (unprefixed) env var names, same pattern as GEMINI_API_KEY
    # above, so they match the documented RAG_BACKEND/DATABASE_URL/... names.
    rag_backend: Literal["memory", "paradedb"] = Field(
        default_factory=lambda: os.getenv("RAG_BACKEND", "memory")
    )
    database_url: Optional[str] = Field(default_factory=lambda: os.getenv("DATABASE_URL"))
    # Stage-2.5: every ParadeDB write/query is scoped by these. A single
    # RAGEngine instance serves one tenant/mailbox; there is no unscoped
    # production retrieval method.
    tenant_id: str = Field(default_factory=lambda: os.getenv("TENANT_ID", "default"))
    mailbox_id: str = Field(default_factory=lambda: os.getenv("MAILBOX_ID", "default"))
    # Must match the vector column dimension in the migration and the encoder
    # actually used (HashingEncoder/MiniLM-L6-v2 both emit 384-dim vectors).
    # Changing this requires a migration + re-embedding backfill, not a config
    # edit against mixed-dimension rows.
    embedding_model_id: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    hybrid_lexical_weight: float = Field(default_factory=lambda: float(os.getenv("HYBRID_LEXICAL_WEIGHT", "1.0")))
    hybrid_dense_weight: float = Field(default_factory=lambda: float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0")))
    hybrid_rrf_k: int = Field(default_factory=lambda: int(os.getenv("HYBRID_RRF_K", "60")))
    hybrid_candidate_limit: int = Field(default_factory=lambda: int(os.getenv("HYBRID_CANDIDATE_LIMIT", "40")))

    # Stage-3 Gmail sync. All optional: RAG_BACKEND=memory needs none of it,
    # and nothing here is read unless a mailbox is actually being connected or
    # synced. Same unprefixed-env-var pattern as the Stage-2 settings above.
    gmail_client_id: Optional[str] = Field(default_factory=lambda: os.getenv("GMAIL_CLIENT_ID"))
    gmail_client_secret: Optional[str] = Field(default_factory=lambda: os.getenv("GMAIL_CLIENT_SECRET"))
    gmail_redirect_uri: Optional[str] = Field(default_factory=lambda: os.getenv("GMAIL_REDIRECT_URI"))
    gmail_pubsub_topic: Optional[str] = Field(default_factory=lambda: os.getenv("GMAIL_PUBSUB_TOPIC"))
    gmail_pubsub_subscription: Optional[str] = Field(
        default_factory=lambda: os.getenv("GMAIL_PUBSUB_SUBSCRIPTION")
    )
    gmail_pubsub_audience: Optional[str] = Field(default_factory=lambda: os.getenv("GMAIL_PUBSUB_AUDIENCE"))
    gmail_pubsub_service_account: Optional[str] = Field(
        default_factory=lambda: os.getenv("GMAIL_PUBSUB_SERVICE_ACCOUNT")
    )
    # Base64 32-byte AES-256 key for refresh-token encryption at rest. Rotating
    # it means re-consenting every mailbox (the stored ciphertext is not
    # re-encryptable without the old key), hence the key_id column.
    gmail_token_encryption_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GMAIL_TOKEN_ENCRYPTION_KEY")
    )
    gmail_token_key_id: str = Field(default_factory=lambda: os.getenv("GMAIL_TOKEN_KEY_ID", "local"))

    # Stage-4 LLM contextualization. Disabled by default and inert when off:
    # ingestion enqueues nothing, no provider is constructed, and no LLM package
    # is imported. Any OpenAI-compatible endpoint works; nothing is hard-coded
    # to a particular model or runtime, and no model is ever downloaded.
    context_enabled: bool = Field(
        default_factory=lambda: os.getenv("CONTEXT_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    context_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("CONTEXT_BASE_URL") or os.getenv("MEDHA_BASE_URL")
    )
    context_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("CONTEXT_MODEL") or os.getenv("MEDHA_MODEL")
    )
    # Secret: read from the environment only, never written to a config file and
    # never rendered into a log line, an error, or an API response.
    context_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("CONTEXT_API_KEY") or os.getenv("MEDHA_API_KEY")
    )
    context_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("CONTEXT_TIMEOUT_SECONDS", "30"))
    )
    context_max_tokens: int = Field(default_factory=lambda: int(os.getenv("CONTEXT_MAX_TOKENS", "96")))
    # Part of the job fingerprint: bumping it re-contextualizes every chunk.
    context_prompt_version: Optional[str] = Field(
        default_factory=lambda: os.getenv("CONTEXT_PROMPT_VERSION")
    )

    # Stage-5 evidence graph extraction. Disabled by default and inert when off:
    # ingestion enqueues nothing, no provider is constructed, and no LLM package
    # is imported. Same OpenAI-compatible seam as Stage 4; reuses the MEDHA_*
    # fallbacks so one endpoint can serve both. Graph results are built and
    # queryable but not wired into the answer path or query router.
    graph_extraction_enabled: bool = Field(
        default_factory=lambda: os.getenv("GRAPH_EXTRACTION_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    graph_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPH_BASE_URL") or os.getenv("MEDHA_BASE_URL")
    )
    graph_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPH_MODEL") or os.getenv("MEDHA_MODEL")
    )
    # Secret: environment only, never written to a config file or a log line.
    graph_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPH_API_KEY") or os.getenv("MEDHA_API_KEY")
    )
    graph_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("GRAPH_TIMEOUT_SECONDS", "60"))
    )
    graph_max_tokens: int = Field(default_factory=lambda: int(os.getenv("GRAPH_MAX_TOKENS", "800")))
    # Both are folded into the job fingerprint: bumping either re-extracts.
    graph_prompt_version: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPH_PROMPT_VERSION")
    )
    graph_schema_version: Optional[str] = Field(
        default_factory=lambda: os.getenv("GRAPH_SCHEMA_VERSION")
    )

    # Stage-6 deterministic query planner + evidence-backed graph retrieval.
    # Enabled by default but inert without graph data: with no graph rows every
    # route resolves to zero citable chunks and falls back to the existing
    # hybrid retriever, so an existing deployment retrieves byte-identically.
    # The planner is deterministic -- never an LLM, embeddings, spaCy, or a
    # network call -- and only the ParadeDB retrieval path consults it; the
    # memory backend is untouched.
    graph_planner_enabled: bool = Field(
        default_factory=lambda: os.getenv("GRAPH_PLANNER_ENABLED", "true").lower() in ("1", "true", "yes")
    )
    # Bounded weight for the graph-evidence branch in the reused weighted-RRF
    # fusion (bm25 + dense + graph). 1.0 keeps it a peer of the lexical/dense
    # branches; lower de-emphasizes graph-sourced chunks.
    graph_branch_weight: float = Field(default_factory=lambda: float(os.getenv("GRAPH_BRANCH_WEIGHT", "1.0")))
    graph_candidate_limit: int = Field(default_factory=lambda: int(os.getenv("GRAPH_CANDIDATE_LIMIT", "20")))
    graph_temporal_candidate_limit: int = Field(
        default_factory=lambda: int(os.getenv("GRAPH_TEMPORAL_CANDIDATE_LIMIT", "10"))
    )

    # Stage-7 grounded answering + bounded Self-RAG. Disabled by default and
    # import-light when off: no provider is constructed and no HTTP client is
    # imported, so the memory/deterministic answer path is byte-identical.
    # Reuses the same OpenAI-compatible seam as Stage 4/5 (MEDHA_* fallbacks).
    # The retry ceiling is fixed at two attempts in code (GroundedAnswerer);
    # it is intentionally not configurable. Graph facts remain retrieval cues:
    # answers cite the underlying email chunks, never synthetic fact rows.
    answer_generation_enabled: bool = Field(
        default_factory=lambda: os.getenv("ANSWER_GENERATION_ENABLED", "false").lower() in ("1", "true", "yes")
    )
    answer_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANSWER_BASE_URL") or os.getenv("MEDHA_BASE_URL")
    )
    answer_model: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANSWER_MODEL") or os.getenv("MEDHA_MODEL")
    )
    # Secret: environment only, never written to a config file or a log line.
    answer_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANSWER_API_KEY") or os.getenv("MEDHA_API_KEY")
    )
    answer_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("ANSWER_TIMEOUT_SECONDS", "60"))
    )
    answer_max_tokens: int = Field(default_factory=lambda: int(os.getenv("ANSWER_MAX_TOKENS", "800")))
    # How many clean, deduplicated evidence chunks feed the LLM per attempt.
    # Bounded; the one retry widens retrieval to twice this.
    answer_evidence_budget: int = Field(
        default_factory=lambda: int(os.getenv("ANSWER_EVIDENCE_BUDGET", "6"))
    )

    # Stage-8 PDF attachment extraction + local OCR + page-level citations.
    # PDF-only by design. Extraction runs in the Gmail extraction worker, never
    # on the sync path, so a slow parse/OCR never blocks Gmail sync. Enqueuing
    # attachment jobs is enabled by default (it is deterministic, non-LLM work),
    # but it only ever fires on the Postgres-backed Gmail path; the memory
    # backend enqueues nothing and imports none of it.
    attachment_extraction_enabled: bool = Field(
        default_factory=lambda: os.getenv("ATTACHMENT_EXTRACTION_ENABLED", "true").lower()
        in ("1", "true", "yes")
    )
    # Oversized guard: attachments larger than this are rejected safely
    # (extraction_status='failed', reason 'oversized') and never enter retrieval.
    attachment_max_bytes: int = Field(
        default_factory=lambda: int(os.getenv("ATTACHMENT_MAX_BYTES", str(20_000_000)))
    )
    # OCR fallback for image-only/scanned PDF pages. Off by default and optional:
    # the Tesseract backend lives behind the 'ocr' extra. When disabled or
    # unavailable, a page with no usable native text is recorded as unavailable
    # rather than having text invented for it.
    attachment_ocr_enabled: bool = Field(
        default_factory=lambda: os.getenv("ATTACHMENT_OCR_ENABLED", "false").lower()
        in ("1", "true", "yes")
    )

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.index_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
