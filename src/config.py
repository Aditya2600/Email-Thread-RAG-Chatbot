from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        env_file=PROJECT_ROOT / ".env",
        extra="ignore",
    )

    # -------------------------
    # Paths
    # -------------------------
    project_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    raw_data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    index_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "indexes")
    log_dir: str = "runs"

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

    # -------------------------
    # Database
    # -------------------------
    db_url: str = "sqlite:///./data/rag.db"

    # -------------------------
    # Local LLM / Ollama
    # -------------------------
    ollama_base_url: str = "http://localhost:11434"
    answer_model: str = "gemma3:12b"
    planner_model: str = "gemma3:12b"

    # -------------------------
    # Models
    # -------------------------
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rewrite_model_name: str = "t5-small"

    # Deprecated compatibility knobs. Runtime model calls should use Ollama only.
    enable_cloud_rewrite: bool = False
    cloud_rewrite_provider: Optional[str] = None
    cloud_rewrite_model: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # -------------------------
    # Retrieval
    # -------------------------
    bm25_k1: float = 1.5
    bm25_b: float = 0.55
    rrf_k: int = 25

    top_k_retrieve: int = 15
    top_k_final: int = 5

    grounding_threshold: float = 0.45
    abstain_ratio: float = 0.40
    token_budget: int = 6000

    max_recent_turns: int = 8
    rewrite_turn_window: int = 6
    answer_stream_chunk_size: int = 24
    rewrite_timeout_seconds: int = 15

    retrieval_thresholds: RetrievalThresholds = Field(default_factory=RetrievalThresholds)
    ocr_thresholds: OCRThresholds = Field(default_factory=OCRThresholds)

    # -------------------------
    # API / UI
    # -------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_host: str = "0.0.0.0"
    ui_port: int = 7860
    api_base_url: str = "http://localhost:8000"

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_names(cls, values):
        if not isinstance(values, dict):
            return values
        values = dict(values)
        legacy_map = {
            "embedding_model_name": "embed_model",
            "reranker_model_name": "rerank_model",
            "bm25_top_k": "top_k_retrieve",
            "dense_top_k": "top_k_retrieve",
            "evidence_top_k": "top_k_final",
            "abstain_threshold": "abstain_ratio",
        }
        for legacy_name, canonical_name in legacy_map.items():
            if legacy_name in values and canonical_name not in values:
                values[canonical_name] = values[legacy_name]
        if "runs_dir" in values and "log_dir" not in values:
            values["log_dir"] = str(values["runs_dir"])
        return values

    @property
    def runs_dir(self) -> Path:
        path = Path(self.log_dir)
        return path if path.is_absolute() else self.project_root / path

    @property
    def embedding_model_name(self) -> str:
        return self.embed_model

    @property
    def reranker_model_name(self) -> str:
        return self.rerank_model

    @property
    def bm25_top_k(self) -> int:
        return self.top_k_retrieve

    @property
    def dense_top_k(self) -> int:
        return self.top_k_retrieve

    @property
    def evidence_top_k(self) -> int:
        return self.top_k_final

    @property
    def fused_top_k(self) -> int:
        return max(self.top_k_retrieve, self.top_k_final)

    @property
    def abstain_threshold(self) -> float:
        return self.abstain_ratio

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


settings = get_settings()
