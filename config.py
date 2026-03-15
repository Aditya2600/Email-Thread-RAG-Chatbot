from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent
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
    rewrite_model_name: str = "t5-small"
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
