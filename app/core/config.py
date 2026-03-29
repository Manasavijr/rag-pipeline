from functools import lru_cache
from typing import List, Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Service ──────────────────────────────────────────────────────────────
    APP_NAME: str = "rag-pipeline"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 8080

    # ── Ollama ───────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TIMEOUT: int = 120

    # ── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── FAISS Vector Store ───────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/indexes/faiss_index"
    FAISS_TOP_K: int = 5

    # ── Document Ingestion ───────────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".md"]

    # ── RAG ──────────────────────────────────────────────────────────────────
    MAX_CONTEXT_DOCS: int = 5
    MIN_RELEVANCE_SCORE: float = 0.3

    # ── Storage ──────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    INDEX_DIR: str = "data/indexes"

    # ── GCP ──────────────────────────────────────────────────────────────────
    GCP_PROJECT_ID: Optional[str] = None
    GCS_BUCKET_NAME: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
