from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    filename: str
    chunks_created: int
    pages: Optional[int] = None
    index_size: int
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IndexStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_size: int
    embedding_model: str
    supported_formats: List[str]


# ── Q&A ───────────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.1, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    model: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Summarization ─────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    filename: Optional[str] = None   # summarize specific doc
    query: Optional[str] = None      # focused summary around a topic
    style: str = Field("concise", pattern="^(concise|detailed|bullet)$")
    max_length: int = Field(500, ge=50, le=2000)


class SummarizeResponse(BaseModel):
    summary: str
    style: str
    source: str
    chunks_used: int
    model: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Classification ────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: Optional[str] = None
    filename: Optional[str] = None
    categories: List[str] = Field(
        default=["legal", "financial", "technical", "medical", "general"],
        min_length=2,
        max_length=10,
    )


class ClassifyResponse(BaseModel):
    category: str
    confidence: str   # high / medium / low
    reasoning: str
    all_scores: Dict[str, str]
    model: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_connected: bool
    index_loaded: bool
    total_chunks: int
    model: str
    uptime_seconds: float
