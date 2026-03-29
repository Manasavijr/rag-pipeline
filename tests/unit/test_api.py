"""
Unit tests for RAG Pipeline API.
Run: pytest tests/ -v --asyncio-mode=auto
"""

import os
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("DEBUG", "true")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    vector_store = MagicMock()
    vector_store.is_loaded = True
    vector_store.chunk_count = 42
    vector_store.doc_count = 3

    llm = MagicMock()
    llm.check_connection.return_value = True
    llm._get_llm.return_value = MagicMock(model="llama3.2")
    llm.answer.return_value = "The contract term is 12 months."
    llm.summarize.return_value = "This document covers key financial metrics for Q3 2025."
    llm.classify.return_value = {
        "category": "legal",
        "confidence": "high",
        "reasoning": "Contains contract terminology and legal clauses.",
        "all_scores": {"legal": "high", "financial": "low", "technical": "low"},
    }

    vector_store.search.return_value = [
        (
            MagicMock(
                page_content="The contract term is 12 months with auto-renewal.",
                metadata={"source": "contract.pdf", "page": 1},
            ),
            0.92,
        )
    ]

    return {"vector_store": vector_store, "llm": llm, "loader": MagicMock()}


@pytest.fixture
def client(mock_pipeline):
    with patch(
        "pipeline.ingestion.document_loader.DocumentLoader.__init__",
        return_value=None,
    ), patch(
        "pipeline.retrieval.vector_store.VectorStore.__init__",
        return_value=None,
    ), patch(
        "pipeline.generation.llm_chain.LLMChainManager.__init__",
        return_value=None,
    ), patch(
        "pipeline.retrieval.vector_store.VectorStore.load_existing",
        return_value=False,
    ), patch(
        "pipeline.generation.llm_chain.LLMChainManager.check_connection",
        return_value=True,
    ):
        from app.main import app
        from fastapi.testclient import TestClient
        app.state.pipeline = mock_pipeline
        with TestClient(app, raise_server_exceptions=False) as c:
            app.state.pipeline = mock_pipeline
            yield c


# ── Health Tests ──────────────────────────────────────────────────────────────

def test_liveness(client):
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == "1.0.0"
    assert "ollama_connected" in data
    assert "index_loaded" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


# ── Q&A Tests ─────────────────────────────────────────────────────────────────

def test_ask_question(client, mock_pipeline):
    client.app.state.pipeline = mock_pipeline
    r = client.post(
        "/api/v1/qa/ask",
        json={"question": "What is the contract term?", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert len(data["sources"]) > 0


def test_ask_no_index(client, mock_pipeline):
    mock_pipeline["vector_store"].is_loaded = False
    client.app.state.pipeline = mock_pipeline
    r = client.post("/api/v1/qa/ask", json={"question": "What is this?"})
    assert r.status_code == 400


def test_ask_question_too_short(client):
    r = client.post("/api/v1/qa/ask", json={"question": "Hi"})
    assert r.status_code == 422


def test_ask_empty_question(client):
    r = client.post("/api/v1/qa/ask", json={"question": "   "})
    assert r.status_code == 422


# ── Summarization Tests ───────────────────────────────────────────────────────

def test_summarize(client, mock_pipeline):
    client.app.state.pipeline = mock_pipeline
    r = client.post("/api/v1/summarize", json={"style": "concise"})
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert "chunks_used" in data
    assert data["style"] == "concise"


def test_summarize_bullet_style(client, mock_pipeline):
    client.app.state.pipeline = mock_pipeline
    r = client.post("/api/v1/summarize", json={"style": "bullet", "query": "financial metrics"})
    assert r.status_code == 200


def test_summarize_invalid_style(client):
    r = client.post("/api/v1/summarize", json={"style": "invalid_style"})
    assert r.status_code == 422


# ── Classification Tests ──────────────────────────────────────────────────────

def test_classify_text(client, mock_pipeline):
    client.app.state.pipeline = mock_pipeline
    r = client.post(
        "/api/v1/classify",
        json={
            "text": "This agreement is entered into between parties for legal services.",
            "categories": ["legal", "financial", "technical"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["category"] in ["legal", "financial", "technical"]
    assert data["confidence"] in ["high", "medium", "low"]
    assert "reasoning" in data


def test_classify_no_input(client):
    r = client.post("/api/v1/classify", json={"categories": ["legal", "financial"]})
    assert r.status_code == 400


def test_classify_too_few_categories(client):
    r = client.post(
        "/api/v1/classify",
        json={"text": "some text", "categories": ["only_one"]},
    )
    assert r.status_code == 422


# ── Ingestion Stats Test ──────────────────────────────────────────────────────

def test_ingest_stats(client, mock_pipeline):
    client.app.state.pipeline = mock_pipeline
    r = client.get("/api/v1/ingest/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_chunks" in data
    assert "embedding_model" in data
    assert "supported_formats" in data
