"""
RAG Pipeline — FastAPI Application Entry Point
Privacy-preserving local RAG with Ollama + FAISS + LangChain
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, ingest, qa, summarize, classify
from app.core.config import settings
from app.core.logging_config import setup_logging
from pipeline.ingestion.document_loader import DocumentLoader
from pipeline.retrieval.vector_store import VectorStore
from pipeline.generation.llm_chain import LLMChainManager

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Pipeline...")

    # Create required directories
    for d in [settings.RAW_DIR, settings.PROCESSED_DIR, settings.INDEX_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Initialize pipeline components
    loader = DocumentLoader()
    vector_store = VectorStore()
    llm = LLMChainManager()

    # Try to load existing index
    if vector_store.load_existing():
        logger.info(f"Loaded existing index with {vector_store.chunk_count} chunks")
    else:
        logger.info("No existing index found — starting fresh")

    # Check Ollama connection
    if llm.check_connection():
        logger.info(f"Ollama connected — model: {settings.OLLAMA_MODEL}")
    else:
        logger.warning(
            f"Ollama not connected at {settings.OLLAMA_BASE_URL}. "
            f"Start with: ollama serve && ollama pull {settings.OLLAMA_MODEL}"
        )

    app.state.pipeline = {
        "loader": loader,
        "vector_store": vector_store,
        "llm": llm,
    }

    logger.info("RAG Pipeline ready!")
    yield

    logger.info("Shutting down RAG Pipeline...")


app = FastAPI(
    title="Local RAG & LLM Intelligence Pipeline",
    description=(
        "Privacy-preserving RAG system with local Ollama LLMs, "
        "FAISS vector search, and LangChain orchestration. "
        "No API keys required — 100% local inference."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{ms:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingestion"])
app.include_router(qa.router, prefix="/api/v1/qa", tags=["Q&A"])
app.include_router(summarize.router, prefix="/api/v1/summarize", tags=["Summarization"])
app.include_router(classify.router, prefix="/api/v1/classify", tags=["Classification"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "RAG Pipeline",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)


@app.get("/ui", include_in_schema=False)
async def custom_ui():
    from fastapi.responses import HTMLResponse
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Pipeline API</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border-bottom: 1px solid #2d3748; padding: 32px 48px; }
        .header h1 { font-size: 28px; font-weight: 700; color: #fff; }
        .header p { color: #718096; margin-top: 8px; font-size: 15px; }
        .badge { display: inline-block; background: #2d3748; color: #68d391; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-top: 12px; margin-right: 8px; }
        .badge-purple { color: #b794f4; }
        .container { max-width: 960px; margin: 0 auto; padding: 40px 48px; }
        .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 32px; }
        .card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; transition: border-color 0.2s; }
        .card:hover { border-color: #4a5568; }
        .card h3 { font-size: 16px; font-weight: 600; color: #fff; margin-bottom: 8px; }
        .card p { color: #718096; font-size: 14px; line-height: 1.6; }
        .method { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; margin-right: 8px; }
        .post { background: #276749; color: #68d391; }
        .get { background: #2a4365; color: #63b3ed; }
        .delete { background: #742a2a; color: #fc8181; }
        .endpoint { font-family: monospace; font-size: 13px; color: #a0aec0; }
        .endpoints { margin-top: 16px; display: flex; flex-direction: column; gap: 8px; }
        .ep { display: flex; align-items: center; padding: 8px 12px; background: #0f1117; border-radius: 6px; }
        .links { display: flex; gap: 16px; margin-top: 24px; }
        .btn { padding: 10px 24px; border-radius: 8px; font-size: 14px; font-weight: 600; text-decoration: none; }
        .btn-primary { background: #553c9a; color: white; }
        .btn-secondary { background: #2d3748; color: #e2e8f0; }
        .status { display: flex; align-items: center; gap: 8px; margin-top: 32px; padding: 16px 20px; background: #1a1f2e; border: 1px solid #276749; border-radius: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: #68d391; }
        .section-title { font-size: 18px; font-weight: 600; color: #fff; margin-top: 40px; margin-bottom: 4px; }
        .section-sub { color: #718096; font-size: 14px; margin-bottom: 20px; }
        .privacy-banner { background: #1a1535; border: 1px solid #553c9a; border-radius: 8px; padding: 16px 20px; margin-top: 24px; display: flex; align-items: center; gap: 12px; }
        .privacy-banner span { color: #b794f4; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 Local RAG & LLM Intelligence Pipeline</h1>
        <p>Privacy-preserving document Q&A, summarization, and classification — no API keys, 100% local inference</p>
        <span class="badge">v1.0.0</span>
        <span class="badge badge-purple">Ollama · llama3.2</span>
        <span class="badge badge-purple">FAISS · LangChain</span>
    </div>
    <div class="container">
        <div class="status">
            <div class="dot"></div>
            <span style="color:#68d391;font-weight:600;">Service Running</span>
            <span style="color:#718096;margin-left:8px;">· Embedding model loaded · FAISS index ready · Ollama connected</span>
        </div>

        <div class="privacy-banner">
            <span>🔐 <strong style="color:#b794f4;">Privacy-first:</strong> All inference runs locally — no data sent to external APIs. Upload sensitive documents safely.</span>
        </div>

        <div class="links">
            <a class="btn btn-primary" href="/docs">Swagger UI</a>
            <a class="btn btn-secondary" href="/health">Health Check</a>
            <a class="btn btn-secondary" href="https://github.com/Manasavijr/rag-pipeline" target="_blank">GitHub</a>
        </div>

        <div class="section-title">API Endpoints</div>
        <div class="section-sub">Upload documents then query them with natural language</div>

        <div class="grid">
            <div class="card">
                <h3>📄 Document Ingestion</h3>
                <p>Upload PDF, DOCX, TXT, or Markdown files. Chunks and indexes into FAISS automatically.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/ingest/upload</span></div>
                    <div class="ep"><span class="method get">GET</span><span class="endpoint">/api/v1/ingest/stats</span></div>
                    <div class="ep"><span class="method delete">DEL</span><span class="endpoint">/api/v1/ingest/clear</span></div>
                </div>
            </div>
            <div class="card">
                <h3>💬 Q&A</h3>
                <p>Ask natural language questions over indexed documents. Returns answer with source attribution and relevance scores.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/qa/ask</span></div>
                </div>
            </div>
            <div class="card">
                <h3>📝 Summarization</h3>
                <p>Generate concise, detailed, or bullet-point summaries. Focus on specific topics or entire documents.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/summarize</span></div>
                </div>
            </div>
            <div class="card">
                <h3>🏷️ Classification</h3>
                <p>Classify documents into custom categories (legal, financial, technical, etc.) with confidence scoring.</p>
                <div class="endpoints">
                    <div class="ep"><span class="method post">POST</span><span class="endpoint">/api/v1/classify</span></div>
                </div>
            </div>
        </div>

        <div class="section-title">Quick Start</div>
        <div class="section-sub">Three steps to query your documents</div>
        <div class="card" style="margin-top:0;">
            <div style="display:flex;flex-direction:column;gap:12px;">
                <div style="display:flex;align-items:flex-start;gap:16px;">
                    <span style="background:#553c9a;color:#b794f4;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">1</span>
                    <div><p style="color:#fff;font-weight:600;">Upload a document</p><code style="color:#68d391;font-size:12px;">POST /api/v1/ingest/upload  (multipart file)</code></div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:16px;">
                    <span style="background:#553c9a;color:#b794f4;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">2</span>
                    <div><p style="color:#fff;font-weight:600;">Ask a question</p><code style="color:#68d391;font-size:12px;">POST /api/v1/qa/ask  {"question": "What are the payment terms?"}</code></div>
                </div>
                <div style="display:flex;align-items:flex-start;gap:16px;">
                    <span style="background:#553c9a;color:#b794f4;border-radius:50%;width:28px;height:28px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0;">3</span>
                    <div><p style="color:#fff;font-weight:600;">Get answer with sources</p><code style="color:#68d391;font-size:12px;">← answer + source docs + relevance scores</code></div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    return HTMLResponse(content=html)
