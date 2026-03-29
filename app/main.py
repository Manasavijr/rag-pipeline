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
