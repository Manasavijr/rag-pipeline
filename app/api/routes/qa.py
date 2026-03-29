"""
Q&A endpoint — POST /api/v1/qa/ask
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.schemas import QuestionRequest, QuestionResponse, SourceDocument

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request):
    return request.app.state.pipeline


@router.post("/ask", response_model=QuestionResponse)
async def ask(payload: QuestionRequest, pipeline=Depends(get_pipeline)):
    """
    Ask a question over indexed documents.

    - Retrieves top-k semantically relevant chunks from FAISS
    - Sends context + question to local Ollama LLM
    - Returns answer with source attribution
    """
    if not pipeline["vector_store"].is_loaded:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload documents via /api/v1/ingest/upload first.",
        )

    if not pipeline["llm"].check_connection():
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not running. Start it with: ollama serve (then: ollama pull {pipeline['llm']._get_llm().model})",
        )

    t0 = time.perf_counter()
    try:
        # Retrieve relevant chunks
        results = pipeline["vector_store"].search(
            payload.question, top_k=payload.top_k
        )

        # Generate answer
        answer = pipeline["llm"].answer(
            question=payload.question,
            context_docs=results,
            temperature=payload.temperature,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Format sources
        sources = [
            SourceDocument(
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                score=score,
            )
            for doc, score in results
        ]

        return QuestionResponse(
            question=payload.question,
            answer=answer,
            sources=sources,
            model=pipeline["llm"]._get_llm().model,
            latency_ms=round(latency_ms, 2),
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Q&A failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Q&A error: {str(e)}")
