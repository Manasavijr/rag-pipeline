"""
Summarization endpoint — POST /api/v1/summarize
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.schemas import SummarizeRequest, SummarizeResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request):
    return request.app.state.pipeline


@router.post("", response_model=SummarizeResponse)
async def summarize(payload: SummarizeRequest, pipeline=Depends(get_pipeline)):
    """
    Summarize indexed documents.

    - If filename provided: summarize that specific document
    - If query provided: focused summary around that topic
    - Supports concise, detailed, and bullet-point styles
    """
    if not pipeline["vector_store"].is_loaded:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload documents first.",
        )

    if not pipeline["llm"].check_connection():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    t0 = time.perf_counter()
    try:
        # Build search query
        if payload.query:
            query = payload.query
        elif payload.filename:
            query = f"main topics and key information in {payload.filename}"
        else:
            query = "main topics, key findings, and important information"

        source_label = payload.filename or "all indexed documents"

        # Retrieve relevant chunks
        results = pipeline["vector_store"].search(query, top_k=10)
        if not results:
            raise HTTPException(status_code=404, detail="No relevant content found.")

        # Filter by filename if specified
        if payload.filename:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("source", "") == payload.filename
            ]
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail=f"No content found for file: {payload.filename}",
                )

        # Generate summary
        focus = payload.query or "main topics, key findings, and important conclusions"
        summary = pipeline["llm"].summarize(
            context_docs=results,
            style=payload.style,
            focus=focus,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return SummarizeResponse(
            summary=summary,
            style=payload.style,
            source=source_label,
            chunks_used=len(results),
            model=pipeline["llm"]._get_llm().model,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization error: {str(e)}")
