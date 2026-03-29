"""
Classification endpoint — POST /api/v1/classify
"""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas.schemas import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request):
    return request.app.state.pipeline


@router.post("", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, pipeline=Depends(get_pipeline)):
    """
    Classify a document or text into a category.

    - If text provided: classify the raw text directly
    - If filename provided: retrieve and classify that document
    - Custom categories supported
    """
    if not payload.text and not payload.filename:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'text' or 'filename'",
        )

    if not pipeline["llm"].check_connection():
        raise HTTPException(status_code=503, detail="Ollama is not running.")

    t0 = time.perf_counter()
    try:
        if payload.text:
            # Classify raw text directly
            text_to_classify = payload.text
            source = "direct_input"
        else:
            # Retrieve document from index
            if not pipeline["vector_store"].is_loaded:
                raise HTTPException(status_code=400, detail="No documents indexed.")

            results = pipeline["vector_store"].search(
                f"document content from {payload.filename}", top_k=5
            )
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("source", "") == payload.filename
            ]
            if not results:
                raise HTTPException(
                    status_code=404,
                    detail=f"No content found for: {payload.filename}",
                )
            text_to_classify = " ".join([doc.page_content for doc, _ in results[:3]])
            source = payload.filename

        # Run classification
        result = pipeline["llm"].classify(
            text=text_to_classify,
            categories=payload.categories,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        return ClassifyResponse(
            category=result["category"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            all_scores=result["all_scores"],
            model=pipeline["llm"]._get_llm().model,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
