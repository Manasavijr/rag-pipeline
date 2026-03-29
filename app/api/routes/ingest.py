"""
Ingestion endpoints — upload and index documents.
POST /api/v1/ingest/upload
DELETE /api/v1/ingest/clear
GET /api/v1/ingest/stats
"""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.schemas.schemas import IndexStatsResponse, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request):
    return request.app.state.pipeline


@router.post("/upload", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline=Depends(get_pipeline),
):
    """
    Upload and index a document (PDF, TXT, DOCX, MD).
    The document is chunked, embedded, and added to the FAISS index.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {settings.SUPPORTED_EXTENSIONS}",
        )

    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Save to disk temporarily
    save_path = Path(settings.RAW_DIR) / file.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(content)

    t0 = time.perf_counter()
    try:
        chunks, pages = pipeline["loader"].load(str(save_path))
        pipeline["vector_store"].add_documents(chunks)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(f"Ingested {file.filename}: {len(chunks)} chunks, {pages} pages")

        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            pages=pages,
            index_size=pipeline["vector_store"].chunk_count,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")


@router.get("/stats", response_model=IndexStatsResponse)
async def get_stats(pipeline=Depends(get_pipeline)):
    """Return current index statistics."""
    return IndexStatsResponse(
        total_documents=pipeline["vector_store"].doc_count,
        total_chunks=pipeline["vector_store"].chunk_count,
        index_size=pipeline["vector_store"].chunk_count,
        embedding_model=settings.EMBEDDING_MODEL,
        supported_formats=settings.SUPPORTED_EXTENSIONS,
    )


@router.delete("/clear")
async def clear_index(pipeline=Depends(get_pipeline)):
    """Clear the entire FAISS index. Use with caution."""
    pipeline["vector_store"].clear()
    return {"message": "Index cleared successfully", "status": "ok"}
