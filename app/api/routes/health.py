import time
from fastapi import APIRouter, Request
from app.core.config import settings
from app.schemas.schemas import HealthResponse

router = APIRouter()
_START_TIME = time.time()


@router.get("", response_model=HealthResponse)
async def health(request: Request):
    pipeline = getattr(request.app.state, "pipeline", {})
    vector_store = pipeline.get("vector_store")
    llm = pipeline.get("llm")

    ollama_connected = llm.check_connection() if llm else False
    index_loaded = vector_store.is_loaded if vector_store else False
    chunk_count = vector_store.chunk_count if vector_store else 0

    status = "healthy" if (index_loaded or True) else "degraded"

    return HealthResponse(
        status=status,
        version=settings.VERSION,
        ollama_connected=ollama_connected,
        index_loaded=index_loaded,
        total_chunks=chunk_count,
        model=settings.OLLAMA_MODEL,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@router.get("/live")
async def liveness():
    return {"status": "alive"}


@router.get("/ready")
async def readiness(request: Request):
    pipeline = getattr(request.app.state, "pipeline", {})
    vector_store = pipeline.get("vector_store")
    if vector_store:
        return {"status": "ready"}
    return {"status": "not_ready"}
