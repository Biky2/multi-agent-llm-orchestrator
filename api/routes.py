import asyncio
import json
import logging
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from api.sse import create_session_queue, stream_events
from config.settings import get_settings
from graph.state import OrchestratorState
from graph.workflow import run_workflow
from memory.postgres_client import get_task_history
from memory.redis_client import get_session_state, ping_redis
from memory.postgres_client import ping_postgres

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1)
    session_id: str | None = None


class TaskResponse(BaseModel):
    session_id: str
    stream_url: str


def _fresh_state(session_id: str, user_task: str) -> OrchestratorState:
    return OrchestratorState(
        session_id=session_id,
        user_task=user_task,
        findings={},
        plan=[],
        results={},
        final_output="",
        confidence_score=0.0,
        retry_count=0,
        current_agent="",
    )


def _state_from_redis(data: dict, user_task: str) -> OrchestratorState:
    return OrchestratorState(
        session_id=data.get("session_id", ""),
        user_task=user_task or data.get("user_task", ""),
        findings=data.get("findings", {}),
        plan=data.get("plan", []),
        results=data.get("results", {}),
        final_output=data.get("final_output", ""),
        confidence_score=float(data.get("confidence_score", 0.0)),
        retry_count=int(data.get("retry_count", 0)),
        current_agent=data.get("current_agent", ""),
    )


@router.post("/task", response_model=TaskResponse)
async def submit_task(body: TaskRequest) -> TaskResponse:
    session_id = body.session_id or str(uuid.uuid4())

    existing = await get_session_state(session_id)
    if existing:
        state = _state_from_redis(existing, body.task)
        if body.task:
            state["user_task"] = body.task
    else:
        state = _fresh_state(session_id, body.task)

    create_session_queue(session_id)

    asyncio.create_task(run_workflow(state))

    return TaskResponse(
        session_id=session_id,
        stream_url=f"/stream/{session_id}",
    )


@router.get("/stream/{session_id}")
async def stream_session(session_id: str) -> EventSourceResponse:
    async def event_generator():
        async for event in stream_events(session_id):
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@router.get("/history/{session_id}")
async def get_history(session_id: str) -> dict:
    record = await get_task_history(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record


async def _check_ollama() -> str:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
            if response.status_code == 200:
                return "ok"
            return "unavailable"
    except Exception:
        return "unavailable"


@router.get("/health")
async def health_check() -> dict:
    settings = get_settings()
    redis_ok = await ping_redis()
    postgres_ok = await ping_postgres()
    ollama_status = await _check_ollama()

    return {
        "redis": "ok" if redis_ok else "error",
        "postgres": "ok" if postgres_ok else "error",
        "ollama": ollama_status,
        "huggingface_fallback": bool(settings.HUGGINGFACE_API_KEY),
    }
