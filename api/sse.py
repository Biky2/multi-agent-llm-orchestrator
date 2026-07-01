import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

_queues: dict[str, asyncio.Queue] = {}


def create_session_queue(session_id: str) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    _queues[session_id] = queue
    return queue


def get_session_queue(session_id: str) -> Optional[asyncio.Queue]:
    return _queues.get(session_id)


async def push_event(session_id: str, event: dict[str, Any]) -> None:
    queue = _queues.get(session_id)
    if queue is not None:
        await queue.put(event)
    else:
        logger.warning("No queue for session %s", session_id)


def remove_session_queue(session_id: str) -> None:
    _queues.pop(session_id, None)


async def stream_events(session_id: str) -> AsyncGenerator[dict[str, Any], None]:
    queue = get_session_queue(session_id)
    if queue is None:
        yield {"agent": "system", "status": "error", "message": "Session stream not found"}
        return

    try:
        while True:
            event = await queue.get()
            yield event
            if event.get("status") in ("complete", "error"):
                break
    finally:
        remove_session_queue(session_id)
