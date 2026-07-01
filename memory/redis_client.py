import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis

from config.settings import get_settings

logger = logging.getLogger(__name__)

_redis_client: Optional[aioredis.Redis] = None

SESSION_TTL = 3600


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def ping_redis() -> bool:
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception as exc:
        logger.warning("Redis ping failed: %s", exc)
        return False


def _session_key(session_id: str) -> str:
    return f"session:{session_id}:context"


async def get_session_state(session_id: str) -> Optional[dict[str, Any]]:
    try:
        client = await get_redis()
        raw = await client.get(_session_key(session_id))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to get session state from Redis: %s", exc)
        return None


async def set_session_state(session_id: str, state: dict[str, Any]) -> None:
    try:
        client = await get_redis()
        await client.set(
            _session_key(session_id),
            json.dumps(state),
            ex=SESSION_TTL,
        )
    except Exception as exc:
        logger.warning("Failed to set session state in Redis: %s", exc)


async def delete_session_state(session_id: str) -> None:
    try:
        client = await get_redis()
        await client.delete(_session_key(session_id))
    except Exception as exc:
        logger.warning("Failed to delete session state from Redis: %s", exc)
