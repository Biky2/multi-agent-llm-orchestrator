import logging
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL UNIQUE,
    user_task TEXT NOT NULL,
    agent_outputs JSONB DEFAULT '{}',
    final_output TEXT DEFAULT '',
    confidence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),
    completed BOOLEAN DEFAULT FALSE
)
"""


async def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.DATABASE_URL, echo=False)
    return _engine


async def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = await get_engine()
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def close_postgres() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def ping_postgres() -> bool:
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("PostgreSQL ping failed: %s", exc)
        return False


async def init_db() -> None:
    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.execute(text(CREATE_TABLE_SQL))


async def upsert_task_history(
    session_id: str,
    user_task: str,
    agent_outputs: dict[str, Any],
    final_output: str,
    confidence_score: float,
    completed: bool,
) -> None:
    factory = await get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                """
                INSERT INTO task_history (
                    session_id, user_task, agent_outputs, final_output,
                    confidence_score, completed
                )
                VALUES (
                    :session_id, :user_task, CAST(:agent_outputs AS JSONB),
                    :final_output, :confidence_score, :completed
                )
                ON CONFLICT (session_id) DO UPDATE SET
                    user_task = EXCLUDED.user_task,
                    agent_outputs = EXCLUDED.agent_outputs,
                    final_output = EXCLUDED.final_output,
                    confidence_score = EXCLUDED.confidence_score,
                    completed = EXCLUDED.completed
                """
            ),
            {
                "session_id": session_id,
                "user_task": user_task,
                "agent_outputs": _json_dumps(agent_outputs),
                "final_output": final_output,
                "confidence_score": confidence_score,
                "completed": completed,
            },
        )
        await session.commit()


async def get_task_history(session_id: str) -> Optional[dict[str, Any]]:
    factory = await get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                """
                SELECT id, session_id, user_task, agent_outputs, final_output,
                       confidence_score, created_at, completed
                FROM task_history
                WHERE session_id = :session_id
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"session_id": session_id},
        )
        row = result.mappings().first()
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "session_id": row["session_id"],
            "user_task": row["user_task"],
            "agent_outputs": row["agent_outputs"] or {},
            "final_output": row["final_output"],
            "confidence_score": row["confidence_score"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "completed": row["completed"],
        }


def _json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data)
