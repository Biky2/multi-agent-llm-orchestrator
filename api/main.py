import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import router
from memory.postgres_client import close_postgres, init_db
from memory.redis_client import close_redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AgentFlow orchestrator")
    await init_db()
    yield
    await close_redis()
    await close_postgres()
    logger.info("AgentFlow orchestrator stopped")


app = FastAPI(
    title="AgentFlow",
    description="Multi-Agent LLM Workflow Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def root():
        from fastapi.responses import FileResponse

        return FileResponse(STATIC_DIR / "index.html")
