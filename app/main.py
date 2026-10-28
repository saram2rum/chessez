from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.routers.ai import engine_service as ai_engine_service
from app.routers.ai import router as ai_router
from app.routers.health import router as health_router
from app.routers.pages import router as pages_router
from app.routers.session import router as session_router
from app.routers.ws import router as ws_router


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[1]
    frontend_dir = project_root / settings.frontend_dir

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            ai_engine_service.initialize()
        except RuntimeError as exc:
            logger.warning("AI engine warmup skipped: %s", exc)
        try:
            yield
        finally:
            ai_engine_service.close()

    application = FastAPI(title="Chessez", lifespan=lifespan)

    application.mount("/js", StaticFiles(directory=frontend_dir / "js"), name="js")
    application.mount("/css", StaticFiles(directory=frontend_dir / "css"), name="css")
    application.mount("/images", StaticFiles(directory=frontend_dir / "images"), name="images")
    application.mount("/sounds", StaticFiles(directory=frontend_dir / "sounds"), name="sounds")

    application.include_router(pages_router)
    application.include_router(health_router)
    application.include_router(session_router)
    application.include_router(ai_router)
    application.include_router(ws_router)

    return application


app = create_app()
