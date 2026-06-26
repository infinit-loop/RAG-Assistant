"""Application entrypoint. Builds the FastAPI app, registers middleware and
routes, and warms the FAISS index on startup."""
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.common.config import get_settings
from app.common.logging import setup_logging
from app.middleware.security import ApiKeyMiddleware, ErrorHandlerMiddleware
from app.routes.rag_routes import router
from app.services.rag_service import init_engine
from app.services.session_service import get_session_manager


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Build the FAISS index once at startup.
    init_engine()
    # Re-index any persisted session uploads from the volume.
    get_session_manager()
    yield


def create_app() -> FastAPI:
    setup_logging()
    cfg = get_settings()
    app = FastAPI(title=cfg.APP_NAME, version=cfg.APP_VERSION, lifespan=lifespan)
    # Order matters: error handler outermost, then API-key gate.
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.HOST, port=cfg.PORT, reload=False)