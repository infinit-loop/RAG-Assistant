"""Custom middleware: a simple API-key gate (access-restriction simulation)
and a catch-all that converts AppError into clean JSON responses."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.common.config import get_settings
from app.common.constants import AppError
# Paths that never require a key
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """If API_KEY is configured, require a matching `x-api-key` header.
    If API_KEY is empty, the gate is disabled (open mode for local dev)."""

    async def dispatch(self, request, call_next):
        cfg = get_settings()
        path = request.url.path
        if cfg.API_KEY and not any(path.startswith(p) for p in PUBLIC_PATHS):
            if request.headers.get("x-api-key") != cfg.API_KEY:
                return JSONResponse(status_code=401,
                                    content={"error": "Invalid or missing API key"})
        return await call_next(request)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except AppError as e:
            return JSONResponse(status_code=e.status_code, content={"error": e.message})
        except Exception:  # noqa: BLE001 - last-resort guard
            return JSONResponse(status_code=500,
                                content={"error": "Internal server error"})
