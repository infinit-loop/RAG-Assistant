"""Service layer. Holds a single shared RAGEngine instance so the FAISS index
is built once at startup rather than per request."""
from app.core.engine import RAGEngine

_engine: RAGEngine | None = None


def init_engine() -> RAGEngine:
    """Build the engine (called on app startup)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def get_engine() -> RAGEngine:
    """FastAPI dependency: return the shared engine."""
    if _engine is None:
        return init_engine()
    return _engine
