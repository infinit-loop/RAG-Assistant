"""Logging configuration.

Call setup_logging() once at startup. Every module then uses
`logging.getLogger("app.<area>")` and the lines show up on stdout (and thus in
`docker compose logs`). Verbosity is controlled by LOG_LEVEL (DEBUG shows
retrieval scores and raw LLM expressions; INFO shows the agent's decisions).
"""
import logging
import os
import sys


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)
    # Add our handler once (avoid duplicates on reload / repeated calls).
    if not any(getattr(h, "_rag_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-5s | %(name)-14s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        handler._rag_handler = True  # marker so we don't add it twice
        root.addHandler(handler)
    # Keep third-party noise down.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(area: str) -> logging.Logger:
    """Return a logger named app.<area> (e.g. get_logger('agent'))."""
    return logging.getLogger(f"app.{area}")
