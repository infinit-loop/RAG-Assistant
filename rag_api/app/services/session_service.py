"""Session manager for uploaded documents.

Each session has its own folder under UPLOAD_DIR (persisted on a volume) and its
own in-memory FAISS store built from the files in that folder. Sessions are
isolated from the base corpus and from each other. On startup, existing session
folders on the volume are re-indexed so uploads survive restarts.

Allowed upload types: .txt, .md, .csv (plain-text formats the engine can read
without extra binary parsers).
"""
from __future__ import annotations
import os
import shutil

import pandas as pd

from app.common.config import get_settings
from app.services.rag_service import get_engine

ALLOWED_EXT = {".txt", ".md", ".csv", ".pdf"}


class SessionManager:
    def __init__(self):
        self.cfg = get_settings()
        self.root = self.cfg.UPLOAD_DIR
        os.makedirs(self.root, exist_ok=True)
        self._stores: dict[str, object] = {}   # session_id -> FaissVectorStore
        # session_id -> list[(filename, DataFrame)] for tabular (CSV) uploads,
        # used by the text-to-pandas analytical path.
        self._frames: dict[str, list[tuple[str, object]]] = {}
        self._rebuild_all()

    # ----- paths -----
    def _session_dir(self, session_id: str) -> str:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
        if not safe:
            raise ValueError("invalid session id")
        d = os.path.join(self.root, safe)
        os.makedirs(d, exist_ok=True)
        return d

    # ----- startup: re-index persisted uploads -----
    def _rebuild_all(self):
        if not os.path.isdir(self.root):
            return
        for sid in os.listdir(self.root):
            d = os.path.join(self.root, sid)
            if os.path.isdir(d):
                self._reindex(sid)

    def _reindex(self, session_id: str):
        d = self._session_dir(session_id)
        files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if os.path.splitext(f)[1].lower() in ALLOWED_EXT]
        if files:
            store, _ = get_engine().build_store_from_files(files)
            self._stores[session_id] = store
        # Load CSV uploads as DataFrames for analytical (text-to-pandas) Q&A.
        frames = []
        for path in files:
            if path.lower().endswith(".csv"):
                try:
                    frames.append((os.path.basename(path), pd.read_csv(path)))
                except Exception:
                    continue
        if frames:
            self._frames[session_id] = frames
        else:
            self._frames.pop(session_id, None)

    def get_frames(self, session_id: str) -> list[tuple[str, object]]:
        """Return [(filename, DataFrame), ...] for this session's CSV uploads."""
        return self._frames.get(session_id, [])

    # ----- API used by controllers -----
    def add_file(self, session_id: str, filename: str, data: bytes) -> dict:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            raise ValueError(f"Unsupported file type '{ext}'. "
                             f"Allowed: {', '.join(sorted(ALLOWED_EXT))}")
        max_bytes = self.cfg.MAX_UPLOAD_MB * 1024 * 1024
        if len(data) > max_bytes:
            raise ValueError(f"File too large (>{self.cfg.MAX_UPLOAD_MB} MB).")
        d = self._session_dir(session_id)
        # sanitize filename
        safe_name = os.path.basename(filename).replace("/", "_").replace("\\", "_")
        dest = os.path.join(d, safe_name)
        prev_size = self.store_size(session_id)
        with open(dest, "wb") as f:
            f.write(data)
        self._reindex(session_id)
        new_size = self.store_size(session_id)
        warning = None
        # A PDF that added no chunks has no extractable text layer (scanned/
        # image-only). Remove it and tell the user, rather than keep a dead file.
        if ext == ".pdf" and new_size <= prev_size:
            os.remove(dest)
            self._reindex(session_id)
            warning = (f"'{safe_name}' has no extractable text (it may be a "
                       "scanned/image-only PDF). OCR is not supported, so it was "
                       "not indexed.")
        result = {"session_id": session_id,
                  "files": self.list_files(session_id),
                  "indexed_chunks": self.store_size(session_id)}
        if warning:
            result["warning"] = warning
        return result

    def list_files(self, session_id: str) -> list[str]:
        d = self._session_dir(session_id)
        return [f for f in sorted(os.listdir(d))
                if os.path.splitext(f)[1].lower() in ALLOWED_EXT]

    def get_store(self, session_id: str):
        return self._stores.get(session_id)

    def store_size(self, session_id: str) -> int:
        s = self._stores.get(session_id)
        return s.size if s else 0

    def suggest(self, session_id: str, n: int = 5) -> list[str]:
        store = self.get_store(session_id)
        if not store or store.size == 0:
            return []
        return get_engine().suggest_questions(store, n)

    def answer(self, session_id: str, question: str) -> dict:
        store = self.get_store(session_id)
        if not store or store.size == 0:
            return {"type": "abstain",
                    "answer": "No uploaded documents for this session yet. "
                              "Upload a file first.",
                    "sources": [], "confidence": None, "score": None, "mode": None}
        return get_engine().answer(question, store=store, skip_ambiguity=True)

    def clear(self, session_id: str) -> None:
        d = self._session_dir(session_id)
        shutil.rmtree(d, ignore_errors=True)
        self._stores.pop(session_id, None)
        self._frames.pop(session_id, None)


_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager