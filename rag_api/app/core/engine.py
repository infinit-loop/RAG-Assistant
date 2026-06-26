"""Core RAG engine. Loads the corpus into a FAISS store and produces grounded,
guardrailed answers. Structured-data Q&A lives here too."""
import os
import re
import glob
import pandas as pd

from app.common.config import get_settings
from app.common.constants import (
    INJECTION_PATTERNS, VAGUE_TERMS, GENERIC_NOUNS,
    TYPE_ANSWER, TYPE_CLARIFY, TYPE_ABSTAIN, TYPE_BLOCKED,
)
from app.core.vector_store import FaissVectorStore
from app.utils.text import (chunk_text, mask_sensitive, extract_pdf_text,
                            chunk_pdf_with_pages, clean_snippet)
from app.services.llm_client import LLMClient
from app.common.logging import get_logger

log = get_logger("engine")


class RAGEngine:
    def __init__(self, docs_dir: str | None = None):
        self.cfg = get_settings()
        self.docs_dir = docs_dir or self.cfg.DOCS_DIR
        self.store = FaissVectorStore()
        self.df: pd.DataFrame | None = None
        self.csv_name: str | None = None
        self.llm = LLMClient()
        self._load_and_index()

    # ---------- load ----------
    def _load_and_index(self):
        chunks, sources = [], []
        for path in sorted(glob.glob(os.path.join(self.docs_dir, "*"))):
            name = os.path.basename(path)
            if name.endswith((".md", ".txt")):
                with open(path, encoding="utf-8") as f:
                    for c, s in chunk_text(f.read(), name):
                        chunks.append(c)
                        sources.append(s)
            elif name.endswith(".csv"):
                self.df = pd.read_csv(path)
                self.csv_name = name
                cols = ", ".join(self.df.columns)
                chunks.append(
                    f"Structured operational report '{name}' with columns: "
                    f"{cols}. Contains {len(self.df)} rows of branch/SKU "
                    f"inventory and sales data."
                )
                sources.append(name)
        if chunks:
            self.store.build(chunks, sources)

    # ---------- guardrail checks ----------
    def detect_injection(self, q: str) -> bool:
        return any(p.search(q) for p in INJECTION_PATTERNS)

    def is_ambiguous(self, q: str) -> bool:
        words = re.findall(r"\w+", q.lower())
        if len(words) < self.cfg.AMBIGUITY_MIN_WORDS:
            return True
        vague_hits = sum(w in VAGUE_TERMS for w in words)
        content = [w for w in words if len(w) > 4 and w not in VAGUE_TERMS]
        has_specific_anchor = any(w not in GENERIC_NOUNS for w in content)
        return vague_hits >= 1 and not has_specific_anchor

    # ---------- main answer ----------
    def answer(self, question: str, store: FaissVectorStore | None = None,
               skip_ambiguity: bool = False) -> dict:
        """Answer a question. If `store` is given (e.g. a session's uploaded-doc
        index) it is searched instead of the base corpus.
        `skip_ambiguity` bypasses the clarification gate. The agent sets this
        because its router has already classified the message as a document
        question; the per-answer ambiguity check would otherwise second-guess a
        decision that was already made and wrongly demand clarification for
        clear questions like "what is northwind".
        """
        q = question.strip()
        store = store or self.store

        # Guardrail: prompt-injection mitigation
        if self.detect_injection(q):
            return {
                "type": TYPE_BLOCKED,
                "answer": ("This request looks like an attempt to override the "
                           "assistant's instructions or extract restricted data. "
                           "I can only answer questions grounded in the business "
                           "documents."),
                "sources": [], "confidence": None, "score": None,
            }

        # Ambiguity -> clarification flow (skipped when the agent already routed)
        if not skip_ambiguity and self.is_ambiguous(q):
            return {
                "type": TYPE_CLARIFY,
                "answer": ("Your question is a bit ambiguous. Could you clarify:\n"
                           "  - Which document, report, or process are you asking about?\n"
                           "  - What specifically do you want to know (definition, "
                           "steps, owner, threshold, a number)?\n"
                           "  - Any timeframe or scope that matters?"),
                "sources": [], "confidence": None, "score": None,
            }

        hits = store.search(q, k=self.cfg.TOP_K)
        top = hits[0][2] if hits else 0.0
        log.debug("retrieval: top=%.3f hits=%s", top,
                  [(s, round(sc, 3)) for _, s, sc in hits])

        # Guardrail: min retrieval threshold + lexical coverage -> abstain
        q_terms = {w for w in re.findall(r"[a-z]{4,}", q.lower())
                   if w not in {"what", "explain", "process", "policy", "company"}}
        top_text = hits[0][0].lower() if hits else ""
        covered = any(t in top_text for t in q_terms) if q_terms else True
        if top < self.cfg.MIN_RETRIEVAL_SCORE or not covered:
            log.info("answer: ABSTAIN (top=%.3f < %.3f or not covered=%s)",
                     top, self.cfg.MIN_RETRIEVAL_SCORE, covered)
            return {
                "type": TYPE_ABSTAIN,
                "answer": ("I couldn't find supporting information in the provided "
                           "documents, so I won't guess. Try rephrasing, or this "
                           "topic may not be covered by the current document set."),
                "sources": [], "confidence": None, "score": round(top, 3),
            }

        # Confidence / uncertainty handling
        confidence = "high" if top >= self.cfg.CONFIDENT_SCORE else "low"
        used = [h for h in hits if h[2] >= self.cfg.MIN_RETRIEVAL_SCORE]
        snippets = [mask_sensitive(c) for c, s, sc in used]

        # Grounded generation: if an LLM is configured, synthesize a fluent
        # answer strictly from the (masked) snippets. Otherwise fall back to
        # returning the snippets directly (extractive).
        mode = "extractive"
        if self.llm.enabled:
            try:
                body = self.llm.generate(q, snippets)
                body = mask_sensitive(body)  # mask anything the model echoed
                mode = "generated"
            except Exception as e:  # noqa: BLE001
                log.warning("answer: LLM generate failed (%s) -> extractive", e)
                body = self._format_extractive(used)
        else:
            body = self._format_extractive(used)
        log.info("answer: %s confidence=%s top=%.3f used=%d snippet(s)",
                 mode, confidence, top, len(used))

        if confidence == "low":
            if mode == "generated":
                # The LLM synthesised an answer from the retrieved snippets; a
                # low TF-IDF score on a short/rare-word query doesn't mean the
                # answer is wrong, so keep a light note rather than a banner.
                body = body + ("\n\n(Note: retrieval confidence was low for this "
                               "phrasing - double-check against the source if "
                               "it's important.)")
            else:
                body = ("Low confidence - the evidence below is only loosely "
                        "related to your question. Based on what is "
                        "available:\n\n" + body +
                        "\n\nWhat may be missing: a document that directly "
                        "addresses this exact question. Please verify before "
                        "relying on it.")

        return {
            "type": TYPE_ANSWER,
            "confidence": confidence,
            "score": round(top, 3),
            "mode": mode,
            "answer": body,
            "sources": [{"source": s, "score": round(sc, 3)} for c, s, sc in used],
        }

    @staticmethod
    def _format_extractive(used: list) -> str:
        """Render retrieved hits as readable, source-attributed excerpts*
        """
        blocks = []
        for c, s, sc in used:
            text = clean_snippet(mask_sensitive(c))
            quoted = "\n".join(f"> {ln}" for ln in text.splitlines())
            blocks.append(f"{quoted}\n>\n> — *{s}*")
        return "\n\n".join(blocks)

    # ---------- session / uploaded-document support ----------
    def build_store_from_files(self, file_paths: list[str]) -> tuple[FaissVectorStore, int]:
        """Build a standalone FAISS store from a list of text/markdown/csv files.
        Used for session-scoped uploaded documents. Returns (store, n_chunks)."""
        chunks, sources = [], []
        for path in file_paths:
            name = os.path.basename(path)
            try:
                if name.lower().endswith(".csv"):
                    df = pd.read_csv(path)
                    cols = ", ".join(map(str, df.columns))
                    chunks.append(f"Uploaded spreadsheet '{name}' with columns: "
                                  f"{cols}. {len(df)} rows.")
                    sources.append(name)
                    # also index a sample of rows as text for retrieval
                    for _, row in df.head(50).iterrows():
                        chunks.append("; ".join(f"{c}={row[c]}" for c in df.columns))
                        sources.append(name)
                elif name.lower().endswith(".pdf"):
                    page_chunks = chunk_pdf_with_pages(path, name)
                    if not page_chunks:
                        # No extractable text layer (likely scanned/image PDF).
                        continue
                    for c, s in page_chunks:
                        chunks.append(c)
                        sources.append(s)
                else:
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        for c, s in chunk_text(f.read(), name):
                            chunks.append(c)
                            sources.append(s)
            except Exception:
                continue
        store = FaissVectorStore()
        if chunks:
            store.build(chunks, sources)
        return store, len(chunks)

    def suggest_questions(self, store: FaissVectorStore, n: int = 5) -> list[str]:
        """Generate suggested questions about the documents in `store`.
        Uses the LLM if configured, otherwise a heuristic fallback."""
        # Pull a sample of the indexed text as context.
        sample = "\n".join(store._chunks[:8])[:2500]
        if self.llm.enabled and sample.strip():
            try:
                prompt = (
                    "Based only on the following document excerpts, write "
                    f"{n} concise questions a reader could ask that are answerable "
                    "from this content. Return ONLY the questions, one per line, "
                    "no numbering.\n\n" + sample
                )
                raw = self.llm.generate(prompt, [sample])
                qs = [ln.strip(" -*0123456789.").strip()
                      for ln in raw.splitlines() if ln.strip()]
                qs = [q for q in qs if q.endswith("?")][:n]
                if qs:
                    return qs
            except Exception:
                pass
        return self._heuristic_questions(store, n)

    @staticmethod
    def _heuristic_questions(store: FaissVectorStore, n: int) -> list[str]:
        """Fallback: derive questions from frequent header lines / key terms."""
        import re as _re
        from collections import Counter
        headers, words = [], Counter()
        for c in store._chunks:
            for ln in c.splitlines():
                s = ln.strip("# ").strip()
                if ln.strip().startswith("#") and 3 < len(s) < 60:
                    headers.append(s)
            for w in _re.findall(r"[A-Za-z]{5,}", c.lower()):
                words[w] += 1
        qs = [f"What does the document say about {h.lower()}?" for h in headers[:n]]
        if len(qs) < n:
            common = [w for w, _ in words.most_common(20)
                      if w not in {"document", "should", "which", "their", "these"}]
            for w in common:
                if len(qs) >= n:
                    break
                qs.append(f"What is mentioned about '{w}' in the document?")
        return qs[:n] or ["What is this document about?"]

    # ---------- structured data Q&A ----------
    def query_structured(self, intent: str) -> str:
        if self.df is None:
            return "No structured data loaded."
        d = self.df
        if intent == "top_branch_sales":
            g = d.groupby("branch")["monthly_sales_value"].sum().sort_values(ascending=False)
            return (f"Highest-sales branch: {g.index[0]} "
                    f"(${g.iloc[0]:,.0f} total monthly sales).\n" + g.to_string())
        if intent == "avg_aging":
            return f"Average inventory aging across all SKUs: {d['aging_days'].mean():.1f} days."
        if intent == "top5_aging":
            t = d.nlargest(5, "aging_days")[["sku", "branch", "aging_days"]]
            return "Top 5 SKUs by aging days:\n" + t.to_string(index=False)
        if intent == "aging_over_target":
            over = d[d["aging_days"] > 45]
            return (f"{len(over)} of {len(d)} SKUs exceed the 45-day aging target "
                    f"({len(over)/len(d)*100:.0f}%).")
        raise ValueError(f"Unknown structured intent: {intent}")

    def answer_table(self, question: str, df, table_name: str) -> dict:
        """Analytical Q&A over a DataFrame via text-to-pandas."""
        from app.core.table_qa import answer_tabular
        res = answer_tabular(self.llm, question, df, table_name)
        if res.get("ok"):
            return {
                "type": TYPE_ANSWER,
                "answer": res["answer"],
                "sources": [{"source": table_name, "score": 1.0}],
                "confidence": "high", "score": 1.0, "mode": "table",
                "query": res.get("expr"),
            }
        # An LLM transport/availability error is an infrastructure problem, not a
        # bad question — say so plainly instead of blaming the user's phrasing.
        err = res.get("error") or ""
        if "LLM call failed" in err:
            msg = ("The analytics model is currently unavailable, so I can't run "
                   "the calculation. Please check the LLM configuration and try "
                   "again.")
        else:
            msg = ("I couldn't translate that into a reliable calculation over "
                   "the data. Try rephrasing, or name the exact column you mean.")
        return {
            "type": TYPE_ABSTAIN, "answer": msg,
            "sources": [], "confidence": None, "score": None, "mode": "table",
        }

    def health(self) -> dict:
        from app.utils.pii import active_engine
        return {"indexed_chunks": self.store.size,
                "structured_rows": 0 if self.df is None else len(self.df),
                "llm_enabled": self.llm.enabled,
                "llm_provider": self.llm.provider if self.llm.enabled else None,
                "llm_model": self.llm.model if self.llm.enabled else None,
                "pii_engine": active_engine()}