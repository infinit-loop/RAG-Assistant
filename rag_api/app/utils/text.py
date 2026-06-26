"""Stateless text helpers used by the RAG core."""
import re
from app.common.constants import RESTRICTED_TAG


def extract_pdf_text(path: str) -> str:
    """Extract the text layer from a PDF as one concatenated string.

    Uses pypdf (pure-python, no system deps). Scanned/image-only PDFs have no
    text layer and will return an empty string — those are not supported here
    (no OCR), and the caller treats an empty result as 'unreadable'.
    """
    from pypdf import PdfReader
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def extract_pdf_pages(path: str) -> list[str]:
    """Extract the text of a PDF page by page.

    Returns a list where index i holds the text of page (i+1). Used so chunks
    can be tagged with the page they came from for page-level citations.
    """
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def chunk_pdf_with_pages(path: str, source: str, max_chars: int = 550):
    """Chunk a PDF, tagging each chunk's source with its page number, e.g.
    'manual.pdf (p.7)'. Returns (chunk_text, source_with_page) tuples. If the
    PDF has no extractable text, returns an empty list."""
    pages = extract_pdf_pages(path)
    out = []
    for i, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        page_source = f"{source} (p.{i})"
        out.extend(chunk_text(page_text, page_source, max_chars))
    return out


def mask_sensitive(text: str) -> str:
    """Redact PII from any text we surface. Delegates to the configured PII
    engine (regex by default, or Presidio if PII_ENGINE=presidio)."""
    from app.utils.pii import mask
    return mask(text)


def chunk_text(text: str, source: str, max_chars: int = 550):
    """Split a document into paragraph-ish chunks.

    Lines tagged RESTRICTED are dropped at load time (access simulation).
    Long blocks are split on whitespace boundaries (never mid-word) so the
    retrieved snippets read cleanly. Returns a list of (chunk_text, source)
    tuples.
    """
    clean_lines = [ln for ln in text.splitlines() if not RESTRICTED_TAG.search(ln)]
    text = "\n".join(clean_lines)
    chunks = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())  # normalise internal whitespace
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append((block, source))
        else:
            chunks.extend((c, source) for c in _split_on_words(block, max_chars))
    return chunks


def _split_on_words(block: str, max_chars: int):
    """Greedily pack words into <= max_chars pieces without breaking words."""
    words, cur, out = block.split(), [], []
    length = 0
    for w in words:
        # +1 for the joining space (except the first word in a piece)
        extra = len(w) + (1 if cur else 0)
        if cur and length + extra > max_chars:
            out.append(" ".join(cur))
            cur, length = [w], len(w)
        else:
            cur.append(w)
            length += extra
    if cur:
        out.append(" ".join(cur))
    return out


_SENT_END = re.compile(r"[.!?]['\")\]]?$")


def clean_snippet(text: str) -> str:
    """Tidy an extracted snippet for display.

    Collapses whitespace, and because chunk boundaries can fall mid-sentence,
    marks a leading partial sentence and a trailing partial sentence with an
    ellipsis so the reader can see the quote is excerpted.
    """
    s = " ".join(text.split())
    if not s:
        return s
    # Leading fragment: starts lower-case or with a stray closing punctuation.
    if s[0].islower():
        s = "… " + s
    # Trailing fragment: doesn't end on sentence-terminating punctuation.
    if not _SENT_END.search(s):
        s = s.rstrip(",;:- ") + " …"
    return s


def content_words(text: str, stop: set[str] | None = None) -> set[str]:
    stop = stop or set()
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop}