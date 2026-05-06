"""Document ingestion.

Two entry points:
- `extract_text(...)` returns plain text (used by paste-section workflow and
  by the historical-KB seeding).
- `extract_sections(...)` returns rich `[{title, content, page?}]` records
  used by the per-run chunk index for retrieval. For DOCX it uses the
  built-in heading styles (Heading 1/2/3) — much more accurate than regex.
"""

from __future__ import annotations

import io
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Plain text extraction
# ---------------------------------------------------------------------------

def extract_text_from_path(path: str | Path) -> str:
    path = Path(path)
    return extract_text(path.read_bytes(), path.suffix)


def extract_text(data: bytes, suffix: str) -> str:
    suffix = (suffix or "").lower()
    if suffix in {".txt", ".md", ""}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".docx":
        return _extract_docx(data)
    if suffix == ".pdf":
        return _extract_pdf(data)
    raise ValueError(f"Unsupported file type: {suffix}")


def _extract_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pdf(data: bytes) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---------------------------------------------------------------------------
# Heading-aware section extraction
# ---------------------------------------------------------------------------

# Fallback heading detector for plain-text / PDF — numbered headings
# ("1. Excellence", "1.2 Work plan") and short ALL-CAPS lines.
HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+\S.{2,}|[A-Z][A-Z0-9 \-/&]{4,}$)"
)


def extract_sections(data: bytes, suffix: str) -> list[dict]:
    """Return [{title, content, ...}] for the document.

    For DOCX uses heading styles (most reliable); for PDF and TXT falls back
    to a heading regex over the extracted plain text.
    """
    suffix = (suffix or "").lower()
    if suffix == ".docx":
        sections = _docx_sections_by_style(data)
        if sections:
            return sections
        # Fallback if the DOCX has no styled headings
        return _heuristic_sections(_extract_docx(data))
    if suffix == ".pdf":
        return _pdf_sections(data)
    if suffix in {".txt", ".md", ""}:
        return _heuristic_sections(data.decode("utf-8", errors="replace"))
    raise ValueError(f"Unsupported file type: {suffix}")


def extract_sections_from_path(path: str | Path) -> list[dict]:
    path = Path(path)
    return extract_sections(path.read_bytes(), path.suffix)


# ---- DOCX heading-style extraction ---------------------------------------

def _docx_sections_by_style(data: bytes) -> list[dict]:
    """Group DOCX paragraphs by their parent Heading 1/2/3 style.

    Tables are appended to the most recent section as pipe-joined rows.
    Returns [] if no styled headings are detected (caller falls back).
    """
    from docx import Document

    doc = Document(io.BytesIO(data))
    sections: list[dict] = [{"title": "Preamble", "content": ""}]
    saw_heading = False

    def _push(text: str) -> None:
        if sections[-1]["content"]:
            sections[-1]["content"] += "\n" + text
        else:
            sections[-1]["content"] = text

    # Walk paragraphs in document order. Tables are interleaved by python-docx
    # only via the underlying XML; for simplicity we append all tables at the
    # end of the document under their nearest preceding heading.
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            saw_heading = True
            sections.append({"title": text, "content": "", "style": style})
        else:
            _push(text)

    # Append tables to the end of the last section (cheap heuristic; good
    # enough for evaluation purposes).
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                _push(" | ".join(cells))

    if not saw_heading:
        return []
    # Drop empty sections
    return [s for s in sections if s["content"].strip()]


# ---- PDF page-aware sections ---------------------------------------------

def _pdf_sections(data: bytes) -> list[dict]:
    """One pseudo-section per PDF page (`title` includes the page number).

    Page-level granularity is a robust default; the downstream chunker will
    further split each page into retrieval-sized pieces.
    """
    sections: list[dict] = []
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    sections.append(
                        {"title": f"Page {i}", "content": text, "page": i}
                    )
    except Exception:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(
                    {"title": f"Page {i}", "content": text, "page": i}
                )
    return sections


# ---- Heuristic for plain text / fallback ---------------------------------

def _heuristic_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    current_title = "Preamble"
    current: list[str] = []

    def _flush() -> None:
        if current:
            content = "\n".join(current).strip()
            if content:
                sections.append({"title": current_title, "content": content})

    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped and len(stripped) < 120 and HEADING_RE.match(stripped):
            _flush()
            current_title = stripped
            current = []
        else:
            current.append(raw)
    _flush()
    return sections


# Back-compat alias used elsewhere (kb.py uses extract_text_from_path; this
# stays for any other callers expecting `split_into_sections`).
def split_into_sections(text: str) -> list[dict]:
    return _heuristic_sections(text)
