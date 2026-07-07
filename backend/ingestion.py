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


def _iter_docx_blocks(doc):
    """Yield Paragraph and Table objects in true document order.

    python-docx exposes `doc.paragraphs` and `doc.tables` as separate lists,
    which loses interleaving — tables (effort, risk, budget, deliverables…)
    would detach from the section they belong to. Walking the body XML keeps
    each table with its surrounding text.
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def _table_lines(table) -> list[str]:
    lines: list[str] = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


def _extract_docx(data: bytes) -> str:
    from docx import Document
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)
        else:
            parts.extend(_table_lines(block))
    return "\n".join(parts)


def _extract_pdf(data: bytes) -> str:
    # PyMuPDF is the fastest and never hangs — always try first.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except Exception:
        pass
    # pypdf: reliable fallback, no hang risk.
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        pass
    # pdfplumber last — can hang on complex PDFs; run in a thread with timeout.
    import concurrent.futures

    def _plumber() -> str:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_plumber)
        try:
            return fut.result(timeout=60)
        except Exception:
            return ""


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
    """Group DOCX content by its parent Heading 1/2/3 style.

    Paragraphs and tables are walked in true document order, so each table
    (effort allocation, risk register, deliverables, budget…) lands in the
    section it actually belongs to as pipe-joined rows.
    Returns [] if no styled headings are detected (caller falls back).
    """
    from docx import Document
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    sections: list[dict] = [{"title": "Preamble", "content": ""}]
    saw_heading = False

    def _push(text: str) -> None:
        if sections[-1]["content"]:
            sections[-1]["content"] += "\n" + text
        else:
            sections[-1]["content"] = text

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "").lower() if block.style else ""
            if style.startswith("heading"):
                saw_heading = True
                sections.append({"title": text, "content": "", "style": style})
            else:
                _push(text)
        else:
            for line in _table_lines(block):
                _push(line)

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
    # PyMuPDF — fastest, never hangs.
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                sections.append({"title": f"Page {i}", "content": text, "page": i})
        if sections:
            return sections
    except Exception:
        pass
    # pypdf fallback.
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append({"title": f"Page {i}", "content": text, "page": i})
        if sections:
            return sections
    except Exception:
        pass
    # pdfplumber last resort — timeout guarded.
    import concurrent.futures

    def _plumber_sections() -> list[dict]:
        import pdfplumber

        out: list[dict] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    out.append({"title": f"Page {i}", "content": text, "page": i})
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_plumber_sections)
        try:
            return fut.result(timeout=60)
        except Exception:
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
