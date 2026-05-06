"""Token-aware chunking using LangChain's RecursiveCharacterTextSplitter.

Each chunk carries the parent section title (and page number for PDFs) so
the LLM evaluator can cite where its claims came from.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, MAX_EMBED_CHARS


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        length_function=len,
        keep_separator=True,
    )


def chunk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each section into chunks <= CHUNK_SIZE_CHARS.

    Returns [{title, content, section_index, chunk_index, page?}, ...].
    Any chunk longer than MAX_EMBED_CHARS (a hard guarantee for the embedding
    model) is further split before being returned.
    """
    splitter = make_splitter()
    out: list[dict[str, Any]] = []
    for sec_idx, sec in enumerate(sections):
        content = (sec.get("content") or "").strip()
        if not content:
            continue
        for ch_idx, piece in enumerate(splitter.split_text(content)):
            piece = piece.strip()
            if not piece:
                continue
            for sub in _hard_split(piece, MAX_EMBED_CHARS):
                out.append(
                    {
                        "title": sec.get("title", f"Section {sec_idx + 1}"),
                        "content": sub,
                        "section_index": sec_idx,
                        "chunk_index": ch_idx,
                        **({"page": sec["page"]} if "page" in sec else {}),
                    }
                )
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Final guard — if RecursiveCharacterTextSplitter produced anything
    larger than the embedding budget (rare with separators=["",...]), cut
    on the nearest whitespace boundary.
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return [p for p in pieces if p]
