"""End-to-end evaluation pipeline.

For each evaluation:
  1. Build a per-run chunk index (token-aware splits over heading-aware
     sections) — embedded once, reused across criteria.
  2. For each criterion, retrieve the top-K most relevant chunks from this
     run's index AND retrieve historical ESR snippets from the long-term KB.
  3. Run the per-criterion LLM chain.
  4. Aggregate, summarise, predict band, persist to the SQLite history.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import numpy as np
from langchain_ollama import OllamaEmbeddings

from . import db, kb
from .chunking import chunk_sections
from .config import (
    EMBED_MODEL,
    MAX_PROPOSAL_CHARS_INLINE,
    OLLAMA_BASE_URL,
    PROPOSAL_CHUNK_TOP_K,
)
from .evaluator import evaluate_criterion, predict_band, summarise
from .ingestion import extract_sections, extract_text

ProgressCb = Callable[[str, float], None]

_embeddings: OllamaEmbeddings | None = None


def _embeddings_singleton() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Robust embedding (auto-halves on context-length errors)
# ---------------------------------------------------------------------------

def _embed_with_autosplit(
    embeddings: OllamaEmbeddings, text: str, depth: int = 0
) -> list[list[float]]:
    try:
        return embeddings.embed_documents([text])
    except Exception:
        if depth > 8 or len(text) < 200:
            raise
        mid = len(text) // 2
        split = text.rfind("\n\n", 0, mid)
        if split < mid // 2:
            split = text.rfind("\n", 0, mid)
        if split < mid // 2:
            split = text.rfind(" ", 0, mid)
        if split <= 0:
            split = mid
        return _embed_with_autosplit(
            embeddings, text[:split], depth + 1
        ) + _embed_with_autosplit(embeddings, text[split:], depth + 1)


def _safe_embed_documents(
    embeddings: OllamaEmbeddings, texts: list[str]
) -> tuple[list[list[float]], list[int]]:
    """Embed a batch. On failure, fall back to per-text auto-split.

    Returns (vectors, parents): vectors[k] embeds a fragment of texts[parents[k]].
    """
    try:
        return embeddings.embed_documents(texts), list(range(len(texts)))
    except Exception:
        pass
    vectors: list[list[float]] = []
    parents: list[int] = []
    for i, text in enumerate(texts):
        sub = _embed_with_autosplit(embeddings, text)
        vectors.extend(sub)
        parents.extend([i] * len(sub))
    return vectors, parents


# ---------------------------------------------------------------------------
# Per-run chunk index
# ---------------------------------------------------------------------------

def _build_chunk_index(
    proposal_text: str,
    proposal_bytes: bytes | None,
    proposal_suffix: str | None,
) -> dict[str, Any]:
    """Build the (chunks, vectors) index for this evaluation."""
    if proposal_bytes is not None and proposal_suffix:
        sections = extract_sections(proposal_bytes, proposal_suffix)
    else:
        from .ingestion import _heuristic_sections  # internal fallback

        sections = _heuristic_sections(proposal_text)

    if not sections:
        sections = [{"title": "Document", "content": proposal_text}]

    chunks = chunk_sections(sections)
    if not chunks:
        return {"chunks": [], "vectors": np.zeros((0, 1)), "parents": []}

    embeddings = _embeddings_singleton()
    texts = [c["content"] for c in chunks]
    vectors, parents = _safe_embed_documents(embeddings, texts)
    return {
        "chunks": chunks,
        "vectors": np.asarray(vectors),
        "parents": parents,
    }


def _select_relevant_text(
    index: dict[str, Any],
    criterion_name: str,
    criterion_description: str,
) -> str:
    """Pick the top-K chunks for this criterion and concatenate, ordered by
    original position so the LLM reads them in narrative order."""
    chunks: list[dict] = index["chunks"]
    if not chunks:
        return ""
    vectors: np.ndarray = index["vectors"]
    parents: list[int] = index["parents"]

    embeddings = _embeddings_singleton()
    query_vec = np.asarray(
        embeddings.embed_query(f"{criterion_name}: {criterion_description}")
    )

    norms = np.linalg.norm(vectors, axis=1)
    qn = float(np.linalg.norm(query_vec)) or 1.0
    denom = np.where(norms == 0, 1.0, norms) * qn
    sims = (vectors @ query_vec) / denom

    # Aggregate similarity per original chunk (auto-split fragments share parent).
    n_chunks = len(chunks)
    chunk_sims = np.full(n_chunks, -np.inf)
    for vec_idx, parent_idx in enumerate(parents):
        if sims[vec_idx] > chunk_sims[parent_idx]:
            chunk_sims[parent_idx] = float(sims[vec_idx])

    top_idx = np.argsort(-chunk_sims)[:PROPOSAL_CHUNK_TOP_K]
    selected = sorted(top_idx.tolist())

    blocks: list[str] = []
    total = 0
    for idx in selected:
        c = chunks[idx]
        page = f", p.{c['page']}" if "page" in c else ""
        block = f"[{c['title']}{page}]\n{c['content']}"
        if total + len(block) > MAX_PROPOSAL_CHARS_INLINE:
            break
        blocks.append(block)
        total += len(block) + 2
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_proposal(
    proposal_text: str,
    criteria: list[dict],
    action_type: str,
    call_id: str,
    overall_threshold: float | None = None,
    proposal_acronym: str = "PROPOSAL",
    proposal_bytes: bytes | None = None,
    proposal_suffix: str | None = None,
    progress_cb: ProgressCb | None = None,
    persist: bool = True,
) -> dict:
    if progress_cb:
        progress_cb("Building / loading historical knowledge base", 0.03)
    seeded = kb.build_kb_if_empty()
    kb_total = kb.kb_size()
    if progress_cb:
        progress_cb(
            f"Knowledge base ready ({kb_total} entries"
            + (f", {seeded} newly indexed" if seeded else "")
            + ")",
            0.06,
        )

    if progress_cb:
        progress_cb("Chunking proposal and computing embeddings", 0.10)
    if proposal_bytes is not None and not proposal_text:
        proposal_text = extract_text(proposal_bytes, proposal_suffix or "")
    index = _build_chunk_index(proposal_text, proposal_bytes, proposal_suffix)
    if progress_cb:
        progress_cb(
            f"Indexed {len(index['chunks'])} chunks "
            f"({len(index['vectors'])} embedding vectors)",
            0.18,
        )

    n = max(len(criteria), 1)
    results: list[dict] = []
    for i, crit in enumerate(criteria):
        label = f"Evaluating criterion {i + 1}/{n}: {crit['name']}"
        if progress_cb:
            progress_cb(label, 0.18 + 0.65 * (i / n))
        proposal_excerpt = _select_relevant_text(
            index, crit["name"], crit.get("description", "")
        )
        ev = evaluate_criterion(
            proposal_text=proposal_excerpt,
            criterion_name=crit["name"],
            criterion_description=crit.get("description", ""),
            weight=float(crit["weight"]),
            action_type=action_type,
        )
        results.append(
            {
                "name": crit["name"],
                "weight": float(crit["weight"]),
                "max_score": float(crit.get("max_score", 5.0)),
                **ev,
            }
        )

    weighted_total = round(sum(r["score"] * r["weight"] for r in results), 2)
    max_weighted_total = round(
        sum(r["max_score"] * r["weight"] for r in results), 2
    )

    if progress_cb:
        progress_cb("Generating ESR-style summary", 0.86)
    summary = summarise(results, weighted_total, max_weighted_total, overall_threshold)

    if progress_cb:
        progress_cb("Predicting outcome band", 0.94)
    band = predict_band(results, weighted_total, max_weighted_total, overall_threshold)

    confs = [r.get("confidence", "medium") for r in results]
    overall_conf = (
        "low"
        if confs.count("low") > len(confs) // 2
        else ("high" if confs.count("high") > len(confs) // 2 else "medium")
    )

    result = {
        "proposal_acronym": proposal_acronym,
        "call_id": call_id,
        "action_type": action_type,
        "criteria": results,
        "weighted_total": weighted_total,
        "max_weighted_total": max_weighted_total,
        "overall_threshold": overall_threshold,
        "esr_style_summary": summary,
        "predicted_band": band,
        "overall_confidence": overall_conf,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kb_size": kb_total,
        "chunk_count": len(index["chunks"]),
        "notes": (
            "Internal pre-screening output. Final judgement rests with "
            "human evaluators."
        ),
    }

    if persist:
        if progress_cb:
            progress_cb("Saving run to history database", 0.98)
        db.save_run(result)
    else:
        result["notes"] = (
            "Partial-text run (pasted sections) — not saved to history. "
            "Only full proposal uploads are persisted."
        )
        if progress_cb:
            progress_cb("Skipping save (partial-text run)", 0.98)

    if progress_cb:
        progress_cb("Done", 1.0)
    return result
