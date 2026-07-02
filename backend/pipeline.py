from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable

import numpy as np
from langchain_ollama import OllamaEmbeddings

from . import db
from .chunking import chunk_sections
from .config import (
    CHAT_MODEL,
    EMBED_MODEL,
    MAX_PROPOSAL_CHARS_INLINE,
    MAX_WORKERS,
    NUM_CTX,
    NUM_GPU,
    OLLAMA_BASE_URL,
    PROPOSAL_CHUNK_TOP_K,
)
from .evaluator import evaluate_criterion, predict_band, summarise
from .ingestion import extract_sections, extract_text

ProgressCb = Callable[[str, float], None]

# Keyed by model name so a per-job model override (chosen in the UI) doesn't
# require a restart — each distinct model gets its own cached client.
_embeddings_cache: dict[str, OllamaEmbeddings] = {}


def _embeddings_singleton(model: str = EMBED_MODEL) -> OllamaEmbeddings:
    if model not in _embeddings_cache:
        # OllamaEmbeddings (langchain-ollama) has no main_gpu field — it
        # strictly rejects unknown kwargs, unlike ChatOllama. num_gpu/num_ctx
        # are supported and applied; GPU *device* pinning isn't available for
        # the embedding model via this client (not needed — CUDA_VISIBLE_DEVICES
        # on the host restricts Ollama to a single GPU regardless).
        _embeddings_cache[model] = OllamaEmbeddings(
            model=model, base_url=OLLAMA_BASE_URL,
            num_gpu=NUM_GPU, num_ctx=NUM_CTX,
        )
    return _embeddings_cache[model]


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
        return _embed_with_autosplit(embeddings, text[:split], depth + 1) + _embed_with_autosplit(
            embeddings, text[split:], depth + 1
        )


def _safe_embed_documents(
    embeddings: OllamaEmbeddings, texts: list[str]
) -> tuple[list[list[float]], list[int]]:
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


def build_proposal_index(
    proposal_text: str,
    proposal_bytes: bytes | None,
    proposal_suffix: str | None,
    on_chunk_embedded: Callable[[int, int], None] | None = None,
    embed_model: str = EMBED_MODEL,
) -> dict[str, Any]:
    """Build the (chunks, vectors) embedding index for a proposal.

    on_chunk_embedded(done, total) is called after each chunk is embedded so
    the caller can drive a progress bar. When None, uses faster batch mode.
    """
    if proposal_bytes is not None and proposal_suffix:
        sections = extract_sections(proposal_bytes, proposal_suffix)
    else:
        from .ingestion import _heuristic_sections
        sections = _heuristic_sections(proposal_text)

    if not sections:
        sections = [{"title": "Document", "content": proposal_text}]

    chunks = chunk_sections(sections)
    if not chunks:
        return {"chunks": [], "vectors": np.zeros((0, 1)), "parents": []}

    embeddings = _embeddings_singleton(embed_model)
    texts = [c["content"] for c in chunks]
    n = len(texts)

    if on_chunk_embedded is not None:
        # Per-chunk embedding so the caller can show a progress bar.
        all_vectors: list[list[float]] = []
        all_parents: list[int] = []
        for i, text in enumerate(texts):
            sub = _embed_with_autosplit(embeddings, text)
            all_vectors.extend(sub)
            all_parents.extend([i] * len(sub))
            on_chunk_embedded(i + 1, n)
        return {
            "chunks": chunks,
            "vectors": np.asarray(all_vectors),
            "parents": all_parents,
        }
    else:
        vectors, parents = _safe_embed_documents(embeddings, texts)
        return {
            "chunks": chunks,
            "vectors": np.asarray(vectors),
            "parents": parents,
        }


def get_proposal_excerpt(
    index: dict[str, Any],
    criterion_name: str,
    criterion_description: str,
    embed_model: str = EMBED_MODEL,
) -> str:
    """Select the top-K most relevant chunks for a criterion."""
    chunks: list[dict] = index["chunks"]
    if not chunks:
        return ""
    vectors: np.ndarray = index["vectors"]
    parents: list[int] = index["parents"]

    embeddings = _embeddings_singleton(embed_model)
    query_vec = np.asarray(
        embeddings.embed_query(f"{criterion_name}: {criterion_description}")
    )

    norms = np.linalg.norm(vectors, axis=1)
    qn = float(np.linalg.norm(query_vec)) or 1.0
    denom = np.where(norms == 0, 1.0, norms) * qn
    sims = (vectors @ query_vec) / denom

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


def warmup_llm(chat_model: str = CHAT_MODEL) -> None:
    """Load the LLM into memory. Fast no-op if already loaded."""
    from .evaluator import _chat_json_model
    try:
        _chat_json_model(chat_model).invoke("warmup")
    except Exception:
        pass


def evaluate_proposal(
    proposal_text: str,
    criteria: list[dict],
    action_type: str,
    call_id: str,
    overall_threshold: float | None = None,
    proposal_acronym: str = "PROPOSAL",
    proposal_bytes: bytes | None = None,
    proposal_suffix: str | None = None,
    chunk_index: dict[str, Any] | None = None,
    progress_cb: ProgressCb | None = None,
    persist: bool = True,
    chat_model: str = CHAT_MODEL,
    embed_model: str = EMBED_MODEL,
) -> dict:
    """Full pipeline in one call. Accepts a pre-built chunk_index to skip re-embedding."""
    if chunk_index is not None:
        index = chunk_index
    else:
        if progress_cb:
            progress_cb("Extracting text from file…", 0.02)
        if proposal_bytes is not None and not proposal_text:
            proposal_text = extract_text(proposal_bytes, proposal_suffix or "")
        if progress_cb:
            progress_cb(
                f"Extracted {len(proposal_text):,} characters. Chunking…",
                0.05,
            )

        def _embed_progress(done: int, total: int) -> None:
            if progress_cb and total > 0:
                # Map embedding progress into the 0.05–0.20 band.
                frac = 0.05 + 0.15 * (done / total)
                progress_cb(f"Embedding chunk {done}/{total}", frac)

        index = build_proposal_index(
            proposal_text, proposal_bytes, proposal_suffix,
            on_chunk_embedded=_embed_progress,
            embed_model=embed_model,
        )
        if progress_cb:
            progress_cb(
                f"Indexed {len(index['chunks'])} chunks ({len(index['vectors'])} vectors)",
                0.20,
            )

    if progress_cb:
        progress_cb("Loading LLM model — first load takes a few minutes on CPU…", 0.20)
    warmup_llm(chat_model)
    if progress_cb:
        progress_cb("LLM ready", 0.25)

    if progress_cb:
        progress_cb("Selecting relevant excerpts per criterion…", 0.25)
    excerpts: list[str] = [
        get_proposal_excerpt(index, crit["name"], crit.get("description", ""), embed_model=embed_model)
        for crit in criteria
    ]
    if progress_cb:
        progress_cb("Starting LLM evaluation…", 0.30)

    n = max(len(criteria), 1)
    completed = 0

    def _eval_one(i: int, crit: dict) -> tuple[int, dict]:
        nonlocal completed
        ev = evaluate_criterion(
            proposal_text=excerpts[i],
            criterion_name=crit["name"],
            criterion_description=crit.get("description", ""),
            weight=float(crit["weight"]),
            action_type=action_type,
            chat_model=chat_model,
        )
        completed += 1
        if progress_cb:
            progress_cb(
                f"Criteria evaluated: {completed}/{n} — {crit['name']}",
                0.30 + 0.55 * (completed / n),
            )
        return i, {
            "name": crit["name"],
            "weight": float(crit["weight"]),
            "max_score": float(crit.get("max_score", 5.0)),
            **ev,
        }

    results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=min(n, MAX_WORKERS)) as pool:
        futures = {pool.submit(_eval_one, i, crit): i for i, crit in enumerate(criteria)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            results_map[idx] = res

    results: list[dict] = [results_map[i] for i in range(n)]
    weighted_total = round(sum(r["score"] * r["weight"] for r in results), 2)
    max_weighted_total = round(sum(r["max_score"] * r["weight"] for r in results), 2)

    if progress_cb:
        progress_cb("Generating ESR-style summary…", 0.88)
    summary = summarise(results, weighted_total, max_weighted_total, overall_threshold, chat_model=chat_model)

    if progress_cb:
        progress_cb("Predicting outcome band…", 0.94)
    band = predict_band(results, weighted_total, max_weighted_total, overall_threshold, chat_model=chat_model)

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
        "chunk_count": len(index["chunks"]),
        "notes": "Internal pre-screening output. Final judgement rests with human evaluators.",
    }

    if persist:
        if progress_cb:
            progress_cb("Saving run to history…", 0.98)
        db.save_run(result)

    if progress_cb:
        progress_cb("Done", 1.0)
    return result
