"""Historical knowledge base — Chroma + Ollama embeddings.

Each ESR is decomposed into one record per criterion. Records are embedded
and stored with metadata (source, criterion, score, weight, action_type)
so retrieval can be filtered or ranked semantically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from .config import EMBED_MODEL, INPUT_DOCS_DIR, KB_DIR, OLLAMA_BASE_URL
from .ingestion import extract_text_from_path

KB_COLLECTION = "edf_historical"

_client: chromadb.api.ClientAPI | None = None
_collection: Any = None
_embeddings: OllamaEmbeddings | None = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embeddings


def _get_collection():
    global _client, _collection
    if _collection is None:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(KB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(KB_COLLECTION)
    return _collection


# ---------------------------------------------------------------------------
# ESR parsing
# ---------------------------------------------------------------------------

_CRITERION_HEADER = re.compile(
    r"Criterion\s+(\d+)\s*[-–—]\s*([^\n]+)",
    re.IGNORECASE,
)
_SCORE_RE = re.compile(r"Score[^0-9]{0,40}(\d+[.,]?\d*)", re.IGNORECASE)
_WEIGHT_RE = re.compile(r"Weight\s*[x×]\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)


def parse_esr_into_records(
    text: str,
    source: str,
    action_type: str = "Research",
) -> list[dict]:
    """Decompose an ESR document into per-criterion records."""
    matches = list(_CRITERION_HEADER.finditer(text))
    records: list[dict] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        # Criterion name = everything after the dash, before the parenthesis
        # or the "Weight" keyword, on the header line.
        header_line = m.group(0)
        name = m.group(2)
        name = re.split(r"\(|–|—|Weight", name, maxsplit=1)[0].strip(" -:")

        score_match = _SCORE_RE.search(block)
        score = (
            float(score_match.group(1).replace(",", "."))
            if score_match
            else None
        )
        weight_match = _WEIGHT_RE.search(header_line) or _WEIGHT_RE.search(block)
        weight = (
            float(weight_match.group(1).replace(",", "."))
            if weight_match
            else 1.0
        )

        # Comments: everything after the first "Comments" keyword in the block.
        comments_match = re.search(r"Comments\s*\n?(.*)", block, re.DOTALL | re.IGNORECASE)
        comments = comments_match.group(1).strip() if comments_match else block.strip()
        # Trim aggressively — Chroma metadata has size limits, and we keep
        # only the first 4k chars of evaluator commentary (already plenty).
        comments = comments[:4000]

        if not comments:
            continue
        records.append(
            {
                "source": source,
                "action_type": action_type,
                "criterion": name,
                "score": score,
                "weight": weight,
                "comment_text": comments,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Build / query
# ---------------------------------------------------------------------------

def build_kb_if_empty() -> int:
    """Seed the KB from any *ESR*.pdf files in input_docs/. Idempotent."""
    coll = _get_collection()
    if coll.count() > 0:
        return 0

    pdfs = sorted(
        list(INPUT_DOCS_DIR.glob("*ESR*.pdf"))
        + list(INPUT_DOCS_DIR.glob("*esr*.pdf"))
    )
    if not pdfs:
        return 0

    embeddings = _get_embeddings()
    added = 0
    for pdf_path in pdfs:
        text = extract_text_from_path(pdf_path)
        records = parse_esr_into_records(text, source=pdf_path.stem)
        if not records:
            continue

        texts = [
            (
                f"Criterion: {r['criterion']}\n"
                f"Score: {r['score'] if r['score'] is not None else 'n/a'}/5\n"
                f"Comments:\n{r['comment_text']}"
            )
            for r in records
        ]
        vectors = embeddings.embed_documents(texts)
        ids = [f"{pdf_path.stem}__{i}" for i in range(len(records))]
        metas = [
            {
                "source": r["source"],
                "action_type": r["action_type"],
                "criterion": r["criterion"],
                "score": float(r["score"]) if r["score"] is not None else -1.0,
                "weight": float(r["weight"]),
            }
            for r in records
        ]
        coll.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
        added += len(records)
    return added


def retrieve(criterion_name: str, action_type: str, k: int = 4) -> list[dict]:
    coll = _get_collection()
    if coll.count() == 0:
        return []
    embeddings = _get_embeddings()
    query = f"Criterion: {criterion_name}\nAction type: {action_type}"
    qvec = embeddings.embed_query(query)
    res = coll.query(query_embeddings=[qvec], n_results=k)
    out: list[dict] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[None] * len(ids)])[0]
    for i in range(len(ids)):
        out.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            }
        )
    return out


def kb_size() -> int:
    return _get_collection().count()
