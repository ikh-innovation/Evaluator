"""Document-level relevance gate — refuse to score the wrong document.

Runs once per evaluation, after the chunk index is built and before the
per-criterion LLM fan-out. Without it the pipeline scores anything it is
given: `get_proposal_excerpt` has no similarity floor and always returns its
top-K chunks, and EVAL_PROMPT never asks whether the document is on-topic, so
an unrelated file is graded on how few weaknesses the judge can find in it —
which lands it mid-scale rather than at zero.

Two independent signals, deliberately weighted:

* An LLM classifier that sees the document's opening and the criteria, and
  decides whether this is the right KIND of document. This is the deciding
  vote: it is the only signal that can tell a proposal from a brochure that
  shares its vocabulary.
* Cosine similarity between each criterion and the best-matching chunk.
  Corroborating only — the absolute values depend on the embedding model, so
  a low score never blocks on its own. It becomes the deciding signal (via a
  deliberately low hard floor) ONLY when the LLM check is unavailable, so an
  Ollama outage degrades to a weak check rather than to no check.

The gate fails open on ambiguity. Blocking a genuine proposal is a worse
failure than scoring an off-topic one, so anything short of a confident
"wrong document" verdict proceeds with a warning attached.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    CHAT_MODEL,
    EMBED_MODEL,
    RELEVANCE_GATE_MODE,
    RELEVANCE_HARD_FLOOR_SIMILARITY,
    RELEVANCE_MIN_SIMILARITY,
    RELEVANCE_PROBE_CHARS,
)
from .evaluator import _chat_json_model, _parse_json


class IrrelevantDocumentError(RuntimeError):
    """Raised instead of returning a scorecard for an off-topic document.

    Carries the full assessment in `.report` so the caller can explain the
    refusal and offer to override it.
    """

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(report.get("message") or "Document rejected as irrelevant.")


RELEVANCE_PROMPT = """You are screening an uploaded document BEFORE it enters an
EDF (European Defence Fund) proposal evaluation. Your only job is to decide
whether this is the right KIND of document to be scored against the award
criteria below. You are NOT judging quality — that happens later.

{call_context_block}
Award criteria the document would be scored against:
{criteria_block}

Opening of the uploaded document:
---
{sample}
---

Choose exactly one verdict:

- "relevant" — the document is a research/development project proposal, or a
  substantive part of one (e.g. a Part B technical description). A proposal
  that is incomplete, thin, badly structured, poorly written or simply weak is
  STILL "relevant": low quality is scored later and is not your concern here.
  A proposal from an adjacent technical domain is still "relevant" so long as
  these criteria could sensibly be applied to it.

- "off_topic" — the document could not be scored against these criteria at
  all, because it is not a project proposal of the expected kind. Typical
  cases: a contract, invoice, CV, payslip, marketing brochure, news or journal
  article, user manual, policy paper, meeting minutes, financial statement,
  source code, creative writing, an evaluation report about someone else's
  proposal, an empty template with no project content — or a genuine project
  proposal whose subject matter is wholly unconnected to the criteria above.

Be decisive but conservative. Answer "off_topic" only when a human evaluator
would hand the file straight back as the wrong document. When it is genuinely
a project proposal, answer "relevant" even if it fits the criteria badly.

Output STRICT JSON ONLY (no markdown, no prose):
{{
  "verdict": "relevant" | "off_topic",
  "document_type": "<what the document actually appears to be, 2-6 words>",
  "reason": "<one sentence, quoting a short distinctive phrase from the sample>",
  "addresses_subject_matter": true | false
}}
"""


def _document_sample(
    chunks: list[dict], probe_chars: int = RELEVANCE_PROBE_CHARS
) -> str:
    """The document's opening, section-labelled, capped at probe_chars."""
    parts: list[str] = []
    budget = probe_chars
    for chunk in chunks:
        if budget <= 0:
            break
        block = f"[{chunk.get('title', 'Document')}]\n{chunk.get('content', '')}"
        parts.append(block[:budget])
        budget -= len(block)
    return "\n\n".join(parts)


def _criteria_block(criteria: list[dict]) -> str:
    lines: list[str] = []
    for crit in criteria:
        desc = (crit.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 300:
            desc = desc[:300] + "..."
        lines.append(
            f"- {crit.get('name', '(unnamed)')}: {desc or '(no description)'}"
        )
    return "\n".join(lines) or "- (no criteria supplied)"


def best_similarities(
    index: dict[str, Any],
    criteria: list[dict],
    embed_model: str = EMBED_MODEL,
) -> dict[str, float]:
    """Best cosine similarity between each criterion and any chunk vector.

    Computed here rather than reused from retrieval because retrieval skips
    embedding entirely when the whole document fits inline — which is exactly
    the small-document case this gate must still cover.
    """
    vectors = index.get("vectors")
    if vectors is None or len(vectors) == 0 or not criteria:
        return {}

    # Deferred import: pipeline imports this module, so a module-level import
    # of pipeline here would be circular.
    from .pipeline import _embeddings_singleton

    embeddings = _embeddings_singleton(embed_model)
    matrix = np.asarray(vectors)
    norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(norms == 0, 1.0, norms)

    out: dict[str, float] = {}
    for crit in criteria:
        name = crit.get("name", "(unnamed)")
        query = f"{name}: {crit.get('description', '')}".strip()
        try:
            query_vec = np.asarray(embeddings.embed_query(query))
        except Exception:
            continue
        query_norm = float(np.linalg.norm(query_vec)) or 1.0
        sims = (matrix @ query_vec) / (safe_norms * query_norm)
        out[name] = float(np.max(sims))
    return out


def _classify(
    sample: str,
    criteria: list[dict],
    call_context: str,
    chat_model: str = CHAT_MODEL,
) -> dict[str, Any] | None:
    """Ask the LLM what kind of document this is. None if the call fails."""
    call_context_block = (
        "Call/topic context for this evaluation:\n---\n"
        + call_context.strip()[:4000]
        + "\n---\n"
        if call_context and call_context.strip()
        else "(No call/topic context was supplied for this run.)\n"
    )
    prompt = RELEVANCE_PROMPT.format(
        call_context_block=call_context_block,
        criteria_block=_criteria_block(criteria),
        sample=sample or "(the uploaded document produced no extractable text)",
    )
    try:
        msg = _chat_json_model(chat_model).invoke(prompt)
        parsed = _parse_json(msg.content)
    except Exception:
        return None
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("relevant", "off_topic"):
        return None
    return {
        "verdict": verdict,
        "document_type": str(
            parsed.get("document_type", "") or "unidentified"
        ).strip(),
        "reason": str(parsed.get("reason", "") or "").strip(),
        "addresses_subject_matter": bool(
            parsed.get("addresses_subject_matter", True)
        ),
    }


def assess(
    index: dict[str, Any],
    criteria: list[dict],
    call_context: str = "",
    chat_model: str = CHAT_MODEL,
    embed_model: str = EMBED_MODEL,
) -> dict[str, Any]:
    """Assess whether this document may be scored. Never raises.

    Returns a report carrying `blocked` (bool) and a human-readable `message`.
    """
    chunks: list[dict] = index.get("chunks") or []
    sims = best_similarities(index, criteria, embed_model=embed_model)
    top_similarity = max(sims.values()) if sims else None

    report: dict[str, Any] = {
        "blocked": False,
        "hard_block": False,   # set when no override may proceed (unreadable file)
        "verdict": "relevant",
        "document_type": "",
        "reason": "",
        "message": "",
        "warning": "",
        "top_similarity": (
            round(top_similarity, 4) if top_similarity is not None else None
        ),
        "per_criterion_similarity": {k: round(v, 4) for k, v in sims.items()},
        "min_similarity_threshold": RELEVANCE_MIN_SIMILARITY,
        "classifier_available": False,
        "signals": [],
    }

    # An empty document can never be scored — no further signal needed. This
    # is an unprocessable input rather than a relevance judgement, so it is a
    # hard block: "evaluate anyway" cannot talk the pipeline into scoring text
    # that does not exist.
    if not chunks:
        report.update(
            blocked=True,
            hard_block=True,
            verdict="off_topic",
            document_type="empty or unreadable file",
            reason="No text could be extracted from the uploaded file.",
            message=(
                "No text could be extracted from the uploaded file, so it "
                "cannot be evaluated. Check that the file is a text-based "
                "PDF/DOCX rather than a scan or an image."
            ),
            signals=["empty_document"],
        )
        return report

    sim_low = (
        top_similarity is not None and top_similarity < RELEVANCE_MIN_SIMILARITY
    )
    sim_floor = (
        top_similarity is not None
        and top_similarity < RELEVANCE_HARD_FLOOR_SIMILARITY
    )

    verdict = _classify(
        _document_sample(chunks), criteria, call_context, chat_model=chat_model
    )
    report["classifier_available"] = verdict is not None

    sim_text = (
        f"best criterion match {top_similarity:.2f}, "
        f"expected at least {RELEVANCE_MIN_SIMILARITY:.2f}"
        if top_similarity is not None
        else "no embedding signal available"
    )

    if verdict is not None:
        report["verdict"] = verdict["verdict"]
        report["document_type"] = verdict["document_type"]
        report["reason"] = verdict["reason"]

        if verdict["verdict"] == "off_topic":
            report["signals"] = ["classifier_off_topic"] + (
                ["low_similarity"] if sim_low else []
            )
            report["blocked"] = True
            report["message"] = (
                "This document was not evaluated: it does not appear to be a "
                f"proposal for this call. It reads as {verdict['document_type']}. "
                f"{verdict['reason']} ({sim_text}.)"
            )
            return report

        if sim_low:
            # Classifier says proposal, embeddings disagree — run, but say so.
            report["signals"] = ["low_similarity"]
            report["warning"] = (
                "Relevance check: the document reads as a project proposal, but "
                f"it matches the award criteria weakly ({sim_text}). Scores may "
                "reflect a poor fit between this proposal and this call — verify "
                "the correct call and criteria were selected."
            )
        return report

    # Classifier unavailable — fall back to embeddings alone. Block only far
    # below the threshold, so an LLM outage cannot reject borderline proposals.
    report["signals"] = ["classifier_unavailable"]
    if sim_floor:
        report["signals"].append("below_hard_floor")
        report["blocked"] = True
        report["verdict"] = "off_topic"
        report["document_type"] = "unrelated to the award criteria"
        report["reason"] = (
            f"The document's best match against any criterion is "
            f"{top_similarity:.2f}, below the "
            f"{RELEVANCE_HARD_FLOOR_SIMILARITY:.2f} floor."
        )
        report["message"] = (
            "This document was not evaluated: its content is unrelated to the "
            f"award criteria ({sim_text}). The document-type check could not be "
            "run because the model was unreachable, so this decision rests on "
            "the similarity floor alone."
        )
        return report

    report["warning"] = (
        "Relevance check incomplete: the document-type check could not be run "
        "because the model was unreachable"
        + (
            f", and the document matches the criteria weakly ({sim_text})"
            if sim_low
            else ""
        )
        + ". Confirm the correct document was uploaded."
    )
    return report


def enforce(
    index: dict[str, Any],
    criteria: list[dict],
    call_context: str = "",
    chat_model: str = CHAT_MODEL,
    embed_model: str = EMBED_MODEL,
    force: bool = False,
    mode: str | None = None,
) -> dict[str, Any]:
    """Run the gate, raising IrrelevantDocumentError when it blocks.

    `force=True` (an explicit "evaluate anyway" from the user) and mode="warn"
    both downgrade a block to a warning — the caveat still travels with the
    result rather than being erased.
    """
    mode = (mode or RELEVANCE_GATE_MODE).strip().lower()
    if mode == "off":
        return {"blocked": False, "skipped": True, "warning": "", "message": ""}
    if mode not in ("block", "warn"):
        # A typo in EDF_RELEVANCE_GATE must not silently disable the gate.
        mode = "block"

    report = assess(
        index,
        criteria,
        call_context=call_context,
        chat_model=chat_model,
        embed_model=embed_model,
    )
    report["mode"] = mode
    report["forced"] = bool(force)

    if report["blocked"]:
        if report.get("hard_block") or (mode == "block" and not force):
            raise IrrelevantDocumentError(report)
        # Downgraded: keep the finding, let the run continue.
        report["blocked"] = False
        report["overridden"] = True
        report["warning"] = (
            "RELEVANCE OVERRIDE — this document did not pass the relevance "
            f"check and was evaluated anyway. {report['message']} Treat every "
            "score below as unreliable."
        )
    return report


# Stable marker prepended to a rejected job's error text, so the UI can tell a
# screened-out document from a genuine crash and offer an explicit override.
REJECTION_TAG = "[relevance-gate]"
