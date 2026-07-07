from __future__ import annotations

import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from .config import (
    CHAT_MODEL,
    MAIN_GPU,
    MAX_CALL_CONTEXT_CHARS,
    MAX_PROPOSAL_CHARS_INLINE,
    NUM_CTX,
    NUM_GPU,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT_S,
)


EDF_RUBRIC = """EDF scoring rubric (apply per criterion, 0–5 in 0.5 increments):
0 — "fails". Fails to address the criterion or cannot be assessed.
1 — "poor". Inadequately addressed; serious inherent weaknesses.
2 — "fair". Broadly addressed but significant weaknesses remain.
3 — "good". Addressed well; a number of shortcomings remain.
4 — "very good". Addressed very well; a small number of shortcomings.
5 — "excellent". Addressed convincingly; only minor shortcomings, if any.

Scores of 4.5 and 5.0 are normal outcomes for well-written proposals; scores
of 2.0–3.0 are normal outcomes for weak ones. Choose the band the findings
actually support — do not drift to the middle."""


_BAND_VALUES = {
    "fails": 0.0,
    "poor": 1.0,
    "fair": 2.0,
    "good": 3.0,
    "very good": 4.0,
    "excellent": 5.0,
}


EVAL_PROMPT = """You are an experienced EDF (European Defence Fund) proposal evaluator
with engineering-level domain expertise. You assess ONE award criterion of a
Part B proposal and write findings in the voice of an official Evaluation
Summary Report (ESR): specific, evaluative, concise. You judge the technical
substance of the proposal — feasibility of its specific claims, realism of its
risks, coherence of partner roles — not merely whether document sections exist.

{rubric}

Criterion under evaluation:
- Name: {criterion_name}
- Description: {criterion_description}
- Weight: x{weight}
- Action type: {action_type}
{call_context_block}
INPUT SCOPE: {input_scope}

Proposal input (each block is labelled with its source section):
---
{proposal_text}
---

Rules:
1. Assess ONLY the criterion above.
2. Every strength and every weakness must name at least one specific element
   from the proposal text — a partner, work package, technology, subsystem,
   figure or claim — and include a short verbatim quote in double quotes.
   Self-check before output: if a finding could be written about a randomly
   chosen proposal (e.g. "no quantified metrics", "alignment with call
   objectives not explicit"), either tie it to a concrete passage or delete it.
3. Critique like a domain expert: question the feasibility of specific
   technical claims, the realism of risk treatment, possible duplication with
   prior work as described in the text, and whether each partner's role is
   justified. Prefer one sharp technical finding over three generic ones.
4. CRITICAL — distinguish the proposal from your input. Your input may be only
   part of the document (see INPUT SCOPE). If a section, table, annex or
   figure is referenced in the text but not present in your input, list it
   under "input_gaps". NEVER list it as a weakness or missing evidence, and
   NEVER reduce the score because of it. Penalise only what the visible text
   demonstrably fails to contain or handles badly.
5. Weaknesses are OPTIONAL. If the criterion is addressed convincingly,
   return an empty weaknesses list and a score of 4.5 or 5.0 — do not invent
   criticism for balance. Equally, do not soften genuine deficiencies: a weak
   criterion must receive pointed, specific criticism and a score of 3.0 or
   below. Do not inflate and do not deflate.
6. Score in two steps. First classify each weakness you kept:
   - SIGNIFICANT: it materially endangers the achievement of this
     criterion's objectives (e.g. no credible plan, unmanaged major risk,
     unsubstantiated core claim).
   - MINOR: presentational, or a detail a competently run project settles
     during execution. Wanting yet more sub-detail from a text that already
     gives the level of specificity of a well-prepared proposal (quantified
     milestones, named owners, itemised resources) is at most MINOR — more
     detail is always conceivable and its absence is not a finding.
   Then derive the band: "excellent" = no significant weaknesses (minor ones
   permitted); "very good" = no significant weaknesses but several minor
   ones; "good" = one or two significant shortcomings; "fair" = significant
   weaknesses across several aspects; "poor"/"fails" = inadequately
   addressed or cannot be assessed. Output the band in "band", then set
   "score" to the band value, or 0.5 above/below it only when the findings
   sit between two bands.
7. Confidence reflects input coverage first, evidence second:
   - "high": you saw the full proposal (or every section this criterion
     needs) and found clear evidence.
   - "medium": you saw excerpts, but they cover the aspects of this criterion.
   - "low": material this criterion needs is missing from your input
     (input_gaps), or the visible text is too vague to judge.
8. If call/topic context is provided above, assess compliance and technical
   fit against it concretely, naming the requirement concerned. If none is
   provided, do not speculate about call requirements and do not treat their
   absence as a weakness.
9. "missing_evidence" is for things the PROPOSAL itself should contain for
   this criterion but demonstrably does not anywhere in the visible text —
   never for material that merely sits outside your input.

Output STRICT JSON ONLY (no markdown, no prose) with this exact schema:

{{
  "band": "fails" | "poor" | "fair" | "good" | "very good" | "excellent",
  "score": <number, 0..5, in 0.5 increments, consistent with band>,
  "confidence": "low" | "medium" | "high",
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "missing_evidence": [string, ...],
  "input_gaps": [string, ...],
  "evaluator_comment": string,
  "recommendations": [string, ...]
}}

Illustration of a valid excellent-criterion output (FICTIONAL — never reuse
its entities or wording; shown only to demonstrate that empty weaknesses and a
score of 5.0 are legitimate):
{{"band": "excellent", "score": 5.0, "confidence": "high",
"strengths": ["The division of work assigns the validation campaign to the consortium member that operates the required test infrastructure (\\"WP5 validation trials led by Partner P3 at its accredited facility\\"), matching capability to task."],
"weaknesses": [],
"missing_evidence": [],
"input_gaps": [],
"evaluator_comment": "The work plan is coherent and convincingly resourced; only minor presentational shortcomings.",
"recommendations": []}}
"""

SUMMARY_PROMPT = """You are summarising an EDF Part B proposal evaluation in the
voice of an EDF Evaluation Summary Report.

Per-criterion findings:
{per_criterion_block}

Weighted total: {weighted_total} / {max_weighted_total}
Threshold: {threshold}

Write a concise (5-8 sentence) ESR-style summary that:
- highlights the strongest criteria,
- highlights genuine weaknesses across criteria, if any — do NOT invent
  criticism where the per-criterion findings report none,
- lists up to 3 concrete rewrite priorities before submission, only where the
  findings support them; for a uniformly strong proposal say so instead.
Return plain prose only — no JSON, no markdown."""

BAND_PROMPT = """Based on a weighted total of {weighted_total}/{max_weighted_total}
(threshold {threshold}) and per-criterion scores {scores}, classify the
proposal into EXACTLY ONE of these labels:
- "Likely funded"
- "Reserve list / borderline"
- "Likely rejected"
Return only the label, nothing else."""


# Keyed by model name so a per-job model override (chosen in the UI) doesn't
# require a restart — each distinct model gets its own cached client.
_chat_json_cache: dict[str, ChatOllama] = {}
_chat_text_cache: dict[str, ChatOllama] = {}


def _chat_json_model(model: str = CHAT_MODEL) -> ChatOllama:
    if model not in _chat_json_cache:
        _chat_json_cache[model] = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            format="json",
            timeout=REQUEST_TIMEOUT_S,
            keep_alive=-1,
            num_gpu=NUM_GPU,
            main_gpu=MAIN_GPU,
            num_ctx=NUM_CTX,
        )
    return _chat_json_cache[model]


def _chat_text_model(model: str = CHAT_MODEL) -> ChatOllama:
    if model not in _chat_text_cache:
        _chat_text_cache[model] = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            timeout=REQUEST_TIMEOUT_S,
            keep_alive=-1,
            num_gpu=NUM_GPU,
            main_gpu=MAIN_GPU,
            num_ctx=NUM_CTX,
        )
    return _chat_text_cache[model]


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    obj = re.search(r"(\{.*\})", text, re.DOTALL)
    if obj:
        return json.loads(obj.group(1))
    raise ValueError(f"Could not parse JSON from model output: {text[:400]}")


def _coerce_score(value: Any) -> float:
    try:
        s = float(value)
    except Exception:
        s = 0.0
    s = max(0.0, min(5.0, s))
    return round(s * 2) / 2


_CONF_ORDER = ("low", "medium", "high")


def _cap_confidence(conf: str, cap: str) -> str:
    conf = conf if conf in _CONF_ORDER else "medium"
    return conf if _CONF_ORDER.index(conf) <= _CONF_ORDER.index(cap) else cap


def evaluate_criterion(
    proposal_text: str,
    criterion_name: str,
    criterion_description: str,
    weight: float,
    action_type: str,
    chat_model: str = CHAT_MODEL,
    call_context: str = "",
    full_document: bool = False,
    coverage_ratio: float | None = None,
) -> dict:
    if full_document:
        input_scope = "FULL PROPOSAL — you are seeing the complete document text."
    else:
        pct = (
            f"approximately {coverage_ratio * 100:.0f}%"
            if coverage_ratio is not None
            else "an unknown fraction"
        )
        input_scope = (
            f"PARTIAL — retrieval-selected excerpts covering {pct} of the "
            "document. Sections not shown to you may well exist in the full "
            "proposal; apply rule 4 strictly."
        )

    call_context = (call_context or "").strip()[:MAX_CALL_CONTEXT_CHARS]
    call_context_block = (
        (
            "\nCall/topic context (runtime-provided; judge compliance and "
            "technical fit against it):\n---\n" + call_context + "\n---\n"
        )
        if call_context
        else "\n(No call/topic context was provided for this run.)\n"
    )

    prompt = EVAL_PROMPT.format(
        rubric=EDF_RUBRIC,
        criterion_name=criterion_name,
        criterion_description=criterion_description or "(no description)",
        weight=weight,
        action_type=action_type,
        call_context_block=call_context_block,
        input_scope=input_scope,
        proposal_text=proposal_text[:MAX_PROPOSAL_CHARS_INLINE] or "(empty proposal text)",
    )
    msg = _chat_json_model(chat_model).invoke(prompt)
    parsed = _parse_json(msg.content)

    parsed.setdefault("strengths", [])
    parsed.setdefault("weaknesses", [])
    parsed.setdefault("missing_evidence", [])
    parsed.setdefault("input_gaps", [])
    parsed.setdefault("recommendations", [])
    parsed.setdefault("evaluator_comment", "")
    parsed.setdefault("confidence", "medium")
    parsed.setdefault("band", "")
    parsed["score"] = _coerce_score(parsed.get("score", 0))

    # Confidence is capped mechanically by input coverage — the model cannot
    # report "high" on partial input, and reported input gaps force "low"
    # (respectively "medium" when the full document was supplied, where a gap
    # can only mean an external annex).
    if full_document:
        cap = "medium" if parsed["input_gaps"] else "high"
    else:
        cap = "low" if parsed["input_gaps"] else "medium"
    parsed["confidence"] = _cap_confidence(parsed["confidence"], cap)
    return parsed


def summarise(
    per_criterion: list[dict],
    weighted_total: float,
    max_weighted_total: float,
    threshold: float | None,
    chat_model: str = CHAT_MODEL,
) -> str:
    block = "\n\n".join(
        (
            f"{c['name']} (score {c['score']}/5, weight x{c['weight']}):\n"
            f"  Strengths: {'; '.join(c.get('strengths') or []) or '—'}\n"
            f"  Weaknesses: {'; '.join(c.get('weaknesses') or []) or '—'}"
        )
        for c in per_criterion
    )
    msg = _chat_text_model(chat_model).invoke(
        SUMMARY_PROMPT.format(
            per_criterion_block=block,
            weighted_total=weighted_total,
            max_weighted_total=max_weighted_total,
            threshold=threshold if threshold is not None else "—",
        )
    )
    return msg.content.strip()


_VALID_BANDS = ("Likely funded", "Reserve list / borderline", "Likely rejected")


def predict_band(
    per_criterion: list[dict],
    weighted_total: float,
    max_weighted_total: float,
    threshold: float | None,
    chat_model: str = CHAT_MODEL,
) -> str:
    msg = _chat_text_model(chat_model).invoke(
        BAND_PROMPT.format(
            weighted_total=weighted_total,
            max_weighted_total=max_weighted_total,
            threshold=threshold if threshold is not None else "—",
            scores=[c["score"] for c in per_criterion],
        )
    )
    out = msg.content.strip().strip('"').strip("'")
    for label in _VALID_BANDS:
        if label.lower() in out.lower():
            return label
    if threshold is not None and weighted_total < threshold:
        return "Likely rejected"
    ratio = weighted_total / max_weighted_total if max_weighted_total else 0.0
    return "Likely funded" if ratio >= 0.7 else "Reserve list / borderline"
