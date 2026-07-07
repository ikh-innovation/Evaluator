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

Calibration (from real EDF Evaluation Summary Reports): even proposals that
win funding typically score 3.5–4.5 on most criteria, each carrying explicit
"However, …" weaknesses; a 5.0 is exceptional and must be earned by clearing
every probe of the weakness scan. A run of straight 5.0s with no weaknesses
almost always means the evaluation failed to probe, not that the proposal is
flawless. Equally, do not drift to the middle: follow the findings."""


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
   IMPORTANT: "the proposal does not sufficiently detail / demonstrate /
   clarify X" IS a valid, score-relevant weakness when X is a specific element
   of THIS proposal that the criterion materially needs (e.g. "reliance on
   robust GNSS receivers as the primary navigation source is not clarified") —
   that is exactly how official ESRs phrase weaknesses. Only the fully generic
   form that names no proposal-specific X is banned.
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
5. MANDATORY weakness scan before scoring. In "weakness_scan", list the 3–5
   aspects of the visible text that a hostile expert reviewer would probe for
   THIS criterion — unsubstantiated performance figures, techniques the text
   relies on but under-specifies, internal inconsistencies, unclear
   differentiation from existing systems or prior projects, thin treatment of
   a dependency or risk the concept hinges on. For each probe set "verdict"
   to "weakness" (and carry it into the weaknesses list) or "cleared" (the
   visible text genuinely settles it — say where). Do not invent criticism: a
   probe the text settles is "cleared". But an empty weaknesses list is
   legitimate ONLY when every probe was genuinely cleared — official ESRs
   attach at least one "However, …" weakness to most criteria even for
   proposals that win funding. Do not inflate and do not deflate.
6. Score in two steps. First classify each weakness you kept:
   - SIGNIFICANT: it materially endangers the achievement of this
     criterion's objectives (e.g. no credible plan, unmanaged major risk,
     unsubstantiated core claim).
   - MINOR: it does not endanger the criterion's objectives but a real
     evaluator would still write it as a "However, …" (insufficient detail or
     substantiation of a specific material element, missing quantitative
     comparison the text itself invites, one thinly treated sub-aspect).
   Then derive band and score together: "excellent" (5.0) = every probe
   cleared, no weaknesses; 4.5 = one or two minor weaknesses; "very good"
   (4.0) = several minor weaknesses, none significant; 3.5 = many minor or
   one borderline-significant weakness; "good" (3.0) = one or two significant
   shortcomings; "fair" (2.0) = significant weaknesses across several
   aspects; "poor"/"fails" = inadequately addressed or cannot be assessed.
   Output the band in "band" and the consistent value in "score".
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
  "weakness_scan": [{{"aspect": string, "verdict": "weakness" | "cleared", "reason": string}}, ...],
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "missing_evidence": [string, ...],
  "input_gaps": [string, ...],
  "evaluator_comment": string,
  "recommendations": [string, ...]
}}

Illustration of a valid output (FICTIONAL — never reuse its entities or
wording; shown only to demonstrate the expected critical depth. An "excellent"
with empty weaknesses is equally legitimate, but only when every scan probe is
genuinely cleared):
{{"band": "very good", "score": 4.0, "confidence": "medium",
"weakness_scan": [
  {{"aspect": "claimed all-weather sensing performance", "verdict": "weakness", "reason": "the \\"98% detection rate\\" figure is given without test conditions or a comparison baseline"}},
  {{"aspect": "access to test infrastructure", "verdict": "cleared", "reason": "WP5 trials are assigned to Partner P3 at its accredited facility"}},
  {{"aspect": "fallback when the datalink is jammed", "verdict": "weakness", "reason": "autonomy mode is asserted but the handover logic is not described"}}],
"strengths": ["The division of work assigns the validation campaign to the consortium member that operates the required test infrastructure (\\"WP5 validation trials led by Partner P3 at its accredited facility\\"), matching capability to task."],
"weaknesses": ["The proposal does not sufficiently substantiate the claimed \\"98% detection rate\\": no test conditions or baseline comparison are given, although the impact argument rests on it.", "The handover between remote control and the asserted \\"full autonomous fallback\\" under datalink jamming is not described, leaving a core operational claim unsupported."],
"missing_evidence": [],
"input_gaps": [],
"evaluator_comment": "The work plan is coherent and convincingly resourced. However, the central detection-performance claim and the jamming fallback are insufficiently substantiated.",
"recommendations": ["Substantiate the detection-rate claim with test conditions and a baseline."]}}
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
            temperature=0.0,
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
            temperature=0.0,
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

    parsed.setdefault("weakness_scan", [])
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
    # STEP 0 telemetry: keep the raw model value and the applied cap alongside
    # the final capped value so runs are auditable (see FIX README v2 §STEP 0).
    parsed["model_confidence"] = (
        parsed["confidence"] if parsed["confidence"] in _CONF_ORDER else "medium"
    )
    if full_document:
        cap = "medium" if parsed["input_gaps"] else "high"
    else:
        cap = "low" if parsed["input_gaps"] else "medium"
    parsed["confidence_cap"] = cap
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
