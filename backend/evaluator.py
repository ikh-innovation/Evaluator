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
"However, …" weaknesses. A 5.0 is exceptional but NOT unreachable, and it does
not require an empty weaknesses list — real ESRs award 5.0 to a criterion that
is answered convincingly while still attaching a "However, …". A run of
straight 5.0s almost always means the evaluation failed to probe.

DISCRIMINATION (the most important calibration point): a proposal's criteria
are NOT equally well addressed. In real ESRs the scores of one proposal spread
across roughly 3.5–5.0 — one criterion is the proposal's strongest and one is
its weakest, and the report says which. A scorecard that gives every criterion
the same score has not evaluated the proposal; it has averaged it. You assess
one criterion at a time and cannot see the others, so the discrimination has to
come from judging THIS criterion on its own merits with the full range live:
ask how convincingly this specific text answers this specific criterion, and
let the answer land wherever it lands. Do not settle on a comfortable middle
value, and in particular do not treat 4.0 as a default resting point."""


OFF_TOPIC_NOTICE = """
=== SCREENING NOTICE — READ BEFORE APPLYING THE RULES BELOW ===
Before evaluation this document was screened and identified as {document_type},
NOT a proposal responding to this call.{reason_clause}

The following OVERRIDE the numbered rules below:

(a) Rule 4 is SUSPENDED. Absent proposal content — work packages, partner
    roles, consortium structure, work plan, milestones, risk register — is NOT
    an input gap and NOT a truncation artefact. It is absent because this is
    not a proposal. Treat the criterion as unaddressed and score it so.
(b) Do NOT assume a fuller proposal exists behind your input. It does not.
    Never write "the full proposal likely contains …" or reason that missing
    material "may well exist" elsewhere in the document.
(c) The question is NOT "is the work described here good?" — it is "does THIS
    DOCUMENT, as it stands, address THIS CRITERION as a proposal for this
    call?" The test is not whether the text contains subject matter related to
    the criterion, but whether it constitutes a proposal addressing it. A
    document describing completed work is not a proposal for future work,
    however excellent that work is. Technically outstanding material that does
    not address the criterion as a proposal scores "fails" (0) — this is
    exactly what the rubric's band 0 ("fails to address the criterion or
    cannot be assessed") is for.
(d) Do not manufacture failure either. If part of this document genuinely
    responds to this criterion, say which part and score that on its merits.
    Report what is there, not what you expect to be missing.
=== END SCREENING NOTICE ===
"""


# Nominal value of each rubric band on the canonical EDF 0–5 scale. The model
# always scores on this scale (see EVAL_PROMPT); a criterion configured with a
# different max_score is rescaled in _coerce_score, not asked of the model.
_BAND_VALUES = {
    "fails": 0.0,
    "poor": 1.0,
    "fair": 2.0,
    "good": 3.0,
    "very good": 4.0,
    "excellent": 5.0,
}

# Half-steps (4.5, 3.5, …) sit deliberately between two bands — rule 6 of
# EVAL_PROMPT assigns them meaning — so a score is consistent with its band
# when it lies within half a band of the band's nominal value, not only when it
# equals it exactly.
_BAND_TOLERANCE = 0.5


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
{relevance_block}
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
   EXCEPTION: this rule is SUSPENDED when a SCREENING NOTICE appears above.
   It exists to stop truncated input being mistaken for a deficient proposal;
   it must never be used to excuse a document that is not a proposal at all.
5. MANDATORY weakness scan before scoring. In "weakness_scan", probe the
   aspects of the visible text that a hostile expert reviewer would attack for
   THIS criterion — unsubstantiated performance figures, techniques the text
   relies on but under-specifies, internal inconsistencies, unclear
   differentiation from existing systems or prior projects, thin treatment of
   a dependency or risk the concept hinges on. Probe as many aspects as the
   text genuinely invites: a thin or evasive section deserves more probes than
   a thorough one, so the NUMBER of probes is a property of this text and must
   never be a quota you fill to reach a target score. For each probe set
   "verdict" to one of:
   - "significant": a real shortcoming that materially endangers the
     achievement of this criterion's objectives (no credible plan, an
     unmanaged major risk, an unsubstantiated claim the criterion rests on, a
     required element simply absent).
   - "minor": a real shortcoming a hostile reviewer would write as a
     "However, …" but that does not endanger the criterion's objectives
     (insufficient substantiation of one material element, a missing
     quantitative comparison the text itself invites, one thinly treated
     sub-aspect).
   - "cleared": the visible text genuinely settles the probe — say where.
   Carry every "significant" and "minor" probe into the weaknesses list; never
   carry a "cleared" one. Do not invent criticism and do not soften a real
   finding into "minor" to be safe: the severity split is what separates a 3.0
   from a 4.5, so calling a genuinely dangerous gap "minor" flattens the whole
   scorecard. An empty weaknesses list is legitimate only when every probe was
   genuinely cleared.
6. Score in two steps.
   STEP A — place the band by asking how convincingly THIS text answers THIS
   criterion, judged on its positive substance: how completely the criterion's
   aspects are covered, how well evidenced and technically credible the
   answers are, and how much of it is convincing rather than merely present.
     "excellent" (5.0) — answered convincingly across every aspect; whatever
       remains is genuinely peripheral.
     "very good" (4.0) — answered very well; a small number of real
       shortcomings.
     "good" (3.0) — answered well, but a number of shortcomings remain.
     "fair" (2.0) — broadly addressed; significant weaknesses across several
       aspects.
     "poor" (1.0) / "fails" (0) — inadequately addressed, or cannot be
       assessed from the visible text.
   STEP B — place the half-step inside that band using the classified probes.
   Within a band, move UP a half-step when the shortcomings are peripheral or
   presentational and the substance is strong; move DOWN a half-step when a
   shortcoming touches something the criterion materially needs. The severity
   verdicts bound the result — these limits are enforced after you answer, so
   a score outside them will be moved:
     one "significant" probe  → 4.0 at most
     two                      → 3.5 at most
     three                    → 3.0 at most
     four or more             → 2.5 at most
     every probe "cleared", no weakness kept → 4.5 at least
   Inside those bounds, with no "significant" verdict, the choice between 4.0,
   4.5 and 5.0 is decided by how convincing the positive substance is, NOT by
   counting the minor ones.
   CRITICAL — weaknesses subtract from a level set by the proposal's positive
   substance; they do not by themselves define it. Two criteria carrying the
   same number of minor weaknesses must NOT receive the same score: the one
   whose visible text answers its criterion more completely and more
   convincingly scores higher. Scoring by weakness count alone is the single
   most common failure of this evaluation and it produces a scorecard where
   every criterion reads 4.0.
   Before you commit to a score, state in "band_rationale" why this band and
   not the one immediately above and not the one immediately below, referring
   to the substance you found — one or two sentences. If you cannot say why
   the band above is wrong, the band above is probably right.
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
  "band_rationale": string,
  "confidence": "low" | "medium" | "high",
  "weakness_scan": [{{"aspect": string, "verdict": "significant" | "minor" | "cleared", "reason": string}}, ...],
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "missing_evidence": [string, ...],
  "input_gaps": [string, ...],
  "evaluator_comment": string,
  "recommendations": [string, ...]
}}

Two illustrations of valid output (FICTIONAL — never reuse their entities or
wording; shown only to demonstrate the expected critical depth and the SHAPE of
the JSON). The bands and probe counts below are arbitrary: they are two points
on the scale, not the expected answer, and they must not anchor your own score.
Your band is whatever STEP A of rule 6 gives for the text in front of you —
including "excellent", "good", "fair" or lower when that is what it gives.

A — strong criterion, shortcomings peripheral, most probes cleared:
{{"band": "excellent", "score": 4.5, "band_rationale": "Every aspect the criterion asks for is answered with evidence and the two open points are presentational, which rules out 4.0; it is not a clean 5.0 because the baseline comparison the text itself invites is absent.", "confidence": "medium",
"weakness_scan": [
  {{"aspect": "access to test infrastructure", "verdict": "cleared", "reason": "WP5 trials are assigned to Partner P3 at its accredited facility"}},
  {{"aspect": "traceability of requirements to work packages", "verdict": "cleared", "reason": "Table 3.1 maps each capability requirement onto a numbered WP task"}},
  {{"aspect": "comparison against the incumbent system", "verdict": "minor", "reason": "the gain over the fielded baseline is asserted qualitatively although the text invites a quantitative comparison"}}],
"strengths": ["The division of work assigns the validation campaign to the consortium member that operates the required test infrastructure (\\"WP5 validation trials led by Partner P3 at its accredited facility\\"), matching capability to task.", "Requirements are traced onto tasks individually (\\"each capability requirement is mapped onto a numbered WP task in Table 3.1\\"), making the work plan auditable."],
"weaknesses": ["The claimed improvement over the fielded baseline (\\"a substantial gain over current in-service equipment\\") is not quantified, although the argument invites the comparison."],
"missing_evidence": [],
"input_gaps": [],
"evaluator_comment": "The criterion is addressed convincingly and the work is credibly resourced. However, the advantage over the in-service baseline is asserted rather than quantified.",
"recommendations": ["Quantify the gain over the fielded baseline against a named reference system."]}}

B — the criterion's core claim is unsupported, so a probe is "significant":
{{"band": "good", "score": 3.0, "band_rationale": "The concept is coherent and the consortium credible, so this is above \\"fair\\"; it cannot reach \\"very good\\" because the detection-performance figure the whole impact case rests on is unsubstantiated and the jamming fallback is undescribed.", "confidence": "medium",
"weakness_scan": [
  {{"aspect": "claimed all-weather sensing performance", "verdict": "significant", "reason": "the \\"98% detection rate\\" figure carries the impact case but is given without test conditions or a comparison baseline"}},
  {{"aspect": "fallback when the datalink is jammed", "verdict": "significant", "reason": "autonomy mode is asserted but the handover logic is not described"}},
  {{"aspect": "partner capability for the sensor payload", "verdict": "cleared", "reason": "Partner P2 is the fielded supplier of the equivalent payload"}}],
"strengths": ["The sensor payload is assigned to the consortium member that already supplies the equivalent fielded equipment (\\"Partner P2, supplier of the in-service payload\\"), so the hardware path is credible."],
"weaknesses": ["The proposal does not substantiate the claimed \\"98% detection rate\\": no test conditions or baseline comparison are given, although the impact argument rests on it.", "The handover between remote control and the asserted \\"full autonomous fallback\\" under datalink jamming is not described, leaving a core operational claim unsupported."],
"missing_evidence": ["Test conditions and baseline for the detection-rate figure."],
"input_gaps": [],
"evaluator_comment": "The concept is coherent and the payload path credibly resourced. However, the central detection-performance claim and the jamming fallback are both unsupported, and the criterion's objectives depend on them.",
"recommendations": ["Substantiate the detection-rate claim with test conditions and a baseline.", "Describe the control handover under datalink denial."]}}
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

# No BAND_PROMPT: the outcome band is arithmetic on the weighted total, not a
# model judgement — see predict_band.


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


def _coerce_score(value: Any, max_score: float = 5.0) -> float | None:
    """Model score on the canonical 0–5 rubric → the criterion's own scale.

    A criterion may be configured with a max_score other than 5 (the UI allows
    1–10). The model is never asked to reason on that scale — it applies the
    EDF rubric it knows, 0–5 in 0.5 increments, and the result is rescaled
    here. Clamping to 5 and then scaling is what keeps the score inside the
    criterion's own maximum: clamping at a literal 5.0 while max_weighted_total
    counted max_score made the ceiling unreachable at max_score > 5, and let
    weighted_total exceed max_weighted_total at max_score < 5.

    Returns None when the model emitted nothing numeric, so the caller can fall
    back to the band label instead of silently recording a 0 ("fails").
    """
    try:
        s = float(value)
    except Exception:
        return None
    s = max(0.0, min(5.0, s))
    return round(s * 2) / 2 * (max_score / 5.0)


def _band_for_score(score_0_5: float) -> str:
    """Nearest rubric band for a canonical 0–5 score; ties go to the lower band."""
    return min(
        _BAND_VALUES,
        key=lambda b: (abs(_BAND_VALUES[b] - score_0_5), _BAND_VALUES[b]),
    )


def _reconcile_band_and_score(
    band: Any, score: Any, max_score: float = 5.0
) -> tuple[str, float, bool]:
    """Force the model's band and score onto a single consistent verdict.

    The two are generated independently and drift: a "good" band carrying a 5.0
    printed as "good" in the report while the 5.0 landed in the weighted total.
    The band leads, because it is what rule 6 derives from the classified
    weaknesses; the score is pulled to the nearest value that band admits,
    which preserves the half-step resolution the rubric relies on. A missing or
    unrecognised band is derived from the score instead, so the pair is always
    self-consistent.

    Returns (band, score, adjusted) — `adjusted` is True when the model's own
    pair was inconsistent and had to be moved.
    """
    scale = max_score / 5.0 if max_score > 0 else 0.0
    label = str(band or "").strip().lower()
    coerced = _coerce_score(score, max_score)

    if label not in _BAND_VALUES:
        if coerced is None:
            return "fails", 0.0, True
        derived = _band_for_score(coerced / scale if scale else 0.0)
        return derived, coerced, True

    nominal = _BAND_VALUES[label] * scale
    if coerced is None:
        return label, nominal, True

    low = max(0.0, nominal - _BAND_TOLERANCE * scale)
    high = min(max_score, nominal + _BAND_TOLERANCE * scale)
    adjusted = min(high, max(low, coerced))
    return label, adjusted, adjusted != coerced


# Highest canonical 0–5 score a criterion may keep given how many of its own
# weakness probes the model classified "significant"; 4+ significant probes use
# _SEVERITY_CEILING_MIN. Rule 6 STEP B states this relationship in prose, but
# the model does not reliably honour it: the observed failure is always the
# same direction — every criterion lands on "very good"/4.0 whatever the scan
# found, because the band is chosen before the severities are weighed.
_SEVERITY_CEILING = {1: 4.0, 2: 3.5, 3: 3.0}
_SEVERITY_CEILING_MIN = 2.5

# Mirror of the ceilings: a criterion whose scan the text genuinely settled —
# no significant probe, no minor probe, and enough probes for that to mean
# something — has been answered convincingly and must not be parked at 4.0.
# Without this the envelope could only push scores down, which is how "4.0
# everywhere" became "4.0 or lower everywhere" rather than a real spread.
_CLEAN_SCAN_FLOOR = 4.5
_CLEAN_SCAN_MIN_PROBES = 2
# The floor presumes a criterion that was actually addressed. A band at "fair"
# or below says it was not, so the probes cannot mean what the floor assumes.
_FLOOR_ELIGIBLE_BANDS = ("good", "very good", "excellent")

# Legacy label: the scan verdict used to be binary ("weakness" | "cleared").
# Runs and any cached output predating the severity split map onto "minor".
_LEGACY_WEAKNESS_VERDICT = "weakness"


def _severity_counts(scan: Any) -> tuple[int, int, int]:
    """(significant, minor, cleared) over a weakness_scan, tolerant of junk."""
    significant = minor = cleared = 0
    for probe in scan if isinstance(scan, list) else []:
        if not isinstance(probe, dict):
            continue
        verdict = str(probe.get("verdict") or "").strip().lower()
        if verdict == "significant":
            significant += 1
        elif verdict in ("minor", _LEGACY_WEAKNESS_VERDICT):
            minor += 1
        elif verdict == "cleared":
            cleared += 1
    return significant, minor, cleared


def _apply_severity_envelope(
    band: str,
    score: float,
    max_score: float,
    significant: int,
    minor: int,
    cleared: int,
    allow_floor: bool = True,
) -> tuple[str, float, str]:
    """Constrain a reconciled (band, score) by the scan's own severity counts.

    Returns (band, score, note) — note is "" when nothing moved, otherwise a
    short human-readable reason recorded as telemetry. The band is re-derived
    from the moved score so the pair stays self-consistent, exactly as
    _reconcile_band_and_score guarantees for the model's own output.
    """
    scale = max_score / 5.0 if max_score > 0 else 0.0
    if not scale:
        return band, score, ""
    canonical = score / scale

    if significant:
        ceiling = _SEVERITY_CEILING.get(significant, _SEVERITY_CEILING_MIN)
        if canonical > ceiling:
            return (
                _band_for_score(ceiling),
                ceiling * scale,
                f"capped at {ceiling:g}/5 by {significant} significant "
                f"weakness probe(s) in the scan",
            )
    elif (
        allow_floor
        and not minor
        and cleared >= _CLEAN_SCAN_MIN_PROBES
        and band in _FLOOR_ELIGIBLE_BANDS
        and canonical < _CLEAN_SCAN_FLOOR
    ):
        return (
            _band_for_score(_CLEAN_SCAN_FLOOR),
            _CLEAN_SCAN_FLOOR * scale,
            f"raised to {_CLEAN_SCAN_FLOOR:g}/5 — all {cleared} scan probes "
            "were cleared and no weakness was kept",
        )

    return band, score, ""


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
    max_score: float = 5.0,
    chat_model: str = CHAT_MODEL,
    call_context: str = "",
    full_document: bool = False,
    coverage_ratio: float | None = None,
    off_topic: bool = False,
    off_topic_document_type: str = "",
    off_topic_reason: str = "",
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
        if off_topic:
            # The stock wording invites the model to assume a fuller proposal
            # sits behind the excerpts — the precise reasoning that lets an
            # off-topic document escape penalty. Say what partial really means.
            input_scope = (
                f"PARTIAL — retrieval-selected excerpts covering {pct} of the "
                "document. What you cannot see is more of THIS document, which "
                "the screening notice below identifies as not a proposal."
            )

    relevance_block = (
        OFF_TOPIC_NOTICE.format(
            document_type=off_topic_document_type or "not a proposal for this call",
            reason_clause=(
                f" Screening verdict: {off_topic_reason}" if off_topic_reason else ""
            ),
        )
        if off_topic
        else ""
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
        relevance_block=relevance_block,
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
    parsed.setdefault("band_rationale", "")

    # Band and score are reconciled into one verdict — the model reports them
    # independently and they drift. Raw values are kept as telemetry so a run
    # stays auditable (same idiom as model_confidence / confidence_cap below).
    parsed["model_band"] = parsed["band"]
    parsed["model_score"] = parsed.get("score")
    parsed["band"], parsed["score"], parsed["score_adjusted"] = (
        _reconcile_band_and_score(parsed.get("band"), parsed.get("score"), max_score)
    )

    # Then hold that verdict to the severities the scan itself assigned. The
    # model chooses its band before weighing them, which is what collapsed every
    # criterion onto "very good"/4.0; the envelope is the deterministic half of
    # rule 6 STEP B. The floor is suppressed on an off-topic run — there the
    # screening notice, not the probe tally, governs what the scan means.
    (
        parsed["significant_weakness_count"],
        parsed["minor_weakness_count"],
        parsed["cleared_probe_count"],
    ) = _severity_counts(parsed["weakness_scan"])
    parsed["band"], parsed["score"], parsed["severity_envelope"] = (
        _apply_severity_envelope(
            parsed["band"],
            parsed["score"],
            max_score,
            parsed["significant_weakness_count"],
            parsed["minor_weakness_count"],
            parsed["cleared_probe_count"],
            allow_floor=not off_topic,
        )
    )

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
            f"{c['name']} (score {c['score']}/{c.get('max_score', 5.0):g}, "
            f"weight x{c['weight']}):\n"
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
_FUNDED, _BORDERLINE, _REJECTED = _VALID_BANDS

# Share of the maximum above which a passing proposal is called fundable
# rather than borderline. Only consulted once the threshold gate is cleared.
_FUNDED_RATIO = 0.7


def predict_band(
    per_criterion: list[dict],
    weighted_total: float,
    max_weighted_total: float,
    threshold: float | None,
    chat_model: str = CHAT_MODEL,
) -> str:
    """Classify the run from its weighted total. Deterministic, not model-led.

    This used to ask the LLM for the label and take whatever came back, with
    the threshold consulted only if no label matched — so a below-threshold
    proposal could be stored as "Likely funded" while the PDF's own PASS/FAIL
    tile, computed from the same threshold, said FAIL. The rule below is the
    one that was previously the unreachable fallback, promoted to be the rule:
    the threshold decides pass/fail, and share-of-maximum separates a
    comfortable pass from a borderline one.

    per_criterion and chat_model are accepted for call-site compatibility;
    neither is needed now that no model call is made.
    """
    if threshold is not None and weighted_total < threshold:
        return _REJECTED
    ratio = weighted_total / max_weighted_total if max_weighted_total else 0.0
    if threshold is None and ratio < 0.5:
        return _REJECTED
    return _FUNDED if ratio >= _FUNDED_RATIO else _BORDERLINE
