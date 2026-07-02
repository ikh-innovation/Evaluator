from __future__ import annotations

import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from .config import (
    CHAT_MODEL,
    MAIN_GPU,
    MAX_PROPOSAL_CHARS_INLINE,
    NUM_CTX,
    NUM_GPU,
    OLLAMA_BASE_URL,
    REQUEST_TIMEOUT_S,
)


EDF_RUBRIC = """EDF scoring rubric (apply per criterion, 0–5 in 0.5 increments):
0 — Fails to address the criterion or cannot be assessed.
1 — Poor. Inadequately addressed; serious inherent weaknesses.
2 — Fair. Broadly addressed but significant weaknesses remain.
3 — Good. Addressed well; a number of shortcomings remain.
4 — Very good. Addressed very well; small number of shortcomings.
5 — Excellent. Addressed convincingly; only minor shortcomings, if any."""


EVAL_PROMPT = """You are an EDF (European Defence Fund) proposal evaluation assistant.
You produce assessments in the voice of an official Evaluation Summary Report (ESR),
grounded ONLY in the proposal text provided below.

{rubric}

Criterion under evaluation:
- Name: {criterion_name}
- Description: {criterion_description}
- Weight: x{weight}
- Action type: {action_type}

Proposal excerpts to evaluate (each block is labelled with its source section):
---
{proposal_text}
---

Rules:
1. Assess ONLY the criterion above.
2. Every strength and weakness must be grounded in the proposal text. Quote a short
   phrase from the proposal verbatim inside the string, in double quotes. If you
   cannot find evidence, do not claim it.
3. Use formal, ESR-style language: specific, evaluative, concise.
4. Apply the EDF rubric strictly. Do NOT inflate scores.
5. Confidence: "high" if you found multiple evidence quotes; "medium" if one or two;
   "low" if the proposal was vague or evidence was thin.
6. Output STRICT JSON ONLY (no markdown, no prose) with this exact schema:

{{
  "score": <number, 0..5, in 0.5 increments>,
  "confidence": "low" | "medium" | "high",
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "missing_evidence": [string, ...],
  "evaluator_comment": string,
  "recommendations": [string, ...]
}}
"""

SUMMARY_PROMPT = """You are summarising an EDF Part B proposal evaluation in the
voice of an EDF Evaluation Summary Report.

Per-criterion findings:
{per_criterion_block}

Weighted total: {weighted_total} / {max_weighted_total}
Threshold: {threshold}

Write a concise (5-8 sentence) ESR-style summary that:
- highlights the strongest criteria,
- highlights the most critical weaknesses across criteria,
- lists 2-3 concrete rewrite priorities before submission.
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


def evaluate_criterion(
    proposal_text: str,
    criterion_name: str,
    criterion_description: str,
    weight: float,
    action_type: str,
    chat_model: str = CHAT_MODEL,
) -> dict:
    prompt = EVAL_PROMPT.format(
        rubric=EDF_RUBRIC,
        criterion_name=criterion_name,
        criterion_description=criterion_description or "(no description)",
        weight=weight,
        action_type=action_type,
        proposal_text=proposal_text[:MAX_PROPOSAL_CHARS_INLINE] or "(empty proposal text)",
    )
    msg = _chat_json_model(chat_model).invoke(prompt)
    parsed = _parse_json(msg.content)

    parsed.setdefault("strengths", [])
    parsed.setdefault("weaknesses", [])
    parsed.setdefault("missing_evidence", [])
    parsed.setdefault("recommendations", [])
    parsed.setdefault("evaluator_comment", "")
    parsed.setdefault("confidence", "medium")
    parsed["score"] = _coerce_score(parsed.get("score", 0))
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
