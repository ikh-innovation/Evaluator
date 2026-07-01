from __future__ import annotations

import csv
import io
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import db, jobs  # noqa: E402
from backend.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL  # noqa: E402
from backend.criteria_io import criteria_to_xlsx_bytes, xlsx_to_criteria  # noqa: E402


DEFAULT_RESEARCH_CRITERIA: list[dict[str, Any]] = [
    {
        "name": "Excellence and potential of disruption",
        "description": (
            "Excellence of the overall concept; soundness of approach; "
            "compliance with topic objectives; advantage over existing "
            "defence products / disruption potential."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Innovation and technological development",
        "description": (
            "Innovation potential, ground-breaking concepts, integration of "
            "prior R&D while avoiding duplication, spin-off potential."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Competitiveness",
        "description": (
            "Competitive advantage vs. existing solutions, market growth "
            "potential across EU, IP strategy."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
    {
        "name": "EDTIB autonomy and security of supply",
        "description": (
            "Contribution to EDTIB autonomy, non-dependency on third "
            "countries, security of supply, alignment with CDP priorities."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
    {
        "name": "Creation of new cross-border cooperation",
        "description": (
            "New cross-border cooperation, especially SMEs and mid-caps with "
            "substantial participation."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Quality and efficiency of implementation",
        "description": (
            "Work plan structure, milestones / deliverables, management, risk "
            "treatment, allocation of work across the consortium."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
]

DEFAULT_DEVELOPMENT_CRITERIA: list[dict[str, Any]] = DEFAULT_RESEARCH_CRITERIA + [
    {
        "name": "Contribution to industrial policy objectives",
        "description": (
            "Contribution to industrial-policy objectives of the EDF, "
            "including increase of production capacity in the EU defence industry."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
]

OVERALL_THRESHOLD_DEFAULT = 30.0


@dataclass
class Criterion:
    name: str
    description: str
    weight: float
    max_score: float = 5.0


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EDF Proposal Evaluator",
    page_icon=":bar_chart:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
button[data-testid="stBaseButton-primary"] {
    background-color: #dc2626 !important;
    border-color: #dc2626 !important;
    color: #fff !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #b91c1c !important;
    border-color: #b91c1c !important;
}
button[data-testid="stBaseButton-primary"]:disabled {
    background-color: #9ca3af !important;
    border-color: #9ca3af !important;
    color: #e5e7eb !important;
}

/* ── Queue status bar ─────────────────────────────────────────────────── */
.edf-queue-bar {
    display: flex;
    align-items: center;
    gap: .9rem;
    width: 100%;
    padding: .55rem 1rem;
    border: 1px solid #e5e7eb;
    background: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 100%);
    font-size: .92rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
}
.edf-queue-bar.idle {
    border-radius: 10px;
    margin: 0 0 .8rem 0;
    color: #475569;
}
.edf-queue-bar.busy {
    border-radius: 10px 10px 0 0;
    border-bottom: none;
    background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
    border-color: #fcd34d;
    color: #78350f;
}
.edf-queue-bar .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #94a3b8; flex-shrink: 0;
}
.edf-queue-bar.busy .dot {
    background: #f59e0b;
    box-shadow: 0 0 0 0 rgba(245, 158, 11, .6);
    animation: edf-pulse 1.4s infinite;
}
@keyframes edf-pulse {
    0%   { box-shadow: 0 0 0 0   rgba(245, 158, 11, .55); }
    70%  { box-shadow: 0 0 0 10px rgba(245, 158, 11, 0);  }
    100% { box-shadow: 0 0 0 0   rgba(245, 158, 11, 0);   }
}
.edf-queue-bar .summary  { font-weight: 700; white-space: nowrap; }
.edf-queue-bar .msg      { color: #78350f; font-size: .82rem; opacity: .85;
                            flex: 1; min-width: 0;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.edf-queue-bar .progress { height: 6px; flex: 0 0 160px; max-width: 28%;
                            background: rgba(255, 255, 255, .55);
                            border-radius: 4px; overflow: hidden; }
.edf-queue-bar .progress > div {
    height: 100%; background: #f59e0b; transition: width .4s ease;
}

/* ── Sliding chip strip (the row immediately below the bar header) ────── */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) {
    flex-wrap: nowrap !important;
    overflow-x: auto;
    overflow-y: visible;
    scroll-behavior: smooth;
    align-items: center !important;
    gap: .55rem !important;
    padding: .6rem .9rem .7rem .9rem;
    margin: 0 0 .8rem 0;
    background: linear-gradient(90deg, #fef3c7 0%, #fde68a 100%);
    border: 1px solid #fcd34d;
    border-top: 1px dashed rgba(252, 211, 77, .55);
    border-radius: 0 0 10px 10px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
    scrollbar-width: thin;
    scrollbar-color: #d97706 rgba(255, 255, 255, .25);
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar { height: 6px; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-track { background: rgba(255,255,255,.25); border-radius: 4px; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-thumb { background: #d97706; border-radius: 4px; }

/* Each column = one (chip + ×) pair. Shrinks to content; kids in a single
   horizontal row with zero gap so the chip and × visually become one pill. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] > div > * {
    width: auto !important;
    flex: 0 0 auto !important;
}

/* Flatten ALL of Streamlit's inner wrappers so chip & × align exactly. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stElementContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdown"],
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdownContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: stretch !important;
}

/* ── The chip — left half of the pill. ─────────────────────────────────
   All variant colors live on the chip; CSS :has() on the column then
   propagates the same colors to the × button so they always agree. */
.qbar-chip {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    height: 2rem;
    padding: 0 .55rem 0 .85rem;
    border: 1px solid #e5e7eb;
    border-right: none;
    border-radius: 8px 0 0 8px;
    background: #ffffff;
    color: #334155;
    font-size: .88rem;
    line-height: 1;
    white-space: nowrap;
    box-shadow: 0 1px 0 rgba(15, 23, 42, .04);
}
.qbar-chip.running   { border-color: #fcd34d; background: #fffbeb; color: #92400e; }
.qbar-chip.mine      { border-color: #93c5fd; background: #eff6ff; color: #1e40af; }
.qbar-chip.cancelling{ border-color: #cbd5e1; background: #f1f5f9; color: #64748b;
                        text-decoration: line-through; opacity: .7; }
.qbar-chip code      { background: transparent; padding: 0; color: inherit; font-size: .8rem; opacity: .75; }
.qbar-submitter      { font-size: .75rem; opacity: .65; font-style: italic; margin-left: .15rem; }

/* ── The × — right half of the same pill. Matches the chip's height,
   border, and background; just the rounded corners flip to the right. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button {
    height: 2rem !important;
    min-height: 2rem !important;
    width: 1.55rem !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    border: 1px solid #e5e7eb !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    background: #ffffff !important;
    color: #94a3b8 !important;
    cursor: pointer;
    box-shadow: 0 1px 0 rgba(15, 23, 42, .04) !important;
    transition: color .12s, background-color .12s, border-color .12s;
}
/* The button's internal <p> wraps the "✕" label — flatten it. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button p,
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button div {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* Variant colors flow from the chip to the × via :has() on the column. */
[data-testid="stColumn"]:has(.qbar-chip.running) [data-testid="stButton"] > button {
    background: #fffbeb !important;
    border-color: #fcd34d !important;
    color: #b45309 !important;
}
[data-testid="stColumn"]:has(.qbar-chip.mine) [data-testid="stButton"] > button {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #3b82f6 !important;
}

/* Hover — only the × box changes, never the chip. Subtle pink tint. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:hover {
    color: #dc2626 !important;
    background: #fee2e2 !important;
    border-color: #fca5a5 !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:focus,
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:active {
    outline: none !important;
    box-shadow: 0 1px 0 rgba(15, 23, 42, .04) !important;
}

/* Cancelling: a static "…" pill stamped on the right, same dimensions. */
.qbar-cancel-pending {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.55rem;
    height: 2rem;
    border: 1px solid #cbd5e1;
    border-left: none;
    border-radius: 0 8px 8px 0;
    background: #f1f5f9;
    color: #64748b;
    font-size: .85rem;
    opacity: .85;
    box-shadow: 0 1px 0 rgba(15, 23, 42, .04);
}
.qbar-cancel-locked {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.55rem;
    height: 2rem;
    border: 1px solid #cbd5e1;
    border-left: none;
    border-radius: 0 8px 8px 0;
    background: #f8fafc;
    color: #94a3b8;
    font-size: .78rem;
    opacity: .7;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "criteria": None,
    "action_type": "Research",
    "call_id": "",
    "proposal_acronym": "",
    "submitted_by": "",
    # Upload (raw bytes — worker handles extraction / embedding)
    "embedded_key": None,
    "proposal_bytes": None,
    "proposal_suffix": None,
    "upload_nonce": 0,          # bumped after each submit to reset the paste/upload widgets
    # Final result (last completed run that belongs to this session)
    "result": None,
    # Dialogs
    "pending_dialog": None,
    "detail_call_id": None,
    # Background queue tracking
    "session_id": None,
    "my_job_ids": [],          # list[str] — jobs submitted from this session and not yet surfaced
    "known_active_ids": None,  # last-seen active job ids; any drop (any user) triggers a refresh
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.session_id is None:
    st.session_state.session_id = uuid.uuid4().hex[:12]

if st.session_state.criteria is None:
    st.session_state.criteria = [Criterion(**c) for c in DEFAULT_RESEARCH_CRITERIA]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _criterion_dicts() -> list[dict[str, Any]]:
    return [
        {"name": c.name, "description": c.description,
         "weight": c.weight, "max_score": c.max_score}
        for c in st.session_state.criteria
    ]


def _score_color(score: float, max_score: float = 5.0) -> str:
    r = score / max_score if max_score else 0.0
    return "green" if r >= 0.7 else ("orange" if r >= 0.5 else "red")



def _run_to_csv(run: dict[str, Any]) -> bytes:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["Field", "Value"])
    for label, key in [
        ("Proposal acronym", "proposal_acronym"), ("Call ID", "call_id"),
        ("Action type", "action_type"), ("Generated at", "generated_at"),
        ("Weighted total", "weighted_total"), ("Max weighted total", "max_weighted_total"),
        ("Threshold", "overall_threshold"), ("Predicted band", "predicted_band"),
        ("Overall confidence", "overall_confidence"), ("ESR-style summary", "esr_style_summary"),
    ]:
        w.writerow([label, run.get(key, "")])
    w.writerow([])
    w.writerow(["Criterion", "Score", "Weight", "Max score", "Contribution", "Confidence",
                "Strengths", "Weaknesses", "Missing evidence", "Evaluator comment", "Recommendations"])
    for c in run.get("criteria", []):
        score = c.get("score", 0) or 0
        weight = c.get("weight", 0) or 0
        w.writerow([
            c.get("name", ""), score, weight, c.get("max_score", ""),
            round(score * weight, 4), c.get("confidence", ""),
            "\n".join(c.get("strengths") or []), "\n".join(c.get("weaknesses") or []),
            "\n".join(c.get("missing_evidence") or []), c.get("evaluator_comment", ""),
            "\n".join(c.get("recommendations") or []),
        ])
    return ("﻿" + out.getvalue()).encode("utf-8")


def _render_criterion_card(c: dict) -> None:
    score = c["score"]
    weight = c["weight"]
    max_score = c.get("max_score", 5.0)
    color = _score_color(score, max_score)
    color_hex = {"green": "#166534", "orange": "#92400e", "red": "#991b1b"}[color]
    bg_hex = {"green": "#dcfce7", "orange": "#fef3c7", "red": "#fee2e2"}[color]

    head = st.columns([5, 1, 1, 1])
    with head[0]:
        st.markdown(f"**{c['name']}**")
        st.caption(f"Confidence: {c.get('confidence', '—').title()}")
    head[1].markdown(
        f"<small>Score</small><br>"
        f"<span style='background:{bg_hex};color:{color_hex};font-weight:700;"
        f"padding:.2rem .5rem;border-radius:6px'>{score:g} / {max_score:g}</span>",
        unsafe_allow_html=True,
    )
    head[2].markdown(f"<small>Weight</small><br><b>×{weight:g}</b>", unsafe_allow_html=True)
    head[3].markdown(f"<small>Contribution</small><br><b>{score * weight:g}</b>", unsafe_allow_html=True)
    st.progress(min(score / max_score, 1.0))
    with st.expander("Details"):
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Strengths**")
            for s in c.get("strengths") or ["—"]:
                st.markdown(f"- {s}")
        with cols[1]:
            st.markdown("**Weaknesses**")
            for s in c.get("weaknesses") or ["—"]:
                st.markdown(f"- {s}")
        with cols[2]:
            st.markdown("**Missing evidence**")
            for s in c.get("missing_evidence") or ["—"]:
                st.markdown(f"- {s}")
        if c.get("evaluator_comment"):
            st.markdown("**Evaluator comment**")
            st.write(c["evaluator_comment"])
        if c.get("recommendations"):
            st.markdown("**Recommendations**")
            for s in c["recommendations"]:
                st.markdown(f"- {s}")


# ---------------------------------------------------------------------------
# Queue bar (top, full-width, auto-refreshing)
# ---------------------------------------------------------------------------

def _escape(text: Any) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _promote_finished_my_jobs() -> bool:
    """Surface results of this session's finished jobs.

    Walks `st.session_state.my_job_ids`, removes any whose status is
    done/failed (or unknown), and for the most recent done one updates
    `st.session_state.result`. Returns True if anything changed.
    """
    pending: list[str] = list(st.session_state.my_job_ids)
    if not pending:
        return False
    still_pending: list[str] = []
    newest_done_call_id: str | None = None
    newest_done_at: str = ""
    failed_messages: list[tuple[str, str]] = []
    for jid in pending:
        info = jobs.get_job(jid)
        if info is None:
            continue  # dropped — silently forget
        status = info["status"]
        if status in ("queued", "running"):
            still_pending.append(jid)
            continue
        if status == "done":
            ts = info.get("finished_at") or ""
            if ts >= newest_done_at:
                newest_done_at = ts
                newest_done_call_id = info.get("result_call_id") or info.get("call_id")
        elif status == "failed":
            failed_messages.append((info.get("proposal_acronym", "—"), info.get("error") or "Evaluation failed."))

    changed = still_pending != pending
    st.session_state.my_job_ids = still_pending

    if newest_done_call_id:
        run = db.get_run(newest_done_call_id)
        if run is not None:
            st.session_state.result = run
            st.toast(f"Evaluation finished: {run.get('proposal_acronym', '—')}", icon="✅")
    for acronym, err in failed_messages:
        st.toast(f"Evaluation failed for {acronym}: {err[:120]}", icon="⚠️")

    return changed or bool(newest_done_call_id) or bool(failed_messages)


@st.fragment(run_every="2s")
def _queue_bar() -> None:
    """Top, full-width auto-refreshing queue indicator."""
    active = jobs.list_active()
    my_ids = set(st.session_state.my_job_ids)

    promoted = _promote_finished_my_jobs()

    current_ids = {j["job_id"] for j in active}
    known_ids = st.session_state.known_active_ids
    someone_finished = bool(known_ids) and bool(known_ids - current_ids)
    st.session_state.known_active_ids = current_ids

    if promoted or someone_finished:
        # Force a full app rerun so the result panel and History reflect the new
        # state, even when the finished job belongs to another user/session.
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()

    if not active:
        st.markdown(
            '<div class="edf-queue-bar idle">'
            '<span class="dot"></span>'
            '<span class="summary">Queue idle</span>'
            '<span style="color:#64748b">No proposals being processed.</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    running = [j for j in active if j["status"] == "running"]
    queued = [j for j in active if j["status"] == "queued"]
    summary_parts: list[str] = []
    if running:
        summary_parts.append(f"{len(running)} running")
    if queued:
        summary_parts.append(f"{len(queued)} queued")
    summary = " · ".join(summary_parts) or "Working…"

    progress_html = ""
    if running:
        frac = running[0].get("progress_frac") or 0.0
        pct = max(0, min(100, int(round(float(frac) * 100))))
        msg = _escape(running[0].get("progress_msg") or "")
        progress_html = (
            f'<div class="progress" title="{msg}"><div style="width:{pct}%"></div></div>'
        )

    # ── Top half of the bar: status + summary + running-job progress ──────
    running_msg = _escape(running[0]["progress_msg"]) if running else ""
    st.markdown(
        '<div class="edf-queue-bar busy">'
        '<span class="dot"></span>'
        f'<span class="summary">{summary}</span>'
        f'<span class="msg">{running_msg}</span>'
        f"{progress_html}"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Bottom half: horizontally-sliding chip strip with inline × ────────
    # One column per (chip + cancel) pair; CSS lays each column out as a
    # horizontal flex row and makes the parent strip scroll left/right.
    strip_cols = st.columns(len(active), gap="small")
    for idx, j in enumerate(active):
        with strip_cols[idx]:
            acronym = _escape(j.get("proposal_acronym") or "—")
            call_id = _escape(j.get("call_id") or "")
            submitter = _escape(j.get("submitted_by") or "—")
            is_cancelling = bool(j.get("cancel_requested"))

            classes = ["qbar-chip"]
            if is_cancelling:
                classes.append("cancelling")
            elif j["status"] == "running":
                classes.append("running")
            if j["job_id"] in my_ids:
                classes.append("mine")

            if is_cancelling:
                marker = "⏳"
            elif j["status"] == "running":
                marker = "▶"
            else:
                marker = "•"
            you = " (you)" if j["job_id"] in my_ids else ""

            st.markdown(
                f'<div class="{" ".join(classes)}" title="{call_id} · submitted by {submitter}">{marker} '
                f"<b>{acronym}</b>{you} <code>{call_id}</code> "
                f'<span class="qbar-submitter">by {submitter}</span></div>',
                unsafe_allow_html=True,
            )
            is_owner = bool(st.session_state.submitted_by.strip()) and (
                (j.get("submitted_by") or "").strip().lower()
                == st.session_state.submitted_by.strip().lower()
            )

            if is_cancelling:
                st.markdown(
                    '<span class="qbar-cancel-pending" title="Cancelling — worker will stop at the next checkpoint">…</span>',
                    unsafe_allow_html=True,
                )
            elif is_owner:
                if st.button(
                    "✕",
                    key=f"cancel_{j['job_id']}",
                    help=f"Cancel {j.get('proposal_acronym') or 'this run'}",
                ):
                    jobs.cancel(j["job_id"])
                    st.rerun()
            else:
                st.markdown(
                    f'<span class="qbar-cancel-locked" title="Only {submitter} can cancel this run">🔒</span>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

@st.dialog("Refresh criteria", width="small")
def refresh_criteria_dialog() -> None:
    st.download_button(
        "Download current criteria as XLSX",
        data=criteria_to_xlsx_bytes(_criterion_dicts()),
        file_name="edf_criteria_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.divider()
    up = st.file_uploader("Upload edited XLSX", type=["xlsx"], key="dlg_criteria_xlsx")
    if up is not None:
        if st.button("Apply", type="primary", use_container_width=True):
            try:
                parsed = xlsx_to_criteria(up.read())
                st.session_state.criteria = [Criterion(**c) for c in parsed]
                st.success(f"Loaded {len(parsed)} criteria.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse XLSX: {exc}")


@st.dialog("Wipe history?", width="small")
def wipe_db_dialog() -> None:
    st.warning("Permanently deletes every stored run.")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with cols[1]:
        if st.button("Wipe everything", type="primary", use_container_width=True):
            removed = db.wipe_all()
            st.success(f"Removed {removed} run(s).")
            st.rerun()


def _render_run_details(run: dict[str, Any]) -> None:
    st.markdown(f"### {run.get('proposal_acronym', '—')}  ·  `{run.get('call_id')}`")
    st.caption(
        f"Generated: {run.get('generated_at', '—')}  ·  "
        f"Action type: {run.get('action_type', '—')}  ·  "
        f"Confidence: {run.get('overall_confidence', '—')}"
    )
    weighted = run.get("weighted_total", 0.0)
    max_w = run.get("max_weighted_total", 1.0) or 1.0
    threshold = run.get("overall_threshold")
    passed = (threshold is None) or (weighted >= threshold)
    cols = st.columns(4)
    cols[0].metric("Weighted total", f"{weighted:g}", f"/ {max_w:g}")
    cols[1].metric("Threshold", f"{threshold:g}" if threshold is not None else "—")
    cols[2].metric("Pass / fail", "PASS" if passed else "FAIL")
    cols[3].metric("Predicted band", run.get("predicted_band", "—"))
    st.markdown("**ESR-style summary**")
    st.write(run.get("esr_style_summary", "—"))
    st.markdown("**Per-criterion**")
    for c in run.get("criteria", []):
        with st.expander(f"{c['name']} — {c['score']}/5 · weight ×{c['weight']}"):
            _render_criterion_card(c)
    act = st.columns([1, 1, 1])
    with act[0]:
        st.download_button("Export CSV", data=_run_to_csv(run),
                           file_name=f"evaluation_{run.get('call_id', 'edf')}.csv",
                           mime="text/csv", use_container_width=True)
    with act[1]:
        st.download_button("Export JSON", data=json.dumps(run, indent=2).encode(),
                           file_name=f"evaluation_{run.get('call_id', 'edf')}.json",
                           mime="application/json", use_container_width=True)
    with act[2]:
        if st.button("Delete this run", type="secondary", use_container_width=True):
            db.delete_run(run.get("call_id", ""))
            st.session_state.pending_dialog = "history"
            st.rerun()


@st.dialog("History of past runs", width="large")
def history_dialog() -> None:
    all_runs = db.list_runs()
    if not all_runs:
        st.info("No runs saved yet.")
        return
    query = st.text_input("Search", placeholder="Filter by Call ID, acronym, or band…").strip()
    runs = all_runs if not query else [
        r for r in all_runs
        if query.lower() in (r.get("call_id") or "").lower()
        or query.lower() in (r.get("proposal_acronym") or "").lower()
        or query.lower() in (r.get("predicted_band") or "").lower()
        or query.lower() in (r.get("action_type") or "").lower()
    ]
    st.caption(f"Showing {len(runs)} of {len(all_runs)} run(s).")
    if not runs:
        st.info(f"No matches for '{query}'.")
        return
    header = st.columns([2, 3, 1.5, 2, 1.2, 0.9])
    for col, label in zip(header, ["Acronym", "Call ID", "Date", "Band", "Score", ""]):
        col.markdown(f"**{label}**")
    st.divider()
    for r in runs:
        cols = st.columns([2, 3, 1.5, 2, 1.2, 0.9])
        cols[0].markdown(f"**{r.get('proposal_acronym') or '—'}**")
        cols[1].markdown(f"`{r.get('call_id')}`")
        cols[2].markdown((r.get("generated_at") or "")[:10] or "—")
        cols[3].markdown(r.get("predicted_band") or "—")
        cols[4].markdown(f"{r['weighted_total']:g}/{r['max_weighted_total']:g}")
        with cols[5]:
            if st.button("View", key=f"hist_view_{r['call_id']}", use_container_width=True):
                st.session_state.pending_dialog = "detail"
                st.session_state.detail_call_id = r["call_id"]
                st.rerun()


@st.dialog("Run details", width="large")
def run_detail_dialog(call_id: str) -> None:
    run = db.get_run(call_id)
    if run is None:
        st.error("Run no longer exists.")
        if st.button("Back to history", use_container_width=True):
            st.session_state.pending_dialog = "history"
            st.rerun()
        return
    if st.button("Back to history", use_container_width=True):
        st.session_state.pending_dialog = "history"
        st.rerun()
    st.divider()
    _render_run_details(run)


# ---------------------------------------------------------------------------
# Dialog dispatcher
# ---------------------------------------------------------------------------

_pending = st.session_state.pending_dialog
st.session_state.pending_dialog = None
if _pending == "history":
    history_dialog()
elif _pending == "detail" and st.session_state.detail_call_id:
    run_detail_dialog(st.session_state.detail_call_id)

# ---------------------------------------------------------------------------
# Top queue bar (full width, auto-refreshing)
# ---------------------------------------------------------------------------

_queue_bar()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("EDF Proposal Evaluator")
st.caption(f"LLM Model =`{CHAT_MODEL}` · Embeddings Model=`{EMBED_MODEL}`")

# ---------------------------------------------------------------------------
# Sidebar — always editable. Edits apply to the *next* submission; jobs
# already queued or running use the criteria/threshold snapshotted at submit.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Call configuration")
    st.session_state.call_id = st.text_input(
        "Call ID *", value=st.session_state.call_id,
        placeholder="e.g. EDF-2024-LS-RA-CHALLENGE",
    )
    st.session_state.proposal_acronym = st.text_input(
        "Proposal acronym *", value=st.session_state.proposal_acronym,
        placeholder="e.g. SWARMER",
    )
    new_action = st.selectbox(
        "Action type", ["Research", "Development"],
        index=0 if st.session_state.action_type == "Research" else 1,
    )
    st.session_state.submitted_by = st.text_input(
        "Submitted by *", value=st.session_state.submitted_by,
        placeholder="e.g. Jane Doe",
        help="Shown in the shared queue so others know whose job is running.",
    )
    if new_action != st.session_state.action_type:
        st.session_state.action_type = new_action
        defaults = DEFAULT_RESEARCH_CRITERIA if new_action == "Research" else DEFAULT_DEVELOPMENT_CRITERIA
        st.session_state.criteria = [Criterion(**c) for c in defaults]

    st.divider()
    st.subheader("Criteria & weights")
    to_delete: list[int] = []
    for idx, crit in enumerate(st.session_state.criteria):
        with st.expander(f"{idx + 1}. {crit.name}", expanded=False):
            crit.name = st.text_input("Name", value=crit.name, key=f"name_{idx}")
            crit.description = st.text_area("Description", value=crit.description,
                                             key=f"desc_{idx}", height=80)
            c1, c2 = st.columns(2)
            with c1:
                crit.weight = st.number_input("Weight", min_value=0.0, max_value=10.0,
                                               value=float(crit.weight), step=0.5,
                                               key=f"w_{idx}")
            with c2:
                crit.max_score = st.number_input("Max score", min_value=1.0, max_value=10.0,
                                                  value=float(crit.max_score), step=0.5,
                                                  key=f"ms_{idx}")
            if st.button("Remove", key=f"del_{idx}"):
                to_delete.append(idx)
    for idx in reversed(to_delete):
        st.session_state.criteria.pop(idx)

    if st.button("+ Add criterion", use_container_width=True):
        st.session_state.criteria.append(
            Criterion(name=f"Criterion {len(st.session_state.criteria) + 1}",
                      description="", weight=1.0, max_score=5.0)
        )
    a, b = st.columns(2)
    with a:
        if st.button("Refresh criteria", use_container_width=True):
            refresh_criteria_dialog()
    with b:
        if st.button("Reset defaults", use_container_width=True):
            defaults = DEFAULT_RESEARCH_CRITERIA if st.session_state.action_type == "Research" else DEFAULT_DEVELOPMENT_CRITERIA
            st.session_state.criteria = [Criterion(**c) for c in defaults]
            st.rerun()

    st.divider()
    overall_threshold = st.number_input(
        "Overall pass threshold", min_value=0.0, max_value=200.0,
        value=OVERALL_THRESHOLD_DEFAULT, step=0.5,
    )

    st.divider()
    runs_count = len(db.list_runs())
    if st.button(
        f"History ({runs_count} run{'s' if runs_count != 1 else ''})" if runs_count else "History",
        use_container_width=True, disabled=runs_count == 0,
    ):
        st.session_state.pending_dialog = "history"
        st.session_state.detail_call_id = None
        st.rerun()
    if runs_count:
        if st.button("Wipe history", type="secondary", use_container_width=True):
            wipe_db_dialog()

    if st.session_state.my_job_ids:
        st.divider()
        n_mine = len(st.session_state.my_job_ids)
        st.info(
            f"You have {n_mine} submission{'s' if n_mine != 1 else ''} in flight. "
            "Results will appear here automatically when each one finishes."
        )


# ---------------------------------------------------------------------------
# MAIN AREA — always visible. Evaluations run in a background worker, so the
# UI stays interactive: you can browse history, edit criteria, queue another
# proposal, all while one is being processed.
# ---------------------------------------------------------------------------

st.subheader("1. Upload Part B")

pasted = st.text_area(
    "Paste sections directly (not saved to history)",
    placeholder="Paste a section of Part B here…",
    height=120,
    key=f"pasted_{st.session_state.upload_nonce}",
)

uploaded = st.file_uploader(
    "Or upload the full proposal (DOCX recommended; PDF / TXT accepted)",
    type=["docx", "pdf", "txt", "md"],
    accept_multiple_files=False,
    key=f"uploaded_{st.session_state.upload_nonce}",
)

# ── Stash uploaded bytes; the worker handles extraction/chunking/embedding ─
has_content = False

if uploaded and not pasted.strip():
    new_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.embedded_key != new_key:
        st.session_state.embedded_key = new_key
        st.session_state.proposal_bytes = uploaded.getvalue()
        st.session_state.proposal_suffix = (
            ("." + uploaded.name.rsplit(".", 1)[-1]).lower() if "." in uploaded.name else ""
        )
    if st.session_state.proposal_bytes:
        size_kb = len(st.session_state.proposal_bytes) / 1024
        st.info(
            f"📎 Ready to queue: **{uploaded.name}** "
            f"({size_kb:.1f} KB). Text extraction and embedding will run on the worker after you press *Queue evaluation*.",
            icon="📎",
        )
        has_content = True

elif pasted.strip():
    has_content = True

# ── Config summary + Run button ────────────────────────────────────────────
st.subheader("2. Configuration")
total_weight = sum(c.weight for c in st.session_state.criteria)
max_total = sum(c.weight * c.max_score for c in st.session_state.criteria)
threshold_ratio = overall_threshold / max_total if max_total else 0.0

with st.container(border=True):
    cols = st.columns([2, 1, 1, 1, 1])
    cols[0].markdown(
        f"<small>Proposal · Call</small><br>"
        f"<b>{st.session_state.proposal_acronym or '—'}</b>  `{st.session_state.call_id or '—'}`",
        unsafe_allow_html=True,
    )
    cols[1].markdown(f"<small>Action</small><br><b>{st.session_state.action_type}</b>", unsafe_allow_html=True)
    cols[2].markdown(f"<small>Criteria</small><br><b>{len(st.session_state.criteria)}</b>", unsafe_allow_html=True)
    cols[3].markdown(f"<small>Max score</small><br><b>{max_total:g}</b>", unsafe_allow_html=True)
    cols[4].markdown(
        f"<small>Threshold</small><br><b>{overall_threshold:g}</b> · {threshold_ratio * 100:.0f}%",
        unsafe_allow_html=True,
    )

run_cols = st.columns([3, 2, 3])
with run_cols[1]:
    run_clicked = st.button(
        "Queue evaluation",
        type="primary",
        disabled=not has_content,
        use_container_width=True,
        key="run_eval_btn",
    )


def _validate() -> list[str]:
    errors: list[str] = []
    if not st.session_state.call_id.strip():
        errors.append("Call ID is required.")
    if not st.session_state.proposal_acronym.strip():
        errors.append("Proposal acronym is required.")
    if not st.session_state.submitted_by.strip():
        errors.append("Submitted by is required.")
    if not st.session_state.criteria:
        errors.append("At least one criterion is required.")
    return errors


if run_clicked:
    errors = _validate()
    if errors:
        for e in errors:
            st.error(e)
    else:
        is_full_proposal = uploaded is not None and not pasted.strip()
        proposal_bytes_for_job = st.session_state.proposal_bytes if is_full_proposal else None
        suffix_for_job = st.session_state.proposal_suffix if is_full_proposal else None
        pasted_for_job = pasted.strip() if pasted.strip() else None

        try:
            job_id = jobs.enqueue(
                call_id=st.session_state.call_id.strip(),
                proposal_acronym=st.session_state.proposal_acronym.strip(),
                action_type=st.session_state.action_type,
                overall_threshold=float(overall_threshold),
                criteria=_criterion_dicts(),
                proposal_bytes=proposal_bytes_for_job,
                proposal_suffix=suffix_for_job,
                pasted_text=pasted_for_job,
                persist=is_full_proposal,
                submitted_by=st.session_state.submitted_by.strip(),
            )
            st.session_state.my_job_ids = list(st.session_state.my_job_ids) + [job_id]
            st.toast(
                f"Queued '{st.session_state.proposal_acronym.strip()}' "
                f"({'pasted text — not saved to history' if not is_full_proposal else 'saved to history when done'})",
                icon="📥",
            )
            # Reset the proposal-scoped fields for the next submission; Call ID,
            # Action type, criteria/weights, threshold and Submitted by carry over.
            st.session_state.proposal_acronym = ""
            st.session_state.embedded_key = None
            st.session_state.proposal_bytes = None
            st.session_state.proposal_suffix = None
            st.session_state.upload_nonce += 1
            st.rerun()
        except Exception as exc:
            st.error(f"Could not queue evaluation: {exc}")

# ── Latest result for this session ─────────────────────────────────────────
result = st.session_state.result
if result:
    st.divider()
    st.header("Evaluation result")

    weighted = result.get("weighted_total", 0.0)
    max_weighted = result.get("max_weighted_total", 1.0) or 1.0
    threshold = result.get("overall_threshold")
    passed = (threshold is None) or (weighted >= threshold)
    band = result.get("predicted_band", "—")
    score_pct = (weighted / max_weighted) * 100 if max_weighted else 0.0

    with st.container(border=True):
        cols = st.columns(4)
        cols[0].metric("Weighted total", f"{weighted:g}", f"/ {max_weighted:g}")
        cols[1].metric("Threshold", f"{threshold:g}" if threshold is not None else "—")
        cols[2].metric("Pass / Fail", "PASS" if passed else "FAIL",
                       delta=f"{'+' if passed else '−'}{abs(weighted - (threshold or 0)):g}",
                       delta_color="normal" if passed else "inverse")
        cols[3].metric("Predicted band", band)
        st.progress(min(weighted / max_weighted, 1.0), text=f"{score_pct:.1f}% of max")
        st.caption(
            f"{result.get('proposal_acronym', '—')} · `{result.get('call_id', '—')}` · "
            f"{result.get('action_type', '—')} · Confidence: {result.get('overall_confidence', '—')}"
        )

    with st.container(border=True):
        st.markdown("##### ESR-style summary")
        st.write(result.get("esr_style_summary", "—"))

    st.markdown("##### Per-criterion breakdown")
    for c in result.get("criteria", []):
        with st.container(border=True):
            _render_criterion_card(c)

    foot = st.columns([1, 1])
    with foot[0]:
        st.download_button(
            "Download JSON",
            data=json.dumps(result, indent=2).encode("utf-8"),
            file_name=f"evaluation_{result.get('call_id', 'edf')}.json",
            mime="application/json", use_container_width=True,
        )
    with foot[1]:
        st.download_button(
            "Download CSV",
            data=_run_to_csv(result),
            file_name=f"evaluation_{result.get('call_id', 'edf')}.csv",
            mime="text/csv", use_container_width=True,
        )
