from __future__ import annotations

import csv
import io
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ollama
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import auth, db, jobs  # noqa: E402
from backend.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL  # noqa: E402
from backend.criteria_io import criteria_to_xlsx_bytes, xlsx_to_criteria  # noqa: E402

AUTH_PORT = 8502


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
# Login gate — nothing below renders without a valid session from the auth app.
# ---------------------------------------------------------------------------

if not st.session_state.get("authed_username"):
    _sid = st.query_params.get("sid")
    _username = auth.get_username(_sid) if _sid else None
    if _username:
        st.session_state.authed_username = _username
        st.session_state.authed_sid = _sid
    else:
        st.title("EDF Proposal Evaluator")
        st.warning("Please log in first.")
        try:
            _host_header = st.context.headers.get("host")
        except Exception:
            _host_header = None
        _login_url = auth.peer_url(_host_header, AUTH_PORT)
        st.markdown(
            f'<a href="{_login_url}" target="_self" '
            'style="display:inline-block;padding:.5rem 1rem;border:1px solid #000;'
            'background:#000;color:#fff;text-decoration:none;font-weight:600;">'
            "Go to login →</a>",
            unsafe_allow_html=True,
        )
        st.stop()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Brutalist palette: white/black/gray only. Red is reserved for errors
   and destructive actions (Wipe history, Delete this run) — nothing else. ── */
button[data-testid="stBaseButton-primary"] {
    background-color: #dc2626 !important;
    border: 1px solid #dc2626 !important;
    border-radius: 0 !important;
    color: #fff !important;
    box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #991b1b !important;
    border-color: #991b1b !important;
}
button[data-testid="stBaseButton-primary"]:disabled {
    background-color: #d4d4d4 !important;
    border-color: #d4d4d4 !important;
    color: #737373 !important;
}
/* The main non-destructive CTA ("Queue evaluation"): solid black, marked via
   an adjacent invisible span since Streamlit gives buttons no other stable,
   CSS-selectable hook — same :has()-on-column technique used below for the
   queue-bar chip/× color-linking. */
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"] {
    background: #000 !important;
    color: #fff !important;
    border-color: #000 !important;
}
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"]:hover {
    background: #262626 !important;
}
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"]:disabled {
    background: #d4d4d4 !important;
    color: #737373 !important;
    border-color: #d4d4d4 !important;
}

/* ── Queue status bar ─────────────────────────────────────────────────── */
/* Fixed to the bottom of the viewport with margin on all sides, so it reads
   as a distinct floating panel rather than sitting flush under the page
   content — same white/black palette as the rest of the site, no inversion.
   stMain (the content area, excluding the sidebar) is made the containing
   block for our fixed elements via a no-op transform, so left/right below
   are relative to the content area only — never hidden under the sidebar,
   and correct even when the sidebar is resized or collapsed. */
[data-testid="stAppViewContainer"] .block-container {
    padding-bottom: 7rem !important;
}
[data-testid="stMain"] {
    transform: translateZ(0);
}
.edf-queue-bar {
    position: fixed;
    left: 1.25rem;
    right: 1.25rem;
    bottom: 1.25rem;
    z-index: 999;
    display: flex;
    align-items: center;
    gap: .9rem;
    padding: .6rem 1rem;
    border: 1px solid #000;
    background: #ffffff;
    font-size: .92rem;
}
.edf-queue-bar.idle {
    color: #525252;
    border-color: #d4d4d4;
}
.edf-queue-bar.busy {
    border-bottom: none;
    color: #000;
}
.edf-queue-bar .dot {
    width: 10px; height: 10px; border-radius: 50%;
    border: 2px solid #000; background: transparent; flex-shrink: 0;
}
.edf-queue-bar.busy .dot {
    background: #000;
}
.edf-queue-bar .summary  { font-weight: 700; white-space: nowrap; }
.edf-queue-bar .msg      { color: #525252; font-size: .82rem;
                            flex: 1; min-width: 0;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.edf-queue-bar .progress { height: 6px; flex: 0 0 160px; max-width: 28%;
                            background: #e5e5e5; overflow: hidden; }
.edf-queue-bar .progress > div {
    height: 100%; background: #000; transition: width .4s ease;
}

/* ── Sliding chip strip — fixed directly above the bar, same width/offset. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) {
    position: fixed !important;
    left: 1.25rem !important;
    right: 1.25rem !important;
    bottom: 4rem !important;
    z-index: 999 !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    overflow-y: visible;
    scroll-behavior: smooth;
    align-items: center !important;
    gap: .55rem !important;
    padding: .6rem .9rem .7rem .9rem;
    background: #f5f5f5;
    border: 1px solid #000;
    border-top: 1px dashed #a3a3a3;
    scrollbar-width: thin;
    scrollbar-color: #000 #e5e5e5;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar { height: 6px; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-track { background: #e5e5e5; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-thumb { background: #000; }

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
/* The markdown container's internal <p> wraps the "…"/"🔒" chips — flatten
   it the same way the button's <p> is flattened, so both sit at the same
   baseline/height as the × button. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    display: contents !important;
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
    border: 1px solid #000;
    border-right: none;
    background: #ffffff;
    color: #000;
    font-size: .88rem;
    line-height: 1;
    white-space: nowrap;
}
.qbar-chip.running   { background: #000; color: #fff; }
.qbar-chip.mine      { font-weight: 700; }
.qbar-chip.cancelling{ border-color: #a3a3a3; background: #f5f5f5; color: #737373;
                        text-decoration: line-through; opacity: .7; }
.qbar-chip code      { background: transparent; padding: 0; color: inherit; font-size: .8rem; opacity: .75; }
.qbar-submitter      { font-size: .75rem; opacity: .7; font-style: italic; margin-left: .15rem; }

/* ── The × — right half of the same pill. Matches the chip's height,
   border, and background; no rounded corners under the brutalist palette. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button {
    height: 2rem !important;
    min-height: 2rem !important;
    width: 1.55rem !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    border: 1px solid #000 !important;
    border-left: none !important;
    background: #ffffff !important;
    color: #000 !important;
    cursor: pointer;
    box-shadow: none !important;
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
    background: #000 !important;
    border-color: #000 !important;
    color: #fff !important;
}

/* Hover — only the × box changes, never the chip. Reserved destructive red. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:hover {
    color: #fff !important;
    background: #dc2626 !important;
    border-color: #dc2626 !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:focus,
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:active {
    outline: none !important;
    box-shadow: none !important;
}

/* Cancelling: a static "…" pill stamped on the right, same dimensions. */
.qbar-cancel-pending {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.55rem;
    height: 2rem;
    border: 1px solid #a3a3a3;
    border-left: none;
    background: #f5f5f5;
    color: #737373;
    font-size: .85rem;
    opacity: .85;
}
.qbar-cancel-locked {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.55rem;
    height: 2rem;
    border: 1px solid #a3a3a3;
    border-left: none;
    background: #f5f5f5;
    color: #737373;
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
    "selected_chat_model": CHAT_MODEL,
    "selected_embed_model": EMBED_MODEL,
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

if not st.session_state.submitted_by:
    st.session_state.submitted_by = st.session_state.authed_username

is_admin = st.session_state.authed_username.strip().lower() == "admin"

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


def _param_count_b(size_str: str | None) -> float:
    """Parse '30.5B' / '137M' into a billions-of-params float, for sorting."""
    if not size_str:
        return 0.0
    s = size_str.strip().upper()
    try:
        if s.endswith("B"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 1000.0
        if s.endswith("K"):
            return float(s[:-1]) / 1_000_000.0
        return float(s)
    except ValueError:
        return 0.0


@st.cache_data(ttl=60)
def _list_models() -> tuple[list[str], list[str]]:
    """Locally pulled Ollama chat models and embedding models, each sorted
    most billions of parameters to least."""
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        pulled = client.list().models
    except Exception:
        return [], []
    chat: list[tuple[float, str]] = []
    embed: list[tuple[float, str]] = []
    for m in pulled:
        size_b = _param_count_b(m.details.parameter_size if m.details else None)
        try:
            caps = client.show(m.model).capabilities or []
        except Exception:
            caps = []
        (embed if "embedding" in caps else chat).append((size_b, m.model))
    chat.sort(key=lambda t: -t[0])
    embed.sort(key=lambda t: -t[0])
    return [n for _, n in chat], [n for _, n in embed]


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

    head = st.columns([5, 1, 1, 1])
    with head[0]:
        st.markdown(f"**{c['name']}**")
        st.caption(f"Confidence: {c.get('confidence', '—').title()}")
    head[1].markdown(
        f"<small>Score</small><br>"
        f"<span style='border:1px solid #000;font-weight:700;"
        f"padding:.2rem .5rem;'>{score:g} / {max_score:g}</span>",
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

    _promote_finished_my_jobs()  # own job(s): toasts + result panel handled inside

    current_ids = {j["job_id"] for j in active}
    known_ids = st.session_state.known_active_ids
    someone_finished = bool(known_ids) and bool(known_ids - current_ids)
    st.session_state.known_active_ids = current_ids

    if someone_finished:
        # No forced full-app rerun here — that used to race against anyone
        # actively typing elsewhere on the page and could wipe in-progress
        # keystrokes. A toast is enough; History/results catch up on the
        # user's own next natural interaction (db.list_runs() is always
        # queried fresh whenever that tab/panel renders).
        st.toast("Another proposal finished processing — History has been updated.", icon="✅")

    if not active:
        st.markdown(
            '<div class="edf-queue-bar idle">'
            '<span class="dot"></span>'
            '<span class="summary">Queue idle</span>'
            '<span style="color:#525252">No proposals being processed.</span>'
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
            try:
                removed = db.wipe_all(requested_by=st.session_state.authed_username)
                st.success(f"Removed {removed} run(s).")
                st.rerun()
            except db.NotAuthorizedError as exc:
                st.error(str(exc))


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
        if is_admin:
            if st.button("Delete this run", type="primary", use_container_width=True):
                try:
                    db.delete_run(run.get("call_id", ""), requested_by=st.session_state.authed_username)
                    st.rerun()
                except db.NotAuthorizedError as exc:
                    st.error(str(exc))
        else:
            st.caption("Only the admin user can delete runs.")


def history_tab() -> None:
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
            st.rerun()
        return
    if st.button("Back to history", use_container_width=True):
        st.rerun()
    st.divider()
    _render_run_details(run)


# ---------------------------------------------------------------------------
# Dialog dispatcher
# ---------------------------------------------------------------------------

_pending = st.session_state.pending_dialog
st.session_state.pending_dialog = None
if _pending == "detail" and st.session_state.detail_call_id:
    run_detail_dialog(st.session_state.detail_call_id)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("EDF Proposal Evaluator")
st.caption(
    f"LLM Model =`{st.session_state.selected_chat_model}` · "
    f"Embeddings Model=`{st.session_state.selected_embed_model}`"
)

# ---------------------------------------------------------------------------
# Sidebar — always editable. Edits apply to the *next* submission; jobs
# already queued or running use the criteria/threshold snapshotted at submit.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(str(Path(__file__).parent / "ikh_logo.png"), width=220)

    _acct_cols = st.columns([3, 1])
    _acct_cols[0].caption(f"**{st.session_state.authed_username}**")
    if _acct_cols[1].button("Log out", use_container_width=True):
        auth.revoke_session(st.session_state.get("authed_sid"))
        st.session_state.authed_username = None
        st.session_state.authed_sid = None
        st.session_state.submitted_by = ""
        st.query_params.clear()
        st.rerun()

    st.divider()
    _chat_options, _embed_options = _list_models()
    if st.session_state.selected_chat_model not in _chat_options:
        _chat_options = [st.session_state.selected_chat_model] + _chat_options
    if st.session_state.selected_embed_model not in _embed_options:
        _embed_options = [st.session_state.selected_embed_model] + _embed_options
    _model_cols = st.columns(2)
    with _model_cols[0]:
        st.session_state.selected_chat_model = st.selectbox(
            "Chat model", _chat_options,
            index=_chat_options.index(st.session_state.selected_chat_model),
        )
    with _model_cols[1]:
        st.session_state.selected_embed_model = st.selectbox(
            "Embedding model", _embed_options,
            index=_embed_options.index(st.session_state.selected_embed_model),
        )

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

    if is_admin:
        st.divider()
        runs_count = len(db.list_runs())
        if runs_count:
            if st.button("Wipe history", type="primary", use_container_width=True):
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

tab_evaluate, tab_history = st.tabs(["Evaluate", "History"])

with tab_evaluate:
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
        st.markdown('<span class="edf-cta-marker"></span>', unsafe_allow_html=True)
        run_clicked = st.button(
            "Queue evaluation",
            type="secondary",
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
                    chat_model=st.session_state.selected_chat_model,
                    embed_model=st.session_state.selected_embed_model,
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

with tab_history:
    history_tab()

# ---------------------------------------------------------------------------
# Queue status — bottom of page, auto-refreshing.
# ---------------------------------------------------------------------------

st.divider()
_queue_bar()
