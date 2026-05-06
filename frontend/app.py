"""EDF Proposal Evaluator — Streamlit frontend (live backend).

Run:
    pip install -r ../requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import db  # noqa: E402
from backend.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL  # noqa: E402
from backend.criteria_io import (  # noqa: E402
    criteria_to_xlsx_bytes,
    xlsx_to_criteria,
)
from backend.pipeline import evaluate_proposal  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

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

DEFAULT_DEVELOPMENT_CRITERIA: list[dict[str, Any]] = (
    DEFAULT_RESEARCH_CRITERIA
    + [
        {
            "name": "Contribution to industrial policy objectives",
            "description": (
                "Contribution to industrial-policy objectives of the EDF, "
                "including increase of production capacity in the EU defence "
                "industry."
            ),
            "weight": 1,
            "max_score": 5.0,
        },
    ]
)

OVERALL_THRESHOLD_DEFAULT = 30.0


@dataclass
class Criterion:
    name: str
    description: str
    weight: float
    max_score: float = 5.0


def _criterion_dicts() -> list[dict[str, Any]]:
    return [
        {
            "name": c.name,
            "description": c.description,
            "weight": c.weight,
            "max_score": c.max_score,
        }
        for c in st.session_state.criteria
    ]


def _run_to_csv(run: dict[str, Any]) -> bytes:
    """Flatten a stored run into a CSV (UTF-8 BOM for Excel)."""
    out = io.StringIO()
    w = csv.writer(out)

    w.writerow(["Field", "Value"])
    for label, key in [
        ("Proposal acronym", "proposal_acronym"),
        ("Call ID", "call_id"),
        ("Action type", "action_type"),
        ("Generated at", "generated_at"),
        ("Weighted total", "weighted_total"),
        ("Max weighted total", "max_weighted_total"),
        ("Threshold", "overall_threshold"),
        ("Predicted band", "predicted_band"),
        ("Overall confidence", "overall_confidence"),
        ("ESR-style summary", "esr_style_summary"),
    ]:
        w.writerow([label, run.get(key, "")])
    w.writerow([])

    w.writerow(
        [
            "Criterion",
            "Score",
            "Weight",
            "Max score",
            "Contribution",
            "Confidence",
            "Strengths",
            "Weaknesses",
            "Missing evidence",
            "Evaluator comment",
            "Recommendations",
            "Citations",
        ]
    )
    for c in run.get("criteria", []):
        score = c.get("score", 0) or 0
        weight = c.get("weight", 0) or 0
        w.writerow(
            [
                c.get("name", ""),
                score,
                weight,
                c.get("max_score", ""),
                round(score * weight, 4),
                c.get("confidence", ""),
                "\n".join(c.get("strengths") or []),
                "\n".join(c.get("weaknesses") or []),
                "\n".join(c.get("missing_evidence") or []),
                c.get("evaluator_comment", ""),
                "\n".join(c.get("recommendations") or []),
                "\n".join(c.get("citations") or []),
            ]
        )

    # UTF-8 BOM so Excel auto-detects encoding correctly
    return ("﻿" + out.getvalue()).encode("utf-8")


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

_BAND_COLOR = {
    "Likely funded": "green",
    "Reserve list / borderline": "orange",
    "Likely rejected": "red",
}


def _score_color(score: float, max_score: float = 5.0) -> str:
    ratio = score / max_score if max_score else 0.0
    if ratio >= 0.7:
        return "green"
    if ratio >= 0.5:
        return "orange"
    return "red"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .edf-hero h1 { font-size: 2.6rem; margin: 0 0 .25rem 0; }
          .edf-hero .meta { color: #6b7280; font-size: .85rem; }
          .edf-band {
            display: inline-block; padding: .25rem .75rem; border-radius: 999px;
            font-weight: 600; font-size: .85rem;
          }
          .edf-band.green  { background: #dcfce7; color: #166534; }
          .edf-band.orange { background: #ffedd5; color: #9a3412; }
          .edf-band.red    { background: #fee2e2; color: #991b1b; }
          .edf-pill {
            display: inline-block; padding: .15rem .6rem; border-radius: 999px;
            font-weight: 600; font-size: .8rem; margin-right: .35rem;
          }
          .edf-pill.green  { background: #dcfce7; color: #166534; }
          .edf-pill.orange { background: #ffedd5; color: #9a3412; }
          .edf-pill.red    { background: #fee2e2; color: #991b1b; }
          .edf-summary {
            border-left: 4px solid #6366f1; padding: .25rem 0 .25rem .85rem;
            color: #1f2937; font-size: 1.02rem; line-height: 1.55;
          }
          .edf-chip {
            display: inline-block; padding: .15rem .55rem; border-radius: 6px;
            font-size: .8rem; margin: .1rem .25rem .1rem 0;
            background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb;
          }
          .edf-chip b { color: #111827; }
          .edf-chip.accent { background: #eef2ff; border-color: #c7d2fe; color: #3730a3; }
          .edf-chip.warn   { background: #fef3c7; border-color: #fde68a; color: #92400e; }
          /* Compact, light-yellow Run-evaluation button */
          div:has(> .edf-run-anchor) + div button,
          div:has(> .edf-run-anchor) + div [data-testid="stButton"] button {
            background-color: #fef9c3 !important;
            color: #713f12 !important;
            border: 1px solid #fde68a !important;
            font-weight: 600 !important;
            padding: .35rem 1rem !important;
            font-size: .9rem !important;
            box-shadow: none !important;
          }
          div:has(> .edf-run-anchor) + div button:hover,
          div:has(> .edf-run-anchor) + div [data-testid="stButton"] button:hover {
            background-color: #fde68a !important;
            border-color: #fbbf24 !important;
            color: #78350f !important;
          }
          div:has(> .edf-run-anchor) + div button:disabled,
          div:has(> .edf-run-anchor) + div [data-testid="stButton"] button:disabled {
            background-color: #f9fafb !important;
            color: #9ca3af !important;
            border-color: #e5e7eb !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Refresh criteria dialog (download template + upload XLSX)
# ---------------------------------------------------------------------------

@st.dialog("Refresh criteria", width="small")
def refresh_criteria_dialog() -> None:
    st.caption(
        "Update the criteria list by editing an XLSX. Download the current "
        "set, edit / extend it, then upload it back here."
    )

    st.markdown("**1. Download template**")
    st.caption(
        "Exports the current criteria. Columns: `name`, `description`, "
        "`weight`, `max_score`."
    )
    st.download_button(
        "Download XLSX template",
        data=criteria_to_xlsx_bytes(_criterion_dicts()),
        file_name="edf_criteria_template.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True,
    )

    st.divider()
    st.markdown("**2. Upload edited XLSX**")
    uploaded = st.file_uploader(
        "Choose an .xlsx file",
        type=["xlsx"],
        accept_multiple_files=False,
        key="dlg_criteria_xlsx",
    )
    if uploaded is not None:
        if st.button(
            "Apply uploaded criteria", type="primary", use_container_width=True
        ):
            try:
                parsed = xlsx_to_criteria(uploaded.read())
                st.session_state.criteria = [Criterion(**c) for c in parsed]
                st.success(f"Loaded {len(parsed)} criteria from XLSX.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse XLSX: {exc}")


# ---------------------------------------------------------------------------
# Wipe history dialog
# ---------------------------------------------------------------------------

@st.dialog("Wipe history database?", width="small")
def wipe_db_dialog() -> None:
    st.warning(
        "This permanently deletes **every** stored run. Cannot be undone.",
        icon=":material/warning:",
    )
    cols = st.columns(2)
    with cols[0]:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with cols[1]:
        if st.button(
            "Wipe everything",
            type="primary",
            use_container_width=True,
        ):
            removed = db.wipe_all()
            st.success(f"Removed {removed} run(s) from history.")
            st.rerun()


# ---------------------------------------------------------------------------
# Run-details renderer (shared between dialogs)
# ---------------------------------------------------------------------------

def _render_run_details(run: dict[str, Any]) -> None:
    st.markdown(
        f"### {run.get('proposal_acronym', '—')}  ·  `{run.get('call_id')}`"
    )
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
    cols[1].metric(
        "Threshold", f"{threshold:g}" if threshold is not None else "—"
    )
    cols[2].metric("Pass / fail", "PASS" if passed else "FAIL")
    cols[3].metric("Predicted band", run.get("predicted_band", "—"))

    st.markdown("**ESR-style summary**")
    st.write(run.get("esr_style_summary", "—"))

    st.markdown("**Per-criterion**")
    for c in run.get("criteria", []):
        with st.expander(
            f"{c['name']} — {c['score']}/5 · weight ×{c['weight']}"
        ):
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
            st.markdown("**Evaluator-style comment**")
            st.write(c.get("evaluator_comment", "—"))
            st.markdown("**Recommendations**")
            for s in c.get("recommendations") or ["—"]:
                st.markdown(f"- {s}")
            st.caption(
                "Citations: " + (", ".join(c.get("citations") or []) or "—")
            )

    actions = st.columns([1, 1, 1])
    with actions[0]:
        st.download_button(
            ":material/table_view: Export CSV",
            data=_run_to_csv(run),
            file_name=f"evaluation_{run.get('call_id', 'edf')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with actions[1]:
        st.download_button(
            ":material/code: Export JSON",
            data=json.dumps(run, indent=2).encode("utf-8"),
            file_name=f"evaluation_{run.get('call_id', 'edf')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with actions[2]:
        if st.button(
            ":material/delete: Delete this run",
            type="secondary",
            use_container_width=True,
        ):
            db.delete_run(run.get("call_id", ""))
            st.session_state.pending_dialog = "history"
            st.rerun()


# ---------------------------------------------------------------------------
# History list dialog
# ---------------------------------------------------------------------------

@st.dialog("History of past runs", width="large")
def history_dialog() -> None:
    all_runs = db.list_runs()
    if not all_runs:
        st.info("No runs saved yet. Each completed evaluation is saved here.")
        return

    query = st.text_input(
        ":material/search: Search",
        placeholder="Filter by Call ID, acronym, action type, or band…",
        key="history_search_input",
    ).strip()

    runs = all_runs
    if query:
        ql = query.lower()
        runs = [
            r
            for r in all_runs
            if ql in (r.get("call_id") or "").lower()
            or ql in (r.get("proposal_acronym") or "").lower()
            or ql in (r.get("predicted_band") or "").lower()
            or ql in (r.get("action_type") or "").lower()
        ]

    st.caption(
        f"Showing {len(runs)} of {len(all_runs)} run(s). "
        "Click View to open the full LLM evaluation in a new popup."
    )

    if not runs:
        st.info(f"No matching runs for '{query}'.")
        return

    # Table-style header
    header = st.columns([2, 3, 1.5, 2, 1.2, 0.9])
    for col, label in zip(
        header,
        ["Acronym", "Call ID", "Date", "Band", "Score", ""],
    ):
        col.markdown(f"**{label}**")
    st.markdown(
        "<hr style='margin:.2rem 0; border:none; "
        "border-top:1px solid #e5e7eb'>",
        unsafe_allow_html=True,
    )

    for r in runs:
        cols = st.columns([2, 3, 1.5, 2, 1.2, 0.9])
        with cols[0]:
            st.markdown(f"**{r.get('proposal_acronym') or '—'}**")
        with cols[1]:
            st.markdown(
                f"<code style='font-size:.8rem'>{r.get('call_id')}</code>",
                unsafe_allow_html=True,
            )
        with cols[2]:
            d = (r.get("generated_at") or "")[:10]
            st.markdown(d or "—")
        with cols[3]:
            band = r.get("predicted_band") or "—"
            color = _BAND_COLOR.get(band, "blue")
            st.markdown(
                f"<span class='edf-pill {color}'>{band}</span>",
                unsafe_allow_html=True,
            )
        with cols[4]:
            st.markdown(
                f"{r['weighted_total']:g}/{r['max_weighted_total']:g}"
            )
        with cols[5]:
            if st.button(
                "View",
                key=f"hist_view_{r['call_id']}",
                use_container_width=True,
            ):
                # Set a pending-dialog flag and rerun. st.rerun() inside a
                # dialog dismisses it; the top-of-script dispatcher will
                # then open the run-detail dialog as a *new* popup.
                st.session_state.pending_dialog = "detail"
                st.session_state.detail_call_id = r["call_id"]
                st.rerun()


# ---------------------------------------------------------------------------
# Run-detail dialog (opened from the history list)
# ---------------------------------------------------------------------------

@st.dialog("Past run details", width="large")
def run_detail_dialog(call_id: str) -> None:
    run = db.get_run(call_id)
    if run is None:
        st.error("Run no longer exists in history.")
        if st.button(
            ":material/arrow_back: Back to history",
            use_container_width=True,
        ):
            st.session_state.pending_dialog = "history"
            st.rerun()
        return

    if st.button(
        ":material/arrow_back: Back to history",
        use_container_width=True,
    ):
        st.session_state.pending_dialog = "history"
        st.rerun()

    st.divider()
    _render_run_details(run)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EDF Proposal Evaluator (internal)",
    page_icon=":bar_chart:",
    layout="wide",
)

# Session state init — note Call ID and acronym are intentionally empty.
if "criteria" not in st.session_state:
    st.session_state.criteria = [Criterion(**c) for c in DEFAULT_RESEARCH_CRITERIA]
if "result" not in st.session_state:
    st.session_state.result = None
if "action_type" not in st.session_state:
    st.session_state.action_type = "Research"
if "call_id" not in st.session_state:
    st.session_state.call_id = ""
if "proposal_acronym" not in st.session_state:
    st.session_state.proposal_acronym = ""
# Pending-run pattern: when set, the next rerun executes the LLM. Keeps the
# sidebar rendered with disabled buttons during the run so a dialog click
# can't interrupt the evaluation.
if "pending_run_params" not in st.session_state:
    st.session_state.pending_run_params = None

# Dialog dispatcher slots — set by buttons, consumed at the top of the next
# rerun to open the right popup. Lets us navigate between separate dialogs.
if "pending_dialog" not in st.session_state:
    st.session_state.pending_dialog = None
if "detail_call_id" not in st.session_state:
    st.session_state.detail_call_id = None

is_running: bool = st.session_state.pending_run_params is not None

_inject_css()

st.title("EDF Proposal Evaluator")
st.caption(
    "Internal pre-screening tool — simulates an ESR-style evaluation of an "
    "EDF Part B proposal. Final judgement always rests with human evaluators."
)
st.caption(
    f"Backend: Ollama @ `{OLLAMA_BASE_URL}` · chat=`{CHAT_MODEL}` · "
    f"embed=`{EMBED_MODEL}`"
)

# Dialog dispatcher: consume the flag (so X-close doesn't reopen the dialog
# on the next interaction) and open the requested popup. This is how the
# History list and the per-run Detail popup hand off to each other.
_pending_dialog = st.session_state.pending_dialog
st.session_state.pending_dialog = None
if _pending_dialog == "history":
    history_dialog()
elif _pending_dialog == "detail" and st.session_state.detail_call_id:
    run_detail_dialog(st.session_state.detail_call_id)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Call configuration")

    st.session_state.call_id = st.text_input(
        "Call ID *",
        value=st.session_state.call_id,
        placeholder="e.g. EDF-2024-LS-RA-CHALLENGE",
    )
    st.session_state.proposal_acronym = st.text_input(
        "Proposal acronym *",
        value=st.session_state.proposal_acronym,
        placeholder="e.g. SWARMER",
    )

    new_action = st.selectbox(
        "Action type",
        ["Research", "Development"],
        index=0 if st.session_state.action_type == "Research" else 1,
    )
    if new_action != st.session_state.action_type:
        st.session_state.action_type = new_action
        defaults = (
            DEFAULT_RESEARCH_CRITERIA
            if new_action == "Research"
            else DEFAULT_DEVELOPMENT_CRITERIA
        )
        st.session_state.criteria = [Criterion(**c) for c in defaults]

    st.divider()
    st.subheader("Criteria & weights")
    st.caption(
        "Defaults loaded for the selected action type. Edit on demand, or "
        "import / export the full set as XLSX."
    )

    to_delete: list[int] = []
    for idx, crit in enumerate(st.session_state.criteria):
        with st.expander(f"{idx + 1}. {crit.name}", expanded=False):
            crit.name = st.text_input("Name", value=crit.name, key=f"name_{idx}")
            crit.description = st.text_area(
                "Description",
                value=crit.description,
                key=f"desc_{idx}",
                height=80,
            )
            cols = st.columns(2)
            with cols[0]:
                crit.weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(crit.weight),
                    step=0.5,
                    key=f"w_{idx}",
                )
            with cols[1]:
                crit.max_score = st.number_input(
                    "Max score",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(crit.max_score),
                    step=0.5,
                    key=f"ms_{idx}",
                )
            if st.button("Remove", key=f"del_{idx}"):
                to_delete.append(idx)

    for idx in reversed(to_delete):
        st.session_state.criteria.pop(idx)

    if st.button(
        "+ Add criterion", use_container_width=True, disabled=is_running
    ):
        st.session_state.criteria.append(
            Criterion(
                name=f"New criterion {len(st.session_state.criteria) + 1}",
                description="",
                weight=1.0,
                max_score=5.0,
            )
        )

    if st.button(
        ":material/refresh: Refresh criteria",
        use_container_width=True,
        disabled=is_running,
        help=(
            "Open a popup to download the current criteria as XLSX or "
            "upload an edited one."
        ),
    ):
        refresh_criteria_dialog()

    if st.button(
        "Reset to defaults", use_container_width=True, disabled=is_running
    ):
        defaults = (
            DEFAULT_RESEARCH_CRITERIA
            if st.session_state.action_type == "Research"
            else DEFAULT_DEVELOPMENT_CRITERIA
        )
        st.session_state.criteria = [Criterion(**c) for c in defaults]
        st.rerun()

    st.divider()
    overall_threshold = st.number_input(
        "Overall pass threshold",
        min_value=0.0,
        max_value=200.0,
        value=OVERALL_THRESHOLD_DEFAULT,
        step=0.5,
    )

    # ---- History ---------------------------------------------------------
    st.divider()
    runs_count = len(db.list_runs())
    history_label = (
        f":material/history: History of past runs  ({runs_count})"
        if runs_count
        else ":material/history: History of past runs"
    )
    if st.button(
        history_label,
        use_container_width=True,
        disabled=is_running or runs_count == 0,
        help=(
            "Open a popup to search past evaluations as a table. "
            "Click any row to see the full LLM evaluation."
            if runs_count
            else "Run an evaluation on a full proposal to populate the history."
        ),
    ):
        st.session_state.pending_dialog = "history"
        st.session_state.detail_call_id = None
        st.rerun()

    if runs_count:
        if st.button(
            ":material/delete_forever: Wipe history database",
            type="secondary",
            use_container_width=True,
            disabled=is_running,
            help="Permanently delete every saved run.",
        ):
            wipe_db_dialog()

    if is_running:
        st.divider()
        st.warning(
            "Evaluation in progress — sidebar actions disabled until it finishes.",
            icon=":material/hourglass_top:",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.subheader("1. Upload Part B")
uploaded = st.file_uploader(
    "Drop the proposal here (DOCX recommended; PDF / TXT accepted)",
    type=["docx", "pdf", "txt", "md"],
    accept_multiple_files=False,
)
pasted = st.text_area(
    "…or paste sections directly",
    placeholder=(
        "Paste a section of Part B here if you want to evaluate just "
        "part of the proposal."
    ),
    height=160,
    help=(
        "Pasted-section runs are evaluated but NOT saved to the history "
        "database. Only full uploaded proposals are persisted."
    ),
)
if pasted.strip():
    st.info(
        "Pasted-text mode: this run will not be saved to the history "
        "database. Upload a full proposal file to persist a run.",
        icon=":material/info:",
    )

st.subheader("2. Configuration summary")
total_weight = sum(c.weight for c in st.session_state.criteria)
max_total = sum(c.weight * c.max_score for c in st.session_state.criteria)
threshold_ratio = overall_threshold / max_total if max_total else 0.0
with st.container(border=True):
    sum_cols = st.columns([2, 1, 1, 1, 1])
    with sum_cols[0]:
        st.markdown(
            f"<small>Proposal · Call</small><br>"
            f"<b>{st.session_state.proposal_acronym or '—'}</b>  "
            f"<span style='color:#6b7280;font-family:ui-monospace,monospace;"
            f"font-size:.85rem'>{st.session_state.call_id or '—'}</span>",
            unsafe_allow_html=True,
        )
    with sum_cols[1]:
        st.markdown(
            f"<small>Action</small><br><b>{st.session_state.action_type}</b>",
            unsafe_allow_html=True,
        )
    with sum_cols[2]:
        st.markdown(
            f"<small>Criteria</small><br><b>"
            f"{len(st.session_state.criteria)}</b>",
            unsafe_allow_html=True,
        )
    with sum_cols[3]:
        st.markdown(
            f"<small>Max score</small><br><b>{max_total:g}</b>"
            f"<span style='color:#6b7280;font-size:.8rem'>"
            f" (w {total_weight:g})</span>",
            unsafe_allow_html=True,
        )
    with sum_cols[4]:
        st.markdown(
            f"<small>Threshold</small><br><b>{overall_threshold:g}</b>"
            f"<span style='color:#6b7280;font-size:.8rem'>"
            f" · {threshold_ratio * 100:.0f}%</span>",
            unsafe_allow_html=True,
        )

# Compact, light-yellow Run button — centred, small
run_btn_cols = st.columns([3, 2, 3])
with run_btn_cols[1]:
    st.markdown(
        '<div class="edf-run-anchor"></div>', unsafe_allow_html=True
    )
    run_clicked = st.button(
        "Run evaluation",
        disabled=(uploaded is None and not pasted.strip()),
        use_container_width=True,
        key="run_eval_btn",
    )


def _validate() -> list[str]:
    errors: list[str] = []
    if not st.session_state.call_id.strip():
        errors.append("Call ID is required.")
    if not st.session_state.proposal_acronym.strip():
        errors.append("Proposal acronym is required.")
    if not st.session_state.criteria:
        errors.append("At least one criterion is required.")
    return errors


if run_clicked and not is_running:
    errors = _validate()
    if errors:
        for e in errors:
            st.error(e)
    else:
        proposal_bytes: bytes | None = None
        proposal_suffix: str | None = None
        proposal_text = pasted.strip()
        if not proposal_text and uploaded is not None:
            proposal_bytes = uploaded.read()
            name = uploaded.name.lower()
            proposal_suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ""
            from backend.ingestion import extract_text as _extract

            try:
                proposal_text = _extract(proposal_bytes, proposal_suffix)
            except Exception as exc:
                st.error(f"Could not extract text: {exc}")
                proposal_text = ""

        if not proposal_text:
            st.error("Could not extract any text from the input.")
        else:
            is_full_proposal = (
                uploaded is not None and not pasted.strip()
            )
            # Stash everything for the deferred run, then rerun so the sidebar
            # re-renders with all interactive buttons disabled before the LLM
            # call starts. This prevents a dialog click from interrupting it.
            st.session_state.pending_run_params = {
                "proposal_text": proposal_text,
                "proposal_bytes": proposal_bytes,
                "proposal_suffix": proposal_suffix,
                "criteria": _criterion_dicts(),
                "action_type": st.session_state.action_type,
                "call_id": st.session_state.call_id.strip(),
                "overall_threshold": overall_threshold,
                "proposal_acronym": st.session_state.proposal_acronym.strip(),
                "persist": is_full_proposal,
            }
            st.rerun()


if is_running:
    params = st.session_state.pending_run_params
    progress = st.progress(0.0, text="Starting…")
    status = st.status("Running evaluation…", expanded=True)

    def progress_cb(message: str, fraction: float) -> None:
        progress.progress(min(max(fraction, 0.0), 1.0), text=message)
        status.write(f"• {message}")

    try:
        result = evaluate_proposal(
            proposal_text=params["proposal_text"],
            criteria=params["criteria"],
            action_type=params["action_type"],
            call_id=params["call_id"],
            overall_threshold=params["overall_threshold"],
            proposal_acronym=params["proposal_acronym"],
            proposal_bytes=params["proposal_bytes"],
            proposal_suffix=params["proposal_suffix"],
            progress_cb=progress_cb,
            persist=params["persist"],
        )
        st.session_state.result = result
        status.update(label="Evaluation complete", state="complete")
        if params["persist"]:
            st.toast("Saved to history.", icon=":material/save:")
        else:
            st.toast(
                "Partial-text run — not saved to history.",
                icon=":material/info:",
            )
    except Exception as exc:
        status.update(label="Evaluation failed", state="error")
        st.error(
            f"**Evaluation failed: `{type(exc).__name__}: {exc}`**\n\n"
            f"Resolved config in this session:\n"
            f"- Ollama URL: `{OLLAMA_BASE_URL}`\n"
            f"- Chat model: `{CHAT_MODEL}`\n"
            f"- Embed model: `{EMBED_MODEL}`\n\n"
            f"Common causes:\n"
            f"1. Ollama not running — start it (`ollama serve`).\n"
            f"2. Required models not pulled — run "
            f"`ollama pull {CHAT_MODEL}` and `ollama pull {EMBED_MODEL}`.\n"
            f"3. Streamlit running stale code after a config change — "
            f"fully kill the process (Ctrl+C) and `streamlit run` again."
        )
        with st.expander("Full traceback", expanded=True):
            st.code("".join(traceback.format_exception(exc)))
    finally:
        st.session_state.pending_run_params = None
    # Refresh the sidebar (re-enable buttons) by triggering one final rerun.
    st.rerun()


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

result = st.session_state.result
if result:
    st.divider()
    st.header(":material/insights: Evaluation result")

    weighted = result.get("weighted_total", 0.0)
    max_weighted = result.get("max_weighted_total", 1.0) or 1.0
    threshold = result.get("overall_threshold", overall_threshold)
    passed = (threshold is None) or (weighted >= threshold)
    band = result.get("predicted_band", "—")
    band_color = _BAND_COLOR.get(band, "blue")
    score_pct = (weighted / max_weighted) * 100 if max_weighted else 0.0
    overall_color = "green" if passed else "red"

    # ---- Hero card -------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            f"<div class='edf-hero'>"
            f"<h1 style='color:#{('166534' if passed else '991b1b')}'>"
            f"{weighted:g}"
            f"<span style='color:#9ca3af;font-size:.55em'> / "
            f"{max_weighted:g}</span></h1>"
            f"<div class='meta'>"
            f"<b>{result.get('proposal_acronym', '—')}</b> · "
            f"<code>{result.get('call_id', '—')}</code> · "
            f"{result.get('action_type', '—')} · "
            f"generated {result.get('generated_at', '—')[:19].replace('T', ' ')}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.progress(
            min(weighted / max_weighted, 1.0),
            text=f"{score_pct:.1f}% of max",
        )
        cols = st.columns(4)
        with cols[0]:
            st.metric(
                "Threshold",
                f"{threshold:g}" if threshold is not None else "—",
            )
        with cols[1]:
            st.metric(
                "Pass / Fail",
                "PASS" if passed else "FAIL",
                delta=(
                    f"{'+' if passed else '−'}{abs(weighted - (threshold or 0)):g}"
                ),
                delta_color="normal" if passed else "inverse",
            )
        with cols[2]:
            st.markdown("<small>Predicted band</small>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='edf-band {band_color}'>{band}</span>",
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.metric(
                "Confidence", result.get("overall_confidence", "—").title()
            )

    # ---- ESR-style summary ----------------------------------------------
    with st.container(border=True):
        st.markdown("##### :material/description: ESR-style summary")
        st.markdown(
            f"<div class='edf-summary'>"
            f"{result.get('esr_style_summary', '—')}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- Per-criterion --------------------------------------------------
    st.markdown("##### :material/list_alt: Per-criterion breakdown")
    for c in result.get("criteria", []):
        score = c["score"]
        weight = c["weight"]
        max_score = c.get("max_score", 5.0)
        contribution = score * weight
        color = _score_color(score, max_score)
        with st.container(border=True):
            head = st.columns([5, 1, 1, 1])
            with head[0]:
                st.markdown(f"**{c['name']}**")
                st.caption(f"Confidence: {c.get('confidence', '—').title()}")
            with head[1]:
                st.markdown(
                    f"<small>Score</small><br>"
                    f"<span class='edf-pill {color}'>"
                    f"{score:g} / {max_score:g}</span>",
                    unsafe_allow_html=True,
                )
            with head[2]:
                st.markdown(
                    f"<small>Weight</small><br><b>×{weight:g}</b>",
                    unsafe_allow_html=True,
                )
            with head[3]:
                st.markdown(
                    f"<small>Contribution</small><br><b>{contribution:g}</b>",
                    unsafe_allow_html=True,
                )
            st.progress(min(score / max_score, 1.0))
            with st.expander(":material/expand_more: Details"):
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("**:material/check_circle: Strengths**")
                    for s in c.get("strengths") or ["—"]:
                        st.markdown(f"- {s}")
                with cols[1]:
                    st.markdown("**:material/error: Weaknesses**")
                    for s in c.get("weaknesses") or ["—"]:
                        st.markdown(f"- {s}")
                with cols[2]:
                    st.markdown("**:material/help: Missing evidence**")
                    for s in c.get("missing_evidence") or ["—"]:
                        st.markdown(f"- {s}")
                st.markdown("**:material/format_quote: Evaluator-style comment**")
                st.write(c.get("evaluator_comment", "—"))
                st.markdown("**:material/lightbulb: Recommended improvements**")
                for s in c.get("recommendations") or ["—"]:
                    st.markdown(f"- {s}")
                cits = ", ".join(c.get("citations") or []) or "—"
                st.caption(f"Citations: {cits}")

    # ---- Footer actions -------------------------------------------------
    foot = st.columns([1, 1])
    with foot[0]:
        st.download_button(
            ":material/download: Download result as JSON",
            data=json.dumps(result, indent=2).encode("utf-8"),
            file_name=f"evaluation_{result.get('call_id', 'edf')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with foot[1]:
        with st.expander(":material/code: Raw JSON"):
            st.json(result, expanded=False)
