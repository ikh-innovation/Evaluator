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
HERE = Path(__file__).resolve().parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend import auth, db, jobs  # noqa: E402
from backend.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL  # noqa: E402
from backend.criteria_io import criteria_to_xlsx_bytes, xlsx_to_criteria  # noqa: E402
from backend.default_criteria import (  # noqa: E402
    DEFAULT_DEVELOPMENT_CRITERIA,
    DEFAULT_RESEARCH_CRITERIA,
)

import charts  # noqa: E402
import theme  # noqa: E402
from pdf_report import run_to_pdf  # noqa: E402

AUTH_PORT = 8502
OVERALL_THRESHOLD_DEFAULT = 30.0


@dataclass
class Criterion:
    name: str
    description: str
    weight: float
    max_score: float = 5.0


# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EDF Proposal Evaluator",
    page_icon=str(theme.LOGO_PATH) if theme.LOGO_PATH.exists() else ":bar_chart:",
    layout="wide",
)

theme.inject_css()

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
        _gate = st.columns([1, 1.1, 1])[1]
        with _gate:
            st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
            st.image(theme.logo_bytes(with_tagline=True), width=150)
            st.markdown(
                '<div class="edf-apptitle" style="margin-top:.8rem">'
                "EDF Proposal Evaluator</div>"
                '<div class="edf-appsub">Internal pre-screening for European '
                "Defence Fund Part B proposals</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            try:
                _host_header = st.context.headers.get("host")
            except Exception:
                _host_header = None
            _login_url = auth.peer_url(_host_header, AUTH_PORT)
            st.markdown(
                f'<a href="{_login_url}" target="_self" style="display:block;'
                f"padding:.6rem 1rem;border-radius:4px;background:{theme.AMBER};"
                'color:#fff;text-decoration:none;font-weight:600;text-align:center;'
                'font-size:.9rem;">Sign in to continue →</a>',
                unsafe_allow_html=True,
            )
            st.caption("Your session has expired or you have not signed in yet.")
        st.stop()

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "criteria": None,
    "action_type": "Research",
    "call_id": "",
    "call_context": "",
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


def _escape(text: Any) -> str:
    return (
        str(text if text is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


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
    w.writerow(["Criterion", "Score", "Band", "Weight", "Max score", "Contribution",
                "Confidence", "Input coverage", "Strengths", "Weaknesses",
                "Missing evidence", "Input gaps", "Evaluator comment", "Recommendations"])
    for c in run.get("criteria", []):
        score = c.get("score", 0) or 0
        weight = c.get("weight", 0) or 0
        w.writerow([
            c.get("name", ""), score, c.get("band", ""), weight, c.get("max_score", ""),
            round(score * weight, 4), c.get("confidence", ""),
            "full document" if c.get("full_document_input") else c.get("input_coverage", ""),
            "\n".join(c.get("strengths") or []), "\n".join(c.get("weaknesses") or []),
            "\n".join(c.get("missing_evidence") or []),
            "\n".join(c.get("input_gaps") or []), c.get("evaluator_comment", ""),
            "\n".join(c.get("recommendations") or []),
        ])
    return ("﻿" + out.getvalue()).encode("utf-8")


@st.cache_data(ttl=900, max_entries=24, show_spinner=False)
def _pdf_bytes(run: dict[str, Any]) -> bytes:
    """Cached PDF render.

    st.download_button needs the bytes up front, and the result panel is a
    2-second fragment — without this the whole report would be re-laid-out
    on every refresh tick.
    """
    return run_to_pdf(run)


def _verdict(run: dict[str, Any]) -> tuple[float, float, float | None, bool, float]:
    """(weighted, max_weighted, threshold, passed, fraction-of-max)."""
    weighted = float(run.get("weighted_total") or 0.0)
    max_weighted = float(run.get("max_weighted_total") or 0.0) or 1.0
    threshold = run.get("overall_threshold")
    threshold = float(threshold) if threshold is not None else None
    passed = (threshold is None) or (weighted >= threshold)
    return weighted, max_weighted, threshold, passed, weighted / max_weighted


def _fmt_when(value: Any, with_time: bool = True) -> str:
    """'2026-08-06T09:41:12Z' → '2026-08-06 09:41'."""
    text = str(value or "").replace("T", " ").replace("Z", "").strip()
    if not text:
        return "—"
    return text[:16] if with_time else text[:10]


# ---------------------------------------------------------------------------
# Result rendering — one renderer shared by the live result panel and the
# history detail dialog, so the two can never drift apart.
# ---------------------------------------------------------------------------

def _criterion_card(c: dict[str, Any]) -> None:
    score = float(c.get("score") or 0.0)
    max_score = float(c.get("max_score") or 5.0) or 5.0
    weight = float(c.get("weight") or 0.0)
    frac = min(score / max_score, 1.0) if max_score else 0.0
    color = theme.score_color(frac)

    meta = [f"Confidence: {(c.get('confidence') or '—').title()}"]
    band = (c.get("band") or "").strip()
    if band:
        meta.append(f"Band: {band.title()}")
    if c.get("full_document_input"):
        meta.append("Input: full document")
    elif c.get("input_coverage") is not None:
        meta.append(f"Input: excerpts (~{float(c['input_coverage']) * 100:.0f}% of document)")

    head = st.columns([6, 1.5, 1, 1.4], vertical_alignment="center")
    with head[0]:
        st.markdown(
            f'<span class="edf-dense"></span>'
            f'<div class="edf-crit-name">{_escape(c.get("name") or "—")}</div>'
            f'<div class="edf-crit-meta">{_escape(" · ".join(meta))}</div>',
            unsafe_allow_html=True,
        )
    head[1].markdown(
        f'<div class="edf-score-badge"><b>{score:g}</b>'
        f"<span>/ {max_score:g}</span></div>",
        unsafe_allow_html=True,
    )
    head[2].markdown(
        f'<div class="edf-tile-inline"><div class="edf-crit-meta">Weight</div>'
        f'<div class="edf-num">×{weight:g}</div></div>',
        unsafe_allow_html=True,
    )
    head[3].markdown(
        f'<div class="edf-tile-inline"><div class="edf-crit-meta">Contribution</div>'
        f'<div class="edf-num">{score * weight:g}'
        f"<small> / {max_score * weight:g}</small></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(theme.meter(frac, color), unsafe_allow_html=True)

    with st.expander("Findings, evidence gaps and recommendations"):
        cols = st.columns(3)
        cols[0].markdown(
            theme.bullet_group("Strengths", c.get("strengths"), "good"),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            theme.bullet_group(
                "Weaknesses", c.get("weaknesses"), "bad",
                empty="No significant weaknesses identified.",
            ),
            unsafe_allow_html=True,
        )
        cols[2].markdown(
            theme.bullet_group("Missing evidence", c.get("missing_evidence")),
            unsafe_allow_html=True,
        )
        if c.get("input_gaps"):
            st.markdown(
                theme.bullet_group(
                    "Input gaps — not seen by the model, not penalised",
                    c["input_gaps"], "gap",
                ),
                unsafe_allow_html=True,
            )
        if c.get("evaluator_comment"):
            st.markdown(
                '<div class="edf-lgroup"><div class="lg-h"><i></i>Evaluator comment</div>'
                f"{theme.prose(c['evaluator_comment'])}</div>",
                unsafe_allow_html=True,
            )
        if c.get("recommendations"):
            st.markdown(
                theme.bullet_group("Recommendations", c["recommendations"], "gap"),
                unsafe_allow_html=True,
            )


def _verdict_panel(run: dict[str, Any]) -> None:
    weighted, max_weighted, threshold, passed, frac = _verdict(run)

    cols = st.columns([1.05, 3], vertical_alignment="center")
    with cols[0]:
        # "content", not "stretch" — the donut is square by construction and
        # stretching it to the column width distorts the ring.
        st.altair_chart(
            charts.overall_donut(weighted, max_weighted, passed), width="content"
        )
    with cols[1]:
        if threshold is None:
            thr_value, thr_sub = "—", "not configured"
            margin = "no threshold set"
        else:
            delta = weighted - threshold
            thr_value = f"{threshold:g}"
            thr_sub = f"{threshold / max_weighted * 100:.0f}% of maximum"
            margin = f"{'+' if delta >= 0 else '−'}{abs(delta):g} against threshold"
        st.markdown(
            theme.tiles(
                theme.tile("Weighted total", f"{weighted:g}"
                           f"<small> / {max_weighted:g}</small>",
                           f"{frac * 100:.1f}% of maximum", "navy"),
                theme.tile("Pass threshold", thr_value, thr_sub, "accent"),
                theme.tile("Outcome", "PASS" if passed else "FAIL", margin,
                           "pass" if passed else "fail"),
                theme.tile("Predicted band", str(run.get("predicted_band") or "—"),
                           f"Confidence: {(run.get('overall_confidence') or '—').title()}"),
            ),
            unsafe_allow_html=True,
        )


def render_run(run: dict[str, Any], key: str, *, exports: bool = True,
               deletable: bool = False) -> None:
    """Full read-out for one run: verdict, summary, charts, criterion detail."""
    if run.get("ingest_warning"):
        st.warning(run["ingest_warning"], icon="⚠️")

    _verdict_panel(run)

    criteria = run.get("criteria") or []

    st.markdown(
        theme.section("Assessment summary", "ESR style, generated from the criterion findings"),
        unsafe_allow_html=True,
    )
    st.markdown(theme.prose(run.get("esr_style_summary") or "—"), unsafe_allow_html=True)

    score_chart = charts.criterion_scores(criteria)
    contrib_chart = charts.contribution_bars(criteria)
    if score_chart is not None or contrib_chart is not None:
        st.markdown(
            theme.section("Score profile", "where the marks were won and lost"),
            unsafe_allow_html=True,
        )
        cc = st.columns(2)
        if score_chart is not None:
            with cc[0]:
                st.caption("Score against maximum, per criterion")
                st.altair_chart(score_chart, width="stretch")
        if contrib_chart is not None:
            with cc[1]:
                st.caption("Weighted contribution against the weighted cap")
                st.altair_chart(contrib_chart, width="stretch")

    profile = charts.band_profile(criteria)
    if profile is not None:
        st.caption("Consistency across criteria — a flat line is a uniform proposal")
        st.altair_chart(profile, width="stretch")

    if criteria:
        st.markdown(
            theme.section("Criterion detail", f"{len(criteria)} criteria assessed"),
            unsafe_allow_html=True,
        )
        for c in criteria:
            with st.container(border=True):
                _criterion_card(c)

    if exports or deletable:
        st.markdown(theme.section("Export"), unsafe_allow_html=True)
        cid = run.get("call_id", "edf")
        cols = st.columns(4)
        if exports:
            with cols[0]:
                st.download_button(
                    "PDF report", data=_pdf_bytes(run),
                    file_name=f"evaluation_{cid}.pdf", mime="application/pdf",
                    width="stretch", key=f"{key}_pdf",
                )
            with cols[1]:
                st.download_button(
                    "CSV", data=_run_to_csv(run),
                    file_name=f"evaluation_{cid}.csv", mime="text/csv",
                    width="stretch", key=f"{key}_csv",
                )
            with cols[2]:
                st.download_button(
                    "JSON", data=json.dumps(run, indent=2).encode("utf-8"),
                    file_name=f"evaluation_{cid}.json", mime="application/json",
                    width="stretch", key=f"{key}_json",
                )
        with cols[3]:
            if deletable and is_admin:
                if st.button("Delete this run", type="primary", width="stretch",
                             key=f"{key}_del"):
                    try:
                        db.delete_run(str(cid), requested_by=st.session_state.authed_username)
                        st.rerun()
                    except db.NotAuthorizedError as exc:
                        st.error(str(exc))
            elif deletable:
                st.caption("Only the admin user can delete runs.")


# ---------------------------------------------------------------------------
# Queue bar (bottom, full-width, auto-refreshing)
# ---------------------------------------------------------------------------

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
            failed_messages.append(
                (info.get("proposal_acronym", "—"), info.get("error") or "Evaluation failed.")
            )

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
    """Sticky, full-width auto-refreshing queue indicator."""
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
            '<span class="msg">No proposals are being processed.</span>'
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
            f'<span class="pct">{pct}%</span>'
            f'<div class="progress" title="{msg}"><div style="width:{pct}%"></div></div>'
        )

    # ── Top half of the bar: status + summary + running-job progress ──────
    running_msg = _escape(running[0].get("progress_msg") or "") if running else ""
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

            st.markdown(
                f'<div class="{" ".join(classes)}" '
                f'title="{call_id} · submitted by {submitter}">'
                f'<span class="lamp"></span><b>{acronym}</b> '
                f"<code>{call_id}</code>"
                f'<span class="qbar-submitter">{submitter}</span></div>',
                unsafe_allow_html=True,
            )
            is_owner = bool(st.session_state.submitted_by.strip()) and (
                (j.get("submitted_by") or "").strip().lower()
                == st.session_state.submitted_by.strip().lower()
            )

            if is_cancelling:
                st.markdown(
                    '<span class="qbar-cancel-pending" title="Cancelling — the worker '
                    'will stop at the next checkpoint">…</span>',
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
                    f'<span class="qbar-cancel-locked" '
                    f'title="Only {submitter} can cancel this run">🔒</span>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

@st.dialog("Criteria spreadsheet", width="small")
def refresh_criteria_dialog() -> None:
    st.caption(
        "Download the current criteria, edit names, descriptions, weights and "
        "maximum scores in Excel, then upload the file back."
    )
    st.download_button(
        "Download current criteria (XLSX)",
        data=criteria_to_xlsx_bytes(_criterion_dicts()),
        file_name="edf_criteria_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )
    st.divider()
    up = st.file_uploader("Upload edited XLSX", type=["xlsx"], key="dlg_criteria_xlsx")
    if up is not None:
        if st.button("Apply to this session", type="primary", width="stretch"):
            try:
                parsed = xlsx_to_criteria(up.read())
                st.session_state.criteria = [Criterion(**c) for c in parsed]
                st.success(f"Loaded {len(parsed)} criteria.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not parse XLSX: {exc}")


@st.dialog("Wipe history?", width="small")
def wipe_db_dialog() -> None:
    st.warning("This permanently deletes every stored run for every user.", icon="⚠️")
    cols = st.columns(2)
    with cols[0]:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    with cols[1]:
        if st.button("Wipe everything", type="primary", width="stretch"):
            try:
                removed = db.wipe_all(requested_by=st.session_state.authed_username)
                st.success(f"Removed {removed} run(s).")
                st.rerun()
            except db.NotAuthorizedError as exc:
                st.error(str(exc))


@st.dialog("Evaluation detail", width="large")
def run_detail_dialog(call_id: str) -> None:
    run = db.get_run(call_id)
    if run is None:
        st.error("This run no longer exists.")
        if st.button("Back to history", width="stretch"):
            st.rerun()
        return

    top = st.columns([5, 1], vertical_alignment="center")
    with top[0]:
        st.markdown(
            f'<div class="edf-apptitle">{_escape(run.get("proposal_acronym") or "—")}</div>'
            f'<div class="edf-appsub"><b>{_escape(run.get("call_id") or "—")}</b> · '
            f'{_escape(run.get("action_type") or "—")} · '
            f'evaluated {_escape(_fmt_when(run.get("generated_at")))}</div>',
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown('<span class="edf-ghost-marker"></span>', unsafe_allow_html=True)
        if st.button("Close", width="stretch", key="detail_close"):
            st.rerun()
    st.divider()
    render_run(run, key=f"detail_{call_id}", exports=True, deletable=True)


# ---------------------------------------------------------------------------
# Dialog dispatcher
# ---------------------------------------------------------------------------

_pending = st.session_state.pending_dialog
st.session_state.pending_dialog = None
if _pending == "detail" and st.session_state.detail_call_id:
    run_detail_dialog(st.session_state.detail_call_id)


# ---------------------------------------------------------------------------
# Header bar — logo, product name, active models, signed-in user.
# Relocated here from the sidebar so the sidebar is purely configuration.
# ---------------------------------------------------------------------------

_hdr = st.columns([0.50, 0.40, 0.10], vertical_alignment="center")
with _hdr[0]:
    st.markdown(
        theme.appbar(
            "EDF Proposal Evaluator",
            "Internal pre-screening of European Defence Fund Part B proposals · "
            "<b>not an official evaluation</b>",
        ),
        unsafe_allow_html=True,
    )
with _hdr[1]:
    st.markdown(
        '<div class="edf-appmeta">'
        + theme.chip(st.session_state.selected_chat_model, key="LLM", max_len=22)
        + theme.chip(st.session_state.selected_embed_model, key="EMBED", max_len=18)
        + theme.chip(st.session_state.authed_username,
                     variant="solid" if is_admin else "accent",
                     key="ADMIN" if is_admin else "USER", max_len=16)
        + "</div>",
        unsafe_allow_html=True,
    )
with _hdr[2]:
    st.markdown('<span class="edf-logout-slot"></span>', unsafe_allow_html=True)
    if st.button("Sign out", width="stretch", key="logout_btn"):
        auth.revoke_session(st.session_state.get("authed_sid"))
        st.session_state.authed_username = None
        st.session_state.authed_sid = None
        st.session_state.submitted_by = ""
        st.query_params.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — always editable. Edits apply to the *next* submission; jobs
# already queued or running use the criteria/threshold snapshotted at submit.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(theme.section("Call configuration", step=1), unsafe_allow_html=True)
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
        defaults = (
            DEFAULT_RESEARCH_CRITERIA if new_action == "Research"
            else DEFAULT_DEVELOPMENT_CRITERIA
        )
        st.session_state.criteria = [Criterion(**c) for c in defaults]
    st.session_state.call_context = st.text_area(
        "Call topic context (optional)",
        value=st.session_state.call_context,
        height=110,
        placeholder="Paste the call topic description here…",
        help=(
            "Taken from the call fiche / topic description of the specific "
            "call. Supplied per run — never baked into the tool."
        ),
    )

    st.markdown(
        theme.section("Criteria & weights", f"{len(st.session_state.criteria)} active", step=2),
        unsafe_allow_html=True,
    )
    to_delete: list[int] = []
    for idx, crit in enumerate(st.session_state.criteria):
        with st.expander(f"{idx + 1}.  {crit.name}", expanded=False):
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
            if st.button("Remove criterion", key=f"del_{idx}", width="stretch"):
                to_delete.append(idx)
    for idx in reversed(to_delete):
        st.session_state.criteria.pop(idx)

    if st.button("+  Add criterion", width="stretch"):
        st.session_state.criteria.append(
            Criterion(name=f"Criterion {len(st.session_state.criteria) + 1}",
                      description="", weight=1.0, max_score=5.0)
        )
    a, b = st.columns(2)
    with a:
        if st.button("Spreadsheet", width="stretch",
                     help="Import / export criteria as XLSX"):
            refresh_criteria_dialog()
    with b:
        if st.button("Reset", width="stretch", help="Restore the default criteria set"):
            defaults = (
                DEFAULT_RESEARCH_CRITERIA if st.session_state.action_type == "Research"
                else DEFAULT_DEVELOPMENT_CRITERIA
            )
            st.session_state.criteria = [Criterion(**c) for c in defaults]
            st.rerun()

    st.markdown(theme.section("Pass threshold", step=3), unsafe_allow_html=True)
    overall_threshold = st.number_input(
        "Overall weighted minimum", min_value=0.0, max_value=200.0,
        value=OVERALL_THRESHOLD_DEFAULT, step=0.5,
        label_visibility="collapsed",
    )
    _max_total = sum(c.weight * c.max_score for c in st.session_state.criteria)
    st.caption(
        f"{overall_threshold:g} of a possible {_max_total:g}"
        + (f" · {overall_threshold / _max_total * 100:.0f}% of maximum" if _max_total else "")
    )

    st.divider()
    _chat_options, _embed_options = _list_models()
    if st.session_state.selected_chat_model not in _chat_options:
        _chat_options = [st.session_state.selected_chat_model] + _chat_options
    if st.session_state.selected_embed_model not in _embed_options:
        _embed_options = [st.session_state.selected_embed_model] + _embed_options
    st.markdown('<span class="edf-model-lock-marker"></span>', unsafe_allow_html=True)
    _model_cols = st.columns(2)
    with _model_cols[0]:
        st.session_state.selected_chat_model = st.selectbox(
            "Chat model", _chat_options,
            index=_chat_options.index(st.session_state.selected_chat_model),
            disabled=True,
        )
    with _model_cols[1]:
        st.session_state.selected_embed_model = st.selectbox(
            "Embedding model", _embed_options,
            index=_embed_options.index(st.session_state.selected_embed_model),
            disabled=True,
        )

    if st.session_state.my_job_ids:
        n_mine = len(st.session_state.my_job_ids)
        st.info(
            f"{n_mine} submission{'s' if n_mine != 1 else ''} of yours "
            "still in flight. Results appear automatically as each finishes.",
            icon="⏳",
        )

    if is_admin:
        st.divider()
        if len(db.list_runs()):
            if st.button("Wipe history", type="primary", width="stretch"):
                wipe_db_dialog()


# ---------------------------------------------------------------------------
# Result panel — its own auto-refreshing fragment (not a forced full-app
# rerun) so it appears the moment your own job finishes, without disturbing
# anything you might be typing elsewhere on the page.
# ---------------------------------------------------------------------------

@st.fragment(run_every="2s")
def _render_result_panel() -> None:
    result = st.session_state.result
    if not result:
        return

    st.divider()
    head = st.columns([6, 1], vertical_alignment="center")
    with head[0]:
        st.markdown(
            theme.section(
                "Evaluation result",
                f"{result.get('proposal_acronym') or '—'} · "
                f"{result.get('call_id') or '—'} · "
                f"{_fmt_when(result.get('generated_at'))}",
            ),
            unsafe_allow_html=True,
        )
    with head[1]:
        st.markdown('<span class="edf-ghost-marker"></span>', unsafe_allow_html=True)
        if st.button("Dismiss", key="dismiss_result", width="stretch",
                     help="Hide this panel — the run stays in History"):
            st.session_state.result = None
            st.rerun()

    render_run(result, key="live", exports=True, deletable=False)


# ---------------------------------------------------------------------------
# History + insights data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _criterion_stats(call_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    """Mean share-of-max per criterion name across the given runs.

    Cached on the tuple of call IDs: the full result JSON has to be loaded
    per run to reach the criterion breakdown, which is too heavy to redo on
    every 3-second fragment refresh.
    """
    totals: dict[str, list[float]] = {}
    for cid in call_ids:
        run = db.get_run(cid)
        if not run:
            continue
        for c in run.get("criteria") or []:
            name = str(c.get("name") or "").strip()
            max_score = float(c.get("max_score") or 5.0) or 5.0
            if not name:
                continue
            totals.setdefault(name, []).append(
                max(0.0, min(1.0, float(c.get("score") or 0.0) / max_score))
            )
    stats = [
        {"name": name, "mean": sum(vals) / len(vals), "n": len(vals)}
        for name, vals in totals.items() if vals
    ]
    stats.sort(key=lambda s: s["mean"])
    return stats


def _portfolio_tiles(runs: list[dict[str, Any]]) -> str:
    total = len(runs)
    fracs, passes = [], 0
    for r in runs:
        max_w = float(r.get("max_weighted_total") or 0.0)
        if max_w:
            fracs.append(float(r.get("weighted_total") or 0.0) / max_w)
        thr = r.get("overall_threshold")
        if thr is None or float(r.get("weighted_total") or 0.0) >= float(thr):
            passes += 1
    mean = (sum(fracs) / len(fracs) * 100) if fracs else 0.0
    best = (max(fracs) * 100) if fracs else 0.0
    latest = _fmt_when(runs[0].get("generated_at")) if runs else "—"
    return theme.tiles(
        theme.tile("Runs stored", f"{total}", f"latest {latest}", "navy"),
        theme.tile("Above threshold", f"{passes}<small> / {total}</small>",
                   f"{(passes / total * 100) if total else 0:.0f}% pass rate", "accent"),
        theme.tile("Mean score", f"{mean:.0f}<small>%</small>", "share of maximum"),
        theme.tile("Best score", f"{best:.0f}<small>%</small>", "share of maximum"),
    )


@st.fragment(run_every="3s")
def history_tab() -> None:
    all_runs = db.list_runs()
    if not all_runs:
        st.markdown(
            theme.empty_state(
                "No evaluations stored yet",
                "Queue a proposal on the Evaluate tab — completed runs from full "
                "document uploads are saved here automatically.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(theme.section("Portfolio"), unsafe_allow_html=True)
    st.markdown(_portfolio_tiles(all_runs), unsafe_allow_html=True)

    st.markdown(theme.section("Stored evaluations"), unsafe_allow_html=True)
    filt = st.columns([3, 1.4], vertical_alignment="bottom")
    with filt[0]:
        query = st.text_input(
            "Search", placeholder="Filter by acronym, Call ID, band or action type…",
            label_visibility="collapsed",
        ).strip()
    with filt[1]:
        bands = ["All outcomes"] + sorted(
            {str(r.get("predicted_band") or "—") for r in all_runs}
        )
        band_pick = st.selectbox("Band", bands, label_visibility="collapsed")

    runs = all_runs
    if query:
        q = query.lower()
        runs = [
            r for r in runs
            if q in (r.get("call_id") or "").lower()
            or q in (r.get("proposal_acronym") or "").lower()
            or q in (r.get("predicted_band") or "").lower()
            or q in (r.get("action_type") or "").lower()
        ]
    if band_pick != "All outcomes":
        runs = [r for r in runs if str(r.get("predicted_band") or "—") == band_pick]

    st.caption(f"Showing {len(runs)} of {len(all_runs)} run(s).")
    if not runs:
        st.markdown(
            theme.empty_state("No matches", "Try a shorter search term or clear the band filter."),
            unsafe_allow_html=True,
        )
        return

    widths = [2.4, 1.5, 2.2, 1.9, 0.9]
    head = st.columns(widths)
    for col, label in zip(
        head, ["Proposal", "Evaluated", "Predicted band", "Weighted score", ""]
    ):
        col.markdown(f'<div class="edf-hhead">{label}</div>', unsafe_allow_html=True)

    for r in runs:
        weighted, max_w, threshold, passed, frac = _verdict(r)
        band = str(r.get("predicted_band") or "—")
        cols = st.columns(widths, vertical_alignment="center")
        with cols[0]:
            st.markdown(
                '<span class="edf-hrow-marker"></span>'
                f'<div class="edf-acr">{_escape(r.get("proposal_acronym") or "—")}</div>'
                f'<span class="edf-cid">{_escape(r.get("call_id") or "—")}</span>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f'<div class="edf-date">{_escape(_fmt_when(r.get("generated_at"), False))}</div>'
                f'<div class="edf-crit-meta">{_escape(r.get("action_type") or "—")}</div>',
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                theme.chip(
                    band, dot=True,
                    variant={"Likely funded": "pass",
                             "Likely rejected": "fail"}.get(band, "accent"),
                ),
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                f'<div class="edf-num">{weighted:g}<small> / {max_w:g}</small>'
                f'&nbsp;&nbsp;<small>{frac * 100:.0f}%</small></div>'
                + theme.meter(frac, theme.score_color(frac)),
                unsafe_allow_html=True,
            )
        with cols[4]:
            if st.button("Open", key=f"hist_view_{r['call_id']}", width="stretch"):
                st.session_state.pending_dialog = "detail"
                st.session_state.detail_call_id = r["call_id"]
                st.rerun()


def insights_tab() -> None:
    all_runs = db.list_runs()
    if len(all_runs) < 1:
        st.markdown(
            theme.empty_state(
                "Nothing to analyse yet",
                "Insights appear once evaluations have been stored in History.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(theme.section("Portfolio at a glance"), unsafe_allow_html=True)
    st.markdown(_portfolio_tiles(all_runs), unsafe_allow_html=True)

    trend = charts.history_trend(all_runs)
    mix = charts.band_mix(all_runs)
    if trend is not None or mix is not None:
        st.markdown(
            theme.section("Outcomes over time", "oldest to newest"), unsafe_allow_html=True
        )
        cols = st.columns([1.6, 1])
        if trend is not None:
            with cols[0]:
                st.caption("Weighted score as a share of the maximum")
                st.altair_chart(trend, width="stretch")
        else:
            cols[0].caption("At least two stored runs are needed for a trend line.")
        if mix is not None:
            with cols[1]:
                st.caption("Predicted outcome band")
                st.altair_chart(mix, width="stretch")

    stats = _criterion_stats(tuple(r["call_id"] for r in all_runs))
    spread = charts.score_spread(all_runs)
    avg = charts.criterion_averages(stats)
    if avg is not None or spread is not None:
        st.markdown(
            theme.section("Where proposals lose marks", "weakest criteria first"),
            unsafe_allow_html=True,
        )
        cols = st.columns([1.6, 1])
        if avg is not None:
            with cols[0]:
                st.caption("Mean score as a share of maximum, per criterion")
                st.altair_chart(avg, width="stretch")
        if spread is not None:
            with cols[1]:
                st.caption("Distribution of overall scores")
                st.altair_chart(spread, width="stretch")
        elif avg is not None:
            cols[1].caption("At least four stored runs are needed for a distribution.")


# ---------------------------------------------------------------------------
# MAIN AREA — always visible. Evaluations run in a background worker, so the
# UI stays interactive: you can browse history, edit criteria, queue another
# proposal, all while one is being processed.
# ---------------------------------------------------------------------------

tab_evaluate, tab_history, tab_insights = st.tabs(["Evaluate", "History", "Insights"])

with tab_evaluate:
    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown(
            theme.section("Proposal input", "DOCX recommended · PDF, TXT, MD accepted", step=1),
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload the full Part B",
            type=["docx", "pdf", "txt", "md"],
            accept_multiple_files=False,
            key=f"uploaded_{st.session_state.upload_nonce}",
            label_visibility="collapsed",
        )
        with st.expander("…or paste sections directly (not saved to history)"):
            pasted = st.text_area(
                "Pasted text",
                placeholder="Paste a section of Part B here…",
                height=160,
                key=f"pasted_{st.session_state.upload_nonce}",
                label_visibility="collapsed",
            )

        # ── Stash uploaded bytes; the worker handles extraction/embedding ──
        has_content = False

        if uploaded and not pasted.strip():
            new_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.embedded_key != new_key:
                st.session_state.embedded_key = new_key
                st.session_state.proposal_bytes = uploaded.getvalue()
                st.session_state.proposal_suffix = (
                    ("." + uploaded.name.rsplit(".", 1)[-1]).lower()
                    if "." in uploaded.name else ""
                )
            if st.session_state.proposal_bytes:
                size_kb = len(st.session_state.proposal_bytes) / 1024
                st.info(
                    f"**{uploaded.name}** · {size_kb:.1f} KB ready to queue. Text "
                    "extraction and embedding run on the worker after you queue it.",
                    icon="📎",
                )
                has_content = True
        elif pasted.strip():
            st.info(
                f"{len(pasted.split()):,} words of pasted text — this run will "
                "**not** be saved to history.",
                icon="📝",
            )
            has_content = True

    with right:
        st.markdown(theme.section("Run summary", step=2), unsafe_allow_html=True)
        total_criteria = len(st.session_state.criteria)
        max_total = sum(c.weight * c.max_score for c in st.session_state.criteria)
        threshold_ratio = overall_threshold / max_total if max_total else 0.0
        st.markdown(
            theme.tiles(
                theme.tile("Proposal", _escape(st.session_state.proposal_acronym or "—"),
                           _escape(st.session_state.call_id or "no Call ID set"),
                           "accent" if st.session_state.proposal_acronym else ""),
                theme.tile("Action type", _escape(st.session_state.action_type),
                           f"{total_criteria} criteria", "navy"),
                theme.tile("Maximum score", f"{max_total:g}",
                           f"{sum(c.weight for c in st.session_state.criteria):g} total weight"),
                theme.tile("Threshold", f"{overall_threshold:g}",
                           f"{threshold_ratio * 100:.0f}% of maximum"),
            ),
            unsafe_allow_html=True,
        )
        if st.session_state.call_context.strip():
            st.caption(
                f"Call topic context supplied ({len(st.session_state.call_context.split()):,} words) — "
                "technical fit will be judged against it."
            )
        else:
            st.caption(
                "No call topic context supplied — only the proposal itself will be "
                "assessed. Add it in the sidebar for a fit judgement."
            )

        st.write("")
        cta = st.columns([1])
        with cta[0]:
            st.markdown('<span class="edf-cta-marker"></span>', unsafe_allow_html=True)
            run_clicked = st.button(
                "Queue evaluation",
                type="secondary",
                disabled=not has_content,
                width="stretch",
                key="run_eval_btn",
            )
        if not has_content:
            st.caption("Upload a document or paste text to enable queueing.")

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
                st.error(e, icon="⚠️")
        else:
            is_full_proposal = uploaded is not None and not pasted.strip()
            proposal_bytes_for_job = (
                st.session_state.proposal_bytes if is_full_proposal else None
            )
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
                    call_context=st.session_state.call_context.strip() or None,
                )
                st.session_state.my_job_ids = list(st.session_state.my_job_ids) + [job_id]
                st.toast(
                    f"Queued '{st.session_state.proposal_acronym.strip()}' "
                    + ("(pasted text — not saved to history)" if not is_full_proposal
                       else "(saved to history when done)"),
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

    _render_result_panel()

with tab_history:
    history_tab()

with tab_insights:
    insights_tab()


# ---------------------------------------------------------------------------
# Queue status — bottom of page, auto-refreshing.
# ---------------------------------------------------------------------------

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
_queue_bar()
