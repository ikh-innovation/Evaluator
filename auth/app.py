from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
for _path in (str(ROOT), str(FRONTEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from backend import auth  # noqa: E402

import theme  # noqa: E402  (shared design system, lives with the frontend)

FRONTEND_PORT = 8501

st.set_page_config(
    page_title="EDF Evaluator — Sign in",
    page_icon=str(theme.LOGO_PATH) if theme.LOGO_PATH.exists() else ":closed_lock_with_key:",
    layout="centered",
)

theme.inject_css()

try:
    _host_header = st.context.headers.get("host")
except Exception:
    _host_header = None

# Centred, narrow sign-in card.
_left, _mid, _right = st.columns([1, 2.1, 1])

with _mid:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    st.image(theme.logo_bytes(with_tagline=True), width=150)
    st.markdown(
        '<div class="edf-apptitle" style="margin-top:1.4rem">EDF Proposal Evaluator</div>'
        '<div class="edf-appsub">Internal pre-screening of European Defence Fund '
        "Part B proposals · <b>not an official evaluation</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="height:2px;background:' + theme.AMBER + ';margin:.9rem 0 1.2rem 0;'
        'width:64px;border-radius:1px"></div>',
        unsafe_allow_html=True,
    )

    with st.form("login", border=False):
        username = st.text_input(
            "Your name", placeholder="e.g. Jane Doe",
            help="Used to label your submissions in the queue and history.",
        )
        st.markdown('<span class="edf-cta-marker"></span>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Sign in", width="stretch")

    if submitted:
        if not username.strip():
            st.error("Enter your name to continue.", icon="⚠️")
        else:
            session_id = auth.create_session(username.strip())
            app_url = f"{auth.peer_url(_host_header, FRONTEND_PORT)}/?sid={session_id}"
            st.success(f"Signed in as **{username.strip()}**.")
            st.markdown(
                f'<a href="{app_url}" target="_self" style="display:block;'
                f"margin-top:.6rem;padding:.6rem 1rem;border-radius:4px;"
                f"background:{theme.NAVY};color:#fff;text-decoration:none;"
                'font-weight:600;text-align:center;font-size:.9rem;">'
                "Continue to the evaluator →</a>",
                unsafe_allow_html=True,
            )
            st.caption(f"If the button does not work, open: {app_url}")

    st.write("")
    st.caption(
        "Runs entirely on local infrastructure — no proposal text leaves this network."
    )
