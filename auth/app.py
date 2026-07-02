from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import auth  # noqa: E402

FRONTEND_PORT = 8501


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value


AUTH_PASSWORD = _require_env("AUTH_PASSWORD")

st.set_page_config(page_title="EDF Evaluator — Login", layout="centered")

try:
    _host_header = st.context.headers.get("host")
except Exception:
    _host_header = None

st.title("EDF Proposal Evaluator")
st.subheader("Log in")

with st.form("login"):
    username = st.text_input("Username", placeholder="e.g. Jane Doe")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Log in")

if submitted:
    if not username.strip():
        st.error("Enter a username.")
    elif password != AUTH_PASSWORD:
        st.error("Invalid password.")
    else:
        session_id = auth.create_session(username.strip())
        app_url = f"{auth.peer_url(_host_header, FRONTEND_PORT)}/?sid={session_id}"
        st.success(f"Logged in as **{username.strip()}**.")
        st.markdown(
            f'<a href="{app_url}" target="_self" '
            'style="display:inline-block;padding:.5rem 1rem;border:1px solid #000;'
            'background:#000;color:#fff;text-decoration:none;font-weight:600;">'
            "Continue to app →</a>",
            unsafe_allow_html=True,
        )
        st.caption(f"If the button doesn't work, open this link: {app_url}")
