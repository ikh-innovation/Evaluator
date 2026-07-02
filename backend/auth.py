"""Minimal shared-password login: username + a session id, nothing more.

Sessions live in the same SQLite file as jobs/runs (reuses jobs._connect(),
which is WAL-enabled and already the shared connection helper for that file
— avoids a third near-duplicate connection helper in this module).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import jobs

_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def create_session(username: str) -> str:
    session_id = uuid.uuid4().hex
    with jobs._connect() as conn:
        conn.executescript(_SESSIONS_SCHEMA)
        conn.execute(
            "INSERT INTO sessions (session_id, username, created_at) VALUES (?, ?, ?)",
            (session_id, username, datetime.now(timezone.utc).isoformat()),
        )
    return session_id


def get_username(session_id: str | None) -> str | None:
    if not session_id:
        return None
    with jobs._connect() as conn:
        conn.executescript(_SESSIONS_SCHEMA)
        row = conn.execute(
            "SELECT username FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return row["username"] if row else None


def peer_url(host_header: str | None, port: int) -> str:
    """Same hostname the browser used to reach this app, different port.

    Lets the login page link to the main app (and vice versa) correctly
    whether someone's on `localhost` or a LAN IP from another machine.
    """
    hostname = (host_header or "localhost").split(":")[0]
    return f"http://{hostname}:{port}"
