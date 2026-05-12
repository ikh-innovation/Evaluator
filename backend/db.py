from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import RUNS_DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    call_id            TEXT PRIMARY KEY,
    proposal_acronym   TEXT NOT NULL,
    action_type        TEXT NOT NULL,
    weighted_total     REAL NOT NULL,
    max_weighted_total REAL NOT NULL,
    overall_threshold  REAL,
    predicted_band     TEXT,
    overall_confidence TEXT,
    generated_at       TEXT NOT NULL,
    result_json        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_generated_at ON runs(generated_at);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    RUNS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RUNS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            conn.executescript(_SCHEMA)
            yield conn
    finally:
        conn.close()


def save_run(result: dict[str, Any]) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
              call_id, proposal_acronym, action_type,
              weighted_total, max_weighted_total, overall_threshold,
              predicted_band, overall_confidence, generated_at, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["call_id"],
                result.get("proposal_acronym", ""),
                result.get("action_type", ""),
                float(result.get("weighted_total", 0.0)),
                float(result.get("max_weighted_total", 0.0)),
                (
                    float(result["overall_threshold"])
                    if result.get("overall_threshold") is not None
                    else None
                ),
                result.get("predicted_band", ""),
                result.get("overall_confidence", ""),
                result.get("generated_at", ""),
                json.dumps(result, ensure_ascii=False),
            ),
        )


def list_runs() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT call_id, proposal_acronym, action_type,
                   weighted_total, max_weighted_total, overall_threshold,
                   predicted_band, overall_confidence, generated_at
              FROM runs
             ORDER BY generated_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(call_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM runs WHERE call_id = ?", (call_id,)
        ).fetchone()
    if row is None:
        return None
    return json.loads(row["result_json"])


def delete_run(call_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM runs WHERE call_id = ?", (call_id,))


def wipe_all() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM runs")
        return cur.rowcount or 0
