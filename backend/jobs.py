"""Shared job queue for proposal evaluations.

Jobs are stored in the same SQLite file as runs. The raw proposal payload
(bytes + suffix, or pasted text) is pickled to disk per job and removed
once the job ends — the worker handles extraction, chunking, embedding,
and LLM evaluation as one sequential pipeline.

The queue is a simple FIFO with statuses: queued | running | done | failed.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DATA_DIR, RUNS_DB_PATH


JOB_PAYLOADS_DIR: Path = DATA_DIR / "job_payloads"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL,
    call_id           TEXT NOT NULL,
    proposal_acronym  TEXT NOT NULL,
    action_type       TEXT NOT NULL,
    overall_threshold REAL,
    criteria_json     TEXT NOT NULL,
    submitted_by      TEXT,
    submitted_at      TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT,
    progress_msg      TEXT,
    progress_frac     REAL,
    payload_path      TEXT,
    persist           INTEGER NOT NULL DEFAULT 1,
    error             TEXT,
    result_call_id    TEXT,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    chat_model        TEXT,
    embed_model       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_submitted_at ON jobs(submitted_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _migrate(conn: sqlite3.Connection) -> None:
    """Inline schema migrations for the jobs table."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if not cols:
        return
    # payload_path used to be index_path (different content).
    if "payload_path" not in cols:
        if "index_path" in cols:
            conn.execute("ALTER TABLE jobs RENAME COLUMN index_path TO payload_path")
        else:
            conn.execute("ALTER TABLE jobs ADD COLUMN payload_path TEXT")
    # cancel_requested added when per-job cancel buttons were introduced.
    if "cancel_requested" not in cols:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
        )
    # chat_model/embed_model added when per-job model selection was introduced.
    if "chat_model" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN chat_model TEXT")
    if "embed_model" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN embed_model TEXT")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    RUNS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RUNS_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # DELETE (rollback journal), not WAL: WAL relies on mmap'd shared-memory
    # coordination between processes, which does not reliably propagate
    # across separate container mount namespaces on Docker Desktop for
    # Windows bind mounts — writes from one container (e.g. auth) could be
    # invisible to reads from another (e.g. frontend), even though they
    # share the same host file. DELETE mode uses plain file locking, which
    # works correctly across containers; write throughput here is trivially
    # low so WAL's concurrency benefit isn't needed. Idempotent — only flips
    # the file's journaling mode once.
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
    except sqlite3.DatabaseError:
        pass
    try:
        with conn:
            conn.executescript(_SCHEMA)
            _migrate(conn)
            yield conn
    finally:
        conn.close()


def _ensure_payloads_dir() -> None:
    JOB_PAYLOADS_DIR.mkdir(parents=True, exist_ok=True)


def enqueue(
    *,
    call_id: str,
    proposal_acronym: str,
    action_type: str,
    overall_threshold: float | None,
    criteria: list[dict[str, Any]],
    proposal_bytes: bytes | None,
    proposal_suffix: str | None,
    pasted_text: str | None,
    persist: bool,
    submitted_by: str | None,
    chat_model: str | None = None,
    embed_model: str | None = None,
    call_context: str | None = None,
) -> str:
    """Pickle the raw payload to disk and insert a queued job. Returns job_id.

    The worker reads this payload, then extracts/chunks/embeds/evaluates as
    one sequential pipeline. Persist=True only makes sense for uploaded files,
    since pasted text is treated as not-saved-to-history.
    """
    if not proposal_bytes and not pasted_text:
        raise ValueError("Either proposal_bytes or pasted_text must be provided.")

    _ensure_payloads_dir()
    job_id = uuid.uuid4().hex
    payload_path = JOB_PAYLOADS_DIR / f"{job_id}.pkl"
    payload = {
        "proposal_bytes": proposal_bytes,
        "proposal_suffix": proposal_suffix,
        "pasted_text": pasted_text,
        "call_context": call_context,
    }
    with payload_path.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
              job_id, status, call_id, proposal_acronym, action_type,
              overall_threshold, criteria_json, submitted_by, submitted_at,
              payload_path, persist, progress_msg, progress_frac,
              chat_model, embed_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "queued",
                call_id,
                proposal_acronym,
                action_type,
                overall_threshold,
                json.dumps(criteria, ensure_ascii=False),
                submitted_by,
                _utcnow_iso(),
                str(payload_path),
                1 if persist else 0,
                "Waiting in queue…",
                0.0,
                chat_model,
                embed_model,
            ),
        )
    return job_id


def claim_next() -> dict[str, Any] | None:
    """Atomically pick the oldest queued job and mark it running.

    Returns the full job row (with criteria parsed) or None if nothing pending.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' "
            "ORDER BY submitted_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, "
            "progress_msg = ?, progress_frac = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (_utcnow_iso(), "Starting…", 0.0, row["job_id"]),
        )
        if cur.rowcount == 0:
            return None
        out = dict(row)
    out["status"] = "running"
    out["criteria"] = json.loads(out.pop("criteria_json"))
    return out


def update_progress(job_id: str, message: str, fraction: float) -> None:
    fraction = max(0.0, min(1.0, float(fraction)))
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET progress_msg = ?, progress_frac = ? WHERE job_id = ?",
            (message, fraction, job_id),
        )


def update_progress_and_check_cancel(job_id: str, message: str, fraction: float) -> bool:
    """Combined progress write + cancel-flag read in one connection.

    Returns True if cancellation has been requested for this job.
    """
    fraction = max(0.0, min(1.0, float(fraction)))
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET progress_msg = ?, progress_frac = ? WHERE job_id = ?",
            (message, fraction, job_id),
        )
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def _remove_payload(payload_path: str | None) -> None:
    if not payload_path:
        return
    try:
        Path(payload_path).unlink(missing_ok=True)
    except OSError:
        pass


def mark_done(job_id: str, result_call_id: str) -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_path FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.execute(
            "UPDATE jobs SET status = 'done', finished_at = ?, "
            "progress_msg = ?, progress_frac = ?, result_call_id = ? "
            "WHERE job_id = ?",
            (_utcnow_iso(), "Completed", 1.0, result_call_id, job_id),
        )
    _remove_payload(row["payload_path"] if row else None)


def mark_failed(job_id: str, error: str) -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_path FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?, "
            "progress_msg = ?, error = ? WHERE job_id = ?",
            (_utcnow_iso(), "Failed", error[:1000], job_id),
        )
    _remove_payload(row["payload_path"] if row else None)


def list_active() -> list[dict[str, Any]]:
    """Return currently running + queued jobs, in execution order."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT job_id, status, call_id, proposal_acronym, action_type,
                   submitted_by, submitted_at, started_at,
                   progress_msg, progress_frac, cancel_requested
              FROM jobs
             WHERE status IN ('running', 'queued')
             ORDER BY CASE status WHEN 'running' THEN 0 ELSE 1 END,
                      submitted_at ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT job_id, status, call_id, proposal_acronym, action_type,
                   submitted_by, submitted_at, started_at, finished_at,
                   progress_msg, progress_frac, error, result_call_id
              FROM jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def load_payload(job_id: str) -> dict[str, Any]:
    """Return {proposal_bytes, proposal_suffix, pasted_text}."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_path FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    if row is None or not row["payload_path"]:
        raise FileNotFoundError(f"No payload stored for job {job_id}")
    with Path(row["payload_path"]).open("rb") as fh:
        return pickle.load(fh)


def recover_orphans() -> int:
    """Called on worker startup. Marks any 'running' jobs as failed so the
    queue doesn't get stuck if the process was killed mid-evaluation, and
    cleans up the orphan payload pickles so they don't leak disk space.

    Returns the number of jobs reset.
    """
    orphan_payloads: list[str] = []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT payload_path FROM jobs WHERE status = 'running'"
        ).fetchall()
        orphan_payloads = [r["payload_path"] for r in rows if r["payload_path"]]
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?, "
            "progress_msg = ?, error = ? WHERE status = 'running'",
            (_utcnow_iso(), "Failed", "Worker restarted before job finished."),
        )
        count = cur.rowcount or 0
    for p in orphan_payloads:
        _remove_payload(p)
    return count


def cancel(job_id: str) -> str | None:
    """Request cancellation of a job. Returns the resulting status, or None
    if the job is unknown.

    - queued  → marked 'failed' immediately ('Cancelled by user.')
    - running → cancel_requested flag set; the worker bails at the next
                progress checkpoint and marks the job 'failed' itself.
    - done/failed → no-op (returns the existing status).

    The transition is done as an atomic conditional UPDATE so a worker
    that claims the job in between our read and write can't race us.
    """
    payload_to_remove: str | None = None
    new_status: str | None = None
    with _connect() as conn:
        # Always set the cancel flag (covers both running and the race window
        # where the worker claims a job between our read and our UPDATE).
        conn.execute(
            "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?",
            (job_id,),
        )

        # Then try the queued→failed transition atomically. rowcount tells
        # us whether the job actually was still queued at that moment.
        cur = conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?, "
            "progress_msg = ?, error = ? "
            "WHERE job_id = ? AND status = 'queued'",
            (_utcnow_iso(), "Cancelled", "Cancelled by user.", job_id),
        )
        if cur.rowcount > 0:
            row = conn.execute(
                "SELECT payload_path FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            payload_to_remove = row["payload_path"] if row else None
            new_status = "failed"
        else:
            row = conn.execute(
                "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                conn.execute(
                    "UPDATE jobs SET progress_msg = ? WHERE job_id = ?",
                    ("Cancelling…", job_id),
                )
            new_status = row["status"]

    if payload_to_remove:
        _remove_payload(payload_to_remove)
    return new_status


def is_cancel_requested(job_id: str) -> bool:
    """Read the cancel flag — called by the worker at progress checkpoints."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])
