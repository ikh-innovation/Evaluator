"""Background worker that drains the shared job queue.

Started lazily on the first Streamlit rerun via start_worker_once(). One
daemon thread processes jobs FIFO. Each job's raw payload (uploaded bytes
or pasted text) is loaded from the pickle written by jobs.enqueue and fed
into evaluate_proposal(), which extracts text, chunks, embeds, and runs
the LLM evaluation as one sequential pipeline.
"""

from __future__ import annotations

import threading
import time
import traceback

import ollama

from . import db, jobs
from .config import CHAT_MODEL, EMBED_MODEL, OLLAMA_BASE_URL
from .pipeline import evaluate_proposal


_thread: threading.Thread | None = None
_thread_lock = threading.Lock()
_POLL_INTERVAL_S = 1.0

# Tracks whichever chat model the most recent job actually used, so the
# idle-unload below evicts the right one — jobs can override the config
# default via a per-job model selection in the UI.
_last_chat_model: str = CHAT_MODEL


class _JobCancelled(Exception):
    """Raised inside the progress callback to abort a running job."""


def _process_one(job: dict) -> None:
    global _last_chat_model
    job_id: str = job["job_id"]
    chat_model = job.get("chat_model") or CHAT_MODEL
    embed_model = job.get("embed_model") or EMBED_MODEL
    _last_chat_model = chat_model

    def _progress(message: str, fraction: float) -> None:
        try:
            cancel_requested = jobs.update_progress_and_check_cancel(
                job_id, message, fraction
            )
        except Exception:
            cancel_requested = False
        if cancel_requested:
            raise _JobCancelled()

    try:
        payload = jobs.load_payload(job_id)
    except Exception as exc:
        jobs.mark_failed(job_id, f"Could not load job payload: {exc}")
        return

    try:
        result = evaluate_proposal(
            proposal_text=payload.get("pasted_text") or "",
            criteria=job["criteria"],
            action_type=job["action_type"],
            call_id=job["call_id"],
            overall_threshold=job["overall_threshold"],
            proposal_acronym=job["proposal_acronym"],
            proposal_bytes=payload.get("proposal_bytes"),
            proposal_suffix=payload.get("proposal_suffix"),
            chunk_index=None,
            progress_cb=_progress,
            persist=bool(job["persist"]),
            chat_model=chat_model,
            embed_model=embed_model,
            call_context=payload.get("call_context") or "",
        )
    except _JobCancelled:
        jobs.mark_failed(job_id, "Cancelled by user.")
        return
    except Exception as exc:
        jobs.mark_failed(job_id, f"{exc}\n{traceback.format_exc()[:600]}")
        return

    # evaluate_proposal saves when persist=True. If persist=False (paste-only)
    # the result still needs to be reachable so the user can view it once;
    # store it under the call_id so the dashboard can pick it up.
    if not bool(job["persist"]):
        try:
            db.save_run(result)
        except Exception:
            pass

    jobs.mark_done(job_id, result.get("call_id", job["call_id"]))


def _unload_chat_model() -> bool:
    """Best-effort: free the chat model's VRAM once the queue is empty.

    Chat calls pin the model with keep_alive=-1 so it survives gaps between
    LLM calls within/between jobs; this explicitly evicts it the moment
    there's nothing left queued, rather than leaving it loaded forever.
    Returns whether the request succeeded, so callers can retry on failure.
    """
    try:
        ollama.Client(host=OLLAMA_BASE_URL).generate(model=_last_chat_model, keep_alive=0)
        return True
    except Exception:
        return False


def _loop() -> None:
    jobs.recover_orphans()
    idle = False
    while True:
        try:
            job = jobs.claim_next()
        except Exception:
            job = None
        if job is None:
            if not idle:
                idle = _unload_chat_model()
            time.sleep(_POLL_INTERVAL_S)
            continue
        idle = False
        try:
            _process_one(job)
        except Exception as exc:
            try:
                jobs.mark_failed(job["job_id"], f"Worker error: {exc}")
            except Exception:
                pass


def start_worker_once() -> None:
    """Idempotent. Safe to call on every Streamlit rerun."""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return
        t = threading.Thread(target=_loop, name="edf-eval-worker", daemon=True)
        t.start()
        _thread = t


def is_alive() -> bool:
    return _thread is not None and _thread.is_alive()
