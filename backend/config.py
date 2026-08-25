from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RUNS_DB_PATH: Path = DATA_DIR / "runs.sqlite"
INPUT_DOCS_DIR: Path = ROOT / "input_docs"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load repo-root .env into the environment on every launch path.

    Under `docker compose` the values are already injected as container env, so
    each key is applied with setdefault — real environment always wins and this
    never clobbers it. When the app is launched directly (e.g. `streamlit run`),
    this is what makes the process pick up the .env. No external dependency:
    a minimal KEY=VALUE parser (ignores blanks, `#` comments, optional `export`
    prefix, and surrounding quotes).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {name!r}. "
            f"Set it in your .env file (see .env.example)."
        )
    return value


OLLAMA_BASE_URL: str = _require_env("OLLAMA_BASE_URL")

CHAT_MODEL: str = _require_env("EDF_CHAT_MODEL")
EMBED_MODEL: str = _require_env("EDF_EMBED_MODEL")

PROPOSAL_CHUNK_TOP_K: int = int(os.environ.get("EDF_PROPOSAL_TOP_K", "16"))
CHUNK_SIZE_CHARS: int = int(os.environ.get("EDF_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP_CHARS: int = int(os.environ.get("EDF_CHUNK_OVERLAP", "150"))
MAX_PROPOSAL_CHARS_INLINE: int = int(os.environ.get("EDF_MAX_INLINE_CHARS", "24000"))
MAX_EMBED_CHARS: int = int(os.environ.get("EDF_MAX_EMBED_CHARS", "2500"))
# Retrieval diversity: at most this many chunks per section title, so one
# semantically dominant section cannot crowd out everything else.
MAX_CHUNKS_PER_SECTION: int = int(os.environ.get("EDF_MAX_CHUNKS_PER_SECTION", "3"))
# Cap on runtime-supplied call/topic context injected into each criterion prompt.
MAX_CALL_CONTEXT_CHARS: int = int(os.environ.get("EDF_MAX_CALL_CONTEXT_CHARS", "8000"))

REQUEST_TIMEOUT_S: float = float(os.environ.get("EDF_REQUEST_TIMEOUT", "600"))

MAX_WORKERS: int = int(os.environ.get("EDF_MAX_WORKERS", "2"))

# Number of model layers to offload to GPU. 99 = use all available GPU layers.
# Set EDF_NUM_GPU=0 to force CPU-only.
NUM_GPU: int = int(os.environ.get("EDF_NUM_GPU", "99"))
# NUM_GPU: int = 0
# GPU device index to use (0 = first GPU, e.g. RTX 5070).
MAIN_GPU: int = int(os.environ.get("EDF_MAIN_GPU", "0"))

# Ollama context window (tokens). Sized for the worst-case criterion prompt:
# ~24k chars of proposal excerpts + ~8k chars of call context + instructions
# + JSON output ≈ 11k tokens. Capping this avoids Ollama's auto-sizing
# (observed defaulting to 262144) wasting VRAM/time on an oversized KV-cache.
NUM_CTX: int = int(os.environ.get("EDF_NUM_CTX", "16384"))
# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------
# Refuses to score documents that are not proposals for the call being
# evaluated. Without it the pipeline scores anything: retrieval always returns
# its top-K chunks (no similarity floor) and the judge prompt never asks
# whether the document is on-topic, so an unrelated file lands mid-scale.
#
#   "block" (default) — raise IrrelevantDocumentError, produce no scorecard.
#   "warn"            — evaluate anyway, attach the verdict to ingest_warning.
#   "off"             — skip the check entirely.
RELEVANCE_GATE_MODE: str = os.environ.get("EDF_RELEVANCE_GATE", "block").strip().lower()

# Cosine similarity between a criterion and the best-matching chunk. These are
# corroborating signals only — the absolute values depend on EDF_EMBED_MODEL,
# so the LLM verdict decides and similarity never blocks on its own except via
# the hard floor below, which applies only when the LLM check is unavailable.
RELEVANCE_MIN_SIMILARITY: float = float(
    os.environ.get("EDF_RELEVANCE_MIN_SIMILARITY", "0.35")
)
RELEVANCE_HARD_FLOOR_SIMILARITY: float = float(
    os.environ.get("EDF_RELEVANCE_HARD_FLOOR", "0.20")
)

# How much of the document's opening the relevance classifier sees.
RELEVANCE_PROBE_CHARS: int = int(os.environ.get("EDF_RELEVANCE_PROBE_CHARS", "6000"))
