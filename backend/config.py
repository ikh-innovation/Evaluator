from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RUNS_DB_PATH: Path = DATA_DIR / "runs.sqlite"
INPUT_DOCS_DIR: Path = ROOT / "input_docs"


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

PROPOSAL_CHUNK_TOP_K: int = int(os.environ.get("EDF_PROPOSAL_TOP_K", "4"))
CHUNK_SIZE_CHARS: int = int(os.environ.get("EDF_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP_CHARS: int = int(os.environ.get("EDF_CHUNK_OVERLAP", "150"))
MAX_PROPOSAL_CHARS_INLINE: int = int(os.environ.get("EDF_MAX_INLINE_CHARS", "5000"))
MAX_EMBED_CHARS: int = int(os.environ.get("EDF_MAX_EMBED_CHARS", "2500"))

REQUEST_TIMEOUT_S: float = float(os.environ.get("EDF_REQUEST_TIMEOUT", "600"))

MAX_WORKERS: int = int(os.environ.get("EDF_MAX_WORKERS", "2"))

# Number of model layers to offload to GPU. 99 = use all available GPU layers.
# Set EDF_NUM_GPU=0 to force CPU-only.
NUM_GPU: int = int(os.environ.get("EDF_NUM_GPU", "99"))
# NUM_GPU: int = 0
# GPU device index to use (0 = first GPU, e.g. RTX 5070).
MAIN_GPU: int = int(os.environ.get("EDF_MAIN_GPU", "0"))

# Ollama context window (tokens). Actual prompts here are small (~2-2.5k
# tokens worst case); capping this avoids Ollama's auto-sizing (observed
# defaulting to 262144) wasting VRAM/time on an oversized KV-cache.
NUM_CTX: int = int(os.environ.get("EDF_NUM_CTX", "8192"))