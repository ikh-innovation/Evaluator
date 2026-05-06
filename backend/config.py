"""Runtime configuration for the EDF Evaluator backend.

All values can be overridden via environment variables so the same code runs
on a developer laptop and on the GPU server without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
KB_DIR: Path = DATA_DIR / "kb"
RUNS_DB_PATH: Path = DATA_DIR / "runs.sqlite"
INPUT_DOCS_DIR: Path = ROOT / "input_docs"

OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat model used for criterion evaluations + summaries.
# Override with `EDF_CHAT_MODEL=qwen2.5:14b` etc.
CHAT_MODEL: str = os.environ.get("EDF_CHAT_MODEL", "gemma3:12b")

# Embedding model. Pull once with `ollama pull nomic-embed-text`.
EMBED_MODEL: str = os.environ.get("EDF_EMBED_MODEL", "nomic-embed-text")

# Retrieval / chunking knobs
KB_TOP_K: int = int(os.environ.get("EDF_KB_TOP_K", "4"))
PROPOSAL_CHUNK_TOP_K: int = int(os.environ.get("EDF_PROPOSAL_TOP_K", "8"))
# Token-aware proposal chunking (~3 chars/token rough estimate)
CHUNK_SIZE_CHARS: int = int(os.environ.get("EDF_CHUNK_SIZE", "1500"))
CHUNK_OVERLAP_CHARS: int = int(os.environ.get("EDF_CHUNK_OVERLAP", "200"))
# Hard cap on text we feed to the chat model per criterion. Must fit in
# the chat model's context window (gemma3:12b defaults to 8192 tokens).
MAX_PROPOSAL_CHARS_INLINE: int = int(os.environ.get("EDF_MAX_INLINE_CHARS", "12000"))
# Hard cap per single embedding call. Ollama's nomic-embed-text enforces
# ~2048 tokens; dense PDF text (tables, ToCs) can hit 1 token / 1.5 chars,
# so we stay well under that. Pipeline auto-halves further on failure.
MAX_EMBED_CHARS: int = int(os.environ.get("EDF_MAX_EMBED_CHARS", "2500"))

REQUEST_TIMEOUT_S: float = float(os.environ.get("EDF_REQUEST_TIMEOUT", "300"))
