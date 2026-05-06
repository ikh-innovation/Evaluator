"""One-shot: parse any *ESR*.pdf in input_docs/ and index it into Chroma.

Run once before the first Streamlit launch to avoid the indexing delay
on the first evaluation:

    python -m scripts.build_kb
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import kb  # noqa: E402


def main() -> None:
    added = kb.build_kb_if_empty()
    total = kb.kb_size()
    if added:
        print(f"Indexed {added} new criterion records. KB size: {total}.")
    elif total:
        print(f"KB already populated ({total} records). Nothing to do.")
    else:
        print(
            "No ESR PDFs found in input_docs/ (looked for *ESR*.pdf). "
            "Drop one in and re-run."
        )


if __name__ == "__main__":
    main()
