# EDF Proposal Evaluator

Internal pre-screening tool for EDF (European Defence Fund) Part B proposals.
Runs entirely on a local Ollama instance — no external APIs, no cloud.

> **Not an official evaluator.** Output is internal pre-screening only; final judgement always rests with human evaluators.

---

## What it does

1. Upload a proposal (DOCX, PDF, or TXT) — it is chunked and embedded locally via Ollama (`nomic-embed-text`).
2. Click **Run evaluation** — the LLM (`gemma3:4b`) evaluates each criterion one at a time, displaying results as they finish.
3. A weighted score, ESR-style summary, and predicted outcome band are generated.
4. Results are saved to a local SQLite database; export as CSV or JSON.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with:

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

---

## Setup

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

Open `http://localhost:8501`.

---

## Project layout

```
backend/
  config.py       — env-var driven settings (models, paths, timeouts)
  ingestion.py    — DOCX / PDF / TXT text extraction + section detection
  chunking.py     — RecursiveCharacterTextSplitter wrapping
  pipeline.py     — per-run embedding index, excerpt retrieval, eval orchestration
  evaluator.py    — LLM prompts, criterion scoring, summary, band prediction
  criteria_io.py  — XLSX import / export for criteria
  db.py           — SQLite run history (save, list, get, delete, wipe)
frontend/
  app.py          — Streamlit UI (header, evaluate / history / insights tabs, queue bar)
  theme.py        — design system: brand palette, stylesheet, small HTML components
  charts.py       — Altair chart builders (score profile, contribution, trends)
  pdf_report.py   — print-quality PDF report (ReportLab, vector charts)
  ikh_logo.png    — brand mark; cropped at runtime for the header and the PDF
.streamlit/
  config.toml     — native Streamlit theme tokens (must match theme.py)
data/
  runs.sqlite     — run history (auto-created)
input_docs/       — reference documents (not required for evaluation)
requirements.txt
```

---

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `EDF_CHAT_MODEL` | `gemma3:4b` | LLM for evaluation |
| `EDF_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EDF_REQUEST_TIMEOUT` | `600` | LLM request timeout (seconds) |
| `EDF_MAX_WORKERS` | `2` | Parallel criterion evaluation threads |

---

## Notes

- The LLM is kept loaded in memory (`keep_alive=-1`) for the duration of the session.
- Each proposal is embedded fresh per run using a numpy in-memory index — no persistent vector store.
- Criteria and weights are fully configurable in the sidebar or via XLSX upload/download.
- Pasted-text runs are not saved to history.

### UI conventions

- Palette is taken from the logo: navy `#202039`, amber `#F08003`, white. Amber marks
  the single primary action on a screen; red is reserved for destructive actions
  (Wipe history, Delete run); green/red appear only on a pass/fail verdict.
- `frontend/theme.py` and `.streamlit/config.toml` both hold colour values — change
  them together.
- Results, history detail and the PDF all read from one renderer / one template, so
  the three can't drift apart.
