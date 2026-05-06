# EDF Proposal Evaluator

Internal, on-prem AI assistant that pre-screens **EDF (European Defence
Fund) Part B proposals** against configurable evaluation criteria, in the
voice of an official ESR (Evaluation Summary Report). The system uses a
local LLM (Ollama) and retrieval-augmented generation (LangChain + Chroma)
over a knowledge base of past ESRs and prior runs.

> **Not an official evaluator.** Output is internal pre-screening; final
> judgement always rests with human evaluators. The tool exists to surface
> weaknesses and likely scoring concerns *before* submission, so authors
> can iterate.

---

## What it does

- Accepts a Part B proposal (DOCX / PDF / TXT) or a pasted excerpt.
- Lets you configure the call's criteria, weights, and threshold (per
  EDF call fiche §9). Defaults are loaded for Research / Development.
- Criteria can be edited in the UI **or** uploaded / downloaded as XLSX.
- For each criterion, retrieves relevant excerpts from the proposal AND
  relevant historical examples from the knowledge base, then asks the LLM
  to produce an ESR-style assessment (score 0–5 in 0.5 increments,
  strengths, weaknesses, missing evidence, recommendations, citations).
- Aggregates weighted total, predicts an outcome band ("Likely funded /
  Reserve list / Likely rejected"), generates an ESR-style summary.
- Persists every full-proposal run to a SQLite history database, and
  decomposes each run into per-criterion records that re-enter the
  retrieval pool — **the system gets richer with every run**.
- Reviewer can promote a run to "ground truth" so future evaluations
  weight it as a strong signal.

---

## Architecture (step by step)

```
                  ┌─────────────────────────────────┐
                  │        Streamlit frontend       │
                  │  upload · criteria · history    │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────┐
                  │        evaluate_proposal()      │
                  │       (backend/pipeline.py)     │
                  └────────────────┬────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
 │ Document        │      │ Per-criterion    │      │ Aggregation      │
 │ ingestion +     │      │ LLM evaluation   │      │ + summary +      │
 │ chunking        │      │ (with RAG)       │      │ band prediction  │
 │ (ingestion.py + │      │ (evaluator.py)   │      │                  │
 │  chunking.py)   │      │                  │      │                  │
 └────────┬────────┘      └─────────┬────────┘      └────────┬─────────┘
          │                         │                        │
          │  proposal chunks +      │  retrieved RAG ctx +   │  result dict
          │  embeddings cached      │  proposal excerpts     │  (JSON-serialisable)
          │  for the run            │                        │
          ▼                         ▼                        ▼
 ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
 │ Ollama (local)  │      │  Chroma KB       │      │ SQLite           │
 │ chat=gemma3:12b │      │  data/kb/        │      │ data/runs.sqlite │
 │ embed=nomic-... │      │  ESRs + prior    │      │ One row per      │
 │                 │      │  runs (mixed     │      │ Call ID          │
 │                 │      │  tiers)          │      │                  │
 └─────────────────┘      └──────────────────┘      └──────────────────┘
```

### 1. Document ingestion ([backend/ingestion.py](backend/ingestion.py))

- DOCX → `python-docx` paragraph + table extraction. Section detection
  uses **heading styles** (Heading 1/2/3) when present — much more reliable
  than regex.
- PDF → `pdfplumber` (with `pypdf` fallback). Each page becomes a
  candidate section labelled with its page number.
- TXT / MD → plain decode plus a heuristic numbered/all-caps heading
  detector.

### 2. Token-aware chunking ([backend/chunking.py](backend/chunking.py))

- LangChain `RecursiveCharacterTextSplitter` with separators
  `["\n\n", "\n", ". ", "; ", " ", ""]`, default chunk size 1500 chars,
  200 overlap.
- Each chunk inherits `title` (heading or page label) and `chunk_index`
  metadata so citations can point back to the source location.
- A second guard splits any chunk that still exceeds the embedding
  budget (`MAX_EMBED_CHARS`, default 2500) at the nearest line break.

### 3. Per-run chunk index ([backend/pipeline.py](backend/pipeline.py))

- All proposal chunks are embedded **once at the start of the run** via
  Ollama (`nomic-embed-text`) and held in memory.
- A defensive `_safe_embed_documents` helper auto-halves any chunk whose
  embedding call fails, so dense PDF text (table-of-contents dotted
  lines, ToCs) never aborts the run.
- For each criterion, the top-K (default 8) chunks are retrieved by
  cosine similarity to the criterion description. Retrieved chunks are
  re-ordered to original document position so the LLM reads them in
  narrative flow.

### 4. RAG over historical knowledge ([backend/kb.py](backend/kb.py))

The Chroma collection contains two record types:

| Type              | Source                          | Trust     | When indexed                              |
|-------------------|---------------------------------|-----------|-------------------------------------------|
| `esr_criterion`   | `input_docs/*ESR*.pdf`          | Gold      | First-time KB seeding (`build_kb_if_empty`)|
| `evaluation_unit` | Every completed run             | Advisory  | Auto-indexed at end of `evaluate_proposal`|

Each `evaluation_unit` carries `reviewer_approved: bool` metadata. The
reviewer can flip this to `True` from the run-detail popup, promoting
the unit into a higher-trust tier.

For each criterion, `kb.retrieve(...)` returns the top-K records of any
type, ranked by cosine. The evaluator prompt labels each one with its
tier so the LLM can weight them appropriately:

- `[OFFICIAL ESR]` — gold standard, real evaluator output.
- `[PRIOR RUN — REVIEWED]` — past system run promoted by a human.
- `[PRIOR RUN — system]` — past system run, not yet reviewed; advisory.

### 5. Per-criterion LLM evaluation ([backend/evaluator.py](backend/evaluator.py))

- Model: `gemma3:12b` via Ollama, `format="json"`, temperature 0.1.
- Prompt embeds the **official EDF 0–5 rubric** (call fiche §9) so the
  model calibrates against EU language.
- Anti-hallucination contract: every strength and weakness must include
  a verbatim quoted phrase from the proposal text. No quote → don't
  claim it. Confidence is rule-based on quote density.
- Output is strict JSON with `score`, `confidence`, `strengths`,
  `weaknesses`, `missing_evidence`, `evaluator_comment`,
  `recommendations`, `citations`. Robust JSON parser with fenced /
  embedded fallbacks. Score snapped to 0.5 increments, clamped to 0–5.

### 6. Aggregation, summary, band

- Weighted total = Σ score × weight. Pass/fail vs. configured threshold.
- ESR-style summary: a second LLM call that takes the per-criterion
  block and produces a 5–8 sentence editorial summary with rewrite
  priorities.
- Predicted band: a third LLM call classifies into `Likely funded /
  Reserve list / borderline / Likely rejected`, with a deterministic
  fallback if the model output drifts.

### 7. Persistence + learning loop ([backend/db.py](backend/db.py) + kb)

- The full result dict (criteria + scores + comments + JSON) is written
  to `data/runs.sqlite`, keyed on Call ID with `INSERT OR REPLACE`. This
  powers the History popup.
- The same result is **decomposed into per-criterion records** and added
  to the Chroma KB. From the next run forward, retrieval will surface
  these alongside the official ESR records.
- Pasted-text runs (no full file) are NOT persisted — they're treated as
  exploratory probes.
- Deleting a run from the History popup also removes its KB records, so
  a wrong run can't poison the retrieval pool.

### 8. Frontend ([frontend/app.py](frontend/app.py))

- Sidebar: Call ID + Acronym (required), action type, criteria editor
  with XLSX import/export popup, threshold slider, History popup with
  search + click-to-detail (separate popup), KB stats line.
- Main: file uploader + paste textarea, compact configuration summary,
  light-yellow Run button (centered, small), result section with score
  hero, ESR summary, per-criterion cards, JSON / CSV export.
- Defensive run pattern: clicking Run stashes inputs and reruns; on the
  next pass the sidebar renders with all interactive buttons disabled
  before the LLM call begins, so a sidebar dialog click can't interrupt
  the evaluation.

---

## Repository layout

```
EDF_Evaluator/
├── ARCHITECTURE.md            # high-level architecture for company review
├── README.md                  # this file
├── requirements.txt
├── previous_ESRs/             # *** drop your historical ESR PDFs here ***
├── backend/
│   ├── config.py              # env-overridable settings
│   ├── ingestion.py           # DOCX/PDF/TXT extraction + sectioning
│   ├── chunking.py            # token-aware chunk splitter
│   ├── kb.py                  # Chroma KB (ESR + evaluation-unit records)
│   ├── evaluator.py           # per-criterion LLM chain + EDF rubric
│   ├── pipeline.py            # full evaluate_proposal orchestration
│   ├── db.py                  # SQLite history of runs
│   └── criteria_io.py         # XLSX import / export
├── frontend/
│   └── app.py                 # Streamlit UI
├── scripts/
│   └── build_kb.py            # batch ESR indexer (run before launching)
├── data/                      # created at runtime (gitignored)
│   ├── kb/                    # Chroma persistence
│   └── runs.sqlite            # run history
└── input_docs/                # legacy fallback for *ESR*.pdf
```

---

## Running it

### Prerequisites

- Python 3.11+ (tested in conda env `genai`).
- Ollama running locally (`ollama serve`).
- Models pulled:
  ```
  ollama pull gemma3:12b
  ollama pull nomic-embed-text
  ```

### Install + launch (recommended order)

```bash
cd EDF_Evaluator
pip install -r requirements.txt

# 1. Drop every historical ESR PDF you have into ./previous_ESRs/
#    (any filename — e.g. SWARMER_ESR.pdf, ATHENA_ESR_2023.pdf, ...)

# 2. Build the knowledge base. Run this whenever you add new ESRs.
python -m scripts.build_kb

# 3. Start the UI
streamlit run frontend/app.py
```

Open `http://localhost:8501`.

The script is **idempotent** — re-running it after dropping new PDFs into
`previous_ESRs/` only adds the new ones (existing records are silently
refreshed). To do a clean rebuild, `rm -rf data/kb/` first.

### Environment overrides (all optional)

| Env var                  | Default                  | Purpose                                |
|--------------------------|--------------------------|----------------------------------------|
| `OLLAMA_BASE_URL`        | `http://localhost:11434` | Ollama daemon URL                      |
| `EDF_CHAT_MODEL`         | `gemma3:12b`             | Chat model id (`ollama pull` first)    |
| `EDF_EMBED_MODEL`        | `nomic-embed-text`       | Embedding model id                     |
| `EDF_KB_TOP_K`           | `4`                      | Records retrieved per criterion        |
| `EDF_PROPOSAL_TOP_K`     | `6`                      | Proposal chunks per criterion          |
| `EDF_CHUNK_SIZE`         | `1500`                   | Chunk size in chars                    |
| `EDF_CHUNK_OVERLAP`      | `200`                    | Chunk overlap                          |
| `EDF_MAX_EMBED_CHARS`    | `2500`                   | Hard per-call embedding budget         |
| `EDF_MAX_INLINE_CHARS`   | `12000`                  | Hard per-criterion LLM budget          |
| `EDF_REQUEST_TIMEOUT`    | `300`                    | Ollama request timeout (s)             |

---

## Feeding the system more samples (the learning loop)

Three ways to enrich the knowledge base, from highest to lowest trust:

1. **Drop real ESR PDFs into `previous_ESRs/` and run the indexer.**
   ```bash
   cp ~/path/to/SWARMER_ESR.pdf previous_ESRs/
   cp ~/path/to/ATHENA_ESR.pdf  previous_ESRs/
   python -m scripts.build_kb
   ```
   Each PDF is decomposed into per-criterion blocks and indexed in the
   gold tier (`esr_criterion`). These dominate retrieval. Adding more
   ESRs and re-running the script is the **single most effective way**
   to make evaluations better.
2. **Run real proposals through the system** — every full-proposal run
   is auto-indexed as an `evaluation_unit`. After ~20 runs, retrieval
   typically surfaces useful patterns even without official ESRs.
3. **Promote runs you trust** — open a past run in the History popup and
   tick "Mark this run as reviewer-approved". The records jump from the
   advisory tier to the reviewer-approved tier and the LLM is told to
   weight them as a strong signal in future evaluations.

The KB stats line in the sidebar shows the current breakdown:
`X official ESR · Y prior runs (Z reviewer-approved)`.

### Performance tuning

Generation speed scales inversely with prompt size. The two knobs that
matter most:

- `EDF_KB_TOP_K` (default 4) — how many historical records each
  criterion's prompt includes. Lower = faster, higher = more context.
- `EDF_PROPOSAL_TOP_K` (default 6) — how many proposal chunks each
  criterion's prompt includes. Same trade-off.

If gemma3:12b feels too slow on CPU, the cleanest speed-up is a smaller
model (`EDF_CHAT_MODEL=qwen2.5:7b-instruct` or `llama3.1:8b`) at the cost
of some calibration quality. Long-term, run on the GPU server.

---

## Honest caveats

- **Calibration is anchored on the gold tier.** Quality scales with the
  number of real ESRs you feed in. A KB with one official ESR is much
  weaker than one with twenty.
- **System-generated examples can compound errors.** That's why they're
  tagged separately and the prompt explicitly tells the model to ignore
  their numerical scores. Promote them to "reviewer-approved" only after
  a human has actually validated the evaluation.
- **No fine-tuning.** This is pure retrieval-augmented prompting. If you
  want a model that internalises EDF style, that's a separate fine-tuning
  step (LoRA on the per-criterion findings).
- **The tool is positioned as pre-screening only.** Final judgement is
  human, by policy and by design.
