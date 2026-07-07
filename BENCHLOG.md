# BENCHLOG — EDF Proposal Evaluator fix campaign

Append-only log of each fix, the commit it landed in, the model used, and the
per-criterion metrics observed. Follows the working order in `FIX README v2`.

---

## STEP 0 — Reproduce & verify (telemetry + guards)

**Code commit:** _pending — telemetry changes staged in working tree, not yet committed._
**Model (deployment):** `qwen3:30b-a3b-instruct-2507-q4_K_M` chat (instruct/non-thinking — compatible with `format="json"`), `qwen3-embedding:4b` embed.
**Window/budgets (from `.env`):** `EDF_NUM_CTX=32768`, `EDF_MAX_INLINE_CHARS=48000`, `EDF_PROPOSAL_TOP_K=32`, `EDF_MAX_CHUNKS_PER_SECTION=4`. Worst-case criterion prompt+output ≈ 17.1k tokens (fits 32768 with headroom).

### Findings (from static audit + run-history inspection; no new run executed here)

1. **No v2 run exists yet.** All 11 rows in `data/runs.sqlite` predate the v2
   prompting commit `61097d8` (Tue Jul 7 2026); the newest (`ACHITOS_2`,
   Jul 6) has **none** of the v2 per-criterion fields (`band`, `input_coverage`,
   `input_gaps`, `full_document_input`). Every stored run was produced by **v1**
   code.
   - **Consequence for the plan:** the "externally observed v2 run reporting
     `confidence: high` on the partial branch" was in fact a **v1** run. v1 had
     no coverage-based confidence cap at all (its rule was "high if multiple
     quotes"), so the contradiction the plan flags is explained, not a v2 bug.
   - A genuine v2 baseline still has to be generated (runbook below).

2. **Every stored run is `overall_confidence: high`, band "Likely funded",
   weighted total 35.5–36.5 / 45.** Consistent with §7b: v1 confidence never
   reflected coverage.

3. **Context window vs prompt size — resolved by config.** The initial `.env`
   had `EDF_NUM_CTX=8192`, below the worst-case partial-branch prompt (~9,595
   tokens: 6,382-char skeleton + 24,000-char excerpt + 8,000-char call context),
   so Ollama would silently truncate the excerpt/output. Resolved by raising
   `EDF_NUM_CTX=32768` **and** enlarging the excerpt budgets in tandem
   (`EDF_MAX_INLINE_CHARS=48000`, `EDF_PROPOSAL_TOP_K=32`,
   `EDF_MAX_CHUNKS_PER_SECTION=4`) to lift per-criterion coverage. New worst case
   ≈ 17.1k tokens < 32768. **If Ollama OOMs on the deployment GPU:** drop
   `EDF_NUM_CTX` to 24576/16384 (and `EDF_MAX_INLINE_CHARS` proportionally), or
   enable KV-cache quantization on the Ollama host.

4. **UI shows the capped confidence, not a raw model field.**
   `frontend/app.py:_render_criterion_card` reads `c.get("confidence")` from the
   stored run; capping happens in `evaluate_criterion` before persistence. No
   stale-cache path (result/history fragments re-query `db.get_run` each render).

### Changes landed in this step (all deterministic; no scoring behaviour changed)

- **Telemetry — model-reported confidence preserved.** `evaluate_criterion`
  now records `model_confidence` (raw, pre-cap) and `confidence_cap` (the cap
  applied) alongside the final capped `confidence`. — `backend/evaluator.py`
- **Telemetry — what the model saw.** `get_proposal_excerpt` now returns
  `selected_chunk_ids`; the per-criterion result now carries `section_titles`
  and `selected_chunk_ids` in addition to `input_coverage`/`full_document_input`.
  Persisted automatically (whole result is JSON-serialised by `db.save_run`).
  — `backend/pipeline.py`
- **Wrong-document guard.** `detect_esr_markers` flags a document that opens
  with evaluation-report language (generic, call-/proposal-agnostic phrase set).
  `evaluate_proposal` sets `result["ingest_warning"]`; the UI shows it as a
  `st.warning` in both the live result panel and history detail.
  — `backend/ingestion.py`, `backend/pipeline.py`, `frontend/app.py`

### Static verification (no model/embedding calls)

- Full-document + empty-index branches of `get_proposal_excerpt` return the new
  `selected_chunk_ids` / `section_titles` keys. ✓
- `detect_esr_markers`: flags `SWARMER …_ESR.pdf` (`evaluation summary report`,
  `proposal evaluation form`, `evaluation result`); silent on the Part B
  template and the proposal-like DOCX. ✓
- `.env` resolves `config.CHAT_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M`,
  `EMBED_MODEL=qwen3-embedding:4b`, `NUM_CTX=8192`; no gemma. ✓

### Runbook — generate the v2 baseline on YOUR deployment host

> A real regression baseline needs an actual Part B proposal file. The repo
> contains only the **blank template** and an **ESR** — neither is a proposal.
> Supply the real proposal (and its call fiche) locally; do not commit them.

1. Ensure `.env` (already created) is present and Ollama has both qwen3 models.
2. `docker compose up --build` (compose auto-loads `.env`).
3. In the UI: set Call ID + acronym, **paste the call topic context**, upload the
   real Part B, Queue evaluation. (Watch for the ⚠️ ingest warning — if it fires,
   you uploaded an ESR/scoresheet, not the proposal.)
4. Export the run JSON and paste the per-criterion telemetry table below:

| Criterion | full_doc | coverage | model_conf | cap | final_conf | score | band | #weak | #input_gaps |
|---|---|---|---|---|---|---|---|---|---|
| 1 Excellence | | | | | | | | | |
| 2 Innovation | | | | | | | | | |
| 3 Competitiveness | | | | | | | | | |
| 4 EDTIB autonomy | | | | | | | | | |
| 5 Cross-border coop | | | | | | | | | |
| 6 Implementation | | | | | | | | | |

5. Sanity-check against §3.3 expectation: a full Part B is far over budget →
   every criterion should be `full_doc=False` with `coverage ≈ 0.02–0.05`. If any
   criterion shows `full_doc=True`, the uploaded file was too small (likely not a
   full proposal) — flag before proceeding to FIX 1.

_Do not tune anything to this baseline; it is the regression reference only._

---

<!-- Append FIX 1a, 1c, 1b, 1d, … below, one section per commit. -->
