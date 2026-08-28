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

## STEP 0b — Deployment drift fixed; temperature 0; TOP_K rebalance (2026-07-07)

**Model (deployment):** `qwen3:30b-a3b-instruct-2507-q4_K_M` chat, `qwen3-embedding:4b` embed — verified live via `/api/ps` (21.8 GB VRAM, ctx 32768).

### Root cause of "evaluations feel too fast / shallow"

The running compose stack was built **Jul 3** — before every prompting/window
commit (61097d8, 9959cbe, 8a8ba3e). The live containers ran the **v1 prompt**
with `MAX_INLINE=5000`, `TOP_K=4`, temps 0.1/0.3 (verified by introspecting
`backend.config` inside the old containers). Per-criterion prompts were ~2k
tokens → seconds-fast, tiny-input evaluations. Additionally the Dockerfiles do
not copy `.env` and compose only forwarded 6 vars, so the enlarged retrieval
budgets **never reached the containers at all**.

### Changes

- `backend/evaluator.py`: temperature **0.0** for both the JSON eval client and
  the text (summary/band) client — full determinism at fixed input.
- `docker-compose.yaml`: `env_file: .env` added to auth/frontend/backend so
  every `.env` key (incl. `EDF_MAX_INLINE_CHARS`, `EDF_PROPOSAL_TOP_K`,
  `EDF_MAX_CHUNKS_PER_SECTION`) reaches the containers.
- `.env`: `EDF_PROPOSAL_TOP_K` 32 → **64**. Observed avg chunk is ~1,100 chars
  (not the nominal 1,500), so at 32 the chunk count was the binding cap and only
  ~18.5k of the 48k char budget was used. At 64 the char budget binds.
- Stack rebuilt (`docker compose up -d --build`). **Operational rule: rebuild
  after every code change — containers COPY the code, they do not mount it.**

### In-container smoke test (Part B template, 865k chars / 768 chunks)

| Step | Result |
|---|---|
| DOCX parse (heading styles) | 30 sections, 768 chunks, 0.4 s |
| Embedding index | 150.7 s |
| Retrieval (criterion 1) | 47,952 chars (budget-bound), coverage 5.28%, pulls §1.1/§1.2/§1.3/§2.1… |
| LLM eval (criterion 1) | 14.4 s, valid JSON, band=excellent, quote-anchored strengths |
| Confidence chain | model=high → cap=medium (partial input) → final=medium ✓ |

At TOP_K=32 the same retrieval returned 18,487 chars / 2.00% coverage and
missed §1.3 (the criterion's own section); at 64 it is included.

---

## FIX 1 — Leniency collapse fixed: mandatory weakness scan + ESR calibration (2026-07-07)

**Baseline failure (v2 prompt, run `SWARMER3`):** 45.0/45, every criterion
`excellent`/5.0, **zero** weaknesses, zero missing_evidence. Official DEFIS ESR
for the same proposal (101224078/SWARMER): **38.00/45** (3.5, 4.0, 4.5, 4.0,
5.0, 4.5), every criterion carrying "However, …" weaknesses.

**Root cause:** the v2 anti-generic-criticism guards outlawed the exact genre
real ESRs write. Rule 2's self-check ordered deletion of findings like "no
quantified metrics"; rule 6 declared "wanting more sub-detail … is not a
finding" — yet the real ESR's weaknesses are precisely "does not sufficiently
detail/demonstrate/clarify X". With all weaknesses suppressed, rule 6 mapped
empty-weaknesses → excellent → 5.0 mechanically. The single excellent/5.0
illustration anchored the same pattern. Temp 0 made it deterministic.

**Changes (`backend/evaluator.py`, `backend/pipeline.py`):**
- Rubric calibration: funded proposals typically score 3.5–4.5 per criterion;
  straight 5.0s mean the evaluation failed to probe.
- Rule 2: "does not sufficiently detail/demonstrate X" is a VALID weakness when
  X is proposal-specific and material; only the generic no-X form stays banned.
- Rule 5 → mandatory `weakness_scan` (3–5 hostile probes per criterion, each
  `weakness` or `cleared`); empty weaknesses legitimate only if all cleared.
- Rule 6 band ladder mapped to half-steps (5.0 = all cleared; 4.5 = 1–2 minor;
  4.0 = several minor; 3.5 = many minor / one borderline; …).
- Illustration replaced with a "very good 4.0" example incl. scan + weaknesses.
- Retrieval: adversarial query per criterion ("risks, limitations, assumptions,
  dependencies…") so weakness-bearing content reaches the excerpt.
- Run-level calibration warning when a whole run has zero weaknesses.

**Verification (full pipeline in-container, same document, 232 s):**

| Criterion | tool | real ESR | #weak | scan |
|---|---|---|---|---|
| Excellence | 4.0 | 3.5 | 4 | 5 probes / 3 weak |
| Innovation | 4.0 | 4.0 | 4 | 5 / 3 |
| Competitiveness | 4.0 | 4.5 | 5 | 5 / 4 |
| EDTIB autonomy | 4.0 | 4.0 | 4 | 5 / 3 |
| Cross-border coop | 4.0 | 5.0 | 3 | 5 / 3 |
| Implementation | 4.0 | 4.5 | 5 | 5 / 4 |
| **Weighted total** | **36.0/45** | **38.00/45** | | |

Finding quality: the tool's top Excellence weakness ("claims 'high-precision
positioning' in GNSS-denied environments without specifying the sensor fusion
architecture") independently converges on the real ESR's top weakness
(GNSS-denied navigation reliance not clarified).

**Known residual:** per-criterion discrimination is compressed — every
criterion lands on 4.0 (real spread 3.5–5.0). The scan now finds 3–5 probes
everywhere; the band ladder maps "several minor" → 4.0 uniformly. Next
candidate fix: weigh cleared-vs-weak probe ratio into the half-step choice, or
let strong criteria (all probes cleared/minor presentational only) reach 4.5–5.0
more readily.

---

## FIX 2 — Score compression fixed: substance-led banding + severity envelope (2026-08-28)

**Reported symptom:** across more than three proposals every report came back at
**80%** — every criterion 4.0/5, "very good". Confirmed in `data/runs.sqlite`:
of the stored per-criterion scores, `4.0` appears 6× against one `3.0`, one
`2.0` and one `1`, and **every** stored `band` is `"very good"`. This is the
residual FIX 1 flagged and did not close.

**Root cause — three compounding prompt defects, all pushing to the same value:**

1. **Rule 5 imposed a quota.** "list the 3–5 aspects … a hostile reviewer would
   probe" made the probe count a target rather than a property of the text. The
   model filled it every time: FIX 1's own verification table shows **5 probes
   on all six criteria**, 3–4 of them "weakness".
2. **Rule 6 scored by weakness COUNT.** The ladder read "4.5 = one or two minor
   weaknesses; 4.0 = several minor weaknesses". Feed a rule that maps "several
   minor" → 4.0 with a quota that manufactures "several minor" every time and
   4.0 is arithmetically forced, whatever the proposal says. The ladder was also
   purely subtractive from a notional 5.0 — positive substance never entered the
   band, so a criterion could not be scored *up* for answering well.
3. **Rule 6 asked for a classification the schema could not express.** It told
   the model to classify each weakness SIGNIFICANT or MINOR, but
   `weakness_scan[].verdict` only admitted `"weakness" | "cleared"` and no other
   field carried severity. The classification was invisible, unverifiable, and
   in practice skipped — so the one signal that could separate a 3.0 from a 4.5
   never left the model.
   Aggravating factor: the single illustration was `"band": "very good",
   "score": 4.0` — a direct anchor at `temperature=0.0`.

**Changes (`backend/evaluator.py`):**
- Rubric gains a **DISCRIMINATION** paragraph: a proposal's criteria are not
  equally well addressed, real ESRs spread 3.5–5.0, an identical score on every
  criterion is an evaluation that averaged rather than judged. Also corrects
  FIX 1's overreach that 5.0 requires an empty weaknesses list — real ESRs award
  5.0 with a "However, …" attached, which had made the top of the scale dead.
- Rule 5: quota removed — probe count is set by the text ("a thin or evasive
  section deserves more probes than a thorough one") and explicitly must not be
  a proxy for the score. `verdict` becomes **three-way**:
  `"significant" | "minor" | "cleared"`, each defined in place, with a warning
  against softening a real finding to "minor" to be safe.
- Rule 6 restructured into **STEP A (band from positive substance)** — how
  completely and how convincingly this text answers this criterion — and
  **STEP B (half-step from the classified probes)**, with the explicit rule that
  two criteria carrying the same number of minor weaknesses must *not* score the
  same. STEP B states the enforced bounds numerically (see envelope below).
- New required output field **`band_rationale`**: one or two sentences on why
  this band and not the one above and not the one below. Forcing the comparison
  is what breaks the drift to a comfortable middle — "if you cannot say why the
  band above is wrong, the band above is probably right."
- Illustration replaced by **two** examples at **4.5 (`excellent`)** and **3.0
  (`good`)**, prefaced by a note that the bands shown are arbitrary. Neither
  sits on 4.0; the anchor is gone.

**Deterministic backstop (`backend/evaluator.py`) — `_apply_severity_envelope`:**
prose alone did not survive contact with the model before, so the severities the
model itself records now bound the score it keeps. Applied after
`_reconcile_band_and_score`, band re-derived from any moved score:

| model's own scan | score bound |
|---|---|
| 1 "significant" probe | ≤ 4.0 |
| 2 | ≤ 3.5 |
| 3 | ≤ 3.0 |
| 4+ | ≤ 2.5 |
| every probe "cleared", no weakness kept (≥2 probes) | **≥ 4.5** |

The floor matters as much as the ceilings: ceilings alone would have turned "4.0
everywhere" into "4.0 or lower everywhere" rather than a spread. It is
suppressed on an off-topic (overridden-gate) run, where the screening notice and
not the probe tally governs, and it only applies to bands "good" and above so an
unaddressed criterion with an empty scan can never be lifted.
Telemetry added per criterion: `significant_weakness_count`,
`minor_weakness_count`, `cleared_probe_count`, `severity_envelope` (the
adjustment note, `""` when nothing moved), alongside the existing
`model_band`/`model_score`.

**Run-level guard (`backend/pipeline.py`):** a run whose criteria all land on the
identical score (≥3 criteria) now raises a calibration warning — the collapse is
reported as undifferentiated rather than passing as agreement. A second warning
lists criteria the severity envelope moved.

**Surfaced (`frontend/app.py`, `frontend/pdf_report.py`):** "Why this band" and
the score-adjustment note per criterion, in both the UI card and the PDF.

### Verification (static — no Ollama on this host, see caveat)

| Check | Result |
|---|---|
| `ast.parse` on all five changed files | ✓ |
| `EVAL_PROMPT.format()`, normal + off-topic branch | ✓ (14.3k-char skeleton) |
| Brace escaping — `{{`/`}}` balanced 10/10, 9 placeholders all bound | ✓ |
| Both illustrations parse as JSON, carry all 11 schema keys, verdicts in the new enum, kept probes == weaknesses | ✓ |
| Illustrations stable under `_reconcile_band_and_score` + envelope (an example the code would correct teaches the wrong mapping) | ✓ |
| `_severity_counts` — mixed/legacy `"weakness"`/junk/non-list | ✓ (legacy → minor) |
| Envelope: 12 ceiling/floor cases, `max_score` 5 and 10, `max_score=0` guard | ✓ |
| Band/score self-consistency over all 480 (sig, minor, cleared, band) combinations | ✓ 0 inconsistent |
| Worst-case prompt vs `EDF_NUM_CTX=32768` | ✓ 19.5k in + 1.2k out, 12.0k headroom (was ~17.1k) |

**⚠ Not yet verified against the model.** No Python or running Ollama on the
authoring host, so no end-to-end run was executed. The intended effect —
per-criterion spread — is a claim about model behaviour and is unproven until
the runbook below is executed on the deployment host.

### Runbook — confirm the fix on the deployment host

1. `docker compose up -d --build` (containers COPY the code; a rebuild is
   mandatory — see STEP 0b).
2. `python benchmarks/run_synthetic.py` — the direction test is the cheapest
   probe and now prints the severity counts and `band_rationale`. Expect the
   excellent section ≥ 4.5 (the clean-scan floor should fire) and the deficient
   section ≤ 3.0, gap ≥ 1.5.
3. Re-run the SWARMER Part B and compare against the real DEFIS ESR
   (3.5, 4.0, 4.5, 4.0, 5.0, 4.5 — **38.00/45**):

| Criterion | real ESR | tool (FIX 1) | tool (FIX 2) | sig/minor/cleared | envelope |
|---|---|---|---|---|---|
| Excellence | 3.5 | 4.0 | | | |
| Innovation | 4.0 | 4.0 | | | |
| Competitiveness | 4.5 | 4.0 | | | |
| EDTIB autonomy | 4.0 | 4.0 | | | |
| Cross-border coop | 5.0 | 4.0 | | | |
| Implementation | 4.5 | 4.0 | | | |
| **Weighted total** | **38.00/45** | **36.0/45** | | | |

   Success is **spread**, not a closer total: ≥3 distinct scores across the six
   criteria, and no run-level "every criterion received the SAME score" warning.
   Rank correlation against the ESR column matters more than absolute agreement.
4. If every criterion still lands on one value, read the new
   `significant_weakness_count` column first. All-zero means the model is still
   refusing to call anything significant — tighten the "significant" definition
   in rule 5 rather than touching the envelope, which can only bound a score and
   cannot create discrimination on its own.

---

<!-- Append FIX 2a, 3, … below, one section per commit. -->
