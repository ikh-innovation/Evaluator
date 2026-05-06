# EDF Proposal Evaluator — High-Level Architecture

Internal, on-prem, AI-assisted pre-screening tool for EDF Part B proposals.
Final judgement remains with human evaluators — this tool produces an
ESR-style simulated assessment to surface weaknesses before submission.

> Scope confirmed against `input_docs/EDF Proposal Evaluator_AI Team.docx`,
> the EDF-2024-LS-RA-CHALLENGE call fiche, and the SWARMER ESR sample.

---

## 1. Goals & Non-Goals

**Goals**
- Configurable criteria + weights per call (criteria change every few months).
- Criterion-level evaluation: score (0–5, half-points), strengths, weaknesses,
  missing evidence, recommended improvements, ESR-style language.
- RAG over historical Part B proposals + ESRs, retrieving at **criterion level**.
- Weighted total score, threshold check, confidence indicator.
- Full audit trail; reviewer override; DOCX/PDF export.

**Non-goals (v1)**
- Auto-submission to the EU portal.
- Replacing human evaluators.
- Fine-tuning custom models (kept for a later phase).

---

## 2. Inputs and Outputs (in plain terms)

**You give the system:**
- A Part B proposal — full DOCX, or just selected sections.
- A set of criteria with weights — hard-coded as a default per call type
  (Research / Development), but admin-editable on demand when EDF rules change.
- Reference samples — past Part B proposals paired with their ESRs (scores +
  evaluator comments). *Today we have 1 sample (SWARMER); the system is
  designed so quality scales as more pairs are added.*

**The agents give back:**
- A predicted score per criterion (0–5, half-points) with weighted total
  and pass/fail vs. threshold.
- Predicted ranking band ("likely funded / reserve / rejected") based on
  patterns in the sample set.
- Evaluator-style comments per criterion — strengths, weaknesses, missing
  evidence — written in the language of real ESRs from the samples.
- Concrete improvement suggestions, citations to which sample passages
  drove each judgement, and a confidence flag.

**Important caveat about today:** with 1 sample, predicted scores and
rankings are indicative only. The system leans on the official rubric +
call fiche; sample-driven calibration kicks in once the corpus grows
(rule of thumb: ~20+ Part B/ESR pairs before rankings become trustworthy).

---

## 3. High-Level Component View

```
                        ┌────────────────────────────────────┐
                        │           Reviewer UI              │
                        │  (upload, configure, review, HITL) │
                        └───────────────┬────────────────────┘
                                        │ REST/WebSocket
                        ┌───────────────▼────────────────────┐
                        │       Backend API (FastAPI)        │
                        │  auth · jobs · audit · exports     │
                        └─┬───────────┬───────────┬──────────┘
                          │           │           │
                ┌─────────▼──┐  ┌─────▼─────┐  ┌──▼─────────────┐
                │ Ingestion  │  │ Evaluation │  │ Report builder │
                │  pipeline  │  │  engine    │  │ DOCX / PDF     │
                │ (docx,pdf) │  │ (LangGraph)│  └────────────────┘
                └─────┬──────┘  └─────┬──────┘
                      │               │ prompt + retrieved ctx
                      │               │
                      │           ┌───▼───────────────┐
                      │           │  LLM serving      │
                      │           │  Ollama / vLLM    │
                      │           │  (GPU server)     │
                      │           └───────────────────┘
                      │
              ┌───────▼─────────┐    ┌──────────────────────┐
              │ Object storage  │    │  Vector DB (Qdrant)  │
              │ MinIO  (DOCX/   │    │  per-criterion chunks│
              │ PDF originals)  │    │  + rich metadata     │
              └─────────────────┘    └──────────┬───────────┘
                                                │
                                     ┌──────────▼───────────┐
                                     │ Postgres             │
                                     │ configs · audit ·    │
                                     │ users · proposals    │
                                     └──────────────────────┘
```

---

## 4. Data Model (the bits that matter)

**Call configuration** (admin-editable, versioned)
- `call_id`, `action_type` (Research / Development), `call_year`
- `criteria[]`: `{ name, description, weight, max_score, individual_threshold? }`
- `overall_threshold`
- `section_to_criterion_map` (which Part B sections feed which criterion)

**Proposal record**
- File ref (MinIO), parsed structure JSON, version, status

**Historical knowledge unit** (the atom of the RAG index)
A single ESR criterion block decomposed into one record:
```
{
  source_proposal: "SWARMER",
  call_id: "EDF-2024-LS-RA-CHALLENGE",
  action_type: "Research",
  criterion: "Implementation",
  score: 4.5,
  weight: 1,
  comment_text: "...",
  linked_proposal_sections: ["WP structure", "risks", "milestones"],
  outcome: "funded" | "reserve" | "rejected",
  consortium_size: 9,
  trl_range: "...",
  tags: ["risk-mitigation", "WBS"]
}
```
Each record is embedded; metadata is queryable as filters.

**Audit log** — every prompt, retrieved context, model response, reviewer
edit, with timestamps and user. Required by the spec's non-functional reqs.

---

## 5. Core Workflow (per evaluation run)

1. **Upload + configure.** Reviewer uploads Part B DOCX, selects an active
   call configuration.
2. **Ingestion.**
   - Parse DOCX (`python-docx` + `unstructured`), keep tables/figures.
   - Section the document (WBS, risks, milestones, consortium, impact, …).
   - Persist normalized JSON + raw file in MinIO.
3. **Section-to-criterion routing.**
   - Combine config-driven mapping with semantic match (embed each section,
     match against criterion descriptions).
4. **Per-criterion evaluation (LangGraph DAG).** For each criterion:
   - Retrieve top-K historical units filtered by `criterion` + `action_type`
     + (optionally) `outcome`.
   - Render the prompt template from the spec (§9 of the use-case doc).
   - Call LLM via Ollama, request structured JSON output.
   - Optional self-critic node: re-checks score ↔ comments consistency,
     flags hallucinated facts (anything not citable to retrieved chunks
     or the proposal itself).
5. **Aggregate.** Weighted total, threshold pass/fail, per-criterion
   confidence (function of retrieval quality + model-stated confidence).
6. **Render report.** Per-criterion table + ESR-style summary + rewrite
   priorities. Export DOCX/PDF.
7. **Human-in-the-loop.** Reviewer can override any score or comment;
   overrides are logged and (later) reused as feedback signal.

---

## 6. Recommended Tech Stack (all open-source, on-prem)

| Layer              | Choice                                          | Notes |
|--------------------|-------------------------------------------------|-------|
| LLM serving        | **Ollama** (start), vLLM (if throughput needed) | GPU server |
| LLM model          | Qwen2.5-32B / 72B Instruct, or Llama-3.3-70B    | Long context (128K), strong reasoning, open weights |
| Embeddings         | `bge-large-en-v1.5` or `nomic-embed-text`       | Served via Ollama |
| Orchestration      | **LangChain + LangGraph**                       | LangGraph for the multi-step DAG (per-criterion → aggregate → critic) |
| Vector DB          | **Qdrant**                                      | On-prem, fast, rich metadata filtering |
| Doc parsing        | `python-docx`, `unstructured`, `pdfplumber`     | DOCX for Part B, PDF for ESRs |
| Backend            | **FastAPI** (Python)                            | Native LangChain integration |
| DB                 | Postgres                                        | Configs, audit, users |
| Object storage     | MinIO                                           | S3-compatible, on-prem |
| Auth + RBAC        | Keycloak (OIDC)                                 | Per spec's RBAC requirement |
| Frontend           | **Streamlit** (Phase 0–1) → React/Next (later)  | Streamlit gets you to a working internal tool fastest |
| Export             | `python-docx`, `weasyprint`                     | DOCX + PDF reports |
| Deployment         | Docker Compose on GPU server → k8s later        | All services co-located, no external egress |

---

## 7. Phasing

**Phase 0 — Prototype (1–2 weeks).** Single user, Streamlit + LangChain +
Ollama + Chroma. Hard-code one call's criteria. Validate output quality on
2–3 historical proposals where the real ESR scores are known. Goal:
prove the LLM produces useful, non-hallucinated, ESR-style output.

**Phase 1 — Internal MVP.** FastAPI + React/Streamlit, Qdrant, Postgres,
MinIO. Criteria admin UI. Audit log. DOCX export. RBAC via Keycloak.

**Phase 2 — Feedback loop.** Capture reviewer overrides as labeled signal.
Use as improved RAG few-shot examples. Version history per proposal.
Tables/figures handling, OCR fallback.

**Phase 3 — Calibration.** Light fine-tuning (LoRA) on the accumulated
override corpus. Continuous calibration as new ESRs land.

---

## 8. Risks & Open Questions to Align On

- **Sensitivity classification.** Are ESRs / Part Bs handled at
  EU-RESTRICTED level? Drives RBAC granularity, network isolation,
  potentially encryption-at-rest specifics.
- **Historical corpus size.** How many `(Part B, ESR)` pairs do we have?
  Below ~20 pairs, RAG quality is the main bottleneck; we'll lean harder
  on the criterion-rubric and the call fiche, less on retrieval.
- **GPU budget.** Available VRAM determines max model size and whether
  we can run a critic pass (two LLM calls per criterion).
- **Hallucination guardrail.** Spec says: "Do not invent facts not present
  in the proposal." We'll enforce this via (a) structured output with
  required `evidence_quote` fields, (b) a self-critic node, (c) UI
  highlighting any claim without a citation.
- **ESR template fidelity.** Should the exported DOCX mirror the official
  EDF ESR layout (criterion-by-criterion with weight × score)?
- **Concurrency.** Are multiple proposals evaluated in parallel during
  pre-submission crunch periods? Drives queue/worker design.

---

## 9. What I'd Build First (concrete next step)

A throwaway Phase-0 spike on a single criterion (e.g., *Implementation*)
using SWARMER's Part B + its real ESR as ground truth: see if the
LLM, given the rubric + retrieved snippets, lands within ±0.5 of the
real evaluator score and produces comments a bid manager would accept.
That single experiment de-risks the entire architecture.
