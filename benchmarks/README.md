# Benchmark harness

Regression-tests the evaluator against official (proposal, ESR) pairs and
against synthetic direction tests. n=1 cannot distinguish skill from luck —
adding pair #2, #3, … must be trivial, so each pair is just a folder.

> Proposal documents and official ESRs are sensitive. `pairs/` contents are
> gitignored except the pair descriptors you explicitly choose to commit.

## Adding a pair

Create `benchmarks/pairs/<name>/pair.json`:

```json
{
  "acronym": "ACRONYM",
  "call_id": "EDF-20XX-...",
  "action_type": "Research",
  "proposal_file": "partb.docx",
  "call_context_file": "call_topic.txt",
  "official_weighted_total": 38.0,
  "official": {
    "Excellence and potential of disruption": {
      "score": 3.5,
      "praised_aspects": ["keyword or phrase the official ESR praised", "..."],
      "criticised_aspects": ["..."]
    },
    "...one entry per criterion...": {}
  }
}
```

- `proposal_file` / `call_context_file` are relative to the pair folder.
  `call_context_file` is optional (plain text with the call topic description
  and functional requirements).
- `criteria` (optional): full criteria list as in the XLSX schema; defaults to
  the standard set for the `action_type`.
- `praised_aspects`: short keywords/phrases from the official ESR's positive
  comments. Any AI weakness or missing-evidence item that mentions one of
  these is counted as a **contradiction** — the primary failure metric.

## Running

Inside the backend image (repo mounted at `/repo`):

```bash
docker run --rm -v "<repo>:/repo" -w /repo \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e EDF_CHAT_MODEL=gemma3:12b -e EDF_EMBED_MODEL=nomic-embed-text \
  evaluator-backend python benchmarks/run_benchmark.py
```

Options: `--pair <name>` (run one pair), `--max-delta 0.5`,
`--max-total-delta 2.0` (regression thresholds; exit code 1 when breached or
when any contradiction is found).

Reports are written to `benchmarks/results/` (JSON) and printed as a table.

## Metrics

Per pair and aggregated across pairs:

| Metric | Meaning |
|---|---|
| per-criterion Δ | AI score − official score |
| weighted total Δ | AI weighted total − official weighted total |
| contradictions | AI weaknesses/missing-evidence that mention an aspect the official ESR praised |
| specificity ratio | fraction of AI weaknesses containing a verbatim quote (≥ 12 chars) actually present in the proposal text |
| confidence/coverage | per-criterion confidence vs. fraction of the document the model saw |

## Synthetic direction tests

`run_synthetic.py` needs no official ESR: it feeds a deliberately excellent
and a deliberately deficient implementation section (fictional content) to the
same criterion prompt and asserts score direction — the excellent one must
reach ≥ 4.0 with few/no weaknesses, the deficient one must fall ≤ 3.0 with
pointed criticism. This guards against both critique bias and its mirror
image, praise bias.

```bash
docker run --rm -v "<repo>:/repo" -w /repo \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e EDF_CHAT_MODEL=gemma3:12b -e EDF_EMBED_MODEL=nomic-embed-text \
  evaluator-backend python benchmarks/run_synthetic.py
```
