"""Benchmark runner: evaluates (proposal, official ESR) pairs and reports
per-criterion deltas, contradiction counts and specificity ratios.

Each pair is a folder under benchmarks/pairs/ with a pair.json descriptor —
see benchmarks/README.md. Exit code 1 when a regression threshold is breached
or any contradiction with officially praised aspects is found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import CHAT_MODEL  # noqa: E402
from backend.default_criteria import (  # noqa: E402
    DEFAULT_DEVELOPMENT_CRITERIA,
    DEFAULT_RESEARCH_CRITERIA,
)
from backend.ingestion import extract_text  # noqa: E402
from backend.pipeline import evaluate_proposal  # noqa: E402

PAIRS_DIR = ROOT / "benchmarks" / "pairs"
RESULTS_DIR = ROOT / "benchmarks" / "results"

_QUOTE_RE = re.compile(r"[\"“”']([^\"“”']{12,})[\"“”']")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _finding_is_specific(finding: str, proposal_norm: str) -> bool:
    """A finding is 'specific' when it quotes ≥12 chars verbatim from the
    proposal — the enforceable proxy for 'names a partner/technology/WP/claim
    from the input document'."""
    return any(_norm(q) in proposal_norm for q in _QUOTE_RE.findall(finding))


def _contradictions(findings: list[str], praised: list[str]) -> list[str]:
    out = []
    for f in findings:
        fn = _norm(f)
        if any(_norm(p) in fn for p in praised if p.strip()):
            out.append(f)
    return out


def run_pair(pair_dir: Path, chat_model: str) -> dict:
    spec = json.loads((pair_dir / "pair.json").read_text(encoding="utf-8"))

    proposal_path = pair_dir / spec["proposal_file"]
    proposal_bytes = proposal_path.read_bytes()
    suffix = proposal_path.suffix

    call_context = ""
    if spec.get("call_context_file"):
        cc_path = pair_dir / spec["call_context_file"]
        if cc_path.suffix.lower() in {".txt", ".md"}:
            call_context = cc_path.read_text(encoding="utf-8", errors="replace")
        else:
            call_context = extract_text(cc_path.read_bytes(), cc_path.suffix)

    criteria = spec.get("criteria")
    if not criteria:
        criteria = (
            DEFAULT_DEVELOPMENT_CRITERIA
            if spec.get("action_type", "Research").lower() == "development"
            else DEFAULT_RESEARCH_CRITERIA
        )

    print(f"[{pair_dir.name}] evaluating {spec['acronym']} "
          f"({len(proposal_bytes) / 1024:.0f} KB, {len(criteria)} criteria)…",
          flush=True)

    result = evaluate_proposal(
        proposal_text="",
        criteria=criteria,
        action_type=spec.get("action_type", "Research"),
        call_id=f"benchmark-{pair_dir.name}",
        proposal_acronym=spec["acronym"],
        proposal_bytes=proposal_bytes,
        proposal_suffix=suffix,
        persist=False,
        chat_model=chat_model,
        call_context=call_context,
        progress_cb=lambda msg, frac: print(f"  {frac * 100:3.0f}% {msg}", flush=True),
        # Benchmarks measure scoring calibration against official ESR scores,
        # not the relevance gate — and their corpus is known-good proposals.
        # Bypass the gate so a false rejection can never abort a benchmark run.
        force_evaluate=True,
    )

    proposal_norm = _norm(extract_text(proposal_bytes, suffix))
    official: dict = spec.get("official", {})
    official_lookup = {k.strip().lower(): v for k, v in official.items()}

    rows = []
    contradiction_total = 0
    weak_total = 0
    weak_specific = 0
    for c in result["criteria"]:
        off = official_lookup.get(c["name"].strip().lower())
        delta = (c["score"] - off["score"]) if off and "score" in off else None
        praised = (off or {}).get("praised_aspects", [])
        contras = _contradictions(
            (c.get("weaknesses") or []) + (c.get("missing_evidence") or []), praised
        )
        contradiction_total += len(contras)
        weaknesses = c.get("weaknesses") or []
        weak_total += len(weaknesses)
        weak_specific += sum(
            1 for w in weaknesses if _finding_is_specific(w, proposal_norm)
        )
        rows.append({
            "criterion": c["name"],
            "ai_score": c["score"],
            "official_score": (off or {}).get("score"),
            "delta": delta,
            "band": c.get("band", ""),
            "confidence": c.get("confidence", ""),
            "input_coverage": c.get("input_coverage"),
            "full_document_input": c.get("full_document_input"),
            "n_weaknesses": len(weaknesses),
            "contradictions": contras,
        })

    weighted_delta = (
        result["weighted_total"] - spec["official_weighted_total"]
        if spec.get("official_weighted_total") is not None
        else None
    )
    return {
        "pair": pair_dir.name,
        "acronym": spec["acronym"],
        "chat_model": chat_model,
        "ai_weighted_total": result["weighted_total"],
        "official_weighted_total": spec.get("official_weighted_total"),
        "weighted_delta": weighted_delta,
        "contradiction_count": contradiction_total,
        "specificity_ratio": (weak_specific / weak_total) if weak_total else None,
        "criteria": rows,
        "raw_result": result,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", help="run a single pair by folder name")
    ap.add_argument("--chat-model", default=CHAT_MODEL)
    ap.add_argument("--max-delta", type=float, default=0.5,
                    help="regression threshold for per-criterion |Δ|")
    ap.add_argument("--max-total-delta", type=float, default=2.0,
                    help="regression threshold for weighted-total |Δ|")
    args = ap.parse_args()

    if not PAIRS_DIR.is_dir():
        print(f"No pairs directory at {PAIRS_DIR}. See benchmarks/README.md.")
        return 1
    pair_dirs = sorted(
        d for d in PAIRS_DIR.iterdir() if (d / "pair.json").is_file()
    )
    if args.pair:
        pair_dirs = [d for d in pair_dirs if d.name == args.pair]
    if not pair_dirs:
        print("No benchmark pairs found. See benchmarks/README.md to add one.")
        return 1

    reports = [run_pair(d, args.chat_model) for d in pair_dirs]

    failed = False
    print("\n=== Benchmark report ===")
    for rep in reports:
        print(f"\n{rep['pair']} ({rep['acronym']}, model {rep['chat_model']})")
        print(f"{'criterion':45s} {'AI':>5s} {'off':>5s} {'Δ':>6s} "
              f"{'conf':>7s} {'cover':>6s} {'#weak':>5s}")
        for r in rep["criteria"]:
            delta = f"{r['delta']:+.1f}" if r["delta"] is not None else "  n/a"
            off = f"{r['official_score']:.1f}" if r["official_score"] is not None else "n/a"
            cover = ("full" if r["full_document_input"]
                     else f"{(r['input_coverage'] or 0) * 100:.0f}%")
            print(f"{r['criterion'][:45]:45s} {r['ai_score']:5.1f} {off:>5s} "
                  f"{delta:>6s} {r['confidence']:>7s} {cover:>6s} "
                  f"{r['n_weaknesses']:5d}")
            if r["delta"] is not None and abs(r["delta"]) > args.max_delta:
                print(f"  !! per-criterion |Δ| > {args.max_delta}")
                failed = True
            for contra in r["contradictions"]:
                print(f"  !! CONTRADICTION (official ESR praised this): {contra}")
                failed = True
        if rep["weighted_delta"] is not None:
            print(f"weighted total: AI {rep['ai_weighted_total']} vs official "
                  f"{rep['official_weighted_total']} (Δ {rep['weighted_delta']:+.1f})")
            if abs(rep["weighted_delta"]) > args.max_total_delta:
                print(f"  !! weighted-total |Δ| > {args.max_total_delta}")
                failed = True
        if rep["specificity_ratio"] is not None:
            print(f"specificity ratio (weaknesses quoting the proposal): "
                  f"{rep['specificity_ratio'] * 100:.0f}%")
            if rep["specificity_ratio"] < 0.5:
                print("  !! specificity ratio < 50%")
                failed = True

    if len(reports) > 1:
        deltas = [abs(r["delta"]) for rep in reports for r in rep["criteria"]
                  if r["delta"] is not None]
        contras = sum(rep["contradiction_count"] for rep in reports)
        print(f"\n=== Aggregate over {len(reports)} pairs ===")
        if deltas:
            print(f"mean |Δ| {sum(deltas) / len(deltas):.2f}, max |Δ| {max(deltas):.1f}")
        print(f"total contradictions: {contras}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"{stamp}_{args.chat_model.replace(':', '_').replace('/', '_')}.json"
    out_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nFull report: {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
