"""Synthetic direction tests — no official ESR needed.

Feeds a deliberately excellent and a deliberately deficient implementation
section (fictional content) to the same criterion prompt and asserts the
score direction. Guards against critique bias (excellent text dragged down
by invented weaknesses) and its mirror image, praise bias (deficient text
scored leniently).

Hard failures (exit 1):
  - excellent section scored < 4.0, or given > 2 weaknesses
  - deficient section scored > 3.0, or given < 2 weaknesses
  - score gap < 1.0
Soft warnings: excellent < 4.5, excellent weaknesses > 0, gap < 1.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import CHAT_MODEL  # noqa: E402
from backend.evaluator import evaluate_criterion  # noqa: E402

SYNTH = ROOT / "benchmarks" / "synthetic"

CRITERION_NAME = "Quality and efficiency of implementation"
CRITERION_DESC = (
    "Work plan structure, milestones / deliverables, management, risk "
    "treatment, allocation of work across the consortium."
)


def _run(label: str, path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    print(f"\n--- {label}: {path.name} ({len(text)} chars) ---", flush=True)
    res = evaluate_criterion(
        proposal_text=text,
        criterion_name=CRITERION_NAME,
        criterion_description=CRITERION_DESC,
        weight=1.0,
        action_type="Research",
        chat_model=CHAT_MODEL,
        full_document=True,
        coverage_ratio=1.0,
    )
    print(json.dumps(
        {k: res[k] for k in
         ("band", "score", "band_rationale", "confidence",
          "significant_weakness_count", "minor_weakness_count",
          "cleared_probe_count", "severity_envelope",
          "strengths", "weaknesses",
          "missing_evidence", "input_gaps", "evaluator_comment")},
        indent=2, ensure_ascii=False,
    ))
    return res


def main() -> int:
    excellent = _run("EXCELLENT", SYNTH / "excellent_implementation.txt")
    deficient = _run("DEFICIENT", SYNTH / "deficient_implementation.txt")

    failures: list[str] = []
    warnings: list[str] = []

    if excellent["score"] < 4.0:
        failures.append(f"excellent section scored {excellent['score']} (< 4.0)")
    elif excellent["score"] < 4.5:
        warnings.append(f"excellent section scored {excellent['score']} (< 4.5)")
    n_weak_exc = len(excellent["weaknesses"])
    if n_weak_exc > 2:
        failures.append(f"excellent section given {n_weak_exc} weaknesses (> 2)")
    elif n_weak_exc > 0:
        warnings.append(f"excellent section given {n_weak_exc} weakness(es)")

    if deficient["score"] > 3.0:
        failures.append(f"deficient section scored {deficient['score']} (> 3.0)")
    if len(deficient["weaknesses"]) < 2:
        failures.append(
            f"deficient section given only {len(deficient['weaknesses'])} weaknesses"
        )

    gap = excellent["score"] - deficient["score"]
    if gap < 1.0:
        failures.append(f"score gap {gap:.1f} (< 1.0)")
    elif gap < 1.5:
        warnings.append(f"score gap {gap:.1f} (< 1.5)")

    print("\n=== Synthetic direction test ===")
    print(f"excellent: {excellent['score']} ({excellent.get('band')}), "
          f"{n_weak_exc} weaknesses | deficient: {deficient['score']} "
          f"({deficient.get('band')}), {len(deficient['weaknesses'])} weaknesses")
    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    if not failures:
        print("PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
