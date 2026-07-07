"""Default EDF award-criteria sets.

Single source of truth shared by the frontend UI and the benchmark harness.
Descriptions double as retrieval queries: each comma/semicolon-separated
aspect issues its own embedding query (see pipeline._split_aspects), so keep
them enumerative.
"""

from __future__ import annotations

from typing import Any

DEFAULT_RESEARCH_CRITERIA: list[dict[str, Any]] = [
    {
        "name": "Excellence and potential of disruption",
        "description": (
            "Excellence of the overall concept; soundness of approach; "
            "compliance with topic objectives; advantage over existing "
            "defence products / disruption potential."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Innovation and technological development",
        "description": (
            "Innovation potential, ground-breaking concepts, integration of "
            "prior R&D while avoiding duplication, spin-off potential."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Competitiveness",
        "description": (
            "Competitive advantage vs. existing solutions, market growth "
            "potential across EU, IP strategy."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
    {
        "name": "EDTIB autonomy and security of supply",
        "description": (
            "Contribution to EDTIB autonomy, non-dependency on third "
            "countries, security of supply, alignment with CDP priorities."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
    {
        "name": "Creation of new cross-border cooperation",
        "description": (
            "New cross-border cooperation, especially SMEs and mid-caps with "
            "substantial participation."
        ),
        "weight": 2,
        "max_score": 5.0,
    },
    {
        "name": "Quality and efficiency of implementation",
        "description": (
            "Work plan structure, milestones / deliverables, management, risk "
            "treatment, allocation of work across the consortium."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
]

DEFAULT_DEVELOPMENT_CRITERIA: list[dict[str, Any]] = DEFAULT_RESEARCH_CRITERIA + [
    {
        "name": "Contribution to industrial policy objectives",
        "description": (
            "Contribution to industrial-policy objectives of the EDF, "
            "including increase of production capacity in the EU defence industry."
        ),
        "weight": 1,
        "max_score": 5.0,
    },
]
