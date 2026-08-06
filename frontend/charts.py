"""Altair chart builders for the evaluator UI.

Every chart is drawn in the IKH palette (navy / amber on white), with the
chrome stripped back: no gridlines unless they carry information, no
legends where a direct label does the job, tabular figures, hairline axes.

Altair ships with Streamlit, so this adds no dependency. Charts are
returned (not rendered) so callers control layout and width.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

from theme import (
    AMBER,
    AMBER_TINT,
    INK_SOFT,
    LINE,
    LINE_STRONG,
    MUTED,
    NAVY,
    NEGATIVE,
    OUTCOME_COLORS,
    SURFACE_ALT,
    score_color,
)

FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def _style(chart: alt.Chart | alt.LayerChart) -> alt.Chart | alt.LayerChart:
    """Shared chrome removal + brand typography."""
    return (
        chart.configure_view(stroke=None, fill=None)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=MUTED,
            titleColor=MUTED,
            labelFontSize=10,
            titleFontSize=10,
            titleFontWeight=600,
            labelFontWeight=500,
            domainColor=LINE_STRONG,
            tickColor=LINE_STRONG,
            gridColor=LINE,
            gridDash=[2, 3],
            titlePadding=8,
        )
        .configure_legend(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=INK_SOFT,
            titleColor=MUTED,
            labelFontSize=10,
            titleFontSize=10,
            symbolType="square",
            orient="bottom",
            direction="horizontal",
            titleFontWeight=600,
        )
        .configure_text(font=FONT)
        .configure_title(font=FONT, color=NAVY, fontSize=12, fontWeight=600, anchor="start")
    )


def _short(name: str, limit: int = 34) -> str:
    name = str(name or "—")
    return name if len(name) <= limit else name[: limit - 1].rstrip() + "…"


# The score and contribution charts sit side by side, so their row bands have
# to line up exactly: same height per row, same axis config, and neither may
# drop a label (Vega hides overlapping tick labels by default, which silently
# left two of six criteria unlabelled).
def _bar_height(n_rows: int) -> int:
    return max(140, 34 * n_rows + 14)


def _bar_axis(order: list[str]) -> alt.Y:
    return alt.Y(
        "name:N", sort=order, title=None,
        axis=alt.Axis(labelLimit=240, labelFontSize=10.5, labelColor=INK_SOFT,
                      domain=False, ticks=False, labelPadding=8, labelOverlap=False),
    )


# ══════════════════════════════════════════════════════════════════════════
# Single-run charts
# ══════════════════════════════════════════════════════════════════════════


def criterion_scores(criteria: list[dict[str, Any]]) -> alt.LayerChart | None:
    """Score-out-of-max per criterion: a light track with a filled bar on top,
    coloured by how much of the maximum was achieved, value labelled inline."""
    rows = []
    for c in criteria or []:
        max_score = float(c.get("max_score") or 5.0) or 5.0
        score = float(c.get("score") or 0.0)
        frac = max(0.0, min(1.0, score / max_score))
        rows.append(
            {
                "name": _short(c.get("name")),
                "full": str(c.get("name") or "—"),
                "score": score,
                "max_score": max_score,
                "frac": frac,
                "pct": f"{frac * 100:.0f}%",
                "label": f"{score:g} / {max_score:g}",
                "band": (c.get("band") or "—").title(),
                "color": score_color(frac),
                "weight": float(c.get("weight") or 0.0),
            }
        )
    if not rows:
        return None

    df = pd.DataFrame(rows)
    order = df["name"].tolist()
    height = _bar_height(len(df))
    y = _bar_axis(order)

    track = (
        alt.Chart(df)
        .mark_bar(color=SURFACE_ALT, cornerRadius=3, height=13)
        .encode(x=alt.X("max_score:Q", title=None,
                        axis=alt.Axis(grid=False, labels=False, ticks=False, domain=False)),
                y=y)
    )
    bar = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, height=13)
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, float(df["max_score"].max())])),
            y=y,
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("full:N", title="Criterion"),
                alt.Tooltip("label:N", title="Score"),
                alt.Tooltip("band:N", title="Band"),
                alt.Tooltip("weight:Q", title="Weight", format="g"),
                alt.Tooltip("pct:N", title="Of max"),
            ],
        )
    )
    label = (
        alt.Chart(df)
        .mark_text(align="left", dx=7, fontSize=10, fontWeight=600, color=INK_SOFT)
        .encode(x=alt.X("max_score:Q"), y=y, text="label:N")
    )
    return _style(
        alt.layer(track, bar, label).properties(height=height, padding={"left": 0, "right": 34})
    )


def contribution_bars(criteria: list[dict[str, Any]]) -> alt.LayerChart | None:
    """What each criterion actually contributes to the weighted total
    (score × weight) against the most it could have contributed. Makes the
    effect of the weights visible, which the raw scores hide."""
    rows = []
    for c in criteria or []:
        weight = float(c.get("weight") or 0.0)
        max_score = float(c.get("max_score") or 5.0) or 5.0
        score = float(c.get("score") or 0.0)
        got, cap = score * weight, max_score * weight
        rows.append(
            {
                "name": _short(c.get("name")),
                "full": str(c.get("name") or "—"),
                "got": got,
                "cap": cap,
                "lost": cap - got,
                "label": f"{got:g}",
                "share": f"{(got / cap * 100) if cap else 0:.0f}% of ×{weight:g} cap",
            }
        )
    if not rows:
        return None

    df = pd.DataFrame(rows)
    order = df["name"].tolist()
    height = _bar_height(len(df))
    y = _bar_axis(order)

    cap = (
        alt.Chart(df)
        .mark_bar(color=SURFACE_ALT, cornerRadius=3, height=13)
        .encode(x=alt.X("cap:Q", title=None,
                        axis=alt.Axis(grid=True, tickCount=5, tickMinStep=1,
                                      domain=False, ticks=False)), y=y)
    )
    got = (
        alt.Chart(df)
        .mark_bar(color=NAVY, cornerRadius=3, height=13)
        .encode(
            x=alt.X("got:Q"),
            y=y,
            tooltip=[
                alt.Tooltip("full:N", title="Criterion"),
                alt.Tooltip("got:Q", title="Contributed", format="g"),
                alt.Tooltip("cap:Q", title="Maximum", format="g"),
                alt.Tooltip("share:N", title="Share"),
            ],
        )
    )
    label = (
        alt.Chart(df)
        .mark_text(align="left", dx=7, fontSize=10, fontWeight=600, color=INK_SOFT)
        .encode(x=alt.X("cap:Q"), y=y, text="label:N")
    )
    return _style(
        alt.layer(cap, got, label).properties(height=height, padding={"left": 0, "right": 30})
    )


def overall_donut(weighted: float, max_weighted: float, passed: bool,
                  size: int = 168) -> alt.LayerChart:
    """Achievement donut: filled arc for the share of the maximum reached,
    percentage in the middle. Amber on a pass, red on a fail — the only
    place colour carries the verdict."""
    max_weighted = float(max_weighted or 1.0) or 1.0
    frac = max(0.0, min(1.0, float(weighted) / max_weighted))
    fill = AMBER if passed else NEGATIVE
    df = pd.DataFrame(
        [
            {"k": "achieved", "v": frac, "c": fill, "o": 0},
            {"k": "remaining", "v": 1 - frac, "c": SURFACE_ALT, "o": 1},
        ]
    )
    arc = (
        alt.Chart(df)
        .mark_arc(innerRadius=size * 0.30, outerRadius=size * 0.42, cornerRadius=2, stroke=None)
        .encode(
            theta=alt.Theta("v:Q", stack=True, sort=alt.EncodingSortField("o")),
            order=alt.Order("o:Q"),
            color=alt.Color("c:N", scale=None, legend=None),
        )
    )
    centre = (
        alt.Chart(pd.DataFrame([{"t": f"{frac * 100:.0f}%"}]))
        .mark_text(fontSize=size * 0.20, fontWeight=700, color=NAVY, dy=-4)
        .encode(text="t:N")
    )
    sub = (
        alt.Chart(pd.DataFrame([{"t": "of maximum"}]))
        .mark_text(fontSize=9.5, color=MUTED, dy=size * 0.15, fontWeight=500)
        .encode(text="t:N")
    )
    return _style(
        alt.layer(arc, centre, sub).properties(width=size, height=size)
    )


def band_profile(criteria: list[dict[str, Any]]) -> alt.Chart | None:
    """How the criterion scores are spread across the 0–max range — a quick
    read on whether a proposal is uniformly mediocre or spiky."""
    rows = [
        {
            "name": _short(c.get("name"), 26),
            "score": float(c.get("score") or 0.0),
            "max_score": float(c.get("max_score") or 5.0) or 5.0,
        }
        for c in criteria or []
    ]
    if len(rows) < 3:
        return None
    df = pd.DataFrame(rows)
    df["frac"] = df["score"] / df["max_score"]
    base = alt.Chart(df).encode(
        x=alt.X("name:N", sort=df["name"].tolist(), title=None,
                axis=alt.Axis(labelAngle=-35, labelLimit=110, labelFontSize=9.5,
                              domain=False, ticks=False)),
        y=alt.Y("frac:Q", title="Share of max",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, grid=True, domain=False, ticks=False)),
    )
    area = base.mark_area(
        line={"color": AMBER, "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=AMBER_TINT, offset=0),
                   alt.GradientStop(color=AMBER, offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        opacity=0.35,
        interpolate="monotone",
    )
    dots = base.mark_point(filled=True, size=52, color=NAVY).encode(
        tooltip=[alt.Tooltip("name:N", title="Criterion"),
                 alt.Tooltip("score:Q", title="Score", format="g"),
                 alt.Tooltip("frac:Q", title="Of max", format=".0%")]
    )
    return _style(alt.layer(area, dots).properties(height=190))


# ══════════════════════════════════════════════════════════════════════════
# History / portfolio charts
# ══════════════════════════════════════════════════════════════════════════


def history_trend(runs: list[dict[str, Any]]) -> alt.LayerChart | None:
    """Score as a share of maximum over time, with the portfolio mean as a
    reference rule. Oldest → newest left to right."""
    rows = []
    for r in runs or []:
        max_w = float(r.get("max_weighted_total") or 0.0)
        if not max_w:
            continue
        rows.append(
            {
                "when": str(r.get("generated_at") or "")[:19].replace("T", " "),
                "acr": str(r.get("proposal_acronym") or "—"),
                "frac": float(r.get("weighted_total") or 0.0) / max_w,
                "score": float(r.get("weighted_total") or 0.0),
                "max": max_w,
                "band": str(r.get("predicted_band") or "—"),
            }
        )
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows)
    df["when"] = pd.to_datetime(df["when"], errors="coerce", format="mixed")
    df = df.dropna(subset=["when"]).sort_values("when")
    if len(df) < 2:
        return None

    base = alt.Chart(df).encode(
        x=alt.X("when:T", title=None,
                axis=alt.Axis(format="%d %b", tickCount=6, grid=False, domain=False, ticks=False)),
        y=alt.Y("frac:Q", title="Share of max",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=5, grid=True, domain=False, ticks=False)),
    )
    area = base.mark_area(
        line={"color": AMBER, "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#FFFFFF", offset=0),
                   alt.GradientStop(color=AMBER_TINT, offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
        interpolate="monotone",
    )
    dots = base.mark_point(filled=True, size=58, color=NAVY, stroke="#fff", strokeWidth=1.2).encode(
        tooltip=[
            alt.Tooltip("acr:N", title="Proposal"),
            alt.Tooltip("when:T", title="Evaluated", format="%d %b %Y %H:%M"),
            alt.Tooltip("score:Q", title="Weighted", format="g"),
            alt.Tooltip("frac:Q", title="Of max", format=".0%"),
            alt.Tooltip("band:N", title="Band"),
        ]
    )
    mean_rule = (
        alt.Chart(df)
        .mark_rule(color=LINE_STRONG, strokeDash=[4, 4], strokeWidth=1)
        .encode(y="mean(frac):Q")
    )
    return _style(alt.layer(area, mean_rule, dots).properties(height=200))


def band_mix(runs: list[dict[str, Any]]) -> alt.LayerChart | None:
    """Count of runs per predicted outcome band."""
    counts: dict[str, int] = {}
    for r in runs or []:
        counts[str(r.get("predicted_band") or "—")] = counts.get(
            str(r.get("predicted_band") or "—"), 0
        ) + 1
    if not counts:
        return None
    order = ["Likely funded", "Reserve list / borderline", "Likely rejected"]
    rows = [
        {"band": b, "n": counts[b], "c": OUTCOME_COLORS.get(b, MUTED)}
        for b in order
        if b in counts
    ] + [
        {"band": b, "n": n, "c": MUTED}
        for b, n in counts.items()
        if b not in order
    ]
    df = pd.DataFrame(rows)
    y = alt.Y("band:N", sort=df["band"].tolist(), title=None,
              axis=alt.Axis(labelLimit=200, labelFontSize=10.5, labelColor=INK_SOFT,
                            domain=False, ticks=False, labelPadding=8))
    bar = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, height=16)
        .encode(
            x=alt.X("n:Q", title=None,
                    axis=alt.Axis(tickMinStep=1, grid=True, domain=False, ticks=False)),
            y=y,
            color=alt.Color("c:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("n:Q", title="Runs")],
        )
    )
    label = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=10, fontWeight=700, color=INK_SOFT)
        .encode(x="n:Q", y=y, text="n:Q")
    )
    return _style(
        alt.layer(bar, label).properties(height=max(90, 34 * len(df)),
                                         padding={"left": 0, "right": 20})
    )


def criterion_averages(stats: list[dict[str, Any]]) -> alt.LayerChart | None:
    """Mean share-of-max per criterion across the whole history — shows which
    criteria the portfolio consistently underperforms on.

    `stats` items: {"name": str, "mean": 0–1 float, "n": int}
    """
    rows = [
        {
            "name": _short(s["name"], 30),
            "full": s["name"],
            "mean": float(s["mean"]),
            "n": int(s.get("n") or 0),
            "label": f"{float(s['mean']) * 100:.0f}%",
            "c": score_color(float(s["mean"])),
        }
        for s in stats or []
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    y = _bar_axis(df["name"].tolist())
    track = (
        alt.Chart(df)
        .mark_bar(color=SURFACE_ALT, cornerRadius=3, height=13)
        .encode(x=alt.X("max:Q", title=None,
                        axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
                y=y)
        .transform_calculate(max="1")
    )
    bar = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3, height=13)
        .encode(
            x=alt.X("mean:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            y=y,
            color=alt.Color("c:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("full:N", title="Criterion"),
                alt.Tooltip("label:N", title="Mean of max"),
                alt.Tooltip("n:Q", title="Runs"),
            ],
        )
    )
    label = (
        alt.Chart(df)
        .mark_text(align="left", dx=7, fontSize=10, fontWeight=600, color=INK_SOFT)
        .encode(x=alt.X("max:Q", scale=alt.Scale(domain=[0, 1])), y=y, text="label:N")
        .transform_calculate(max="1")
    )
    return _style(
        alt.layer(track, bar, label).properties(height=_bar_height(len(df)),
                                                padding={"left": 0, "right": 34})
    )


def score_spread(runs: list[dict[str, Any]]) -> alt.Chart | None:
    """Distribution of share-of-max across all stored runs."""
    fracs = []
    for r in runs or []:
        max_w = float(r.get("max_weighted_total") or 0.0)
        if max_w:
            fracs.append(float(r.get("weighted_total") or 0.0) / max_w)
    if len(fracs) < 4:
        return None
    df = pd.DataFrame({"frac": fracs})
    return _style(
        alt.Chart(df)
        .mark_bar(color=NAVY, cornerRadius=2, opacity=0.9)
        .encode(
            x=alt.X("frac:Q", bin=alt.Bin(maxbins=12), title="Share of max",
                    axis=alt.Axis(format="%", grid=False, domain=False, ticks=False)),
            y=alt.Y("count():Q", title="Runs",
                    axis=alt.Axis(tickMinStep=1, grid=True, domain=False, ticks=False)),
            tooltip=[alt.Tooltip("count():Q", title="Runs")],
        )
        .properties(height=190)
    )
