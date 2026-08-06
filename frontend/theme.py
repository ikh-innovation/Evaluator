"""Brand design system for the EDF Proposal Evaluator.

Single source of truth for colour and for the handful of small HTML
components that Streamlit's own widgets can't express cleanly.

The palette is taken straight from the IKH logo (`ikh_logo.png`): navy
`#202039` and amber `#F08003` on white. Nothing else is decorative —
green/red appear only on a pass/fail verdict, grey carries everything
structural.

Note: the stylesheet below is a *literal* string with the hex values
written out, deliberately not interpolated — CSS is full of `{}` and
f-string/.format escaping over 600 lines is a reliable source of bugs.
When you change a token here, change it in the CSS block too.
"""

from __future__ import annotations

import io
from functools import lru_cache
from html import escape as _h
from pathlib import Path

LOGO_PATH = Path(__file__).resolve().parent / "ikh_logo.png"

# The shipped PNG is a 945×945 square with generous whitespace and the
# "crafting empowering technologies" tagline underneath. Neither crop is
# useful at header size, so trim to the content bounding boxes (measured
# once off the actual file): the bare IKH mark, and mark + tagline.
_CROP_MARK = (223, 199, 690, 444)
_CROP_FULL = (223, 199, 690, 758)

# ── Tokens ────────────────────────────────────────────────────────────────
NAVY = "#202039"          # brand ink — text, dark fills
NAVY_DEEP = "#16162A"     # pressed / gradient end
NAVY_MID = "#2E2E4D"      # hover on navy fills
INK_SOFT = "#4A4A66"      # secondary text
MUTED = "#71718C"         # captions, meta
LINE = "#E8E9EF"          # hairline borders
LINE_STRONG = "#D5D7E0"   # emphasised borders
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F6F7FA"   # subtle fills, zebra rows

AMBER = "#F08003"         # brand accent
AMBER_DEEP = "#C96A02"    # hover
AMBER_TINT = "#FFF3E3"    # accent wash
AMBER_LINE = "#FFDCB0"    # accent border

POSITIVE = "#12715A"
POSITIVE_TINT = "#E8F3EF"
NEGATIVE = "#B02A1F"
NEGATIVE_TINT = "#FBECEA"

# Ordered scale used for criterion bands (fails → excellent).
BAND_SCALE = ["#B02A1F", "#C4592B", "#E0902F", "#F08003", "#5B7A5A", "#12715A"]

# Outcome bands produced by backend.evaluator.predict_band.
OUTCOME_COLORS = {
    "Likely funded": POSITIVE,
    "Reserve list / borderline": AMBER,
    "Likely rejected": NEGATIVE,
}


_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════
   EDF Proposal Evaluator — IKH brand theme
   navy #202039 · amber #F08003 · white. Sleek, flat, hairline borders,
   4–6px radii, no gradients or shadows beyond a 1px lift.
   ══════════════════════════════════════════════════════════════════════ */

:root {
    --edf-navy: #202039;
    --edf-navy-mid: #2E2E4D;
    --edf-ink-soft: #4A4A66;
    --edf-muted: #71718C;
    --edf-line: #E8E9EF;
    --edf-line-strong: #D5D7E0;
    --edf-alt: #F6F7FA;
    --edf-amber: #F08003;
    --edf-amber-deep: #C96A02;
    --edf-amber-tint: #FFF3E3;
    --edf-amber-line: #FFDCB0;
    --edf-radius: 6px;
    --edf-radius-sm: 4px;
    --edf-sans: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
    --edf-mono: "SFMono-Regular", ui-monospace, "JetBrains Mono", Menlo,
                 Consolas, monospace;
}

/* ── Canvas ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
    font-family: var(--edf-sans);
    color: var(--edf-navy);
}
[data-testid="stAppViewContainer"] > .main { background: #FFFFFF; }

/* Streamlit's own chrome: keep it, mute it. The floating status widget and
   the "hosted with Streamlit" footer are noise in an internal tool. */
/* Streamlit's header is position:fixed and still owns the sidebar-expand
   control, so it stays — but it is flattened to a plain white scroll mask.
   The main container's top padding must clear its height (2.6rem) or the
   app bar renders underneath it. */
[data-testid="stHeader"] {
    background: rgba(255,255,255,.94) !important;
    backdrop-filter: saturate(180%) blur(6px);
    border-bottom: none !important;
    height: 2.6rem;
}
[data-testid="stStatusWidget"], footer, #MainMenu { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { right: .4rem; }

/* Tighter, wider content column — this is a data-dense tool. */
[data-testid="stMainBlockContainer"], .block-container {
    padding: 3.5rem 2.4rem 5.5rem 2.4rem;
    max-width: 1600px;
}

/* ── Type scale ─────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--edf-sans) !important;
    color: var(--edf-navy) !important;
    letter-spacing: -.01em;
    font-weight: 650;
}
h1 { font-size: 1.55rem !important; }
h2 { font-size: 1.2rem !important;  padding-top: .2rem !important; }
h3 { font-size: 1.02rem !important; }
h4, h5 { font-size: .93rem !important; }
p, li, label, .stMarkdown { font-family: var(--edf-sans); }
code, kbd, .stCode, [data-testid="stMarkdownContainer"] code {
    font-family: var(--edf-mono) !important;
    background: var(--edf-alt) !important;
    border: 1px solid var(--edf-line);
    border-radius: 3px;
    color: var(--edf-ink-soft) !important;
    font-size: .82em !important;
    padding: .06em .34em !important;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--edf-muted) !important;
    font-size: .8rem !important;
}
hr, [data-testid="stDivider"] hr {
    border-color: var(--edf-line) !important;
    margin: 1rem 0 !important;
}

/* ── App header bar ─────────────────────────────────────────────────────
   The logo + title + subtitle are ONE inline-flex lockup (see theme.appbar)
   rather than separate Streamlit columns: columns each carry their own
   wrapper margins and baseline, which left the subtitle and the account
   chips visibly out of alignment. Only the chips and the sign-out button
   remain as sibling columns, both vertically centred against the lockup. */
[data-testid="stHorizontalBlock"]:has(.edf-appbar-marker) {
    border-bottom: 2px solid var(--edf-amber);
    padding-bottom: .6rem;
    margin-bottom: .4rem;
    align-items: center !important;
    gap: .5rem !important;
}
[data-testid="stHorizontalBlock"]:has(.edf-appbar-marker) [data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.edf-appbar-marker) [data-testid="stColumn"] > div {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.edf-appbar {
    display: flex;
    align-items: center;
    gap: .9rem;
    min-width: 0;
}
.edf-appbar img {
    display: block;
    width: auto;
    flex: 0 0 auto;
}
.edf-titles { min-width: 0; }
.edf-apptitle {
    font-size: 1.3rem;
    font-weight: 680;
    letter-spacing: -.015em;
    color: var(--edf-navy);
    line-height: 1.15;
}
.edf-appsub {
    font-size: .76rem;
    color: var(--edf-muted);
    margin-top: .2rem;
    letter-spacing: .01em;
}
.edf-appsub b { color: var(--edf-ink-soft); font-weight: 600; }

/* Account chips: right-aligned, single row, vertically centred on the bar. */
.edf-appmeta {
    display: flex;
    gap: .35rem;
    justify-content: flex-end;
    align-items: center;
    flex-wrap: wrap;
    row-gap: .3rem;
}
[data-testid="stColumn"]:has(.edf-logout-slot) button[data-testid="stBaseButton-secondary"] {
    height: 1.7rem !important; min-height: 1.7rem !important;
    font-size: .74rem !important; padding: 0 .6rem !important;
    color: var(--edf-muted) !important;
}
[data-testid="stColumn"]:has(.edf-logout-slot) button[data-testid="stBaseButton-secondary"]:hover {
    color: var(--edf-navy) !important; border-color: var(--edf-navy) !important;
}

/* ── Chips / pills ──────────────────────────────────────────────────── */
.edf-chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    height: 1.5rem;
    padding: 0 .55rem;
    border: 1px solid var(--edf-line-strong);
    border-radius: 999px;
    background: #FFFFFF;
    color: var(--edf-ink-soft);
    font-size: .74rem;
    font-weight: 550;
    white-space: nowrap;
    line-height: 1;
}
.edf-chip .k {
    color: var(--edf-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-size: .66rem;
}
.edf-chip.solid   { background: var(--edf-navy); border-color: var(--edf-navy); color: #fff; }
.edf-chip.accent  { background: var(--edf-amber-tint); border-color: var(--edf-amber-line); color: var(--edf-amber-deep); }
.edf-chip.pass    { background: #E8F3EF; border-color: #BFDED4; color: #12715A; }
.edf-chip.fail    { background: #FBECEA; border-color: #F0CFCB; color: #B02A1F; }
.edf-chip .dot    { width: 6px; height: 6px; border-radius: 50%; background: currentColor; flex: 0 0 auto; }

/* ── Section header (label + optional hint, amber tick on the left) ─── */
.edf-section {
    display: flex;
    align-items: baseline;
    gap: .6rem;
    row-gap: .1rem;
    flex-wrap: wrap;
    margin: 1.35rem 0 .6rem 0;
    padding-left: .6rem;
    border-left: 3px solid var(--edf-amber);
}
/* The label must never wrap mid-phrase — the hint after it may. */
.edf-section .t {
    font-size: .82rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: .09em;
    color: var(--edf-navy);
    white-space: nowrap;
}
.edf-section .h { font-size: .78rem; color: var(--edf-muted); min-width: 0; }
.edf-step {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.15rem; height: 1.15rem; border-radius: 50%;
    background: var(--edf-navy); color: #fff;
    font-size: .68rem; font-weight: 700; margin-right: .1rem;
}

/* ── Metric tiles ───────────────────────────────────────────────────── */
.edf-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: .6rem;
}
.edf-tile {
    border: 1px solid var(--edf-line);
    border-radius: var(--edf-radius);
    background: #FFFFFF;
    padding: .7rem .85rem;
    position: relative;
    overflow: hidden;
}
.edf-tile::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--edf-line-strong);
}
.edf-tile.accent::before  { background: var(--edf-amber); }
.edf-tile.navy::before    { background: var(--edf-navy); }
.edf-tile.pass::before    { background: #12715A; }
.edf-tile.fail::before    { background: #B02A1F; }
.edf-tile .k {
    font-size: .68rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: var(--edf-muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.edf-tile .v {
    font-size: 1.5rem; font-weight: 660; line-height: 1.15;
    color: var(--edf-navy); margin-top: .18rem;
    font-variant-numeric: tabular-nums;
}
.edf-tile .v small { font-size: .8rem; font-weight: 500; color: var(--edf-muted); }
.edf-tile.pass .v { color: #12715A; }
.edf-tile.fail .v { color: #B02A1F; }
.edf-tile .s { font-size: .72rem; color: var(--edf-muted); margin-top: .15rem; }

/* ── Meter (thin score bar used inside cards and history rows) ──────── */
.edf-meter {
    height: 5px; border-radius: 3px; background: var(--edf-alt);
    overflow: hidden; border: 1px solid var(--edf-line);
}
.edf-meter > i { display: block; height: 100%; background: var(--edf-amber); border-radius: 3px; }
.edf-meter.navy > i { background: var(--edf-navy); }

/* ── Cards: st.container(border=True) ───────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
    border-radius: var(--edf-radius);
}
div[data-testid="stVerticalBlockBorderWrapper"][style*="border"] {
    border-color: var(--edf-line) !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: .1rem;
    border-bottom: 1px solid var(--edf-line);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    height: 2.5rem;
    padding: 0 1rem;
    background: transparent !important;
    color: var(--edf-muted) !important;
    font-size: .86rem;
    font-weight: 600;
    letter-spacing: .01em;
    border-radius: 0;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--edf-navy) !important; }
.stTabs [aria-selected="true"] { color: var(--edf-navy) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--edf-amber) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ── Buttons ────────────────────────────────────────────────────────── */
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    border-radius: var(--edf-radius-sm) !important;
    border: 1px solid var(--edf-line-strong) !important;
    background: #FFFFFF !important;
    color: var(--edf-navy) !important;
    font-family: var(--edf-sans) !important;
    font-size: .84rem !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: background-color .12s, border-color .12s, color .12s;
}
button[data-testid="stBaseButton-secondary"]:hover,
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    border-color: var(--edf-navy) !important;
    background: var(--edf-alt) !important;
    color: var(--edf-navy) !important;
}
button[data-testid="stBaseButton-secondary"]:disabled {
    background: var(--edf-alt) !important;
    border-color: var(--edf-line) !important;
    color: #A6A6B8 !important;
}
/* Destructive / primary — reserved for Wipe history, Delete run. */
button[data-testid="stBaseButton-primary"] {
    background: #B02A1F !important;
    border: 1px solid #B02A1F !important;
    border-radius: var(--edf-radius-sm) !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
button[data-testid="stBaseButton-primary"]:hover {
    background: #8E2118 !important; border-color: #8E2118 !important;
}
button[data-testid="stBaseButton-primary"]:disabled {
    background: var(--edf-alt) !important; border-color: var(--edf-line) !important;
    color: #A6A6B8 !important;
}
/* The one true CTA ("Queue evaluation", "Sign in") — amber, marked by an
   adjacent invisible span since Streamlit gives buttons no stable CSS hook.
   Form submit buttons carry a different testid, hence both selectors. */
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondaryFormSubmit"],
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"] {
    background: var(--edf-amber) !important;
    border-color: var(--edf-amber) !important;
    color: #fff !important;
    height: 2.6rem;
    font-size: .9rem !important;
    letter-spacing: .01em;
}
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--edf-amber-deep) !important;
    border-color: var(--edf-amber-deep) !important;
}
[data-testid="stColumn"]:has(.edf-cta-marker) button[data-testid="stBaseButton-secondary"]:disabled {
    background: var(--edf-alt) !important;
    border-color: var(--edf-line) !important;
    color: #A6A6B8 !important;
}
/* Quiet icon/ghost buttons (dismiss ✕, row actions). */
[data-testid="stColumn"]:has(.edf-ghost-marker) button[data-testid="stBaseButton-secondary"] {
    border-color: transparent !important;
    background: transparent !important;
    color: var(--edf-muted) !important;
}
[data-testid="stColumn"]:has(.edf-ghost-marker) button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--edf-alt) !important;
    color: var(--edf-navy) !important;
}
[data-testid="stDownloadButton"] button { width: 100%; }

/* ── Inputs ─────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div {
    border-radius: var(--edf-radius-sm) !important;
    border-color: var(--edf-line-strong) !important;
    background: #FFFFFF !important;
    color: var(--edf-navy) !important;
    font-family: var(--edf-sans) !important;
    font-size: .86rem !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--edf-amber) !important;
    box-shadow: 0 0 0 2px rgba(240,128,3,.16) !important;
}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
    font-size: .78rem !important;
    font-weight: 600 !important;
    color: var(--edf-ink-soft) !important;
    letter-spacing: .01em;
}
[data-testid="stNumberInput"] button { border-color: var(--edf-line) !important; }

/* File uploader — flatter, amber on hover/drag. */
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--edf-line-strong) !important;
    border-radius: var(--edf-radius) !important;
    background: var(--edf-alt) !important;
    padding: 1rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--edf-amber) !important;
    background: var(--edf-amber-tint) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    border-radius: var(--edf-radius-sm) !important;
    border: 1px solid var(--edf-line-strong) !important;
    background: #fff !important;
    color: var(--edf-navy) !important;
}
[data-testid="stFileUploaderFile"] { background: #fff; border: 1px solid var(--edf-line); border-radius: var(--edf-radius-sm); padding: .4rem .55rem; }

/* Model selectors are locked (disabled) so the UI still shows which models
   are in use, but the dropdown arrow — a clickability affordance — is
   removed entirely rather than just greyed out. */
[data-testid="stHorizontalBlock"]:has(.edf-model-lock-marker) [data-testid="stSelectbox"] svg {
    display: none !important;
}

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] details {
    border: 1px solid var(--edf-line) !important;
    border-radius: var(--edf-radius) !important;
    background: #FFFFFF;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    background: var(--edf-alt);
    padding: .5rem .75rem !important;
    font-size: .84rem;
    font-weight: 600;
    color: var(--edf-navy);
}
[data-testid="stExpander"] summary:hover { background: var(--edf-amber-tint); }
[data-testid="stExpander"] summary p { font-size: .84rem !important; font-weight: 600 !important; }
[data-testid="stExpander"] details[open] summary { border-bottom: 1px solid var(--edf-line); }

/* ── Progress bars ──────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div > div { background: var(--edf-alt) !important; height: 6px !important; }
[data-testid="stProgress"] > div > div > div > div {
    background: var(--edf-amber) !important; border-radius: 3px;
}

/* ── Alerts / notifications ─────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--edf-radius) !important;
    border: 1px solid var(--edf-line) !important;
    border-left-width: 3px !important;
    box-shadow: none !important;
    font-size: .84rem;
}
[data-testid="stAlertContentInfo"], [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
    background: var(--edf-alt) !important; border-left-color: var(--edf-navy) !important; color: var(--edf-navy) !important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
    background: var(--edf-amber-tint) !important; border-left-color: var(--edf-amber) !important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
    background: #FBECEA !important; border-left-color: #B02A1F !important;
}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background: #E8F3EF !important; border-left-color: #12715A !important;
}
[data-testid="stToast"] {
    border-radius: var(--edf-radius) !important;
    border: 1px solid var(--edf-line-strong) !important;
    background: #fff !important;
    color: var(--edf-navy) !important;
}

/* ── Dialogs ────────────────────────────────────────────────────────── */
[data-testid="stDialog"] > div > div {
    border-radius: var(--edf-radius) !important;
    border: 1px solid var(--edf-line-strong);
    border-top: 3px solid var(--edf-amber);
}

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid var(--edf-line);
}
[data-testid="stSidebarContent"] { padding-top: .6rem; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .5rem; }
[data-testid="stSidebar"] h2 { font-size: .82rem !important; text-transform: uppercase; letter-spacing: .09em; color: var(--edf-muted) !important; }
[data-testid="stSidebar"] h3 { font-size: .78rem !important; text-transform: uppercase; letter-spacing: .09em; color: var(--edf-muted) !important; }
[data-testid="stSidebar"] hr { margin: .8rem 0 !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button {
    color: var(--edf-navy) !important;
}

/* ── Altair/Vega charts: transparent, no chrome ─────────────────────── */
[data-testid="stVegaLiteChart"] { background: transparent; }
[data-testid="stVegaLiteChart"] .vega-embed summary { display: none !important; }
[data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg { max-width: 100%; }

/* ══════════════════════════════════════════════════════════════════════
   QUEUE BAR — sticky, bottom of the page.
   position: sticky, not fixed — this is the correct tool here. The bar is
   the last element on the page, which scrolls at the document level (not
   an inner scroll box), so sticky inherits its horizontal position from
   normal document flow — the same flow that already excludes the sidebar
   and re-adjusts live when it's dragged or collapsed, with no hardcoded
   width or JS measurement. Two earlier attempts (a containing-block
   transform trick, then a hardcoded fixed-position offset) each fixed one
   dimension of this while breaking the other; sticky avoids the tradeoff.
   ══════════════════════════════════════════════════════════════════════ */
.edf-queue-bar {
    position: sticky;
    z-index: 999;
    display: flex;
    align-items: center;
    gap: .8rem;
    padding: .55rem .9rem;
    border: 1px solid var(--edf-line-strong);
    border-radius: var(--edf-radius);
    background: rgba(255,255,255,.97);
    backdrop-filter: saturate(180%) blur(8px);
    font-size: .84rem;
    color: var(--edf-navy);
}
/* The bar renders before the chip strip in the DOM, so when both are stuck
   the bar's offset has to clear the strip's full height — otherwise the two
   swap places between "stuck" and "at the end of the document". Idle has no
   strip below it, so it sits at the bottom on its own. The 4.5rem is a hair
   under the strip's measured height (~3.6rem + 1.1rem offset) so the panels
   overlap by a pixel or two rather than risking a visible seam. */
.edf-queue-bar.idle {
    bottom: 1.1rem;
    color: var(--edf-muted);
}
.edf-queue-bar.busy {
    bottom: 4.5rem;
    border-color: var(--edf-amber-line);
    border-left: 3px solid var(--edf-amber);
    border-bottom: none;
    border-radius: var(--edf-radius) var(--edf-radius) 0 0;
}
/* Status lamp: hollow when idle, amber and breathing while working. */
.edf-queue-bar .dot {
    width: 8px; height: 8px; border-radius: 50%;
    border: 1.5px solid var(--edf-line-strong); background: transparent; flex-shrink: 0;
}
.edf-queue-bar.busy .dot {
    background: var(--edf-amber); border-color: var(--edf-amber);
    animation: edf-pulse 1.6s ease-in-out infinite;
}
@keyframes edf-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(240,128,3,.45); }
    50%      { box-shadow: 0 0 0 5px rgba(240,128,3,0); }
}
.edf-queue-bar .summary {
    font-weight: 650; white-space: nowrap; font-size: .8rem;
    text-transform: uppercase; letter-spacing: .06em;
}
.edf-queue-bar .msg {
    color: var(--edf-ink-soft); font-size: .8rem;
    flex: 1; min-width: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.edf-queue-bar .pct {
    font-size: .76rem; font-weight: 650; color: var(--edf-amber-deep);
    font-variant-numeric: tabular-nums; flex: 0 0 auto;
}
.edf-queue-bar .progress {
    height: 5px; flex: 0 0 170px; max-width: 26%;
    background: var(--edf-alt); border-radius: 3px; overflow: hidden;
    border: 1px solid var(--edf-line);
}
.edf-queue-bar .progress > div {
    height: 100%; background: var(--edf-amber);
    border-radius: 3px; transition: width .4s ease;
}

/* ── Sliding chip strip — sticky directly below the bar, same reasoning. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) {
    position: sticky !important;
    bottom: 1.1rem !important;
    z-index: 999 !important;
    flex-wrap: nowrap !important;
    overflow-x: auto;
    overflow-y: visible;
    scroll-behavior: smooth;
    align-items: center !important;
    gap: .4rem !important;
    padding: .5rem .8rem;
    background: rgba(246,247,250,.97);
    backdrop-filter: saturate(180%) blur(8px);
    border: 1px solid var(--edf-amber-line);
    border-top: 1px dashed var(--edf-line-strong);
    border-left: 3px solid var(--edf-amber);
    border-radius: 0 0 var(--edf-radius) var(--edf-radius);
    scrollbar-width: thin;
    scrollbar-color: var(--edf-line-strong) transparent;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar { height: 5px; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-track { background: transparent; }
[data-testid="stHorizontalBlock"]:has(.qbar-chip)::-webkit-scrollbar-thumb {
    background: var(--edf-line-strong); border-radius: 3px;
}

/* Each column = one (chip + ×) pair. Shrinks to content; kids in a single
   horizontal row with zero gap so the chip and × become one pill. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    width: auto !important;
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] > div {
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) > [data-testid="stColumn"] > div > * {
    width: auto !important;
    flex: 0 0 auto !important;
}

/* Flatten ALL of Streamlit's inner wrappers so chip & × align exactly. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stElementContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    display: inline-flex !important;
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdown"],
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdownContainer"] {
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    display: inline-flex !important;
    align-items: stretch !important;
}
/* The markdown container's internal <p> wraps the "…"/"🔒" chips — flatten
   it the same way the button's <p> is flattened, so both sit at the same
   baseline/height as the × button. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
    display: contents !important;
}

/* ── The chip — left half of the pill. ──────────────────────────────────
   All variant colours live on the chip; CSS :has() on the column then
   propagates the same colours to the × button so they always agree. */
.qbar-chip {
    display: inline-flex;
    align-items: center;
    gap: .38rem;
    height: 1.85rem;
    padding: 0 .5rem 0 .7rem;
    border: 1px solid var(--edf-line-strong);
    border-right: none;
    border-radius: var(--edf-radius-sm) 0 0 var(--edf-radius-sm);
    background: #FFFFFF;
    color: var(--edf-navy);
    font-size: .8rem;
    line-height: 1;
    white-space: nowrap;
}
.qbar-chip .lamp {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--edf-line-strong); flex: 0 0 auto;
}
.qbar-chip.running {
    background: var(--edf-navy); border-color: var(--edf-navy); color: #fff;
}
.qbar-chip.running .lamp { background: var(--edf-amber); }
.qbar-chip.mine { font-weight: 650; }
.qbar-chip.mine::after {
    content: "YOU";
    font-size: .58rem; font-weight: 700; letter-spacing: .08em;
    padding: .1rem .28rem; border-radius: 2px;
    background: var(--edf-amber-tint); color: var(--edf-amber-deep);
}
.qbar-chip.running.mine::after { background: var(--edf-amber); color: #fff; }
.qbar-chip.cancelling {
    border-color: var(--edf-line); background: var(--edf-alt); color: #A6A6B8;
    text-decoration: line-through;
}
.qbar-chip code {
    background: transparent !important; border: none !important;
    padding: 0 !important; color: inherit !important;
    font-size: .72rem !important; opacity: .7;
}
.qbar-submitter { font-size: .7rem; opacity: .65; margin-left: .1rem; }

/* ── The × — right half of the same pill. Matches the chip's height,
   border and background. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button {
    height: 1.85rem !important;
    min-height: 1.85rem !important;
    width: 1.5rem !important;
    padding: 0 !important;
    margin: 0 !important;
    line-height: 1 !important;
    font-size: .8rem !important;
    font-weight: 500 !important;
    border: 1px solid var(--edf-line-strong) !important;
    border-left: none !important;
    border-radius: 0 var(--edf-radius-sm) var(--edf-radius-sm) 0 !important;
    background: #FFFFFF !important;
    color: var(--edf-muted) !important;
    cursor: pointer;
    box-shadow: none !important;
    transition: color .12s, background-color .12s, border-color .12s;
}
/* The button's internal <p> wraps the "✕" label — flatten it. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button p,
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button div {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
}
/* Variant colours flow from the chip to the × via :has() on the column. */
[data-testid="stColumn"]:has(.qbar-chip.running) [data-testid="stButton"] > button {
    background: var(--edf-navy) !important;
    border-color: var(--edf-navy) !important;
    color: rgba(255,255,255,.72) !important;
}
/* Hover — only the × box changes, never the chip. Destructive red. */
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:hover {
    color: #fff !important;
    background: #B02A1F !important;
    border-color: #B02A1F !important;
}
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:focus,
[data-testid="stHorizontalBlock"]:has(.qbar-chip) [data-testid="stButton"] > button:active {
    outline: none !important;
    box-shadow: none !important;
}
/* Cancelling / locked: static pills stamped on the right, same dimensions. */
.qbar-cancel-pending, .qbar-cancel-locked {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem;
    height: 1.85rem;
    border: 1px solid var(--edf-line);
    border-left: none;
    border-radius: 0 var(--edf-radius-sm) var(--edf-radius-sm) 0;
    background: var(--edf-alt);
    color: #A6A6B8;
    font-size: .76rem;
}

/* ══════════════════════════════════════════════════════════════════════
   HISTORY — table-like rows built from st.columns.
   ══════════════════════════════════════════════════════════════════════ */
.edf-hrow {
    display: flex; align-items: center; gap: .5rem;
    min-height: 2.1rem;
}
.edf-hhead {
    font-size: .66rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: var(--edf-muted);
    padding-bottom: .35rem;
}
.edf-acr { font-weight: 650; font-size: .9rem; color: var(--edf-navy); }
.edf-cid {
    font-family: var(--edf-mono); font-size: .74rem; color: var(--edf-muted);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block;
}
.edf-num { font-variant-numeric: tabular-nums; font-weight: 600; font-size: .86rem; color: var(--edf-navy); }
.edf-num small { color: var(--edf-muted); font-weight: 500; }
.edf-date { font-size: .78rem; color: var(--edf-ink-soft); font-variant-numeric: tabular-nums; }

/* Zebra + hover on the history rows. Streamlit wraps each st.columns() in a
   stHorizontalBlock; the marker span scopes this to history rows only. */
[data-testid="stHorizontalBlock"]:has(.edf-hrow-marker) {
    border-bottom: 1px solid var(--edf-line);
    padding: .3rem .4rem;
    border-radius: var(--edf-radius-sm);
    align-items: center !important;
    transition: background-color .1s;
}
[data-testid="stHorizontalBlock"]:has(.edf-hrow-marker):hover { background: var(--edf-alt); }
[data-testid="stHorizontalBlock"]:has(.edf-hrow-marker) [data-testid="stElementContainer"] { margin-bottom: 0 !important; }
[data-testid="stHorizontalBlock"]:has(.edf-hrow-marker) button[data-testid="stBaseButton-secondary"] {
    height: 1.8rem !important; min-height: 1.8rem !important;
    font-size: .76rem !important; padding: 0 .6rem !important;
}
[data-testid="stHorizontalBlock"]:has(.edf-hrow-marker) button[data-testid="stBaseButton-secondary"]:hover {
    background: var(--edf-navy) !important; border-color: var(--edf-navy) !important; color: #fff !important;
}

/* ── Criterion card header ──────────────────────────────────────────── */
.edf-crit-name { font-weight: 650; font-size: .95rem; color: var(--edf-navy); }
.edf-tile-inline .edf-crit-meta {
    font-size: .66rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; margin: 0;
}
.edf-crit-meta { font-size: .74rem; color: var(--edf-muted); margin-top: .12rem; }
.edf-score-badge {
    display: inline-flex; align-items: baseline; gap: .12rem;
    padding: .18rem .5rem; border-radius: var(--edf-radius-sm);
    border: 1px solid var(--edf-line-strong); background: #fff;
    font-variant-numeric: tabular-nums;
}
.edf-score-badge b { font-size: 1rem; font-weight: 680; color: var(--edf-navy); }
.edf-score-badge span { font-size: .74rem; color: var(--edf-muted); }

/* Labelled bullet groups inside the criterion detail. */
.edf-lgroup { margin-bottom: .5rem; }
.edf-lgroup .lg-h {
    font-size: .7rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: var(--edf-muted);
    display: flex; align-items: center; gap: .35rem; margin-bottom: .25rem;
}
.edf-lgroup .lg-h i { width: 3px; height: .72rem; border-radius: 2px; background: var(--edf-line-strong); }
.edf-lgroup.good  .lg-h i { background: #12715A; }
.edf-lgroup.bad   .lg-h i { background: #B02A1F; }
.edf-lgroup.gap   .lg-h i { background: var(--edf-amber); }
.edf-lgroup ul { margin: 0; padding-left: 1.05rem; }
.edf-lgroup li { font-size: .82rem; line-height: 1.5; color: var(--edf-ink-soft); margin-bottom: .18rem; }
.edf-lgroup p.none { font-size: .82rem; color: #A6A6B8; margin: 0; font-style: italic; }

/* ── Empty state ────────────────────────────────────────────────────── */
.edf-empty {
    border: 1px dashed var(--edf-line-strong);
    border-radius: var(--edf-radius);
    background: var(--edf-alt);
    padding: 2.2rem 1rem;
    text-align: center;
}
.edf-empty .t { font-size: .92rem; font-weight: 650; color: var(--edf-navy); }
.edf-empty .s { font-size: .8rem; color: var(--edf-muted); margin-top: .3rem; }

/* ── Prose block (ESR summary) — readable measure, justified ────────── */
.edf-prose {
    font-size: .88rem; line-height: 1.68; color: var(--edf-ink-soft);
    max-width: 82ch; text-align: justify; hyphens: auto;
}
.edf-prose p { margin: 0 0 .6rem 0; }

/* Kill the extra bottom margin Streamlit puts under every element in the
   dense header/tile rows. */
[data-testid="stHorizontalBlock"]:has(.edf-dense) [data-testid="stElementContainer"] {
    margin-bottom: 0 !important;
}
</style>
"""


def inject_css() -> None:
    """Install the stylesheet. Safe to call once per script run."""
    import streamlit as st

    st.markdown(_CSS, unsafe_allow_html=True)


# ── Brand assets ──────────────────────────────────────────────────────────


@lru_cache(maxsize=2)
def logo_data_uri(with_tagline: bool = False) -> str:
    """The logo as an inline data URI.

    Inlining lets the logo live inside a flex lockup together with the
    title, which is the only way to get the two reliably centred on one
    line — st.image sits in its own column with its own wrapper margins.
    """
    import base64

    return "data:image/png;base64," + base64.b64encode(
        logo_bytes(with_tagline)
    ).decode("ascii")


def appbar(title: str, subtitle_html: str, logo_height: int = 34) -> str:
    """Logo + title + subtitle as a single horizontal lockup."""
    return (
        '<div class="edf-appbar"><span class="edf-appbar-marker"></span>'
        f'<img src="{logo_data_uri()}" alt="IKH" style="height:{logo_height}px">'
        f'<div class="edf-titles"><div class="edf-apptitle">{_h(title)}</div>'
        f'<div class="edf-appsub">{subtitle_html}</div></div>'
        "</div>"
    )


@lru_cache(maxsize=2)
def logo_bytes(with_tagline: bool = False) -> bytes:
    """The logo cropped to its content, as PNG bytes.

    Falls back to the untouched file if Pillow is unavailable, so a missing
    optional dependency can never take the page down.
    """
    try:
        from PIL import Image

        with Image.open(LOGO_PATH) as im:
            crop = im.convert("RGBA").crop(_CROP_FULL if with_tagline else _CROP_MARK)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return LOGO_PATH.read_bytes()


# ── Small HTML components ─────────────────────────────────────────────────


def section(title: str, hint: str = "", step: int | None = None) -> str:
    """An uppercase section rule with an amber tick and optional step number."""
    step_html = f'<span class="edf-step">{step}</span>' if step is not None else ""
    hint_html = f'<span class="h">{_h(hint)}</span>' if hint else ""
    return (
        f'<div class="edf-section">{step_html}'
        f'<span class="t">{_h(title)}</span>{hint_html}</div>'
    )


def chip(text: str, variant: str = "", key: str = "", dot: bool = False,
         max_len: int | None = None) -> str:
    """A small rounded pill. `key` renders an uppercase label before the value.

    `max_len` shortens the value and keeps the full text as a tooltip —
    needed for Ollama model tags, which are long enough to wrap the header
    chips onto a second line.
    """
    label = str(text or "—")
    shown = label
    if max_len and len(label) > max_len:
        shown = label[: max_len - 1].rstrip() + "…"
    cls = f"edf-chip {variant}".strip()
    dot_html = '<span class="dot"></span>' if dot else ""
    key_html = f'<span class="k">{_h(key)}</span>' if key else ""
    title = f' title="{_h(label)}"' if shown != label else ""
    return f'<span class="{cls}"{title}>{dot_html}{key_html}{_h(shown)}</span>'


def tile(label: str, value: str, sub: str = "", variant: str = "") -> str:
    """One metric tile. `value` may contain a single <small> unit suffix."""
    sub_html = f'<div class="s">{_h(sub)}</div>' if sub else ""
    return (
        f'<div class="edf-tile {variant}">'
        f'<div class="k">{_h(label)}</div>'
        f'<div class="v">{value}</div>'
        f"{sub_html}</div>"
    )


def tiles(*items: str) -> str:
    """Lay out tiles in a responsive auto-fit grid."""
    return f'<div class="edf-tiles">{"".join(items)}</div>'


def meter(frac: float, color: str | None = None, navy: bool = False) -> str:
    pct = max(0.0, min(1.0, float(frac))) * 100
    cls = "edf-meter navy" if navy else "edf-meter"
    style = f"width:{pct:.1f}%" + (f";background:{color}" if color else "")
    return f'<div class="{cls}"><i style="{style}"></i></div>'


def bullet_group(label: str, items: list[str] | None, tone: str = "",
                 empty: str = "None recorded.") -> str:
    """A labelled bullet list with a coloured tick — the detail-panel workhorse."""
    clean = [i for i in (items or []) if str(i).strip()]
    body = (
        "<ul>" + "".join(f"<li>{_h(str(i))}</li>" for i in clean) + "</ul>"
        if clean
        else f'<p class="none">{_h(empty)}</p>'
    )
    return (
        f'<div class="edf-lgroup {tone}">'
        f'<div class="lg-h"><i></i>{_h(label)}</div>{body}</div>'
    )


def empty_state(title: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{_h(sub)}</div>' if sub else ""
    return f'<div class="edf-empty"><div class="t">{_h(title)}</div>{sub_html}</div>'


def prose(text: str) -> str:
    """Render multi-paragraph model output at a readable measure."""
    paras = [p.strip() for p in str(text or "—").split("\n") if p.strip()]
    body = "".join(f"<p>{_h(p)}</p>" for p in paras) or "<p>—</p>"
    return f'<div class="edf-prose">{body}</div>'


def score_color(frac: float) -> str:
    """Map a 0–1 achievement ratio onto the band scale."""
    idx = min(len(BAND_SCALE) - 1, max(0, int(round(float(frac) * (len(BAND_SCALE) - 1)))))
    return BAND_SCALE[idx]
