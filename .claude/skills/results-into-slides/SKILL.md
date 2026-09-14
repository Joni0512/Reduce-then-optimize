---
name: results-into-slides
description: Whenever new experiment results (charts, comparison numbers, sweep outcomes) are ready, add them into the matching topic slide deck (e.g. SRL results go into the SRL deck, not a random new file) as a properly designed slide - never a plain bullet-point dump. Use this proactively once a result is confirmed, not only when explicitly asked to make a slide.
---

# Put results into the matching deck, properly designed

Don't just report numbers in chat and stop - if a result is meeting/thesis
relevant, it belongs in the matching deck. Don't create a new deck per
result either; find the existing topic deck first.

## 1. Find the right deck

This project keeps topic-organized decks, mainly under `figures_export/`
and the repo root:
- SRL results/design → `figures_export/srl_meeting_deck.pptx` (chronological
  meeting/results deck) and/or `srl_design_overview_en.pptx` (repo root,
  concept/design-decision deck) - check which one the current result fits:
  a NEW finding/result → meeting deck; a design DECISION (which of several
  options was chosen and why) → design deck. Both may apply.
- Other topics: check `figures_export/<TOPIC>/` (e.g. `figures_export/NYC/`,
  `figures_export/LiLim_ReduceThenOptimize_SIL/`) for an existing deck
  before creating a new one.

If genuinely unsure which deck (or whether a new one is warranted), ask -
don't guess and don't default to creating a new file.

## 2. Match the existing deck's visual design exactly

Never freehand a new style. Before adding anything:
1. Unpack the target deck's slides and read 2-3 existing slides' raw XML to
   get the actual color hex values, fonts, and layout shapes in use (don't
   assume - this project has used the same "Midnight Executive"-ish navy
   (`1E2761`) + gold/amber accent palette across multiple decks, but confirm
   per-deck since a numbered-step layout, a two-card comparison layout, and
   a stat-callout layout are all different existing templates within the
   same decks).
2. Pick the closest-matching EXISTING slide as a template (comparison of
   options → the two-card layout; a chart with a headline number → the
   stat-callout layout; a sequence/pipeline → the numbered-step layout).
   Duplicate it (`scripts/add_slide.py` from the `pptx` skill, or manual
   `[Content_Types].xml` + `presentation.xml.rels` + `sldIdLst` surgery if
   copying a slide across two different decks - see this session's history
   for the exact mechanical steps if needed) rather than building a layout
   from scratch.
3. Replace text run-by-run (`<a:t>` contents), keep every other XML
   attribute untouched, so spacing/fonts/colors carry over exactly.
4. For a chart result: embed the PNG (see `experiment-results-export`
   skill - PNG version goes into the slide, PDF stays for the thesis doc)
   using the deck's existing pattern for image placement, not a raw
   full-bleed screenshot.

## 3. Content rule: no flat bullet dumps

A results/design slide needs actual structure, not a wall of bullets:
- **Comparing options/configs** → side-by-side cards (see the existing
  "Phase 1 vs Phase 2" / "Option A vs Option B" slides for the pattern:
  numbered circle badge, bold header, 3-4 short bullets max per card, a
  bottom callout box stating the decision/synthesis).
- **Reporting a headline metric** → a big stat number (60-72pt) with a small
  label underneath, next to (not instead of) the actual chart.
- **A process/sequence of steps** → numbered step badges in a row, not a
  numbered list of paragraphs.
- Every content bullet stays short (fragment, not a full sentence) - this
  matches the project's general "Zusammenfassungen als Stichpunkte"
  communication rule, just applied to slide design instead of chat text.
- Every slide needs at least one non-text visual element (chart, stat
  callout, numbered badge) - a title + plain bullet list is the thing to
  avoid.

## 4. Always validate + visually check before calling it done

```bash
python scripts/office/validate.py <deck>.pptx --original <deck_before_edit>.pptx
python scripts/office/soffice.py --headless --convert-to pdf <deck>.pptx
pdftoppm -jpeg -r 150 <deck>.pdf <prefix>
```
Look at the actual rendered slide image (not just "it compiled") - check for
text overflow, and confirm the new slide's font/color actually matches its
neighbors, not just "looks plausible."

## 5. Which deck(s) to touch for a given result type

| Result type | Deck |
|---|---|
| New SRL/SIL sweep or training-curve result | `figures_export/srl_meeting_deck.pptx` |
| A design decision (chosen option, locked-in config) | `srl_design_overview_en.pptx` (repo root) |
| NYC-specific experiment result | `figures_export/NYC/` deck matching the sub-topic |
| Li&Lim/SIL pipeline result | `figures_export/LiLim_ReduceThenOptimize_SIL/` deck matching the sub-topic |

If a result spans two decks' scope (e.g. it's both a new SRL result AND
finalizes a design decision), add it to both, worded appropriately for each
(meeting deck: what was found; design deck: what was decided going forward).
