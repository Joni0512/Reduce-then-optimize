---
name: experiment-results-export
description: Whenever an experiment/training run produces results (service-rate curves, comparison charts, sweep outcomes), always save them as both PNG and PDF, not just PNG. Use this any time a plot is generated from experiment output, a run finishes, or the user asks to "save"/"export" results - do it proactively, don't wait to be asked each time.
---

# Always export results as PNG + PDF

Every plot generated from experiment results in this project (Li&Lim/NYC
PDPTW runs, SIL/SRL training curves, sweep comparisons, seed comparisons)
must be saved in BOTH formats, every time, without being asked:

- **PNG** - for quick viewing, pasting into chat, pasting into slides (pptx
  embeds PNG natively and cleanly).
- **PDF** - vector format, required for the actual thesis LaTeX document
  (`\includegraphics` wants vector PDFs for print-quality figures, not
  raster PNGs) and for any figure that might need to scale up.

## How to do it (matplotlib)

```python
fig.savefig(output_path.with_suffix(".png"), dpi=150)
fig.savefig(output_path.with_suffix(".pdf"))
```

Same `fig`, two calls, right next to each other - don't generate the PDF as
an afterthought in a separate pass. If a script already exists that only
saves PNG (e.g. `srl_training_loop.py`'s `srl_train_val_curves.png`, several
`scripts/analysis/make_*_chart.py` files), add the PDF line alongside it
when touched, rather than leaving it PNG-only.

## Where the files go

Keep both formats in the same output directory, same basename, different
extension - never split them into separate folders. This repo's convention:
research/experiment output plots live under `outputs/<run>/...` or
`figures_export/<topic>/...` (the latter specifically for figures destined
for slides/thesis, see `results-into-slides` skill for what happens next).

## When this does NOT apply

One-off debugging/diagnostic plots that will never leave the terminal or be
referenced again (a quick sanity check while iterating) don't need PDF -use
judgment. Anything that's a candidate for a slide, a thesis figure, or that
the user references again later (by name, in a chat, in a doc) gets both.
