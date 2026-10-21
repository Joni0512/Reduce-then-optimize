"""
2026-10-12: One-slide overview of the two pruners (Request Pruner, Pair Pruner) for the thesis talk.
Top: pipeline figure (figures_export/04_Method/reduce_then_optimize_pipeline.png, built by
scripts/analysis/make_pipeline_overview_figure.py). Bottom: two boxes with a few bullet points each.

Bullet content verified against the code (2026-10-12):
  Request Pruner - rtv_solver/pipeline/request_pruner.py (kept = score >= threshold; force-keeps
    urgent and already-committed requests; pruned requests are deferred, not rejected) and
    request_pruner_mlp.py (feed-forward node classifier, per-request features); expert-prefix label from
    rtv_solver/pipeline/build_request_pruner_prefix_dataset.py (per window and vehicle, the first K entries of
    the vehicle's reference route projected onto the window batch get label 1, K = max cardinality).
    min_retention_fraction 0.3 is the value used in all SIL runs of the pruner comparison (Config default differs).
  Pair Pruner - rtv_solver/pipeline/request_graph_pruner.py (GNN edge scores on the full RR graph; pair allowed
    iff score >= threshold; trip [A,B,C] only if (A,B), (A,C), (B,C) are all allowed); trained on the reference
    solution (docstring of build_request_pruner_prefix_dataset.py).
No result numbers on the slide on purpose (design slide); add them from the result tables if wanted.

Style as in build_request_pruner_deck.py (16:9, Cambria navy title, Calibri body).
Run: ./venv/bin/python3 build_pruner_overview_slide.py
"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Emu, Pt

NAVY = RGBColor(0x1E, 0x27, 0x61)
PETROL = RGBColor(0x27, 0x65, 0x75)
GRAY = RGBColor(0x60, 0x60, 0x60)
LIGHT = RGBColor(0xF3, 0xF5, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FIG = "figures_export/04_Method/reduce_then_optimize_pipeline.png"
OUT = "pruner_overview_slide.pptx"
FIG_ASPECT = 3520 / 1078  # width / height of the PNG

REQUEST_BULLETS = [
    "MLP node classifier on per-request features",
    "Label: first K requests of each vehicle's expert route per window (K = max. cardinality)",
    "Score < threshold: request is deferred to the next window, not rejected",
    "Always kept: urgent and already committed requests; min. retention fraction 0.3",
]
PAIR_BULLETS = [
    "GNN edge classifier (message passing) on the full request-request graph",
    "Trained on the reference solution; pair allowed if score ≥ threshold (0.5)",
    "Trip [A, B, C] only if (A,B), (A,C) and (B,C) are all allowed",
    "Shrinks the RTV search space; the Request Pruner shrinks the request set",
]

prs = Presentation()
prs.slide_width = Emu(12191695)
prs.slide_height = Emu(6858000)
slide = prs.slides.add_slide(prs.slide_layouts[6])

# title
tb = slide.shapes.add_textbox(Emu(457200), Emu(228600), Emu(11277295), Emu(600000))
r = tb.text_frame.paragraphs[0].add_run()
r.text = "Reduce: Request Pruner and Pair Pruner"
r.font.name = "Cambria"; r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY

# pipeline figure (centered, width-limited so the boxes below still fit)
fig_w = Emu(10600000)
fig_h = Emu(int(fig_w / FIG_ASPECT))
fig_left = Emu(int((12191695 - fig_w) / 2))
fig_top = Emu(820000)
slide.shapes.add_picture(FIG, fig_left, fig_top, width=fig_w)

# two boxes
box_top = Emu(fig_top + fig_h + 130000)
box_h = Emu(6858000 - box_top - 230000)
gap = Emu(240000)
margin = Emu(457200)
box_w = Emu(int((12191695 - 2 * margin - gap) / 2))


def add_box(left, heading, bullets):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, box_top, box_w, box_h)
    shp.adjustments[0] = 0.04
    shp.fill.solid(); shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = PETROL; shp.line.width = Pt(1.5)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(180000)
    tf.margin_top = Emu(90000); tf.margin_bottom = Emu(60000)
    p = tf.paragraphs[0]
    rr = p.add_run(); rr.text = heading
    rr.font.name = "Cambria"; rr.font.size = Pt(18); rr.font.bold = True; rr.font.color.rgb = PETROL
    p.space_after = Pt(4)
    from pptx.enum.text import PP_ALIGN
    p.alignment = PP_ALIGN.LEFT
    for b in bullets:
        pb = tf.add_paragraph()
        pb.alignment = PP_ALIGN.LEFT
        pb.space_after = Pt(3)
        rb = pb.add_run(); rb.text = "•  " + b
        rb.font.name = "Calibri"; rb.font.size = Pt(13); rb.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    from pptx.enum.text import MSO_ANCHOR
    tf.vertical_anchor = MSO_ANCHOR.TOP


add_box(margin, "Request Pruner (per request)", REQUEST_BULLETS)
add_box(Emu(margin + box_w + gap), "Pair Pruner (per request pair)", PAIR_BULLETS)

prs.save(OUT)
print(f"saved {OUT}")
