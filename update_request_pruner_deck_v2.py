"""
2026-09-23 (v2): Second in-place update of request_pruner_deck.pptx:
  1. Inserts a new "Mixed-Class Runtime & Savings" slide (mirrors the class-1
     runtime slide) right after the existing mixed-class service-rate slide.
  2. Rewrites the Summary & Open Points slide's bullets to match the final
     numbers (previous ones were stale: 63-146% -> 61-146%, mixed-class
     11%->8% and no longer single-seed for 2/3 configs, "Next:" list partly done).

Still leaves slides 0-4 (title/root-cause/fix) untouched - only the results slides
and the summary the user asked to refresh.

Run: ./venv/bin/python3 update_request_pruner_deck_v2.py
"""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DECK = "request_pruner_deck.pptx"
SIL_DIR = "figures_export/Request Pruner SIL"

NAVY = RGBColor(0x1E, 0x27, 0x61)
GRAY = RGBColor(0x60, 0x60, 0x60)


def add_title(slide, title, subtitle=None):
    tb = slide.shapes.add_textbox(Emu(457200), Emu(292608), Emu(11277295), Emu(640080))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.name = "Cambria"; r.font.size = Pt(32); r.font.bold = True; r.font.color.rgb = NAVY
    if subtitle:
        tb2 = slide.shapes.add_textbox(Emu(457200), Emu(932688), Emu(11277295), Emu(365760))
        p2 = tb2.text_frame.paragraphs[0]
        r2 = p2.add_run(); r2.text = subtitle
        r2.font.name = "Calibri"; r2.font.size = Pt(14); r2.font.color.rgb = GRAY


def add_caption(slide, text, top):
    tb = slide.shapes.add_textbox(Emu(457200), top, Emu(11277295), Emu(457200))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name = "Calibri"; r.font.size = Pt(11); r.font.color.rgb = GRAY
    tb.text_frame.word_wrap = True


def set_text(shape, text):
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    if p0.runs:
        p0.runs[0].text = text
        for r in p0.runs[1:]:
            r.text = ""
    else:
        p0.text = text
    for para in tf.paragraphs[1:]:
        for r in para.runs:
            r.text = ""


prs = Presentation(DECK)
BLANK = prs.slide_layouts[6]

# ---- 1. New mixed-class runtime slide ----
s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: Mixed-Class Runtime & Savings",
          "Same runs as the previous slide — total wall-clock (train+val), % = saving vs. no-pruner baseline")
s.shapes.add_picture(f"{SIL_DIR}/sil_mixedclass_requestpruner_prefixlabel_final_comparison_runtime.png",
                      Emu(2895600), Emu(1417320), Emu(6400800), Emu(6600000))
add_caption(s, "The expert-prefix-label pruner saves less runtime than the RHO-label pruner at bi200/ss100 (34.1% vs. 39.9%) "
               "but far more at bi400/ss200 (34.1% overall, while also beating the baseline's service rate).", Emu(6800000))

# move the new slide (currently last) to right after the old mixed-class slide (index 7 -> new slide goes to index 8)
xml_slides = prs.slides._sldIdLst
slides = list(xml_slides)
xml_slides.remove(slides[-1])
xml_slides.insert(8, slides[-1])

# ---- 2. Refresh the Summary & Open Points slide (now pushed to index 9) ----
s = prs.slides[9]
texts = [sh for sh in s.shapes if sh.shape_type == 17]  # TEXT_BOX only - excludes the numbered ovals (AUTO_SHAPE)
# texts[0] = title "Summary & Open Points" (unchanged), texts[1..6] = the 6 numbered bullets
bullets_final = [
    "The Request Pruner is not fundamentally broken — it was trained on the wrong label for SIL's legacy imitation rule.",
    "Expert-prefix labels recover 61–146% of the SIL damage at class-1 (all 3 configs, n=5 seeds) — beating the no-pruner baseline at bi400/ss80 and bi400/ss200.",
    "A ~13.6pp residual gap remains at bi200/ss100 class-1, not fully explained (a window-count/divergence hypothesis was tested and refuted).",
    "bi400/ss80 required a full HP re-sweep after the reused RHO-label architecture proved mis-calibrated — the re-swept pruner then became the single best result overall (81.8% ± 0.6%, +5.8pp over baseline).",
    "Mixed-class recovers far less damage than class-1 (bi200/ss100: 8% vs. 61%, n=5 seeds both) — confirmed under replication, not a single-seed artifact. bi400/ss200 mixed-class beats its baseline (n=5).",
    "Open: bi400/ss80 mixed-class on more seeds (currently 1, repaired pruner), then Request Pruner + Pair Pruner combined (already supported by trip_handler.py).",
]
for tb, txt in zip(texts[1:], bullets_final):
    set_text(tb, txt)

prs.save(DECK)
print(f"saved {DECK}, {len(prs.slides)} slides")
