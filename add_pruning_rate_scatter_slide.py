"""
2026-09-24: Adds the pruning-rate-vs-service-rate scatter slide (mixed-class, bi200/ss100,
per-instance breakdown) - explains the lc-vs-lr/lrc service-rate gap found this session.
Inserted right after "Why Does bi200/ss100 Underperform?" (same diagnostic thread),
search-by-title so this survives the deck's repeated external-resave slide loss.

Run: ./venv/bin/python3 add_pruning_rate_scatter_slide.py
"""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor

DECK = "request_pruner_deck.pptx"
SIL_DIR = "figures_export/Request Pruner SIL"

NAVY = RGBColor(0x1E, 0x27, 0x61)
GRAY = RGBColor(0x60, 0x60, 0x60)


def add_title(slide, title, subtitle=None):
    tb = slide.shapes.add_textbox(Emu(457200), Emu(292608), Emu(11277295), Emu(640080))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = title
    r.font.name = "Cambria"; r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = NAVY
    if subtitle:
        tb2 = slide.shapes.add_textbox(Emu(457200), Emu(932688), Emu(11277295), Emu(365760))
        p2 = tb2.text_frame.paragraphs[0]
        r2 = p2.add_run(); r2.text = subtitle
        r2.font.name = "Calibri"; r2.font.size = Pt(13); r2.font.color.rgb = GRAY


def add_caption(slide, text, top):
    tb = slide.shapes.add_textbox(Emu(457200), top, Emu(11277295), Emu(700000))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name = "Calibri"; r.font.size = Pt(11); r.font.color.rgb = GRAY
    tb.text_frame.word_wrap = True


prs = Presentation(DECK)
BLANK = prs.slide_layouts[6]

insert_after = None
for i, s in enumerate(prs.slides):
    for sh in s.shapes:
        if sh.shape_type == 17 and sh.has_text_frame and sh.text_frame.text.startswith("Why Does bi200/ss100 Underperform?"):
            insert_after = i
            break
    if insert_after is not None:
        break
if insert_after is None:
    raise SystemExit("Could not find the 'Why Does bi200/ss100 Underperform?' slide - aborting, nothing changed.")

s = prs.slides.add_slide(BLANK)
add_title(s, "Root Cause of the Instance Gap: Pruning Rate",
           "bi200/ss100 mixed-class, expert-prefix-label pruner, seed 42, best-val epoch, all 12 val instances")
s.shapes.add_picture(f"{SIL_DIR}/pruning_vs_service_rate_scatter.png",
                      Emu(2971800), Emu(1264920), Emu(6248400))
add_caption(s,
    "Pruning rate almost fully explains the lc-vs-lr/lrc service-rate gap: lc instances (clustered "
    "pickup/dropoff) are pruned lightly (4-14%) and stay near-fully served (89-94%); lr/lrc instances "
    "(random/mixed geometry) are pruned 3-15x harder (31-56%) and collapse toward 0% service. Not a bug - "
    "the pruner is miscalibrated specifically for spatially scattered requests, where fewer candidates "
    "happen to match the K=2 expert-route prefix the legacy imitation-scoring rule needs.",
    Emu(7700000))

xml_slides = prs.slides._sldIdLst
slides = list(xml_slides)
xml_slides.remove(slides[-1])
xml_slides.insert(insert_after + 1, slides[-1])

prs.save(DECK)
print(f"saved {DECK}, {len(prs.slides)} slides, new slide inserted at index {insert_after + 1}")
