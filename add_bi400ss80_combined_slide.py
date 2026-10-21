"""
2026-09-25: Adds the bi400/ss80 combined-pruner comparison slide (5/5 seeds now
complete), analog to the bi200/ss100 and bi400/ss200 combined-pruner slides. Inserted
right after the bi400/ss200 one, search-by-title so this survives the deck's repeated
external-resave slide loss.

Run: ./venv/bin/python3 add_bi400ss80_combined_slide.py
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
        if sh.shape_type == 17 and sh.has_text_frame and sh.text_frame.text.startswith("New Results: Combining Request + Pair Pruner (bi400/ss200)"):
            insert_after = i
            break
    if insert_after is not None:
        break
if insert_after is None:
    raise SystemExit("Could not find the bi400/ss200 combined-pruner slide - aborting, nothing changed.")

s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: Combining Request + Pair Pruner (bi400/ss80)",
           "class1, threshold=0.5, min_retention_fraction=0.3, n=4-5 seeds (42,1,2,3[,4])")
s.shapes.add_picture(f"{SIL_DIR}/bi400ss80_class1_combined_pruner_service_rate.png",
                      Emu(228600), Emu(1264920), Emu(5943600))
s.shapes.add_picture(f"{SIL_DIR}/bi400ss80_class1_combined_pruner_runtime.png",
                      Emu(6248400), Emu(1264920), Emu(5943600))
add_caption(s,
    "Both pruner stages beat the no-pruner baseline (79.32±0.64% → 82.58±0.72% → 82.20±0.96%), "
    "consistent with bi400/ss200. The Pair Pruner is service-rate-neutral here but cuts runtime by another "
    "~30% on top of the Request Pruner's own saving (63.9→44.5 min, cluster-only seeds - seed42 ran "
    "locally and is excluded from the runtime bar as a non-comparable outlier). Baseline is n=4 (seed 4 "
    "pending). All values are \"Best val service rate\" (the saved checkpoint), not a fixed epoch.",
    Emu(7700000))

xml_slides = prs.slides._sldIdLst
slides = list(xml_slides)
xml_slides.remove(slides[-1])
xml_slides.insert(insert_after + 1, slides[-1])

prs.save(DECK)
print(f"saved {DECK}, {len(prs.slides)} slides, new slide inserted at index {insert_after + 1}")
