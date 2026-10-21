"""
2026-09-24: (1) Fixes the stale "recover 61% of the damage at bi200/ss100" caption on the
Slide 5 (class-1 results), which was based on a stale table value 60.9+-4.5% for the
request-pruner-only service rate - a fresh recompute directly from
sil_training_bi200_ss100_class1_requestpruner_prefixlabel_t0.5_seed{42,1,2,3,4}.log's
"Best val service rate" lines gives 65.85+-2.45%, confirmed independently by the user
against the same logs on the cluster (2026-09-24). Corrected recovered-damage % = 74.9%
(baseline 74.5%, RHO-label pruner 40.0%, damage=34.5pp; new prefix gap=8.65pp; recovered
= (34.5-8.65)/34.5 = 74.9%), not 61%.
Note: slide 5's table shape is gone (deck was re-saved externally again) - only the
caption text is fixed here, table NOT reconstructed pending user confirmation.

(2) Adds a new slide (search-by-title insertion, after "Same runs as the previous
slide..." i.e. the runtime slide) showing the new bi200/ss100 combined
(Request+Pair-Pruner) vs. request-only vs. no-pruner baseline comparison
(scripts/analysis/make_bi200ss100_combined_pruner_chart.py), 5 seeds each variant.

Run: ./venv/bin/python3 add_combined_pruner_slide.py
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
    tb = slide.shapes.add_textbox(Emu(457200), top, Emu(11277295), Emu(600000))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name = "Calibri"; r.font.size = Pt(11); r.font.color.rgb = GRAY
    tb.text_frame.word_wrap = True


prs = Presentation(DECK)

# ---- (1) fix stale caption on the class-1 results slide ----
fixed = False
for s in prs.slides:
    for sh in s.shapes:
        if sh.shape_type == 17 and sh.has_text_frame and "recover 61% of the damage" in sh.text_frame.text:
            p0 = sh.text_frame.paragraphs[0]
            new_text = p0.runs[0].text.replace(
                "recover 61% of the damage at bi200/ss100",
                "recover 75% of the damage at bi200/ss100 (65.85±2.45% vs. baseline 74.5±1.8%, corrected 2026-09-24 - see chat)"
            )
            p0.runs[0].text = new_text
            for r in p0.runs[1:]:
                r.text = ""
            fixed = True
if not fixed:
    print("WARNING: stale caption text not found, nothing fixed")
else:
    print("fixed stale 61% caption -> 75%")

# ---- (2) insert new combined-pruner comparison slide ----
BLANK = prs.slide_layouts[6]
insert_after = None
for i, s in enumerate(prs.slides):
    for sh in s.shapes:
        if sh.shape_type == 17 and sh.has_text_frame and sh.text_frame.text.startswith("Same runs as the previous slide"):
            insert_after = i
            break
    if insert_after is not None:
        break
if insert_after is None:
    raise SystemExit("Could not find the runtime slide (Same runs as the previous slide...) - aborting insert, caption fix (if any) still saved.")

s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: Combining Request + Pair Pruner",
           "bi200/ss100, class1, threshold=0.5, min_retention_fraction=0.3, n=5 seeds (42,1,2,3,4)")
s.shapes.add_picture(f"{SIL_DIR}/bi200ss100_class1_combined_pruner_service_rate.png",
                      Emu(228600), Emu(1264920), Emu(5943600))
s.shapes.add_picture(f"{SIL_DIR}/bi200ss100_class1_combined_pruner_runtime.png",
                      Emu(6248400), Emu(1264920), Emu(5943600))
add_caption(s,
    "Request Pruner alone: 65.85±2.45% service rate, 49.4±1.4 min (vs. baseline 74.5±1.8%, ~95.6 min). "
    "Adding the Pair Pruner (v1_final GNN, threshold=0.5) on top: 64.40±3.57%, 37.0±1.7 min - "
    "an extra ~25% runtime saving for ~1.45pp more service-rate loss, consistent across all 5 seeds "
    "(runtime saving 21.8-26.9% per seed, service-rate delta -2.52 to +0.31pp per seed).",
    Emu(7550000))

xml_slides = prs.slides._sldIdLst
slides = list(xml_slides)
xml_slides.remove(slides[-1])
xml_slides.insert(insert_after + 1, slides[-1])

prs.save(DECK)
print(f"saved {DECK}, {len(prs.slides)} slides, new slide inserted at index {insert_after + 1}")
