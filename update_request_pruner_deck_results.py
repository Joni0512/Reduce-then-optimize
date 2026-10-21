"""
2026-09-23: In-place update of request_pruner_deck.pptx's RESULT slides only (5, 6, 7)
with the now-complete 5-seed data (bi400/ss80 class-1 finished, bi400/ss200 mixed-class
finished). Deliberately does NOT touch slides 0-4 and 8 - the user edited titles/wording
there directly in the deck (see chat 2026-09-23, "ich habe ja auch paar dokumentationen
aktualisiert ... das soll lassen nur die ergebnisse anpassen") and those edits must survive.

Approach: open the EXISTING file (not rebuild from scratch like build_request_pruner_deck.py
does), edit table cell text and caption/subtitle runs in place (preserves existing font/style),
and only fully replace the two picture shapes per result slide (swap to the final 3-config /
5-seed chart images) since python-pptx has no partial-image-update API.

Run: ./venv/bin/python3 update_request_pruner_deck_results.py
"""
from pptx import Presentation
from pptx.util import Emu
import copy

DECK = "request_pruner_deck.pptx"
SIL_DIR = "figures_export/Request Pruner SIL"


def set_text(shape, text):
    """Overwrite a text shape's content, keeping the first run's font/style."""
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


def set_table_cell(table, row, col, text):
    cell = table.cell(row, col)
    p0 = cell.text_frame.paragraphs[0]
    if p0.runs:
        p0.runs[0].text = text
        for r in p0.runs[1:]:
            r.text = ""
    else:
        p0.text = text


def replace_picture(slide, old_pic_shape, new_path):
    left, top, width, height = old_pic_shape.left, old_pic_shape.top, old_pic_shape.width, old_pic_shape.height
    old_pic_shape._element.getparent().remove(old_pic_shape._element)
    slide.shapes.add_picture(new_path, left, top, width=width, height=height)


def find_shapes(slide):
    texts, pic, table = [], None, None
    for sh in slide.shapes:
        if sh.shape_type == 13:
            pic = sh
        elif sh.shape_type == 19:
            table = sh.table
        elif sh.has_text_frame:
            texts.append(sh)
    return texts, pic, table


prs = Presentation(DECK)

# ---------------------------------------------------------------- Slide 5: class-1 service rate
s = prs.slides[5]
texts, pic, table = find_shapes(s)
set_text(texts[1], "Class-1, threshold=0.5, min_retention_fraction=0.3, n=5 seeds (42,1,2,3,4) — all three configs complete")
replace_picture(s, pic, f"{SIL_DIR}/sil_class1_requestpruner_prefixlabel_final_comparison.png")
# table: add the bi400/ss80 row, refresh bi200/ss100's numbers (now truly n=5, not n=4)
set_table_cell(table, 1, 1, "74.5 ± 1.8%"); set_table_cell(table, 1, 2, "40.0 ± 5.9%")
set_table_cell(table, 1, 3, "60.9 ± 4.5%"); set_table_cell(table, 1, 4, "34.5 / 13.6pp"); set_table_cell(table, 1, 5, "61%")
set_table_cell(table, 2, 1, "51.8 ± 1.6%"); set_table_cell(table, 2, 2, "43.6 ± 6.8%")
set_table_cell(table, 2, 3, "55.6 ± 1.7%"); set_table_cell(table, 2, 4, "8.2 / −3.8pp"); set_table_cell(table, 2, 5, "146% (beats baseline)")
# clone row 2 as a template for the new bi400/ss80 row
tbl_el = table._tbl
new_tr = copy.deepcopy(tbl_el.tr_lst[2])
tbl_el.append(new_tr)
set_table_cell(table, 3, 0, "bi400/ss80")
set_table_cell(table, 3, 1, "76.0 ± 1.0%"); set_table_cell(table, 3, 2, "57.9 ± 8.7%")
set_table_cell(table, 3, 3, "81.8 ± 0.6%"); set_table_cell(table, 3, 4, "18.1 / −5.8pp"); set_table_cell(table, 3, 5, "132% (beats baseline)")
set_text(texts[2], "Expert-prefix labels beat the no-pruner baseline at bi400/ss80 and bi400/ss200, and recover 61% of the damage at bi200/ss100 — bi400/ss80 required a full HP re-sweep first (see Root Cause / Fix slides).")

# ---------------------------------------------------------------- Slide 6: runtime
s = prs.slides[6]
texts, pic, table = find_shapes(s)
replace_picture(s, pic, f"{SIL_DIR}/sil_class1_requestpruner_prefixlabel_final_comparison_runtime.png")
set_text(texts[2], "The expert-prefix-label pruner saves as much or more runtime than the RHO-label pruner at every config — largest at bi400/ss80 (42.2% vs. 11.5%), where it is also by far the most accurate.")

# ---------------------------------------------------------------- Slide 7: mixed-class
s = prs.slides[7]
texts, pic, table = find_shapes(s)
set_text(texts[1], "threshold=0.5, min_retention_fraction=0.3 — bi200/ss100 and bi400/ss200: n=5 seeds; bi400/ss80: pending (repaired pruner, seed=42 only so far)")
replace_picture(s, pic, f"{SIL_DIR}/sil_mixedclass_requestpruner_prefixlabel_final_comparison.png")
set_table_cell(table, 1, 1, "65.5 ± 1.3%"); set_table_cell(table, 1, 2, "29.0 ± 1.5%")
set_table_cell(table, 1, 3, "31.9 ± 2.3%"); set_table_cell(table, 1, 4, "36.5 / 33.6pp")
set_table_cell(table, 2, 1, "45.4 ± 1.4%"); set_table_cell(table, 2, 2, "25.0 ± 2.7%")
set_table_cell(table, 2, 3, "46.9 ± 3.5%"); set_table_cell(table, 2, 4, "20.3 / −1.6pp (beats baseline)")
set_text(texts[2], "With full 5-seed data, bi400/ss200 mixed-class confirms beating the no-pruner baseline (recovered damage >100%). "
                    "bi200/ss100 mixed-class recovers only 8% of the damage — the class-1 vs. mixed-class gap holds up under replication, not just a single-seed artifact anymore.")

prs.save(DECK)
print(f"saved {DECK}, {len(prs.slides)} slides (only 5, 6, 7 touched)")
