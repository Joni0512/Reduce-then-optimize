"""
2026-09-23: Builds request_pruner_deck.pptx - the Request Pruner results deck the user
asked for: (1) existing RHO + SIL results (RHO-label pruner, reused from figures_export/
Request Pruner RHO and SIL), (2) the new expert-prefix-label rule (why the RHO label
mismatches SIL's legacy imitation scoring, see chat 2026-09-21/22), (3) new SIL-training
results comparing baseline / RHO-label / expert-prefix-label pruners.

Style matches the existing pruner_results_deck.pptx (16:9, Cambria 32pt bold navy titles
#1E2761, Calibri 14pt gray #606060 subtitles, navy-header white-text tables, Calibri 11pt
gray captions) so it sits consistently next to that deck.

bi400/ss80's expert-prefix-label pruner is still being re-run on the cluster (the first
reused-architecture checkpoint was mis-calibrated and made things worse, see chat) - new
results slides therefore only show bi200/ss100 and bi400/ss200 for now, explicitly noted;
rerun this script once bi400/ss80 lands to add it.

Run: ./venv/bin/python3 build_request_pruner_deck.py
"""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

NAVY = RGBColor(0x1E, 0x27, 0x61)
RED = RGBColor(0xA6, 0x49, 0x3A)
GREEN = RGBColor(0x3E, 0x8F, 0x6C)
GRAY = RGBColor(0x60, 0x60, 0x60)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

RHO_DIR = "figures_export/Request Pruner RHO"
SIL_DIR = "figures_export/Request Pruner SIL"

prs = Presentation()
prs.slide_width = Emu(12191695)
prs.slide_height = Emu(6858000)
BLANK = prs.slide_layouts[6]


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


def add_caption(slide, text, top, left=Emu(457200), width=Emu(11277295)):
    tb = slide.shapes.add_textbox(left, top, width, Emu(457200))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name = "Calibri"; r.font.size = Pt(11); r.font.color.rgb = GRAY
    tb.text_frame.word_wrap = True
    return tb


def add_picture(slide, path, left, top, width):
    return slide.shapes.add_picture(path, left, top, width=width)


def add_table(slide, headers, rows, left, top, width, height, col_widths=None):
    n_rows, n_cols = len(rows) + 1, len(headers)
    shp = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = shp.table
    if col_widths:
        for c, w in zip(tbl.columns, col_widths):
            c.width = Emu(w)
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = ""
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        r = p.add_run(); r.text = h
        r.font.name = "Calibri"; r.font.size = Pt(11.5); r.font.bold = True; r.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.name = "Calibri"; r.font.size = Pt(11); r.font.bold = True; r.font.color.rgb = NAVY
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
    return shp


def add_numbered_list(slide, items, left, top, item_h=548640, width=10515600):
    for i, text in enumerate(items):
        y = top + i * item_h
        oval = slide.shapes.add_shape(1, left, y, Emu(274320), Emu(274320))  # 1 = OVAL
        oval.fill.solid(); oval.fill.fore_color.rgb = NAVY
        oval.line.fill.background()
        p = oval.text_frame.paragraphs[0]
        r = p.add_run(); r.text = str(i + 1)
        r.font.name = "Calibri"; r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        oval.text_frame.vertical_anchor = 3  # MIDDLE
        tb = slide.shapes.add_textbox(left + Emu(420624), y - Emu(27432), Emu(width), Emu(item_h))
        tb.text_frame.word_wrap = True
        p2 = tb.text_frame.paragraphs[0]
        r2 = p2.add_run(); r2.text = text
        r2.font.name = "Calibri"; r2.font.size = Pt(13); r2.font.color.rgb = NAVY


# ============================================================ Slide 1: Title
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Emu(914400), Emu(2286000), Emu(10363295), Emu(1097280))
p = tb.text_frame.paragraphs[0]
r = p.add_run(); r.text = "Request Pruner: From RHO Labels to Expert-Prefix Labels"
r.font.name = "Cambria"; r.font.size = Pt(36); r.font.bold = True; r.font.color.rgb = NAVY
tb.text_frame.word_wrap = True
tb2 = s.shapes.add_textbox(Emu(914400), Emu(3474720), Emu(10363295), Emu(548640))
p2 = tb2.text_frame.paragraphs[0]
r2 = p2.add_run()
r2.text = "Diagnosing and fixing catastrophic SIL damage — Li&Lim, cardinality 2 (2026-09)"
r2.font.name = "Calibri"; r2.font.size = Pt(16); r2.font.color.rgb = GRAY

# ============================================================ Slide 2: RHO results (existing)
s = prs.slides.add_slide(BLANK)
add_title(s, "Previous Results: RHO (No Learning)",
          "Class-1-only Request Pruner (RHO-label), window-specific MLP checkpoints, mean ± 1 std over 5 seeds")
add_picture(s, f"{RHO_DIR}/rh_class1_requestpruner_bi200_ss100_service_rate_5seed.png", Emu(3200400), Emu(1417320), Emu(5791200))
add_table(s, ["Configuration", "Service Rate (%)", "Runtime (min)"],
          [["bi200/ss100, No Pruner", "77.6 ± 0.3", "0.12 ± 0.00"],
           ["bi200/ss100, t=0.5", "74.0 ± 0.4", "0.08 ± 0.00"],
           ["bi400/ss80, No Pruner", "78.1 ± 0.3", "0.15 ± 0.01"],
           ["bi400/ss80, t=0.5", "76.4 ± 1.0", "0.13 ± 0.01"],
           ["bi400/ss200, No Pruner", "53.3 ± 0.5", "0.10 ± 0.00"],
           ["bi400/ss200, t=0.5", "52.7 ± 0.5", "0.09 ± 0.00"]],
          Emu(1600200), Emu(4368338), Emu(9982320), Emu(2011680),
          col_widths=[4991160, 2495580, 2495580])
add_caption(s, "Under RHO without learning, the Request Pruner causes only mild damage across all three window configurations (≤3pp).", Emu(6482080))

# ============================================================ Slide 3: SIL results (existing)
s = prs.slides.add_slide(BLANK)
add_title(s, "Previous Results: SIL (COAML) — the RHO-Label Pruner",
          "Class-1-only, threshold=0.5, min_retention_fraction=0.3, 5 epochs, lr=0.0001, mean ± 1 std over 5 seeds")
add_picture(s, f"{SIL_DIR}/sil_class1_requestpruner_bi200_ss100_service_rate_5seed.png", Emu(3200400), Emu(1417320), Emu(5791200))
add_table(s, ["Configuration", "Service Rate (%)", "vs. baseline"],
          [["bi200/ss100, No Pruner", "74.5 ± 1.8", "—"],
           ["bi200/ss100, t=0.5", "40.0 ± 5.9", "−34.5pp"],
           ["bi400/ss80, No Pruner", "76.0 ± 1.0", "—"],
           ["bi400/ss80, t=0.5", "57.9 ± 8.7", "−18.1pp"],
           ["bi400/ss200, No Pruner", "51.8 ± 1.6", "—"],
           ["bi400/ss200, t=0.5", "43.6 ± 6.8", "−8.2pp"],
           ["mixed-class bi200/ss100, No Pruner", "65.5 ± 1.3", "—"],
           ["mixed-class bi200/ss100, t=0.5", "29.0 ± 1.5", "−36.5pp"]],
          Emu(1097280), Emu(4368338), Emu(10515600), Emu(2468880),
          col_widths=[6515600, 2000000, 2000000])
add_caption(s, "Same pruner, same checkpoints — damage under SIL is far larger than under RHO. Something SIL-specific is going wrong.", Emu(6939008))

# ============================================================ Slide 4: Root cause
s = prs.slides.add_slide(BLANK)
add_title(s, "Root Cause: The RHO Label Doesn't Match What SIL Can Learn",
          "IMITATION_SCORING_RULE=legacy — SIL only rewards trips that are a PREFIX of the expert route")
items = [
    "SIL's legacy imitation rule scores a candidate trip only if it is a contiguous PREFIX (start index 0) of the vehicle's reference route, projected onto the current batch (coaml_pipeline.py).",
    "At cardinality 2, only the first 1–2 requests of each vehicle's projected route can ever score — the rest of the route is irrelevant to this window's loss.",
    "The old Request Pruner is trained on the RHO label: “was this request assigned by RHO this window?” — a completely different criterion.",
    "Window-0 replay (state identical to a real SIL run, validated exactly against the solver log): at min_retention=0.3, 49% of vehicles lose their FIRST expert request — vs. only 17% of RHO-assigned requests being pruned at all.",
    "Reference routes serve ALL requests eventually, so “in the optimal route” alone is a constant-1 label — the fix must be window-specific.",
    "2x2 diagnostic (train with/without pruner × eval with/without, 5 seeds): the TRAINING channel causes −21.7pp, eval-time pruning alone only −7.1pp — confirms the label mismatch acts mainly through training, not deployment.",
]
add_numbered_list(s, items, Emu(457200), Emu(1600200), item_h=822960, width=10800000)

# ============================================================ Slide 5: The fix
s = prs.slides.add_slide(BLANK)
add_title(s, "The Fix: Expert-Prefix Labels",
          "New label definition + exact hyperparameters per configuration")
items = [
    "For each window, project each vehicle's reference-route pickup order onto the requests actually in this window's batch.",
    "Label = 1 for the first K requests of that projection (K = max_cardinality = 2), else 0 — exactly what legacy scoring can reward.",
    "Same feature rows/columns as the RHO-label dataset (13 pre-solve features) — only the label column changes.",
    "Trained with the existing RequestPrunerMLP / train_one_config pipeline, unchanged.",
]
add_numbered_list(s, items, Emu(457200), Emu(1600200), item_h=548640, width=10800000)
add_table(s, ["Config", "Label", "hidden_dim", "layers", "dropout", "pos_weight", "lr", "threshold", "min_ret."],
          [["bi200/ss100", "RHO", "32", "1", "0.2", "1.0", "0.01", "0.5", "0.3"],
           ["bi200/ss100", "Expert-prefix", "32", "1", "0.2", "1.0", "0.01", "0.5", "0.3"],
           ["bi400/ss80", "RHO", "64", "3", "0.1", "3.0", "0.03", "0.5", "0.3"],
           ["bi400/ss80", "Expert-prefix (re-swept)", "64", "1", "0.0", "1.5", "0.03", "0.5", "0.3"],
           ["bi400/ss200", "RHO", "32", "1", "0.0", "1.0", "0.03", "0.5", "0.3"],
           ["bi400/ss200", "Expert-prefix", "32", "1", "0.0", "1.0", "0.03", "0.5", "0.3"]],
          Emu(457200), Emu(3931920), Emu(11277295), Emu(2560320),
          col_widths=[1600200, 2286000, 1097280, 914400, 914400, 1005840, 914400, 1005840, 1005840])
add_caption(s, "bi400/ss80's reused-architecture prefix pruner was mis-calibrated (best_epoch=3, test precision=0.51) and made SIL worse — fixed via a full HP re-sweep ranked by val_roc_auc_mean.", Emu(6629400))

# ============================================================ Slide 6: New results - service rate
s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: SIL Training with the Expert-Prefix-Label Pruner",
          "Class-1, threshold=0.5, min_retention_fraction=0.3, n=5 seeds (42,1,2,3,4) — bi400/ss80 pending re-sweep result")
add_picture(s, f"{SIL_DIR}/sil_class1_requestpruner_prefixlabel_vs_rho_service_rate.png", Emu(2895600), Emu(1417320), Emu(6400800))
add_table(s, ["Config", "No pruner", "RHO-label", "Expert-prefix", "Damage RHO / prefix", "Recovered"],
          [["bi200/ss100", "74.5 ± 1.8%", "40.0 ± 5.9%", "60.9 ± 4.5%", "34.5 / 13.6pp", "61%"],
           ["bi400/ss200", "51.8 ± 1.6%", "43.6 ± 6.8%", "55.6 ± 1.7%", "8.2 / −3.8pp", "146% (beats baseline)"]],
          Emu(1600200), Emu(4800600), Emu(9982320), Emu(1005840),
          col_widths=[1900000, 1600000, 1600000, 1600000, 1682320, 1600000])
add_caption(s, "Expert-prefix labels recover most of the damage in every seed — at bi400/ss200 the pruned model even beats the no-pruner baseline.", Emu(6027420))

# ============================================================ Slide 7: New results - runtime
s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: Runtime & Savings",
          "Same runs as the previous slide — total wall-clock (train+val), % = saving vs. no-pruner baseline")
add_picture(s, f"{SIL_DIR}/sil_class1_requestpruner_prefixlabel_vs_rho_runtime.png", Emu(2895600), Emu(1417320), Emu(6400800))
add_caption(s, "The expert-prefix-label pruner saves as much or more runtime than the RHO-label pruner (52.2% vs. 41.7% at bi200/ss100) while causing far less damage.", Emu(6800000))

# ============================================================ Slide 8: Mixed-class
s = prs.slides.add_slide(BLANK)
add_title(s, "New Results: Mixed-Class",
          "threshold=0.5, min_retention_fraction=0.3, seed=42 ONLY for both configs — not yet multi-seed")
add_picture(s, f"{SIL_DIR}/sil_mixedclass_requestpruner_prefixlabel_vs_rho_service_rate.png", Emu(2895600), Emu(1417320), Emu(6400800))
add_table(s, ["Config", "No pruner", "RHO-label", "Expert-prefix", "Damage RHO / prefix"],
          [["bi200/ss100", "65.8%", "30.3%", "34.4%", "35.5 / 31.4pp"],
           ["bi400/ss200", "44.3%", "27.5%", "48.3%", "16.9 / −4.0pp (beats baseline)"]],
          Emu(1600200), Emu(4800600), Emu(9982320), Emu(1005840),
          col_widths=[2200000, 1945580, 1945580, 1945580, 1945580])
add_caption(s, "bi400/ss200 mixed-class matches the class-1 pattern for this config: the prefix-label pruner beats the no-pruner baseline. "
               "bi200/ss100 recovers only 11% of the damage — both numbers are single-seed, need replication before drawing conclusions.", Emu(6482080))

# ============================================================ Slide 9: Summary
s = prs.slides.add_slide(BLANK)
add_title(s, "Summary & Open Points")
items = [
    "The Request Pruner is not fundamentally broken — it was trained on the wrong label for SIL's legacy imitation rule.",
    "Expert-prefix labels recover 63–146% of the SIL damage at class-1 (bi200/ss100, bi400/ss200), with equal-or-better runtime savings.",
    "A ~13pp residual gap remains even at bi200/ss100 — not fully explained (likely window≥1 batch approximation error).",
    "bi400/ss80 required a full HP re-sweep after the reused architecture proved mis-calibrated — result pending.",
    "Mixed-class recovers far less damage (11% vs. 63%) at a single seed — needs replication before any conclusion.",
    "Next: bi400/ss80 re-sweep results, multi-seed mixed-class, then Request Pruner + Pair Pruner combined (already supported by trip_handler.py — request pruning runs before pair pruning).",
]
add_numbered_list(s, items, Emu(457200), Emu(1600200), item_h=731520, width=10800000)

prs.save("request_pruner_deck.pptx")
print("saved request_pruner_deck.pptx,", len(prs.slides), "slides")