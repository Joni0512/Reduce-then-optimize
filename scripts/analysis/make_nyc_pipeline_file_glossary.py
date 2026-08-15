"""Renders a glossary of the key NYC RHO/SIL pipeline files as a PNG table.

Uses manual row placement (not ax.table) because ax.table does not size rows
to multi-line cell content, causing severe text overlap with long descriptions.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sections = [
    ("Manifeste (Daten)", [
        ("..._v50.json\n(leeres Solve-\nManifest)",
         "requests + depot + travel_time_matrix + driver_runs mit state=frischer\n"
         "Fahrzeugstart, manifest=[] (leer). Von build_nyc_solve_manifest.py gebaut."),
        ("..._v50_expert.json\n(Experten-\nManifest)",
         "Gleiche requests/depot/ttm wie oben, aber driver_runs.manifest ist mit\n"
         "einer geloesten Route vorbefuellt (y*-Ziel fuer SIL-Imitation). Von\n"
         "build_sil_expert_manifest.py gemerged."),
        ("result_driver_\nruns.json",
         "Output eines RHO-Laufs: state = Simulations-Endzustand (spaet am Tag!),\n"
         "manifest = die vom Solver gefundene Route. NIE direkt als --input_file\n"
         "nehmen (state passt nicht zu leerem manifest -> ~0% Service Rate)."),
    ]),
    ("Ergebnisse eines Laufs", [
        ("results.json",
         "Statistiken: serviced, total_requests, total_time (Runtime),\n"
         "vmt/pmt, per_vehicle_distance_m, etc."),
        ("config.json",
         "Alle Config-Parameter des Laufs (Seed, step_size, batch_interval,\n"
         "USE_REQUEST_GRAPH_PRUNER, LEARNING_RATE, ...) + Git-Commit-Hash."),
    ]),
    ("Wichtige Skripte", [
        ("main.py",
         "Einstiegspunkt. --mode offline (RHO) / coaml (SIL) / online.\n"
         "Ruft PayloadParser.clear_vehicle_manifests() vor jedem Solve auf."),
        ("build_nyc_solve_\nmanifest.py",
         "Baut das leere Solve-Manifest (..._v50.json) aus den NYC-Rohdaten\n"
         "(requests_20.csv) fuer einen Kalendertag."),
        ("build_sil_expert_\nmanifest.py",
         "Merged state (aus leerem Solve-Manifest) + manifest (aus RHO-Result)\n"
         "zu einem neuen Experten-Manifest fuer SIL-Training."),
        ("training_loop.py",
         "COAMLTrainingLoop._run_train_val_payloads(): SIL-Trainingsschleife\n"
         "(NYC-Einzeltag-Pfad). Nutzt --input_file zweimal: als geleerten\n"
         "Solve-State UND (unveraendert) als imitation_solution_path fuer y*."),
        ("request_graph_\nfeature_builder.py",
         "Node-/Edge-Features fuer den Pair-Pruner-GNN (Distanz, Zeitfenster-\n"
         "Ueberlappung, Richtung, ...). geographic=True -> Haversine (NYC)."),
    ]),
]

NAVY = "#1E2761"
SECTION_BG = "#DCE3F5"
ROW_BG_A = "#FFFFFF"
ROW_BG_B = "#F5F6FA"
BORDER = "#D8D8D8"

LINE_H = 0.30          # height per text line, in data units
ROW_PAD = 0.16          # vertical padding per row (top+bottom combined)
SECTION_H = 0.55
COL0_W = 3.0            # file/script name column width
COL1_W = 9.3            # description column width
TOTAL_W = COL0_W + COL1_W

# pre-compute total figure height
total_h = 0.9  # top margin for title
for section_title, rows in sections:
    total_h += SECTION_H
    for name, desc in rows:
        n_lines = max(name.count("\n"), desc.count("\n")) + 1
        total_h += n_lines * LINE_H + ROW_PAD

fig_w = 13.0
fig_h = total_h * 0.62
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, TOTAL_W)
ax.set_ylim(0, total_h)
ax.invert_yaxis()
ax.axis("off")

y = 0.55  # start below title margin
for section_title, rows in sections:
    ax.add_patch(Rectangle((0, y), TOTAL_W, SECTION_H, facecolor=SECTION_BG, edgecolor=BORDER, linewidth=0.8))
    ax.text(0.12, y + SECTION_H / 2, section_title, va="center", ha="left",
            fontsize=13, fontweight="bold", color=NAVY)
    y += SECTION_H

    for i, (name, desc) in enumerate(rows):
        n_lines = max(name.count("\n"), desc.count("\n")) + 1
        row_h = n_lines * LINE_H + ROW_PAD
        bg = ROW_BG_A if i % 2 == 0 else ROW_BG_B

        ax.add_patch(Rectangle((0, y), COL0_W, row_h, facecolor=bg, edgecolor=BORDER, linewidth=0.6))
        ax.add_patch(Rectangle((COL0_W, y), COL1_W, row_h, facecolor=bg, edgecolor=BORDER, linewidth=0.6))

        ax.text(0.12, y + row_h / 2, name, va="center", ha="left",
                fontsize=10.5, fontweight="bold", family="monospace", color="#111111")
        ax.text(COL0_W + 0.15, y + row_h / 2, desc, va="center", ha="left",
                fontsize=10, color="#333333")

        y += row_h

ax.text(TOTAL_W / 2, 0.05, "NYC RHO/SIL Pipeline — Wichtige Dateien und ihr Zweck",
        ha="center", va="top", fontsize=16, fontweight="bold", color=NAVY)

plt.tight_layout()
out_path = "figures_export/nyc_pipeline_file_glossary.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print("saved:", out_path)
