"""2026-11-04: mean +- SD over seeds 1-3 of the window-graph pair-pruner grid (outputs/pair_pruner_window_rho3060/*.json)."""
import glob, json
from collections import defaultdict
import numpy as np

runs = defaultdict(list)
for f in sorted(glob.glob("outputs/pair_pruner_window_rho3060/*.json")):
    r = json.load(open(f)); h, b = r["hyperparameters"], r["best"]
    runs[(h["message_passing_steps"], h["pos_weight"], h["learning_rate"])].append((b, r["runtime_s"]["training"]))
keys = ["val_auc", "val_mean_window_auc", "val_recall_at_95pct_pruned", "val_kept_share_at_0.5", "val_recall_at_0.5"]
print(f"{'layers':>6}{'pw':>4}{'lr':>7}{'n':>3}" + "".join(f"{k.replace('val_',''):>28}" for k in keys) + f"{'train_s':>9}")
rows = []
for (L, pw, lr), v in runs.items():
    cells = [np.array([b[k] for b, _ in v]) for k in keys]
    rows.append((cells[0].mean(), L, pw, lr, len(v), cells, np.mean([t for _, t in v])))
for _, L, pw, lr, n, cells, t in sorted(rows, key=lambda x: -x[0]):
    print(f"{L:>6}{pw:>4g}{lr:>7g}{n:>3}" + "".join(f"{c.mean():>17.3f} ± {c.std(ddof=1):.3f}" for c in cells) + f"{t:>9.0f}")
