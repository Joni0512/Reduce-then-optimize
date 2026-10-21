"""
2026-10-19: own accuracy of the FINAL c2 expert-prefix-label Request Pruner per instance GROUP (class x geometry) on the 12
mixedclass val/test instances (the pruner never saw them in training: they are its "test" split). Extends
rtv_solver/pipeline/analyze_prefix_pruner_own_accuracy.py (class-1 val only, three configs) to class 2 and to groups.
Same logic: threshold 0.5, min_retention 0.3, window 0 (exact state, no assignment yet) - per vehicle with a non-empty projected
reference route: is its TRUE first request pruned ("lost 1st target"), is any of its first K=2 requests pruned ("lost any of K")?
Plus the pruned share over all windows (replay of the RHO-baseline windows, approximate from window 1 on).
Difference to the original script: URGENT_SLACK is set to the config's step_size (the trip handler passes STEP_SIZE, the original
left the module default 100 for all configs).
Checkpoints/datasets: as in analyze_prefix_pruner_own_accuracy.py (c2 pruners, the ones used by the c2 mixedclass runs).
Run: ./venv/bin/python3 scripts/analysis/analyze_prefix_pruner_accuracy_by_group.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from rtv_solver.pipeline import analyze_request_pruner_false_prunes as A
from rtv_solver.pipeline.analyze_prefix_pruner_own_accuracy import CONFIGS, MIN_RETENTION, THRESHOLD
from rtv_solver.pipeline.build_request_pruner_prefix_dataset import K, expert_by_vehicle

TEST = ["lc108", "lc109", "lc207", "lc208", "lr111", "lr112", "lr210", "lr211", "lrc107", "lrc108", "lrc207", "lrc208"]
STEP = {"bi200_ss100": 100.0, "bi400_ss80": 80.0, "bi400_ss200": 200.0}


def group(stem):
    g = re.match(r"[a-z]+", stem).group()
    return (f"class{stem[len(g)]}", g)


def main():
    rows = []
    for cfg, info in CONFIGS.items():
        A.DATASET_DIR, A.CKPT, A.THRESHOLD, A.URGENT_SLACK = info["dataset_dir"], info["ckpt"], THRESHOLD, STEP[cfg]
        manifest = json.load(open(A.DATASET_DIR / "manifest.json"))
        feature_index = {n: i for i, n in enumerate(manifest["feature_names"])}
        model, ck = A.load_model()
        agg = defaultdict(lambda: dict(veh=0, lost1=0, lostK=0, cand=0, pruned=0, pos=0, pos_pruned=0))
        for stem in TEST:
            df = A.replay_instance(stem, model, ck, feature_index)
            pr_all, _ = A.apply_pruner(df, MIN_RETENTION)
            cand = ~(df["active"] | df["urgent"])
            a = agg[group(stem)]
            a["cand"] += int(cand.sum()); a["pruned"] += int(pr_all.sum())
            lab1 = cand & (df["label"] == 1.0)
            a["pos"] += int(lab1.sum()); a["pos_pruned"] += int((lab1 & pr_all).sum())
            w0 = df[df.window == 0].reset_index(drop=True)
            pw0, _ = A.apply_pruner(w0, MIN_RETENTION)
            pr = dict(zip(w0.request, pw0))
            batch = set(w0.request)
            for seq in expert_by_vehicle(stem).values():
                proj = [r for r in seq if r in batch]
                if not proj:
                    continue
                a["veh"] += 1
                a["lost1"] += int(pr[proj[0]])
                a["lostK"] += int(any(pr[r] for r in proj[:K]))
        for g, a in sorted(agg.items()):
            rows.append({"config": cfg, "class": g[0], "geometry": g[1], "vehicles_w0": a["veh"],
                         "lost_1st_target_%": 100 * a["lost1"] / max(a["veh"], 1),
                         "lost_any_of_K_%": 100 * a["lostK"] / max(a["veh"], 1),
                         "label1_pruned_%_all_windows": 100 * a["pos_pruned"] / max(a["pos"], 1),
                         "pruned_share_%_all_windows": 100 * a["pruned"] / max(a["cand"], 1)})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.float_format", "{:.1f}".format):
        print(df.to_string(index=False))
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/prefix_pruner_accuracy_by_group.csv", index=False)


if __name__ == "__main__":
    main()
