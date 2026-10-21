"""
2026-09-23: Tests the hypothesis raised in chat for why the expert-prefix-label Request
Pruner recovers much more SIL damage at bi400/ss200 (class-1: beats baseline; mixed:
beats baseline) than at bi200/ss100 (class-1: 63% of damage recovered; mixed: only 8%).

Hypothesis: the prefix label is computed from RHO-baseline batches (see
build_request_pruner_prefix_dataset.py), which is EXACT only in window 0 - from window 1
onward the real SIL trajectory (mode="eval" during validation) diverges from the RHO
baseline as soon as anything gets pruned/assigned differently, so the label becomes a
worse approximation the more windows an instance has. bi200/ss100 has ~312 eval-mode
windows (class-1, 6 val instances, 5 epochs) vs. ~174 for bi400/ss200 - almost 2x -
which could explain the difference IF divergence actually grows with window index.

Method (fully offline, no new cluster run):
  1. RHO-baseline batch per (instance, window_index): request IDs already stored in
     dataset/request_pruning_{cfg}/{split}/{instance}.npz (node_ids, window_index) -
     built once from the RHO baseline logs, used for both the RHO-label and
     prefix-label datasets.
  2. Real SIL-eval batch per (instance, window_index): parsed directly from an existing
     synced prefix-label SIL run's main.log, "=== Iteration {epoch}-{window} COAML
     Solver (mode: eval)" blocks (epoch 5 = the reported result's checkpoint), reading
     the following "N new and already assigned request(s) in payload - [ids]" line.
  3. Jaccard similarity between the two request-ID sets per window index -> how much the
     real trajectory has diverged from the RHO-baseline trajectory the label was built
     from. Plotted/tabulated vs. window index, per config.

Limitation: only covers the (instance, window_index) pairs present in BOTH the dataset
and the SIL log (val instances only, one seed per config, epoch 5 only).

Run: python -m rtv_solver.pipeline.analyze_prefix_label_batch_divergence
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

VAL_CLASS1 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]

CONFIGS = {
    "bi200_ss100": {
        "dataset_dir": Path("dataset/request_pruning_bi200_ss100"),
        "sil_log": Path("sil_training_bi200_ss100_class1_requestpruner_prefixlabel_t0.5_seed42.log"),
    },
    "bi400_ss200": {
        "dataset_dir": Path("dataset/request_pruning_bi400_ss200"),
        "sil_log": Path("sil_training_bi400_ss200_class1_requestpruner_prefixlabel_t0.5_seed42.log"),
    },
}

ITER_RE = re.compile(r"Iteration (\d+)-(\d+) COAML Solver \(mode: eval\)")
PAYLOAD_RE = re.compile(r"\d+ new and already assigned request\(s\) in payload - \[([^\]]*)\]")


def rho_baseline_batches(dataset_dir: Path, instances: list[str]) -> dict:
    """{instance: {window_index: set(request_ids)}}"""
    out = {}
    for inst in instances:
        for split in ("train", "val", "test"):
            p = dataset_dir / split / f"{inst}.npz"
            if p.exists():
                z = np.load(p)
                d = {}
                for wid, rid in zip(z["window_index"], z["node_ids"]):
                    d.setdefault(int(wid), set()).add(int(rid))
                out[inst] = d
                break
    return out


def sil_eval_batches(log_path: Path, instances: list[str], epoch: int = 5) -> dict:
    """{instance: {window_index: set(request_ids)}} - from the given epoch's validation
    pass for each instance.

    2026-09-23: two earlier attempts at this both got the instance/window attribution
    wrong (see git history of this function) because "Loading input data" block
    boundaries don't line up cleanly with epoch boundaries: the ~23-instance TRAINING
    loop between two epochs' validation passes is untracked, and window indices RESET
    to 0 for every new instance within the same epoch, so any block-boundary heuristic
    based only on the 6 tracked val instances' own load-line positions either bleeds
    into a neighboring instance's data or crosses an epoch boundary. Fixed by a single
    linear scan that tracks "current instance" from EVERY manifest load (train instances
    included, just not recorded) and "current epoch/window" from Iteration lines - so
    instance identity is always correctly attributed regardless of which instance comes
    next. Verified against the RHO-baseline dataset: window 0 must be IDENTICAL by
    construction (same fixed initial payload, no assignment has happened yet) - this is
    the correctness check used to catch both earlier bugs.
    """
    LOAD_RE = re.compile(r"manifests/(\w+)\.json")
    out = {inst: {} for inst in instances}
    current_instance = None
    cur_epoch = cur_window = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m_load = LOAD_RE.search(line)
        if m_load:
            current_instance = m_load.group(1)
            cur_epoch = cur_window = None  # new instance -> any pending state is stale
            continue
        m_iter = ITER_RE.search(line)
        if m_iter:
            cur_epoch, cur_window = int(m_iter.group(1)), int(m_iter.group(2))
            continue
        if cur_epoch == epoch and current_instance in instances:
            m_pl = PAYLOAD_RE.search(line)
            if m_pl:
                ids = {int(x) for x in m_pl.group(1).split(",") if x.strip()}
                out[current_instance][cur_window] = ids
                cur_epoch = None  # consumed - avoid a later stray line matching again
    return out


def main():
    rows = []
    for cfg, info in CONFIGS.items():
        if not info["sil_log"].exists():
            print(f"[skip] {cfg}: log not synced locally ({info['sil_log']})")
            continue
        rho = rho_baseline_batches(info["dataset_dir"], VAL_CLASS1)
        sil = sil_eval_batches(info["sil_log"], VAL_CLASS1)
        for inst in VAL_CLASS1:
            common_windows = sorted(set(rho.get(inst, {})) & set(sil.get(inst, {})))
            for w in common_windows:
                a, b = rho[inst][w], sil[inst][w]
                union = a | b
                jaccard = len(a & b) / len(union) if union else 1.0
                rows.append({"config": cfg, "instance": inst, "window": w,
                             "jaccard": jaccard, "rho_size": len(a), "sil_size": len(b)})

    if not rows:
        print("No data - check that the SIL logs listed in CONFIGS are synced locally.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("outputs/prefix_label_batch_divergence.csv", index=False)

    print("Mean batch-overlap (Jaccard similarity, RHO-baseline batch vs. real SIL eval-mode "
          "batch) by window index bin, per config - LOWER means the prefix label (computed "
          "from the RHO-baseline batch) is a worse approximation of what the real SIL run saw.\n")
    df["window_bin"] = pd.cut(df["window"], [-1, 0, 1, 2, 4, 100],
                               labels=["w0 (exact by construction)", "w1", "w2", "w3-4", "w5+"])
    piv = df.groupby(["config", "window_bin"], observed=True)["jaccard"].agg(["mean", "count"])
    print(piv.to_string())

    print("\nOverall mean Jaccard per config (all windows):")
    print(df.groupby("config")["jaccard"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
