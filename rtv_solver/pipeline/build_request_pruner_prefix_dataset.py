"""
2026-09-21: Request-pruner dataset with EXPERT-PREFIX labels instead of RHO labels
(bi200/ss100, cardinality 2).

Why: SIL's legacy imitation scoring only rewards a trip that is a contiguous PREFIX
of the vehicle's reference route projected onto the full window batch
(imitation_handler.score_combinations_legacy). A pruned first-expert-request leaves that
vehicle without any positive trip. The existing pruner was trained on RHO labels
("assigned in this window") and, in window 0, prunes the first expert request of ~49% of
the vehicles at min_retention 0.3 (vs. 17% of the RHO-assigned requests). The Pair Pruner
is trained on the reference solution and is ~neutral in SIL.

Label (per window w, per vehicle v): project v's reference-route pickup order onto the
requests present in w's batch; the first K entries get label 1 (K = max cardinality, 2
here), everything else in the batch gets 0. Requests of different vehicles are disjoint.
Reference routes serve ALL requests, so "in the route" alone would be constant 1.

Same rows/features/window structure as the RHO dataset (dataset/request_pruning_bi200_ss100),
only `label` is replaced. Limitation: the window batches are those of the RHO-baseline
trajectory (exact in window 0; from window 1 on an approximation of what a SIL run sees,
because the state diverges as soon as anything is assigned/pruned differently).

Run: python -m rtv_solver.pipeline.build_request_pruner_prefix_dataset
"""
import json
from pathlib import Path

import numpy as np

SRC_DIR = Path("dataset/request_pruning_bi200_ss100")
DST_DIR = Path("dataset/request_pruning_bi200_ss100_prefix")
MANIFEST_DIR = Path("solutions/li_lim/manifests")
K = 2  # = MAX_CARDINALITY of the SIL runs


def expert_by_vehicle(stem: str, manifest_dir: Path = MANIFEST_DIR) -> dict:
    """{vehicle_id: pickup order of the reference route}."""
    d = json.load(open(manifest_dir / f"{stem}.json"))
    out = {}
    for dr in d["driver_runs"]:
        vid = dr["manifest"][0]["run_id"] if dr["manifest"] else dr["state"]["run_id"]
        out[vid] = [e["booking_id"] for e in dr["manifest"] if e["action"] == "pickup"]
    return out


def prefix_labels(ids: np.ndarray, win: np.ndarray, stem: str, k: int = K) -> np.ndarray:
    label = np.zeros(len(ids), dtype=np.float32)
    routes = expert_by_vehicle(stem)
    for w in np.unique(win):
        idx = np.where(win == w)[0]
        batch = {int(ids[i]) for i in idx}
        prefix = set()
        #Here is the projection of the reference route onto the batch and we take the first k requests of that projection as label 1.
        for seq in routes.values():
            proj = [r for r in seq if r in batch]
            prefix.update(proj[:k])
        for i in idx:
            if int(ids[i]) in prefix:
                label[i] = 1.0
    return label


def main():
    manifest = json.load(open(SRC_DIR / "manifest.json"))
    manifest["label_definition"] = (
        f"1.0 if the request is among the first K={K} requests of any vehicle's reference route "
        "projected onto this window's batch (what legacy imitation scoring can reward), else 0.0. "
        "Batches = RHO-baseline windows (exact in window 0, approximation afterwards). "
        "Rows/features identical to dataset/request_pruning_bi200_ss100."
    )
    manifest["source_dataset"] = str(SRC_DIR)
    for split in ("train", "val", "test"):
        (DST_DIR / split).mkdir(parents=True, exist_ok=True)
        pos_new = pos_old = rows = 0
        for p in sorted((SRC_DIR / split).glob("*.npz")):
            z = np.load(p)
            new = prefix_labels(z["node_ids"], z["window_index"], p.stem)
            np.savez(
                DST_DIR / split / p.name,
                node_ids=z["node_ids"], window_index=z["window_index"],
                timestamp=z["timestamp"], label=new, features=z["features"],
            )
            pos_new += new.sum(); pos_old += z["label"].sum(); rows += len(new)
            if p.stem in manifest["instances"]:
                manifest["instances"][p.stem]["positive_rate"] = float(new.mean())
        print(f"{split:5s}: rows={rows:6d}  positive rate prefix={pos_new / rows:.3f}  (RHO label was {pos_old / rows:.3f})")
    json.dump(manifest, open(DST_DIR / "manifest.json", "w"), indent=2)
    print(f"written to {DST_DIR}")


if __name__ == "__main__":
    main()
