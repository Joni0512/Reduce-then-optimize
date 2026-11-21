"""
2026-11-19: SF request-pruner pre-check (user: "Request Pruner auch machen ... wie bei Li&Lim", step 1 "billiger
Vorab-Check, kein neuer Lauf"). Counts, from EXISTING RHO runs without any pruner, how many (request, window) rows
would be positive under both label definitions of request_pruner_dataset_builder.py:
  * assigned                    = request gets a vehicle in this window (Li&Lim default)
  * pickup_before_next_resolve  = assigned now AND pickup (final/result_driver_runs.json) < window t + step_size (NYC)
and how both relate to the time rule (keep iff the request was already assigned in an earlier window, or its
earliest_pickup_time < t + step_size; approximation of trip_handler's USE_REQUEST_TIME_RULE_FILTER without
TIME_RULE_DEFER_ACTIVE). Lesson from NYC (memory project_nyc_window_pruners_2026_11): "assigned" was 94-99 % positive
(nothing to prune), "pickup_before_next_resolve" ~= the time rule (P(1 | rule prunes) = 0).
Runs: outputs/new_tests/sf_lodes_nodepot/inst{1,2}_bi*_..._uniformroad*_s42 (pair-pruner label runs, instance seeds 1
(train) / 2 (val), start seed 42, mc2, 150 vehicles, time rule off, no depot return). If a config has two run dirs,
the one with more windows is used. pickup label only where final/result_driver_runs.json exists locally.
Output: figures_export/SF/sf_request_pruner_label_check.csv
"""
import csv, glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANI = ROOT / "solutions" / "sf" / "manifests"
OUT = ROOT / "figures_export" / "SF" / "sf_request_pruner_label_check.csv"


def status_lines(p):
    return [json.loads(l) for l in open(p) if json.loads(l).get("message") == "Status"]


def earliest_pickup(instance):
    reqs = json.load(open(MANI / f"{instance}.json"))["requests"]
    k = next(k for k in ("pickup_time_window_start", "pickup_time", "earliest_pickup_time") if k in reqs[0])
    return {int(r["booking_id"] if "booking_id" in r else r["id"]): float(r[k]) for r in reqs}, k


rows = []
for top in sorted(glob.glob(str(ROOT / "outputs/new_tests/sf_lodes_nodepot/inst*_bi*_uniformroad*_s42"))):
    name = Path(top).name
    inst, bi, ss = name.split("_")[0], int(name.split("_bi")[1].split("_")[0]), int(name.split("_ss")[1].split("_")[0])
    runs = glob.glob(f"{top}/run_offline_*/*/assignment_data.jsonl")
    if not runs:
        print("skip (no assignment_data.jsonl locally):", name); continue
    log = max(runs, key=lambda p: len(status_lines(p)))
    instance = Path(log).parent.name.rsplit("_", 2)[0]
    ep, key = earliest_pickup(instance)
    final = Path(log).parent / "final" / "result_driver_runs.json"
    pick = None
    if final.exists():
        fr = json.load(open(final)); fr = fr["driver_runs"] if isinstance(fr, dict) else fr
        pick = {int(s["booking_id"]): s["scheduled_time"] for d in fr for s in d["manifest"] if s["action"] == "pickup" and int(s["booking_id"]) >= 0}
    seen_assigned = set()
    n = a1 = p1 = keep = a1_keep = a1_prune = p1_keep = p1_prune = 0
    win = 0
    for e in status_lines(log):
        st = e["extra"]["status"]; t = e["extra"]["timestamp"]
        assigned = {int(r) for r in st["assigned_requests"]}; active = assigned | {int(r) for r in st["unserved_requests"]}
        if not active: continue
        win += 1
        for r in active:
            a = r in assigned
            p = a and pick is not None and r in pick and pick[r] < t + ss
            k = r in seen_assigned or ep[r] < t + ss
            n += 1; a1 += a; p1 += p; keep += k
            if k: a1_keep += a; p1_keep += p
            else: a1_prune += a; p1_prune += p
        seen_assigned |= assigned
    pr = n - keep
    rows.append({"instance": inst, "batch_interval": bi, "step_size": ss, "rtv21600": "rtv21600" in name, "windows": win, "rows": n,
                 "requests_per_window": round(n / win, 1), "pos_assigned": round(a1 / n, 4),
                 "pos_pickup_before_next": round(p1 / n, 4) if pick else "",
                 "time_rule_keeps": round(keep / n, 4),
                 "P_assigned_given_rule_keeps": round(a1_keep / keep, 4) if keep else "",
                 "P_assigned_given_rule_prunes": round(a1_prune / pr, 4) if pr else "",
                 "P_pickup_given_rule_keeps": round(p1_keep / keep, 4) if (pick and keep) else "",
                 "P_pickup_given_rule_prunes": round(p1_prune / pr, 4) if (pick and pr) else "",
                 "earliest_pickup_field": key, "log": str(Path(log).relative_to(ROOT))})
rows.sort(key=lambda r: (r["batch_interval"], r["instance"]))
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print({k: v for k, v in r.items() if k != "log"})
