---
name: cluster-job-cleanup
description: Check LRZ cluster job/coordinator status and free disk space once a run is done, keeping only the results that matter. Use whenever the user asks to check on a cluster run/sweep, mentions disk space or quota problems (local or cluster), or a training run has finished and its raw output is no longer needed. Also use proactively after confirming a wandb sweep/cluster job has completed, without waiting to be asked.
---

# Cluster job status + cleanup

Goal: don't make the user re-explain this every time. When a run is confirmed
done (or disk space is tight), act directly - check status, identify what to
keep vs. delete, then delete the disposable parts. Only ask first if it's
unclear whether a specific result is still needed (e.g. an unfinished/failed
run, or something that might be the ONLY copy of an unpublished result).

## 1. Check what's actually running (do this before touching anything)

```bash
squeue -M serial -u <user>
```

Cross-check against any expected coordinator processes - a cluster-side
coordinator/agent script normally runs on a LOGIN node (not a compute node,
since compute nodes here have no internet), detached via `disown -h` or a
`tmux`/`screen` session so it survives SSH disconnects. Check BOTH login
nodes (`cm4login1` AND `cm4login2`) - a coordinator started on one node is
invisible to `ps aux` on the other, and it's easy to end up with duplicate
coordinators against the same sweep if a fix was re-applied on the wrong
node without killing the original session first (this happened once this
session - see git log around 2026-09-13 for the srl-training-loop-sweep
duplicate-coordinator incident).

```bash
ps aux | grep <coordinator_script_name>   # run on EACH login node
tmux ls                                    # run on EACH login node
```

If a duplicate/zombie coordinator is found (same sweep_id, old code in
memory from before a fix was pulled): kill it (`tmux kill-session -t <name>`
or kill the PID), then `scancel -M serial <job_id>` for whatever job it
submitted. A run left in "running" state in wandb with a dead coordinator
behind it will never get a result reported - that's expected, not a bug;
leave it or mark it crashed in the wandb UI, don't try to resurrect it.

## 2. Check wandb for what's actually finished

```python
import wandb
api = wandb.Api()
sweep = api.sweep("<entity>/<project>/<sweep_id>")
for r in sweep.runs:
    print(r.name, r.state, r.summary.get("test_service_rate"))
```

A run in state `finished` has a reported metric - safe to consider done.
`running` = still going, don't touch its output dir. `failed`/`crashed` with
no metric = the compute-side result.json was likely never written (job
killed, e.g. by a time limit) - safe to clean up its (probably incomplete)
output dir.

## 3. What to KEEP per finished run (never delete these)

For this repo's SRL/SIL run outputs (`outputs/<run_type>/<run_id_or_label>/`):
- `result.json` (cluster trial summary) or `results.json`/`result_driver_runs.json` (single-instance runs)
- `*_curves.csv` / `*_curves.png` (training/validation curves - these ARE the thesis figures)
- `best_actor_checkpoint.pt` / `best_val_checkpoint.pt` / `coaml_model_weights_best_val.pt` (the actual trained model, needed to reproduce test numbers or run further eval)
- Any `.tex`/`.md`/slide export that was generated FROM the run (nothing to check here, just don't delete)

## 4. What's SAFE to delete once the above is confirmed present

- Per-instance/per-epoch intermediate manifest and log dirs under a run's
  `output_dir` (e.g. `train/epoch_*/`, `val/epoch_*/`, `critic_pretrain/*/`
  in `srl_training_loop.py`'s output layout) - these are solve_pdptw's raw
  working files, not referenced by anything once the curves/checkpoint exist.
- Oversized `.err`/`.log` files from crashed/verbose runs (this repo has a
  recurring pattern of accidentally-verbose per-iteration `print()` calls
  producing multi-GB `.err` files - see coaml_pipeline.py's known debug-print
  issue - these are pure noise, always safe to delete regardless of run status).
- `wandb/run-*` local sync directories once you've confirmed the run synced
  to the cloud (the data lives in wandb's servers, not just locally) - check
  with `wandb sync --view <dir>` if unsure, or just trust that a `finished`/
  `failed` state visible via the API means it already synced.
- Failed/crashed run output dirs with no `result.json` at all (nothing to keep).

## 5. Disk space triage (when quota/space is the actual complaint, not just tidiness)

Check both the Mac (`df -h /`, `diskutil apfs list` if `df` numbers look
inconsistent with a Data volume vs. system volume split) and the cluster
(`quota -s`, `du -sh ~/Reduce-then-optimize/* | sort -rh | head -15`).
Biggest recurring offenders in this repo, largest-impact first:
1. `outputs/` (can reach tens of GB - old completed experiments; only ask
   before deleting something here if it's NOT a finished run per step 3-4 above)
2. Stray multi-GB `.err`/`.log` files at the repo root (always safe, see step 4)
3. `wandb/` local run directories (safe once synced, see step 4)
4. macOS-only: local Time Machine snapshots eating the Data volume even
   though `df -h /` shows plenty of "Avail" - use `diskutil apfs list` to see
   real container free space, `tmutil thinlocalsnapshots` / delete named
   snapshot dates if that's the actual cause, not this repo's own files.

Act on 2-3 without asking whenever they're found during a status check -
they are unconditionally disposable. Only pause to ask before touching (1)
if it's ambiguous whether a specific instance/seed's results are still
needed for the thesis.
