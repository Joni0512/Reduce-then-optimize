---
name: figures-export-organization
description: Whenever a figure, chart, table export, or slide file is saved into figures_export/, put it directly into the matching topic subfolder - never loose at the figures_export/ root. Use this any time a plot/table/deck is about to be saved under figures_export, or when the user mentions figures_export being messy/disorganized.
---

# Never drop files loose into figures_export/ root

`figures_export/` already has topic subfolders - use them, don't add to the
root. As of 2026-09-14 there were 5184 loose files at the root despite these
folders existing, because new exports kept skipping them. Don't repeat that.

## Existing topic subfolders (check for more before assuming this list is complete)

| Folder | Covers |
|---|---|
| `SRL/` | Structured RL: critic training, actor-critic sweeps, TD-bootstrap, replay buffer, target-critic, GAT/GCN aggregator comparisons, reward_mode comparisons |
| `NYC/` | NYC dataset-specific results (RHO on NYC data, NYC-specific pruners, dataset stats) |
| `LiLim_ReduceThenOptimize_SIL/` | SIL (behavior cloning) results on Li&Lim instances |
| `Request Pruner RHO/`, `Request Pruner SIL/` | Request-pruner GNN results, split by which pipeline generated them |
| `Pair Pruner RHO/`, `Pair Pruner SIL/` | Pair-pruner GNN results, same RHO/SIL split |

## How to pick the right folder for a NEW file

Match by what the result is actually about, not just filename prefix
(prefixes are a strong hint but confirm against content):
- Actor-critic / SRL training curves, sweeps, TD-bootstrap, GAT/GCN critic
  comparisons, reward_mode comparisons → `SRL/`
- Anything computed on NYC data specifically → `NYC/`
- Plain SIL (imitation learning, no critic) results on Li&Lim → `LiLim_ReduceThenOptimize_SIL/`
- Request/pair pruner GNN metrics → the matching `Request Pruner <RHO|SIL>/`
  or `Pair Pruner <RHO|SIL>/` (RHO = generated via the rolling-horizon
  pipeline, SIL = generated via the imitation-learning pipeline)
- Genuinely new topic with no existing folder (e.g. the RHO-outcome-advantage
  method once it exists) → ask whether to create a new subfolder rather than
  defaulting to root or silently reusing the closest existing one.

## What this means in practice

- When a script/plot-generation call writes to `figures_export/...`, always
  include the topic subfolder in the path - e.g.
  `figures_export/SRL/srl_training_local_vs_local_positive_chart.png`, never
  `figures_export/srl_training_local_vs_local_positive_chart.png`.
- Both the PNG and PDF (see `experiment-results-export` skill) go in the
  SAME subfolder, same basename.
- A `.pptx`/`.tex` file that's a byproduct of a specific topic's results
  also goes in that topic's subfolder, not the root - e.g. a one-off slide
  export about an SRL finding belongs in `SRL/`, even though the main decks
  (`srl_meeting_deck.pptx`, `srl_design_overview_en.pptx`) stay at repo
  root/their own established location (see `results-into-slides` skill).

## The existing 5184-file backlog

Not cleaned up as part of adding this skill - reorganizing thousands of
existing loose files risks breaking references from other documents/decks
that already point at their current root-level paths, and needs the user's
explicit go-ahead before moving files at that scale. Only clean up backlog
files as a distinct, explicitly-requested task, not as a side effect of
saving something new.
