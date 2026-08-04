"""
Feature-group ablation for ScoringMLP (SIL trip-scoring MLP).

For each of the four feat_builder.py feature groups (state/vehicle/trip/tripcost)
plus the future-request grid, this neutralizes that group's columns at
model-scoring time (leaving the reject-action flag and every other column
untouched) and re-runs mode="eval" (the model's own learned policy, ILP
re-assignment included) on the six standard SIL validation files. Pooled
service rate is compared against the unmodified baseline. Uses the already
trained bi200_ss100_class1_legacy checkpoint - no retraining, no synthetic
data, only the real Li&Lim validation instances.
"""
from __future__ import annotations

import argparse
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn as nn

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.pipeline.feat_builder import (
    FeatureBuilder,
    StateFeatures,
    VehicleFeatures,
    TripFeatures,
    TripCostFeatures,
)
from rtv_solver.pipeline.model_simpleScoring import ScoringMLP
from rtv_solver.structure.config import Config
from rtv_solver.training_loop import (
    TRAINING_FILES,
    VALIDATION_FILES,
    _load_and_clear_payload,
    _split_train_val_files,
    _save_validation_results,
)
from rtv_solver.util.helper import save_json

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"
# Defaults reproduce the legacy-scoring-rule ablation; run_ablation() takes the
# checkpoint/label/scoring-rule as parameters so the exponential_prefix
# counterpart (same bi200/ss100/class1 setup, see
# run_sil_training_bi200_ss100_class1_exp_prefix.sh) can reuse this file
# instead of duplicating it.
CHECKPOINT = REPO_ROOT / (
    "outputs/sil_training_bi200_ss100_class1_legacy/batch_lilim_coaml_seed42/"
    "mc2_bi200_ss100_20260727_010701/coaml_model_weights_best_val.pt"
)
RUN_LABEL_BASE = "ablation_bi200_ss100_class1_legacy"
IMITATION_SCORING_RULE = "legacy"
# 2026-07-30: no timeout/gtimeout binary on this machine - cap each per-file
# solver instance at 30 min wall-clock via signal.alarm instead, skip on expiry.
PER_FILE_TIMEOUT_SEC = 1800

GRID_PREFIX = "fr_grid_"
GROUP_PREFIXES = {
    "state": "s_",
    "vehicle": "v_",
    "trip": "t_",
    "tripcost": "tc_",
    "grid": GRID_PREFIX,
}


def _neutral_values() -> dict[str, float]:
    """Per-feature neutral default, taken from each dataclass's own field
    defaults (e.g. v_norm_remaining_am_cap defaults to 1.0 = full capacity,
    not 0.0). Grid cells use 0.0 = no future demand, matching FeatureBuilder's
    own convention for an empty request list (add_reject_action_entries)."""
    values: dict[str, float] = {}
    for group_defaults in (StateFeatures(), VehicleFeatures(), TripFeatures(), TripCostFeatures()):
        values.update({k: float(v) for k, v in asdict(group_defaults).items()})
    for row in range(7):
        for col in range(7):
            values[f"{GRID_PREFIX}{row}_{col}"] = 0.0
    values[FeatureBuilder.REJECT_FLAG_FEATURE_NAME] = 0.0
    return values


NEUTRAL_VALUES = _neutral_values()
# Alphabetical order matches build_matrix()'s `sorted(name for name, value in
# example.items() ...)` - this key set is fixed regardless of instance content,
# so it can be derived once here instead of running a real trip_handler first.
FEATURE_NAMES = sorted(NEUTRAL_VALUES.keys())
assert len(FEATURE_NAMES) == FeatureBuilder.FEATURE_SIZE, (
    f"Derived {len(FEATURE_NAMES)} feature names, expected {FeatureBuilder.FEATURE_SIZE} - "
    "feat_builder.py's dataclasses changed since this script was written."
)


class AblatedScoringModel(nn.Module):
    """Wraps a trained ScoringMLP; overwrites one feature group's columns with
    their neutral default before scoring. `ablate_group=None` is a pass-through
    (baseline)."""

    def __init__(self, base_model: ScoringMLP, ablate_group: str | None):
        super().__init__()
        if ablate_group is not None and ablate_group not in GROUP_PREFIXES:
            raise ValueError(f"Unknown ablation group: {ablate_group}")
        self.base_model = base_model
        self.feature_dim = base_model.feature_dim
        self.hidden_dim = base_model.hidden_dim
        self.ablate_group = ablate_group

        mask = torch.zeros(len(FEATURE_NAMES))
        fill = torch.zeros(len(FEATURE_NAMES))
        if ablate_group is not None:
            prefix = GROUP_PREFIXES[ablate_group]
            for idx, name in enumerate(FEATURE_NAMES):
                if name.startswith(prefix):
                    mask[idx] = 1.0
                    fill[idx] = NEUTRAL_VALUES[name]
        self.register_buffer("_mask", mask)
        self.register_buffer("_fill", fill)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablate_group is not None:
            x = x * (1.0 - self._mask) + self._fill * self._mask
        return self.base_model(x)


def _build_config(label: str, run_label_base: str, imitation_scoring_rule: str) -> Config:
    """Reconstructs the Config that run_sil_training_bi200_ss100_class1_legacy.sh /
    run_sil_training_bi200_ss100_class1_exp_prefix.sh used (same bi200/ss100/mc2/
    5-epoch/no-pruner setup, only --imitation_scoring_rule differs - see main.py's
    argparse defaults). Several argparse defaults differ from Config's own
    dataclass field defaults (RTV_TIMEOUT, ILP_PENALTY, LARGEST_TSP,
    SHARE_COST_FACTOR, DWELL_ALIGHT) - those are set explicitly below to match
    what the checkpoint was actually trained under, rather than relying on
    Config()'s dataclass defaults."""
    namespace = argparse.Namespace(
        config_file="",
        override=[],
        server_url="http://127.0.0.1:5001/",
        max_thread_cnt=16,
        rtv_timeout=1800,
        ilp_timeout=120,
        ilp_penalty=100_000,
        travel_time_margin=5,
        debug="False",
        rebalancing="False",
        keep_active="False",
        return_depot="True",
        intermediate_location="False",
        dwell_pickup=180,
        dwell_alight=90,
        share_cost_factor=5,
        walk_distance_cutoff=0,
        mode="coaml",
        max_cardinality=2,
        largest_tsp=16,
        step_size=100,
        batch_interval=200,
        seed=42,
        epochs=5,
        learning_rate=0.0001,
        hidden_dim=64,
        num_samples=20,
        sigma=0.2,
        y_star_type="best_ordered_match",
        coaml_model_weights="",
        coaml_solve_mode="train",
        input_file="solutions/li_lim/manifests/lc101.json",
        input_dir="solutions/li_lim/manifests/",
        val_input_file="",
        imitation_solution_file="solutions/li_lim/manifests/lc101.json",
        output_dir=f"{run_label_base}/{label}",
        extra_validation_files="",
        extra_training_files="",
        override_training_files="",
        override_validation_files="",
        use_request_graph_pruner="False",
        request_graph_model_path=(
            "outputs/models_v2_gnnv2/rgnn_mixed_c2_pw10_v2/"
            "rgnn_mixed_c2_pw10_v2_best_val_f3.pt"
        ),
        request_graph_threshold=0.5,
        use_request_pruner="False",
        imitation_scoring_rule=imitation_scoring_rule,
        request_pruner_model_path=(
            "outputs/request_pruner_mlp/request_pruner_mlp_h32_l1_d0p0_pw1p0/"
            "request_pruner_mlp_h32_l1_d0p0_pw1p0_best_val_f3.pt"
        ),
        request_pruner_threshold=0.3,
    )
    return Config.from_args(namespace)


class _AblationTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _AblationTimeout()


def _run_one_file(config: Config, model: nn.Module, v_path: Path):
    """Runs mode="eval" on one validation file, capped at PER_FILE_TIMEOUT_SEC
    wall-clock. Returns the StatsParser stats object, or None if skipped."""
    out_dir = config.OUTPUT_DIR / "val" / v_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    file_cfg = replace(config, OUTPUT_DIR=out_dir)

    v_cleared = _load_and_clear_payload(v_path)
    pipeline = COAMLPipeline(file_cfg, v_cleared, model=model, imitation_solution_path=v_path)

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_FILE_TIMEOUT_SEC)
    start = time.time()
    try:
        driver_runs = pipeline.solve_pdptw(v_cleared, mode="eval")
    except _AblationTimeout:
        print(f"  [SKIP] {v_path.stem}: exceeded {PER_FILE_TIMEOUT_SEC}s wall-clock cap")
        return None
    finally:
        signal.alarm(0)
    elapsed = time.time() - start

    stats = _save_validation_results(file_cfg, v_cleared, driver_runs)
    print(f"  {v_path.stem}: serviced {stats.serviced}/{stats.total_requests} ({elapsed:.1f}s)")
    return stats


def run_ablation(
    checkpoint_path: Path = CHECKPOINT,
    run_label_base: str = RUN_LABEL_BASE,
    imitation_scoring_rule: str = IMITATION_SCORING_RULE,
) -> None:
    _, val_files = _split_train_val_files(DATA_DIR, set(TRAINING_FILES), set(VALIDATION_FILES))
    if not val_files:
        raise ValueError(f"No validation files found in {DATA_DIR}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    base_model = ScoringMLP(
        feature_dim=checkpoint.get("feature_dim", FeatureBuilder.FEATURE_SIZE),
        hidden_dim=checkpoint.get("hidden_dim", 64),
    )
    base_model.load_state_dict(checkpoint["model_state_dict"])
    base_model.eval()

    summary: dict[str, dict] = {}
    for group in (None, "state", "vehicle", "trip", "tripcost", "grid"):
        label = group or "baseline"
        print(f"=== Ablation: {label} ===")
        model = AblatedScoringModel(base_model, group)
        model.eval()

        config = _build_config(label, run_label_base, imitation_scoring_rule)

        pooled_serviced = 0
        pooled_total = 0
        skipped: list[str] = []
        for v_path in val_files:
            stats = _run_one_file(config, model, v_path)
            if stats is None:
                skipped.append(v_path.stem)
                continue
            pooled_serviced += stats.serviced
            pooled_total += stats.total_requests

        service_rate = pooled_serviced / max(pooled_total, 1)
        summary[label] = {
            "serviced": pooled_serviced,
            "total_requests": pooled_total,
            "service_rate": service_rate,
            "skipped_files": skipped,
            "output_dir": str(config.OUTPUT_DIR),
        }
        print(f"  -> pooled service rate: {service_rate:.4f} ({pooled_serviced}/{pooled_total})")

    baseline_rate = summary["baseline"]["service_rate"]
    print("\n=== Summary (delta vs baseline) ===")
    for label, entry in summary.items():
        delta = entry["service_rate"] - baseline_rate
        print(f"  {label:10s} service_rate={entry['service_rate']:.4f}  delta={delta:+.4f}")

    summary_path = REPO_ROOT / "outputs" / run_label_base / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, summary_path)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    run_ablation()
