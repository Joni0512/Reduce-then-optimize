"""
Same config as srl_local_twin_critic_trial.py (twin_critic_test_1), except
use_actor_warmstart=False - the actor is randomly initialized (via COAMLPipeline's own default
build_scoring_model) instead of loaded from a SIL checkpoint. See chat: the reference SRL paper
(Hoppe et al., "Structured Reinforcement Learning for Combinatorial Decision-Making") uses "the
same random model initialization for SIL, PPO, and SRL" - no SIL-pretrained warmstart - so this
tests whether our own twin-critic result (val=0.6696 with warmstart) holds up without it.

Usage: ./venv/bin/python3 -m rtv_solver.pipeline.srl_local_twin_critic_no_warmstart_trial
"""
import traceback

import torch

from rtv_solver.pipeline.srl_training_loop import run_srl_training_loop, REPO_ROOT, _pooled_service_rate
from rtv_solver.pipeline.srl_train_val_test_split import TEST_INSTANCES
from rtv_solver.pipeline.candidate_scoring_gnn import build_scoring_model
from rtv_solver.pipeline import select_feature_builder_class
from rtv_solver.structure.config import Config

# Still passed to run_srl_training_loop (required arg), but never actually loaded since
# use_actor_warmstart=False below skips every load_model_weights() call.
ACTOR_CHECKPOINT = str(list((REPO_ROOT / "outputs/outputs/sil_training_bi200_ss100_mixed_balanced_legacy_mlp_seed1").rglob("coaml_model_weights_best_val.pt"))[0])

EPOCHS = 20
VAL_EVERY_N_EPOCHS = 5
GAMMA = 0.99
TAU = 0.005
BATCH_INTERVAL = 200
STEP_SIZE = 100
SEED = 42

REWARD_MODE = "local_positive"
ACTOR_LR = 0.0031923396291543868
CRITIC_LR = 0.0010833586558285635


def main() -> None:
    run_id = "twin_critic_no_warmstart_1"
    output_dir = REPO_ROOT / "outputs" / "srl_training_sweep" / run_id
    print(f"=== srl_local_twin_critic_no_warmstart_trial {run_id}: reward_mode={REWARD_MODE} actor_lr={ACTOR_LR} critic_lr={CRITIC_LR} tau={TAU} gamma={GAMMA} use_twin_critic=True use_actor_warmstart=False ===")
    try:
        result = run_srl_training_loop(
            reward_mode=REWARD_MODE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
            output_dir=output_dir, actor_checkpoint=ACTOR_CHECKPOINT,
            gamma=GAMMA, tau=TAU, epochs=EPOCHS, val_every_n_epochs=VAL_EVERY_N_EPOCHS,
            batch_interval=BATCH_INTERVAL, step_size=STEP_SIZE, seed=SEED,
            use_twin_critic=True, use_actor_warmstart=False,
        )

        config_template = Config(OUTPUT_DIR=output_dir, BATCH_INTERVAL=BATCH_INTERVAL, STEP_SIZE=STEP_SIZE, SEED=SEED)
        active_feature_builder = select_feature_builder_class(config_template)
        model = build_scoring_model("mlp", feature_dim=active_feature_builder.FEATURE_SIZE, hidden_dim=64)
        checkpoint = torch.load(result.best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        test_service_rate = _pooled_service_rate(TEST_INSTANCES, model, config_template, output_dir, epoch_num=result.best_epoch, tag="test")
        print(f"=== srl_local_twin_critic_no_warmstart_trial {run_id} DONE: best_epoch={result.best_epoch} best_val_service_rate={result.best_val_service_rate:.4f} test_service_rate={test_service_rate:.4f} ===")
    except Exception as e:
        crash_path = output_dir / "crash_traceback.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(crash_path, "w") as f:
            traceback.print_exc(file=f)
        print(f"!!! srl_local_twin_critic_no_warmstart_trial {run_id} FAILED: {e!r} - full traceback written to {crash_path}")


if __name__ == "__main__":
    main()
