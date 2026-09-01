"""
2026-08-21: first real SRL actor-critic training test (Algorithm 1, all 6
steps - see chat/figures_export/srl_actor_critic_integration_steps.tex).

Single instance, repeated for --episodes episodes. BOTH the actor (fresh,
untrained ScoringMLP - mode="srl", per-iteration FenchelYoungLoss against the
critic-informed softmax target action) and the critic (fresh CriticGNN,
reward_mode="local", per-episode Huber loss against real outcomes) start
from scratch and learn simultaneously - classic actor-critic moving-target
setup, first check whether any learning signal emerges at all before scaling
to a multi-instance run.

sigma=1.0 (NOT config.py's default 0.2) and num_samples=10 (NOT the default
20) - chosen to match what tau=0.02 was actually calibrated against
(diagnose_q_spread.py/test_softmax_tau.py, see chat), not COAMLPipeline's
general-purpose defaults.
"""
# 2026-08-31: commit 69b521f (29.08., after ACTOR_CHECKPOINT below was
# trained) flipped feat_builder.py's ENABLE_PICKUP_SLACK_FEATURE default to
# True, raising v1's FEATURE_SIZE from 84 to 85 - breaks loading
# ACTOR_CHECKPOINT (trained at 84) for every script that imports this module,
# not just the ones that remembered to patch it themselves (bit
# pretrain_and_save_shared_critic.py, sweep_srl_replaybuffer_trial.py,
# replicate_v1_top10_seeds.py, and test_targetcritic_plus_replaybuffer_lrc207.py
# separately - see chat). Patched centrally here instead, at import time,
# so any script importing train_srl_single_instance gets it automatically.
# Temporary, local-only override until the user decides whether to retrain
# the SRL actor checkpoint with the new feature enabled.
from rtv_solver.pipeline import feat_builder as _feat_builder_module
_feat_builder_module.FeatureBuilder.ENABLE_PICKUP_SLACK_FEATURE = False
_feat_builder_module.FeatureBuilder.FEATURE_SIZE = (
    _feat_builder_module.FeatureBuilder._BASE_FEATURE_SIZE
    + (_feat_builder_module.FeatureBuilder._TRIP_COMPOSITION_FEATURE_SIZE if _feat_builder_module.FeatureBuilder.ENABLE_TRIP_COMPOSITION_FEATURES else 0)
)

import argparse
import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.stats_parser import StatsParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.pipeline.replay_buffer import ReplayBuffer
from rtv_solver.pipeline.train_critic import TRAIN_INSTANCES as DEFAULT_CRITIC_PRETRAIN_INSTANCES
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"


def train(
    instance: str = "lc101",
    episodes: int = 20,
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    num_samples: int = 10,
    sigma: float = 1.0,
    tau: float = 0.02,
    actor_checkpoint: str | None = None,
    freeze_critic: bool = False,
    critic_pretrain_epochs: int = 10,
    shared_critic: torch.nn.Module | None = None,
    label_suffix: str = "",
    target_critic_update_interval: int | None = None,
    target_critic_polyak_tau: float | None = None,
    use_replay_buffer: bool = False,
    replay_capacity: int = 40,
    replay_batch_size: int = 12,
    replay_update_group_size: int = 3,
) -> None:
    # 2026-08-23: actor_checkpoint added after the first "untrained actor"
    # test degenerated to service_rate=0.0 for all 20 episodes (see chat) -
    # an untrained ScoringMLP scores real trip rows much more negatively
    # than the near-all-zero reject row (more nonzero features -> larger
    # random dot-product magnitude at init), so CO_ScoreMaximization always
    # picks "reject everything" and never receives a gradient signal to
    # escape that. Starting from an already SIL-trained checkpoint avoids
    # this cold-start failure mode - SRL then fine-tunes a working policy
    # instead of trying to discover one from scratch.
    # 2026-08-25: label_suffix added so runs against a different actor
    # checkpoint (e.g. mixed-class vs. class-1-only) don't silently overwrite
    # each other's output directory under the same instance name - see chat.
    label = (f"{instance}_pretrained" + ("_frozencritic" if freeze_critic else "") if actor_checkpoint else instance) + label_suffix
    input_path = MANIFEST_DIR / f"{instance}.json"
    output_dir = REPO_ROOT / "outputs" / "train_srl_single_instance" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2026-08-21: NUM_SAMPLES/SIGMA/SRL_TAU set explicitly here (not left at
    # config.py's defaults) - see module docstring for why.
    config = Config(
        OUTPUT_DIR=output_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed,
        NUM_SAMPLES=num_samples, SIGMA=sigma, SRL_TAU=tau,
    )
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

    # 2026-08-21: fresh, untrained actor AND critic - the critic is built
    # ONCE, reused (and updated) across all episodes, same "build once
    # outside the loop" pattern train_critic.py already uses. The actor
    # (self.model) is fresh per COAMLPipeline() call too, but its weights are
    # carried across episodes manually below via `model=pipeline.model`.
    if shared_critic is not None:
        # 2026-08-25: reuse an already-pretrained critic passed in from
        # outside, instead of building+pretraining a fresh one per instance -
        # needed for the 12-instance frozen-critic sweep (see chat):
        # pretraining 12x from scratch would be wasteful, one shared critic
        # is pretrained once by the caller and reused here.
        #
        # 2026-08-28: deepcopy added - this function used to take
        # `shared_critic` BY REFERENCE, which was harmless for freeze_critic=
        # True (the critic never gets a gradient step, so it can't drift
        # either way) but would silently corrupt the "pretrained but still
        # co-adapting" variant (freeze_critic=False + shared_critic): without
        # the copy, each test instance's 20 SRL episodes would keep mutating
        # the SAME object, so instance 2 would start from instance 1's
        # already-adapted weights instead of the original pretrained
        # snapshot - instances would silently stop being independent/
        # comparable. Deepcopying makes every call start from the exact same
        # pretrained state regardless of what earlier calls did to it.
        critic = copy.deepcopy(shared_critic)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    else:
        critic = CriticGNN()
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        if freeze_critic:
            # 2026-08-24: diagnostic mode (see chat) - investigating whether the
            # service-rate collapse on lr111/lr112/lrc107 comes from the
            # actor/critic co-adapting simultaneously ("moving target" problem).
            # Pretrain the critic once, upfront, on a fixed actor (the same
            # actor_checkpoint the SRL episodes below start from) - then during
            # the SRL loop, train_critic=False keeps these weights fixed, so only
            # the actor keeps changing.
            if not actor_checkpoint:
                raise ValueError("freeze_critic requires actor_checkpoint (need a fixed policy to pretrain the frozen critic against).")
            print(f"Pretraining critic ({critic_pretrain_epochs} epochs, then freezing) on {DEFAULT_CRITIC_PRETRAIN_INSTANCES}...")
            for epoch in range(critic_pretrain_epochs):
                for pretrain_instance in DEFAULT_CRITIC_PRETRAIN_INSTANCES:
                    pretrain_input_path = MANIFEST_DIR / f"{pretrain_instance}.json"
                    pretrain_output_dir = output_dir / "critic_pretrain" / pretrain_instance
                    pretrain_output_dir.mkdir(parents=True, exist_ok=True)
                    pretrain_config = Config(
                        OUTPUT_DIR=pretrain_output_dir, MODE="coaml", BATCH_INTERVAL=batch_interval, STEP_SIZE=step_size, SEED=seed,
                    )
                    setup_loggers(pretrain_config.OUTPUT_DIR)
                    set_seed(pretrain_config.SEED, pretrain_config.DEBUG)
                    pretrain_payload = PayloadParser.load_input_data(pretrain_input_path)
                    pretrain_cleared_payload = PayloadParser.clear_vehicle_manifests(pretrain_payload)
                    pretrain_pipeline = COAMLPipeline(
                        pretrain_config, pretrain_cleared_payload, imitation_solution_path=pretrain_input_path,
                        critic=critic, critic_optimizer=critic_optimizer,
                    )
                    pretrain_pipeline.load_model_weights(actor_checkpoint)
                    pretrain_pipeline.solve_pdptw(pretrain_cleared_payload, mode="eval", train_critic=True, reward_mode="local")
                print(f"critic pretrain epoch {epoch} done")

    rows = []
    actor_optimizer = None
    model = None
    # 2026-08-25: best-checkpoint tracking, per the reference SRL
    # implementation (Julia/Flux DVSP code, see chat) - they keep the
    # best-validation-service-rate snapshot across training and return THAT,
    # not just whatever the final episode happened to land on. Directly
    # guards against the collapse-then-recover-then-collapse-again pattern
    # we saw (e.g. lc108's 40-episode run peaked mid-training, then drifted
    # back down by episode 39).
    best_service_rate = -1.0
    best_episode = -1
    best_state_dict = None

    # 2026-08-26: SRL Option B (see chat) - target_critic is a periodically
    # hard-copied, otherwise-frozen snapshot of the live critic, used ONLY to
    # score candidates for the actor's softmax target action
    # (coaml_pipeline.py's self.target_critic). The live critic keeps
    # training exactly as before (Huber loss vs. G_t/r_t, unaffected).
    # Only meaningful when the critic is NOT frozen - if freeze_critic=True,
    # the live critic never moves anyway, so target_critic would always be
    # identical to it regardless of the copy interval.
    # 2026-08-31: Polyak (soft) update - alternative to the hard periodic
    # copy above, see chat. Instead of staying frozen for
    # target_critic_update_interval episodes then jumping all at once, the
    # target_critic is nudged a little toward the live critic EVERY episode:
    #   target_param <- tau * live_param + (1 - tau) * target_param
    # tau=0.5 (user's first, deliberately simple test) means each episode's
    # target_critic is an equal-weight blend of "last episode's target_critic"
    # and "this episode's live critic" - smoother than the hard-copy jump,
    # but still much more responsive than typical DDPG/TD3 tau (~0.005), by
    # design for this first, easy test.
    if target_critic_update_interval is not None and target_critic_polyak_tau is not None:
        raise ValueError("target_critic_update_interval and target_critic_polyak_tau are two different target_critic update mechanisms (hard periodic copy vs. soft blend every episode) - pick one, not both.")

    target_critic = None
    if target_critic_update_interval is not None:
        if freeze_critic:
            raise ValueError("target_critic_update_interval has no effect when freeze_critic=True - the live critic never changes, so a periodic copy of it is always identical.")
        target_critic = copy.deepcopy(critic)
    elif target_critic_polyak_tau is not None:
        if freeze_critic:
            raise ValueError("target_critic_polyak_tau has no effect when freeze_critic=True - the live critic never changes, so blending toward it changes nothing.")
        target_critic = copy.deepcopy(critic)

    # 2026-08-28: replay buffer (see chat) - built ONCE outside the episode
    # loop, same "persists across episodes" pattern as critic/critic_optimizer
    # above, so it actually accumulates cross-episode history within this
    # instance's run instead of being rebuilt empty every episode.
    replay_buffer = None
    if use_replay_buffer:
        if freeze_critic:
            raise ValueError("use_replay_buffer has no effect when freeze_critic=True - the critic never trains, so there is nothing for the buffer to feed.")
        replay_buffer = ReplayBuffer(capacity=replay_capacity)

    for episode in range(episodes):
        if target_critic is not None and target_critic_update_interval is not None and episode % target_critic_update_interval == 0:
            # 2026-08-26: hard copy, same pattern as the reference SRL
            # implementation's `target_critic = deepcopy(π.critic_model)` -
            # done at the START of the interval (episode 0, then every
            # target_critic_update_interval-th episode), so the actor trains
            # against a fixed target action for the whole interval, only
            # then getting a fresh (now-improved) target_critic snapshot.
            target_critic.load_state_dict(critic.state_dict())
            print(f"episode {episode}: target_critic copied from live critic")
        elif target_critic is not None and target_critic_polyak_tau is not None:
            # 2026-08-31: soft blend every episode, see the setup comment
            # above for the formula/reasoning.
            with torch.no_grad():
                for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(target_critic_polyak_tau * live_param.data + (1 - target_critic_polyak_tau) * target_param.data)
            print(f"episode {episode}: target_critic polyak-blended (tau={target_critic_polyak_tau}) toward live critic")

        pipeline = COAMLPipeline(
            config, cleared_payload, imitation_solution_path=input_path,
            model=model, critic=critic, critic_optimizer=critic_optimizer,
            target_critic=target_critic,
            replay_buffer=replay_buffer, replay_batch_size=replay_batch_size,
            replay_update_group_size=replay_update_group_size,
        )
        if actor_checkpoint and episode == 0:
            # 2026-08-23: only load on episode 0 - after that, `model` (the
            # already-updated weights, carried below) is passed back in on
            # every subsequent episode, so re-loading later would overwrite
            # the SRL fine-tuning progress with the original checkpoint again.
            pipeline.load_model_weights(actor_checkpoint)
        if actor_optimizer is None:
            # 2026-08-21: actor_optimizer built once, after the pipeline's
            # first fresh model exists, then reused (holding Adam's momentum
            # state across episodes) - model itself is passed back in on
            # every later episode (below) so weights persist too.
            actor_optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=actor_lr)

        final_driver_runs = pipeline.solve_pdptw(cleared_payload, mode="srl", optimizer=actor_optimizer, train_critic=not freeze_critic, reward_mode="local")
        model = pipeline.model  # carry actor weights into the next episode

        # 2026-08-21: service rate independent of the critic's reward_mode
        # choice - same StatsParser call solve_pdptw() already does
        # internally for the critic's G_t/r_t, reused here just to read off
        # serviced_requests directly (a real, human-readable metric, unlike
        # the loss values which only measure "how close to the target").
        full_payload_object = PayloadParser.get_payload_object(
            cleared_payload, dwell_pickup_default=config.DWELL_PICKUP, dwell_alight_default=config.DWELL_ALIGHT, online=False,
        )
        all_requests = RequestHandler(full_payload_object.requests, config=config).get_all_requests()
        stats_payload = {
            PayloadKeys.DEPOT: cleared_payload[PayloadKeys.DEPOT],
            PayloadKeys.REQUESTS: cleared_payload[PayloadKeys.REQUESTS],
            PayloadKeys.DRIVERS: final_driver_runs,
            PayloadKeys.TIME_MATRIX: cleared_payload.get(PayloadKeys.TIME_MATRIX, None),
        }
        _, episode_stats, _ = StatsParser(config, payload=stats_payload).evaluate(stats_payload)
        num_requests = len(all_requests)
        num_serviced = len(episode_stats.serviced_requests)
        service_rate = num_serviced / num_requests if num_requests > 0 else 0.0

        if service_rate > best_service_rate:
            best_service_rate = service_rate
            best_episode = episode
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

        # 2026-08-23 bugfix: pipeline.loss_history can contain None entries
        # (coaml_pipeline.py appends None whenever an iteration's FY loss was
        # skipped, e.g. an empty y_star) - filter those out before averaging,
        # crashed on lrc107 (0 feasible trip costs iteration) without this.
        valid_losses = [l for l in pipeline.loss_history if l is not None]
        mean_fy_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None
        critic_loss_val = pipeline.last_episode_loss if hasattr(pipeline, "last_episode_loss") else None
        returns = pipeline.last_episode_returns if hasattr(pipeline, "last_episode_returns") else []

        # 2026-08-24: mean target-action accept/reject mass across this
        # episode's iterations (see coaml_pipeline.py's srl_target_action_log,
        # investigating the service-rate collapse - see chat).
        target_log = pipeline.srl_target_action_log if hasattr(pipeline, "srl_target_action_log") else []
        if target_log:
            mean_accept_mass = sum(e["accept_mass"] for e in target_log) / len(target_log)
            mean_reject_mass = sum(e["reject_mass"] for e in target_log) / len(target_log)
        else:
            mean_accept_mass = None
            mean_reject_mass = None

        print(f"episode {episode}: service_rate={service_rate:.3f} ({num_serviced}/{num_requests})  mean_fy_loss={mean_fy_loss}  critic_loss={critic_loss_val}  buffered_iters={len(returns)}  accept_mass={mean_accept_mass}  reject_mass={mean_reject_mass}")
        rows.append({
            "episode": episode,
            "service_rate": service_rate,
            "mean_fy_loss": mean_fy_loss,
            "mean_accept_mass": mean_accept_mass,
            "mean_reject_mass": mean_reject_mass,
            "critic_loss": critic_loss_val,
        })

    print(f"Best episode: {best_episode} with service_rate={best_service_rate:.3f}")
    best_checkpoint_path = output_dir / "best_actor_checkpoint.pt"
    torch.save({"model_state_dict": best_state_dict, "episode": best_episode, "service_rate": best_service_rate}, best_checkpoint_path)
    print(f"Saved best checkpoint to {best_checkpoint_path}")

    csv_path = output_dir / "srl_training_curves.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "service_rate", "mean_fy_loss", "critic_loss", "mean_accept_mass", "mean_reject_mass"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    episodes_x = [r["episode"] for r in rows]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    axes[0].plot(episodes_x, [r["service_rate"] for r in rows], marker="o", color="tab:green")
    axes[0].set_title("Service rate per episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Service rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes_x, [r["mean_fy_loss"] for r in rows], marker="o", color="tab:blue")
    axes[1].set_title("Actor: mean Fenchel-Young loss per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("FY loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes_x, [r["critic_loss"] for r in rows], marker="o", color="tab:orange")
    axes[2].set_title("Critic: Huber loss per episode (reward_mode=local)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Huber loss")
    axes[2].grid(True, alpha=0.3)

    # 2026-08-24: target-action accept/reject mass, investigating whether the
    # softmax target action itself drifts toward "reject everything" over
    # episodes (see chat) - same diagnostic idea as the cold-start collapse,
    # applied here to the SRL target action instead of the actor's own scores.
    axes[3].plot(episodes_x, [r["mean_accept_mass"] for r in rows], marker="o", color="tab:purple", label="accept mass")
    axes[3].plot(episodes_x, [r["mean_reject_mass"] for r in rows], marker="o", color="tab:red", label="reject mass")
    axes[3].set_title("Target action a_hat: mean accept vs. reject mass")
    axes[3].set_xlabel("Episode")
    axes[3].set_ylabel("Summed weight")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"SRL actor-critic joint training, instance={instance}, {episodes} episodes")
    fig.tight_layout()

    png_path = REPO_ROOT / "figures_export" / f"srl_training_curves_{label}.png"
    pdf_path = REPO_ROOT / "figures_export" / f"srl_training_curves_{label}.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="First real SRL actor-critic joint training test, single instance.")
    parser.add_argument("--instance", type=str, default="lc101")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--actor_checkpoint", type=str, default="", help="SIL-pretrained actor checkpoint to start from. Empty = fresh untrained actor (old default behavior).")
    parser.add_argument("--freeze_critic", action="store_true", help="Pretrain the critic once (on actor_checkpoint), then keep it fixed during the SRL episodes - isolates whether instability comes from actor/critic co-adaptation.")
    parser.add_argument("--critic_pretrain_epochs", type=int, default=10)
    parser.add_argument("--target_critic_update_interval", type=int, default=None, help="SRL Option B (see chat): episodes between target_critic hard-copies. None = no target_critic (actor uses the live, co-adapting critic directly, old default behavior). Only meaningful when NOT --freeze_critic.")
    parser.add_argument("--target_critic_polyak_tau", type=float, default=None, help="Alternative to --target_critic_update_interval (see chat): soft-blend target_critic toward the live critic every episode by this weight, instead of hard-copying periodically. Mutually exclusive with --target_critic_update_interval.")
    parser.add_argument("--use_replay_buffer", action="store_true", help="Train the critic from a cross-episode Monte Carlo replay buffer instead of one averaged step per episode (see chat). Only meaningful when NOT --freeze_critic.")
    parser.add_argument("--replay_capacity", type=int, default=40)
    parser.add_argument("--replay_batch_size", type=int, default=12)
    parser.add_argument("--replay_update_group_size", type=int, default=3)
    args = parser.parse_args()

    train(
        instance=args.instance,
        episodes=args.episodes,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        num_samples=args.num_samples,
        sigma=args.sigma,
        tau=args.tau,
        actor_checkpoint=args.actor_checkpoint or None,
        freeze_critic=args.freeze_critic,
        critic_pretrain_epochs=args.critic_pretrain_epochs,
        target_critic_update_interval=args.target_critic_update_interval,
        target_critic_polyak_tau=args.target_critic_polyak_tau,
        use_replay_buffer=args.use_replay_buffer,
        replay_capacity=args.replay_capacity,
        replay_batch_size=args.replay_batch_size,
        replay_update_group_size=args.replay_update_group_size,
    )
