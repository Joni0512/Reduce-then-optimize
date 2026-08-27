"""
2026-08-18: multi-episode training loop for the SRL critic (Phase 2),
"Variante A" from chat - instance-split, fixed actor.

One CriticGNN + one Adam optimizer are built ONCE, outside the instance loop,
and reused across all episodes (unlike test_critic_single_instance.py, which
builds a fresh critic per run and only ever does one gradient step). The
actor is held fixed throughout: every instance is solved with mode="eval"
using the SAME loaded actor checkpoint, so only the critic ever learns here -
see chat for why actor training (perturb-and-softmax against Q) is
deliberately deferred until the critic itself is shown to work.

Train/val split is over Li&Lim class-1 instances (lc101-lc109, 9 total):
lc101-lc106 train (critic takes a gradient step per episode), lc107-lc109
validation (forward pass + loss only, train_critic=False - never influences
the critic's weights, see coaml_pipeline.py's solve_pdptw docstring).
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from rtv_solver.coaml_pipeline import COAMLPipeline
from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.pipeline.critic_gnn import CriticGNN
from rtv_solver.structure.config import Config
from rtv_solver.util.helper import set_seed
from rtv_solver.util.logger import setup_loggers

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "solutions" / "li_lim" / "manifests"

TRAIN_INSTANCES = ["lc101", "lc102", "lc103", "lc104", "lc105", "lc106"]
VAL_INSTANCES = ["lc107", "lc108", "lc109"]


def _run_episode(
    instance: str,
    critic: torch.nn.Module,
    critic_optimizer: torch.optim.Optimizer,
    actor_checkpoint: str,
    batch_interval: int,
    step_size: int,
    seed: int,
    output_dir: Path,
    train_critic: bool,
    reward_mode: str,
) -> dict:
    input_path = MANIFEST_DIR / f"{instance}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"No manifest for instance '{instance}': {input_path}")

    instance_output_dir = output_dir / instance
    instance_output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        OUTPUT_DIR=instance_output_dir,
        MODE="coaml",
        BATCH_INTERVAL=batch_interval,
        STEP_SIZE=step_size,
        SEED=seed,
    )
    setup_loggers(config.OUTPUT_DIR)
    set_seed(config.SEED, config.DEBUG)

    payload = PayloadParser.load_input_data(input_path)
    cleared_payload = PayloadParser.clear_vehicle_manifests(payload)

    # 2026-08-18: critic/critic_optimizer are passed in from the outer loop
    # (built once, reused across episodes) - only the pipeline itself and the
    # actor's ScoringMLP are rebuilt fresh per instance, same pattern main.py
    # uses for the actor's own weight loading (coaml_pipeline.py ~357-361).
    pipeline = COAMLPipeline(
        config,
        cleared_payload,
        imitation_solution_path=input_path,
        critic=critic,
        critic_optimizer=critic_optimizer,
    )
    pipeline.load_model_weights(actor_checkpoint)

    pipeline.solve_pdptw(cleared_payload, mode="eval", train_critic=train_critic, reward_mode=reward_mode)

    returns = pipeline.last_episode_returns
    predictions = pipeline.last_episode_predictions
    loss = pipeline.last_episode_loss if hasattr(pipeline, "last_episode_loss") else None
    return {"instance": instance, "returns": returns, "predictions": predictions, "loss": loss}


def train_critic(
    actor_checkpoint: str,
    batch_interval: int = 200,
    step_size: int = 100,
    seed: int = 42,
    epochs: int = 1,
    actor_seed_label: str = "unknown",
    # 2026-08-18: default flipped to "local" (r_t) per user preference after
    # comparing both - "cumulative" (G_t) was judged too strict/pessimistic,
    # see chat. Still selectable via --reward_mode cumulative if needed.
    reward_mode: str = "local",
    train_instances: list[str] | None = None,
    val_instances: list[str] | None = None,
    run_label: str = "",
) -> None:
    # 2026-08-18: train_instances/val_instances default to the class-1 split
    # (module-level TRAIN_INSTANCES/VAL_INSTANCES) but are overridable - e.g.
    # for the class-2 6/3 subset (lc201,lc202,lc203,lc205,lr201,lr202 /
    # lc207,lc208,lr210) used to test the critic against a
    # harder/lower-service-rate actor, see chat.
    train_instances = train_instances if train_instances is not None else TRAIN_INSTANCES
    val_instances = val_instances if val_instances is not None else VAL_INSTANCES

    dir_suffix = f"_{run_label}" if run_label else ""
    output_dir = REPO_ROOT / "outputs" / "critic_training" / f"bi{batch_interval}_ss{step_size}_reward-{reward_mode}{dir_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    critic = CriticGNN()
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # 2026-08-18: one row per episode (instance x epoch), written incrementally
    # so a crash mid-run doesn't lose earlier epochs - see chat, user wants
    # this saved as CSV + a labeled plot after training.
    csv_path = output_dir / "critic_losses.csv"
    rows: list[dict] = []

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch} ===")

        print("--- Training instances ---")
        for instance in train_instances:
            result = _run_episode(
                instance, critic, critic_optimizer, actor_checkpoint,
                batch_interval, step_size, seed, output_dir, train_critic=True, reward_mode=reward_mode,
            )
            print(f"{instance}: loss={result['loss']:.4f}  G_t={[round(g, 2) for g in result['returns']]}")
            rows.append({"epoch": epoch, "split": "train", "instance": instance, "loss": result["loss"]})

        print("--- Validation instances ---")
        val_losses = []
        for instance in val_instances:
            result = _run_episode(
                instance, critic, critic_optimizer, actor_checkpoint,
                batch_interval, step_size, seed, output_dir, train_critic=False, reward_mode=reward_mode,
            )
            val_losses.append(result["loss"])
            print(f"{instance}: loss={result['loss']:.4f}  G_t={[round(g, 2) for g in result['returns']]}")
            rows.append({"epoch": epoch, "split": "val", "instance": instance, "loss": result["loss"]})

        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"epoch {epoch}: mean val loss = {mean_val_loss:.4f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split", "instance", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved losses to {csv_path}")

    _plot_losses(rows, output_dir, actor_seed_label, batch_interval, step_size, reward_mode, run_label)


def _plot_losses(
    rows: list[dict],
    output_dir: Path,
    actor_seed_label: str,
    batch_interval: int,
    step_size: int,
    reward_mode: str = "cumulative",
    run_label: str = "",
) -> None:
    epochs = sorted({r["epoch"] for r in rows})
    n_train = len({r["instance"] for r in rows if r["split"] == "train"})
    n_val = len({r["instance"] for r in rows if r["split"] == "val"})
    train_mean = [
        sum(r["loss"] for r in rows if r["epoch"] == e and r["split"] == "train")
        / len([r for r in rows if r["epoch"] == e and r["split"] == "train"])
        for e in epochs
    ]
    val_mean = [
        sum(r["loss"] for r in rows if r["epoch"] == e and r["split"] == "val")
        / len([r for r in rows if r["epoch"] == e and r["split"] == "val"])
        for e in epochs
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_mean, marker="o", label=f"Train loss (mean over {n_train} instances)")
    ax.plot(epochs, val_mean, marker="o", label=f"Validation loss (mean over {n_val} instances)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Huber loss (Q_pred vs {'G_t' if reward_mode == 'cumulative' else 'r_t'})")
    title_suffix = f", {run_label}" if run_label else ""
    ax.set_title(
        f"SRL Critic Training Loss (reward: {reward_mode}{title_suffix})\n"
        f"Fixed actor: MLP checkpoint, seed {actor_seed_label} | "
        f"batch_interval={batch_interval}, step_size={step_size}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "critic_losses.png"
    fig.savefig(plot_path, dpi=150)

    # 2026-08-18: also saved as PDF + PNG into figures_export/, matching the
    # existing convention there (vector PDF for the thesis, PNG since that
    # was being manually copied there by hand every time before - see chat).
    export_dir = REPO_ROOT / "figures_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    label_suffix = f"_{run_label}" if run_label else ""
    export_stem = f"critic_losses_bi{batch_interval}_ss{step_size}_seed{actor_seed_label}_reward-{reward_mode}{label_suffix}"
    pdf_path = export_dir / f"{export_stem}.pdf"
    png_path = export_dir / f"{export_stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)

    plt.close(fig)
    print(f"Saved plot to {plot_path}, {pdf_path} and {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-episode SRL critic training (Variante A, instance split).")
    parser.add_argument("--actor_checkpoint", type=str, required=True, help="Fixed actor checkpoint used for every episode (mode=eval).")
    parser.add_argument("--batch_interval", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the solver run itself (RTV tie-breaks etc.), not the actor's training seed.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--actor_seed_label", type=str, default="unknown", help="Label only (e.g. '5') identifying which actor training seed --actor_checkpoint came from, used in the saved plot's title.")
    parser.add_argument(
        "--reward_mode", type=str, default="local", choices=["cumulative", "local"],
        help="'cumulative' = G_t, the Monte Carlo return (default). 'local' = r_t, -1 only in the "
        "window a request's deadline permanently passes unserved - see mc_return_builder.py.",
    )
    parser.add_argument("--train_instances", type=str, default="", help="Comma-separated instance stems, e.g. 'lc201,lc202,...'. Empty = class-1 default (lc101-lc106).")
    parser.add_argument("--val_instances", type=str, default="", help="Comma-separated instance stems. Empty = class-1 default (lc107-lc109).")
    parser.add_argument("--run_label", type=str, default="", help="Free-text label appended to output dir/plot title/filenames, e.g. 'class2'.")
    args = parser.parse_args()

    train_critic(
        actor_checkpoint=args.actor_checkpoint,
        batch_interval=args.batch_interval,
        step_size=args.step_size,
        seed=args.seed,
        epochs=args.epochs,
        actor_seed_label=args.actor_seed_label,
        reward_mode=args.reward_mode,
        train_instances=[s.strip() for s in args.train_instances.split(",") if s.strip()] or None,
        val_instances=[s.strip() for s in args.val_instances.split(",") if s.strip()] or None,
        run_label=args.run_label,
    )
