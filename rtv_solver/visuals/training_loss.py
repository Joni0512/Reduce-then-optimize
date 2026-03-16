from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_loss_per_file(
    losses_per_file: dict[str, list[Optional[float]]],
    output_dir: Path | str,
) -> None:
    """
    Plot loss per training file as separate lines. Legend with file stems below plot.
    """
    plt.figure()
    for stem, losses in losses_per_file.items():
        valid = [l for l in losses if l is not None]
        if not valid:
            continue
        plt.plot(range(1, len(valid) + 1), valid, label=stem, alpha=0.8)
    plt.axhline(0, linestyle="--", linewidth=1, color="gray", alpha=0.5)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Training Loss per File")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(8, len(losses_per_file)),
        fontsize=8,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_loss_per_file.png", bbox_inches="tight")
    plt.close()


def plot_loss(loss_history, iterations, output_dir):
    """Plot training loss over iterations with background iteration markers."""
    num_points = len(loss_history)
    x_values = range(1, num_points + 1)

    plt.figure()

    if iterations > 0 and num_points > 0:
        spacing = num_points / iterations
        for idx in range(1, iterations + 1):
            plt.axvline(
                x=idx * spacing,
                color="red",
                linestyle="-",
                linewidth=1,
                alpha=0.2,
                zorder=0,
            )

    plt.plot(x_values, loss_history, marker='o', label="Loss")
    plt.axhline(0, linestyle="--", linewidth=1, label="Zero")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_filename = "training_loss.png"
    plt.savefig(Path(output_dir) / plot_filename)
    plt.close()