import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_loss(loss_history, iterations):
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
    plt.show()