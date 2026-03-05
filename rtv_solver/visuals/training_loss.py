import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_loss(loss_history):
    """Plot training loss over iterations."""
    plt.figure()
    plt.plot(loss_history, marker='o', label="Loss")
    plt.axhline(0, linestyle="--", linewidth=1, label="Zero")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()