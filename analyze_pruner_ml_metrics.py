from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

# Passe den Pfad an, falls deine Prediction-Datei anders heißt.
INPUT = ROOT / "outputs" / "pruner_predictions_pw5.csv"

OUT = ROOT / "outputs" / "new_tests" / "card2_pw5" / "analysis" / "ml_metrics"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


def fbeta(precision, recall, beta):
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    y_pred = (scores >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + fp + tn + fn
    positives = tp + fn
    negatives = tn + fp
    kept = tp + fp
    pruned = tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_edges": total,
        "positive_edges": positives,
        "negative_edges": negatives,
        "kept_edges": kept,
        "pruned_edges": pruned,
        "edge_keep_rate": kept / total if total else 0.0,
        "edge_reduction_rate": pruned / total if total else 0.0,
        "positive_edges_pruned": fn,
        "positive_edges_pruned_rate": fn / positives if positives else 0.0,
        "negative_edges_pruned_rate": tn / negatives if negatives else 0.0,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "specificity": specificity,
        "f1": fbeta(precision, recall, 1),
        "f2": fbeta(precision, recall, 2),
        "f3": fbeta(precision, recall, 3),
    }


def summarize_group(df, group_cols):
    rows = []

    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        base = dict(zip(group_cols, keys))

        for th in THRESHOLDS:
            row = base.copy()
            row.update(compute_metrics(sub["label"], sub["score"], th))
            rows.append(row)

    return pd.DataFrame(rows)


def plot_metric(df, group_name, metric, title, ylabel):
    plt.figure()

    for key, sub in df.groupby(group_name):
        avg = sub.groupby("threshold")[metric].mean().reset_index()
        plt.plot(avg["threshold"], avg[metric], marker="o", label=str(key))

    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(OUT / f"{metric}_by_threshold_{group_name}.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_overall(df, metric, title, ylabel):
    avg = df.groupby("threshold")[metric].mean().reset_index()

    plt.figure()
    plt.plot(avg["threshold"], avg[metric], marker="o")
    plt.xlabel("Threshold")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(OUT / f"overall_{metric}_by_threshold.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}\n"
            "Bitte setze INPUT auf deine Prediction-CSV mit Spalten: score,label."
        )

    df = pd.read_csv(INPUT)

    required = {"score", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need at least score,label.")

    # Falls label boolean/string ist
    df["label"] = df["label"].astype(int)
    df["score"] = df["score"].astype(float)

    # Optional nur pw5 filtern
    if "pos_weight" in df.columns:
        df = df[df["pos_weight"].astype(str).isin(["5", "5.0", "pw5"])].copy()

    # Overall metrics
    overall_rows = []
    for th in THRESHOLDS:
        overall_rows.append(compute_metrics(df["label"], df["score"], th))
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "ml_metrics_overall_pw5.csv", index=False)

    # By class
    if "class" in df.columns:
        by_class = summarize_group(df, ["class"])
        by_class.to_csv(OUT / "ml_metrics_by_class_pw5.csv", index=False)

        for metric, ylabel in [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("accuracy", "Accuracy"),
            ("f2", "F2 score"),
            ("f3", "F3 score"),
            ("edge_reduction_rate", "Edge reduction rate"),
            ("positive_edges_pruned_rate", "Positive edges pruned rate"),
        ]:
            plot_metric(by_class, "class", metric, f"{ylabel} by threshold and class", ylabel)

    # By instance
    if "instance" in df.columns:
        by_instance = summarize_group(df, ["instance"])
        by_instance.to_csv(OUT / "ml_metrics_by_instance_pw5.csv", index=False)

    # By class + instance
    if "class" in df.columns and "instance" in df.columns:
        by_class_instance = summarize_group(df, ["class", "instance"])
        by_class_instance.to_csv(OUT / "ml_metrics_by_class_instance_pw5.csv", index=False)

    # Overall plots
    for metric, ylabel in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
        ("f1", "F1 score"),
        ("f2", "F2 score"),
        ("f3", "F3 score"),
        ("edge_reduction_rate", "Edge reduction rate"),
        ("positive_edges_pruned_rate", "Positive edges pruned rate"),
        ("negative_edges_pruned_rate", "Negative edges pruned rate"),
    ]:
        plot_overall(overall, metric, f"Overall {ylabel} by threshold", ylabel)

    print("Saved:")
    print(OUT / "ml_metrics_overall_pw5.csv")
    if "class" in df.columns:
        print(OUT / "ml_metrics_by_class_pw5.csv")
    if "instance" in df.columns:
        print(OUT / "ml_metrics_by_instance_pw5.csv")
    if "class" in df.columns and "instance" in df.columns:
        print(OUT / "ml_metrics_by_class_instance_pw5.csv")

    print("\nOverall metrics:")
    print(
        overall[
            [
                "threshold",
                "precision",
                "recall",
                "accuracy",
                "f2",
                "f3",
                "edge_reduction_rate",
                "positive_edges_pruned_rate",
            ]
        ].round(4)
    )


if __name__ == "__main__":
    main()
