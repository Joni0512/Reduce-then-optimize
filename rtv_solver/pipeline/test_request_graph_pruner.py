from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rtv_solver.handlers.payload_parser import PayloadParser
from rtv_solver.handlers.request_handler import RequestHandler
from rtv_solver.handlers.network_handler import NetworkHandler
from rtv_solver.schema.payload_keys import PayloadKeys
from rtv_solver.structure.config import Config

from rtv_solver.pipeline.request_graph_pruner import RequestGraphPruner


PAYLOAD_PATH = Path("solutions/li_lim/manifests/lc107.json")
MODEL_PATH = Path("outputs/request_graph_gnn.pt")

OUTPUT_DIR = Path("outputs")
OUTPUT_CSV = OUTPUT_DIR / "pruner_allowed_pairs_metrics.csv"
OUTPUT_PNG = OUTPUT_DIR / "pruner_allowed_pairs_metrics.png"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def save_table_as_png(df: pd.DataFrame, output_path: Path):
    display_df = df.copy()

    for col in ["edge_reduction"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

    display_df["threshold"] = display_df["threshold"].map(lambda x: f"{x:.1f}")

    fig_height = max(4, 0.4 * len(display_df))
    fig_width = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Config()
    payload = PayloadParser.load_input_data(PAYLOAD_PATH)

    NetworkHandler.init_from_payload(
        payload=payload,
        server_url=config.SERVER_URL,
    )

    request_handler = RequestHandler(
        payload[PayloadKeys.REQUESTS],
        config,
    )

    requests = request_handler.get_all_requests()

    rows = []

    print("Instance:", PAYLOAD_PATH.stem)
    print("Requests:", len(requests))

    for threshold in THRESHOLDS:
        pruner = RequestGraphPruner(
            model_path=MODEL_PATH,
            threshold=threshold,
        )

        allowed_pairs, stats = pruner.build_allowed_pairs(requests)

        row = {
            "instance": PAYLOAD_PATH.stem,
            "threshold": threshold,
            "requests": stats["num_requests"],
            "edges_total": stats["num_edges_total"],
            "edges_kept": stats["num_edges_kept"],
            "edges_removed": stats["num_edges_total"] - stats["num_edges_kept"],
            "edge_reduction": stats["edge_reduction"],
        }

        rows.append(row)

        print(
            f"Threshold {threshold:.1f} | "
            f"kept {row['edges_kept']}/{row['edges_total']} | "
            f"removed {row['edges_removed']} | "
            f"reduction {row['edge_reduction']:.3f}"
        )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(OUTPUT_CSV, index=False)
    save_table_as_png(results_df, OUTPUT_PNG)

    print("\nPRUNER METRICS")
    print(results_df.to_string(index=False))

    print(f"\nSaved CSV to: {OUTPUT_CSV}")
    print(f"Saved PNG to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()