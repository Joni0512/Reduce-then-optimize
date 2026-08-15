"""
2026-08-10: Service Rate + Runtime bar charts and LaTeX tables for the
class-1-only Pair Pruner (GNN v1_final) threshold sweep (bi200/ss100,
bi400/ss200), re-verifying earlier pair-pruner results against the
standard (non-mixed-class) SIL split. Baseline was run once and reused
across the three thresholds. Runtime is main.log first-to-last-line
wall-clock time (whole run: train + val).
"""
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

VAL_6 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
CONFIGS = ["bi200_ss100", "bi400_ss200"]
CONFIG_LABELS = {"bi200_ss100": "bi200/ss100", "bi400_ss200": "bi400/ss200"}
VARIANTS = ["baseline", "pair_t0.4", "pair_t0.5", "pair_t0.6"]
VARIANT_LABELS = {
    "baseline": "No Pruner",
    "pair_t0.4": "Pair Pruner\n(t=0.4)",
    "pair_t0.5": "Pair Pruner\n(t=0.5)",
    "pair_t0.6": "Pair Pruner\n(t=0.6)",
}
VARIANT_COLORS = {
    "baseline": "#9AA5A9",
    "pair_t0.4": "#3E8F6C",
    "pair_t0.5": "#276575",
    "pair_t0.6": "#1E2761",
}
OUT_DIR = Path("figures_export")


def find_run_dir(cfg, variant):
    dirs = sorted(Path(f"outputs/outputs/sil_training_{cfg}_class1_pairpruner_{variant}").glob("*/mc2_*"))
    return dirs[-1] if dirs else None


def service_rate(run_dir):
    total_serviced = total_requests = 0
    for inst in VAL_6:
        with open(run_dir / "val" / "epoch_4" / inst / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]
    return 100.0 * total_serviced / total_requests


def runtime_minutes(run_dir):
    lines = (run_dir / "main.log").read_text(errors="ignore").splitlines()
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    dt0 = datetime.strptime(lines[0][:23], fmt)
    dt1 = datetime.strptime(lines[-1][:23], fmt)
    return (dt1 - dt0).total_seconds() / 60.0


def make_bar_figure(data, cfg, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.suptitle(f"SIL {CONFIG_LABELS[cfg]}, class-1-only — Pair Pruner (v1_final)\n{title}", fontsize=12)
    xs = range(len(VARIANTS))
    vals = [data[v] for v in VARIANTS]
    colors = [VARIANT_COLORS[v] for v in VARIANTS]
    bars = ax.bar(xs, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=9.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved {OUT_DIR / out_stem}.png / .pdf")


def make_latex_table(results, out_path):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Configuration & Variant & Service Rate (\%) & Runtime (min) \\",
        r"\midrule",
    ]
    for cfg in CONFIGS:
        for i, variant in enumerate(VARIANTS):
            sr, rt = results[cfg][variant]
            cfg_cell = CONFIG_LABELS[cfg].replace("/", "/") if i == 0 else ""
            variant_label = VARIANT_LABELS[variant].replace("\n", " ")
            lines.append(f"{cfg_cell} & {variant_label} & {sr:.1f} & {rt:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\caption{Class-1-only Pair Pruner (GNN v1\_final) threshold sweep: service rate (6-instance validation set) and total wall-clock runtime (train + val, 5 epochs).}",
        r"\label{tab:class1_pairpruner_v1final_sweep}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    results = {}
    for cfg in CONFIGS:
        results[cfg] = {}
        service = {}
        runtime = {}
        print(f"=== {cfg} ===")
        for variant in VARIANTS:
            run_dir = find_run_dir(cfg, variant)
            if run_dir is None:
                print(f"{variant:10s}  NO OUTPUT DIR - skipping")
                continue
            sr = service_rate(run_dir)
            rt = runtime_minutes(run_dir)
            service[variant] = sr
            runtime[variant] = rt
            results[cfg][variant] = (sr, rt)
            print(f"{variant:10s}  service_rate={sr:5.1f}%   runtime={rt:6.1f} min")

        make_bar_figure(service, cfg, "Service Rate (%)", "Service Rate (6-instance val set)",
                         f"class1_pairpruner_v1final_{cfg}_service_rate", ylim=(0, 100))
        make_bar_figure(runtime, cfg, "Runtime (minutes, full 5-epoch run)", "Runtime",
                         f"class1_pairpruner_v1final_{cfg}_runtime")

    make_latex_table(results, OUT_DIR / "class1_pairpruner_v1final_sweep_table.tex")
