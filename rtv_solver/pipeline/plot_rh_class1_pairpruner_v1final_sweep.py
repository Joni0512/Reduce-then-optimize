"""
2026-08-10: Service Rate + Runtime bar charts and LaTeX table for the
class-1-only RHO WITHOUT learning (--mode offline, myopic rolling-horizon
heuristic, no SIL/ML training) Pair Pruner (GNN v1_final) threshold sweep,
bi200/ss100, bi400/ss80, bi400/ss200. Counterpart to
plot_class1_pairpruner_v1final_sweep.py (the SIL version) - same window
configs, same 6-instance class-1 set, same GNN checkpoint, same thresholds,
seed=42 (explicit in every run script and in this plot's titles/table so
results can't be silently mixed up with a different seed/config later).
Baseline was run once per instance and reused across the three thresholds.
Each instance gets its own main.log (offline mode calls main.py once per
instance), so runtime per variant is the SUM of each instance's own
first-to-last-log-line wall-clock time, not a single continuous log like
the SIL/coaml runs.
"""
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

VAL_6 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
CONFIGS = ["bi200_ss100", "bi400_ss80", "bi400_ss200"]
CONFIG_LABELS = {"bi200_ss100": "bi200/ss100", "bi400_ss80": "bi400/ss80", "bi400_ss200": "bi400/ss200"}
SEED = 42
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


def instance_run_dir(cfg, variant, inst):
    base = Path(f"outputs/new_tests/rh_class1_pairpruner_v1final/{cfg}_seed{SEED}/{variant}/{inst}")
    dirs = sorted(base.glob("run_offline_*/*"))
    return dirs[-1] if dirs else None


def service_rate_and_runtime(cfg, variant):
    total_serviced = total_requests = 0
    total_minutes = 0.0
    for inst in VAL_6:
        run_dir = instance_run_dir(cfg, variant, inst)
        if run_dir is None:
            raise FileNotFoundError(f"no run dir for {cfg}/{variant}/{inst}")
        with open(run_dir / "final" / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]

        lines = (run_dir / "main.log").read_text(errors="ignore").splitlines()
        fmt = "%Y-%m-%d %H:%M:%S,%f"
        dt0 = datetime.strptime(lines[0][:23], fmt)
        dt1 = datetime.strptime(lines[-1][:23], fmt)
        total_minutes += (dt1 - dt0).total_seconds() / 60.0

    sr = 100.0 * total_serviced / total_requests
    return sr, total_minutes, total_serviced, total_requests


def make_bar_figure(data, cfg, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.suptitle(f"RHO without learning, {CONFIG_LABELS[cfg]}, class-1-only, seed={SEED}\nPair Pruner (v1_final) — {title}", fontsize=11.5)
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


def make_tex_figure_wrapper(stem, cfg, metric_desc):
    label = f"fig:{stem}"
    tex = (
        "\\begin{figure}[h]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.8\\textwidth]{{{stem}.pdf}}\n"
        f"    \\caption{{{metric_desc} of the class-1-only Pair Pruner (GNN v1\\_final) threshold sweep under RHO without learning (myopic rolling horizon, no SIL training), {CONFIG_LABELS[cfg]}, seed={SEED}.}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n"
    )
    (OUT_DIR / f"{stem}.tex").write_text(tex)
    print(f"saved {OUT_DIR / stem}.tex")


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
            cfg_cell = CONFIG_LABELS[cfg] if i == 0 else ""
            variant_label = VARIANT_LABELS[variant].replace("\n", " ")
            lines.append(f"{cfg_cell} & {variant_label} & {sr:.1f} & {rt:.2f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        rf"\caption{{Class-1-only Pair Pruner (GNN v1\_final) threshold sweep under RHO without learning (myopic rolling horizon, no SIL training), seed={SEED}: service rate (6-instance set) and total wall-clock runtime (sum over 6 instances).}}",
        r"\label{tab:rh_class1_pairpruner_v1final_sweep}",
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
        print(f"=== {cfg} (seed={SEED}) ===")
        for variant in VARIANTS:
            sr, rt, serviced, total = service_rate_and_runtime(cfg, variant)
            service[variant] = sr
            runtime[variant] = rt
            results[cfg][variant] = (sr, rt)
            print(f"{variant:10s}  service_rate={sr:5.1f}% ({serviced}/{total})   runtime={rt:6.2f} min")

        sr_stem = f"rh_class1_pairpruner_v1final_{cfg}_service_rate"
        rt_stem = f"rh_class1_pairpruner_v1final_{cfg}_runtime"
        make_bar_figure(service, cfg, "Service Rate (%)", "Service Rate (6-instance set)", sr_stem, ylim=(0, 100))
        make_bar_figure(runtime, cfg, "Runtime (minutes, sum over 6 instances)", "Runtime", rt_stem, value_fmt="{:.2f}")
        make_tex_figure_wrapper(sr_stem, cfg, "Service rate")
        make_tex_figure_wrapper(rt_stem, cfg, "Runtime")

    make_latex_table(results, OUT_DIR / "rh_class1_pairpruner_v1final_sweep_table.tex")
