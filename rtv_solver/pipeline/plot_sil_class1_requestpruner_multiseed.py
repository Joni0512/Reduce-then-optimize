"""
2026-08-11: 5-seed (42, 1, 2, 3, 4) mean +/- std Service Rate and Runtime
bar charts and LaTeX table for the class-1-only SIL (COAML) Request Pruner
threshold sweep (t=0.4/0.5/0.6), bi200/ss100, bi400/ss80, bi400/ss200.
Each config uses its OWN window-specific Request Pruner MLP checkpoint
(outputs/request_pruner_mlp_bi<BI>_ss<SS>/...) - see
build_window_specific_request_pruners.py for why. Baseline was run once
per seed and reused across thresholds. --epochs 5 --learning_rate 0.0001
per thesis Table 3. Counterpart to plot_rh_class1_pairpruner_v1final_
multiseed.py (RHO version) and plot_class1_pairpruner_v1final_sweep.py
(single-seed Pair Pruner SIL version).
"""
import json
import statistics
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

VAL_6 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
CONFIGS = ["bi200_ss100", "bi400_ss80", "bi400_ss200"]
CONFIG_LABELS = {"bi200_ss100": "bi200/ss100", "bi400_ss80": "bi400/ss80", "bi400_ss200": "bi400/ss200"}
SEEDS = [42, 1, 2, 3, 4]
VARIANTS = ["baseline", "request_t0.4", "request_t0.5", "request_t0.6"]
VARIANT_LABELS = {
    "baseline": "No Pruner",
    "request_t0.4": "Request Pruner\n(t=0.4)",
    "request_t0.5": "Request Pruner\n(t=0.5)",
    "request_t0.6": "Request Pruner\n(t=0.6)",
}
VARIANT_COLORS = {
    "baseline": "#9AA5A9",
    "request_t0.4": "#A6493A",
    "request_t0.5": "#8B1E1E",
    "request_t0.6": "#5A1010",
}
OUT_DIR = Path("figures_export")


def find_run_dir(cfg, variant, seed):
    dirs = sorted(Path(f"outputs/outputs/sil_training_{cfg}_class1_requestpruner_{variant}_seed{seed}").glob("*/mc2_*"))
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


def mean_std(values):
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def make_bar_figure(means, stds, cfg, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.suptitle(
        f"SIL (COAML), {CONFIG_LABELS[cfg]}, class-1-only, mean over 5 seeds ({','.join(map(str, SEEDS))})\n"
        f"Request Pruner (window-specific MLP) — {title}",
        fontsize=11,
    )
    xs = range(len(VARIANTS))
    vals = [means[v] for v in VARIANTS]
    errs = [stds[v] for v in VARIANTS]
    colors = [VARIANT_COLORS[v] for v in VARIANTS]
    bars = ax.bar(xs, vals, yerr=errs, capsize=5, color=colors, width=0.6,
                   error_kw={"ecolor": "#333333", "elinewidth": 1.2})
    for b, v, e in zip(bars, vals, errs):
        ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + e),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
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
    seeds_str = ", ".join(map(str, SEEDS))
    tex = (
        "\\begin{figure}[h]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.8\\textwidth]{{{stem}.pdf}}\n"
        f"    \\caption{{{metric_desc} (mean $\\pm$ 1 std over 5 seeds: {seeds_str}) of the class-1-only Request Pruner (window-specific MLP checkpoint) threshold sweep under SIL (COAML, 5 epochs, lr=0.0001), {CONFIG_LABELS[cfg]}.}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n"
    )
    (OUT_DIR / f"{stem}.tex").write_text(tex)
    print(f"saved {OUT_DIR / stem}.tex")


def make_latex_table(results, out_path):
    seeds_str = ", ".join(map(str, SEEDS))
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
            sr_m, sr_s, rt_m, rt_s = results[cfg][variant]
            cfg_cell = CONFIG_LABELS[cfg] if i == 0 else ""
            variant_label = VARIANT_LABELS[variant].replace("\n", " ")
            lines.append(
                f"{cfg_cell} & {variant_label} & {sr_m:.1f} $\\pm$ {sr_s:.1f} & {rt_m:.1f} $\\pm$ {rt_s:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        rf"\caption{{Class-1-only Request Pruner (window-specific MLP checkpoints) threshold sweep under SIL (COAML, 5 epochs, lr=0.0001): service rate (6-instance val set) and total wall-clock runtime (train+val), mean $\pm$ 1 std over 5 seeds ({seeds_str}).}}",
        r"\label{tab:sil_class1_requestpruner_multiseed_sweep}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    results = {}
    for cfg in CONFIGS:
        results[cfg] = {}
        means_sr, stds_sr = {}, {}
        means_rt, stds_rt = {}, {}
        print(f"=== {cfg} (seeds={SEEDS}) ===")
        for variant in VARIANTS:
            srs, rts = [], []
            for seed in SEEDS:
                run_dir = find_run_dir(cfg, variant, seed)
                if run_dir is None:
                    raise FileNotFoundError(f"no run dir for {cfg}/{variant}/seed{seed}")
                srs.append(service_rate(run_dir))
                rts.append(runtime_minutes(run_dir))
            sr_m, sr_s = mean_std(srs)
            rt_m, rt_s = mean_std(rts)
            means_sr[variant], stds_sr[variant] = sr_m, sr_s
            means_rt[variant], stds_rt[variant] = rt_m, rt_s
            results[cfg][variant] = (sr_m, sr_s, rt_m, rt_s)
            print(f"{variant:14s}  service_rate={sr_m:5.1f}% (+/-{sr_s:.1f})   runtime={rt_m:6.1f} min (+/-{rt_s:.1f})   raw_sr={['%.1f' % x for x in srs]}")

        sr_stem = f"sil_class1_requestpruner_{cfg}_service_rate_5seed"
        rt_stem = f"sil_class1_requestpruner_{cfg}_runtime_5seed"
        make_bar_figure(means_sr, stds_sr, cfg, "Service Rate (%)", "Service Rate (6-instance val set)", sr_stem, ylim=(0, 100))
        make_bar_figure(means_rt, stds_rt, cfg, "Runtime (minutes, full 5-epoch run)", "Runtime", rt_stem)
        make_tex_figure_wrapper(sr_stem, cfg, "Service rate")
        make_tex_figure_wrapper(rt_stem, cfg, "Runtime")

    make_latex_table(results, OUT_DIR / "sil_class1_requestpruner_multiseed_sweep_table.tex")
