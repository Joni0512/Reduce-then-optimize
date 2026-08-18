"""
2026-08-18: Service Rate + Runtime bar chart and LaTeX table for the
min_retention_fraction sweep (0.3/0.4/0.5/0.6) of the class-1-only Request
Pruner under RHO without learning (myopic rolling horizon, no SIL
training), bi200/ss100, threshold fixed at 0.5, seed=42. min_retention_fraction
is the retention-floor safety measure (see request_pruner.py) - was a
hardcoded constructor default with no CLI override until
--request_pruner_min_retention_fraction was added (main.py/config.py/
trip_handler.py) specifically for this sweep.
"""
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

VAL_6 = ["lc108", "lc109", "lr111", "lr112", "lrc107", "lrc108"]
SEED = 42
THRESHOLD = "0.5"
MIN_RETENTIONS = ["0.3", "0.4", "0.5", "0.6"]
COLORS = ["#5A1010", "#A6493A", "#276575", "#3E8F6C"]
OUT_DIR = Path("figures_export")


def find_run_dir(mr, inst):
    base = Path(f"outputs/new_tests/rh_bi200_ss100_class1_requestpruner_retentionsweep_seed{SEED}/minret{mr}/{inst}")
    dirs = sorted(base.glob("run_offline_*/*"))
    return dirs[-1] if dirs else None


def service_rate_and_runtime(mr):
    total_serviced = total_requests = 0
    total_minutes = 0.0
    for inst in VAL_6:
        run_dir = find_run_dir(mr, inst)
        with open(run_dir / "final" / "results.json") as f:
            stats = json.load(f)["stats"]
        total_serviced += stats["serviced"]
        total_requests += stats["total_requests"]
        lines = (run_dir / "main.log").read_text(errors="ignore").splitlines()
        fmt = "%Y-%m-%d %H:%M:%S,%f"
        dt0 = datetime.strptime(lines[0][:23], fmt)
        dt1 = datetime.strptime(lines[-1][:23], fmt)
        total_minutes += (dt1 - dt0).total_seconds() / 60.0
    return 100.0 * total_serviced / total_requests, total_minutes, total_serviced, total_requests


def make_bar_figure(data, ylabel, title, out_stem, value_fmt="{:.1f}", ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fig.suptitle(
        f"RHO without learning, bi200/ss100, class-1-only, seed={SEED}\n"
        f"Request Pruner (t={THRESHOLD}) — {title}, min_retention_fraction sweep",
        fontsize=11,
    )
    xs = range(len(MIN_RETENTIONS))
    vals = [data[mr] for mr in MIN_RETENTIONS]
    bars = ax.bar(xs, vals, color=COLORS, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(value_fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"min_retention\n={mr}" for mr in MIN_RETENTIONS], fontsize=9.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{out_stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"saved {OUT_DIR / out_stem}.png / .pdf")


def make_tex_figure_wrapper(stem, metric_desc):
    label = f"fig:{stem}"
    tex = (
        "\\begin{figure}[h]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.8\\textwidth]{{{stem}.pdf}}\n"
        f"    \\caption{{{metric_desc} of the class-1-only Request Pruner min\\_retention\\_fraction sweep (threshold fixed at {THRESHOLD}) under RHO without learning (myopic rolling horizon, no SIL training), bi200/ss100, seed={SEED}.}}\n"
        f"    \\label{{{label}}}\n"
        "\\end{figure}\n"
    )
    (OUT_DIR / f"{stem}.tex").write_text(tex)
    print(f"saved {OUT_DIR / stem}.tex")


def make_latex_table(results, out_path):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{cc}",
        r"\toprule",
        r"min\_retention\_fraction & Service Rate (\%) \\",
        r"\midrule",
    ]
    for mr in MIN_RETENTIONS:
        sr, rt = results[mr]
        lines.append(f"{mr} & {sr:.1f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Class-1-only Request Pruner min\_retention\_fraction sweep (threshold fixed at {THRESHOLD}) under RHO without learning, bi200/ss100, seed={SEED}: service rate (6-instance set).}}",
        r"\label{tab:rh_bi200ss100_requestpruner_retentionsweep}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    service, runtime, results = {}, {}, {}
    for mr in MIN_RETENTIONS:
        sr, rt, serviced, total = service_rate_and_runtime(mr)
        service[mr] = sr
        runtime[mr] = rt
        results[mr] = (sr, rt)
        print(f"min_retention={mr}  service_rate={sr:5.1f}% ({serviced}/{total})   runtime={rt:.2f} min")

    sr_stem = "rh_bi200ss100_requestpruner_retentionsweep_service_rate"
    rt_stem = "rh_bi200ss100_requestpruner_retentionsweep_runtime"
    make_bar_figure(service, "Service Rate (%)", "Service Rate (6-instance set)", sr_stem, ylim=(0, 100))
    make_bar_figure(runtime, "Runtime (minutes, sum over 6 instances)", "Runtime", rt_stem, value_fmt="{:.2f}")
    make_tex_figure_wrapper(sr_stem, "Service rate")
    make_tex_figure_wrapper(rt_stem, "Runtime")
    make_latex_table(results, OUT_DIR / "rh_bi200ss100_requestpruner_retentionsweep_table.tex")
