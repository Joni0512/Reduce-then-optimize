"""
2026-10-08: overview table + validation curves of ALL finished runs with the NEW buffered ("collect, then
update") SRL actor scheme (actor batch size 16, buffer = one epoch), see chat. Local runs are read from their
result.json; the three cluster actor_lr-scan runs (jobs 223553/54/55) are NOT available locally - their numbers
are copied from the cluster logs srl_buf_<jobid>.log that the user pasted into the chat on 2026-10-08
(complete after the last paste). Running sweep trials are not included.

Common settings of every run: twin critic, no warmstart, pickup_slack=True, batch_interval=200/step_size=100,
10 critic pretrain epochs, local_positive, gamma 0.99, tau 0.005, seed as listed. n=1 per configuration.
Reference line: RHO validation service rate = mean of 7 repeated RHO runs (keep_active=False), batch_interval=200/step_size=100.

Writes PNG + PDF (+ .tex for the table) into figures_export/SRL/.
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
REPO_ROOT_FOR_EXTRA = REPO
OUT = REPO / "figures_export" / "SRL"
SWEEP = REPO / "outputs" / "srl_training_sweep"
# 2026-10-09: RHO reference = mean of the 7 repeated RHO validation runs (keep_active=False, same commitment rule as the SRL loop), NOT the old single
# run 0.6891 (rho_val_bi200_rerun.log, keep_active unknown) - see chat.
import glob as _glob
import statistics as _st
RHO_VAL = _st.mean(json.load(open(p))['mean'] for p in sorted(_glob.glob(str(Path(__file__).resolve().parents[2] / 'outputs/rho_val_runs/bi200_ss100_card2_keepactiveFalse_seed42/run*/summary.json'))))
BASE_CLR = 0.0010833586558285635


def load_local(run_dir: str, label: str):
    d = json.load(open(SWEEP / run_dir / "result.json"))
    return {
        "label": label, "where": "local", "actor_lr": d["actor_lr"], "critic_lr": d["critic_lr"], "seed": d["seed"],
        "epochs": d["epochs"], "best_val": d["best_val_service_rate"], "best_epoch": d["best_epoch"],
        "test": d["test_service_rate"], "val": {r["epoch"]: r["service_rate"] for r in d["val_curve"]},
        "source": f"{run_dir}/result.json",
    }


RUNS = [
    load_local("buf_ep20_ps_alr0p0031923396291543868_s42", "A"),
    load_local("buf_ep20_ps_alr0p0031923396291543868_s42_clr0p001", "B"),
    load_local("buf_ep30_ps_alr0p0031923396291543868_s42_clr0p0015", "C"),
    load_local("buf_ep30_ps_alr0p0031923396291543868_s42_clr0p002", "D"),
    load_local("bufsweep_local_44a8ed4b", "E"),
    {"label": "F", "where": "cm4 223553", "actor_lr": 0.0064, "critic_lr": BASE_CLR, "seed": 42, "epochs": 20,
     "best_val": 0.7048, "best_epoch": 20, "test": 0.6815, "val": {5: 0.6894, 10: 0.6938, 15: 0.6828, 20: 0.7048}, "source": "srl_buf_223553.log"},
    {"label": "G", "where": "cm4 223554", "actor_lr": 0.0128, "critic_lr": BASE_CLR, "seed": 42, "epochs": 20,
     "best_val": 0.6853, "best_epoch": 15, "test": 0.6717, "val": {5: 0.6613, 10: 0.6575, 15: 0.6853, 20: 0.6421}, "source": "srl_buf_223554.log"},
    {"label": "H", "where": "cm4 223555", "actor_lr": 0.0255, "critic_lr": BASE_CLR, "seed": 42, "epochs": 20,
     "best_val": 0.0, "best_epoch": 5, "test": 0.0, "val": {5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}, "source": "srl_buf_223555.log"},
]
# 2026-10-09: finished sweep trials (shared wandb sweep): ethereal-1 is run E above; the others are added here. different-2 ran on serial_std
# (job 5648341), its per-epoch values are copied from the cluster print-out pasted into the chat; likely-4 / snowy-8 / feasible-10 ran on the Mac.
RUNS.append({"label": "I", "where": "serial 5648341 (sweep)", "actor_lr": 0.008850194777397007, "critic_lr": 0.001006074781381208, "seed": 6444, "epochs": 20,
             "best_val": 0.7282, "best_epoch": 15, "test": 0.7636, "val": {5: 0.702, 10: 0.618, 15: 0.728, 20: 0.720}, "source": "wandb different-sweep-2 / cluster result.json"})
for _lab, _dir in (("J", "bufsweep_local_19a6e5ec"), ("K", "bufsweep_local_057eb45d"), ("L", "bufsweep_local_751e6a08")):
    _r = load_local(_dir, _lab)
    _r["where"] = "local (sweep)"
    RUNS.append(_r)
# 2026-10-09/10: sweep trials finished later. true-11 (bufsweep_local_0794a737), hearty-1 (fe2e6f98) and deft-1 (e1d84337, wandb label "crashed" after the
# one-hour pause, but result.json and metrics are complete) ran on the Mac; peachy-5 / scarlet-6 / expert-7 ran on cm4 - their val curves and test come from
# wandb (logged by the cluster coordinator), the result.json stay on the cluster.
for _lab, _dir, _name in (("M", "bufsweep_local_0794a737", "true-11"), ("N", "bufsweep_local_fe2e6f98", "hearty-1"), ("O", "bufsweep_local_e1d84337", "deft-1")):
    _r = load_local(_dir, _lab)
    _r["where"] = f"local (sweep {_name})"
    RUNS.append(_r)
for _lab, _name, _alr, _clr, _seed, _best, _test, _val in (
    ("P", "peachy-5", 0.010131457010671273, 0.000748794234095617, 7166, 0.69747921745702, 0.7372892593400672, {5: 0.5852, 10: 0.6737, 15: 0.6779, 20: 0.6975}),
    ("Q", "scarlet-6", 0.002645611423567577, 0.0009366757372402624, 9331, 0.717988781063143, 0.6857514618311265, {5: 0.6181, 10: 0.7134, 15: 0.6895, 20: 0.718}),
    ("R", "expert-7", 0.009854264248957368, 0.0008543763311135315, 3590, 0.5860555697847595, 0.6046098088754398, {5: 0.4557, 10: 0.4668, 15: 0.5709, 20: 0.5861}),
):
    RUNS.append({"label": _lab, "where": f"cm4 (sweep {_name})", "actor_lr": _alr, "critic_lr": _clr, "seed": _seed, "epochs": 20, "best_val": _best,
                 "best_epoch": max(_val, key=_val.get), "test": _test, "val": _val, "source": f"wandb {_name}"})
# 2026-10-10: further finished local runs (v4 sweep trials and the 4 repeat seeds of config E) are picked up automatically from their result.json.
_known = {r["source"].split("/")[0] for r in RUNS if "source" in r}
_extra = sorted(glob.glob(str(REPO_ROOT_FOR_EXTRA / "outputs/srl_training_sweep/bufsweep_local_*/result.json"))) + sorted(glob.glob(str(REPO_ROOT_FOR_EXTRA / "outputs/srl_training_sweep/bestE_s*/result.json")))
_loaded_dirs = {"bufsweep_local_44a8ed4b", "bufsweep_local_19a6e5ec", "bufsweep_local_057eb45d", "bufsweep_local_751e6a08", "bufsweep_local_0794a737", "bufsweep_local_fe2e6f98", "bufsweep_local_e1d84337"}
for _i, _p in enumerate(_extra):
    _d = Path(_p).parent.name
    if _d in _loaded_dirs:
        continue
    _r = load_local(_d, "X%d" % _i)
    _r["where"] = "local (repeat of E)" if _d.startswith("bestE") else "local (sweep v4)"
    if _r["best_val"] < 0.1:
        pass  # collapsed trials stay in RUNS (the statistics scripts exclude best_val < 0.1)
    RUNS.append(_r)
for _r in RUNS:
    if _r["label"] == "E":
        _r["where"] = "local (sweep)"

COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73", "D": "#CC79A7", "E": "#D55E00", "F": "#56B4E9", "G": "#000000", "H": "#999999",
          "I": "#8C564B", "J": "#17BECF", "K": "#BCBD22", "L": "#E377C2",
          "M": "#1F77B4", "N": "#FF7F0E", "O": "#2CA02C", "P": "#D62728", "Q": "#9467BD", "R": "#7F7F7F"}


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def make_chart():
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in RUNS:
        xs = sorted(r["val"])
        cluster = r["where"].startswith("cm4")
        ax.plot(xs, [r["val"][x] for x in xs], marker="o", linewidth=2, markersize=6, color=COLORS.get(r["label"], "#888888"),
                linestyle="--" if (cluster or r["where"].startswith("serial")) else "-",
                label=f"{r['label']}: actor_lr={r['actor_lr']:g}, critic_lr={r['critic_lr']:.5g}, seed={r['seed']}, {r['epochs']} ep, {r['where']}")
    ax.axhline(RHO_VAL, color="#444444", linestyle=":", linewidth=1.5, label=f"RHO validation, mean of 7 runs (batch_interval=200/step_size=100, keep_active=False): {RHO_VAL:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation service rate (mean, 9 instances)")
    ax.set_xticks([5, 10, 15, 20, 25, 30])
    ax.set_ylim(-0.02, 0.85)
    ax.grid(True, alpha=0.3)
    ax.set_title("Buffered actor update (collect, then update, batch 16): validation service rate per epoch", fontsize=12)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    fig.text(0.01, 0.005, "Twin critic, no warmstart, pickup_slack, batch_interval=200/step_size=100, 10 pretrain epochs; n=1 per configuration. Dashed = cluster runs "
             "(values copied from cluster logs). Sources: result.json files / srl_buf_<jobid>.log.", fontsize=6.5, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = OUT / "buffered_actor_val_curves_all_runs"
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_table():
    cols = ["Run", "Where", "actor_lr", "critic_lr", "Seed", "Epochs", "Best val (epoch)", "Test", "Val @5", "@10", "@15", "@20", "@25", "@30"]
    rows = []
    for r in RUNS:
        v = r["val"]
        rows.append([
            r["label"], r["where"], f"{r['actor_lr']:g}", f"{r['critic_lr']:.5g}", str(r["seed"]), str(r["epochs"]),
            f"{r['best_val']:.4f} ({r['best_epoch']})",
            fmt(r["test"], 4), *[fmt(v.get(e)) if (e in v or e <= r["epochs"]) else "-" for e in (5, 10, 15, 20, 25, 30)],
        ])
    fig, ax = plt.subplots(figsize=(17, 3.6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    tbl.scale(1, 1.5)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor("#bbbbbb")
        if i == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
    ax.set_title("Buffered actor update, all finished runs (seed per row, n=1 each); common: twin critic, no warmstart, pickup_slack, batch_interval=200/step_size=100",
                 fontsize=9)
    out = OUT / "buffered_actor_results_table"
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    tex = ["\\begin{tabular}{llrrrrrrrrrrrr}", "\\hline",
           "Run & Where & actor LR & critic LR & Seed & Epochs & Best val (epoch) & Test & Val@5 & @10 & @15 & @20 & @25 & @30 \\\\", "\\hline"]
    for row in rows:
        tex.append(" & ".join(c.replace("_", "\\_").replace("(running)", "(running)") for c in row) + " \\\\")
    tex += ["\\hline", "\\end{tabular}"]
    (OUT / "buffered_actor_results_table.tex").write_text("\n".join(tex) + "\n")
    return rows, cols


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    make_chart()
    rows, cols = make_table()
    print(" | ".join(cols))
    for row in rows:
        print(" | ".join(row))
