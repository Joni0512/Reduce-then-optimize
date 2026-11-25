"""2026-11-25: collect the 12-instance RHO comparison LKH-3 step solver vs RTV+ILP (max_cardinality=2),
5 windows each, into CSV/Markdown tables + PNG/PDF charts (figures_export/LKH RHO/).
Sources: outputs/rho_lkh/12inst/*/run_*/*/final/results.json + lkh_step_stats.json,
         outputs/rho_ilp/12inst/*_mc2/run_*/*/final/results.json, reference VMT solutions/li_lim/liLim_solution_scores.json.
Runtime = results.json stats.total_time; runs overlapping the Mac sleep 2026-11-25 03:08:22-12:43:48 are flagged and
corrected by subtracting the sleep length (34 526 s) - only lr210 batch_interval=200/step_size=50 LKH is affected."""
import glob, json, os, csv
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/joni/Desktop/Masterarbeit/Reduce-then-optimize"
INST = "lc108 lc109 lc207 lc208 lr111 lr112 lr210 lr211 lrc107 lrc108 lrc207 lrc208".split()
WIN = [(40, 10), (100, 25), (120, 30), (200, 50), (400, 100)]
SLEEP = (datetime(2026, 11, 25, 3, 8, 22).timestamp(), datetime(2026, 11, 25, 12, 43, 48).timestamp())
REF = json.load(open(f"{ROOT}/solutions/li_lim/liLim_solution_scores.json"))
OUT = f"{ROOT}/figures_export/LKH RHO"
os.makedirs(OUT, exist_ok=True)


def load(d):
    fs = sorted(glob.glob(f"{d}/run_*/*/final/results.json"))
    if not fs:
        return None
    f = fs[-1]
    r = json.load(open(f))
    s = r["stats"]
    run = os.path.dirname(os.path.dirname(f))
    end = os.path.getmtime(f)
    t = s["total_time"]
    start = end - t
    slept = max(0.0, min(end, SLEEP[1]) - max(start, SLEEP[0]))
    row = dict(served=s["serviced"], total=s["total_requests"], vmt=s["vmt"], wait=s["average_wait_time"],
               viol=len(r["violations"]), runtime_raw=t, runtime=t - slept, slept=slept > 0,
               src=os.path.relpath(f, ROOT))
    st = f"{run}/lkh_step_stats.json"
    if os.path.exists(st):
        k = json.load(open(st))
        row.update(steps=k["steps"], fallback=k["steps_fallback_ilp"], dropped=k["dropped_requests"],
                   lkh_time=k["lkh_time_s"], run_name=os.path.basename(os.path.dirname(run)))
    return row


rows = []
for i in INST:
    for bi, ss in WIN:
        l = load(f"{ROOT}/outputs/rho_lkh/12inst/{i}_bi{bi}_ss{ss}_mt10000_special")
        g = load(f"{ROOT}/outputs/rho_ilp/12inst/{i}_bi{bi}_ss{ss}_mc2")
        rows.append(dict(inst=i, bi=bi, ss=ss, lkh=l, ilp=g, ref=REF.get(i)))

# ---- CSV + Markdown
cols = ["instance", "batch_interval", "step_size", "requests", "ref_vmt",
        "ILP_served", "ILP_vmt", "ILP_wait", "ILP_viol", "ILP_runtime_s",
        "LKH_served", "LKH_vmt", "LKH_wait", "LKH_viol", "LKH_runtime_s", "LKH_steps", "LKH_fallback_steps",
        "LKH_dropped", "note", "ILP_source", "LKH_source"]
def f1(x): return "" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))
table = []
for r in rows:
    l, g = r["lkh"], r["ilp"]
    tot = (l or g)["total"]
    note = []
    if l is None: note.append("LKH TIMEOUT 7200 s")
    if g is None: note.append("ILP TIMEOUT 7200 s")
    if l and l["slept"]: note.append(f"LKH runtime minus sleep (raw {l['runtime_raw']:.0f} s)")
    table.append([r["inst"], r["bi"], r["ss"], tot, r["ref"],
                  *(([g["served"], g["vmt"], g["wait"], g["viol"], g["runtime"]]) if g else ["", "", "", "", ">7200"]),
                  *(([l["served"], l["vmt"], l["wait"], l["viol"], l["runtime"], l.get("steps"), l.get("fallback"), l.get("dropped")])
                    if l else ["", "", "", "", ">7200", "", "", ""]),
                  "; ".join(note), g["src"] if g else "", l["src"] if l else ""])
with open(f"{OUT}/12inst_lkh_vs_ilp_mc2.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols); w.writerows(table)
with open(f"{OUT}/12inst_lkh_vs_ilp_mc2.md", "w") as fh:
    c = cols[:-2]
    fh.write("| " + " | ".join(c) + " |\n|" + "---|" * len(c) + "\n")
    for t in table:
        fh.write("| " + " | ".join(f1(x) for x in t[:-2]) + " |\n")

# ---- summaries printed for chat
print("PER WINDOW (sum over 12 instances; timeouts excluded pairwise)")
for bi, ss in WIN:
    rs = [r for r in rows if r["bi"] == bi and r["lkh"] and r["ilp"]]
    n_to = sum(1 for r in rows if r["bi"] == bi and not (r["lkh"] and r["ilp"]))
    print(bi, ss, "n", len(rs), "timeouts", n_to,
          "served ILP", sum(r["ilp"]["served"] for r in rs), "LKH", sum(r["lkh"]["served"] for r in rs),
          "of", sum(r["ilp"]["total"] for r in rs),
          "VMT ILP %.1f LKH %.1f" % (sum(r["ilp"]["vmt"] for r in rs), sum(r["lkh"]["vmt"] for r in rs)),
          "rt med ILP %.1f LKH %.1f" % (np.median([r["ilp"]["runtime"] for r in rs]), np.median([r["lkh"]["runtime"] for r in rs])),
          "rt max ILP %.0f LKH %.0f" % (max(r["ilp"]["runtime"] for r in rs), max(r["lkh"]["runtime"] for r in rs)),
          "fallback steps", sum(r["lkh"]["fallback"] for r in rs), "dropped", sum(r["lkh"]["dropped"] for r in rs),
          "viol", sum(r["ilp"]["viol"] + r["lkh"]["viol"] for r in rs))
print("PER INSTANCE")
for i in INST:
    rs = [r for r in rows if r["inst"] == i]
    print(i, " ".join(
        f"{r['bi']}/{r['ss']}:ILP{(r['ilp'] or {}).get('served','TO')}/LKH{(r['lkh'] or {}).get('served','TO')}"
        f"({(r['lkh'] or {}).get('fallback','-')}fb)" for r in rs),
        "ref", rs[0]["ref"], "tot", (rs[0]['ilp'] or rs[0]['lkh'])['total'])

# ---- charts: served rate, VMT ratio to reference, runtime
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
def style(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

def grid_chart(metric, ylabel, fname, title, log=False):
    fig, axes = plt.subplots(len(WIN), 1, figsize=(11, 2.1 * len(WIN)), sharex=True)
    x = np.arange(len(INST)); wdt = 0.38
    for ax, (bi, ss) in zip(axes, WIN):
        for k, (key, col, lab, off) in enumerate([("ilp", BLUE, "RTV + ILP (max_cardinality=2)", -wdt / 2 - 0.01),
                                                    ("lkh", ORANGE, "LKH-3 step (MAX_TRIALS=10000, SPECIAL)", wdt / 2 + 0.01)]):
            vals = []
            for i in INST:
                r = next(r for r in rows if r["inst"] == i and r["bi"] == bi)
                vals.append(metric(r, key))
            v = np.array([np.nan if a is None else a for a in vals], float)
            ax.bar(x + off, np.nan_to_num(v), wdt, color=col, label=lab, edgecolor="white", linewidth=0.5)
            for xi, a in zip(x, vals):
                if a is None:
                    ax.text(xi + off, 0.02, "TO", ha="center", va="bottom", fontsize=7, color=INK, transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"batch_interval={bi}\nstep_size={ss}", fontsize=7, color=INK)
        if log: ax.set_yscale("log")
        style(ax)
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(INST, fontsize=8, color=INK)
    h, lb = axes[0].get_legend_handles_labels(); fig.legend(h, lb, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.975), ncol=2)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, f"{ylabel}. TO = 7200 s wall-clock cap hit. Source: outputs/rho_lkh/12inst, outputs/rho_ilp/12inst "
             "(collect script scripts/rho_lkh/collect_12inst.py). LKH: RUNS=1, TIME_LIMIT=10 s, R1 rejection, ILP fallback mc2.",
             fontsize=6, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{fname}.{ext}", dpi=200)
    plt.close(fig)

grid_chart(lambda r, k: None if r[k] is None else 100 * r[k]["served"] / r[k]["total"],
           "Service rate [%]", "12inst_service_rate", "RHO on 12 Li&Lim instances: service rate, RTV+ILP vs LKH-3 per step")
grid_chart(lambda r, k: None if r[k] is None else r[k]["runtime"],
           "Runtime [s], log scale", "12inst_runtime", "RHO on 12 Li&Lim instances: runtime per run, RTV+ILP vs LKH-3 per step", log=True)
grid_chart(lambda r, k: None if r[k] is None else r[k]["vmt"] / r["ref"],
           "VMT / reference VMT (published best, all requests)", "12inst_vmt_ratio",
           "RHO on 12 Li&Lim instances: VMT relative to reference solution")
print("written to", OUT)
