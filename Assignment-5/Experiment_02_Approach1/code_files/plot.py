import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.35,
    "grid.linestyle": "--",
})

COLORS  = {"def": "#1f77b4", "imm": "#ff7f0e", "a4": "#2ca02c"}
MARKERS = {"def": "o",       "imm": "s",        "a4": "^"}
LABELS  = {
    "def": "Deferred (with ins/del)",
    "imm": "Immediate (with ins/del)",
    "a4":  "A4 Mover (no ins/del)",
}

GRIDS = [
    {"idx": 1, "NX": 250,  "NY": 100,  "label": "Grid 1 (NX=250, NY=100)"},
    {"idx": 2, "NX": 500,  "NY": 200,  "label": "Grid 2 (NX=500, NY=200)"},
    {"idx": 3, "NX": 1000, "NY": 400,  "label": "Grid 3 (NX=1000, NY=400)"},
]

THREAD_COUNTS = [1, 2, 4, 8, 16]

def load_csv(fpath):
    if not os.path.exists(fpath):
        print(f"  [WARNING] Not found: {os.path.basename(fpath)}")
        return {}

    data = np.loadtxt(fpath, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    result = {}
    for t in np.unique(data[:, 0]).astype(int):
        rows = data[data[:, 0] == t]
        result[t] = {
            "interp": np.mean(rows[:, 2]),
            "mover":  np.mean(rows[:, 3]),
            "total":  np.mean(rows[:, 4]),
        }
    return result

def load_grid(g, machine_suffix=""):
    suffix = f"_cluster" if machine_suffix == "cluster" else ""
    return {
        "def": load_csv(os.path.join(BASE, f"exp2_deferred_grid{g}{suffix}.csv")),
        "imm": load_csv(os.path.join(BASE, f"exp2_immediate_grid{g}{suffix}.csv")),
        "a4":  load_csv(os.path.join(BASE, f"exp2_a4mover_grid{g}{suffix}.csv")),
    }

def compute_speedup(data_dict, col="mover"):
    
    if not data_dict or 1 not in data_dict:
        return [], []
    t1 = data_dict[1][col]
    threads  = sorted(data_dict.keys())
    speedups = [t1 / data_dict[t][col] for t in threads]
    return threads, speedups

# Speedup vs Threads
print("\n" + "="*60)
print("PLOT A — Speedup vs Threads")
print("="*60)

for grid in GRIDS:
    g = grid["idx"]

    lab     = load_grid(g, "lab")
    cluster = load_grid(g, "cluster")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        f"Experiment 2 — Speedup vs Threads | {grid['label']}",
        fontsize=13, fontweight="bold"
    )

    for ax, machine_label, data in [
        (axes[0], "Lab PC",      lab),
        (axes[1], "HPC Cluster", cluster),
    ]:
        # Ideal reference
        max_t = max(THREAD_COUNTS)
        ax.plot([1, max_t], [1, max_t], "k--", linewidth=1.2, label="Ideal Speedup")

        for key in ["def", "imm", "a4"]:
            ts, sp = compute_speedup(data[key], col="mover")
            if ts:
                ax.plot(
                    ts, sp,
                    color=COLORS[key], marker=MARKERS[key],
                    label=LABELS[key], linewidth=1.8, markersize=8,
                    markerfacecolor="white", markeredgewidth=2,
                )

        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Speedup  (T₁ / Tₙ)")
        ax.set_title(machine_label)
        ax.set_xticks(THREAD_COUNTS)
        ax.set_xlim(0.5, max_t + 1)
        ax.set_ylim(bottom=0.5)
        ax.legend()

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp2_speedup_grid{g}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp2_speedup_grid{g}.png")

# Absolute Mover Time vs Threads
print("\nPLOT B — Absolute Mover Time vs Threads")

for grid in GRIDS:
    g = grid["idx"]

    lab     = load_grid(g, "lab")
    cluster = load_grid(g, "cluster")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        f"Experiment 2 — Mover Time vs Threads | {grid['label']}",
        fontsize=13, fontweight="bold"
    )

    for ax, machine_label, data in [
        (axes[0], "Lab PC",      lab),
        (axes[1], "HPC Cluster", cluster),
    ]:
        for key in ["def", "imm", "a4"]:
            d = data[key]
            if not d:
                continue
            ts = sorted(d.keys())
            times = [d[t]["mover"] for t in ts]
            ax.plot(
                ts, times,
                color=COLORS[key], marker=MARKERS[key],
                label=LABELS[key], linewidth=1.8, markersize=8,
                markerfacecolor="white", markeredgewidth=2,
            )

        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Mean Mover Time (s)")
        ax.set_title(machine_label)
        ax.set_xticks(THREAD_COUNTS)
        ax.set_xlim(0.5, max(THREAD_COUNTS) + 1)
        ax.legend()

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp2_movertime_grid{g}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp2_movertime_grid{g}.png")

# Interpolation vs Mover Time Breakdown - Immediate
print("\nPLOT C1 — Interpolation vs Mover Time Breakdown")

BAR_COLORS = {"interp": "#4e79a7", "mover": "#f28e2b"}

for grid in GRIDS:
    g = grid["idx"]

    lab_d     = load_csv(os.path.join(BASE, f"exp2_immediate_grid{g}.csv"))
    cluster_d = load_csv(os.path.join(BASE, f"exp2_immediate_grid{g}_cluster.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        f"Experiment 2 — Time Breakdown (Immediate) | {grid['label']}",
        fontsize=13, fontweight="bold"
    )

    for ax, machine_label, data in [
        (axes[0], "Lab PC",      lab_d),
        (axes[1], "HPC Cluster", cluster_d),
    ]:
        if not data:
            ax.set_title(f"{machine_label} — No Data")
            continue

        ts      = sorted(data.keys())
        interps = [data[t]["interp"] for t in ts]
        movers  = [data[t]["mover"]  for t in ts]
        x       = np.arange(len(ts))
        width   = 0.5

        ax.bar(x, interps, width, label="Interpolation", color=BAR_COLORS["interp"])
        ax.bar(x, movers,  width, bottom=interps,        label="Mover",         color=BAR_COLORS["mover"])

        ax.set_xticks(x)
        ax.set_xticklabels([f"T={t}" for t in ts])
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Mean Time per Iteration (s)")
        ax.set_title(machine_label)
        ax.legend()

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp2_breakdown_immediate_grid{g}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp2_breakdown_grid{g}.png")

# Interpolation vs Mover Time Breakdown - Deffered
print("\nPLOT C2 — Interpolation vs Mover Time Breakdown")

BAR_COLORS = {"interp": "#4e79a7", "mover": "#f28e2b"}

for grid in GRIDS:
    g = grid["idx"]

    lab_d     = load_csv(os.path.join(BASE, f"exp2_deferred_grid{g}.csv"))
    cluster_d = load_csv(os.path.join(BASE, f"exp2_deferred_grid{g}_cluster.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        f"Experiment 2 — Time Breakdown (Deferred) | {grid['label']}",
        fontsize=13, fontweight="bold"
    )

    for ax, machine_label, data in [
        (axes[0], "Lab PC",      lab_d),
        (axes[1], "HPC Cluster", cluster_d),
    ]:
        if not data:
            ax.set_title(f"{machine_label} — No Data")
            continue

        ts      = sorted(data.keys())
        interps = [data[t]["interp"] for t in ts]
        movers  = [data[t]["mover"]  for t in ts]
        x       = np.arange(len(ts))
        width   = 0.5

        ax.bar(x, interps, width, label="Interpolation", color=BAR_COLORS["interp"])
        ax.bar(x, movers,  width, bottom=interps,        label="Mover",         color=BAR_COLORS["mover"])

        ax.set_xticks(x)
        ax.set_xticklabels([f"T={t}" for t in ts])
        ax.set_xlabel("Number of Threads")
        ax.set_ylabel("Mean Time per Iteration (s)")
        ax.set_title(machine_label)
        ax.legend()

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp2_breakdown_deffered_grid{g}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: exp2_breakdown_grid{g}.png")

# PLOT D — Speedup across all 3 grids on one figure (Lab PC only)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle(
    "Experiment 2 — Speedup vs Threads (Lab PC, All Grids)",
    fontsize=13, fontweight="bold"
)

for ax, grid in zip(axes, GRIDS):
    g   = grid["idx"]
    lab = load_grid(g, "lab")

    max_t = max(THREAD_COUNTS)
    ax.plot([1, max_t], [1, max_t], "k--", linewidth=1.2, label="Ideal")

    for key in ["def", "imm", "a4"]:
        ts, sp = compute_speedup(lab[key], col="mover")
        if ts:
            ax.plot(
                ts, sp,
                color=COLORS[key], marker=MARKERS[key],
                label=LABELS[key], linewidth=1.8, markersize=7,
                markerfacecolor="white", markeredgewidth=2,
            )

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.set_title(grid["label"])
    ax.set_xticks(THREAD_COUNTS)
    ax.set_xlim(0.5, max_t + 1)
    ax.set_ylim(bottom=0.5)
    ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "exp2_speedup_all_grids_lab.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: exp2_speedup_all_grids_lab.png")

# PLOT E — Speedup across all 3 grids on one figure (HPC cluster only)
print("\nPLOT E — All grids speedup comparison (HPC Cluster)")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle(
    "Experiment 2 — Speedup vs Threads (HPC Cluster, All Grids)",
    fontsize=13, fontweight="bold"
)

for ax, grid in zip(axes, GRIDS):
    g   = grid["idx"]
    lab = load_grid(g, "lab")

    max_t = max(THREAD_COUNTS)
    ax.plot([1, max_t], [1, max_t], "k--", linewidth=1.2, label="Ideal")

    for key in ["def", "imm", "a4"]:
        ts, sp = compute_speedup(cluster[key], col="mover")
        if ts:
            ax.plot(
                ts, sp,
                color=COLORS[key], marker=MARKERS[key],
                label=LABELS[key], linewidth=1.8, markersize=7,
                markerfacecolor="white", markeredgewidth=2,
            )

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Speedup")
    ax.set_title(grid["label"])
    ax.set_xticks(THREAD_COUNTS)
    ax.set_xlim(0.5, max_t + 1)
    ax.set_ylim(bottom=0.5)
    ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, "exp2_speedup_all_grids_cluster.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: exp2_speedup_all_grids_cluster.png")

print("\nAll Experiment 2 plots done.")