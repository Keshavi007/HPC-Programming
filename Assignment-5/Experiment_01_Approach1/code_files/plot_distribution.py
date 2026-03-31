import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    1: {"NX": 250,  "NY": 100},
    2: {"NX": 500,  "NY": 200},
    3: {"NX": 1000, "NY": 400},
}
APPROACHES = {
    "def": "Deferred",
    "imm": "Immediate",
}
MACHINES = {
    "labPC":   "Lab PC",
    "cluster": "HPC Cluster",
}

HIST_BINS   = 50
OUTPUT_DIR  = BASE

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})

def load_particles(fpath):
    """Load x,y from a comma-separated .out file. Returns (xs, ys) as np arrays."""
    data = np.loadtxt(fpath, delimiter=",")
    return data[:, 0], data[:, 1]

def plot_distribution(xs, ys, NX, NY, title_tag, out_path):
    """
    2×2 figure:
      [0,0] Scatter   [0,1] X-histogram
      [1,0] Y-histogram  [1,1] Cell heatmap
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Particle Distribution — {title_tag}", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(xs, ys, s=1.5, alpha=0.4, color="#1f77b4")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Scatter (n={len(xs):,})")
    ax.set_aspect("equal")

    ax = axes[0, 1]
    ax.hist(xs, bins=HIST_BINS, color="#1f77b4", alpha=0.75, edgecolor="white")
    
    # Expected uniform line
    expected = len(xs) / HIST_BINS
    ax.axhline(expected, color="red", linestyle="--", linewidth=1.5, label=f"Expected ({expected:,.0f})")
    ax.set_xlabel("x"); ax.set_ylabel("Count")
    ax.set_title("X-coordinate Distribution")
    ax.legend(fontsize=8)

    
    ax = axes[1, 0]
    ax.hist(ys, bins=HIST_BINS, color="#ff7f0e", alpha=0.75, edgecolor="white")
    expected = len(ys) / HIST_BINS
    ax.axhline(expected, color="red", linestyle="--", linewidth=1.5, label=f"Expected ({expected:,.0f})")
    ax.set_xlabel("y"); ax.set_ylabel("Count")
    ax.set_title("Y-coordinate Distribution")
    ax.legend(fontsize=8)

    
    ax = axes[1, 1]
    
    hNX = NX
    hNY = NY
    heatmap, xedges, yedges = np.histogram2d(
        xs, ys, bins=[hNX, hNY], range=[[0, 1], [0, 1]]
    )
    im = ax.imshow(
        heatmap.T, origin="lower", aspect="auto",
        extent=[0, 1, 0, 1],
        cmap="YlOrRd",
        norm=mcolors.LogNorm(vmin=max(1, heatmap.min()), vmax=heatmap.max())
    )
    fig.colorbar(im, ax=ax, label="Particle count (log scale)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Cell-wise Count ({hNX}×{hNY} bins)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")

print("=" * 60)
print("Particle Distribution Plots — Assignment 5 Q3")
print("=" * 60)

total = 0

for approach, approach_label in APPROACHES.items():
    for machine, machine_label in MACHINES.items():
        for config, grid in CONFIGS.items():

            fname = f"particle_sample_{approach}_{machine}_config{config}.out"
            fpath = os.path.join(BASE, fname)

            xs, ys = load_particles(fpath)

            title_tag = (
                f"{approach_label} | {machine_label} | "
                f"Grid {config} (NX={grid['NX']}, NY={grid['NY']})"
            )
            out_fname = (
                f"dist_{approach}_{machine}_config{config}.png"
            )
            out_path = os.path.join(OUTPUT_DIR, out_fname)

            print(f"\n[{approach_label} | {machine_label} | Config {config}]")
            plot_distribution(xs, ys, grid["NX"], grid["NY"], title_tag, out_path)
            total += 1

print(f"\nDone. {total} distribution plots generated.")
