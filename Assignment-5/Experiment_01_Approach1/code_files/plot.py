import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

COLORS = {
    "def_lab": "#1f77b4",
    "def_cluster": "#aec7e8",
    "imm_lab": "#d62728",
    "imm_cluster": "#f4a582",
}

MARKERS = {
    "def_lab": "o",
    "def_cluster": "s",
    "imm_lab": "^",
    "imm_cluster": "D",
}

GRID_LABELS = {
    1: "Grid 1: NX=250, NY=100",
    2: "Grid 2: NX=500, NY=200",
    3: "Grid 3: NX=1000, NY=400",
}

PARTICLE_LABELS = ["$10^2$", "$10^4$", "$10^6$", "$10^8$", "$10^9$"]


def find_csv(approach, machine, config):
    fname = f"exp1_{approach}_grid{config}_{machine}.csv"
    path = os.path.join(BASE, fname)
    if os.path.exists(path):
        return path
    return None


def load_grid_csv(path, config):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "config" not in df.columns:
        df["config"] = config
    else:
        df["config"] = df["config"].fillna(config).astype(int)
    return df


def load_series(approach, machine):
    frames = []
    for config in [1, 2, 3]:
        path = find_csv(approach, machine, config)
        if path is None:
            print(f"[WARN] Missing file for {approach} {machine} grid {config}")
            continue
        print(f"[LOAD] {os.path.basename(path)}")
        frames.append(load_grid_csv(path, config))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


df_def_lab = load_series("deferred", "labpc")
df_def_cluster = load_series("deferred", "cluster")
df_imm_lab = load_series("immediate", "labpc")
df_imm_cluster = load_series("immediate", "cluster")


def plot_line(ax, df, config, ycol, key, label):
    if df is None:
        return
    sub = df[df["config"] == config].sort_values("particles")
    if sub.empty or ycol not in sub.columns:
        return
    ax.plot(
        sub["particles"], sub[ycol],
        color=COLORS[key],
        marker=MARKERS[key],
        linewidth=1.8,
        markersize=6,
        label=label,
    )


for config in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_line(ax, df_def_lab, config, "total_time", "def_lab", "Deferred - Lab PC")
    plot_line(ax, df_def_cluster, config, "total_time", "def_cluster", "Deferred - Cluster")
    plot_line(ax, df_imm_lab, config, "total_time", "imm_lab", "Immediate - Lab PC")
    plot_line(ax, df_imm_cluster, config, "total_time", "imm_cluster", "Immediate - Cluster")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Total Execution Time (s)")
    ax.set_title(f"Experiment 1 - Execution Time vs Particles\n{GRID_LABELS[config]}")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp1_total_time_config{config}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for config in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_line(ax, df_def_lab, config, "interp_time", "def_lab", "Deferred - Lab PC")
    plot_line(ax, df_def_cluster, config, "interp_time", "def_cluster", "Deferred - Cluster")
    plot_line(ax, df_imm_lab, config, "interp_time", "imm_lab", "Immediate - Lab PC")
    plot_line(ax, df_imm_cluster, config, "interp_time", "imm_cluster", "Immediate - Cluster")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Interpolation Time (s)")
    ax.set_title(f"Experiment 1 - Interpolation Time vs Particles\n{GRID_LABELS[config]}")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp1_interp_time_config{config}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for config in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_line(ax, df_def_lab, config, "mover_time", "def_lab", "Deferred - Lab PC")
    plot_line(ax, df_def_cluster, config, "mover_time", "def_cluster", "Deferred - Cluster")
    plot_line(ax, df_imm_lab, config, "mover_time", "imm_lab", "Immediate - Lab PC")
    plot_line(ax, df_imm_cluster, config, "mover_time", "imm_cluster", "Immediate - Cluster")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Mover Time (s)")
    ax.set_title(f"Experiment 1 - Mover Time vs Particles\n{GRID_LABELS[config]}")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp1_mover_time_config{config}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


for config in [1, 2, 3]:
    fig, ax = plt.subplots(figsize=(7, 5))

    for df, key, label in [
        (df_def_lab, "def_lab", "Deferred - Lab PC"),
        (df_def_cluster, "def_cluster", "Deferred - Cluster"),
        (df_imm_lab, "imm_lab", "Immediate - Lab PC"),
        (df_imm_cluster, "imm_cluster", "Immediate - Cluster"),
    ]:
        if df is None:
            continue
        sub = df[df["config"] == config].sort_values("PPC")
        if sub.empty or "PPC" not in sub.columns or "per_particle_time" not in sub.columns:
            continue
        ax.plot(
            sub["PPC"], sub["per_particle_time"],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=1.8,
            markersize=6,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particles per Cell (PPC)")
    ax.set_ylabel("Per-Particle Execution Time (s)")
    ax.set_title(f"Experiment 1 - Per-Particle Time vs PPC\n{GRID_LABELS[config]}")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"exp1_per_particle_ppc_config{config}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def stacked_bar_plot(df, machine_label, approach_label, filename):
    if df is None:
        print(f"Skipping stacked bar for {approach_label} {machine_label} (no data)")
        return

    configs = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, config in zip(axes, configs):
        sub = df[df["config"] == config].sort_values("particles")
        if sub.empty:
            continue

        x = np.arange(len(sub))
        width = 0.55

        ax.bar(
            x,
            sub["interp_time"].values,
            width,
            label="Interpolation",
            color="#4e79a7",
        )
        ax.bar(
            x,
            sub["mover_time"].values,
            width,
            bottom=sub["interp_time"].values,
            label="Mover",
            color="#f28e2b",
        )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(PARTICLE_LABELS[:len(sub)], rotation=30, ha="right", fontsize=9)
        ax.set_xlabel("Number of Particles")
        ax.set_ylabel("Time (s)")
        ax.set_title(GRID_LABELS[config], fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Experiment 1 - Interpolation vs Mover Time\n"
        f"{approach_label} Approach - {machine_label}",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")


stacked_bar_plot(
    df_def_lab, "Lab PC", "Deferred",
    os.path.join(OUTPUT_DIR, "exp1_breakdown_deferred_lab.png")
)
stacked_bar_plot(
    df_imm_lab, "Lab PC", "Immediate",
    os.path.join(OUTPUT_DIR, "exp1_breakdown_immediate_lab.png")
)
stacked_bar_plot(
    df_imm_cluster, "Cluster", "Immediate",
    os.path.join(OUTPUT_DIR, "exp1_breakdown_immediate_cluster.png")
)
stacked_bar_plot(
    df_def_cluster, "Cluster", "Deferred",
    os.path.join(OUTPUT_DIR, "exp1_breakdown_deferred_cluster.png")
)

print("\nAll Experiment 1 plots generated.")
