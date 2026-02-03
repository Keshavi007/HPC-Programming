import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

KERNEL = "add"
df = pd.read_csv(
    "results/avg_algo_times_add.csv",
    skipinitialspace=True
)

kernel_info = {
    "copy":  {"bytes": 16, "flops": 0},
    "scale": {"bytes": 16, "flops": 1},
    "add":   {"bytes": 24, "flops": 1},
    "triad": {"bytes": 32, "flops": 2},
}
BYTES_PER_ELEM = kernel_info[KERNEL]["bytes"]
FLOPS_PER_ELEM = kernel_info[KERNEL]["flops"]

# Rename columns 
df.columns = ["problem_size", "runs", "total_ops", "avg_e2e_time", "avg_algo_time"]
# df.columns = ["problem_size", "runs", "total_ops", "e2e_time", "algo_time"]

# ensure numeric types
for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Throughput (MFLOPs)

if FLOPS_PER_ELEM > 0:
    total_flops = FLOPS_PER_ELEM * df["problem_size"] * df["runs"]
    df["Throughput (MFLOPs)"] = (
        total_flops / df["avg_algo_time"] / (2 ** 20)
    )

# Bandwidth calculation (bits / second)
bytes_moved = BYTES_PER_ELEM  * df["problem_size"] * df["runs"]
df["Bandwidth (GB/s)"] = bytes_moved / df["avg_algo_time"] / 1e9

if FLOPS_PER_ELEM > 0:
    plt.figure()
    plt.plot(
        np.log2(df["problem_size"]),
        np.log10(df["Throughput (MFLOPs)"]),
        marker="o"
    )
    plt.xlabel("log2(Problem Size)")
    plt.ylabel("log10(Throughput) [MFLOPs]")
    plt.title("Compute Throughput vs Problem Size")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(f"results/throughput_{KERNEL}.png")
    plt.close()

plt.figure()
plt.plot(
    np.log2(df["problem_size"]),
    df["Bandwidth (GB/s)"],
    marker="o"
)
plt.xlabel("log2(Problem Size)")
plt.ylabel("Bandwidth (GB/s)")
plt.title("Memory Bandwidth vs Problem Size")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(f"results/bandwidth_{KERNEL}.png")
plt.close()

print("Plots saved successfully:")

