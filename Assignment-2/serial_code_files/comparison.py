import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dfA = pd.read_csv("results/probA.csv")

dfA.columns = ["problem_size", "loop_type", "avg_algo_time", "avg_mflops"]

dfA_best = dfA[dfA["loop_type"] == 6].copy()

dfA_best["method"] = "Conventional"

dfB = pd.read_csv("results/probB.csv")
dfB.columns = ["problem_size", "avg_algo_time", "avg_mflops"]
dfB["method"] = "Transpose"

dfC = pd.read_csv("results/probC.csv")
dfC.columns = ["problem_size", "avg_algo_time", "avg_mflops"]
dfC["method"] = "Blocked"

df_all = pd.concat([
    dfA_best[["problem_size", "avg_algo_time", "avg_mflops", "method"]],
    dfB,
    dfC
])

df_all = df_all.sort_values(["method", "problem_size"])

plt.figure()

for method in df_all["method"].unique():
    subset = df_all[df_all["method"] == method]
    plt.plot(
        np.log2(subset["problem_size"]),
        np.log10(subset["avg_algo_time"]),
        marker="o",
        linewidth=2,
        label=method
    )

plt.xlabel("Problem Size (log2 scale)")
plt.ylabel("Execution Time (log10 scale)")
plt.title("Matrix Multiplication Runtime Comparison")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("results/comparison_runtime.png")
plt.close()

plt.figure()
for method in df_all["method"].unique():
    subset = df_all[df_all["method"] == method]
    plt.plot(
        np.log2(subset["problem_size"]),
        subset["avg_mflops"],
        marker="o",
        linewidth=2,
        label=method
    )

plt.xlabel("Problem Size (log2 scale)")
plt.ylabel("Performance (MFLOPS)")
plt.title("Matrix Multiplication Performance Comparison")
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("results/comparison_mflops.png")
plt.close()

print("Comparison plots saved:")
print(" - results/comparison_runtime.png")
print(" - results/comparison_mflops.png")
