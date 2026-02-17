import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/probB.csv",
    skipinitialspace=True
)

df.columns = ["problem_size", 
            # "loop_type", for problemA
            "avg_algo_time", 
            "avg_mflops"]

for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# Plot 1: Time vs Problem Size
plt.figure()

plt.plot(
    np.log2(df["problem_size"]),
    np.log10(df["avg_algo_time"]),
    marker="o"
)

# for loop in sorted(df["loop_type"].unique()):
#     subset = df[df["loop_type"] == loop]

#     plt.plot(
#         np.log2(subset["problem_size"]),
#         np.log10(subset["avg_algo_time"]),
#         marker="o",
#         label=f"Loop {loop}"
#     )

plt.xlabel("Problem Size (log2 scale)")
plt.ylabel("Execution Time (log10 scale)")
plt.title("Problem B - Time vs Problem Size (Transpose)")
# plt.title("Problem C - Block Matrix Multiplication (Time)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("results/probB_time_vs_problem_size.png")
plt.close()

# Plot 2: MFLOPS vs Problem Size
plt.figure()

plt.plot(
    np.log2(df["problem_size"]),
    df["avg_mflops"],
    marker="o"
)
# for loop in sorted(df["loop_type"].unique()):
#     subset = df[df["loop_type"] == loop]

#     plt.plot(
#         np.log2(subset["problem_size"]),
#         subset["avg_mflops"],
#         marker="o",
#         label=f"Loop {loop}"
#     )

plt.xlabel("Problem Size (log2 scale)")
plt.ylabel("Performance (MFLOPS)")
plt.title("Problem B - MFLOPS vs Problem Size (Transpose)")
# plt.title("Problem C - Block Matrix Multiplication (MFLOPS)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("results/probB_mflops_vs_problem_size.png")
plt.close()

print("Plots saved successfully:")
print(" - probB_time_vs_problem_size.png")
print(" - probB_mflops_vs_problem_size.png")