import numpy as np
import matplotlib.pyplot as plt

# Load data
lab = np.loadtxt("exp2_lab.csv", delimiter=",")
cluster = np.loadtxt("exp2_cluster.csv", delimiter=",")

problem_index = lab[:,0]
lab_time = lab[:,1]
cluster_time = cluster[:,1]

x = np.arange(len(problem_index))
width = 0.35
# Problem Index vs Interpolation Time
plt.figure()
plt.plot(lab[:,0], lab[:,1], 'o-', label="Lab PC")
plt.plot(cluster[:,0], cluster[:,1], 's-', label="HPC Cluster")

plt.xlabel("Problem Index")
plt.ylabel("Total Interpolation Time (seconds)")
plt.title("Experiment 02: Consistency Across Configurations")
plt.legend()
plt.grid(True)

plt.savefig("exp2_total_interpolation_time.png", dpi=300)
plt.show()

#Bar plot
plt.figure()

plt.bar(x - width/2, lab_time, width, label="Lab PC")
plt.bar(x + width/2, cluster_time, width, label="HPC Cluster")

plt.xlabel("Problem Index")
plt.ylabel("Total Interpolation Time (seconds)")
plt.title("Experiment 02: Execution Time Comparison")
plt.xticks(x, ["1", "2", "3"])
plt.legend()
plt.grid(axis='y')

plt.savefig("exp2_barplot.png", dpi=300)
plt.show()


