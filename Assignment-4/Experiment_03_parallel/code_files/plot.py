import numpy as np
import matplotlib.pyplot as plt

serial_lab = np.loadtxt("exp3_serial_lab.csv", delimiter=",", skiprows=1)
parallel_lab = np.loadtxt("exp3_parallel_lab.csv", delimiter=",", skiprows=1)

serial_cluster = np.loadtxt("exp3_serial_cluster.csv", delimiter=",", skiprows=1)
parallel_cluster = np.loadtxt("exp3_parallel_cluster.csv", delimiter=",", skiprows=1)

# PLOT 1: Iteration vs Times

def plot_iteration(data, title, filename):

    iteration = data[:,0]
    interp = data[:,1]
    mover = data[:,2]
    total = data[:,3]

    plt.figure()
    plt.plot(iteration, interp, 'o-', label="Interpolation Time")
    plt.plot(iteration, mover, 's-', label="Mover Time")
    plt.plot(iteration, total, '^-', label="Total Time")

    plt.xlabel("Iteration")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, dpi=300)
    plt.show()


# Lab PC (Serial)
plot_iteration(serial_lab,
               "Experiment 03 - Lab PC (Serial)",
               "lab_serial_iteration.png")

# HPC Cluster (Serial)
plot_iteration(serial_cluster,
               "Experiment 03 - HPC Cluster (Serial)",
               "cluster_serial_iteration.png")

# PLOT 2: Serial vs Parallel
plt.figure()

plt.plot(serial_lab[:,0], serial_lab[:,2], 'o-', label="Lab Serial Mover")
plt.plot(parallel_lab[:,0], parallel_lab[:,2], 's-', label="Lab Parallel Mover")

plt.plot(serial_cluster[:,0], serial_cluster[:,2], 'o--', label="Cluster Serial Mover")
plt.plot(parallel_cluster[:,0], parallel_cluster[:,2], 's--', label="Cluster Parallel Mover")

plt.xlabel("Iteration")
plt.ylabel("Mover Time (seconds)")
plt.title("Mover Serial vs Parallel Comparison")
plt.legend()
plt.grid(True)

plt.savefig("mover_comparison.png", dpi=300)
plt.show()

# PLOT 3: Speedup

# Compute speedup per iteration
speedup_lab = serial_lab[:,2] / parallel_lab[:,2]
speedup_cluster = serial_cluster[:,2] / parallel_cluster[:,2]

plt.figure()

plt.plot(serial_lab[:,0], speedup_lab, 'o-', label="Lab PC Speedup")
plt.plot(serial_cluster[:,0], speedup_cluster, 's-', label="Cluster Speedup")

plt.xlabel("Iteration")
plt.ylabel("Speedup (Tserial / Tparallel)")
plt.title("Parallel Mover Speedup")
plt.legend()
plt.grid(True)

plt.savefig("speedup_plot.png", dpi=300)
plt.show()

# PRINT AVERAGE SPEEDUP 

print("Average Lab Speedup:", np.mean(speedup_lab))
print("Average Cluster Speedup:", np.mean(speedup_cluster))