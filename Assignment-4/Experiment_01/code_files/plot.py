import numpy as np
import matplotlib.pyplot as plt

lab_data = np.loadtxt("exp1_lab.csv", delimiter=",")
cluster_data = np.loadtxt("exp1_cluster.csv", delimiter=",")

# Function to plot each configuration
def plot_config(config_id):

    lab_config = lab_data[lab_data[:,0] == config_id]
    cluster_config = cluster_data[cluster_data[:,0] == config_id]

    particles_lab = lab_config[:,1]
    time_lab = lab_config[:,2]

    particles_cluster = cluster_config[:,1]
    time_cluster = cluster_config[:,2]

    plt.figure()

    # Plot Lab PC data
    if len(particles_lab) > 0:
        plt.loglog(particles_lab, time_lab, 'o-', label="Lab PC")

    # Plot Cluster data (can include 1e9)
    if len(particles_cluster) > 0:
        plt.loglog(particles_cluster, time_cluster, 's-', label="HPC Cluster")

    plt.xlabel("Number of Particles")
    plt.ylabel("Total Interpolation Time (seconds)")
    plt.title(f"Experiment 01 - Configuration {config_id}")

    plt.legend()
    plt.grid(True, which="both")

    plt.savefig(f"Exp1_config{config_id}.png", dpi=300)
    plt.show()

plot_config(1)
plot_config(2)
plot_config(3)