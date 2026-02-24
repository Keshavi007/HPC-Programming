import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


cases = ['(a)', '(b)', '(c)', '(d)', '(e)']

lab_times = [0.084440, 0.470347, 0.347945, 1.896351, 1.874231]       # Lab PC times
cluster_times = [0.26, 1.15, 0.93, 4.72, 3.28]   # Cluster times


data = {
    "Case": cases,
    "Lab PC Time (s)": lab_times,
    "Cluster Time (s)": cluster_times
}

df = pd.DataFrame(data)

print("\nExecution Time Comparison Table:\n")
print(df.to_string(index=False))

df.to_csv("execution_times.csv", index=False)


x = np.arange(len(cases))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, lab_times, width, label='Lab PC', color='#1f77b4')
plt.bar(x + width/2, cluster_times, width, label='HPC Cluster', color='#d62728')

plt.xlabel('Problem Case')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison: Lab PC vs HPC Cluster')
plt.xticks(x, cases)
plt.legend()

plt.tight_layout()

# Save figure
plt.savefig("execution_time_comparison.png", dpi=300)

plt.show()

print("\nPlot saved as 'execution_time_comparison.png'")
print("Table saved as 'execution_times.csv'")