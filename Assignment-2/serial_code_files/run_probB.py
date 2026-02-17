import os
import csv
import subprocess
from collections import defaultdict

EXECUTABLE = "./probB"
NUM_RUNS = 5
OUTPUT_CSV = "results/probB.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results = defaultdict(lambda: {
    "algo_times": [],
    "mflops": []
})

print(f"Running Problem B benchmark {NUM_RUNS} times...\n")

for run_id in range(NUM_RUNS):
    print(f"\n=== Run {run_id + 1}/{NUM_RUNS} ===")

    proc = subprocess.Popen(
        [EXECUTABLE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    header_seen = False

    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break

        if not line:
            continue

        line = line.strip()
        print(line)

        if "ProblemSize" in line:
            header_seen = True
            continue

        if not header_seen:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        problem_size = int(parts[0])
        algo_time = float(parts[1])
        mflops = float(parts[2])

        results[problem_size]["algo_times"].append(algo_time)
        results[problem_size]["mflops"].append(mflops)

    err = proc.stderr.read()
    if err:
        print("STDERR:", err)

# Write averaged results
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "ProblemSize",
        "AvgAlgoTime",
        "AvgMFLOPS"
    ])

    for problem_size in sorted(results.keys()):
        entry = results[problem_size]

        avg_algo = sum(entry["algo_times"]) / len(entry["algo_times"])
        avg_mflops = sum(entry["mflops"]) / len(entry["mflops"])

        writer.writerow([
            problem_size,
            f"{avg_algo:.9f}",
            f"{avg_mflops:.2f}"
        ])

print(f"\nAveraged results written to: {OUTPUT_CSV}")
