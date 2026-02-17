import os
import csv
import subprocess
from collections import defaultdict

EXECUTABLE = "./probA"
NUM_RUNS = 5
OUTPUT_CSV = "results/probA.csv"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

results = defaultdict(lambda: {
    "algo_times": [],
    "mflops": []
})

print(f"Running benchmark {NUM_RUNS} times...\n")

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

        # Skip header line
        if "ProblemSize" in line:
            header_seen = True
            continue

        if not header_seen:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue

        problem_size = int(parts[0])
        loop_type = int(parts[1])
        algo_time = float(parts[2])
        mflops = float(parts[3])

        key = (problem_size, loop_type)
        results[key]["algo_times"].append(algo_time)
        results[key]["mflops"].append(mflops)

    # Print stderr if any
    err = proc.stderr.read()
    if err:
        print("STDERR:", err)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "ProblemSize",
        "LoopType",
        "AvgAlgoTime",
        "AvgMFLOPS"
    ])

    for (problem_size, loop_type) in sorted(results.keys()):
        entry = results[(problem_size, loop_type)]

        avg_algo = sum(entry["algo_times"]) / len(entry["algo_times"])
        avg_mflops = sum(entry["mflops"]) / len(entry["mflops"])

        writer.writerow([
            problem_size,
            loop_type,
            f"{avg_algo:.9f}",
            f"{avg_mflops:.2f}"
        ])

print(f"\nAveraged results written to: {OUTPUT_CSV}")
