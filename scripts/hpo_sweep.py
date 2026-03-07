#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor

# Scientific Parameter Grid
PARAMETERS = {
    "beam_size": [10, 25, 50],
    "generations": [25, 100, 250] # Test high-depth saturation
}

TASKS_LIMIT = 20 # Representative Golden Slice

def run_experiment(beam, gens):
    timestamp = int(time.time())
    label = f"B{beam}_G{gens}_{timestamp}"
    cmd = [
        sys.executable, "agi.py", "eval",
        "--data", "arc_data/data/evaluation",
        "--tasks", str(TASKS_LIMIT),
        "--beam-size", str(beam),
        "--generations", str(gens),
        "--task-workers", "4", 
        "--workers", "1",
        "--model", "LATEST",
        "--shuffle",
        "--seed", "42"
    ]
    
    print(f"🚀 Launching Experiment: B={beam}, G={gens}...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    
    # Scrape the JSON output path from the logs
    # agi.py saves to reports/eval_<timestamp>.json
    # We can find the latest created JSON file in reports
    reports = [f for f in os.listdir("reports") if f.startswith("eval_") and f.endswith(".json")]
    reports.sort(key=lambda x: os.path.getmtime(os.path.join("reports", x)), reverse=True)
    
    if not reports:
        return {"error": "No report found"}
    
    with open(os.path.join("reports", reports[0]), "r") as f:
        data = json.load(f)
        
    return {
        "beam": beam,
        "gens": gens,
        "solved": data.get("n_solved", 0),
        "pct": data.get("pct_solved", 0),
        "mean_acc": data.get("mean_test_acc", 0),
        "evals": data.get("total_evals", 0),
        "time": data.get("total_elapsed_s", 0)
    }

def main():
    parser = argparse.ArgumentParser(description="Scientific HPO Sweep for ARC Solver")
    parser.add_argument("--workers", type=int, default=1, help="Number of experiments to run in parallel")
    args = parser.parse_args()
    
    combinations = list(itertools.product(PARAMETERS["beam_size"], PARAMETERS["generations"]))
    print(f"🔬 ARC Scientific HPO Sweep: {len(combinations)} configurations")
    print(f"🎯 Target Sample: First {TASKS_LIMIT} tasks")
    print("-" * 60)
    
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_experiment, b, g) for b, g in combinations]
        for f in futures:
            results.append(f.result())
            
    # Sort and print results
    results.sort(key=lambda x: x["solved"], reverse=True)
    
    print("\n📊 HPO SWEEP RESULTS (Sorted by Solve Count)")
    print("-" * 100)
    print("| Beam | Gens | Solved | Pct % | Mean Acc | Total Evals | Time (s) | Evals/Solve |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        eps = r["evals"] / max(r["solved"], 1)
        print(f"| {r['beam']:4} | {r['gens']:4} | {r['solved']:6} | {r['pct']:6.1f} | {r['mean_acc']:8.3f} | {r['evals']/1000:7.1f}k | {r['time']:8.1f} | {eps/1000:7.1f}k |")
    print("-" * 100)

if __name__ == "__main__":
    main()
