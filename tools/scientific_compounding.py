#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse

# SCIENTIFIC EXPERIMENT: Does Micro-Batch Compounding improve solve rates?
# Hypotheses:
# 1. Compounding (N=10) grows the library faster, enabling solve of complex tasks late in the run.
# 2. Standard (N=50) learns too slowly to help with current-batch logic.

def run_agi_task(label, batch_size, beam=10, gens=25, seed=42):
    timestamp = int(time.time())
    model_name = f"compounding_test_{label}_{timestamp}.json"
    cmd = [
        sys.executable, "agi_compounding.py", "train",
        "--data", "arc_data/data/evaluation",
        "--tasks", "25",
        "--batch-size", str(batch_size),
        "--beam-size", "25",
        "--generations", "50",
        "--task-workers", "4",
        "--model", f"models/{model_name}",
        "--shuffle",
        "--seed", str(seed)
    ]
    
    print(f"\n[🚀] Launching Experiment: {label} (Batch={batch_size})")
    start_t = time.time()
    subprocess.run(cmd, check=True)
    end_t = time.time()
    
    # Simple summary metric scrape (since agi_compounding prints the global solved)
    # But for scientific precision, we'll just check the log files later
    print(f"[✅] Experiment {label} completed in {end_t-start_t:.1f}s.")

def main():
    # 1. Control (Standard - learn once after 50 tasks)
    run_agi_task("Control_Baseline", batch_size=50)
    
    # 2. Experiment (Micro-Batch Compounding - learn every 10 tasks)
    run_agi_task("Compounding_Agent", batch_size=10)
    
    print("\n🔬 EXPERIMENT COMPLETED.")
    print("Compare the solve rates in logs/ for the randomized set.")

if __name__ == "__main__":
    main()
