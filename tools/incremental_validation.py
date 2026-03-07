#!/usr/bin/env python3
import sys
import os
import subprocess
import time
import json
import argparse

def run_tier_1(args):
    """Tier 1: Unit Solvability (N=1)"""
    task_id = "00576224" # Tile/Repeat logic
    print(f"\n[🔬 TIER 1] Testing Unit Solvability for Task: {task_id}")
    print("Hypothesis: Deep Search (G=200) can solve complex evaluation tasks.")
    
    cmd = [
        sys.executable, "agi.py", "eval",
        "--task-ids", task_id,
        "--data", "arc_data/data/evaluation",
        "--beam-size", "50",
        "--generations", "200",
        "--task-workers", "1",
        "--model", "LATEST"
    ]
    
    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()
    print(f"[✅ TIER 1] Completed in {end-start:.1f}s.")

def run_tier_2(args):
    """Tier 2: The Compounding Margin (N=10)"""
    print(f"\n[🔬 TIER 2] Testing Compounding Margin (N=10)")
    print("Hypothesis: Continuous compounding improves solve rates on randomized sets.")
    
    timestamp = int(time.time())
    
    # 1. Control (No mid-run learning)
    print("\n[🚀] Launching CONTROL (Static Library)...")
    cmd_control = [
        sys.executable, "agi_compounding.py", "train",
        "--tasks", "10",
        "--batch-size", "10",
        "--data", "arc_data/data/evaluation",
        "--shuffle", "--seed", "42",
        "--beam-size", "25",
        "--generations", "50",
        "--model", f"models/tier2_control_{timestamp}.json"
    ]
    subprocess.run(cmd_control, check=True)
    
    # 2. Experiment (Micro-Batch Compounding)
    print("\n[🚀] Launching EXPERIMENT (Compounding Library)...")
    cmd_exp = [
        sys.executable, "agi_compounding.py", "train",
        "--tasks", "10",
        "--batch-size", "2", # Learn every 2 tasks to maximize compounding
        "--data", "arc_data/data/evaluation",
        "--shuffle", "--seed", "42",
        "--beam-size", "25",
        "--generations", "50",
        "--model", f"models/tier2_exp_{timestamp}.json"
    ]
    subprocess.run(cmd_exp, check=True)

def main():
    parser = argparse.ArgumentParser(description="Incremental Scientific Validation")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    args = parser.parse_args()
    
    if args.tier == 1:
        run_tier_1(args)
    elif args.tier == 2:
        run_tier_2(args)
    else:
        print("Tier not implemented yet.")

if __name__ == "__main__":
    main()
