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

def run_tier_3(args):
    """Tier 3: Statistical Scaling (N=50)"""
    print(f"\n[🔬 TIER 3] Testing Statistical Scaling (N=50)")
    print("Hypothesis: Higher Generations (G=150) will convert 15-20% of evaluation tasks.")
    
    timestamp = int(time.time())
    cmd = [
        sys.executable, "agi.py", "eval",
        "--tasks", "50",
        "--data", "arc_data/data/evaluation",
        "--shuffle", "--seed", "42",
        "--beam-size", "50",
        "--generations", "150",
        "--task-workers", "4",
        "--report", f"reports/tier3_scaling_{timestamp}.md"
    ]
    subprocess.run(cmd, check=True)

def run_tier_4(args):
    """Tier 4: Global Benchmark (N=100)"""
    print(f"\n[🔬 TIER 4] Testing Global Benchmark with Compounding (N=100)")
    print("Hypothesis: Continuous compounding increases solve rates at scale on randomized evaluation tasks.")
    
    timestamp = int(time.time())
    cmd = [
        sys.executable, "agi.py", "train",
        "--tasks", "100",
        "--batch-size", "10",
        "--epochs", "3",
        "--data", "arc_data/data/evaluation",
        "--shuffle", "--seed", "42",
        "--beam-size", "50",
        "--generations", "150",
        "--task-workers", "4",
        "--report", f"reports/tier4_compounding_{timestamp}.md"
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Incremental Scientific Validation")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], required=True)
    args = parser.parse_args()
    
    if args.tier == 1:
        run_tier_1(args)
    elif args.tier == 2:
        run_tier_2(args)
    elif args.tier == 3:
        run_tier_3(args)
    elif args.tier == 4:
        run_tier_4(args)
    else:
        print("Tier not implemented yet.")

if __name__ == "__main__":
    main()
