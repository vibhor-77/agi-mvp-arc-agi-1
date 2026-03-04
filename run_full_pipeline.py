#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
Executes the full End-to-End AGI pipeline sequentially: 
1. Wake-Sleep Training (discovering new generic primitives)
2. Evaluation (testing the hardened DSL on unseen semantic grids)
"""
import argparse
import subprocess
import sys
import datetime
import os

def main():
    parser = argparse.ArgumentParser(description="Run the full Training -> Evaluation AGI Pipeline")
    parser.add_argument("--train-data", type=str, default="arc_data/data/training")
    parser.add_argument("--eval-data", type=str, default="arc_data/data/evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of Wake/Sleep cycles")
    parser.add_argument("--train-tasks", type=int, default=400, help="Number of tasks to train on")
    parser.add_argument("--eval-tasks", type=int, default=400, help="Number of tasks to evaluate on")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processing across tasks")
    parser.add_argument("--task-workers", type=int, default=8, help="Parallel processing within a single task's search")
    parser.add_argument("--beam-size", type=int, default=10, help="Size of the Beam Search queue")
    parser.add_argument("--offspring", type=int, default=20, help="Number of mutations per generation")
    parser.add_argument("--generations", type=int, default=100, help="Number of deep search iterations per task")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed")
    parser.add_argument("--train-task-ids", type=str, default=None, help="Comma-separated list of train task IDs")
    parser.add_argument("--eval-task-ids", type=str, default=None, help="Comma-separated list of eval task IDs")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"models/arc-agi-1_{timestamp}.json"
    train_report = f"reports/train_{timestamp}.md"
    eval_report = f"reports/eval_{timestamp}.md"

    print("\n" + "="*65)
    print("  🚀 PHASE 1: WAKE-SLEEP TRAINING")
    print("="*65)

    train_cmd = [
        sys.executable, "train_wake_sleep.py",
        "--data", args.train_data,
        "--epochs", str(args.epochs),
        "--tasks", str(args.train_tasks),
        "--workers", str(args.workers),
        "--task-workers", str(args.task_workers),
        "--beam-size", str(args.beam_size),
        "--offspring", str(args.offspring),
        "--generations", str(args.generations),
        "--model", model_path,
        "--report", train_report,
    ]
    if args.seed is not None:
        train_cmd.extend(["--seed", str(args.seed)])
    if args.train_task_ids is not None:
        train_cmd.extend(["--task-ids", args.train_task_ids])

    try:
        # Popen inherently pipes to stdout so user sees dynamic print logs
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Training Pipeline failed with exit code {e.returncode}. Aborting.")
        sys.exit(e.returncode)

    print("\n" + "="*65)
    print("  🧠 PHASE 2: EVALUATION")
    print("="*65)

    eval_cmd = [
        sys.executable, "evaluate_agi.py",
        "--data", args.eval_data,
        "--tasks", str(args.eval_tasks),
        "--workers", str(args.workers),
        "--task-workers", str(args.task_workers),
        "--beam-size", str(args.beam_size),
        "--offspring", str(args.offspring),
        "--generations", str(args.generations),
        "--model", model_path,
        "--report", eval_report,
    ]
    if args.seed is not None:
        eval_cmd.extend(["--seed", str(args.seed)])
    if args.eval_task_ids is not None:
        eval_cmd.extend(["--task-ids", args.eval_task_ids])

    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Evaluation Pipeline failed with exit code {e.returncode}. Aborting.")
        sys.exit(e.returncode)

    print("\n" + "="*65)
    print("  🏆 END-TO-END PIPELINE COMPLETE SUMMARY")
    print("="*65)
    print(f"  Model Weights        : {model_path}")
    print(f"  Train Report (.md)   : {train_report}")
    print(f"  Train Report (.html) : {train_report.replace('.md', '.html')}")
    print(f"  Eval  Report (.md)   : {eval_report}")
    print(f"  Eval  Report (.html) : {eval_report.replace('.md', '.html')}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
