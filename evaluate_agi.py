#!/usr/bin/env python3
"""
evaluate_agi.py
===============
Phase 2 of the AGI Pipeline: Evaluation.

This script strictly targets the unseen Evaluation dataset (or test cases) 
using the frozen, hyper-optimized DSL library (`arc_library.json`) built
during the Wake-Sleep training phase.

It does NOT learn new primitives. It simply searches for the shortest MDL
program to solve the provided `train` grids, and applies it to the `test` grid.
"""
import argparse
import os
import sys

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from core.library import PrimitiveLibrary
from core.primitives import registry

def run_evaluation(data_dir: str, num_tasks: int, cfg: BenchmarkConfig) -> None:
    print(f"\n🧠 AGI Evaluation Phase")
    print(f"Loading tasks from: {data_dir}")
    
    tasks = load_tasks_from_dir(data_dir)[:num_tasks]
    if not tasks:
        print("No tasks found. Exiting.")
        return

    # Load the matured library learned from the Training set
    lib = PrimitiveLibrary("arc_library.json")
    print(f"Loaded {len(lib.learned_ops)} learned primitives from Sleep phase.")
    
    # Inject them into the active DSL registry
    lib.register_all(domain="arc")
    active_ops = registry.names(domain="arc")
    
    print(f"Total DSL Size: {len(active_ops)} operations")
    print(f"\n{'='*65}")
    print(f"  EVALUATING {len(tasks)} UNSEEN TASKS")
    print(f"{'='*65}")
    
    # Run a generic BeamSearch, but equipped with the Generative Priors (transition matrix)
    # to guide the discovery tree toward likely structures.
    report = evaluate_tasks(
        tasks=tasks, 
        op_subset=active_ops, 
        cfg=cfg, 
        label=f"Evaluation Mode",
        transition_matrix=lib.transition_matrix,
    )

    print("\n✅ Evaluation Complete!")
    print(f"Metrics saved to: {report.saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Forces evaluation to ONLY look at the eval subset
    parser.add_argument("--data", type=str, default="arc_data/data/evaluation")
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-workers", type=int, default=os.cpu_count() or 1)
    
    args = parser.parse_args()

    # Evaluation config (we can afford a larger beam since we aren't looping epochs)
    cfg = BenchmarkConfig(
        beam_size=30, 
        offspring=50, 
        generations=60, 
        task_workers=args.task_workers, 
        workers=args.workers, 
        baseline_only=True
    )
    
    run_evaluation(args.data, args.tasks, cfg)
