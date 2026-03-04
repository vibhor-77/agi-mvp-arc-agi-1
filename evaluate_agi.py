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

def run_evaluation(data_dir: str, num_tasks: int, cfg: BenchmarkConfig, model_path: str) -> None:
    print(f"\n🧠 AGI Evaluation Phase")
    print(f"Loading tasks from: {data_dir}")
    
    tasks = load_tasks_from_dir(data_dir)[:num_tasks]
    if not tasks:
        print("No tasks found. Exiting.")
        return

    # Load the matured library learned from the Training set
    lib = PrimitiveLibrary(model_path)
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
        learned_ops=lib.learned_ops,
    )

    print("\n✅ Evaluation Complete!")
    print(f"Metrics saved to: {report.saved_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Forces evaluation to ONLY look at the eval subset
    parser.add_argument("--data", type=str, default="arc_data/data/evaluation")
    parser.add_argument("--tasks", type=int, default=400, help="Number of tasks to evaluate")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processing across tasks (default 1 to keep feedback clean)")
    parser.add_argument("--task-workers", type=int, default=8, help="Parallel processing within a single task's search")
    parser.add_argument("--beam-size", type=int, default=10, help="Size of the Beam Search queue")
    parser.add_argument("--offspring", type=int, default=20, help="Number of mutations per generation")
    parser.add_argument("--generations", type=int, default=100, help="Number of deep search iterations per task")
    parser.add_argument("--model", type=str, default="arc_library.json", help="Filepath to load the learned primitive dictionary from")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed for the search engine")
    
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  AGI EVALUATION PARAMETERS")
    print("="*65)
    for arg, value in vars(args).items():
        print(f"  {arg.replace('_', '-'):<15} : {value}")
    print("="*65)

    # Evaluation config 
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, 
        offspring=args.offspring, 
        generations=args.generations, 
        task_workers=args.task_workers, 
        workers=args.workers, 
        baseline_only=True,
        seed=args.seed
    )
    
    run_evaluation(args.data, args.tasks, cfg, args.model)
