#!/usr/bin/env python3
"""
scripts/sweep_hyperparams.py
============================
Automated grid search over `beam_size` and `generations`.
Identifies the "Sweet Spot" between Shallow/Wide exploration and Deep/Narrow refinement
to establish optimal defaults for the AGI command-line runner.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.primitives import registry

def main():
    print("\n🔍 AGI Sweet Spot Hyperparameter Sweep")
    print("Testing combinatorial matrix of [Beam Size] x [Generations]")
    
    # Use a challenging programmatic 5-task subset for fast but rigorous benchmarking on real data
    try:
        tasks = load_tasks_from_dir("arc_data/data/training")[:5]
    except Exception as e:
        print(f"Warning: {e}. Falling back to programmatic benchmarks.")
        tasks = build_benchmark()[:5]

    active_ops = registry.names(domain="arc")

    beam_sizes = [10, 25, 50]
    generation_depths = [20, 50, 100]

    best_acc = 0.0
    best_params = None

    print(f"\nEvaluating {len(tasks)} tasks using {len(active_ops)} ops.")
    print(f"Parallel Worker Threads: {os.cpu_count() or 1}\n")

    for b in beam_sizes:
        for g in generation_depths:
            cfg = BenchmarkConfig(
                beam_size=b,
                offspring=b * 2,  # Keep offspring proportional to beam width 
                generations=g,
                task_workers=os.cpu_count() or 1,
                workers=1,
                baseline_only=True,
                verbose=False # Keep grid search logs clean
            )

            print(f"▶ Testing: Beam={b:2d} | Gens={g:3d} | Offspring={b*2:3d} ... ", end="", flush=True)
            
            t0 = time.time()
            report = evaluate_tasks(
                tasks=tasks, 
                op_subset=active_ops, 
                cfg=cfg, 
                label=f"Sweep(B={b},G={g})"
            )
            elapsed = time.time() - t0

            solves = report.n_solved
            mean_acc = report.mean_test_acc

            print(f"Solved: {solves}/5 | Mean Acc: {mean_acc:.3f} | Time: {elapsed:.1f}s")
            
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (b, g)
                
    print(f"\n🏆 Optimal Sweet Spot Found:")
    print(f"Beam Size:   {best_params[0]}")
    print(f"Generations: {best_params[1]}")
    print(f"Top Accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    main()
