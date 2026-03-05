#!/usr/bin/env python3
"""
scripts/sweep_hyperparams.py
============================
Systematic hyperparameter optimization to find the "Sweet Spot" for AGI performance.
Measures Solve Rate / Time (Inference Efficiency) across generations, offspring, and beam width.
Parallelizes the sweep combinations themselves to saturate all CPU cores.
"""
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.primitives import registry

def run_single_combination(b, o, g, tasks, active_ops, combinations_total, idx):
    """Worker function for a single hyperparameter combination."""
    cfg = BenchmarkConfig(
        beam_size=b,
        offspring=o,
        generations=g,
        task_workers=1, # Sequential tasks within this combination because the pool is at the sweep level
        workers=1,
        baseline_only=True,
        verbose=False
    )

    t0 = time.time()
    report = evaluate_tasks(tasks=tasks, op_subset=active_ops, cfg=cfg, label=f"Sweep")
    elapsed = time.time() - t0

    # ROC Score = (Solve Rate * 100) / (Time in minutes)
    roc_score = (report.n_solved / float(len(tasks)) * 100) / (elapsed / 60.0 + 0.01)

    return {
        "params": {"beam": b, "offspring": o, "generations": g},
        "solves": report.n_solved,
        "mean_acc": report.mean_test_acc,
        "time": elapsed,
        "roc_score": roc_score
    }

def main():
    print("\n🔍 AGI 'Sweet Spot' Optimization Experiment (PARALLEL)")
    print("======================================================")
    
    # 5 representative tasks for fast but rigorous benchmarking
    try:
        tasks = load_tasks_from_dir("arc_data/data/training")[:5]
        task_names = [t.name for t in tasks]
    except Exception:
        tasks = build_benchmark()[:5]
        task_names = ["programmatic_" + str(i) for i in range(5)]

    active_ops = registry.names(domain="arc")

    # Sweep Space
    beam_sizes = [5, 10, 20]
    offspring_counts = [10, 20, 40]
    gen_depths = [25, 50, 100]

    combinations = []
    for b in beam_sizes:
        for o in offspring_counts:
            for g in gen_depths:
                combinations.append((b, o, g))

    print(f"🔄 Sweeping {len(combinations)} combinations in parallel...")
    print(f"📊 Dataset: {len(tasks)} ARC Tasks | {len(active_ops)} Primitives | {os.cpu_count()} Workers\n")

    print(f"{'Beam':<6} | {'Offs':<6} | {'Gen':<5} | {'Solve':<7} | {'MeanAcc':<8} | {'Time':<8} | {'ROC Score'}")
    print("-" * 75)

    results = []
    # Saturate ALL cores by running combinations in parallel
    # We use roughly CPU_COUNT - 1 to leave room for OS
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_combination, b, o, g, tasks, active_ops, len(combinations), i): (b, o, g)
            for i, (b, o, g) in enumerate(combinations)
        }
        
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            
            p = res['params']
            print(f"{p['beam']:<6} | {p['offspring']:<6} | {p['generations']:<5} | {res['solves']:<7} | {res['mean_acc']:<8.3f} | {res['time']:<8.1f} | {res['roc_score']:.2f}")

    # Find the ROC winner
    winner = max(results, key=lambda x: x["roc_score"])
    
    print("\n🏆 OPTIMIZATION FINDINGS")
    print("=======================")
    print(f"Winner (Max Return on Compute):")
    print(f"  > Beam Size:   {winner['params']['beam']}")
    print(f"  > Offspring:   {winner['params']['offspring']}")
    print(f"  > Generations: {winner['params']['generations']}")
    print(f"  > ROC Score:   {winner['roc_score']:.2f}")
    
    # Save findings for documentation
    with open("docs/hyperparameter_sweep_data.json", "w") as f:
        json.dump({"tasks": task_names, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
