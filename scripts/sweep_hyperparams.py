#!/usr/bin/env python3
"""
scripts/sweep_hyperparams.py
============================
Scientific Hyperparameter Optimization for ARC-AGI Search.
Goal: Objective identification of the Search 'Sweet Spot'.

Metrics:
  1. ROC (Math): (SolveRate * 1e6) / (Total Programs Evaluated)
     - Hardware-independent, noise-free metric.
  2. Mean Test Accuracy: Surrogate for search proximity on unsolved tasks.
  3. Absolute Solve Count: Prefers coverage at equal ROC.
"""
import os
import sys
import time
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.primitives import registry

# CONSTANTS FOR SCIENTIFIC RIGOR
N_TASKS = 20        # Increased for statistical significance
FIXED_SEED = 42     # For reproducibility
WORK_UNIT_SCALE = 1e6

def run_single_combination(b, o, g, tasks, active_ops):
    """
    Evaluates a single hyperparameter configuration.
    Returns objective metrics for ranking.
    """
    cfg = BenchmarkConfig(
        beam_size=b,
        offspring=o,
        generations=g,
        task_workers=1, 
        workers=1,
        baseline_only=True,
        verbose=False,
        seed=FIXED_SEED
    )

    t0 = time.time()
    report = evaluate_tasks(tasks=tasks, op_subset=active_ops, cfg=cfg, label=f"Sweep_{b}_{o}_{g}")
    elapsed_s = time.time() - t0
    
    # Mathematical Work: Total programs evaluated across all tasks
    total_evals = sum(r.n_evals for r in report.results)
    
    solve_rate = report.n_solved / float(len(tasks))
    
    # Math ROC: Solves per Million Programs Evaluated
    roc_math = (solve_rate * WORK_UNIT_SCALE) / (total_evals + 1)

    return {
        "params": {"beam": b, "offspring": o, "generations": g},
        "metrics": {
            "n_solved": report.n_solved,
            "solve_rate": solve_rate,
            "mean_acc": report.mean_test_acc,
            "time_s": elapsed_s,
            "n_evals_total": total_evals,
            "roc_math": roc_math
        }
    }

def main():
    print(f"\n🧪 SCIENTIFIC AGI OPTIMIZATION: SCALE {N_TASKS}")
    print(" (Metric: Solves per Million Programs Evaluated)")
    print("=" * 55)
    
    try:
        tasks = load_tasks_from_dir("arc_data/data/training")[:N_TASKS]
    except Exception:
        tasks = build_benchmark()[:N_TASKS]

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

    print(f"📊 Dataset: {len(tasks)} tasks | DSL: {len(active_ops)} ops")
    print(f"🔄 Sweep:   {len(combinations)} configs in parallel")
    print(f"⚙️ CPU:     {os.cpu_count()} cores (8 Performance / 2 Efficiency)")
    print("-" * 55)

    print(f"{'Beam':<5} | {'Offs':<5} | {'Gen':<5} | {'Solve':<5} | {'Acc':<7} | {'Evals (k)':<10} | {'ROC (M)'}")
    print("-" * 75)

    results = []
    # Use CPU_COUNT - 1 to maintain focus on Performance cores
    max_workers = max(1, (os.cpu_count() or 4) - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_combination, b, o, g, tasks, active_ops): (b, o, g)
            for b, o, g in combinations
        }
        
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                
                p = res['params']
                m = res['metrics']
                print(f"{p['beam']:<5} | {p['offspring']:<5} | {p['generations']:<5} | {m['n_solved']:<5} | {m['mean_acc']:<7.3f} | {m['n_evals_total']/1000:<10.1f} | {m['roc_math']:.2f}", flush=True)
            except Exception as e:
                print(f"Error in combination: {e}")

    # Tiered Selection
    sorted_res = sorted(results, 
                        key=lambda x: (x['metrics']['roc_math'], 
                                       x['metrics']['n_solved'], 
                                       x['metrics']['mean_acc']), 
                        reverse=True)
    
    winner = sorted_res[0]
    
    print("\n🏆 SCIENTIFIC WINNER (ROC_MATH)")
    print("-" * 35)
    print(f"Winner: Beam {winner['params']['beam']}, Offspring {winner['params']['offspring']}, Gen {winner['params']['generations']}")
    print(f"ROC_Math: {winner['metrics']['roc_math']:.2f} (Solves per 1M Evals)")
    print(f"Performance: {winner['metrics']['n_solved']} Solves | {winner['metrics']['n_evals_total']} Total Evals")
    
    with open("docs/scientific_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
