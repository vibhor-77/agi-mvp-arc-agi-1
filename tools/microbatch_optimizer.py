import os
import sys
import time
import argparse
import numpy as np
from agi import setup_tasks, PrimitiveLibrary
from domains.arc.runner import BenchmarkConfig, evaluate_tasks
from core.primitives import registry

def run_experiment(batch_size, task_limit=100):
    print(f"\n>>> TESTING BATCH SIZE N={batch_size} (Tasks: {task_limit})")
    timestamp = int(time.time())
    exp_name = f"exp_N{batch_size}_{timestamp}"
    model_path = f"/tmp/{exp_name}.json"
    
    tasks = setup_tasks("arc_data/data/training", task_limit, None)
    lib = PrimitiveLibrary(model_path)
    
    # We use a reduced budget to speed up the scientific comparison
    cfg = BenchmarkConfig(
        beam_size=8, generations=20, max_evals=50000, 
        task_workers=8, timeout_s=30.0, baseline_only=True,
        fail_on_timeout=False
    )
    
    t0 = time.time()
    total_solved = 0
    total_abstractions = 0
    
    # Chunk tasks into batches
    for i in range(0, len(tasks), batch_size):
        chunk = tasks[i : i + batch_size]
        print(f"  [Chunk {i//batch_size + 1}] Processing {len(chunk)} tasks...")
        
        ops = registry.names(domain="arc")
        report = evaluate_tasks(
            chunk, ops, cfg, label=f"N={batch_size}", 
            transition_matrix=lib.transition_matrix, learned_ops=lib.learned_ops
        )
        
        # Soft-success extraction
        successes = {
            r.task_name: r.best_tree 
            for r in report.results 
            if (r.solved or r.test_acc >= 0.90) and r.best_tree
        }
        
        lib.extract_from_tasks(successes, min_size=3, min_tasks=1)
        lib.register_all(domain="arc", overwrite=True)
        lib.save()
        
        solved = sum(1 for r in report.results if r.solved)
        total_solved += solved
        total_abstractions = len(lib.learned_ops)
        print(f"  [Progress] Solved: {total_solved} | Abstractions: {total_abstractions}")

    elapsed = time.time() - t0
    return {
        "N": batch_size,
        "solved": total_solved,
        "abstractions": total_abstractions,
        "time_s": elapsed,
        "efficiency": total_solved / (elapsed / 3600)  # Solves per hour
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="10,25,50,100")
    parser.add_argument("--tasks", type=int, default=100)
    args = parser.parse_args()
    
    sizes = [int(s) for s in args.sizes.split(",")]
    results = []
    
    for N in sizes:
        res = run_experiment(N, task_limit=args.tasks)
        results.append(res)
    
    print("\n" + "="*40)
    print("      MICRO-BATCH OPTIMIZATION RESULTS")
    print("="*40)
    print(f"{'N':>4} | {'Solved':>7} | {'Abs':>5} | {'Time (s)':>8} | {'Solves/Hr':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['N']:4d} | {r['solved']:7d} | {r['abstractions']:5d} | {r['time_s']:8.1f} | {r['efficiency']:10.2f}")

if __name__ == "__main__":
    main()
