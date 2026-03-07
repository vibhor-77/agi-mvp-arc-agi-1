#!/usr/bin/env python3
"""
agi_compounding.py (EXPERIMENTAL)
================================
Implements Deterministic Micro-Batching for Continuous Compounding.
"""
import argparse
import datetime
import os
import sys
import time
from copy import deepcopy

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.library import PrimitiveLibrary
from core.primitives import registry

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_tasks(data_path, limit, task_ids, shuffle=False, seed=None):
    try:
        tasks = load_tasks_from_dir(data_path)
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(tasks)
        else:
            # Traditional curriculum: easier tasks first
            tasks.sort(key=lambda t: sum(len(inp)*len(inp[0]) for inp, _ in t.train_pairs))
            
        if task_ids:
            target_ids = [t.strip() for t in task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
        return tasks[:limit]
    except Exception as e:
        tasks = build_benchmark()
        return tasks[:limit]

def cmd_train(args):
    timestamp = get_timestamp()
    model_path = args.model or f"models/compounding_{timestamp}.json"
    report_path = args.report or f"reports/train_compounding_{timestamp}.md"
    
    tasks = setup_tasks(args.data, args.tasks, args.task_ids, shuffle=args.shuffle, seed=args.seed)
    lib = PrimitiveLibrary(model_path)
    lib.load()
    
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, offspring=args.offspring, generations=args.generations,
        task_workers=args.task_workers, workers=args.workers, timeout_s=args.timeout, max_evals=args.max_evals,
        baseline_only=True, seed=args.seed
    )

    batch_size = args.batch_size or len(tasks)
    print(f"\n{'='*65}\n  CONTINUOUS COMPOUNDING: {len(tasks)} tasks | N={batch_size}\n{'='*65}")
    
    for epoch in range(1, args.epochs + 1):
        total_tasks = len(tasks)
        all_results = []
        cumulative_solved = 0
        cumulative_near = 0
        cumulative_done = 0
        epoch_successes = {}
        
        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]
            chunk_label = f"Epoch {epoch} [{i//batch_size + 1}]"
            
            chunk_stats = {
                "offset": cumulative_done,
                "global_total": total_tasks,
                "global_solved": cumulative_solved,
                "global_near": cumulative_near
            }
            
            ops = registry.names(domain="arc")
            report = evaluate_tasks(
                chunk, ops, cfg, label=chunk_label, 
                transition_matrix=lib.transition_matrix, learned_ops=lib.learned_ops,
                epoch_str=f"{chunk_label} (Total: {len(lib.learned_ops)} ops)",
                report_callback=None, # Inline batching handle results manually
                global_stats=chunk_stats
            )
            
            # Extract from this chunk
            chunk_successes = {
                r.task_name: r.best_tree 
                for r in report.results 
                if (r.solved or r.test_acc >= 0.90) and r.best_tree
            }
            epoch_successes.update(chunk_successes)
            
            # Micro-Sleep / Compound
            if chunk_successes:
                lib.extract_from_tasks(chunk_successes, min_size=3, min_tasks=1)
                lib.register_all(domain="arc", overwrite=True)
                lib.save()
            
            cumulative_solved += report.n_solved
            cumulative_near += report.n_near
            cumulative_done += len(chunk)
            all_results.extend(report.results)
            
            print(f"  [Compounding] Chunk {i//batch_size+1} done. Library @ {len(lib.learned_ops)} ops. Global: {cumulative_solved} solved / {cumulative_done} done.")

        # Save Final Report
        from domains.arc.runner import BenchmarkReport
        final_report = BenchmarkReport(
            label="Compounding Training", 
            results=all_results,
            n_ops=len(registry.names(domain="arc"))
        )
        final_report.save(report_path)
        
        # Also save JSON for compounding
        json_path = report_path.replace(".md", ".json").replace(".html", ".json")
        import json
        with open(json_path, "w") as f_json:
            json.dump(final_report.as_dict(), f_json, indent=2)
            
        print(f"  [Report] Unified report saved to {report_path} (and .json)")

    print(f"\n✅ Training Complete. Model: {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Universal ARC-AGI Solver (Compounding)")
    parser.add_argument("command", choices=["train"])
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--tasks", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data", type=str, default="arc_data/data/training")
    parser.add_argument("--task-workers", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--max-evals", type=int, default=1000000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true", help="Randomize task order for scientific validity")
    parser.add_argument("--task-ids", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--report", type=str, default=None)

    args = parser.parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if args.command == "train":
        cmd_train(args)

if __name__ == "__main__":
    main()
