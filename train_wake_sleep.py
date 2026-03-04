#!/usr/bin/env python3
"""
train_wake_sleep.py
===================
End-to-end evolutionary AGI training loop leveraging Abstraction and Composability.

1. WAKE Phase: Attempt batches of ARC tasks.
2. SLEEP Phase: Extract common functional sub-trees from solved tasks and compress them into new reusable primitives (`lib_op_X`).
3. Repeat: The expanded DSL tackles harder tasks in subsequent epochs.
"""
import argparse
import os
import sys
import datetime

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.domain import ARCTask
from core.library import PrimitiveLibrary
from core.primitives import registry

def run_wake_sleep(tasks: list[ARCTask], epochs: int, cfg: BenchmarkConfig, model_path: str, report_path: str) -> None:
    lib = PrimitiveLibrary(model_path)
    
    print(f"\n🚀 Starting Wake-Sleep Training over {len(tasks)} tasks for {epochs} epochs")
    
    full_report = f"# Wake-Sleep Training Log\n\n**Total Epochs**: {epochs} | **Tasks**: {len(tasks)}\n\n---\n\n"
    
    for epoch in range(1, epochs + 1):
        # The op subset is whatever is currently registered in 'arc' domain
        # This grows every epoch after the sleep phase.
        active_ops = registry.names(domain="arc")
        
        print(f"\n{'='*65}")
        print(f"  EPOCH {epoch}/{epochs}  —  WAKE PHASE  [{len(active_ops)} ops available]")
        print(f"{'='*65}")
        
        # Wake: Attempt to solve all tasks
        report = evaluate_tasks(
            tasks=tasks, 
            op_subset=active_ops, 
            cfg=cfg, 
            label=f"Epoch {epoch} ({len(active_ops)} ops)",
            transition_matrix=lib.transition_matrix,
            learned_ops=lib.learned_ops,
        )
        
        # Sleep: Collect successful trees
        successful_trees = {}
        for r in report.results:
            if r.solved and r.best_tree is not None:
                successful_trees[r.task_name] = r.best_tree
                
        print(f"\n  [Sleep Phase] Extracting abstractions from {len(successful_trees)} solved tasks...")
        
        # For testing purposes, let's artificially set min_tasks=1 if we only have a few tasks solved,
        # otherwise we might not learn anything in small local tests.
        min_tasks = 2 if len(successful_trees) > 2 else 1
        lib.extract_from_tasks(successful_trees, min_size=3, min_tasks=min_tasks)
        
        print(f"  [Sleep Phase] Current Library Size: {len(lib.learned_ops)} abstractions")
        
        # Inject learned abstractions back into active registry for next Epoch
        lib.register_all(domain="arc")
        lib.save()

        # Append Epoch report
        full_report += report.generate_markdown_report() + "\n\n---\n\n"
        
        # Save cumulative report after each epoch
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)

    print("\n🎉 Wake-Sleep Training Complete!")
    print(f"Total Learned Primitives: {len(lib.learned_ops)}")
    print(f"Full introspection report saved to: {report_path}")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="arc_data/data/training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of Wake/Sleep cycles")
    parser.add_argument("--tasks", type=int, default=400, help="Number of tasks to train on")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processing across tasks (default 1 to keep feedback loop clean)")
    parser.add_argument("--task-workers", type=int, default=8, help="Parallel processing within a single task's search")
    parser.add_argument("--beam-size", type=int, default=10, help="Size of the Beam Search queue")
    parser.add_argument("--offspring", type=int, default=20, help="Number of mutations per generation")
    parser.add_argument("--generations", type=int, default=100, help="Number of deep search iterations per task")
    parser.add_argument("--model", type=str, default=f"models/arc-agi-1_{timestamp}.json", help="Filepath to save the learned primitive dictionary")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed for the search engine")
    parser.add_argument("--report", type=str, default=f"reports/train_{timestamp}.md", help="Markdown file to accumulate Introspection diagnostics")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    print("\n" + "="*65)
    print("  WAKE-SLEEP EXECUTOR PARAMETERS")
    print("="*65)
    for arg, value in vars(args).items():
        print(f"  {arg.replace('_', '-'):<15} : {value}")
    print("="*65)

    try:
        tasks = load_tasks_from_dir(args.data)[:args.tasks]
    except Exception as e:
        print(e)
        tasks = [] # fallback to built in programmatic tasks
        from domains.arc.benchmark import build_benchmark
        tasks = build_benchmark()[:args.tasks]

    cfg = BenchmarkConfig(
        beam_size=args.beam_size, 
        offspring=args.offspring, 
        generations=args.generations, 
        task_workers=args.task_workers, 
        workers=args.workers, 
        baseline_only=True,
        seed=args.seed
    )
    
    run_wake_sleep(tasks, args.epochs, cfg, args.model, args.report)
