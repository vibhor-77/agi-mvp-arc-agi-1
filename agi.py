#!/usr/bin/env python3
"""
agi.py
======
The Unified Entry Point for the Universal ARC-AGI Solver.

Usage:
  # 1. Start Training (Wake-Sleep)
  python agi.py train --epochs 5 --task-workers 8

  # 2. Run Evaluation on unseen tasks
  python agi.py eval --model models/latest.json --data arc_data/data/evaluation

  # 3. Comprehensive Pipeline (Train -> Eval)
  python agi.py pipeline --train-tasks 100 --eval-tasks 100
"""
import argparse
import datetime
import os
import sys
import time

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from domains.arc.benchmark import build_benchmark
from core.library import PrimitiveLibrary
from core.primitives import registry

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_tasks(data_path, limit, task_ids):
    try:
        tasks = load_tasks_from_dir(data_path)
        # Curriculum: easier tasks first (smaller grids)
        tasks.sort(key=lambda t: sum(len(inp)*len(inp[0]) for inp, _ in t.train_pairs))
        if task_ids:
            target_ids = [t.strip() for t in task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
        return tasks[:limit]
    except Exception as e:
        print(f"[!] Warning: Could not load from {data_path} ({e}). Falling back to programmatic benchmark.")
        tasks = build_benchmark()
        if task_ids:
            target_ids = [t.strip() for t in task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
        return tasks[:limit]

def cmd_train(args):
    timestamp = get_timestamp()
    model_path = args.model or f"models/arc_model_{timestamp}.json"
    report_path = args.report or f"reports/train_{timestamp}.md"
    
    tasks = setup_tasks(args.data, args.tasks, args.task_ids)
    lib = PrimitiveLibrary(model_path)
    lib.load()
    
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, offspring=args.offspring, generations=args.generations,
        task_workers=args.task_workers, workers=args.workers, timeout_s=args.timeout, max_evals=args.max_evals,
        baseline_only=True, seed=args.seed
    )

    print(f"\n{'='*65}\n  WAKE-SLEEP TRAINING: {len(tasks)} tasks | {args.epochs} epochs\n{'='*65}")
    
    for epoch in range(1, args.epochs + 1):
        ops = registry.names(domain="arc")
        report = evaluate_tasks(
            tasks, ops, cfg, label=f"Epoch {epoch}", 
            transition_matrix=lib.transition_matrix, learned_ops=lib.learned_ops,
            epoch_str=f"Epoch {epoch}/{args.epochs}",
            report_callback=lambda r: r.save(report_path)
        )
        
        # Sleep Phase: Extract from both solved AND near-solved (>=90% accuracy)
        successes = {
            r.task_name: r.best_tree 
            for r in report.results 
            if (r.solved or r.test_acc >= 0.90) and r.best_tree
        }
        # min_tasks=1: extract any composite sub-tree (not just ones shared across tasks).
        # At small scale every solution tends to be unique, so min_tasks=2 yields 0 abstractions.
        # min_size=3: require at least 3 AST nodes so trivial single-op wrappers aren't promoted.
        lib.extract_from_tasks(successes, min_size=3, min_tasks=1)
        lib.register_all(domain="arc")
        lib.save()
        
        usage = sum(1 for r in report.results if r.solved and "lib_op_" in str(r.found_expr))
        print(f"  [Analysis] Epoch {epoch}: {len(successes)} solved | {len(lib.learned_ops)} abstractions | {usage} used learned ops.")

    print(f"\n✅ Training Complete. Model: {model_path}")
    return model_path

def cmd_eval(args):
    timestamp = get_timestamp()
    report_path = args.report or f"reports/eval_{timestamp}.md"
    
    model_path = args.model
    if model_path == "LATEST":
        if not os.path.exists("models"): return
        model_files = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".json")]
        model_path = max(model_files, key=os.path.getmtime) if model_files else None
    
    if not model_path or not os.path.exists(model_path):
        print(f"[!] Error: Model path {model_path} not found.")
        return

    tasks = setup_tasks(args.data, args.tasks, args.task_ids)
    lib = PrimitiveLibrary(model_path)
    lib.load()
    lib.register_all(domain="arc")
    
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, offspring=args.offspring, generations=args.generations,
        task_workers=args.task_workers, workers=args.workers, timeout_s=args.timeout, max_evals=args.max_evals,
        baseline_only=True, seed=args.seed
    )

    print(f"\n{'='*65}\n  AGI EVALUATION: {len(tasks)} tasks | Model: {os.path.basename(model_path)}\n{'='*65}")
    
    report = evaluate_tasks(
        tasks, registry.names(domain="arc"), cfg, label="Evaluation",
        transition_matrix=lib.transition_matrix, learned_ops=lib.learned_ops,
        report_callback=lambda r: r.save(report_path)
    )
    print(f"\n✅ Evaluation Complete. Report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Universal ARC-AGI Solver")
    subparsers = parser.add_subparsers(dest="command")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--tasks", type=int, default=400)
    shared.add_argument("--task-workers", type=int, default=8)
    shared.add_argument("--workers", type=int, default=1)
    shared.add_argument("--beam-size", type=int, default=10)
    shared.add_argument("--offspring", type=int, default=20)
    shared.add_argument("--generations", type=int, default=25)
    shared.add_argument("--max-evals", type=int, default=1000000)
    shared.add_argument("--timeout", type=float, default=60.0)
    shared.add_argument("--seed", type=int, default=None)
    shared.add_argument("--task-ids", type=str, default=None)

    p_train = subparsers.add_parser("train", parents=[shared])
    p_train.add_argument("--data", type=str, default="arc_data/data/training")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--model", type=str, default=None)
    p_train.add_argument("--report", type=str, default=None)

    p_eval = subparsers.add_parser("eval", parents=[shared])
    p_eval.add_argument("--data", type=str, default="arc_data/data/evaluation")
    p_eval.add_argument("--model", type=str, default="LATEST")
    p_eval.add_argument("--report", type=str, default=None)

    p_pipe = subparsers.add_parser("pipeline", parents=[shared])
    p_pipe.add_argument("--train-data", type=str, default="arc_data/data/training")
    p_pipe.add_argument("--eval-data", type=str, default="arc_data/data/evaluation")
    p_pipe.add_argument("--epochs", type=int, default=5)
    p_pipe.add_argument("--train-tasks", type=int, default=400)
    p_pipe.add_argument("--eval-tasks", type=int, default=400)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "pipeline":
        train_args = argparse.Namespace(**vars(args))
        train_args.data = args.train_data
        train_args.tasks = args.train_tasks
        train_args.model = None
        train_args.report = None
        m_path = cmd_train(train_args)
        
        eval_args = argparse.Namespace(**vars(args))
        eval_args.data = args.eval_data
        eval_args.tasks = args.eval_tasks
        eval_args.model = m_path
        eval_args.report = None
        cmd_eval(eval_args)

if __name__ == "__main__":
    main()
