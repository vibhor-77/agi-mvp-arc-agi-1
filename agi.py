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
import json
import os
import sys
import time
from dataclasses import asdict

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

def setup_output_logging(timestamp):
    log_file = f"logs/run_{timestamp}.log"
    os.makedirs("logs", exist_ok=True)
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    f = open(log_file, "a", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)
    return log_file

def _log_run_header(
    title: str,
    args: argparse.Namespace,
    cfg: BenchmarkConfig,
    progress_log_path: str,
    report_path: str,
    model_path: str,
    log_file: str,
) -> None:
    cmdline = " ".join(sys.argv)
    print(f"\n{'='*65}\n  {title}\n{'='*65}")
    print(f"  [Cmd] python {' '.join(sys.argv)}")
    print(f"  [Log File] {log_file} (tail -f to monitor)")
    print(f"  [Out] report={report_path}")
    print(f"  [Out] model={model_path}")
    print(f"  [Log] progress={progress_log_path}")
    print("  [Args] (Full Parameter Set)")
    for k, v in sorted(vars(args).items()):
        print(f"    - {k}: {v}")
    print("  [Config] (Search Internal Parameters)")
    for k, v in sorted(asdict(cfg).items()):
        print(f"    - {k}: {v}")

    start_event = {
        "reason": "run_start",
        "ts": datetime.datetime.now().isoformat(),
        "cmdline": cmdline,
        "args": vars(args),
        "config": asdict(cfg),
        "report_path": report_path,
        "model_path": model_path,
    }
    log_dir = os.path.dirname(progress_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(progress_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(start_event, ensure_ascii=True) + "\n")

def cmd_train(args):
    timestamp = get_timestamp()
    log_file = setup_output_logging(timestamp)
    model_path = args.model or f"models/arc_model_{timestamp}.json"
    report_path = args.report or f"reports/train_{timestamp}.md"
    progress_log_path = args.progress_log or f"logs/train_progress_{timestamp}.jsonl"
    
    tasks = setup_tasks(args.data, args.tasks, args.task_ids)
    lib = PrimitiveLibrary(model_path)
    lib.load()
    
    # We maintain this for cross-process worker serialization
    compact_learned_ops = {
        name: {"expr": meta.get("expr", ""), "arity": int(meta.get("arity", 1))}
        for name, meta in lib.learned_ops.items()
        if meta.get("expr")
    }
    
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, offspring=args.offspring, generations=args.generations,
        task_workers=args.task_workers, workers=args.workers, timeout_s=(None if args.timeout <= 0 else args.timeout), max_evals=args.max_evals,
        max_cost=(None if args.max_cost <= 0 else args.max_cost),
        baseline_only=True, seed=args.seed,
        mem_per_task_worker_gb=args.mem_per_task_worker_gb,
        reserve_mem_gb=args.reserve_mem_gb,
        cpu_reserve=args.cpu_reserve,
        capture_traces=args.capture_traces,
        profile_primitives=args.profile_primitives,
        stall_kill_s=(None if args.stall_kill_s <= 0 else args.stall_kill_s),
        adaptive_primitive_subset=(not args.no_adaptive_primitive_subset),
        primitive_cap=args.primitive_cap,
        progress_interval_s=args.progress_interval_s,
        progress_log_path=progress_log_path,
        max_rss_gb=args.max_rss_gb,
    )

    _log_run_header(
        title=f"WAKE-SLEEP TRAINING: {len(tasks)} tasks | {args.epochs} epochs",
        args=args,
        cfg=cfg,
        progress_log_path=progress_log_path,
        report_path=report_path,
        model_path=model_path,
        log_file=log_file,
    )
    
    for epoch in range(1, args.epochs + 1):
        ops = registry.names(domain="arc")
        
        # Pass 1: Standard Search
        report = evaluate_tasks(
            tasks, ops, cfg, label=f"Epoch {epoch} - Pass 1", 
            transition_matrix=lib.transition_matrix, learned_ops=compact_learned_ops,
            epoch_str=f"Epoch {epoch}/{args.epochs}",
            report_callback=lambda r: r.save(report_path)
        )
        
        # Identify near-misses for Pass 2 refinement (>= 80% acc but not solved)
        near_miss_tasks = [
            t for t in tasks 
            if any(r.task_name == t.name and not r.solved and r.test_acc >= 0.80 for r in report.results)
        ]
        
        if near_miss_tasks:
            print(f"  [Refinement] Found {len(near_miss_tasks)} near-miss tasks. Re-running with 2x budget...")
            refine_cfg = BenchmarkConfig(**asdict(cfg))
            refine_cfg.beam_size *= 2
            if refine_cfg.max_evals: refine_cfg.max_evals *= 2
            refine_cfg.generations += 10
            
            # Show cumulative progress in the scoreboard
            solves_p1 = sum(1 for r in report.results if r.solved)
            near_p1 = sum(1 for r in report.results if not r.solved and r.test_acc >= 0.8)
            global_stats = {
                "offset": len(tasks) - len(near_miss_tasks),
                "global_total": len(tasks),
                "global_solved": solves_p1,
                "global_near": near_p1 - len(near_miss_tasks), # Those not in this pass
            }
            
            refine_report = evaluate_tasks(
                near_miss_tasks, ops, refine_cfg, label=f"Epoch {epoch} - Pass 2 (Refinement)", 
                transition_matrix=lib.transition_matrix, learned_ops=compact_learned_ops,
                epoch_str=f"Epoch {epoch}/{args.epochs} [Refine]",
                report_callback=None,
                global_stats=global_stats
            )
            
            # Merge refined results back into main report
            refined_map = {r.task_name: r for r in refine_report.results}
            refine_fixes = 0
            for i, r in enumerate(report.results):
                if r.task_name in refined_map:
                    best_refined = refined_map[r.task_name]
                    if best_refined.solved and not r.solved:
                        refine_fixes += 1
                        report.results[i] = best_refined
                    elif best_refined.test_acc > r.test_acc:
                        report.results[i] = best_refined
            
            print(f"  [Refinement] Delta: +{refine_fixes} tasks solved via Pass 2.")
            report.label = f"Epoch {epoch} (Refined: +{refine_fixes})"
            report.save(report_path)

        # Sleep Phase: Extract from both solved AND near-solved (>=90% accuracy)
        successes = {
            r.task_name: r.best_tree 
            for r in report.results 
            if (r.solved or r.test_acc >= 0.90) and r.best_tree
        }
        
        # Auto-tune min_tasks: if solve rate is low, be more aggressive/exploratory.
        solves = sum(1 for r in report.results if r.solved)
        mt = 1 if solves < 10 else 2
        print(f"  [Sleep] Solved={solves}. Extracting with min_tasks={mt}...")
        
        lib.extract_from_tasks(successes, min_size=3, min_tasks=mt)
        lib.register_all(domain="arc")
        lib.save()

        # Update learned_ops for worker serialization in next epoch
        compact_learned_ops = {
            name: {"expr": meta.get("expr", ""), "arity": int(meta.get("arity", 1))}
            for name, meta in lib.learned_ops.items()
            if meta.get("expr")
        }
        
        usage = sum(1 for r in report.results if r.solved and "lib_op_" in str(r.found_expr))
        print(f"  [Analysis] Epoch {epoch}: {solves} solved | {len(lib.learned_ops)} abstractions | {usage} used learned ops.")

    print(f"\n✅ Training Complete. Model: {model_path}")
    return model_path

def cmd_eval(args):
    timestamp = get_timestamp()
    report_path = args.report or f"reports/eval_{timestamp}.md"
    progress_log_path = args.progress_log or f"logs/eval_progress_{timestamp}.jsonl"
    
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
    compact_learned_ops = {
        name: {"expr": meta.get("expr", ""), "arity": int(meta.get("arity", 1))}
        for name, meta in lib.learned_ops.items()
        if meta.get("expr")
    }
    
    cfg = BenchmarkConfig(
        beam_size=args.beam_size, offspring=args.offspring, generations=args.generations,
        task_workers=args.task_workers, workers=args.workers, timeout_s=(None if args.timeout <= 0 else args.timeout), max_evals=args.max_evals,
        max_cost=(None if args.max_cost <= 0 else args.max_cost),
        baseline_only=True, seed=args.seed,
        mem_per_task_worker_gb=args.mem_per_task_worker_gb,
        reserve_mem_gb=args.reserve_mem_gb,
        cpu_reserve=args.cpu_reserve,
        capture_traces=args.capture_traces,
        profile_primitives=args.profile_primitives,
        stall_kill_s=(None if args.stall_kill_s <= 0 else args.stall_kill_s),
        adaptive_primitive_subset=(not args.no_adaptive_primitive_subset),
        primitive_cap=args.primitive_cap,
        progress_interval_s=args.progress_interval_s,
        progress_log_path=progress_log_path,
        max_rss_gb=args.max_rss_gb,
    )

    _log_run_header(
        title=f"AGI EVALUATION: {len(tasks)} tasks | Model: {os.path.basename(model_path)}",
        args=args,
        cfg=cfg,
        progress_log_path=progress_log_path,
        report_path=report_path,
        model_path=model_path,
    )
    
    report = evaluate_tasks(
        tasks, registry.names(domain="arc"), cfg, label="Evaluation",
        transition_matrix=lib.transition_matrix, learned_ops=compact_learned_ops,
        report_callback=lambda r: r.save(report_path)
    )
    print(f"\n✅ Evaluation Complete. Report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Universal ARC-AGI Solver")
    subparsers = parser.add_subparsers(dest="command")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--tasks", type=int, default=400)
    shared.add_argument("--task-workers", type=int, default=0, help="Parallel tasks. 0 = auto resource-safe mode.")
    shared.add_argument("--workers", type=int, default=1)
    shared.add_argument("--beam-size", type=int, default=10)
    shared.add_argument("--offspring", type=int, default=20)
    shared.add_argument("--generations", type=int, default=25)
    shared.add_argument("--max-evals", type=int, default=1000000)
    shared.add_argument("--max-cost", type=int, default=0, help="Optional cost-unit budget per task (0=off).")
    shared.add_argument("--timeout", type=float, default=0.0)
    shared.add_argument("--progress-interval-s", type=float, default=5.0, help="Heartbeat interval in seconds.")
    shared.add_argument("--progress-log", type=str, default=None, help="Timestamped JSONL progress log path.")
    shared.add_argument("--max-rss-gb", type=float, default=0.0, help="Abort run if process RSS reaches this limit (0=off).")
    shared.add_argument("--seed", type=int, default=None)
    shared.add_argument("--task-ids", type=str, default=None)
    shared.add_argument("--mem-per-task-worker-gb", type=float, default=3.0)
    shared.add_argument("--reserve-mem-gb", type=float, default=10.0)
    shared.add_argument("--cpu-reserve", type=int, default=2)
    shared.add_argument("--capture-traces", action="store_true", help="Capture eval traces for unsolved tasks (high memory).")
    shared.add_argument("--profile-primitives", action="store_true", help="Profile primitive runtime per task (adds overhead).")
    shared.add_argument("--stall-kill-s", type=float, default=0.0, help="Kill worker only if no eval progress for N seconds (0=off).")
    shared.add_argument("--no-adaptive-primitive-subset", action="store_true")
    shared.add_argument("--primitive-cap", type=int, default=80)

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
    os.makedirs("logs", exist_ok=True)

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
