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
import datetime

from domains.arc.runner import load_tasks_from_dir, BenchmarkConfig, evaluate_tasks
from core.library import PrimitiveLibrary
from core.primitives import registry

def run_evaluation(data_dir: str, num_tasks: int, cfg: BenchmarkConfig, model_path: str, report_path: str) -> None:
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
    
    def live_eval_callback(rep):
        markdown_str = rep.generate_markdown_report()
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_str)

        html_path = report_path.replace(".md", ".html")
        html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AGI Report</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
  <style>body {{ box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }}</style>
</head>
<body class="markdown-body">
  <div id="content"></div>
  <script type="text/markdown" id="md-content">
{markdown_str}
  </script>
  <script>
    document.getElementById('content').innerHTML = marked.parse(document.getElementById('md-content').textContent);
  </script>
</body>
</html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    # Run a generic BeamSearch, but equipped with the Generative Priors (transition matrix)
    # to guide the discovery tree toward likely structures.
    report = evaluate_tasks(
        tasks=tasks, 
        op_subset=active_ops, 
        cfg=cfg, 
        label=f"Evaluation Mode",
        transition_matrix=lib.transition_matrix,
        learned_ops=lib.learned_ops,
        report_callback=live_eval_callback,
    )

    print("\n✅ Evaluation Complete!")
    print(f"Markdown Introspection report saved to: {report_path}")
    print(f"Browser-friendly report saved to: {report_path.replace('.md', '.html')}\n")
    print(report.summary())

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    # Forces evaluation to ONLY look at the eval subset
    parser.add_argument("--data", type=str, default="arc_data/data/evaluation")
    parser.add_argument("--tasks", type=int, default=400, help="Number of tasks to evaluate")
    parser.add_argument("--workers", type=int, default=1, help="Parallel processing across tasks (default 1 to keep feedback clean)")
    parser.add_argument("--task-workers", type=int, default=8, help="Parallel processing within a single task's search")
    parser.add_argument("--beam-size", type=int, default=10, help="Size of the Beam Search queue")
    parser.add_argument("--offspring", type=int, default=20, help="Number of mutations per generation")
    parser.add_argument("--generations", type=int, default=100, help="Number of deep search iterations per task")
    parser.add_argument("--model", type=str, default="LATEST", help="Filepath to load the learned primitive dictionary from. Defaults to latest file in models/")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic random seed for the search engine")
    parser.add_argument("--report", type=str, default=f"reports/eval_{timestamp}.md", help="Markdown file to accumulate Introspection diagnostics")
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated list of task IDs to evaluate explicitly (e.g. 007bbfb7,025d127b)")
    
    args = parser.parse_args()

    if args.model == "LATEST":
        if not os.path.exists("models"):
            print("Error: 'models' directory not found. Please run train_wake_sleep.py first or specify --model.")
            sys.exit(1)
        model_files = [os.path.join("models", f) for f in os.listdir("models") if f.endswith(".json")]
        if not model_files:
            print("Error: No .json models found in 'models' directory.")
            sys.exit(1)
        args.model = max(model_files, key=os.path.getmtime)
        print(f"Auto-selected latest model: {args.model}")

    # Ensure reports directory exists
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

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
    
    try:
        tasks = load_tasks_from_dir(args.data)
        if args.task_ids:
            target_ids = [t.strip() for t in args.task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
            if not tasks:
                print(f"Error: None of the specific task IDs {target_ids} were found in {args.data}")
                sys.exit(1)
        else:
            tasks = tasks[:args.tasks]
    except Exception as e:
        print(e)
        tasks = []
        from domains.arc.benchmark import build_benchmark
        tasks = build_benchmark()
        if args.task_ids:
            target_ids = [t.strip() for t in args.task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
        else:
            tasks = tasks[:args.tasks]

    run_evaluation(args.data, len(tasks), cfg, args.model, args.report)

