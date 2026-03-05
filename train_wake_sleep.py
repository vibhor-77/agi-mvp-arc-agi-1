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
    lib.load()
    
    print(f"\n🚀 Starting Wake-Sleep Training over {len(tasks)} tasks for {epochs} epochs")
    if lib.learned_ops:
        print(f"Loaded {len(lib.learned_ops)} existing learned primitives from disk.")
    
    # Pre-generate the current model.json file so the user can verify its location immediately
    lib.save()
    
    full_report = f"# Wake-Sleep Training Log\n\n**Total Epochs**: {epochs} | **Tasks**: {len(tasks)}\n\n---\n\n"
    
    def _save_live_report(added_markdown: str):
        current_markdown = full_report + added_markdown
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(current_markdown)

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
{current_markdown}
  </script>
  <script>
    document.getElementById('content').innerHTML = marked.parse(document.getElementById('md-content').textContent);
  </script>
</body>
</html>"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    
    for epoch in range(1, epochs + 1):
        # The op subset is whatever is currently registered in 'arc' domain
        # This grows every epoch after the sleep phase.
        active_ops = registry.names(domain="arc")
        
        print(f"\n{'='*65}")
        print(f"  EPOCH {epoch}/{epochs}  —  WAKE PHASE  [{len(active_ops)} ops available]")
        print(f"{'='*65}")
        
        
        def epoch_report_callback(rep):
            _save_live_report(rep.generate_markdown_report() + "\n\n---\n\n")
            
        # Wake: Attempt to solve all tasks
        report = evaluate_tasks(
            tasks=tasks, 
            op_subset=active_ops, 
            cfg=cfg, 
            label=f"Epoch {epoch} ({len(active_ops)} ops)",
            transition_matrix=lib.transition_matrix,
            learned_ops=lib.learned_ops,
            epoch_str=f"Epoch {epoch}/{epochs}",
            report_callback=epoch_report_callback,
        )
        
        # Sleep: Collect successful trees
        successful_trees = {}
        for r in report.results:
            if r.solved and r.best_tree is not None:
                successful_trees[r.task_name] = r.best_tree
                
        print(f"\n  [Sleep Phase] Extracting abstractions from {len(successful_trees)} solved tasks...")
        
        # Adaptive library learning thresholds:
        # - min_size=2: Capture 2-node compositions like grot90(ginv(x)) which are the
        #   most common useful patterns. Previously min_size=3 was too strict.
        # - min_tasks: Scale with number of solved tasks. When few tasks are solved (<10),
        #   be generous (min_tasks=1) to maximize learning. At scale, require at least 2
        #   tasks to share a sub-tree before promoting it.
        if len(successful_trees) < 10:
            min_tasks = 1
        elif len(successful_trees) < 30:
            min_tasks = 2
        else:
            min_tasks = 3
        lib.extract_from_tasks(successful_trees, min_size=2, min_tasks=min_tasks)
        
        print(f"  [Sleep Phase] Current Library Size: {len(lib.learned_ops)} abstractions")
        
        # Inject learned abstractions back into active registry for next Epoch
        lib.register_all(domain="arc")
        lib.save()

        # Append Epoch report for next loops
        full_report += report.generate_markdown_report() + "\n\n---\n\n"

    print("\n🎉 Wake-Sleep Training Complete!")
    print(f"Total Learned Primitives: {len(lib.learned_ops)}")
    print(f"Full introspection report saved to: {report_path}")
    print(f"Browser-friendly report saved to: {report_path.replace('.md', '.html')}")
    print(f"Model saved to: {model_path}\n")

    # Print summary of the model
    if lib.learned_ops:
        print("+" + "-"*63 + "+")
        print("|" + " MODEL ABSTRACTION DICTIONARY SUMMARY ".center(63) + "|")
        print("+" + "-"*63 + "+")
        for name, data in lib.learned_ops.items():
            print(f"  {name}: {data.get('body', 'Unknown AST')}")
    else:
        print("[!] No new abstractions were discovered and stored in the model.")

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
    parser.add_argument("--task-ids", type=str, default=None, help="Comma-separated list of task IDs to evaluate explicitly (e.g. 007bbfb7,025d127b)")
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
        tasks = load_tasks_from_dir(args.data)
        
        # Generative Curriculum Learning
        # Sort by total pixel count across train_pairs, forcing the Search Engine 
        # to establish fundamental abstractions on low dimensions first.
        def _task_complexity(t):
            return sum(len(inp) * len(inp[0]) for inp, out in t.train_pairs) + (len(t.train_pairs) * 10)
            
        tasks.sort(key=_task_complexity)

        if args.task_ids:
            # Filter specifically by task IDs
            target_ids = [t.strip() for t in args.task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
            if not tasks:
                print(f"Error: None of the specific task IDs {target_ids} were found in {args.data}")
                sys.exit(1)
        else:
            tasks = tasks[:args.tasks]
    except Exception as e:
        print(e)
        tasks = [] # fallback to built in programmatic tasks
        from domains.arc.benchmark import build_benchmark
        tasks = build_benchmark()
        if args.task_ids:
            target_ids = [t.strip() for t in args.task_ids.split(",")]
            tasks = [t for t in tasks if t.name in target_ids]
        else:
            tasks = tasks[:args.tasks]

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
