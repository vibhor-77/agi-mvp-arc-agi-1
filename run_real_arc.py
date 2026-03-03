#!/usr/bin/env python3
"""
run_real_arc.py
===============
Convenience script to run the benchmark against the real ARC-AGI-1 dataset.

Prerequisites
-------------
Clone the dataset first (one-time):
    git clone https://github.com/fchollet/ARC-AGI arc_data

This places the JSON task files at:
    arc_data/data/evaluation/   — 400 evaluation tasks (no public answers)
    arc_data/data/training/     — 400 training tasks   (answers included)

Usage
-----
    python run_real_arc.py                          # full eval run (slow)
    python run_real_arc.py --quick                  # fast settings
    python run_real_arc.py --quick --tasks 10       # smoke test, 10 tasks
    python run_real_arc.py --split training         # run on training set instead
    python run_real_arc.py --workers 4              # parallel (uses CPU cores)
    python run_real_arc.py --save my_results.json   # custom output path
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from domains.arc.runner import load_tasks_from_dir, run_benchmark, BenchmarkConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ARC-AGI benchmark against the real dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  git clone https://github.com/fchollet/ARC-AGI arc_data\n"
            "  python run_real_arc.py --quick --tasks 10\n"
        ),
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to directory of ARC JSON files. "
             "Defaults to arc_data/data/<split>/.",
    )
    parser.add_argument(
        "--split", type=str, default="evaluation", choices=["evaluation", "training"],
        help="Which split to use when --data is not set (default: evaluation).",
    )
    parser.add_argument("--quick",        action="store_true", help="Fast run (fewer generations)")
    parser.add_argument("--baseline-only",action="store_true", help="Run baseline DSL only")
    parser.add_argument("--workers",      type=int, default=1,   help="Parallel workers")
    parser.add_argument("--generations",  type=int, default=None, help="Override generations")
    parser.add_argument("--tasks",        type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--save",         type=str, default="results_real.json",
                        help="Output JSON path (default: results_real.json)")
    args = parser.parse_args()

    data_dir = args.data or os.path.join("arc_data", "data", args.split)

    tasks = load_tasks_from_dir(data_dir)
    print(f"Loaded {len(tasks)} tasks from '{data_dir}'.")

    if args.tasks:
        tasks = tasks[: args.tasks]
        print(f"Limiting to first {args.tasks} tasks.")

    cfg = BenchmarkConfig(
        beam_size     = 10 if args.quick else 20,
        offspring     = 25 if args.quick else 50,
        generations   = args.generations or (40 if args.quick else 100),
        workers       = args.workers,
        verbose       = True,
        baseline_only = args.baseline_only,
    )

    run_benchmark(tasks, cfg, save_path=args.save)


if __name__ == "__main__":
    main()
