"""
domains/arc/runner.py
=====================
Run the baseline vs expanded-DSL benchmark and produce a structured report.

Usage (command line)
--------------------
    python -m domains.arc.runner
    python -m domains.arc.runner --quick       # fast (~30 s)
    python -m domains.arc.runner --workers 4   # parallel evaluation

Usage (programmatic)
--------------------
    from domains.arc.runner import run_benchmark, BenchmarkConfig
    from domains.arc.benchmark import get_benchmark

    cfg = BenchmarkConfig(generations=50, beam_size=15, verbose=False)
    report = run_benchmark(get_benchmark(), cfg)
    print(report.summary())
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from core.search import SearchConfig
from core.primitives import registry
from domains.arc.benchmark import build_benchmark
from domains.arc.domain import ARCDomain, ARCTask, grid_cell_accuracy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    Tuning knobs for a benchmark run.

    Attributes
    ----------
    beam_size : int
        Beam width for the search.
    offspring : int
        Offspring per beam member per generation.
    generations : int
        Max generations per task.
    workers : int
        Parallel workers (1 = single-threaded).
    lam : float
        MDL complexity penalty.
    verbose : bool
        Print per-task results while running.
    baseline_only : bool
        Skip the expanded-DSL run (useful for quick checks).
    """
    beam_size: int = 20
    offspring: int = 50
    generations: int = 100
    workers: int = 1
    lam: float = 0.02
    verbose: bool = True
    baseline_only: bool = False


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    task_name: str
    category: str
    true_op: str
    found_expr: str
    train_acc: float
    test_acc: float
    solved: bool          # exact match on all test pairs
    near_solved: bool     # test_acc >= 0.80
    n_nodes: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "category": self.category,
            "true_op": self.true_op,
            "found_expr": self.found_expr,
            "train_acc": round(self.train_acc, 4),
            "test_acc": round(self.test_acc, 4),
            "solved": self.solved,
            "near_solved": self.near_solved,
            "n_nodes": self.n_nodes,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    label: str
    n_ops: int
    results: list[TaskResult] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    @property
    def n_tasks(self) -> int:
        return len(self.results)

    @property
    def n_solved(self) -> int:
        return sum(1 for r in self.results if r.solved)

    @property
    def n_near(self) -> int:
        return sum(1 for r in self.results if r.near_solved)

    @property
    def pct_solved(self) -> float:
        return 100 * self.n_solved / max(self.n_tasks, 1)

    @property
    def mean_test_acc(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.test_acc for r in self.results) / len(self.results)

    def by_category(self) -> dict[str, dict]:
        cats: dict[str, dict] = {}
        for r in self.results:
            cat = r.category
            if cat not in cats:
                cats[cat] = {"solved": 0, "total": 0}
            cats[cat]["total"] += 1
            if r.solved:
                cats[cat]["solved"] += 1
        return cats

    def summary(self) -> str:
        lines = [
            f"\n{'='*65}",
            f"  {self.label}  ({self.n_ops} ops, {self.n_tasks} tasks)",
            f"{'='*65}",
            f"  Solved (exact):   {self.n_solved}/{self.n_tasks}  ({self.pct_solved:.1f}%)",
            f"  Near-solved ≥80%: {self.n_near}/{self.n_tasks}",
            f"  Mean test acc:    {self.mean_test_acc:.3f}",
            f"  Total time:       {self.total_elapsed_s:.0f}s",
            "",
            "  Per-category:",
        ]
        for cat, d in sorted(self.by_category().items()):
            bar = "█" * d["solved"] + "░" * (d["total"] - d["solved"])
            pct = 100 * d["solved"] / d["total"]
            lines.append(f"    {cat}: {bar}  {d['solved']}/{d['total']} ({pct:.0f}%)")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_ops": self.n_ops,
            "n_tasks": self.n_tasks,
            "n_solved": self.n_solved,
            "pct_solved": round(self.pct_solved, 1),
            "mean_test_acc": round(self.mean_test_acc, 4),
            "total_elapsed_s": round(self.total_elapsed_s, 1),
            "results": [r.as_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_tasks(
    tasks: list[ARCTask],
    op_subset: list[str],
    cfg: BenchmarkConfig,
    label: str,
) -> BenchmarkReport:
    """
    Run beam search on every task in *tasks* using *op_subset* as primitives.

    Returns a populated BenchmarkReport.
    """
    report = BenchmarkReport(label=label, n_ops=len(op_subset))
    t0 = time.time()

    search_cfg = SearchConfig(
        beam_size=cfg.beam_size,
        offspring=cfg.offspring,
        generations=cfg.generations,
        workers=cfg.workers,
        converge_threshold=1e-9,
        verbose=False,          # suppress per-generation logs
        seed=None,
    )

    for i, task in enumerate(tasks):
        task_t0 = time.time()

        # Unique seed per task for reproducibility across runs
        search_cfg.seed = i * 7 + 42

        domain = ARCDomain(task, lam=cfg.lam, primitive_subset=op_subset)
        result = domain.solve(config=search_cfg)

        tree = result.best_tree
        train_acc = domain.train_accuracy(tree)
        test_acc  = domain.test_accuracy(tree)
        solved    = domain.check_solution(tree)
        near      = test_acc >= 0.80

        task_elapsed = time.time() - task_t0

        tr = TaskResult(
            task_name=task.name,
            category=task.name.split("_")[0],
            true_op=task.true_op,
            found_expr=str(tree),
            train_acc=train_acc,
            test_acc=test_acc,
            solved=solved,
            near_solved=near,
            n_nodes=tree.size(),
            elapsed_s=task_elapsed,
        )
        report.results.append(tr)

        if cfg.verbose:
            status = "✓" if solved else ("~" if near else "✗")
            print(
                f"  [{i+1:2d}/{len(tasks)}] {status} {task.name[:28]:28s} "
                f"train={train_acc:.2f} test={test_acc:.2f} "
                f"→ {str(tree)[:40]}"
            )

    report.total_elapsed_s = time.time() - t0
    if cfg.verbose:
        print(report.summary())
    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    tasks: list[ARCTask] | None = None,
    cfg: BenchmarkConfig | None = None,
    save_path: str | None = None,
) -> tuple[BenchmarkReport, BenchmarkReport]:
    """
    Run both the baseline (8 ops) and expanded (all ARC ops) benchmarks.

    Parameters
    ----------
    tasks : list[ARCTask] | None
        Tasks to evaluate. Builds the default 80-task benchmark if None.
    cfg : BenchmarkConfig | None
        Configuration. Uses defaults if None.
    save_path : str | None
        If given, write results JSON to this path.

    Returns
    -------
    (baseline_report, expanded_report)
    """
    if tasks is None:
        tasks = build_benchmark()
    if cfg is None:
        cfg = BenchmarkConfig()

    # Define the two op sets
    baseline_ops = [
        "grot90", "grot180", "grot270",
        "grefl_h", "grefl_v",
        "gtrsp", "ginv", "gid",
    ]
    expanded_ops = registry.names(domain="arc")

    print(f"\nBenchmark: {len(tasks)} tasks")
    print(f"Baseline : {len(baseline_ops)} ops")
    print(f"Expanded : {len(expanded_ops)} ops")

    print(f"\n{'='*65}")
    print(f"  BASELINE ({len(baseline_ops)} ops)")
    print(f"{'='*65}")
    baseline = evaluate_tasks(tasks, baseline_ops, cfg, f"Baseline ({len(baseline_ops)} ops)")

    expanded: BenchmarkReport | None = None
    if not cfg.baseline_only:
        print(f"\n{'='*65}")
        print(f"  EXPANDED DSL ({len(expanded_ops)} ops)")
        print(f"{'='*65}")
        expanded = evaluate_tasks(tasks, expanded_ops, cfg, f"Expanded ({len(expanded_ops)} ops)")

        # Head-to-head
        new = [
            e for b, e in zip(baseline.results, expanded.results)
            if e.solved and not b.solved
        ]
        print(f"\n{'='*65}")
        print("  HEAD-TO-HEAD")
        print(f"{'='*65}")
        print(f"  Baseline : {baseline.n_solved}/{len(tasks)} ({baseline.pct_solved:.1f}%)")
        print(f"  Expanded : {expanded.n_solved}/{len(tasks)} ({expanded.pct_solved:.1f}%)")
        print(f"  Gain     : +{expanded.n_solved - baseline.n_solved} tasks")
        if new:
            print("\n  Newly solved by expanded DSL:")
            for r in new[:20]:
                print(f"    ✓ {r.task_name[:30]:30s} → {r.found_expr[:40]}")

    if save_path:
        out = {
            "baseline": baseline.as_dict(),
            "expanded": expanded.as_dict() if expanded else None,
        }
        import pathlib
        pathlib.Path(save_path).write_text(json.dumps(out, indent=2))
        print(f"\n  Results saved → {save_path}")

    return baseline, expanded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ARC-AGI-1 benchmark runner")
    parser.add_argument("--quick",        action="store_true", help="Fast run (fewer generations)")
    parser.add_argument("--baseline-only",action="store_true", help="Run baseline only")
    parser.add_argument("--workers",      type=int, default=1,   help="Parallel workers")
    parser.add_argument("--generations",  type=int, default=None, help="Override generations")
    parser.add_argument("--tasks",        type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--save",         type=str, default="results.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        beam_size   = 10 if args.quick else 20,
        offspring   = 25 if args.quick else 50,
        generations = args.generations or (40 if args.quick else 100),
        workers     = args.workers,
        verbose     = True,
        baseline_only = args.baseline_only,
    )

    tasks = build_benchmark()
    if args.tasks:
        tasks = tasks[: args.tasks]

    run_benchmark(tasks, cfg, save_path=args.save)
