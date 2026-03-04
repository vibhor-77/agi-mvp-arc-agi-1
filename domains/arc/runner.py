"""
domains/arc/runner.py
=====================
Run the baseline vs expanded-DSL benchmark and produce a structured report.

Usage (command line — programmatic benchmark, 76 tasks)
-------------------------------------------------------
    python -m domains.arc.runner
    python -m domains.arc.runner --quick                     # fast (~30 s)
    python -m domains.arc.runner --task-workers 8            # parallel tasks (M1 Max)
    python -m domains.arc.runner --quick --task-workers 8    # fast + parallel

Usage (command line — real ARC-AGI dataset)
-------------------------------------------
    # First clone the dataset:
    #   git clone https://github.com/fchollet/ARC-AGI arc_data
    python -m domains.arc.runner --data arc_data/data/evaluation
    python -m domains.arc.runner --data arc_data/data/evaluation --quick --task-workers 8

Usage (programmatic)
--------------------
    from domains.arc.runner import run_benchmark, load_tasks_from_dir, BenchmarkConfig

    # Programmatic benchmark (76 tasks, no download needed):
    from domains.arc.benchmark import build_benchmark
    cfg = BenchmarkConfig(generations=50, beam_size=15, verbose=False)
    baseline, expanded = run_benchmark(build_benchmark(), cfg)
    print(baseline.summary())

    # Real ARC-AGI dataset:
    tasks = load_tasks_from_dir("arc_data/data/evaluation")
    baseline, expanded = run_benchmark(tasks, cfg)
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from core.tree import Node
from core.search import SearchConfig
from core.primitives import registry
from domains.arc.benchmark import build_benchmark
from domains.arc.domain import ARCDomain, ARCTask, grid_cell_accuracy


# ---------------------------------------------------------------------------
# Real-dataset loader
# ---------------------------------------------------------------------------

def load_tasks_from_dir(data_dir: str) -> list[ARCTask]:
    """
    Load ARC-AGI tasks from a directory of JSON files.

    Each file must follow the standard ARC format::

        {
          "train": [{"input": [...], "output": [...]}, ...],
          "test":  [{"input": [...], "output": [...]}, ...]
        }

    Parameters
    ----------
    data_dir : str
        Path to a directory containing ``*.json`` task files.
        For the official ARC-AGI-1 repo cloned as ``arc_data/``, use:
            ``arc_data/data/evaluation``   (400 hidden eval tasks)
            ``arc_data/data/training``     (400 training tasks with solutions)

    Returns
    -------
    list[ARCTask]
        Tasks sorted by filename for reproducibility.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    ValueError
        If no ``*.json`` files are found in *data_dir*.
    """
    p = pathlib.Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir!r}\n"
            "Clone the dataset first:\n"
            "  git clone https://github.com/fchollet/ARC-AGI arc_data"
        )
    files = sorted(p.glob("*.json"))
    if not files:
        raise ValueError(
            f"No JSON files found in {data_dir!r}. "
            "Check the path — the evaluation tasks are under "
            "'arc_data/data/evaluation/', not 'arc_data/evaluation/'."
        )
    tasks = []
    for f in files:
        d = json.loads(f.read_text())
        d["name"] = f.stem
        tasks.append(ARCTask.from_dict(d))
    return tasks


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
        Parallel candidate evaluations *within* each task's beam search
        (uses multiprocessing). Set to 1 when task_workers > 1 to avoid
        nested pool issues on macOS (this is enforced automatically).
    task_workers : int
        Number of tasks to run in parallel via threads. Defaults to 1
        (sequential).

        **M1 Max recommendation:** ``task_workers=8`` saturates all 8
        performance cores without contention. The 2 efficiency cores and the
        GPU / Neural Engine are not applicable to symbolic tree search, so
        there is no benefit from setting this higher than the number of
        performance cores.

        When ``task_workers > 1``, the inner ``workers`` param is
        automatically forced to 1 to avoid nested multiprocessing pools.
    lam : float
        MDL complexity penalty.
    verbose : bool
        Print per-task start/finish lines and a live running scoreboard.
    baseline_only : bool
        Skip the expanded-DSL run (useful for quick checks).
    """
    beam_size: int = 20
    offspring: int = 50
    generations: int = 100
    workers: int = 1
    task_workers: int = 1
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
    best_tree: Optional[Node] = None

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

def _suppress_on_result(result: Any) -> None:
    pass

def evaluate_tasks(
    tasks: list[ARCTask],
    op_subset: list[str],
    cfg: BenchmarkConfig,
    label: str,
) -> BenchmarkReport:
    """
    Run beam search on every task in *tasks* using *op_subset* as primitives.

    When ``cfg.task_workers > 1`` tasks run concurrently via
    ``ThreadPoolExecutor``; the inner beam search's ``workers`` is forced to 1
    to avoid nested ``multiprocessing.Pool`` issues on macOS.

    With ``cfg.verbose=True`` each task prints a "→ STARTING" line when it
    begins and a "✓/~/✗ DONE" line when it finishes, followed by a live
    scoreboard showing:

        ✓ solved  ✗ unsolved  → active  ⏳ pending  done/total  success%

    Returns a populated BenchmarkReport.
    """
    report = BenchmarkReport(label=label, n_ops=len(op_subset))
    t0 = time.time()

    # Nested multiprocessing pools are unreliable on macOS; force inner=1
    # when tasks themselves run in parallel.
    inner_workers = 1 if cfg.task_workers > 1 else cfg.workers

    n_total = len(tasks)
    lock = threading.Lock()
    counters: dict[str, int] = {"solved": 0, "near": 0, "done": 0, "active": 0}
    ordered_results: list[tuple[int, TaskResult]] = []

    def _scoreboard() -> str:
        done  = counters["done"]
        sol   = counters["solved"]
        act   = counters["active"]
        pend  = n_total - done - act
        pct   = 100.0 * sol / done if done else 0.0
        unsol = done - sol
        return (
            f"  ┌ scoreboard ─ "
            f"✓ solved={sol}  ✗ unsolved={unsol}  "
            f"→ active={act}  ⏳ pending={pend}  "
            f"done={done}/{n_total}  "
            f"success={pct:.1f}%"
        )

    def _run_one(idx: int, task: ARCTask) -> tuple[int, TaskResult]:
        """Evaluate a single task. Safe to run inside a thread."""
        if cfg.verbose:
            with lock:
                counters["active"] += 1
                print(
                    f"  → [{idx+1:3d}/{n_total}] STARTING  {task.name}",
                    flush=True,
                )
                print(_scoreboard(), flush=True)

        search_cfg = SearchConfig(
            beam_size=cfg.beam_size,
            offspring=cfg.offspring,
            generations=cfg.generations,
            workers=inner_workers,
            converge_threshold=1e-9,
            verbose=False,
            seed=idx * 7 + 42,
        )

        task_t0 = time.time()
        domain = ARCDomain(task, lam=cfg.lam, primitive_subset=op_subset)
        # Suppress any domain-level result printing; we handle output here.
        domain.on_result = _suppress_on_result  # type: ignore[method-assign]

        result = domain.solve(config=search_cfg)

        tree      = result.best_tree
        train_acc = domain.train_accuracy(tree)
        test_acc  = domain.test_accuracy(tree)
        solved    = domain.check_solution(tree)
        near      = test_acc >= 0.80
        elapsed   = time.time() - task_t0

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
            elapsed_s=elapsed,
            best_tree=tree,
        )

        if cfg.verbose:
            status = "✓" if solved else ("~" if near else "✗")
            with lock:
                counters["active"] -= 1
                counters["done"]   += 1
                if solved:
                    counters["solved"] += 1
                if near and not solved:
                    counters["near"] += 1
                print(
                    f"  {status} [{idx+1:3d}/{n_total}] DONE      "
                    f"{task.name[:28]:28s} "
                    f"train={train_acc:.2f} test={test_acc:.2f} "
                    f"({elapsed:.1f}s) → {str(tree)[:35]}",
                    flush=True,
                )
                print(_scoreboard(), flush=True)
        else:
            with lock:
                counters["active"] -= 1
                counters["done"]   += 1
                if solved:
                    counters["solved"] += 1

        return idx, tr

    # ---- dispatch tasks -------------------------------------------------------
    if cfg.task_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.task_workers) as exe:
            futures = {exe.submit(_run_one, i, t): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                idx, tr = fut.result()
                ordered_results.append((idx, tr))
    else:
        for i, task in enumerate(tasks):
            idx, tr = _run_one(i, task)
            ordered_results.append((idx, tr))

    # Re-sort to original task order (parallel runs may finish out-of-order)
    ordered_results.sort(key=lambda x: x[0])
    report.results = [tr for _, tr in ordered_results]

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
        Tasks to evaluate. Builds the default 76-task programmatic benchmark
        if None. Pass the result of ``load_tasks_from_dir()`` to run against
        the real ARC-AGI dataset.
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
    if cfg.task_workers > 1:
        print(f"Task parallelism: {cfg.task_workers} threads (inner workers forced to 1)")

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
        pathlib.Path(save_path).write_text(json.dumps(out, indent=2))
        print(f"\n  Results saved → {save_path}")

    return baseline, expanded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ARC-AGI benchmark runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Programmatic benchmark (76 tasks, no download):\n"
            "  python -m domains.arc.runner --quick\n\n"
            "  # Real ARC-AGI-1 evaluation set (400 tasks):\n"
            "  git clone https://github.com/fchollet/ARC-AGI arc_data\n"
            "  python -m domains.arc.runner --data arc_data/data/evaluation\n\n"
            "  # Parallel tasks — M1 Max sweet spot:\n"
            "  python -m domains.arc.runner --task-workers 8\n"
        ),
    )
    parser.add_argument("--data",         type=str, default=None,
                        help="Path to directory of ARC JSON files (e.g. arc_data/data/evaluation). "
                             "If omitted, uses the built-in 76-task programmatic benchmark.")
    parser.add_argument("--quick",        action="store_true", help="Fast run (fewer generations)")
    parser.add_argument("--baseline-only",action="store_true", help="Run baseline only")
    parser.add_argument("--workers",      type=int, default=1,
                        help="Beam-search candidate workers per task (default 1). "
                             "Auto-forced to 1 when --task-workers > 1.")
    parser.add_argument("--task-workers", type=int, default=1,
                        help="Tasks to run in parallel (default 1). "
                             "M1 Max sweet spot: --task-workers 8. "
                             "Forces inner --workers to 1 on macOS.")
    parser.add_argument("--generations",  type=int, default=None, help="Override generations")
    parser.add_argument("--tasks",        type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--save",         type=str, default="results.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        beam_size    = 10 if args.quick else 20,
        offspring    = 25 if args.quick else 50,
        generations  = args.generations or (40 if args.quick else 100),
        workers      = args.workers,
        task_workers = args.task_workers,
        verbose      = True,
        baseline_only = args.baseline_only,
    )

    if args.data:
        print(f"Loading real ARC tasks from: {args.data}")
        tasks = load_tasks_from_dir(args.data)
        print(f"Loaded {len(tasks)} tasks.")
    else:
        print("No --data path given. Using built-in 76-task programmatic benchmark.")
        tasks = build_benchmark()

    if args.tasks:
        tasks = tasks[: args.tasks]

    run_benchmark(tasks, cfg, save_path=args.save)
