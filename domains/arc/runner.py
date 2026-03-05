"""
domains/arc/runner.py
=====================
Run the baseline vs expanded-DSL benchmark and produce a structured report.

Usage (command line — programmatic benchmark, 76 tasks)
-------------------------------------------------------
    python -m domains.arc.runner
    python -m domains.arc.runner --task-workers 8            # parallel tasks (M1 Max)

Usage (command line — real ARC-AGI dataset)
-------------------------------------------
    # First clone the dataset:
    #   git clone https://github.com/fchollet/ARC-AGI arc_data
    python -m domains.arc.runner --data arc_data/data/evaluation
    python -m domains.arc.runner --data arc_data/data/evaluation --task-workers 8

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
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import signal
from typing import Any, Optional

def _ignore_sigint_initializer():
    """Ignore SIGINT in multiprocessing workers so only the parent receives KeyboardInterrupt."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
        Number of tasks to run in parallel via threads. Defaults to os.cpu_count() or 1.

        **M1 Max recommendation:** The default saturates performance cores.

        When ``task_workers > 1``, the inner ``workers`` param is
        automatically forced to 1 to avoid nested multiprocessing pools.
    lam : float
        MDL complexity penalty.
    verbose : bool
        Print per-task start/finish lines and a live running scoreboard.
    baseline_only : bool
        Skip the expanded-DSL run (useful for quick checks).
    """
    beam_size: int = 10
    offspring: int = 20
    generations: int = 25
    workers: int = 1
    task_workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    lam: float = 0.02
    verbose: bool = True
    baseline_only: bool = False
    seed: int | None = None


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
    n_evals: int = 0
    introspection: str = ""
    best_tree: Optional[Node] = None
    trace: list | None = None

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
            "n_evals": self.n_evals,
            "introspection": self.introspection,
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

    def grid_to_html_table(self, grid: list[list[int]]) -> str:
        """Render an ARC grid as an HTML table with correct ARC colors."""
        if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
            return str(grid)
            
        colors = {
            0: "#000000", 1: "#0074D9", 2: "#FF4136", 3: "#2ECC40",
            4: "#FFDC00", 5: "#AAAAAA", 6: "#F012BE", 7: "#FF851B",
            8: "#7FDBFF", 9: "#870C25"
        }
        
        html = ['<table style="border-collapse: collapse; display: inline-block; margin-right: 20px; vertical-align: top;">']
        for row in grid:
            html.append('<tr>')
            for cell in row:
                color = colors.get(cell, "#ffffff")
                html.append(f'<td style="width: 20px; height: 20px; background-color: {color}; border: 1px solid #555;"></td>')
            html.append('</tr>')
        html.append('</table>')
        return "".join(html)

    def generate_markdown_report(self) -> str:
        lines = [f"# AGI Execution Report: {self.label}"]
        lines.append(f"\n## Overall Summary")
        lines.append(f"- **Total Tasks**: {self.n_tasks}")
        lines.append(f"- **Solved**: {self.n_solved} ({self.pct_solved:.1f}%)")
        lines.append(f"- **Near-solved (≥80%)**: {self.n_near}")
        lines.append(f"- **Mean Test Accuracy**: {self.mean_test_acc:.3f}")
        lines.append(f"- **Total Time**: {self.total_elapsed_s:.1f}s")
        
        lines.append(f"\n## Performance by Category")
        cats = self.by_category()
        for cat, d in sorted(cats.items()):
            pct = 100 * d["solved"] / d["total"] if d["total"] > 0 else 0
            lines.append(f"- **{cat}**: {d['solved']}/{d['total']} ({pct:.1f}%)")
            
        lines.append(f"\n## Introspection Analysis (Failures by Category)")
        
        failed_by_cat = {}
        for r in self.results:
            if not r.solved:
                if r.category not in failed_by_cat:
                    failed_by_cat[r.category] = []
                failed_by_cat[r.category].append(r)
                
        if not failed_by_cat:
            lines.append("No failures! All tasks solved.")
        
        for cat, fails in sorted(failed_by_cat.items()):
            lines.append(f"\n### {cat} ({len(fails)} failures)")
            for r in fails[:15]: 
                lines.append(f"- **{r.task_name}**:")
                lines.append(f"  - **Introspection**: {r.introspection}")
                if r.best_tree:
                    ast_str = r.found_expr if len(r.found_expr) < 120 else r.found_expr[:117] + "..."
                    lines.append(f"  - **Best AST**: `{ast_str}`")
                    lines.append(f"  - **Train Acc**: {r.train_acc:.2f} | **Test Acc**: {r.test_acc:.2f}")
                
                # If we have a trace, visualize it!
                if r.trace:
                    lines.append("\n  <div style='overflow-x: auto; white-space: nowrap;'>")
                    for step_name, step_grid in r.trace:
                        lines.append("    <div style='display: inline-block; text-align: center; margin-right: 15px;'>")
                        lines.append(f"      <div style='font-family: monospace; font-size: 12px; margin-bottom: 5px; max-width: 150px; overflow: hidden; text-overflow: ellipsis;'>{step_name}</div>")
                        lines.append(f"      {self.grid_to_html_table(step_grid)}")
                        lines.append("    </div>")
                        if step_name != r.trace[-1][0]: # Arrow between steps
                            lines.append("    <div style='display: inline-block; vertical-align: top; margin-top: 20px; font-weight: bold;'>&#8594;</div>")
                    lines.append("  </div>\n")
            if len(fails) > 15:
                lines.append(f"- *...and {len(fails) - 15} more {cat} tasks.*")
                
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def _suppress_on_result(result: Any) -> None:
    pass

def _run_task_process(
    idx: int, 
    task: ARCTask, 
    cfg: BenchmarkConfig, 
    op_subset: list[str], 
    inner_workers: int,
    transition_matrix: dict[str, dict[str, float]] | None,
    learned_ops: dict[str, dict] | None = None
) -> tuple[int, TaskResult]:
    """Execute the core logic for a single task inside an isolated multiprocessing worker."""
    if learned_ops:
        from core.library import PrimitiveLibrary
        lib = PrimitiveLibrary()
        lib.learned_ops = learned_ops
        lib.register_all(domain="arc")
    if cfg.verbose:
        print(f"  → [{idx+1:3d}] STARTING  {task.name}", flush=True)

    search_cfg = SearchConfig(
        beam_size=cfg.beam_size,
        offspring=cfg.offspring,
        generations=cfg.generations,
        workers=inner_workers,
        converge_threshold=1e-9,
        verbose=False,
        seed=idx * 7 + 42 if cfg.seed is None else cfg.seed + idx,
    )

    task_t0 = time.time()
    domain = ARCDomain(task, lam=cfg.lam, primitive_subset=op_subset)
    domain.on_result = _suppress_on_result  # type: ignore[method-assign]

    result = domain.solve(config=search_cfg, transition_matrix=transition_matrix)

    tree      = result.best_tree
    train_acc = domain.train_accuracy(tree)
    test_acc  = domain.test_accuracy(tree)
    solved    = domain.check_solution(tree)
    near      = test_acc >= 0.80
    elapsed   = time.time() - task_t0

    introspection_msg = ""
    if not solved and tree is not None:
        try:
            # We must re-evaluate the best tree on the first train instance to generate the diff heuristic
            pair = task.train_pairs[0]
            if len(pair) == 2:
                inp, expected = pair[0], pair[1]
                actual, trace = tree.eval_trace([inp], domain._primitives)
                
                if isinstance(actual, list) and len(actual) > 0 and isinstance(actual[0], list):
                    if len(expected) != len(actual) or len(expected[0]) != len(actual[0]):
                        introspection_msg = f"Dimension mismatch: Expected {len(expected)}x{len(expected[0])}, Actual {len(actual)}x{len(actual[0])}"
                    else:
                        mismatches = 0
                        total = len(expected) * len(expected[0])
                        for r in range(len(expected)):
                            for c in range(len(expected[0])):
                                if expected[r][c] != actual[r][c]:
                                    mismatches += 1
                        introspection_msg = f"Pixel mismatch: {mismatches}/{total} pixels incorrect"
                else:
                    introspection_msg = f"Logical Failure: Output is not a valid 2D grid (got {type(actual).__name__})"
            else:
                introspection_msg = "Logical Failure: Invalid train pair format"
        except Exception as e:
            introspection_msg = f"Evaluation Crash: {type(e).__name__}: {str(e)}"
    elif not solved:
        introspection_msg = "Search Capacity Exhausted: No valid logical AST discovered within generation limits."

    tr = TaskResult(
        task_name=task.name,
        category=task.name.split("_")[0],
        true_op=task.true_op,
        found_expr=str(tree),
        train_acc=train_acc,
        test_acc=test_acc,
        solved=solved,
        near_solved=near,
        n_nodes=tree.size() if tree else 0,
        elapsed_s=elapsed,
        n_evals=result.n_evals,
        introspection=introspection_msg,
        best_tree=tree,
        trace=trace if 'trace' in locals() else None,
    )
    return idx, tr

def evaluate_tasks(
    tasks: list[ARCTask],
    op_subset: list[str],
    cfg: BenchmarkConfig,
    label: str,
    transition_matrix: dict[str, dict[str, float]] | None = None,
    learned_ops: dict[str, dict] | None = None,
    epoch_str: str = "",
    report_callback=None,
) -> BenchmarkReport:
    """
    Run beam search on every task in *tasks* using *op_subset* as primitives.

    When ``cfg.task_workers > 1`` tasks run concurrently via
    ``ProcessPoolExecutor``; the inner beam search's ``workers`` is forced to 1
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
    counters: dict[str, int] = {"solved": 0, "near": 0, "done": 0}
    ordered_results: list[tuple[int, TaskResult]] = []

    def _scoreboard() -> str:
        done  = counters["done"]
        sol   = counters["solved"]
        
        # In a ProcessPool, active threads are bounded by task_workers
        act   = min(cfg.task_workers, n_total - done)
        pend  = n_total - done - act
        
        pct   = 100.0 * sol / done if done else 0.0
        unsol = done - sol
        
        prefix = f" [{epoch_str}]" if epoch_str else ""
        return (
            f"  ┌ scoreboard{prefix} ─ "
            f"✓ solved={sol}  ✗ unsolved={unsol}  "
            f"→ active={act}  ⏳ pending={pend}  "
            f"done={done}/{n_total}  "
            f"success={pct:.1f}%"
        )

    def _handle_result(idx: int, tr: TaskResult, task: ARCTask) -> None:
        if cfg.verbose:
            status = "✓" if tr.solved else ("~" if tr.near_solved else "✗")
            counters["done"]   += 1
            if tr.solved:
                counters["solved"] += 1
            if tr.near_solved and not tr.solved:
                counters["near"] += 1
            print(
                f"  {status} [{idx+1:3d}/{n_total}] DONE      "
                f"{task.name[:28]:28s} "
                f"train={tr.train_acc:.2f} test={tr.test_acc:.2f} "
                f"({tr.elapsed_s:.1f}s) → {tr.found_expr[:50]}",
                flush=True,
            )
            if not tr.solved and tr.best_tree is not None and tr.n_nodes > 1:
                print(f"    [FAILURE ANALYSIS] Best AST: {tr.found_expr}")
                pair = task.train_pairs[0]
                if len(pair) == 2:
                    inp, expected = pair[0], pair[1]
                    print("    Input Grid:")
                    for row in inp: print(f"      {row}")
                    print("    Expected Output:")
                    for row in expected: print(f"      {row}")
                    try:
                        domain_phantom = ARCDomain(task, primitive_subset=op_subset)
                        actual = tr.best_tree.eval([inp], domain_phantom._primitives)
                        print("    Actual output from AST:")
                        if isinstance(actual, list) and len(actual) > 0 and isinstance(actual[0], list):
                            for row in actual: print(f"      {row}")
                            
                            # Verbose Pixel-Level Explainability (Diff Map)
                            print("    [DIFF MAP] (Expected vs Actual):")
                            if len(expected) == len(actual) and len(expected[0]) == len(actual[0]):
                                for r in range(len(expected)):
                                    diff_row = []
                                    for c in range(len(expected[0])):
                                        if expected[r][c] == actual[r][c]:
                                            diff_row.append(".") # Match
                                        else:
                                            diff_row.append("X") # Mismatch
                                    print(f"      {diff_row}")
                            else:
                                print(f"      [Dimension Mismatch] Expected: {len(expected)}x{len(expected[0])}, Actual: {len(actual)}x{len(actual[0] if actual else 0)}")
                        else:
                            print(f"      {actual}")
                    except Exception as e:
                        print(f"      [EVAL ERROR] {type(e).__name__}: {e}")
                    print(f"    {'-'*40}")
            print(_scoreboard(), flush=True)
        else:
            counters["done"]   += 1
            if tr.solved:
                counters["solved"] += 1

        if report_callback:
            # Temporarily build the list of ordered results including this current TaskResult
            # before it gets officially appended to the ordered_results array in the main loop
            report.results = [r for _, r in sorted(ordered_results + [(idx, tr)], key=lambda x: x[0])]
            report.total_elapsed_s = time.time() - t0
            report_callback(report)

    # ---- dispatch tasks -------------------------------------------------------
    if cfg.task_workers > 1:
        exe = ProcessPoolExecutor(max_workers=cfg.task_workers, initializer=_ignore_sigint_initializer)
        try:
            futures = {exe.submit(_run_task_process, i, t, cfg, op_subset, inner_workers, transition_matrix, learned_ops): i for i, t in enumerate(tasks)}
            for fut in as_completed(futures):
                idx, tr = fut.result()
                _handle_result(idx, tr, tasks[idx])
                ordered_results.append((idx, tr))
        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt received. Forcefully terminating workers...", flush=True)
            for p in exe._processes.values():
                try:
                    p.kill()
                except Exception:
                    pass
            exe.shutdown(wait=False, cancel_futures=True)
            import os
            os._exit(130)
        finally:
            exe.shutdown(wait=True)
    else:
        try:
            for i, task in enumerate(tasks):
                print(_scoreboard(), flush=True)
                idx, tr = _run_task_process(i, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops)
                _handle_result(idx, tr, task)
                ordered_results.append((idx, tr))
        except KeyboardInterrupt:
            print("\n[!] KeyboardInterrupt received. Aborting evaluation...", flush=True)
            import sys
            sys.exit(130)

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
            "  python -m domains.arc.runner\n\n"
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
    parser.add_argument("--baseline-only",action="store_true", help="Run baseline only")
    parser.add_argument("--workers",      type=int, default=1,
                        help="Beam-search candidate workers per task (default 1). "
                             "Auto-forced to 1 when --task-workers > 1.")
    parser.add_argument("--task-workers", type=int, default=os.cpu_count() or 1,
                        help="Tasks to run in parallel (default: os.cpu_count()). "
                             "Forces inner --workers to 1 on macOS.")
    parser.add_argument("--generations",  type=int, default=100, help="Override generations")
    parser.add_argument("--tasks",        type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--save",         type=str, default="results.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        beam_size    = 10,
        offspring    = 20,
        generations  = args.generations,
        workers      = args.workers,
        task_workers = args.task_workers,
        verbose      = True,
        baseline_only = args.baseline_only,
        seed         = None,
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
