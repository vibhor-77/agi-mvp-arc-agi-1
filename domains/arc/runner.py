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
import multiprocessing as mp
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
    timeout_s : float | None
        Time limit per task in seconds.
    max_evals : int | None
        Maximum program evaluations per task. Primary deterministic limit.
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
    timeout_s: float | None = 300.0  # Optional wall-clock safety
    max_evals: int | None = 1000000  # 1M evaluations (Deterministic Pruning)


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
            
        lines.append(f"\n## Detailed Results (N={self.n_tasks})")
        lines.append("| Task | Status | Evals | Time | Train | Test | Expression |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in self.results:
            status = "✅ SOLVED" if r.solved else ("⚠️ NEAR" if r.near_solved else "❌ FAIL")
            expr = r.found_expr if len(r.found_expr) < 40 else r.found_expr[:37] + "..."
            lines.append(f"| {r.task_name} | {status} | {r.n_evals/1000:5.1f}k | {r.elapsed_s:.1f}s | {r.train_acc:.2f} | {r.test_acc:.2f} | `{expr}` |")
            
        lines.append(f"\n## Introspection Analysis (Failures by Category)")
        failed_by_cat = {}
        for r in self.results:
            if not r.solved:
                if r.category not in failed_by_cat: failed_by_cat[r.category] = []
                failed_by_cat[r.category].append(r)
                
        if not failed_by_cat: lines.append("No failures! All tasks solved.")
        
        for cat, fails in sorted(failed_by_cat.items()):
            lines.append(f"\n### {cat} ({len(fails)} failures)")
            for r in fails[:15]: 
                lines.append(f"- **{r.task_name}**: {r.introspection}")
                if r.best_tree:
                    ast_str = r.found_expr if len(r.found_expr) < 120 else r.found_expr[:117] + "..."
                    lines.append(f"  - **Best AST**: `{ast_str}` | {r.n_evals/1000:.1f}k evals | {r.elapsed_s:.1f}s")
                
                if r.trace:
                    lines.append("\n  <div style='overflow-x: auto; white-space: nowrap;'>")
                    for step_name, step_grid in r.trace:
                        lines.append("    <div style='display: inline-block; text-align: center; margin-right: 15px;'>")
                        lines.append(f"      <div style='font-family: monospace; font-size: 10px;'>{step_name}</div>")
                        lines.append(f"      {self.grid_to_html_table(step_grid)}")
                        lines.append("    </div>")
                        if step_name != r.trace[-1][0]: lines.append("    <div style='display: inline-block; vertical-align: top; margin-top: 20px;'>&#8594;</div>")
                    lines.append("  </div>\n")
            if len(fails) > 15: lines.append(f"- *...and {len(fails) - 15} more {cat} tasks.*")
        return "\n".join(lines)

    def save(self, md_path: str):
        """Persist the report to both Markdown and HTML formats."""
        content = self.generate_markdown_report()
        os.makedirs(os.path.dirname(md_path), exist_ok=True) if os.path.dirname(md_path) else None
        with open(md_path, "w", encoding="utf-8") as f: f.write(content)
        
        html_template = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>AGI Report</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        <style>body { box-sizing: border-box; min-width: 200px; max-width: 980px; margin: 0 auto; padding: 45px; }</style>
        </head><body class="markdown-body"><div id="content"></div>
        <script type="text/markdown" id="md-content">CONTENT_PLACEHOLDER</script>
        <script>document.getElementById('content').innerHTML = marked.parse(document.getElementById('md-content').textContent);</script>
        </body></html>"""
        with open(md_path.replace(".md", ".html"), "w", encoding="utf-8") as f:
            f.write(html_template.replace("CONTENT_PLACEHOLDER", content))


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

_worker_shared_evals = None
_worker_idx = None

def _ignore_sigint_initializer(evals_arr: Any = None):
    global _worker_shared_evals
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if evals_arr is not None:
        _worker_shared_evals = evals_arr

def _run_task_process(
    idx: int, 
    task: ARCTask, 
    cfg: BenchmarkConfig, 
    op_subset: list[str], 
    inner_workers: int,
    transition_matrix: dict[str, dict[str, float]] | None,
    learned_ops: dict[str, dict] | None = None,
    worker_idx: int = 0
) -> tuple[int, TaskResult]:
    global _worker_idx
    _worker_idx = worker_idx
    
    if learned_ops:
        from core.library import PrimitiveLibrary
        lib = PrimitiveLibrary()
        lib.learned_ops = learned_ops
        lib.register_all(domain="arc")
    
    def on_step(n_evals, elapsed):
        if _worker_shared_evals is not None and _worker_idx is not None:
            _worker_shared_evals[_worker_idx] = n_evals

    search_cfg = SearchConfig(
        beam_size=cfg.beam_size,
        offspring=cfg.offspring,
        generations=cfg.generations,
        workers=inner_workers,
        converge_threshold=1e-9,
        verbose=False,
        seed=idx * 7 + 42 if cfg.seed is None else cfg.seed + idx,
        timeout_s=cfg.timeout_s,
        max_evals=cfg.max_evals,
    )

    task_t0 = time.time()
    domain = ARCDomain(task, lam=cfg.lam, primitive_subset=op_subset)
    domain.on_result = lambda x: None  # type: ignore[method-assign]
    
    result = domain.solve(config=search_cfg, transition_matrix=transition_matrix, on_step=on_step)

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

def _worker_wrapper(idx, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops, slot, conn, shared_evals):
    """Top-level wrapper for multiprocessing processes."""
    global _worker_shared_evals
    _worker_shared_evals = shared_evals
    try:
        from domains.arc.runner import _run_task_process
        _, tr = _run_task_process(idx, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops, slot)
        conn.send(tr)
    except Exception as e:
        # Avoid crashing the parent if worker fails
        conn.send(None)
    finally:
        conn.close()

class LiveScoreboard:
    """Encapsulates the status reporting logic for evaluate_tasks."""
    def __init__(self, n_total: int, t0: float, task_workers: int, epoch_str: str = ""):
        self.n_total = n_total
        self.t0 = t0
        self.task_workers = task_workers
        self.epoch_str = epoch_str
        self.counters = {
            "solved": 0, "near": 0, "done": 0,
            "solved_time": 0.0, "near_time": 0.0, "unsolved_time": 0.0,
            "total_evals": 0, "solved_evals": 0, "near_evals": 0, "unsolved_evals": 0
        }
        self.start_times: dict[int, float] = {}
        self.shared_evals: Any = None

    def update(self, tr: TaskResult):
        self.counters["done"] += 1
        self.counters["total_evals"] += tr.n_evals
        if tr.solved:
            self.counters["solved"] += 1
            self.counters["solved_time"] += tr.elapsed_s
            self.counters["solved_evals"] += tr.n_evals
        else:
            self.counters["unsolved_time"] += tr.elapsed_s
            self.counters["unsolved_evals"] += tr.n_evals
            if tr.near_solved:
                self.counters["near"] += 1
                self.counters["near_time"] += tr.elapsed_s
                self.counters["near_evals"] += tr.n_evals

    def render(self) -> str:
        c = self.counters
        done, sol, near = int(c["done"]), int(c["solved"]), int(c["near"])
        unsol = done - sol
        act = min(self.task_workers, self.n_total - done)
        pend = self.n_total - done - act
        elapsed = time.time() - self.t0
        
        # Latency metrics
        task_total_t = c["solved_time"] + c["near_time"] + c["unsolved_time"]
        avg_work_t = task_total_t / done if done > 0 else 0.0
        
        # Eval metrics
        total_evals = int(c["total_evals"])
        avg_work_e = total_evals / done if done > 0 else 0.0
        evals_p_s = total_evals / elapsed if elapsed > 0 else 0.0
        
        # Efficiency metrics
        speedup = task_total_t / elapsed if elapsed > 0 else 0.0
        utilization = 100 * (speedup / self.task_workers) if self.task_workers > 0 else 0.0
        pct = 100 * sol / self.n_total if self.n_total > 0 else 0.0

        # Stragglers
        running = [time.time() - t for t in self.start_times.values()]
        max_run_t = max(running) if running else 0.0
        max_run_e = max(self.shared_evals) if self.shared_evals else 0

        prefix = f" [{self.epoch_str}]" if self.epoch_str else ""
        return (
            f"  ┌ scoreboard{prefix} ─\n"
            f"  │ ✓ solved={sol} ({100*sol/max(done,1):.1f}%)  ⚠️ near={near}  ✗ unsolved={unsol}\n"
            f"  │ → active={act}  ⏳ pending={pend}  done={done}/{self.n_total}  success={pct:.1f}%\n"
            f"  │ TIME:  elapsed={elapsed:.1f}s (Throughput: {elapsed/max(done,1):.2f}s/task | Latency Avg: {avg_work_t:.1f}s)\n"
            f"  │ WORK:  speedup={speedup:.2f}x ({utilization:.1f}% core) | STRAGGLER: {max_run_t:.1f}s, {max_run_e/1000:.1f}k evals\n"
            f"  │ EVALS: total={total_evals/1000:.1f}k ({evals_p_s/1000:.2f}k/s | Per-Task Avg: {avg_work_e/1000:.1f}k)"
        )

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
    report = BenchmarkReport(label=label, n_ops=len(op_subset))
    t0 = time.time()
    inner_workers = 1 if cfg.task_workers > 1 else cfg.workers

    scoreboard = LiveScoreboard(len(tasks), t0, cfg.task_workers, epoch_str)
    ordered_results: list[tuple[int, TaskResult]] = []
    last_report_t = 0.0

    import multiprocessing as mp
    shared_evals = mp.RawArray('i', cfg.task_workers)
    scoreboard.shared_evals = shared_evals

    def _on_done(idx, tr, task):
        nonlocal last_report_t
        if idx in scoreboard.start_times: del scoreboard.start_times[idx]
        scoreboard.update(tr)
        ordered_results.append((idx, tr))
        if cfg.verbose:
            status = "✓" if tr.solved else ("~" if tr.near_solved else "✗")
            print(f"  {status} [{idx+1:3d}/{len(tasks)}] DONE {task.name[:28]:28s} {tr.n_evals/1000:5.1f}k evals ({tr.elapsed_s:.1f}s)")
            print(scoreboard.render(), flush=True)

        if report_callback:
            now = time.time()
            if (scoreboard.counters["done"] % 5 == 0) or (now - last_report_t > 15) or (scoreboard.counters["done"] == len(tasks)):
                report.results = [r for _, r in sorted(ordered_results, key=lambda x: x[0])]
                report.total_elapsed_s = now - t0
                report_callback(report)
                last_report_t = now

    if cfg.task_workers > 1:
        import multiprocessing as mp
        active_processes = {} # {proc_idx: (process, task_idx, start_time)}
        task_queue = list(enumerate(tasks))
        free_slots = list(range(cfg.task_workers))

        try:
            while task_queue or active_processes:
                # Start new tasks if slots available
                while task_slots := [s for s in free_slots if task_queue]:
                    slot = slot_idx = task_slots[0]
                    free_slots.remove(slot)
                    task_idx, task = task_queue.pop(0)

                    if cfg.verbose: print(f"  → [{task_idx+1:3d}] STARTING  {task.name}", flush=True)
                    
                    # Create a pipe for the result
                    parent_conn, child_conn = mp.Pipe()
                    
                    # We reuse _run_task_process but wrap it to send result through the pipe
                    p = mp.Process(
                        target=_worker_wrapper, 
                        args=(task_idx, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops, slot, child_conn, shared_evals)
                    )
                    p.start()
                    active_processes[slot] = (p, task_idx, time.time(), parent_conn)
                    scoreboard.start_times[task_idx] = time.time()

                # Monitor active processes
                finished_slots = []
                for slot, (p, task_idx, start_t, conn) in list(active_processes.items()):
                    now = time.time()
                    elapsed = now - start_t
                    timeout = cfg.timeout_s or 300.0

                    # Check if finished
                    if conn.poll():
                        try:
                            tr = conn.recv()
                            if tr:
                                _on_done(task_idx, tr, tasks[task_idx])
                            else:
                                # Worker crashed
                                pass
                        except EOFError:
                            pass
                        p.join()
                        finished_slots.append(slot)
                    elif not p.is_alive():
                        # Process died unexpectedly
                        p.join()
                        finished_slots.append(slot)
                    elif elapsed > timeout + 10: # Hard kill after 10s grace
                        if cfg.verbose: print(f"  [!] Hard killing straggler {tasks[task_idx].name} ({elapsed:.1f}s > {timeout}s)")
                        p.kill()
                        p.join()
                        finished_slots.append(slot)
                        # On-done with empty result?
                        from domains.arc.runner import TaskResult
                        empty_tr = TaskResult(
                            task_name=tasks[task_idx].name,
                            category=tasks[task_idx].name.split("_")[0],
                            true_op=tasks[task_idx].true_op,
                            found_expr="TIMEOUT",
                            train_acc=0.0,
                            test_acc=0.0,
                            solved=False,
                            near_solved=False,
                            n_nodes=0,
                            elapsed_s=elapsed,
                            n_evals=0,
                            introspection="Hard timeout reached. Killed process."
                        )
                        _on_done(task_idx, empty_tr, tasks[task_idx])

                for slot in finished_slots:
                    del active_processes[slot]
                    free_slots.append(slot)
                    shared_evals[slot] = 0

                time.sleep(0.1) # Avoid busy wait

        except KeyboardInterrupt:
            for slot, (p, _, _, _) in active_processes.items():
                p.kill()
                p.join()
            os._exit(130)
    else:
        for i, task in enumerate(tasks):
            scoreboard.start_times[i] = time.time()
            print(scoreboard.render(), flush=True)
            _, tr = _run_task_process(i, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops)
            _on_done(i, tr, task)

    ordered_results.sort(key=lambda x: x[0])
    report.results = [tr for _, tr in ordered_results]
    report.total_elapsed_s = time.time() - t0
    if cfg.verbose: print(report.summary())
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
