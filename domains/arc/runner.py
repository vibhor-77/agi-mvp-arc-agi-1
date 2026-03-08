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
import math
import multiprocessing as mp
import os
import pathlib
import resource
import subprocess
import sys
import time
import hashlib
import numpy as np
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
    lam: float = 0.05
    verbose: bool = True
    baseline_only: bool = False
    expanded_only: bool = False
    seed: int | None = None
    timeout_s: float | None = None  # Optional wall-clock safety (off by default for deterministic budgets)
    max_evals: int | None = 1000000  # 1M evaluations (Baseline Deterministic Limit)
    max_cost: int | None = None     # Total Pixel-Budget (if set, overrides max_evals)
    max_eval_cost: int | None = None # Per-evaluation Pixel-Budget
    mem_per_task_worker_gb: float = 3.0
    reserve_mem_gb: float = 10.0
    cpu_reserve: int = 2
    capture_traces: bool = False
    stall_kill_s: float | None = None
    adaptive_primitive_subset: bool = True
    primitive_cap: int = 80
    fail_on_timeout: bool = True
    progress_interval_s: float = 5.0
    progress_log_path: str | None = None
    max_rss_gb: float = 0.0
    profile_primitives: bool = False


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
    n_cost: int = 0
    introspection: str = ""
    best_tree: Optional[Node] = None
    trace: list | None = None
    primitive_hotspots: str = ""
    timed_out: bool = False
    worker_error: bool = False

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
            "n_cost": self.n_cost,
            "primitive_hotspots": self.primitive_hotspots,
            "timed_out": self.timed_out,
            "worker_error": self.worker_error,
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
        return sum(1 for r in self.results if r and r.solved)

    @property
    def n_near(self) -> int:
        return sum(1 for r in self.results if r and r.near_solved)

    @property
    def pct_solved(self) -> float:
        return 100 * self.n_solved / max(self.n_tasks, 1)

    @property
    def mean_test_acc(self) -> float:
        valid = [r for r in self.results if r]
        if not valid:
            return 0.0
        return sum(r.test_acc for r in valid) / len(valid)

    @property
    def total_evals(self) -> int:
        return sum(int(r.n_evals) for r in self.results if r)

    @property
    def total_cost(self) -> int:
        return sum(int(r.n_cost) for r in self.results if r)

    @property
    def solved_per_million_evals(self) -> float:
        te = self.total_evals
        if te <= 0:
            return 0.0
        return (self.n_solved * 1_000_000.0) / te

    @property
    def solved_per_million_cost(self) -> float:
        tc = self.total_cost
        if tc <= 0:
            return 0.0
        return (self.n_solved * 1_000_000.0) / tc

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
            f"  Total evals:      {self.total_evals/1000:.1f}k",
            f"  Solves / 1M eval: {self.solved_per_million_evals:.2f}",
            f"  Total cost units: {self.total_cost/1000:.1f}k",
            f"  Solves / 1M cost: {self.solved_per_million_cost:.2f}",
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
            "total_evals": self.total_evals,
            "total_cost": self.total_cost,
            "solves_per_million_evals": round(self.solved_per_million_evals, 2),
            "solves_per_million_cost": round(self.solved_per_million_cost, 2),
            "results": [r.as_dict() for r in self.results if r],
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
        lines.append(f"- **Total Evals**: {self.total_evals}")
        lines.append(f"- **Solves per 1M Evals**: {self.solved_per_million_evals:.2f}")
        lines.append(f"- **Total Cost Units**: {self.total_cost}")
        lines.append(f"- **Solves per 1M Cost Units**: {self.solved_per_million_cost:.2f}")
        lines.append(f"- **Total Time**: {self.total_elapsed_s:.1f}s")
        
        lines.append(f"\n## Performance by Category")
        cats = self.by_category()
        for cat, d in sorted(cats.items()):
            pct = 100 * d["solved"] / d["total"] if d["total"] > 0 else 0
            lines.append(f"- **{cat}**: {d['solved']}/{d['total']} ({pct:.1f}%)")
            
        lines.append(f"\n## Detailed Results (N={self.n_tasks})")
        lines.append("| Task | Status | Evals | Cost | Time | Train | Test | Expression |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in self.results:
            status = "✅ SOLVED" if r.solved else ("⚠️ NEAR" if r.near_solved else "❌ FAIL")
            expr = r.found_expr if len(r.found_expr) < 40 else r.found_expr[:37] + "..."
            lines.append(f"| {r.task_name} | {status} | {r.n_evals/1000:5.1f}k | {r.n_cost/1000:5.1f}k | {r.elapsed_s:.1f}s | {r.train_acc:.2f} | {r.test_acc:.2f} | `{expr}` |")
            
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
                    lines.append(f"  - **Best AST**: `{ast_str}` | {r.n_evals/1000:.1f}k evals | {r.n_cost/1000:.1f}k cost | {r.elapsed_s:.1f}s")
                if r.primitive_hotspots:
                    lines.append(f"  - **Hotspots**: {r.primitive_hotspots}")
                
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
_worker_shared_costs = None
_worker_idx = None


def _detect_total_memory_gb() -> float:
    """Best-effort RAM detection without external dependencies."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out) / (1024**3)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return 16.0

def _process_rss_gb() -> float:
    """Best-effort process RSS in GB."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux reports KiB.
    if sys.platform == "darwin":
        return rss / (1024**3)
    return (rss * 1024.0) / (1024**3)


def _stable_task_seed(task_name: str, base_seed: int | None) -> int:
    """Derive a per-task seed that is stable across task ordering and batch slicing."""
    base = 0 if base_seed is None else int(base_seed)
    digest = hashlib.blake2b(task_name.encode("utf-8"), digest_size=4).digest()
    task_seed = int.from_bytes(digest, byteorder="big", signed=False)
    return (base + task_seed) & 0x7FFFFFFF


def _recommend_task_workers(requested: int, cfg: BenchmarkConfig) -> int:
    """
    Choose a safe task-worker count that stays out of swap while using CPU well.
    """
    if requested <= 0:
        requested = os.cpu_count() or 1
    cpu_total = os.cpu_count() or 1
    cpu_cap = max(1, cpu_total - max(cfg.cpu_reserve, 0))

    total_mem_gb = _detect_total_memory_gb()
    mem_budget_gb = max(1.0, total_mem_gb - max(cfg.reserve_mem_gb, 0.0))
    mem_cap = max(1, int(math.floor(mem_budget_gb / max(cfg.mem_per_task_worker_gb, 0.25))))

    return max(1, min(requested, cpu_cap, mem_cap))

def _run_task_process(
    idx: int, 
    task: ARCTask, 
    cfg: BenchmarkConfig, 
    op_subset: list[str], 
    inner_workers: int,
    transition_matrix: dict[str, dict[str, float]] | None,
    learned_ops: dict[str, dict] | None = None,
    worker_idx: int = 0,
    progress_hook: Any | None = None,
    seed_programs: list[Node] | None = None,
) -> tuple[int, TaskResult]:
    global _worker_idx, _worker_shared_evals, _worker_shared_costs
    _worker_idx = worker_idx
    
    if learned_ops:
        from core.library import PrimitiveLibrary
        from core.tree import Node
        lib = PrimitiveLibrary()
        lib.learned_ops = {
            name: {
                "expr": meta.get("expr", ""),
                "arity": int(meta.get("arity", 1)),
                "node": Node.parse(meta["expr"]) if meta.get("expr") else None,
            }
            for name, meta in learned_ops.items()
            if meta.get("expr")
        }
        lib.register_all(domain="arc")
    
    def on_step(n_evals, elapsed, n_cost=None):
        if _worker_shared_evals is not None and _worker_idx is not None:
            _worker_shared_evals[_worker_idx] = n_evals
        if _worker_shared_costs is not None and _worker_idx is not None:
            _worker_shared_costs[_worker_idx] = 0 if n_cost is None else int(n_cost)
        if cfg.max_rss_gb and cfg.max_rss_gb > 0:
            rss_gb = _process_rss_gb()
            if rss_gb >= cfg.max_rss_gb:
                raise MemoryError(
                    f"RSS limit exceeded: {rss_gb:.2f}GB >= {cfg.max_rss_gb:.2f}GB"
                )
        if progress_hook is not None:
            try:
                progress_hook(n_evals, elapsed, 0 if n_cost is None else int(n_cost))
            except Exception:
                pass

    search_cfg = SearchConfig(
        beam_size=cfg.beam_size,
        offspring=cfg.offspring,
        generations=cfg.generations,
        workers=inner_workers,
        converge_threshold=1e-9,
        verbose=False,
        seed=_stable_task_seed(task.name, cfg.seed),
        timeout_s=cfg.timeout_s,
        max_evals=cfg.max_evals,
        max_cost=cfg.max_cost,
        max_eval_cost=cfg.max_eval_cost,
    )

    task_t0 = time.time()
    selected_ops = op_subset
    if cfg.adaptive_primitive_subset:
        try:
            from domains.arc.domain import select_primitives_for_task
            selected_ops = select_primitives_for_task(task, op_subset, max_ops=cfg.primitive_cap)
        except Exception:
            selected_ops = op_subset
    domain = ARCDomain(
        task,
        lam=cfg.lam,
        primitive_subset=selected_ops,
        seed_programs=seed_programs,
        profile_primitives=cfg.profile_primitives,
        max_eval_cost=search_cfg.max_eval_cost,
    )
    domain.on_result = lambda x: None  # type: ignore[method-assign]
    
    result = domain.solve(config=search_cfg, transition_matrix=transition_matrix, on_step=on_step)

    tree      = result.best_tree
    train_acc = domain.train_accuracy(tree)
    test_acc  = domain.test_accuracy(tree)
    solved    = domain.check_solution(tree)
    
    # LOCAL REFINEMENT: Try to fix near-misses (80%+) on top candidates
    if not solved and result.top_candidates:
        # Refine top 2 (usually winner + runner-up)
        for _, cand_tree, _, _ in result.top_candidates[:2]:
            cand_train_acc = domain.train_accuracy(cand_tree)
            if cand_train_acc >= 0.80:
                refined_tree = domain.super_refine(cand_tree)
                refined_train_acc = domain.train_accuracy(refined_tree)
                if refined_train_acc > train_acc:
                    tree = refined_tree
                    train_acc = refined_train_acc
                    test_acc = domain.test_accuracy(tree)
                    solved = domain.check_solution(tree)
                    # Update result
                    result.best_tree = tree
                    result.best_fitness = domain.evaluate_candidate(tree)[0]
                    result.solved = solved
                    if solved: break

    near      = test_acc >= 0.80
    elapsed   = time.time() - task_t0
    timed_out = (
        cfg.timeout_s is not None
        and elapsed >= cfg.timeout_s
        and (cfg.max_evals is None or result.n_evals < cfg.max_evals)
        and (cfg.max_cost is None or result.n_cost < cfg.max_cost)
    )

    introspection_msg = ""
    if not solved and tree is not None:
        try:
            # We must re-evaluate the best tree on the first train instance to generate the diff heuristic
            pair = task.train_pairs[0]
            if len(pair) == 2:
                inp, expected = pair[0], pair[1]
                if cfg.capture_traces:
                    actual, trace = tree.eval_trace([inp], domain._primitives)
                else:
                    actual = tree.eval([inp], domain._primitives)
                    trace = None
                
                if isinstance(actual, (list, np.ndarray)) and len(actual) > 0:
                    # Uniformize for inspection
                    actual_np = np.asarray(actual)
                    expected_np = np.asarray(expected)
                    if len(expected) != len(actual) or len(expected[0]) != len(actual[0]):
                        mismatches_msg = f"Dimension mismatch: Expected {len(expected)}x{len(expected[0])}, Actual {len(actual)}x{len(actual[0])}"
                    else:
                        mismatches = 0
                        total = len(expected) * len(expected[0])
                        for r in range(len(expected)):
                            for c in range(len(expected[0])):
                                if expected[r][c] != actual[r][c]:
                                    mismatches += 1
                        mismatches_msg = f"Pixel mismatch: {mismatches}/{total} pixels incorrect"
                    
                    # Merge with Domain's own introspection (Complexity Trace)
                    try:
                        _, _, _, domain_intro, _ = domain.evaluate_candidate(tree)
                        introspection_msg = f"{mismatches_msg} ({domain_intro})"
                    except Exception:
                        introspection_msg = mismatches_msg
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
        n_cost=result.n_cost,
        introspection=introspection_msg,
        best_tree=tree,
        trace=trace if 'trace' in locals() else None,
        primitive_hotspots=(
            " | ".join(
                f"{name}:{calls}c/{secs:.3f}s"
                for name, calls, secs in domain.primitive_runtime_top(5)
            )
            if cfg.profile_primitives
            else ""
        ),
        timed_out=timed_out,
        worker_error=False,
    )
    return idx, tr

def _worker_wrapper(
    idx,
    task,
    cfg,
    op_subset,
    inner_workers,
    transition_matrix,
    learned_ops,
    slot,
    conn,
    shared_evals,
    shared_costs,
    seed_programs=None,
):
    """Top-level wrapper for multiprocessing processes."""
    global _worker_shared_evals, _worker_shared_costs
    _worker_shared_evals = shared_evals
    _worker_shared_costs = shared_costs
    try:
        from domains.arc.runner import _run_task_process
        _, tr = _run_task_process(
            idx, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops, slot, seed_programs=seed_programs
        )
        conn.send(tr)
    except Exception as e:
        conn.send(None)
    finally:
        conn.close()

class LiveScoreboard:
    """Encapsulates the status reporting logic for evaluate_tasks."""
    def __init__(self, n_total: int, t0: float, task_workers: int, epoch_str: str = "", global_stats: dict | None = None):
        self.n_total = n_total
        self.t0 = t0
        self.task_workers = task_workers
        self.epoch_str = epoch_str
        self.global_stats = global_stats or {}
        self.counters = {
            "solved": 0, "near": 0, "done": 0,
            "solved_time": 0.0, "near_time": 0.0, "unsolved_time": 0.0,
            "total_evals": 0, "solved_evals": 0, "near_evals": 0, "unsolved_evals": 0,
            "total_cost": 0, "solved_cost": 0, "near_cost": 0, "unsolved_cost": 0,
        }
        self.start_times: dict[int, float] = {}
        self.shared_evals: Any = None

    def update(self, tr: TaskResult):
        self.counters["done"] += 1
        self.counters["total_evals"] += tr.n_evals
        self.counters["total_cost"] += tr.n_cost
        if tr.solved:
            self.counters["solved"] += 1
            self.counters["solved_time"] += tr.elapsed_s
            self.counters["solved_evals"] += tr.n_evals
            self.counters["solved_cost"] += tr.n_cost
        else:
            self.counters["unsolved_time"] += tr.elapsed_s
            self.counters["unsolved_evals"] += tr.n_evals
            self.counters["unsolved_cost"] += tr.n_cost
            if tr.near_solved:
                self.counters["near"] += 1
                self.counters["near_time"] += tr.elapsed_s
                self.counters["near_evals"] += tr.n_evals
                self.counters["near_cost"] += tr.n_cost

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
        total_cost = int(c["total_cost"])
        avg_work_e = total_evals / done if done > 0 else 0.0
        evals_p_s = total_evals / elapsed if elapsed > 0 else 0.0
        avg_work_c = total_cost / done if done > 0 else 0.0
        cost_p_s = total_cost / elapsed if elapsed > 0 else 0.0
        
        # Efficiency metrics
        speedup = task_total_t / elapsed if elapsed > 0 else 0.0
        utilization = 100 * (speedup / self.task_workers) if self.task_workers > 0 else 0.0
        pct = 100 * sol / self.n_total if self.n_total > 0 else 0.0

        # Stragglers
        running = [time.time() - t for t in self.start_times.values()]
        max_run_t = max(running) if running else 0.0
        max_run_e = max(self.shared_evals) if self.shared_evals else 0

        prefix = f" [{self.epoch_str}]" if self.epoch_str else ""
        
        # If global stats provided, make main header global-aware for better intuition
        disp_sol = sol
        disp_near = near
        disp_done = done
        disp_total = self.n_total
        disp_failed = unsol
        disp_pct = pct

        if self.global_stats:
            gs = self.global_stats
            g_offset = gs.get("offset", 0)
            disp_total = gs.get("global_total", self.n_total)
            disp_sol = gs.get("global_solved", 0) + sol
            disp_near = gs.get("global_near", 0) + near
            disp_done = g_offset + done
            disp_failed = disp_done - disp_sol
            disp_pct = 100 * disp_sol / max(disp_total, 1)

        main_scoreboard = (
            f"  ┌ scoreboard{prefix} ─\n"
            f"  │ ✓ solved={disp_sol} ({100*disp_sol/max(disp_done,1):.1f}%)  ⚠️ near={disp_near}  ✗ failed={disp_failed}  ⏳ pending={pend}\n"
            f"  │ → active={act}  done={disp_done}/{disp_total}  success={disp_pct:.1f}%\n"
            f"  │ TIME:  elapsed={elapsed:.1f}s (Throughput: {elapsed/max(done,1):.2f}s/task | Latency Avg: {avg_work_t:.1f}s)\n"
            f"  │ WORK:  speedup={speedup:.2f}x ({utilization:.1f}% core) | STRAGGLER: {max_run_t:.1f}s, {max_run_e/1000:.1f}k evals\n"
            f"  │ EVALS: total={total_evals/1000:.1f}k ({evals_p_s/1000:.2f}k/s | Per-Task Avg: {avg_work_e/1000:.1f}k)\n"
            f"  │ COST:  total={total_cost/1000:.1f}k ({cost_p_s/1000:.2f}k/s | Per-Task Avg: {avg_work_c/1000:.1f}k)"
        )

        if self.global_stats:
            # Add a local pass line instead of a global line at the bottom
            main_scoreboard += f"\n  │ PASS:  done={done}/{self.n_total} | solved={sol} | near={near} | success={pct:.1f}%"

        return main_scoreboard

def evaluate_tasks(
    tasks: list[ARCTask],
    op_subset: list[str],
    cfg: BenchmarkConfig,
    label: str,
    transition_matrix: dict[str, dict[str, float]] | None = None,
    learned_ops: dict[str, dict] | None = None,
    epoch_str: str = "",
    report_callback=None,
    global_stats: dict | None = None,
) -> BenchmarkReport:
    report = BenchmarkReport(label=label, n_ops=len(op_subset))
    t0 = time.time()
    effective_task_workers = _recommend_task_workers(cfg.task_workers, cfg)
    if cfg.verbose and effective_task_workers != cfg.task_workers:
        total_mem = _detect_total_memory_gb()
        print(
            f"  [ResourceGuard] task-workers adjusted {cfg.task_workers} -> {effective_task_workers} "
            f"(cpu={os.cpu_count() or 1}, ram={total_mem:.1f}GB, reserve={cfg.reserve_mem_gb:.1f}GB, "
            f"per-worker={cfg.mem_per_task_worker_gb:.1f}GB)"
        )
    cfg.task_workers = effective_task_workers
    inner_workers = 1 if cfg.task_workers > 1 else cfg.workers

    scoreboard = LiveScoreboard(len(tasks), t0, cfg.task_workers, epoch_str, global_stats=global_stats)
    ordered_results: list[tuple[int, TaskResult]] = []
    last_report_t = 0.0
    last_progress_t = 0.0

    shared_evals = mp.RawArray('i', cfg.task_workers)
    shared_costs = mp.RawArray('q', cfg.task_workers)
    scoreboard.shared_evals = shared_evals

    def _emit_progress(reason: str, force: bool = False):
        nonlocal last_progress_t, last_report_t
        now = time.time()
        if (not force) and (now - last_progress_t < max(0.5, cfg.progress_interval_s)):
            return
        done = int(scoreboard.counters["done"])
        active = min(cfg.task_workers, len(tasks) - done)
        active_evals = [int(v) for v in shared_evals]
        active_costs = [int(v) for v in shared_costs]
        event = {
            "t": round(now - t0, 2),
            "reason": reason,
            "done": done,
            "total": len(tasks),
            "active": active,
            "solved": int(scoreboard.counters["solved"]),
            "near": int(scoreboard.counters["near"]),
            "total_evals": int(scoreboard.counters["total_evals"]),
            "total_cost": int(scoreboard.counters["total_cost"]),
            "inflight_evals": int(sum(active_evals)),
            "inflight_cost": int(sum(active_costs)),
            "active_evals": active_evals,
            "active_costs": active_costs,
        }
        if cfg.verbose:
            print(
                f"  [HB] t={event['t']:.1f}s done={done}/{len(tasks)} active={active} "
                f"solved={event['solved']} near={event['near']} evals={event['total_evals']/1000:.1f}k "
                f"cost={event['total_cost']/1000:.1f}k inflight={event['inflight_evals']/1000:.1f}k evals "
                f"{event['inflight_cost']/1000:.1f}k cost active_evals={event['active_evals']}",
                flush=True,
            )
        if cfg.progress_log_path:
            try:
                log_dir = os.path.dirname(cfg.progress_log_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(cfg.progress_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=True) + "\n")
            except Exception:
                pass
        if report_callback and ((now - last_report_t > 10.0) or force):
            report.results = [r for _, r in sorted(ordered_results, key=lambda x: x[0])]
            report.total_elapsed_s = now - t0
            report_callback(report)
            last_report_t = now
        last_progress_t = now

    def _on_done(idx, tr, task):
        nonlocal last_report_t
        if idx in scoreboard.start_times: del scoreboard.start_times[idx]
        scoreboard.update(tr)
        ordered_results.append((idx, tr))
        if cfg.verbose:
            status = "✓" if tr.solved else ("~" if tr.near_solved else "✗")
            print(
                f"  {status} [{idx+1:3d}/{len(tasks)}] DONE {task.name[:28]:28s} "
                f"{tr.n_evals/1000:5.1f}k evals {tr.n_cost/1000:5.1f}k cost ({tr.elapsed_s:.1f}s)"
            )
            print(scoreboard.render(), flush=True)

        if report_callback:
            now = time.time()
            if (scoreboard.counters["done"] % 5 == 0) or (now - last_report_t > 15) or (scoreboard.counters["done"] == len(tasks)):
                report.results = [r for _, r in sorted(ordered_results, key=lambda x: x[0])]
                report.total_elapsed_s = now - t0
                report_callback(report)
                last_report_t = now
        _emit_progress("task_done", force=False)

    if cfg.task_workers > 1:
        def _worker_error(task: ARCTask, elapsed_s: float, timed_out: bool = False) -> TaskResult:
            return TaskResult(
                task_name=task.name,
                category=task.name.split("_")[0],
                true_op=task.true_op,
                found_expr="TIMEOUT" if timed_out else "WORKER_ERROR",
                train_acc=0.0,
                test_acc=0.0,
                solved=False,
                near_solved=False,
                n_nodes=0,
                elapsed_s=elapsed_s,
                n_evals=0,
                n_cost=0,
                introspection="Worker timed out." if timed_out else "Worker failed to return a result.",
                timed_out=timed_out,
                worker_error=(not timed_out),
            )

        active_processes: dict[int, tuple[Any, int, float, Any, int, float]] = {}
        task_queue = list(enumerate(tasks))
        free_slots = list(range(cfg.task_workers))

        # Culture buffer (Persistent memory for compounding)
        culture_exprs: set[str] = set()
        culture_seeds: list[Node] = []

        try:
            while task_queue or active_processes:
                while free_slots and task_queue:
                    slot = free_slots.pop(0)
                    task_idx, task = task_queue.pop(0)
                    if cfg.verbose:
                        print(f"  → [{task_idx+1:3d}] STARTING  {task.name}", flush=True)
                    parent_conn, child_conn = mp.Pipe()
                    
                    # Pass the CURRENT culture seeds to the worker
                    current_seeds = [s.clone() for s in culture_seeds]
                    
                    p = mp.Process(
                        target=_worker_wrapper,
                        args=(
                            task_idx,
                            task,
                            cfg,
                            op_subset,
                            inner_workers,
                            transition_matrix,
                            learned_ops,
                            slot,
                            child_conn,
                            shared_evals,
                            shared_costs,
                            current_seeds,
                        ),
                    )
                    p.start()
                    now = time.time()
                    active_processes[slot] = (p, task_idx, now, parent_conn, 0, now)
                    scoreboard.start_times[task_idx] = now
                    _emit_progress("task_start", force=True)

                finished_slots: list[int] = []
                for slot, (p, task_idx, start_t, conn, last_eval, last_progress_t) in list(active_processes.items()):
                    now = time.time()
                    elapsed = now - start_t
                    current_eval = int(shared_evals[slot]) if slot < len(shared_evals) else 0
                    if current_eval > last_eval:
                        last_eval = current_eval
                        last_progress_t = now
                        active_processes[slot] = (p, task_idx, start_t, conn, last_eval, last_progress_t)

                    if conn.poll():
                        tr = None
                        try:
                            tr = conn.recv()
                        except EOFError:
                            tr = None
                        if tr is None:
                            tr = _worker_error(tasks[task_idx], elapsed, timed_out=False)
                        
                        # CUMULATIVE CULTURE: If solved, add to future seed programs
                        if tr.solved and tr.best_tree:
                            expr = str(tr.best_tree)
                            if expr not in culture_exprs:
                                culture_exprs.add(expr)
                                culture_seeds.append(tr.best_tree.clone())
                                if cfg.verbose:
                                    print(f"  [Culture] New Pattern Discovered: {expr[:60]}... (Total Seeds: {len(culture_seeds)})")

                        _on_done(task_idx, tr, tasks[task_idx])
                        p.join()
                        finished_slots.append(slot)
                    elif not p.is_alive():
                        p.join()
                        _on_done(task_idx, _worker_error(tasks[task_idx], elapsed, timed_out=False), tasks[task_idx])
                        finished_slots.append(slot)
                    elif cfg.timeout_s is not None and elapsed >= cfg.timeout_s:
                        if cfg.verbose:
                            print(
                                f"  [!] Timeout {tasks[task_idx].name} "
                                f"(elapsed={elapsed:.1f}s, evals={current_eval})",
                                flush=True,
                            )
                        p.kill()
                        p.join()
                        _on_done(task_idx, _worker_error(tasks[task_idx], elapsed, timed_out=True), tasks[task_idx])
                        finished_slots.append(slot)

                for slot in finished_slots:
                    del active_processes[slot]
                    free_slots.append(slot)
                    shared_evals[slot] = 0
                    shared_costs[slot] = 0

                _emit_progress("heartbeat", force=False)
                time.sleep(0.05)

        except KeyboardInterrupt:
            for slot, (p, _, _, _, _, _) in active_processes.items():
                p.kill()
                p.join()
            os._exit(130)
    else:
        for i, task in enumerate(tasks):
            scoreboard.start_times[i] = time.time()
            print(scoreboard.render(), flush=True)
            _emit_progress("task_start", force=True)
            last_task_hb = time.time()

            def _single_progress(n_evals: int, elapsed_s: float, n_cost: int):
                nonlocal last_task_hb
                shared_evals[0] = int(n_evals)
                shared_costs[0] = int(n_cost)
                now = time.time()
                if now - last_task_hb >= max(0.5, cfg.progress_interval_s):
                    if cfg.verbose:
                        print(
                            f"  [HB] task={task.name} evals={n_evals/1000:.1f}k cost={n_cost/1000:.1f}k "
                            f"elapsed={elapsed_s:.1f}s",
                            flush=True,
                        )
                    _emit_progress("single_task_heartbeat", force=False)
                    last_task_hb = now

            _, tr = _run_task_process(
                i, task, cfg, op_subset, inner_workers, transition_matrix, learned_ops, progress_hook=_single_progress
            )
            _on_done(i, tr, task)
            shared_evals[0] = 0
            shared_costs[0] = 0

    ordered_results.sort(key=lambda x: x[0])
    report.results = [tr for _, tr in ordered_results]
    report.total_elapsed_s = time.time() - t0
    _emit_progress("final", force=True)
    hard_failures = [r for r in report.results if r.timed_out or r.worker_error]
    if hard_failures and cfg.fail_on_timeout:
        failed_names = ", ".join(r.task_name for r in hard_failures[:10])
        if len(hard_failures) > 10:
            failed_names += f", ... (+{len(hard_failures)-10} more)"
        raise RuntimeError(
            f"Hard worker failures detected ({len(hard_failures)} tasks): {failed_names}. "
            "Timeout/worker-error is treated as a bug by configuration."
        )
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

    baseline: BenchmarkReport | None = None
    if not cfg.expanded_only:
        print(f"\n{'='*65}")
        print(f"  BASELINE ({len(baseline_ops)} ops)")
        print(f"{'='*65}")
        baseline = evaluate_tasks(tasks, baseline_ops, cfg, f"Baseline ({len(baseline_ops)} ops)")
    else:
        # Create a dummy report so downstream logic doesn't crash
        baseline = BenchmarkReport(label="Skipped", n_ops=len(baseline_ops))
        baseline.results = [None] * len(tasks)

    expanded: BenchmarkReport | None = None
    if not cfg.baseline_only:
        print(f"\n{'='*65}")
        print(f"  EXPANDED DSL ({len(expanded_ops)} ops)")
        print(f"{'='*65}")
        expanded = evaluate_tasks(tasks, expanded_ops, cfg, f"Expanded ({len(expanded_ops)} ops)")

    if expanded and not cfg.baseline_only and not cfg.expanded_only:
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
    parser.add_argument("--expanded-only",action="store_true", help="Run expanded DSL only")
    parser.add_argument("--workers",      type=int, default=1,
                        help="Beam-search candidate workers per task (default 1). "
                             "Auto-forced to 1 when --task-workers > 1.")
    parser.add_argument("--task-workers", type=int, default=0,
                        help="Tasks to run in parallel. 0 = auto resource-safe mode.")
    parser.add_argument("--mem-per-task-worker-gb", type=float, default=3.0,
                        help="Estimated RAM consumed per task worker (used for auto-capping).")
    parser.add_argument("--reserve-mem-gb", type=float, default=10.0,
                        help="RAM reserve to keep free to avoid swap.")
    parser.add_argument("--cpu-reserve", type=int, default=2,
                        help="Number of CPU threads to keep free.")
    parser.add_argument("--capture-traces", action="store_true",
                        help="Capture AST execution traces in results (increases memory use).")
    parser.add_argument("--profile-primitives", action="store_true",
                        help="Profile primitive runtime per task (adds overhead).")
    parser.add_argument("--stall-kill-s", type=float, default=None,
                        help="Optional watchdog: kill a task worker if eval count makes no progress for this many seconds.")
    parser.add_argument("--adaptive-primitive-subset", action="store_true",
                        help="Enable per-task primitive subset selection (default on).")
    parser.add_argument("--no-adaptive-primitive-subset", action="store_true",
                        help="Disable per-task primitive subset selection.")
    parser.add_argument("--primitive-cap", type=int, default=80,
                        help="Max primitive count per task when adaptive subset is enabled.")
    parser.add_argument("--generations",  type=int, default=100, help="Override generations")
    parser.add_argument("--max-evals", type=int, default=1000000, help="Evaluation budget per task.")
    parser.add_argument("--max-cost", type=int, default=0, help="Cost-unit budget per task (0=off).")
    parser.add_argument("--timeout", type=float, default=0.0, help="Per-task timeout seconds (0=off).")
    parser.add_argument("--progress-interval-s", type=float, default=5.0, help="Progress heartbeat interval.")
    parser.add_argument("--progress-log", type=str, default=None, help="JSONL progress log path.")
    parser.add_argument("--max-rss-gb", type=float, default=0.0, help="Abort if process RSS reaches this limit (0=off).")
    parser.add_argument("--tasks",        type=str, default=None, help="Limit number of tasks or path to ID list")
    parser.add_argument("--beam-size",     type=int, default=10, help="Symbolic beam width.")
    parser.add_argument("--save",         type=str, default="results.json", help="Output JSON path")
    args = parser.parse_args()

    cfg = BenchmarkConfig(
        beam_size    = args.beam_size,
        offspring    = 20,
        generations  = args.generations,
        workers      = args.workers,
        task_workers = args.task_workers,
        verbose      = True,
        baseline_only = args.baseline_only,
        expanded_only = args.expanded_only,
        seed         = None,
        timeout_s    = (None if args.timeout <= 0 else args.timeout),
        max_evals    = args.max_evals,
        max_cost     = (None if args.max_cost <= 0 else args.max_cost),
        mem_per_task_worker_gb=args.mem_per_task_worker_gb,
        reserve_mem_gb=args.reserve_mem_gb,
        cpu_reserve=args.cpu_reserve,
        capture_traces=args.capture_traces,
        profile_primitives=args.profile_primitives,
        stall_kill_s=args.stall_kill_s,
        adaptive_primitive_subset=False if args.no_adaptive_primitive_subset else True,
        primitive_cap=args.primitive_cap,
        progress_interval_s=args.progress_interval_s,
        progress_log_path=args.progress_log,
        max_rss_gb=args.max_rss_gb,
    )

    if args.data:
        print(f"Loading real ARC tasks from: {args.data}")
        tasks = load_tasks_from_dir(args.data)
        print(f"Loaded {len(tasks)} tasks.")
    else:
        print("No --data path given. Using built-in 76-task programmatic benchmark.")
        tasks = build_benchmark()

    if args.tasks:
        import os
        if os.path.exists(str(args.tasks)):
            with open(str(args.tasks), "r") as f:
                ids = [line.strip() for line in f if line.strip()]
            tasks = [t for t in tasks if t.name in ids]
            print(f"Filtered to {len(tasks)} tasks from file {args.tasks}.")
        else:
            try:
                limit = int(args.tasks)
                tasks = tasks[:limit]
                print(f"Limited to first {limit} tasks.")
            except ValueError:
                pass

    run_benchmark(tasks, cfg, save_path=args.save)
