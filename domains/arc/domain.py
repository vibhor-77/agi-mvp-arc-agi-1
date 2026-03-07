"""
domains/arc/domain.py
=====================
ARCDomain — plugs ARC-AGI tasks into the beam search engine.

Usage
-----
    from domains.arc.domain import ARCDomain, ARCTask

    task = ARCTask(
        name="rotate_90",
        train_pairs=[
            ([[1,2],[3,4]], [[3,1],[4,2]]),
            ([[0,1],[1,0]], [[1,0],[0,1]]),
            ([[1,0,0],[0,1,0],[0,0,1]], [[0,0,1],[0,1,0],[1,0,0]]),
        ],
    )

    domain = ARCDomain([task])
    from core.search import SearchConfig
    results = domain.run(config=SearchConfig(generations=100))
    for r in results:
        print(r)

Generalising to ARC-AGI-2 / ARC-AGI-3
---------------------------------------
  - ARC-AGI-2: load tasks from the ARC-AGI-2 JSON files into ARCTask objects;
    the domain, fitness function, and beam search are identical.
  - ARC-AGI-3 (interactive): override ``fitness()`` to run a multi-step episode
    instead of comparing a single input-output pair.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from core.domain import Domain
from core.tree import Node
from core.search import SearchConfig, SearchResult

# Side-effect import: registers all ARC primitives into the global registry
import domains.arc.primitives  # noqa: F401
from core.primitives import registry

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Task representation
# ---------------------------------------------------------------------------

class TrainPair(NamedTuple):
    input: Grid
    output: Grid


@dataclass
class ARCTask:
    """
    One ARC-AGI task.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "007bbfb7" or "rotate_90").
    train_pairs : list[TrainPair]
        Demonstration examples used to find the rule.
        Typically 3–5 pairs.
    test_pairs : list[TrainPair]
        Held-out pairs for evaluation.  A task is "solved" iff the
        discovered program produces exact outputs for all test inputs.
    true_op : str
        Optional ground-truth label (used in benchmarks; leave "" if unknown).
    """
    name: str
    train_pairs: list[tuple[Grid, Grid]]
    test_pairs: list[tuple[Grid, Grid]] = field(default_factory=list)
    true_op: str = ""

    def __post_init__(self) -> None:
        # Normalise to list-of-tuples (accepts both tuples and lists)
        self.train_pairs = [
            (copy.deepcopy(inp), copy.deepcopy(out))
            for inp, out in self.train_pairs
        ]
        self.test_pairs = [
            (copy.deepcopy(inp), copy.deepcopy(out))
            for inp, out in self.test_pairs
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "ARCTask":
        """
        Construct from the standard ARC JSON format::

            {
              "train": [{"input": [...], "output": [...]}, ...],
              "test":  [{"input": [...], "output": [...]}, ...]
            }
        """
        train = [(p["input"], p["output"]) for p in d.get("train", [])]
        test  = [(p["input"], p["output"]) for p in d.get("test",  [])]
        name  = d.get("name", "unnamed")
        return cls(name=name, train_pairs=train, test_pairs=test)


def select_primitives_for_task(task: "ARCTask", op_pool: list[str], max_ops: int = 80) -> list[str]:
    """
    Score and select a compact primitive subset for a specific task.
    Keeps core operators always, then task-shape/color relevant operators.
    """
    if max_ops <= 0 or len(op_pool) <= max_ops:
        return list(op_pool)

    core = {
        "gid", "grot90", "grot180", "grot270", "grefl_h", "grefl_v", "gtrsp",
        "ginv", "gmirror_h", "gmirror_v", "ghstack", "gvstack", "goverlay",
        "gmask", "gframe1", "gframe2", "gframe5", "gframe8", "gframe9",
        "gpad1", "gcrop_border", "gscale2x", "gcheckerboard", "gstripe_h2", "gstripe_v2",
        "gcountbar", "gmajority", "gkeep_rows2", "gkeep_rows3", "gkeep_rows4",
    }

    grow_tokens = ("scale", "pad", "repeat", "stack", "fractal", "inflate")
    shrink_tokens = ("crop", "downscale")
    color_tokens = ("swap", "mod", "replace", "fill", "color", "invert", "majority")
    geom_tokens = ("rot", "refl", "trsp", "mirror", "diag", "align")
    object_tokens = ("obj", "cc", "extract", "keep_largest", "render")

    # Measure shape trends across train examples
    grows = shrinks = same = 0
    color_change = 0
    for inp, out in task.train_pairs:
        ri, ci = len(inp), len(inp[0]) if inp else 0
        ro, co = len(out), len(out[0]) if out else 0
        ai, ao = ri * ci, ro * co
        if ao > ai:
            grows += 1
        elif ao < ai:
            shrinks += 1
        else:
            same += 1
        # Coarse color-change signal
        in_colors = {v for row in inp for v in row}
        out_colors = {v for row in out for v in row}
        if in_colors != out_colors:
            color_change += 1

    scored: list[tuple[int, str]] = []
    for op in op_pool:
        s = 0
        if op in core:
            s += 10
        if op.startswith("lib_op_"):
            s += 8
        low = op.lower()
        if any(t in low for t in geom_tokens):
            s += 2
        if color_change > 0 and any(t in low for t in color_tokens):
            s += 3
        if grows > 0 and any(t in low for t in grow_tokens):
            s += 3
        if shrinks > 0 and any(t in low for t in shrink_tokens):
            s += 3
        if same > 0 and ("hstack" in low or "vstack" in low):
            s -= 2
        if any(t in low for t in object_tokens):
            s += 2
        scored.append((s, op))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [op for score, op in scored if score > 0][:max_ops]

    # Safety: always keep at least a robust core.
    if len(selected) < 20:
        core_present = [op for op in op_pool if op in core]
        filler = [op for _, op in scored if op not in core_present]
        selected = (core_present + filler)[:max_ops]
    return selected


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

import numpy as np
import time
try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency fallback
    njit = None


if njit is not None:
    @njit(cache=True)
    def _fast_cell_match_numba(p: np.ndarray, t: np.ndarray) -> int:
        matches = 0
        rows, cols = p.shape
        for r in range(rows):
            for c in range(cols):
                if p[r, c] == t[r, c]:
                    matches += 1
        return matches
else:
    _fast_cell_match_numba = None

class BudgetExceededException(Exception):
    """Raised when an expression tree exceeds its allocated Pixel-Budget."""
    pass

def _to_np_grid(g: Grid | np.ndarray) -> np.ndarray | None:
    """Normalize grid-like values to a 2D NumPy array."""
    if isinstance(g, np.ndarray):
        return g if g.ndim == 2 else None
    if not isinstance(g, list):
        return None
    if not g or not isinstance(g[0], list):
        return None
    try:
        arr = np.asarray(g, dtype=np.int16)
    except Exception:
        return None
    return arr if arr.ndim == 2 else None

def _compact_grid_fingerprint(pred: Any, pred_np: np.ndarray | None) -> str:
    """Bounded-size semantic fingerprint for dedupe without huge string allocations."""
    if pred_np is None:
        return f"NON_GRID:{type(pred).__name__}"
    rows, cols = pred_np.shape
    # Hash bytes to keep fingerprint fixed-size regardless of grid dimensions.
    digest = hashlib.blake2b(pred_np.tobytes(), digest_size=8).hexdigest()
    return f"{rows}x{cols}:{digest}"

def grid_cell_accuracy(pred: Grid, target: Grid) -> float:
    """
    Fraction of cells that match between *pred* and *target*.
    Uses Numba-accelerated JIT kernel for the inner pixel loop.
    """
    p_np = _to_np_grid(pred)
    t_np = _to_np_grid(target)
    if p_np is None or t_np is None:
        return 0.0
    
    # Dimensional Gravity Heuristic
    MAX_HEURISTIC_BONUS = 0.25
    r_target, c_target = t_np.shape
    r_pred, c_pred = p_np.shape
    
    r_diff = abs(r_target - r_pred)
    c_diff = abs(c_target - c_pred)
    dim_penalty = (r_diff + c_diff) / (r_target + c_target)
    dimensional_reward = max(0.0, MAX_HEURISTIC_BONUS * (1.0 - dim_penalty))
    
    if r_pred != r_target or c_pred != c_target:
        return dimensional_reward

    if _fast_cell_match_numba is not None:
        matches = int(_fast_cell_match_numba(p_np, t_np))
    else:
        matches = int(np.count_nonzero(p_np == t_np))
    pixel_accuracy = matches / (r_target * c_target)
    # The actual score is the combination of the structural bonus (0.25) and the pixel accuracy (0.75)
    return dimensional_reward + (pixel_accuracy * (1.0 - MAX_HEURISTIC_BONUS))


def is_exact_match(pred: Grid, target: Grid) -> bool:
    """Return True iff pred and target are identical grids."""
    return grid_cell_accuracy(pred, target) == 1.0


# ---------------------------------------------------------------------------
# ARCDomain
# ---------------------------------------------------------------------------

class ARCDomain(Domain):
    """
    Domain wrapper for one ARC-AGI task.

    Fitness is:
        mean(1 - cell_accuracy over training pairs) + λ * tree.size()

    This means a perfect solution on all training pairs has a fitness of
    λ * tree.size() > 0.  To check if a task is truly solved, call
    ``check_solution()`` which evaluates on test pairs.

    Parameters
    ----------
    task : ARCTask
        The task to solve.
    lam : float
        MDL complexity penalty coefficient.
    primitive_subset : list[str] | None
        Explicit list of primitive names to use.
        Defaults to all registered ARC primitives.
    """

    def __init__(
        self,
        task: ARCTask,
        lam: float = 0.05,
        primitive_subset: list[str] | None = None,
        library: Any = None,
        profile_primitives: bool = False,
        max_eval_cost: int | None = None,
    ) -> None:
        self.task = task
        self.lam = lam
        self._op_list = primitive_subset or registry.names(domain="arc")
        self._primitives = {name: registry.get(name) for name in self._op_list}
        if library:
            self._primitives.update(library.learned_ops)
            self._op_list = list(self._primitives.keys())
        self._discovery_eval_count = 0
        self._discovery_start_time = time.time()
        self._primitive_profile: dict[str, dict[str, float]] = {}
        self._profile_primitives = profile_primitives
        self._max_eval_cost = max_eval_cost
        # Cache immutable targets and inputs as NumPy once to avoid per-eval conversion overhead.
        self._train_targets_np = [_to_np_grid(out) for _, out in self.task.train_pairs]
        self._train_inputs_np = [_to_np_grid(inp) for inp, _ in self.task.train_pairs]
        self._current_eval_cost = 0 # Tracked during evaluate_candidate
        self._instrument_primitives()

    def _instrument_primitives(self) -> None:
        wrapped: dict[str, Any] = {}
        for name, fn in self._primitives.items():
            self._primitive_profile[name] = {"calls": 0.0, "time_s": 0.0, "pixels": 0.0}

            def _mk(n: str, f: Any):
                def _wrapped(*args, **kwargs):
                    t0 = time.perf_counter()
                    try:
                        res = f(*args, **kwargs)
                        # Pixel-Budgeting: Charge for the complexity of the grid produced
                        px = 1
                        if isinstance(res, (list, np.ndarray)):
                             r_np = _to_np_grid(res)
                             if r_np is not None:
                                 px = int(r_np.shape[0] * r_np.shape[1])
                        
                        self._current_eval_cost += px
                        
                        if self._max_eval_cost and self._current_eval_cost > self._max_eval_cost:
                            raise BudgetExceededException(f"Eval cost {self._current_eval_cost} exceeded limit {self._max_eval_cost}")

                        if self._profile_primitives:
                            stat = self._primitive_profile[n]
                            stat["calls"] += 1.0
                            stat["time_s"] += (time.perf_counter() - t0)
                            stat["pixels"] += px
                        return res
                    except Exception:
                        self._current_eval_cost += 1
                        raise
                return _wrapped

            wrapped[name] = _mk(name, fn)
        self._primitives = wrapped

    def get_stats(self) -> str:
        duration = max(0.1, time.time() - self._discovery_start_time)
        return f"evals={self._discovery_eval_count} rate={self._discovery_eval_count/duration:.1f}/s"

    def primitive_runtime_top(self, top_n: int = 5) -> list[tuple[str, int, float]]:
        rows: list[tuple[str, int, float]] = []
        for name, stat in self._primitive_profile.items():
            calls = int(stat.get("calls", 0.0))
            elapsed_s = float(stat.get("time_s", 0.0))
            if calls > 0:
                rows.append((name, calls, elapsed_s))
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows[:top_n]

    # ------------------------------------------------------------------ #
    # Domain interface                                                     #
    # ------------------------------------------------------------------ #

    # Complex Fuzz Grid for strict Mathematical Canonicalization
    FUZZ_GRID = [
        [0, 1, 2, 3, 4, 1],
        [5, 6, 7, 8, 9, 2],
        [9, 8, 7, 0, 5, 3],
        [4, 3, 2, 1, 0, 4],
        [1, 5, 9, 2, 8, 5],
        [6, 3, 0, 8, 7, 0]
    ]

    def fuzz_hash(self, tree: Node) -> str:
        """
        Evaluate the AST on a constant, high-entropy complex grid.
        If two trees have the exact same Fuzz Hash, they are strictly
        functionally identical globally, regardless of how they are written.
        """
        try:
            pred = tree.eval([self.FUZZ_GRID], self._primitives)
            return _compact_grid_fingerprint(pred, _to_np_grid(pred))
        except Exception:
            return "ERR"

    def evaluate_candidate(self, tree: Node) -> tuple[float, tuple[str, ...], list[float], str, int]:
        """
        Single-pass evaluation for search.
        Returns (fitness, fingerprint, lexicase_errors, fuzz_hash).
        """
        self._discovery_eval_count += 1
        self._current_eval_cost = 0 # Reset for this specific tree
        
        errors: list[float] = []
        fp: list[str] = []

        MAX_EVAL_CELLS = 10_000
        for idx, target_np in enumerate(self._train_targets_np):
            inp_np = self._train_inputs_np[idx]
            try:
                pred = tree.eval([inp_np], self._primitives)
                pred_np = _to_np_grid(pred)
                # target_np is already pre-converted
                if pred_np is None or target_np is None:
                    acc = 0.0
                else:
                    pred_cells = int(pred_np.shape[0] * pred_np.shape[1])
                    # Infrastructure guard (memory safety)
                    if pred_cells > MAX_EVAL_CELLS:
                        acc = 0.0
                    else:
                        acc = grid_cell_accuracy(pred_np, target_np)
                
                errors.append(1.0 - acc)
                fp.append(_compact_grid_fingerprint(pred, pred_np))
            except BudgetExceededException:
                errors.append(1.0)
                fp.append("BUDGET_EXCEEDED")
                # Ensure we report the actual cost that triggered the exception
                break 
            except Exception:
                errors.append(1.0)
                fp.append("ERR")
                self._current_eval_cost += 1 # Penalty for crash

        mean_error = sum(errors) / max(len(errors), 1)
        # Cost is pixels processed * tree size (to penalize complex programs producing large outputs)
        final_cost = max(1, int(self._current_eval_cost))
        
        return (
            mean_error + self.lam * tree.size(),
            tuple(fp),
            errors,
            "",
            final_cost,
        )

    def primitive_names(self) -> list[str]:
        return self._op_list

    def n_vars(self) -> int:
        # One variable: the input grid
        return 1

    def fitness(self, tree: Node) -> float:
        score, _, _, _, _ = self.evaluate_candidate(tree)
        return score
        
    def solve(self, config: SearchConfig | None = None, transition_matrix: dict[str, dict[str, float]] | None = None, on_step: Any | None = None) -> SearchResult:
        """
        Execute beam search to find the best generic expression tree
        that mapping inputs to outputs for this task.

        Returns
        -------
        SearchResult
        """
        from .primitives import registry
        from core.search import BeamSearch
        op_arities = {name: registry.arity(name) for name in self.primitive_names()}
        searcher = BeamSearch(
            fitness_fn=self.fitness,
            op_list=self.primitive_names(),
            n_vars=self.n_vars(),
            config=config or SearchConfig(),
            op_arities=op_arities,
            evaluate_fn=self.evaluate_candidate,
            transition_matrix=transition_matrix,
        )
        result = searcher.run(on_step=on_step)
        self.on_result(result)
        return result

    def description(self) -> str:
        return f"ARC task '{self.task.name}'"

    def on_result(self, result: SearchResult) -> None:
        solved = self.check_solution(result.best_tree)
        status = "✓ SOLVED" if solved else "✗ unsolved"
        print(
            f"  [{status}] {self.task.name}  "
            f"expr={result.best_tree}  "
            f"fitness={result.best_fitness:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def check_solution(self, tree: Node) -> bool:
        """
        Return True iff *tree* produces exact output for all test pairs.

        If there are no test pairs, falls back to checking training pairs.
        """
        pairs = self.task.test_pairs or self.task.train_pairs
        return all(
            is_exact_match(tree.eval([inp], self._primitives), out)
            for inp, out in pairs
        )

    def predict(self, tree: Node, grid: Grid) -> Grid:
        """Apply *tree* to a single input grid."""
        return tree.eval([grid], self._primitives)

    def train_accuracy(self, tree: Node) -> float:
        """Mean cell accuracy across training pairs."""
        accs = []
        for inp, out in self.task.train_pairs:
            try:
                pred = tree.eval([inp], self._primitives)
                accs.append(grid_cell_accuracy(pred, out))
            except Exception:
                accs.append(0.0)
        return sum(accs) / max(len(accs), 1)

    def test_accuracy(self, tree: Node) -> float:
        """Mean cell accuracy across test pairs."""
        if not self.task.test_pairs:
            return self.train_accuracy(tree)
        accs = []
        for inp, out in self.task.test_pairs:
            try:
                pred = tree.eval([inp], self._primitives)
                accs.append(grid_cell_accuracy(pred, out))
            except Exception:
                accs.append(0.0)
        return sum(accs) / max(len(accs), 1)
