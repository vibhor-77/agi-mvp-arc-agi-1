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
from dataclasses import dataclass, field
from typing import NamedTuple

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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

import numba
from numba import njit
import numpy as np
import time

@njit(cache=True)
def _fast_cell_match(p: np.ndarray, t: np.ndarray) -> int:
    """Numba-accelerated inner loop for pixel matching."""
    matches = 0
    rows, cols = p.shape
    for r in range(rows):
        for c in range(cols):
            if p[r, c] == t[r, c]:
                matches += 1
    return matches

def grid_cell_accuracy(pred: Grid, target: Grid) -> float:
    """
    Fraction of cells that match between *pred* and *target*.
    Uses Numba-accelerated JIT kernel for the inner pixel loop.
    """
    if not isinstance(pred, list) or not isinstance(target, list):
        return 0.0
    if not pred or not target:
        return 0.0
    if not isinstance(pred[0], list) or not isinstance(target[0], list):
        return 0.0
    
    # Dimensional Gravity Heuristic
    MAX_HEURISTIC_BONUS = 0.25
    r_target, c_target = len(target), len(target[0])
    r_pred, c_pred = len(pred), len(pred[0])
    
    r_diff = abs(r_target - r_pred)
    c_diff = abs(c_target - c_pred)
    dim_penalty = (r_diff + c_diff) / (r_target + c_target)
    dimensional_reward = max(0.0, MAX_HEURISTIC_BONUS * (1.0 - dim_penalty))
    
    if r_pred != r_target or c_pred != c_target:
        return dimensional_reward

    # High performance pixel matching
    try:
        p_np = pred if isinstance(pred, np.ndarray) else np.array(pred, dtype=np.int32)
        t_np = target if isinstance(target, np.ndarray) else np.array(target, dtype=np.int32)
        matches = _fast_cell_match(p_np, t_np)
        pixel_accuracy = matches / (r_target * c_target)
        # The actual score is the combination of the structural bonus (0.25) and the pixel accuracy (0.75)
        return dimensional_reward + (pixel_accuracy * (1.0 - MAX_HEURISTIC_BONUS))
    except:
        # Fallback if Numba/Numpy conversion fails (e.g., malformed grid)
        return dimensional_reward


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
        lam: float = 0.02,
        primitive_subset: list[str] | None = None,
        library: Any = None,
    ) -> None:
        self.task = task
        self.lam = lam
        self._op_list = primitive_subset or registry.names(domain="arc")
        self._primitives = {name: registry.get(name) for name in self._op_list}
        if library:
            self._primitives.update(library.primitives)
            self._op_list = list(self._primitives.keys())
        self._discovery_eval_count = 0
        self._discovery_start_time = time.time()

    def get_stats(self) -> str:
        duration = max(0.1, time.time() - self._discovery_start_time)
        return f"evals={self._discovery_eval_count} rate={self._discovery_eval_count/duration:.1f}/s"

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
            return str(tree.eval([self.FUZZ_GRID], self._primitives))
        except Exception:
            return "ERR"

    def fingerprint(self, tree: Node) -> tuple:
        """
        Evaluate the AST on the training inputs to generate a semantic hash.
        Used to deduplicate functionally equivalent trees during search.
        We return tuple(str(output)) to ensure it's hashable.
        """
        fp = []
        for inp, out in self.task.train_pairs:
            try:
                # evaluate tree on the single input variable (the grid)
                res = tree.eval([inp], self._primitives)
                fp.append(str(res))
            except Exception:
                fp.append("ERR")
        return tuple(fp)
        
    def lexicase_eval(self, tree: Node) -> list[float]:
        self._discovery_eval_count += 1
        errors = []
        for inp, out in self.task.train_pairs:
            try:
                pred = tree.eval([inp], self._primitives)
                err = 1.0 - grid_cell_accuracy(pred, out)
            except Exception:
                err = 1.0
            errors.append(err)
        return errors

    def primitive_names(self) -> list[str]:
        return self._op_list

    def n_vars(self) -> int:
        # One variable: the input grid
        return 1

    def fitness(self, tree: Node) -> float:
        self._discovery_eval_count += 1
        total_error = 0.0
        for inp, out in self.task.train_pairs:
            try:
                pred = tree.eval([inp], self._primitives)
                error = 1.0 - grid_cell_accuracy(pred, out)
            except Exception:
                error = 1.0
            total_error += error
        mean_error = total_error / max(len(self.task.train_pairs), 1)
        return mean_error + self.lam * tree.size()
        
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
            fingerprint_fn=self.fingerprint,
            lexicase_fn=self.lexicase_eval,
            fuzz_hash_fn=self.fuzz_hash,
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
