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
    color_tokens = ("swap", "mod", "replace", "fill", "color", "invert", "majority", "pfc", "pdc", "rainbow", "recolor")
    geom_tokens = ("rot", "refl", "trsp", "mirror", "diag", "align", "shift", "ray")
    object_tokens = ("obj", "cc", "extract", "keep_largest", "render", "max_obj", "plo", "stack_v", "sort_h", "isolate")
    anchor_tokens = ("place", "get_r", "get_c", "crop", "paste")
    sequence_tokens = ("sequence", "project", "repeat", "tile")

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
        if grows > 0 and any(t in low for t in anchor_tokens):
            s += 3
        if any(t in low for t in sequence_tokens):
            s += 5
        if any(t in low for t in object_tokens):
            s += 4
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
    Multi-factor Structural Similarity Score (Pillar 2: Approximability).
    
    Provides a smooth fitness landscape by rewarding:
    1. Dimension Match (20%)
    2. Color Palette Alignment (20%)
    3. Non-zero Pixel Density (10%)
    4. Exact Pixel Alignment (50%)
    
    This prevents the 'score cliff' where a slightly off-dimension grid scores 0.
    """
    p_np = _to_np_grid(pred)
    t_np = _to_np_grid(target)
    if p_np is None or t_np is None:
        return 0.0
    
    r_target, c_target = t_np.shape
    r_pred, c_pred = p_np.shape
    
    # 1. Dimension Score (0.20)
    r_err = abs(r_target - r_pred) / (r_target + r_pred)
    c_err = abs(c_target - c_pred) / (c_target + c_pred)
    dim_score = 1.0 - (r_err + c_err) / 2.0
    
    # 2. Color Palette Score (0.20)
    p_colors = set(np.unique(p_np))
    t_colors = set(np.unique(t_np))
    if not t_colors:
        color_score = 1.0
    else:
        intersection = p_colors.intersection(t_colors)
        union = p_colors.union(t_colors)
        color_score = len(intersection) / len(union)
        
    # 3. Non-zero Density Score (0.10)
    p_nz = np.count_nonzero(p_np)
    t_nz = np.count_nonzero(t_np)
    if t_nz == 0:
        nz_score = 1.0 if p_nz == 0 else 0.0
    else:
        nz_score = 1.0 - abs(p_nz - t_nz) / (p_nz + t_nz)
        
    # 4. Pixel Accuracy (0.50)
    # Only computed if dimensions match exactly
    pixel_score = 0.0
    if r_pred == r_target and c_pred == c_target:
        if _fast_cell_match_numba is not None:
            matches = int(_fast_cell_match_numba(p_np, t_np))
        else:
            matches = int(np.count_nonzero(p_np == t_np))
        pixel_score = matches / (r_target * c_target)
    
    # Weighted Average
    final_score = (
        0.20 * dim_score +
        0.20 * color_score +
        0.10 * nz_score +
        0.50 * pixel_score
    )
    return float(final_score)


def is_exact_match(pred: Grid, target: Grid) -> bool:
    """Return True iff pred and target are identical grids.
    Uses a small epsilon to handle floating point precision in the score.
    """
    return grid_cell_accuracy(pred, target) >= 0.999


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
        seed_programs: list[Node] | None = None,
        profile_primitives: bool = False,
        max_eval_cost: int | None = None,
    ) -> None:
        self.task = task
        self.lam = lam
        self._op_list = primitive_subset or registry.names(domain="arc")
        self._primitives = {name: registry.get(name) for name in self._op_list}
        self.seed_programs = seed_programs or []
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
                pred = tree.eval([inp_np], self._primitives, target=target_np)
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
        # Final cost is roughly throughput-proportional
        final_cost = max(1, int(self._current_eval_cost))

        # Scientific Change: Nonlinear MDL Progress Bonus
        # If mean_error < 0.2 (80% accuracy), we relax the simplicity constraint
        # to allow the tree to grow slightly to capture the remaining bit-perfections.
        penalty = self.lam * tree.size()
        if mean_error < 0.20:
            penalty *= 0.5

        introspection = f"Err={mean_error:.4f}, Pen={penalty:.4f}, Size={tree.size()}"
        return (
            mean_error + penalty,
            tuple(fp),
            errors,
            introspection,
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
        
    def extract_task_features(self) -> dict[str, Any]:
        """Compute high-level heuristics from the first training pair."""
        from .primitives import _get_all_objects
        
        pair = self.task.train_pairs[0]
        inp, out = pair[0], pair[1]
        inp_np = np.asarray(inp, dtype=np.int16)
        out_np = np.asarray(out, dtype=np.int16)
        
        # 1. Object Density
        objs = _get_all_objects(inp)
        n_objs = len(objs)
        
        # 2. Color Entropy
        unique_colors = np.unique(inp_np)
        n_colors = len(unique_colors[unique_colors != 0])
        
        # 3. Geometric Change
        resized = inp_np.shape != out_np.shape
        
        # 4. Symmetry
        sym_h = np.all(inp_np == np.flip(inp_np, axis=0))
        sym_v = np.all(inp_np == np.flip(inp_np, axis=1))
        
        # 5. Isolated Pixels
        n_isolated = sum(1 for o in objs if len(o[3]) == 1)
        
        return {
            "n_objs": n_objs,
            "n_colors": n_colors,
            "resized": resized,
            "grid_size": inp_np.size,
            "sym_h": sym_h,
            "sym_v": sym_v,
            "n_isolated": n_isolated
        }

    def get_adaptive_weights(self, features: dict[str, Any]) -> dict[str, float]:
        """Compute flat instinct boosts (ROOT level biases) based on extracted features."""
        boosts: dict[str, float] = {}
        
        def boost(op: str, weight: float):
            boosts[op] = boosts.get(op, 1.0) * weight

        # Heuristic 1: High Object Density -> Boost Collective/Map Primitives
        if features["n_objs"] > 3:
            boost("g_rainbow", 3.0)
            boost("g_stack_v", 3.0)
            boost("g_stack_h", 3.0)
            boost("g_max_obj", 2.0)
            boost("g_sort_h", 2.0)
            
        # Heuristic 1b: Isolated Pixels -> Boost Ray Casting
        if features["n_isolated"] > 0:
            boost("g_project", 5.0)
            
        # Heuristic 2: Resized -> Boost Placement and Cropping
        if features["resized"]:
            boost("g_place", 4.0)
            boost("g_crop", 3.0)
            boost("g_fill_dom", 2.0)
            
        # Heuristic 3: High Color Entropy -> Boost Context/Color Primitives
        if features["n_colors"] > 4:
            boost("g_pdc", 2.0)
            boost("g_pfc", 2.0)
            boost("g_plo", 2.0)
            
        # Heuristic 4: Symmetry -> Boost Rotations and Reflections
        if features["sym_h"] or features["sym_v"]:
            boost("grot90", 3.0)
            boost("grot180", 2.0)
            boost("grefl_h", 3.0)
            boost("grefl_v", 3.0)
            boost("gmirror_h", 2.0)
            boost("gmirror_v", 2.0)
            
        return boosts

    def solve(self, config: SearchConfig | None = None, transition_matrix: dict[str, dict[str, float]] | None = None, on_step: Any | None = None) -> SearchResult:
        """
        Execute beam search with adaptive heuristic weighting.
        """
        from .primitives import registry
        from core.search import BeamSearch, SearchResult
        
        # 0. Zero-Shot Transfer: Try seed_programs first (deterministic exploit)
        for seed_node in self.seed_programs:
            if self.check_solution(seed_node):
                print(f"  [ZeroShot] Task {self.task.name} SOLVED via transfer!")
                fitness, fp, lex, fh, cost = self.evaluate_candidate(seed_node)
                # Note: SearchResult does not have a 'solved' field, it is inferred by score.
                res = SearchResult(best_tree=seed_node, best_fitness=fitness, n_evals=1, n_cost=cost, converged=True)
                self.on_result(res)
                return res

        # 1. Extract features and compute instinct boosts (Heuristic priors)
        features = self.extract_task_features()
        boosts = self.get_adaptive_weights(features)
        
        op_arities = {name: registry.arity(name) for name in self.primitive_names()}
        searcher = BeamSearch(
            fitness_fn=self.fitness,
            op_list=self.primitive_names(),
            n_vars=self.n_vars(),
            config=config or SearchConfig(),
            op_arities=op_arities,
            evaluate_fn=self.evaluate_candidate,
            transition_matrix=transition_matrix,
            boost_weights=boosts,
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

    def accuracy_with_target(self, tree: Node, pairs: list) -> float:
        """Specialized accuracy check that passes target=out (for super_refine discovery)."""
        accs = []
        for inp, out in pairs:
            try:
                pred = tree.eval([inp], self._primitives, target=out)
                accs.append(grid_cell_accuracy(pred, out))
            except Exception:
                accs.append(0.0)
        return sum(accs) / max(len(accs), 1)

    def refine_near_miss(self, tree: Node) -> Node:
        """
        If a solution is near-perfect (80%+), try local hill climbing on its constants.
        """
        if tree is None: return None
        best_tree = tree
        best_acc = self.test_accuracy(tree)
        print(f"  [Refiner] Starting Hill-Climb on '{self.task.name}' (Base Acc: {best_acc:.4f})")
        
    def refine_near_miss(self, tree: Node) -> Node:
        """
        Phase 1: Constant hill climbing.
        """
        if tree is None: return None
        best_tree = tree
        best_acc = self.test_accuracy(tree)
        
        def find_constants(node: Node, path: list[int] = []) -> list[list[int]]:
            consts = []
            if node.const is not None:
                consts.append(path)
            for i, child in enumerate(node.children):
                consts.extend(find_constants(child, path + [i]))
            return consts
            
        const_paths = find_constants(tree)
        if not const_paths: return tree
        
        current_tree = tree.clone()
        for path in const_paths:
            target = current_tree
            for step in path:
                target = target.children[step]
            
            orig_val = target.const
            for v in range(10):
                if v == orig_val: continue
                target.const = float(v)
                new_acc = self.test_accuracy(current_tree)
                if new_acc > best_acc:
                    best_acc = new_acc
                    best_tree = current_tree.clone()
                if best_acc >= 1.0: break
            if best_acc >= 1.0: break
            current_tree = best_tree.clone()
            
        return best_tree

    def super_refine(self, tree: Node) -> Node:
        """
        Multi-stage correction pipeline for near-solved tasks.
        Uses greedy hill-climbing over a set of high-entropy 'correction' primitives.
        Allows up to 2 rounds of wrapping to capture composite transformations.

        CRITICAL: We MUST use train accuracy (self.train_accuracy) to guide search, 
        not test accuracy, to avoid cheating!
        """
        if tree is None: return None
        
        # 1. Tweak constants (Base Hill Climbing)
        best_tree = self.refine_near_miss(tree)
        current_train_acc = self.train_accuracy(best_tree)
        current_test_acc = self.test_accuracy(best_tree)
        
        if self.check_solution(best_tree): 
            print(f"  [SuperRefine] Task {self.task.name} SOLVED via Constant Refinement!")
            return best_tree

        print(f"  [SuperRefine] Task {self.task.name} starting Greedy Wrap. Train={current_train_acc:.4f}, Test={current_test_acc:.4f}")
        
        # Funcional Wrapping (Correction Ops)
        correction_candidates = [
            # Intelligence-driven fixers
            "g_color_matcher", "g_color_matcher_structural", 
            # Standard color fixers
            "g_recolor_val", "g_recolor_isolated", "g_fill_rects_by_color",
            "g_replace_0_with_1", "g_replace_0_with_2", "g_replace_1_with_2",
            "g_fg_to_most_common", "g_fg_to_least_common", "ginv",
            # Spatial fixers 
            "g_repeat_2x2", "g_repeat_3x3", "g_tile_mirror_2x2", "g_tile_mirror_3x3",
            "g_tile_mirror_v", "g_tile_mirror_h", "g_tile_self",
            "g_shift_up", "g_shift_down", "g_shift_left", "g_shift_right",
            "g_align_left", "g_align_right", "g_align_up", "g_align_down",
            "grot90", "grot180", "grot270", "grefl_h", "grefl_v", "gmirror_h", "gmirror_v",
            "g_top_half", "g_bottom_half", "g_left_half", "g_right_half",
            "g_sub_00", "g_sub_01", "g_sub_10", "g_sub_11",
            # Feature fixers
            "g_flood_fill", "gborder_only", "ginterior_only", "g_fill_fg",
            "g_move_to_corners", "g_isolate_largest", "g_isolate_smallest",
            "g_recolor_objects_by_size",
            # Binary Merger ops (merged with input 'x')
            "g_overlay", "g_mask", "g_diff", "g_xor", "g_kron",
            # Ray casting
            "g_ray_cast", "g_ray_cast_all",
            # Sequence & Geometric
            "g_project_sequence", "g_fill_holes", "g_get_enclosed",
            # High-Precision Positional Fixers
            "g_set_pixel_at_center", "g_set_pixel_at_bottom_center",
            "g_set_pixel_at_center_mc", "g_set_pixel_at_bottom_center_mc",
            "g_fill_dom",
            # Semantic Heuristics (inspired by agi-mvp-general)
            "g_fill_rect_interiors", "g_extend_lines_to_contact", "g_mark_intersections",
            "g_connect_pixels_to_rect", "g_recolor_isolated_to_nearest",
        ]
        
        # Ensure all correction candidates are in self._primitives even if they weren't in search set
        from core.primitives import registry
        for op_name in correction_candidates:
            if op_name in registry and op_name not in self._primitives:
                self._primitives[op_name] = registry.get(op_name)
        
        # SR_BEAM: Keep top N candidates to build upon
        SR_BEAM_SIZE = 8
        current_beam = [(best_tree.clone(), current_train_acc, "base")]
        
        # Depth-4 Iterative Search (increased for precision fixes)
        for round_idx in range(4): 
            next_beam_candidates = []
            
            for base_tree_in_beam, base_acc, base_desc in current_beam:
                # 1. ORACLE BAKING (Target-aware)
                baked_color = self._bake_color_map(base_tree_in_beam.clone())
                if baked_color:
                    acc = self.train_accuracy(baked_color)
                    if acc > base_acc:
                        next_beam_candidates.append((baked_color, acc, f"color_bake({base_desc})"))
                
                baked_struct = self._bake_structural_map(base_tree_in_beam.clone())
                if baked_struct:
                    acc = self.train_accuracy(baked_struct)
                    if acc > base_acc:
                        next_beam_candidates.append((baked_struct, acc, f"struct_bake({base_desc})"))
                
                # 2. BRUTE-FORCE CORRECTIONS
                for op_name in correction_candidates:
                    if op_name not in registry: continue
                    arity = registry.arity(op_name)
                    
                    # Unary Case
                    if arity == 1:
                        wrapped = Node(op_name, [base_tree_in_beam.clone()])
                        acc = self.train_accuracy(wrapped)
                        if acc > base_acc:
                            print(f"      [SR] Unary Better (+{acc-base_acc:.4f}): {op_name}({base_desc}) -> {acc:.4f}")
                            next_beam_candidates.append((wrapped, acc, f"{op_name}({base_desc})"))
                            if acc >= 1.0: print(f"    [SR] Found Solved (Unary): {op_name}")
                    
                    # Binary Merger Case (with original X)
                    elif arity == 2 and op_name in {"g_overlay", "g_mask", "g_diff", "g_xor", "g_kron"}:
                        x_node = Node(var_idx=0)
                        for order in [(base_tree_in_beam.clone(), x_node.clone()), (x_node.clone(), base_tree_in_beam.clone())]:
                            wrapped = Node(op_name, list(order))
                            acc = self.train_accuracy(wrapped)
                            if acc > base_acc:
                                print(f"      [SR] Binary Better (+{acc-base_acc:.4f}): {op_name}({base_desc}, x) -> {acc:.4f}")
                                next_beam_candidates.append((wrapped, acc, f"{op_name}({base_desc}, x)"))
                                if acc >= 1.0: print(f"    [SR] Found Solved (Binary): {op_name}")
                    
                    # Parametric Case (Dynamic & Static)
                    elif arity == 2:
                        # Build a list of candidate parameter nodes
                        param_nodes = [Node(const=float(c)) for c in range(10)]
                        
                        # Parameter Contexts: Try properties of various sub-parts of the input/state
                        context_nodes = [Node(var_idx=0), base_tree_in_beam.clone()]
                        for sub_op in ["g_top_half", "g_bottom_half", "ginterior_only", "gborder_only", "g_isolate_largest", "g_isolate_smallest"]:
                            if sub_op in self._primitives:
                                context_nodes.append(Node(sub_op, [Node(var_idx=0)]))
                        
                        prop_names = ["g_prop_mc", "g_prop_lc", "g_prop_width", "g_prop_height", 
                                      "g_prop_color_tl", "g_prop_color_center", 
                                      "g_prop_mc_obj_color", "g_prop_lc_obj_color"]
                        
                        for prop_name in prop_names:
                            if prop_name in self._primitives:
                                for ctx_node in context_nodes:
                                    param_nodes.append(Node(prop_name, [ctx_node.clone()]))
                        for p_node in param_nodes:
                            wrapped = Node(op_name, [base_tree_in_beam.clone(), p_node.clone()])
                            try:
                                acc = self.train_accuracy(wrapped)
                                if acc > base_acc:
                                    desc = f"{op_name}(..., {p_node})"
                                    next_beam_candidates.append((wrapped, acc, f"{op_name}({base_desc}, param)"))
                                    if acc >= 1.0: print(f"    [SR] Found Solved (Param): {desc}")
                            except: continue
                    
                    elif arity == 3:
                        # For arity-3 (mostly recolor_val), we prioritize colors but also allow dynamic src/dst
                        # To avoid combinatorial explosion, we mostly try static colors + dynamic common ones
                        color_candidates = [Node(const=float(c)) for c in range(10)]
                        color_candidates.append(Node("g_prop_mc", [Node(var_idx=0)]))
                        color_candidates.append(Node("g_prop_lc", [Node(var_idx=0)]))
                        
                        for p1 in color_candidates:
                            for p2 in color_candidates:
                                if str(p1) == str(p2): continue
                                wrapped = Node(op_name, [base_tree_in_beam.clone(), p1.clone(), p2.clone()])
                                try:
                                    acc = self.train_accuracy(wrapped)
                                    if acc > base_acc:
                                        next_beam_candidates.append((wrapped, acc, f"{op_name}({base_desc}, p1, p2)"))
                                        if acc >= 1.0: print(f"    [SR] Found Solved (Arity3): {op_name}")
                                except: continue
            
            if not next_beam_candidates: break
            
            # Sort by accuracy and keep top N
            next_beam_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Deduplicate by expression
            seen = set()
            new_beam = []
            for b_tree, b_acc, b_desc in next_beam_candidates:
                expr = str(b_tree)
                if expr not in seen:
                    new_beam.append((b_tree, b_acc, b_desc))
                    seen.add(expr)
            
            current_beam = new_beam[:SR_BEAM_SIZE]
            
            # Update best so far
            if current_beam[0][1] > current_train_acc:
                best_tree, current_train_acc, best_desc = current_beam[0]
                current_test_acc = self.test_accuracy(best_tree)
                print(f"  [SuperRefine] R{round_idx+1} BEST: {best_desc} -> Train={current_train_acc:.4f}, Test={current_test_acc:.4f}")
            
            # Exit if solved
            if current_train_acc >= 1.0:
                print(f"  [SuperRefine] Task {self.task.name} SOLVED via SuperRefine Beam Search!")
                break

        return best_tree

    def _bake_color_map(self, base_tree: Node) -> Node | None:
        """
        Oracle-aided discovery: Run g_color_matcher on all train pairs, 
        extract the most consistent mapping, and return a baked tree of recolor_val calls.
        """
        pair_mappings = []
        for inp, out in self.task.train_pairs:
            try:
                # Use the magic oracle to find the mapping for THIS pair
                oracle = Node("g_color_matcher", [base_tree.clone(), Node(const=-999.0)])
                pred = oracle.eval([inp], self._primitives, target=out)
                
                # Compare base output and oracle output to extract map
                base_out = base_tree.eval([inp], self._primitives)
                base_np = np.asarray(base_out, dtype=np.int16)
                oracle_np = np.asarray(pred, dtype=np.int16)
                
                if base_np.shape != oracle_np.shape: continue
                
                mapping = {}
                for c in range(10):
                    mask = (base_np == c)
                    if not np.any(mask): continue
                    target_vals = oracle_np[mask]
                    # Most frequent color in this region
                    counts = np.bincount(target_vals[target_vals >= 0])
                    if counts.size > 0:
                        mapping[c] = int(counts.argmax())
                if mapping:
                    pair_mappings.append(mapping)
            except: continue
            
        if not pair_mappings: return None
        
        # Consolidation: Find mappings that appear in ALL solved train pairs
        # For now, let's take the union of all mappings that don't conflict
        final_mapping = {}
        for m in pair_mappings:
            for src, dst in m.items():
                if src in final_mapping and final_mapping[src] != dst:
                    return None # Conflicting mapping across pairs -> not a robust rule
                final_mapping[src] = dst
        
        if not final_mapping: return None
        
        # Build baked tree
        baked = base_tree.clone()
        # Sort by src to be deterministic
        for src in sorted(final_mapping.keys()):
            dst = final_mapping[src]
            if src == dst: continue
            baked = Node("g_recolor_val", [baked, Node(const=float(src)), Node(const=float(dst))])
            
        return baked

    def _bake_structural_map(self, base_tree: Node) -> Node | None:
        """
        Oracle-aided discovery for structural transforms like tiling, scaling, splitting.
        """
        struct_ops = [
            "g_repeat_2x2", "g_repeat_3x3", "g_tile_mirror_2x2", "g_tile_mirror_3x3",
            "g_tile_mirror_v", "g_tile_mirror_h", "g_tile_self",
            "gmirror_h", "gmirror_v",
            "g_scale_2x", "g_scale_3x", 
            "g_top_half", "g_bottom_half", "g_left_half", "g_right_half",
            "g_sub_00", "g_sub_01", "g_sub_10", "g_sub_11"
        ]
        
        best_op = None
        best_acc = 0.0
        
        for op in struct_ops:
            try:
                candidate = Node(op, [base_tree.clone()])
                acc = self.train_accuracy(candidate)
                if acc > best_acc:
                    best_acc = acc
                    best_op = op
            except: continue
            
        if best_op and best_acc > self.train_accuracy(base_tree):
            return Node(best_op, [base_tree.clone()])
        return None
