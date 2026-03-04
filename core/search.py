"""
core/search.py
==============
Domain-agnostic beam search over expression trees.

The search engine knows nothing about specific domains.  It only needs:
  - A fitness function  f(Node) -> float  (lower is better)
  - A list of primitive names to build trees from
  - The number of input variables

The same engine solves:
  - Symbolic regression   (fitness = MSE + λ·size)
  - CartPole policy search (fitness = −mean_steps)
  - ARC grid transforms   (fitness = cell_error + λ·size)
  - Any future domain      (define fitness, register primitives, go)

Algorithm
---------
Each generation:
  1. Every beam member spawns ``offspring`` children via mutation/crossover.
  2. All candidates (beam + offspring) are evaluated.
  3. Top ``beam_size`` unique survivors (by string) form the new beam.
  4. Repeat until convergence or ``generations`` reached.

Performance
-----------
Set ``workers > 1`` to evaluate candidates in parallel across CPU cores.
Parallel evaluation is the main bottleneck for slow fitness functions
(e.g. CartPole episodes).  For fast functions (grid transforms, math),
single-threaded is often faster due to spawn overhead.
"""
from __future__ import annotations

import random
import time
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable

from .tree import Node, random_tree, mutate, crossover


# ---------------------------------------------------------------------------
# Search configuration (dataclass for clean parameterisation)
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """
    All tunable parameters for a beam search run.

    Attributes
    ----------
    beam_size : int
        Number of elite candidates kept between generations.
        Larger beams explore more but cost more per generation.
    offspring : int
        Mutations generated per beam member per generation.
        Total pool size = beam_size * (1 + offspring).
    generations : int
        Hard limit on number of generations.
    max_init_depth : int
        Maximum tree depth for the initial random population.
    workers : int
        CPU processes for parallel fitness evaluation.
        0 or 1 = single-threaded (best for fast fitness functions).
    converge_threshold : float
        Stop early if best fitness falls below this value.
    mutation_rate : float
        Fraction of offspring created by mutation (vs crossover).
    const_range : (float, float)
        Range for randomly generated constant leaves.
    const_sigma : float
        Std dev for Gaussian perturbation of constants.
    verbose : bool
        Print progress every ``log_interval`` generations.
    log_interval : int
        How often to print progress when verbose=True.
    seed : int | None
        Random seed for reproducibility.
    """
    beam_size: int = 20
    offspring: int = 60
    generations: int = 300
    max_init_depth: int = 3
    workers: int = 1
    converge_threshold: float = 1e-9
    mutation_rate: float = 0.85
    const_range: tuple[float, float] = (-3.0, 3.0)
    const_sigma: float = 0.5
    verbose: bool = True
    log_interval: int = 50
    seed: int | None = None


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    Returned by BeamSearch.run().

    Attributes
    ----------
    best_tree : Node
        The best expression tree found.
    best_fitness : float
        Its fitness score (lower is better).
    history : list[dict]
        Per-generation record: {gen, best_fitness, best_expr, elapsed_s}.
    elapsed_s : float
        Total wall-clock seconds.
    converged : bool
        True if search stopped early due to converge_threshold.
    """
    best_tree: Node
    best_fitness: float
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    converged: bool = False


# ---------------------------------------------------------------------------
# Worker helpers (module-level so they survive pickling)
# ---------------------------------------------------------------------------

_worker_fitness_fn: Callable | None = None


def _worker_init(fitness_fn: Callable) -> None:
    global _worker_fitness_fn
    _worker_fitness_fn = fitness_fn


def _worker_eval(tree: Node) -> tuple[float, Node]:
    assert _worker_fitness_fn is not None
    return _worker_fitness_fn(tree), tree


# ---------------------------------------------------------------------------
# BeamSearch
# ---------------------------------------------------------------------------

class BeamSearch:
    """
    Beam search over expression trees guided by a pluggable fitness function.

    Parameters
    ----------
    fitness_fn : Callable[[Node], float]
        Lower is better.  Must be picklable if ``workers > 1``.
    op_list : list[str]
        Names of primitives available for tree construction.
        Typically obtained from ``PrimitiveRegistry.names(domain=...)``.
    n_vars : int
        Number of input variables (determines variable leaf indices).
    config : SearchConfig | None
        Tuning parameters.  Uses defaults if None.
    op_arities : dict[str, int] | None
        Arity mapping for primitives. Uses default=1 if None.

    Examples
    --------
    >>> from core.search import BeamSearch, SearchConfig
    >>> from core.primitives import registry
    >>> import math
    >>>
    >>> xs = [0.1 * i for i in range(30)]
    >>> ys = [math.sin(x**2) for x in xs]
    >>>
    >>> def fitness(tree):
    ...     preds = [tree.eval([x], {n: registry.get(n) for n in registry.names()}) for x in xs]
    ...     mse = sum((p-y)**2 for p,y in zip(preds, ys)) / len(ys)
    ...     return mse + 0.001 * tree.size()
    >>>
    >>> searcher = BeamSearch(fitness, registry.names(domain="math"), n_vars=1)
    >>> result = searcher.run()
    >>> print(result.best_tree)
    """

    def __init__(
        self,
        fitness_fn: Callable[[Node], float],
        op_list: list[str],
        n_vars: int = 1,
        config: SearchConfig | None = None,
        op_arities: dict[str, int] | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.op_list = list(op_list)
        self.n_vars = n_vars
        self.config = config or SearchConfig()
        self.op_arities = op_arities
        self._rng = random.Random(self.config.seed)

    def run(self) -> SearchResult:
        """
        Execute the beam search.

        Returns
        -------
        SearchResult
            Contains best tree found, fitness, history, and timing.
        """
        cfg = self.config
        rng = self._rng

        # ── Initialise population ────────────────────────────────────────
        init_pool = [
            random_tree(self.op_list, self.n_vars, cfg.max_init_depth,
                        cfg.const_range, rng, self.op_arities)
            for _ in range(cfg.beam_size * 5)
        ]
        scored = self._evaluate_all(init_pool)
        scored.sort(key=lambda x: x[0])

        beam: list[Node] = [t for _, t in scored[: cfg.beam_size]]
        best_fitness = scored[0][0]
        best_tree = scored[0][1].clone()

        history: list[dict] = []
        t0 = time.time()

        # ── Main loop ────────────────────────────────────────────────────
        for gen in range(cfg.generations):
            # Generate offspring
            pool = list(beam)
            for parent in beam:
                for _ in range(cfg.offspring):
                    if rng.random() < cfg.mutation_rate or len(beam) < 2:
                        child = mutate(
                            parent, self.op_list, self.n_vars,
                            cfg.const_range, cfg.const_sigma, rng,
                            self.op_arities
                        )
                    else:
                        other = rng.choice(beam)
                        child = crossover(parent, other, rng)
                    pool.append(child)

            # Evaluate
            scored = self._evaluate_all(pool)
            scored.sort(key=lambda x: x[0])

            # Deduplicate by string form, keep top beam_size
            seen: set[str] = set()
            new_beam: list[tuple[float, Node]] = []
            for score, tree in scored:
                key = str(tree)
                if key not in seen:
                    seen.add(key)
                    new_beam.append((score, tree))
                if len(new_beam) == cfg.beam_size:
                    break

            beam = [t for _, t in new_beam]
            gen_best_score = new_beam[0][0]
            gen_best_tree = new_beam[0][1]

            # Track history
            elapsed = time.time() - t0
            history.append({
                "gen": gen,
                "best_fitness": gen_best_score,
                "best_expr": str(gen_best_tree),
                "elapsed_s": round(elapsed, 2),
            })

            if gen_best_score < best_fitness:
                best_fitness = gen_best_score
                best_tree = gen_best_tree.clone()

            # Log
            if cfg.verbose and (gen % cfg.log_interval == 0):
                print(
                    f"  gen={gen:4d}  fitness={best_fitness:.4e}  "
                    f"expr={str(best_tree)[:55]}  [{elapsed:.1f}s]"
                )

            # Early exit
            if best_fitness < cfg.converge_threshold:
                if cfg.verbose:
                    print(f"  ✓ Converged at generation {gen}")
                return SearchResult(
                    best_tree=best_tree,
                    best_fitness=best_fitness,
                    history=history,
                    elapsed_s=time.time() - t0,
                    converged=True,
                )

        return SearchResult(
            best_tree=best_tree,
            best_fitness=best_fitness,
            history=history,
            elapsed_s=time.time() - t0,
            converged=False,
        )

    # ------------------------------------------------------------------ #
    # Internal: batch evaluation (parallel or serial)                     #
    # ------------------------------------------------------------------ #

    def _evaluate_all(self, pool: list[Node]) -> list[tuple[float, Node]]:
        """Evaluate every tree in *pool*, returning (score, tree) pairs."""
        cfg = self.config
        if cfg.workers > 1 and len(pool) > cfg.beam_size:
            with mp.Pool(
                processes=cfg.workers,
                initializer=_worker_init,
                initargs=(self.fitness_fn,),
            ) as p:
                return list(p.map(_worker_eval, pool))
        else:
            # Single-threaded — also used when workers=1
            _worker_init(self.fitness_fn)
            return [_worker_eval(t) for t in pool]
