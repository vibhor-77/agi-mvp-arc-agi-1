"""
core/search.py
==============
Domain-agnostic deterministic beam search over expression trees.

The search engine only needs:
  - A fitness function  f(Node) -> float  (lower is better)
  - A list of primitive names to build trees from
  - The number of input variables

The same engine solves:
  - Symbolic regression   (fitness = MSE + lambda * size)
  - CartPole policy search (fitness = -mean_steps)
  - ARC grid transforms   (fitness = cell_error + lambda * size)
  - Any future domain      (define fitness, register primitives, go)

Algorithm
---------
Each generation:
  1. Keep a scored beam from the previous generation.
  2. Spawn up to ``beam_size * offspring`` new children.
  3. Evaluate only those new children.
  4. Merge old beam + new children and keep the top unique survivors.

This keeps compute accounting simple and deterministic:
  - budgets are enforced by evaluated candidate count / cost
  - solved tasks can still use their full configured budget
  - the hot path avoids novelty rewards, annealing noise, and extra fuzz evals
"""
from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .tree import Node, crossover, mutate, random_tree


@dataclass
class SearchConfig:
    """All tunable parameters for a beam search run."""

    beam_size: int = 10
    offspring: int = 20
    generations: int = 25
    max_init_depth: int = 3
    workers: int = 1
    converge_threshold: float = 1e-9
    mutation_rate: float = 0.5
    const_range: tuple[float, float] = (-3.0, 3.0)
    const_sigma: float = 0.5
    verbose: bool = True
    log_interval: int = 50
    seed: int | None = None
    initial_temp: float = 0.0
    cooling_rate: float = 1.0
    max_evals: int | None = None
    max_cost: int | None = None
    timeout_s: float | None = None


@dataclass
class SearchResult:
    """Returned by BeamSearch.run()."""

    best_tree: Node
    best_fitness: float
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    converged: bool = False
    n_evals: int = 0
    n_cost: int = 0


_worker_fitness_fn: Callable | None = None
_worker_fingerprint_fn: Callable | None = None
_worker_evaluate_fn: Callable | None = None


ScoredTree = tuple[float, Node, tuple | None, int]


def _normalize_eval_output(out: Any) -> tuple[float, tuple | None, int]:
    if not isinstance(out, tuple):
        return (float(out), None, 1)
    if len(out) >= 5:
        score, fp, _lex, _fuzz, cost = out[:5]
        return (float(score), fp, max(1, int(cost)))
    if len(out) == 4:
        score, fp, _lex, _fuzz = out
        return (float(score), fp, 1)
    if len(out) == 3:
        score, fp, cost = out
        return (float(score), fp, max(1, int(cost)))
    if len(out) == 2:
        score, fp = out
        return (float(score), fp, 1)
    if len(out) == 1:
        return (float(out[0]), None, 1)
    return (float("inf"), None, 1)


def _worker_eval(tree: Node) -> ScoredTree:
    if _worker_evaluate_fn is not None:
        score, fp, cost = _normalize_eval_output(_worker_evaluate_fn(tree))
        return (score, tree, fp, cost)
    score = _worker_fitness_fn(tree) if _worker_fitness_fn else float("inf")
    fp = _worker_fingerprint_fn(tree) if _worker_fingerprint_fn else None
    return (score, tree, fp, 1)


class BeamSearch:
    """Deterministic beam search over expression trees."""

    def __init__(
        self,
        fitness_fn: Callable[[Node], float],
        op_list: list[str],
        n_vars: int = 1,
        config: SearchConfig | None = None,
        op_arities: dict[str, int] | None = None,
        fingerprint_fn: Callable[[Node], tuple] | None = None,
        lexicase_fn: Callable[[Node], list[float]] | None = None,
        fuzz_hash_fn: Callable[[Node], str] | None = None,
        evaluate_fn: Callable[[Node], Any] | None = None,
        transition_matrix: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.op_list = list(op_list)
        self.n_vars = n_vars
        self.config = config or SearchConfig()
        self.op_arities = op_arities
        self.fingerprint_fn = fingerprint_fn
        self.evaluate_fn = evaluate_fn
        self.transition_matrix = transition_matrix
        self._rng = random.Random(self.config.seed)
        # Backward-compatible args kept in the signature, but intentionally unused.
        _ = lexicase_fn
        _ = fuzz_hash_fn

    def run(self, on_step: Any | None = None) -> SearchResult:
        cfg = self.config
        rng = self._rng
        t0 = time.time()
        n_evals_total = 0
        n_cost_total = 0
        converged = False
        history: list[dict] = []

        init_count = self._cap_eval_count(cfg.beam_size * 5, n_evals_total)
        if init_count <= 0:
            init_count = 1
        init_pool = [
            random_tree(
                self.op_list,
                self.n_vars,
                cfg.max_init_depth,
                cfg.const_range,
                rng,
                self.op_arities,
                self.transition_matrix,
            )
            for _ in range(init_count)
        ]
        beam_scored, evals_used, cost_used = self._evaluate_and_select(init_pool, cfg.beam_size)
        n_evals_total += evals_used
        n_cost_total += cost_used
        best_score, best_tree, _best_fp, _best_cost = beam_scored[0]
        best_tree = best_tree.clone()

        for gen in range(cfg.generations):
            elapsed = time.time() - t0
            if on_step is not None:
                try:
                    on_step(n_evals_total, elapsed, n_cost_total)
                except TypeError:
                    on_step(n_evals_total, elapsed)

            history.append(
                {
                    "gen": gen,
                    "best_fitness": best_score,
                    "best_expr": str(best_tree),
                    "elapsed_s": round(elapsed, 2),
                }
            )

            if cfg.verbose and (gen % cfg.log_interval == 0):
                print(
                    f"  gen={gen:4d}  fitness={best_score:.4e}  "
                    f"expr={str(best_tree)[:55]}  [{elapsed:.1f}s]"
                )

            if cfg.timeout_s is not None and elapsed >= cfg.timeout_s:
                break
            if cfg.max_evals is not None and n_evals_total >= cfg.max_evals:
                break
            if cfg.max_cost is not None and n_cost_total >= cfg.max_cost:
                break

            children = self._spawn_children(beam_scored)
            if not children:
                break
            remaining = self._cap_eval_count(len(children), n_evals_total)
            if remaining <= 0:
                break
            if remaining < len(children):
                children = children[:remaining]

            child_scored, evals_used, cost_used = self._evaluate_and_select(children, limit=len(children))
            n_evals_total += evals_used
            n_cost_total += cost_used

            combined = beam_scored + child_scored
            beam_scored = self._select_survivors(combined, cfg.beam_size)
            if not beam_scored:
                break

            gen_best_score, gen_best_tree, _gen_fp, _gen_cost = beam_scored[0]
            if gen_best_score < best_score:
                best_score = gen_best_score
                best_tree = gen_best_tree.clone()
                converged = cfg.converge_threshold > 0 and best_score <= cfg.converge_threshold

        return SearchResult(
            best_tree=best_tree,
            best_fitness=float(best_score),
            history=history,
            elapsed_s=time.time() - t0,
            converged=converged,
            n_evals=n_evals_total,
            n_cost=n_cost_total,
        )

    def _cap_eval_count(self, requested: int, n_evals_total: int) -> int:
        max_evals = self.config.max_evals
        if max_evals is None:
            return requested
        remaining = max(0, int(max_evals) - int(n_evals_total))
        return min(requested, remaining)

    def _spawn_children(self, beam_scored: list[ScoredTree]) -> list[Node]:
        cfg = self.config
        rng = self._rng
        beam = [tree for _score, tree, _fp, _cost in beam_scored]
        if not beam:
            return []

        children: list[Node] = []
        seen_exprs = {str(tree) for tree in beam}
        target = max(0, cfg.beam_size * cfg.offspring)
        max_attempts = max(target * 4, 32)
        attempts = 0
        while len(children) < target and attempts < max_attempts:
            attempts += 1
            parent = beam[(attempts - 1) % len(beam)]
            if rng.random() < cfg.mutation_rate or len(beam) < 2:
                child = mutate(
                    parent,
                    self.op_list,
                    self.n_vars,
                    cfg.const_range,
                    cfg.const_sigma,
                    rng,
                    self.op_arities,
                    self.transition_matrix,
                )
            else:
                mate = beam[rng.randrange(len(beam))]
                if mate is parent and len(beam) > 1:
                    mate = beam[(beam.index(parent) + 1) % len(beam)]
                child = crossover(parent, mate, rng)
            expr = str(child)
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            children.append(child)
        return children

    def _evaluate_and_select(self, pool: list[Node], limit: int) -> tuple[list[ScoredTree], int, int]:
        scored = self._evaluate_all(pool)
        return self._select_survivors(scored, limit), len(scored), sum(item[3] for item in scored)

    def _select_survivors(self, pool: list[ScoredTree], limit: int) -> list[ScoredTree]:
        best_by_expr: dict[str, ScoredTree] = {}
        best_by_fp: dict[tuple, ScoredTree] = {}
        for item in pool:
            score, tree, fp, cost = item
            expr = str(tree)
            prev_expr = best_by_expr.get(expr)
            if prev_expr is None or self._is_better(item, prev_expr):
                best_by_expr[expr] = item
            if fp is not None:
                prev_fp = best_by_fp.get(fp)
                if prev_fp is None or self._is_better(item, prev_fp):
                    best_by_fp[fp] = item

        survivors: list[ScoredTree] = []
        seen_exprs: set[str] = set()
        for item in sorted(best_by_expr.values(), key=self._sort_key):
            score, tree, fp, cost = item
            expr = str(tree)
            if fp is not None and best_by_fp.get(fp) is not item:
                continue
            if expr in seen_exprs:
                continue
            seen_exprs.add(expr)
            survivors.append((score, tree, fp, cost))
            if len(survivors) >= limit:
                break

        if survivors:
            return survivors
        return sorted(pool, key=self._sort_key)[:limit]

    def _sort_key(self, item: ScoredTree) -> tuple[float, int, str]:
        score, tree, _fp, _cost = item
        return (float(score), tree.size(), str(tree))

    def _is_better(self, left: ScoredTree, right: ScoredTree) -> bool:
        return self._sort_key(left) < self._sort_key(right)

    def _evaluate_all(self, pool: list[Node]) -> list[ScoredTree]:
        cfg = self.config
        if cfg.workers <= 1:
            results: list[ScoredTree] = []
            for tree in pool:
                if self.evaluate_fn is not None:
                    score, fp, cost = _normalize_eval_output(self.evaluate_fn(tree))
                else:
                    score = self.fitness_fn(tree)
                    fp = self.fingerprint_fn(tree) if self.fingerprint_fn else None
                    cost = 1
                results.append((score, tree, fp, cost))
            return results

        global _worker_fitness_fn, _worker_fingerprint_fn, _worker_evaluate_fn
        _worker_fitness_fn = self.fitness_fn
        _worker_fingerprint_fn = self.fingerprint_fn
        _worker_evaluate_fn = self.evaluate_fn
        with mp.get_context("fork").Pool(cfg.workers) as pool_executor:
            return pool_executor.map(_worker_eval, pool)
