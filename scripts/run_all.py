#!/usr/bin/env python3
"""
scripts/run_all.py
==================
Run all three domains and print a unified summary.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --quick
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_symbolic_regression(quick: bool = False) -> None:
    """Demonstrate symbolic regression on y = sin(x²)."""
    import math
    from domains.symbolic_reg.domain import SymbolicRegressionDomain
    from core.search import SearchConfig

    print("\n" + "=" * 60)
    print("  SYMBOLIC REGRESSION  —  y = sin(x²)")
    print("=" * 60)

    xs = [i * 0.2 for i in range(-20, 21)]
    ys = [math.sin(x ** 2) for x in xs]

    domain = SymbolicRegressionDomain(xs, ys)
    cfg = SearchConfig(
        beam_size   = 5 if quick else 20,
        offspring   = 15 if quick else 40,
        generations = 30 if quick else 150,
        verbose     = False,
        seed        = 42,
    )
    t0 = time.time()
    result = domain.solve(cfg)
    elapsed = time.time() - t0
    print(f"  Found:    {result.best_tree}")
    print(f"  Fitness:  {result.best_fitness:.6f}")
    print(f"  Time:     {elapsed:.1f}s")


def run_cartpole(quick: bool = False) -> None:
    """Demonstrate symbolic RL on CartPole."""
    from domains.cartpole.domain import CartPoleDomain, run_episode
    from core.search import SearchConfig

    print("\n" + "=" * 60)
    print("  CARTPOLE  —  symbolic control policy")
    print("=" * 60)

    domain = CartPoleDomain(
        n_episodes = 3 if quick else 10,
        seeds      = list(range(3 if quick else 10)),
    )
    cfg = SearchConfig(
        beam_size   = 5 if quick else 15,
        offspring   = 10 if quick else 30,
        generations = 15 if quick else 80,
        verbose     = False,
        seed        = 0,
    )
    t0 = time.time()
    result = domain.solve(cfg)
    elapsed = time.time() - t0

    # Evaluate on fresh episodes
    from core.primitives import registry
    prims = {n: registry.get(n) for n in registry.names(domain="math")}
    steps_list = []
    for seed in range(10):
        traj = run_episode(lambda s: result.best_tree.eval(s, prims), seed=seed)
        steps_list.append(len(traj))
    mean_steps = sum(steps_list) / len(steps_list)

    print(f"  Policy:     {result.best_tree}")
    print(f"  Fitness:    {result.best_fitness:.2f}")
    print(f"  Mean steps: {mean_steps:.1f} / 200")
    print(f"  Time:       {elapsed:.1f}s")


def run_arc(quick: bool = False) -> None:
    """Run the ARC-AGI-1 benchmark (baseline vs expanded DSL)."""
    from domains.arc.runner import run_benchmark, BenchmarkConfig
    from domains.arc.benchmark import build_benchmark

    tasks = build_benchmark()
    cfg = BenchmarkConfig(
        beam_size   = 10 if quick else 20,
        offspring   = 25 if quick else 50,
        generations = 40 if quick else 100,
        verbose     = False,
        baseline_only = False,
    )
    baseline, expanded = run_benchmark(tasks, cfg, save_path="results.json")
    print(baseline.summary())
    print(expanded.summary())


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if quick:
        print("Running in quick mode (reduced generations)...")

    run_symbolic_regression(quick)
    run_cartpole(quick)
    run_arc(quick)

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
