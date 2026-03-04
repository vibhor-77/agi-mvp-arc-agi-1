#!/usr/bin/env python3
"""
scripts/run_all.py
==================
Run all three domains and print a unified summary.

Usage:
    python scripts/run_all.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run_symbolic_regression() -> None:
    print("\n" + "="*50)
    print("  PHASE 1: SYMBOLIC REGRESSION (y = sin(x^2) + 2x)")
    print("="*50)

    import math
    from domains.symbolic_reg.domain import SymbolicRegressionDomain
    from core.search import SearchConfig

    domain = SymbolicRegressionDomain.from_function(
        lambda x: math.sin(x**2) + 2*x,
        x_range=(-2.0, 2.0),
        n_points=40,
        lam=0.01,
    )
    # Using low search boundaries for quick smoke tests
    cfg = SearchConfig(
        beam_size   = 5,
        offspring   = 15,
        generations = 30,
        workers     = 1,
        verbose     = True,
        seed        = 42,
    )
    t0 = time.time()
    result = domain.solve(cfg)
    elapsed = time.time() - t0
    print(f"  Found:    {result.best_tree}")
    print(f"  Fitness:  {result.best_fitness:.6f}")
    print(f"  Time:     {elapsed:.1f}s")


def run_cartpole() -> None:
    print("\n" + "="*50)
    print("  PHASE 2: CARTPOLE (Symbolic Controller)")
    print("="*50)

    from domains.cartpole.domain import CartPoleDomain, run_episode
    from core.search import SearchConfig

    # Hardcoding low episodes for quick smoke tests
    domain = CartPoleDomain(
        n_episodes = 3,
        seeds      = list(range(3)),
    )
    cfg = SearchConfig(
        beam_size   = 5,
        offspring   = 10,
        generations = 15,
        workers     = 1,
        verbose     = True,
        seed        = 42,
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



if __name__ == "__main__":
    t0 = time.time()

    # We now default exclusively to fast configurations in run_all
    run_symbolic_regression()
    run_cartpole()

    print(f"\nAll phases finished in {time.time() - t0:.1f}s")
    print("=" * 60)
