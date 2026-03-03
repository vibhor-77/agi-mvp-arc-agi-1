"""
domains/symbolic_reg/domain.py
==============================
Symbolic regression: discover the simplest formula f(x) ≈ y
from a set of (x, y) data points using MDL-guided beam search.

This is the foundational "Level 1" experiment in the system:
prove that the search engine can rediscover known mathematical
formulas from data alone, with no prior knowledge of the form.

Key idea: MDL fitness
  fitness = MSE(tree, data) + λ * tree.size()

The λ term penalises complexity so the search prefers compact
expressions over ones that merely overfit the training data.

Usage
-----
    from domains.symbolic_reg.domain import SymbolicRegressionDomain
    import math

    xs = [i * 0.1 for i in range(-30, 31)]
    ys = [math.sin(x**2) + 2*x for x in xs]

    domain = SymbolicRegressionDomain(xs, ys)
    result = domain.solve()
    print(result.best_tree)      # e.g. "sin(sq(x))" + 2x approximation
"""
from __future__ import annotations

import math
from typing import Callable

from core.domain import Domain
from core.tree import Node
from core.search import SearchConfig, SearchResult
from core.primitives import registry


class SymbolicRegressionDomain(Domain):
    """
    Learn a symbolic formula  f(x_0, ..., x_k) ≈ y  from data.

    Parameters
    ----------
    xs : list[list[float]] | list[float]
        Input data.  Either a list of scalar floats (univariate) or
        a list of vectors (multivariate).
    ys : list[float]
        Target output values.  Must be the same length as *xs*.
    lam : float
        MDL complexity coefficient.  Higher values prefer shorter trees.
    extra_ops : list[str] | None
        Additional primitive names to include (must be registered).
        Defaults to all math primitives.
    """

    def __init__(
        self,
        xs: list,
        ys: list[float],
        lam: float = 0.001,
        extra_ops: list[str] | None = None,
    ) -> None:
        # Normalise xs to list-of-vectors
        if xs and not isinstance(xs[0], (list, tuple)):
            self._xs: list[list[float]] = [[x] for x in xs]
        else:
            self._xs = [list(x) for x in xs]
        self._ys = list(ys)
        self.lam = lam

        ops = extra_ops if extra_ops is not None else registry.names(domain="math")
        self._op_list = ops
        self._primitives = {n: registry.get(n) for n in ops}

    # ------------------------------------------------------------------ #
    # Domain interface                                                     #
    # ------------------------------------------------------------------ #

    def primitive_names(self) -> list[str]:
        return self._op_list

    def n_vars(self) -> int:
        return len(self._xs[0]) if self._xs else 1

    def fitness(self, tree: Node) -> float:
        """MSE over data points + λ * tree size."""
        se = 0.0
        for variables, y_true in zip(self._xs, self._ys):
            try:
                y_pred = tree.eval(variables, self._primitives)
                if not isinstance(y_pred, (int, float)):
                    return 1e9
                if math.isnan(y_pred) or math.isinf(y_pred):
                    return 1e9
                se += (y_pred - y_true) ** 2
            except Exception:
                return 1e9
        mse = se / max(len(self._ys), 1)
        return mse + self.lam * tree.size()

    def description(self) -> str:
        return f"SymbolicRegression({len(self._ys)} points, {self.n_vars()} var(s))"

    def on_result(self, result: SearchResult) -> None:
        mse = self._mse(result.best_tree)
        print(
            f"  Best expression : {result.best_tree}\n"
            f"  MSE             : {mse:.6f}\n"
            f"  Tree size       : {result.best_tree.size()} nodes\n"
            f"  Fitness         : {result.best_fitness:.6f}"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _mse(self, tree: Node) -> float:
        se = 0.0
        for variables, y_true in zip(self._xs, self._ys):
            try:
                y_pred = float(tree.eval(variables, self._primitives))
                se += (y_pred - y_true) ** 2
            except Exception:
                se += 1e9
        return se / max(len(self._ys), 1)

    def predict(self, tree: Node) -> list[float]:
        """Return model predictions for every data point."""
        preds = []
        for variables in self._xs:
            try:
                preds.append(float(tree.eval(variables, self._primitives)))
            except Exception:
                preds.append(float("nan"))
        return preds

    @staticmethod
    def from_function(
        fn: Callable[[float], float],
        x_range: tuple[float, float] = (-3.0, 3.0),
        n_points: int = 40,
        noise: float = 0.0,
        seed: int = 0,
        **kwargs,
    ) -> "SymbolicRegressionDomain":
        """
        Convenience factory: generate data from a known function.

        Example
        -------
            import math
            domain = SymbolicRegressionDomain.from_function(
                lambda x: math.sin(x**2) + 2*x
            )
        """
        import random
        rng = random.Random(seed)
        xs = [x_range[0] + (x_range[1] - x_range[0]) * i / (n_points - 1)
              for i in range(n_points)]
        ys = [fn(x) + rng.gauss(0, noise) for x in xs]
        return SymbolicRegressionDomain(xs, ys, **kwargs)
