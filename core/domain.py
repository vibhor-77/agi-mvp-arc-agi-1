"""
core/domain.py
==============
Abstract base class that every domain must implement.

To add a completely new domain (ARC-AGI-2, Zork, NetHack, MuJoCo, ...):
  1. Subclass ``Domain``
  2. Implement the three required methods
  3. Optionally override ``on_result()`` for domain-specific post-processing

That's it.  The beam search engine, tree representation, and primitive
registry all work unchanged.

Design Rationale
----------------
The three required methods encode the minimal interface:

  primitive_names() -> list[str]
      Which primitives from the registry are in scope for this domain.
      Lets you run symbolic regression on math ops only, ARC on grid ops
      only, or any combination.

  fitness(tree) -> float
      How good is this expression tree?  Lower is better.
      This is the only place domain-specific logic enters the search.

  n_vars() -> int
      How many input variables does a tree leaf refer to?
      e.g. 1 for univariate regression, 4 for CartPole state, 1 for ARC grids.

Example: Adding a Zork domain
------------------------------
    class ZorkDomain(Domain):
        def primitive_names(self):
            return registry.names(domain="zork")   # go_north, pick_up, ...

        def fitness(self, tree):
            # Simulate N steps of Zork using the tree as a policy
            # Return negative mean score (lower = better)
            ...

        def n_vars(self):
            return len(ZORK_STATE_FEATURES)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .tree import Node
from .search import BeamSearch, SearchConfig, SearchResult


class Domain(ABC):
    """
    Abstract base for all problem domains.

    Subclass this to plug a new problem into the beam search engine.
    """

    # ------------------------------------------------------------------ #
    # Required interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def primitive_names(self) -> list[str]:
        """
        Return the list of primitive names available in this domain.

        These names must all be registered in the module-level
        ``core.primitives.registry`` before the search starts.

        Returns
        -------
        list[str]
        """

    @abstractmethod
    def fitness(self, tree: Node) -> float:
        """
        Score one expression tree.  Lower is better.

        This is the only domain-specific method the search engine calls.
        It should be fast — it will be called  beam_size * offspring * generations  times.

        Parameters
        ----------
        tree : Node
            Candidate expression tree to evaluate.

        Returns
        -------
        float
            Non-negative score.  0.0 = perfect solution.
        """

    @abstractmethod
    def n_vars(self) -> int:
        """
        Number of input variables a leaf node can reference.

        For univariate regression: 1.
        For CartPole (x, ẋ, θ, θ̇): 4.
        For ARC (single grid input): 1.
        """

    # ------------------------------------------------------------------ #
    # Optional hooks                                                       #
    # ------------------------------------------------------------------ #

    def primitives_dict(self) -> dict[str, Any]:
        """
        Return {name: callable} for all primitives in this domain.

        Used internally for tree.eval().  Override if your domain needs
        a non-standard lookup (e.g. dynamically generated primitives).
        """
        from .primitives import registry
        return {name: registry.get(name) for name in self.primitive_names()}

    def on_result(self, result: SearchResult) -> None:
        """
        Called after search completes.  Override for domain-specific
        post-processing: saving results, printing interpretable summaries,
        running the best policy on a real environment, etc.

        Default: no-op.
        """

    def description(self) -> str:
        """
        One-line human-readable description of the domain.
        Used in logging and the dashboard.  Override to customise.
        """
        return self.__class__.__name__

    # ------------------------------------------------------------------ #
    # Convenience: run the search                                          #
    # ------------------------------------------------------------------ #

    def solve(self, config: SearchConfig | None = None) -> SearchResult:
        """
        Run beam search for this domain with the given configuration.

        Parameters
        ----------
        config : SearchConfig | None
            Search hyperparameters.  Uses defaults if None.

        Returns
        -------
        SearchResult
        """
        from .primitives import registry
        op_arities = {name: registry.arity(name) for name in self.primitive_names()}
        searcher = BeamSearch(
            fitness_fn=self.fitness,
            op_list=self.primitive_names(),
            n_vars=self.n_vars(),
            config=config or SearchConfig(),
            op_arities=op_arities,
        )
        result = searcher.run()
        self.on_result(result)
        return result
