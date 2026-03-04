"""
core/primitives.py
==================
Central registry for all domain primitives.

Architecture
------------
A "primitive" is any callable that takes one value and returns one value.
It could be a math function (sin, sq), a grid transformation (rotate90),
a control function (threshold), or anything you define.

To add primitives for a new domain (e.g. ARC-AGI-2, Zork, NetHack):
  1. Create a file  domains/<your_domain>/primitives.py
  2. Define your callables
  3. Call  registry.register(name, fn, domain="your_domain", arity=1)
  4. The beam search will automatically include them when you pass domain=

No changes to core/ are ever needed when adding new domains.

Example
-------
    from core.primitives import PrimitiveRegistry

    registry = PrimitiveRegistry()
    registry.register("double", lambda x: x * 2, domain="math")
    registry.register("negate", lambda x: -x,    domain="math")

    fn = registry.get("double")   # -> callable
    math_ops = registry.names(domain="math")   # -> ["double", "negate"]
"""
from __future__ import annotations

from typing import Callable, Any
import math


# ---------------------------------------------------------------------------
# PrimitiveRegistry
# ---------------------------------------------------------------------------

class PrimitiveRegistry:
    """
    A dict-like store mapping  name -> (callable, metadata).

    Primitives can have different arities (number of arguments).
    By default, they are assumed to be unary (arity=1).

    Parameters
    ----------
    None — instantiate and then call .register() to populate.
    """

    def __init__(self) -> None:
        # name -> {"fn": callable, "domain": str, "description": str}
        self._store: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(
        self,
        name: str,
        fn: Callable[[Any], Any],
        *,
        domain: str = "general",
        description: str = "",
        arity: int = 1,
        overwrite: bool = False,
    ) -> "PrimitiveRegistry":
        """
        Register a primitive.

        Parameters
        ----------
        name : str
            Unique identifier used in expression trees and output strings.
        fn : Callable
            Callable matching the specified arity.
        domain : str
            Logical grouping (e.g. "math", "arc", "cartpole").
            Used for filtering when building domain-specific op lists.
        description : str
            Human-readable description shown in documentation / debug output.
        overwrite : bool
            If False (default), raises ValueError if name already exists.

        Returns
        -------
        self  (enables chaining)
        """
        if name in self._store and not overwrite:
            raise ValueError(
                f"Primitive '{name}' already registered. "
                "Pass overwrite=True to replace it."
            )
        self._store[name] = {
            "fn": fn,
            "domain": domain,
            "description": description,
            "arity": arity,
        }
        return self

    def register_many(
        self,
        mapping: dict[str, Callable],
        *,
        domain: str = "general",
        overwrite: bool = False,
    ) -> "PrimitiveRegistry":
        """Register multiple primitives from a name->callable dict."""
        for name, fn in mapping.items():
            self.register(name, fn, domain=domain, overwrite=overwrite)
        return self

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> Callable:
        """Return the callable for *name*.  Raises KeyError if not found."""
        return self._store[name]["fn"]

    def __getitem__(self, name: str) -> Callable:
        return self.get(name)

    def arity(self, name: str) -> int:
        """Return the arity (number of arguments) of primitive *name*."""
        return self._store[name]["arity"]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def names(self, domain: str | None = None) -> list[str]:
        """
        Return all registered primitive names, optionally filtered by domain.

        Parameters
        ----------
        domain : str | None
            If given, return only primitives whose domain matches.
        """
        if domain is None:
            return list(self._store.keys())
        return [n for n, meta in self._store.items() if meta["domain"] == domain]

    def domains(self) -> list[str]:
        """Return all distinct domain names."""
        return sorted(set(m["domain"] for m in self._store.values()))

    def info(self, name: str) -> dict:
        """Return full metadata dict for a primitive."""
        return dict(self._store[name])

    def summary(self) -> str:
        """Multi-line summary of all registered primitives."""
        lines = []
        for domain in self.domains():
            lines.append(f"[{domain}]")
            for name in self.names(domain=domain):
                desc = self._store[name]["description"] or "(no description)"
                lines.append(f"  {name:20s}  {desc}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"PrimitiveRegistry({len(self)} primitives, domains={self.domains()})"


# ---------------------------------------------------------------------------
# Module-level singleton — import this in domain files to register into it
# ---------------------------------------------------------------------------

registry = PrimitiveRegistry()


# ---------------------------------------------------------------------------
# Built-in math primitives (domain="math")
# Always available for symbolic regression / CartPole / general use.
# ---------------------------------------------------------------------------

def _safe(fn: Callable) -> Callable:
    """Wrap a function to return 0.0 on math domain errors."""
    def _wrapped(x):
        try:
            v = fn(x)
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return 0.0
            return v
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            return 0.0
    _wrapped.__name__ = getattr(fn, "__name__", "safe_fn")
    return _wrapped


_MATH_PRIMITIVES: dict[str, tuple[Callable, str]] = {
    # Arithmetic
    "neg":   (lambda x: -x,                           "Negate: -x"),
    "sq":    (lambda x: x * x,                        "Square: x²"),
    "cube":  (lambda x: x * x * x,                    "Cube: x³"),
    "sqrt":  (_safe(lambda x: math.sqrt(abs(x))),     "Square root: √|x|"),
    "abs":   (abs,                                     "Absolute value"),
    "inv":   (_safe(lambda x: 1.0 / x if abs(x) > 1e-9 else 0.0), "Reciprocal: 1/x"),
    # Trig
    "sin":   (_safe(math.sin),                        "Sine"),
    "cos":   (_safe(math.cos),                        "Cosine"),
    "tan":   (_safe(math.tan),                        "Tangent"),
    "asin":  (_safe(math.asin),                       "Arcsine"),
    "acos":  (_safe(math.acos),                       "Arccosine"),
    "atan":  (_safe(math.atan),                       "Arctangent"),
    # Exp / log
    "exp":   (_safe(lambda x: math.exp(min(x, 50))), "Exponential: eˣ"),
    "log":   (_safe(lambda x: math.log(abs(x) + 1e-9)), "Natural log: ln|x|"),
    "log2":  (_safe(lambda x: math.log2(abs(x) + 1e-9)), "Log base 2"),
    "log10": (_safe(lambda x: math.log10(abs(x) + 1e-9)), "Log base 10"),
    # Rounding / steps
    "floor": (math.floor,                             "Floor"),
    "ceil":  (math.ceil,                              "Ceiling"),
    "sign":  (lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0), "Sign"),
    "relu":  (lambda x: max(0.0, x),                  "ReLU: max(0, x)"),
    "tanh":  (math.tanh,                              "Hyperbolic tangent"),
}

for _name, (_fn, _desc) in _MATH_PRIMITIVES.items():
    registry.register(_name, _fn, domain="math", description=_desc)
