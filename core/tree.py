"""
core/tree.py
============
Expression tree data structure used by the beam search engine.

A tree is either a leaf (variable or constant) or an internal node
(a named primitive applied to one child sub-tree).  All primitives
are *unary*, so each internal node has exactly one child.

This single structure is reused across every domain:
  - Symbolic regression:  Node("sin", Node("sq", Leaf(var=0)))  →  sin(x²)
  - ARC grid transforms:  Node("grot90", Node("grefl_h", Leaf(var=0))) → rotate(reflect(grid))
  - CartPole policy:      Node("sq", Leaf(var=2))  →  θ²  used as force signal

Design decisions
----------------
* Only unary nodes — keeps mutation/crossover simple and the search space
  well-defined.  Binary ops can always be built as two-argument primitives
  registered at the domain level if needed.
* Deep-copy on clone() — trees are treated as immutable values during search;
  mutations return new trees.
* Semantic fingerprinting — evaluate on fixed test inputs to detect
  functionally-equivalent trees without symbolic simplification.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    One node in an expression tree.

    Attributes
    ----------
    op : str | None
        Primitive name (must exist in the registry used for evaluation),
        or None for leaves.
    child : Node | None
        Single child for internal nodes; None for leaves.
    var_idx : int | None
        For variable leaves: which input variable (0-indexed).
    const : float | None
        For constant leaves: the numeric value.
    """

    __slots__ = ("op", "child", "var_idx", "const")

    def __init__(
        self,
        op: str | None = None,
        child: "Node | None" = None,
        var_idx: int | None = None,
        const: float | None = None,
    ) -> None:
        self.op = op
        self.child = child
        self.var_idx = var_idx
        self.const = const

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def eval(self, variables: list[Any], primitives: dict[str, Callable]) -> Any:
        """
        Recursively evaluate the tree.

        Parameters
        ----------
        variables : list
            One entry per input variable.  ``variables[i]`` is used when
            a leaf has ``var_idx == i``.
        primitives : dict[str, Callable]
            Maps primitive names to callables.  Typically obtained from
            ``PrimitiveRegistry.names()`` + ``registry.get()``, or simply
            passed as a pre-built {name: fn} dict.

        Returns
        -------
        Any
            Whatever the composed primitives produce — float for math
            domains, Grid (list[list[int]]) for ARC, etc.
        """
        if self.op is None:
            # Leaf node
            if self.var_idx is not None:
                return variables[self.var_idx]
            return self.const

        fn = primitives.get(self.op)
        if fn is None:
            raise KeyError(f"Unknown primitive '{self.op}'. Check your registry.")

        child_val = self.child.eval(variables, primitives)
        return fn(child_val)

    # ------------------------------------------------------------------ #
    # Size / complexity (MDL proxy)                                        #
    # ------------------------------------------------------------------ #

    def size(self) -> int:
        """
        Count all nodes in the tree (self included).

        Used as the complexity term in MDL fitness:
            fitness = error + λ * tree.size()

        A leaf has size 1; Node("sin", Leaf(x)) has size 2.
        """
        if self.child is None:
            return 1
        return 1 + self.child.size()

    # ------------------------------------------------------------------ #
    # String representation                                                #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        if self.op is None:
            if self.var_idx is not None:
                # Nice variable names for up to 8 variables
                _NAMES = ["x", "y", "z", "w", "x0", "x1", "x2", "x3"]
                return _NAMES[self.var_idx] if self.var_idx < len(_NAMES) else f"v{self.var_idx}"
            return f"{self.const:.4g}"
        return f"{self.op}({self.child})"

    def __repr__(self) -> str:
        return f"Node({self})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Node) and str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))

    # ------------------------------------------------------------------ #
    # Deep copy                                                            #
    # ------------------------------------------------------------------ #

    def clone(self) -> "Node":
        """Return an independent deep copy."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    # Sub-tree enumeration (used by library learning)                     #
    # ------------------------------------------------------------------ #

    def all_subtrees(self) -> list["Node"]:
        """
        Return a list of all sub-trees rooted at this node and below,
        in pre-order (self first).

        Useful for Level-2 primitive discovery: collect sub-trees from
        all solutions, find frequent ones, promote to new primitives.
        """
        result: list[Node] = [self]
        if self.child is not None:
            result.extend(self.child.all_subtrees())
        return result

    # ------------------------------------------------------------------ #
    # Semantic fingerprint (deduplication)                                 #
    # ------------------------------------------------------------------ #

    def fingerprint(
        self,
        test_inputs: list[list[Any]],
        primitives: dict[str, Callable],
    ) -> tuple:
        """
        Evaluate on a fixed set of test inputs and return a rounded tuple.

        Two trees with the same fingerprint are *semantically equivalent*
        (they produce identical outputs on all test inputs).  Use this to
        avoid promoting duplicate primitives during library learning.

        Parameters
        ----------
        test_inputs : list[list]
            Each element is a ``variables`` list passed to ``eval()``.
        primitives : dict
            Callable lookup (same as for ``eval()``).

        Returns
        -------
        tuple
            Rounded output values.  Empty tuple on evaluation error.
        """
        try:
            vals = []
            for inp in test_inputs:
                v = self.eval(inp, primitives)
                if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                    vals.append(round(float(v), 6))
                else:
                    vals.append(None)
            return tuple(vals)
        except Exception:
            return ()


# ---------------------------------------------------------------------------
# Tree factories
# ---------------------------------------------------------------------------

def make_leaf_var(var_idx: int) -> Node:
    """Return a variable leaf node."""
    return Node(var_idx=var_idx)


def make_leaf_const(value: float) -> Node:
    """Return a constant leaf node."""
    return Node(const=value)


def make_node(op: str, child: Node) -> Node:
    """Return an internal (operator) node."""
    return Node(op=op, child=child)


def random_tree(
    op_list: list[str],
    n_vars: int,
    max_depth: int = 3,
    const_range: tuple[float, float] = (-3.0, 3.0),
    rng: random.Random | None = None,
) -> Node:
    """
    Build a random expression tree.

    Parameters
    ----------
    op_list : list[str]
        Primitive names to sample operators from.
    n_vars : int
        Number of input variables (variables are 0-indexed).
    max_depth : int
        Maximum tree depth.  A depth-0 tree is always a leaf.
    const_range : (float, float)
        Range for random constant leaves.  Only used for math domains;
        for grid domains constants are unused (all leaves are variables).
    rng : random.Random | None
        Optional RNG instance for reproducibility.

    Returns
    -------
    Node
    """
    if rng is None:
        rng = random.Random()

    # Always emit a leaf at depth 0 or with 30% probability
    if max_depth == 0 or rng.random() < 0.30:
        if n_vars > 0 and rng.random() < 0.75:
            return make_leaf_var(rng.randrange(n_vars))
        val = round(rng.uniform(*const_range), 3)
        return make_leaf_const(val)

    op = rng.choice(op_list)
    child = random_tree(op_list, n_vars, max_depth - 1, const_range, rng)
    return make_node(op, child)


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def mutate(
    node: Node,
    op_list: list[str],
    n_vars: int,
    const_range: tuple[float, float] = (-3.0, 3.0),
    const_sigma: float = 0.5,
    rng: random.Random | None = None,
) -> Node:
    """
    Return a mutated copy of *node*.

    Randomly applies one of three strategies (equal probability):
    1. **Subtree replacement** — pick a random node, replace with a new
       random sub-tree of depth ≤ 2.
    2. **Constant perturbation** — add Gaussian noise to a random constant leaf.
    3. **Op insertion** — wrap a random node with a new operator.

    Parameters
    ----------
    node : Node
        Tree to mutate (not modified in place).
    op_list : list[str]
        Available operator names.
    n_vars : int
        Number of input variables.
    const_range : (float, float)
        Range for newly created constants.
    const_sigma : float
        Standard deviation for Gaussian perturbation of constants.
    rng : random.Random | None
        Optional RNG for reproducibility.
    """
    if rng is None:
        rng = random.Random()

    tree = node.clone()
    r = rng.random()

    if r < 0.50:
        _replace_random_subtree(tree, op_list, n_vars, const_range, rng)
    elif r < 0.80:
        if not _tweak_constant(tree, const_sigma, rng):
            # No constants found — fall back to subtree replacement
            _replace_random_subtree(tree, op_list, n_vars, const_range, rng)
    else:
        # Wrap self with a new op
        op = rng.choice(op_list)
        wrapped = make_node(op, tree)
        return wrapped

    return tree


def crossover(a: Node, b: Node, rng: random.Random | None = None) -> Node:
    """
    Subtree crossover: clone *a*, then splice a random sub-tree from *b*
    into a random position in the clone.

    This combines structural information from two elite candidates,
    analogous to biological recombination.
    """
    if rng is None:
        rng = random.Random()

    child = a.clone()
    donor_subtrees = b.all_subtrees()
    if not donor_subtrees:
        return child
    donor = rng.choice(donor_subtrees).clone()
    _splice_at_random(child, donor, rng)
    return child


# ---------------------------------------------------------------------------
# Private mutation helpers
# ---------------------------------------------------------------------------

def _all_nodes_list(root: Node) -> list[Node]:
    """Flat list of all nodes in the tree."""
    result: list[Node] = []
    stack = [root]
    while stack:
        n = stack.pop()
        result.append(n)
        if n.child is not None:
            stack.append(n.child)
    return result


def _replace_random_subtree(
    root: Node,
    op_list: list[str],
    n_vars: int,
    const_range: tuple[float, float],
    rng: random.Random,
) -> None:
    """In-place: overwrite a randomly chosen node with a new random sub-tree."""
    nodes = _all_nodes_list(root)
    if not nodes:
        return
    target = rng.choice(nodes)
    new = random_tree(op_list, n_vars, max_depth=2, const_range=const_range, rng=rng)
    # Copy attributes in-place so references to `target` elsewhere remain valid
    target.op = new.op
    target.child = new.child
    target.var_idx = new.var_idx
    target.const = new.const


def _tweak_constant(root: Node, sigma: float, rng: random.Random) -> bool:
    """In-place: add Gaussian noise to a random constant leaf.  Returns True if found."""
    constants = [n for n in _all_nodes_list(root) if n.const is not None]
    if not constants:
        return False
    n = rng.choice(constants)
    n.const = round(n.const + rng.gauss(0, sigma), 4)
    return True


def _splice_at_random(root: Node, subtree: Node, rng: random.Random) -> None:
    """In-place: replace a random node (excluding root) with *subtree*."""
    # Collect (parent, 'child') pairs
    pairs: list[Node] = []

    def _collect(node: Node) -> None:
        if node.child is not None:
            pairs.append(node)
            _collect(node.child)

    _collect(root)
    if not pairs:
        return
    parent = rng.choice(pairs)
    parent.child = subtree
