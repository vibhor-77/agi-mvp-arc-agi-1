"""
core/tree.py
============
Expression tree data structure used by the beam search engine.

A tree is either a leaf (variable or constant) or an internal node
(a named primitive applied to sub-trees). Primitives can be unary, binary,
or multi-ary, so each internal node has a list of children.

This single structure is reused across every domain:
  - Symbolic regression:  Node("compose", [Node("sin", ...), Node("sq", Leaf(var=0))])
  - ARC grid transforms:  Node("overlay", [Node("grot90", [Leaf(0)]), Leaf(0)])
  - CartPole policy:      Node("sq", [Leaf(var=2)])  →  θ²  used as force signal

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
    children : list[Node]
        Children for internal nodes; empty list for leaves.
    var_idx : int | None
        For variable leaves: which input variable (0-indexed).
    const : float | None
        For constant leaves: the numeric value.
    """

    __slots__ = ("op", "children", "var_idx", "const")

    def __init__(
        self,
        op: str | None = None,
        children: list["Node"] | None = None,
        var_idx: int | None = None,
        const: float | None = None,
    ) -> None:
        self.op = op
        self.children = children or []
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

        child_vals = [c.eval(variables, primitives) for c in self.children]
        return fn(*child_vals)

    # ------------------------------------------------------------------ #
    # Size / complexity (MDL proxy)                                        #
    # ------------------------------------------------------------------ #

    def size(self) -> int:
        """
        Count all nodes in the tree (self included).

        Used as the complexity term in MDL fitness:
            fitness = error + λ * tree.size()

        A leaf has size 1; Node("sin", [Leaf(x)]) has size 2.
        """
        return 1 + sum(c.size() for c in self.children)

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
        args = ", ".join(str(c) for c in self.children)
        return f"{self.op}({args})"

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
        for c in self.children:
            result.extend(c.all_subtrees())
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


def make_node(op: str, children: list[Node]) -> Node:
    """Return an internal (operator) node."""
    return Node(op=op, children=children)


def random_tree(
    op_list: list[str],
    n_vars: int,
    max_depth: int = 3,
    const_range: tuple[float, float] = (-3.0, 3.0),
    rng: random.Random | None = None,
    op_arities: dict[str, int] | None = None,
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
    op_arities : dict[str, int] | None
        Arity mapping for all primitives. Defaults to 1 for all if not provided.

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
    arity = op_arities.get(op, 1) if op_arities else 1
    children = [
        random_tree(op_list, n_vars, max_depth - 1, const_range, rng, op_arities)
        for _ in range(arity)
    ]
    return make_node(op, children)


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
    op_arities: dict[str, int] | None = None,
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
    op_arities : dict[str, int] | None
        Arities for all primitives.
    """
    if rng is None:
        rng = random.Random()

    tree = node.clone()
    r = rng.random()

    if r < 0.50:
        _replace_random_subtree(tree, op_list, n_vars, const_range, rng, op_arities)
    elif r < 0.80:
        if not _tweak_constant(tree, const_sigma, rng):
            # No constants found — fall back to subtree replacement
            _replace_random_subtree(tree, op_list, n_vars, const_range, rng, op_arities)
    else:
        # Wrap self with a new op
        op = rng.choice(op_list)
        arity = op_arities.get(op, 1) if op_arities else 1
        
        # Self becomes one child (randomly chosen position)
        self_idx = rng.randrange(arity)
        children = []
        for i in range(arity):
            if i == self_idx:
                children.append(tree)
            else:
                children.append(random_tree(op_list, n_vars, max_depth=1, const_range=const_range, rng=rng, op_arities=op_arities))
                
        wrapped = make_node(op, children)
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
        stack.extend(n.children)
    return result


def _replace_random_subtree(
    root: Node,
    op_list: list[str],
    n_vars: int,
    const_range: tuple[float, float],
    rng: random.Random,
    op_arities: dict[str, int] | None = None,
) -> None:
    """In-place: overwrite a randomly chosen node with a new random sub-tree."""
    nodes = _all_nodes_list(root)
    if not nodes:
        return
    target = rng.choice(nodes)
    new = random_tree(op_list, n_vars, max_depth=2, const_range=const_range, rng=rng, op_arities=op_arities)
    # Copy attributes in-place so references to `target` elsewhere remain valid
    target.op = new.op
    target.children = new.children
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
    # Collect (parent, child_idx) pairs
    pairs: list[tuple[Node, int]] = []

    def _collect(node: Node) -> None:
        for i, c in enumerate(node.children):
            pairs.append((node, i))
            _collect(c)

    _collect(root)
    if not pairs:
        return
    parent, idx = rng.choice(pairs)
    parent.children[idx] = subtree
