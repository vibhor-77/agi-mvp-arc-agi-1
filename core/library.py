"""
core/library.py
===============
Manages the extraction, storage, and persistence of discovered abstractions 
(frequent sub-trees) from successfully solved tasks. Central to the Wake-Sleep cycle.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Any

from .tree import Node
from .primitives import registry

def _default_dict_factory() -> dict[str, float]:
    return defaultdict(float)

class PrimitiveLibrary:
    """
    Manages loading, saving, and extracting learned primitives.
    """

    def __init__(self, filepath: str = "library.json"):
        self.filepath = filepath
        self.learned_ops: dict[str, dict] = {}
        # P(child_op | parent_op) weights
        self.transition_matrix: dict[str, dict[str, float]] = defaultdict(_default_dict_factory)

    def extract_from_tasks(self, task_trees: dict[str, Node], min_size: int = 3, min_tasks: int = 2) -> None:
        """
        Extract frequent sub-trees that span across multiple tasks.
        
        Parameters
        ----------
        task_trees : dict[str, Node]
            Mapping of task_name -> best_tree found.
        min_size : int
            Minimum AST node count to be considered a valuable abstraction.
            (e.g. >2 avoids pulling out trivial 1-op abstractions).
        min_tasks : int
            Minimum number of distinct tasks the sub-tree must appear in.
        """
        # mapping of string representation -> set of task names
        subtree_to_tasks = defaultdict(set)
        # mapping of string representation -> actual Node object
        subtree_to_node = {}

        for task_name, tree in task_trees.items():
            # 1. Update transition prior probabilities
            self._update_transitions(tree)
            
            # 2. Extract abstractions
            for sub in tree.all_subtrees():
                if sub.size() >= min_size:
                    key = str(sub)
                    subtree_to_tasks[key].add(task_name)
                    if key not in subtree_to_node:
                        subtree_to_node[key] = sub.clone()

        # Identify frequent abstractions
        for key, tasks in subtree_to_tasks.items():
            if len(tasks) >= min_tasks:
                self._add_to_library(subtree_to_node[key])
                
        self._normalize_transitions()

    def _update_transitions(self, tree: Node) -> None:
        """Walk the AST and count occurrences of (parent_op -> child_op)."""
        if tree.op is None:
            return
            
        for child in tree.children:
            if child.op is not None:
                self.transition_matrix[tree.op][child.op] += 1.0
            self._update_transitions(child)

    def _normalize_transitions(self) -> None:
        """Normalize counts into probabilities P(child|parent)."""
        for parent_op, children_counts in self.transition_matrix.items():
            total = sum(children_counts.values())
            if total > 0:
                for child_op in children_counts:
                    self.transition_matrix[parent_op][child_op] /= total

    def _add_to_library(self, node: Node) -> None:
        """Add a deeply cloned node to the library if it doesn't already exist."""
        key = str(node)
        # Check if already learned
        if any(v.get("expr") == key for v in self.learned_ops.values()):
            return

        op_name = f"lib_op_{len(self.learned_ops) + 1}"
        
        # Calculate arity by finding the max var_idx used + 1
        var_indices = [n.var_idx for n in node.all_subtrees() if n.var_idx is not None]
        arity = max(var_indices) + 1 if var_indices else 0
        
        # NOTE: some learned operations might take 0 arguments if they are just constant-generating trees.
        # But in ARC, usually everything operates on the main grid (var 0).
        if arity == 0:
            arity = 1 # Fallback 

        self.learned_ops[op_name] = {
            "expr": key,
            "arity": arity,
            "node": node
        }
        
    def register_all(self, domain: str = "arc") -> None:
        """Inject all learned primitives into the active environment registry."""
        for name, meta in self.learned_ops.items():
            if name not in registry:
                fn = self._make_callable(meta["node"])
                # We rename the function to debugging purposes
                fn.__name__ = name
                registry.register(
                    name=name,
                    fn=fn,
                    domain=domain,
                    description=f"Learned abstraction: {meta['expr']}",
                    arity=meta["arity"],
                    overwrite=True
                )

    def _make_callable(self, node: Node) -> Callable:
        """Create a lambda/function that evaluates the frozen node AST safely."""
        def _fn(*args: Any) -> Any:
            try:
                # Late binding for primitive mapping
                prims = {n: registry.get(n) for n in registry.names()}
                return node.eval(list(args), prims)
            except Exception:
                # If a sequence of ops fails inside the learned abstraction (e.g. index error on crop)
                # Fail gracefully by passing the first argument back unchanged, or None if error.
                return args[0] if args else None
                
        return _fn

    def save(self) -> None:
        """Persist the library expressions to disk."""
        data = {
            "library": {name: {"expr": meta["expr"], "arity": meta["arity"]} for name, meta in self.learned_ops.items()},
            "transitions": {k: dict(v) for k, v in self.transition_matrix.items()}
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, op_list: list[str] = None) -> None:
        """
        Load library expressions from disk and re-construct the ASTs.
        If op_list is provided, only loads ops whose keys are in that list.
        """
        if not os.path.exists(self.filepath):
            return
            
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except Exception:
            return
            
        if "library" in data:
            for name, meta in data["library"].items():
                if op_list is not None and name not in op_list:
                    continue
                node = Node.parse(meta["expr"])
                self.learned_ops[name] = {
                    "expr": meta["expr"],
                    "arity": meta["arity"],
                    "node": node
                }
                
        if "transitions" in data:
            for parent_op, v in data["transitions"].items():
                for child_op, prob in v.items():
                    self.transition_matrix[parent_op][child_op] = float(prob)
