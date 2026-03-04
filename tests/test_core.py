"""
tests/test_core.py
==================
Unit tests for core/ — primitive registry, expression tree, and beam search.

Tests run with either pytest or plain unittest:
    python -m pytest tests/test_core.py -v
    python -m unittest tests.test_core -v
"""
import math
import random
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.primitives import PrimitiveRegistry, registry
from core.tree import (
    Node, make_leaf_var, make_leaf_const, make_node,
    random_tree, mutate, crossover,
)
from core.search import BeamSearch, SearchConfig


class TestPrimitiveRegistry(unittest.TestCase):

    def test_register_and_get(self):
        r = PrimitiveRegistry()
        r.register("double", lambda x: x * 2, domain="test")
        self.assertEqual(r.get("double")(5), 10)

    def test_register_duplicate_raises(self):
        r = PrimitiveRegistry()
        r.register("f", lambda x: x, domain="test")
        with self.assertRaises(ValueError):
            r.register("f", lambda x: x + 1, domain="test")

    def test_overwrite_allowed(self):
        r = PrimitiveRegistry()
        r.register("f", lambda x: x, domain="test")
        r.register("f", lambda x: x + 1, domain="test", overwrite=True)
        self.assertEqual(r.get("f")(0), 1)

    def test_names_filtered_by_domain(self):
        r = PrimitiveRegistry()
        r.register("a", lambda x: x, domain="alpha")
        r.register("b", lambda x: x, domain="beta")
        r.register("c", lambda x: x, domain="alpha")
        self.assertEqual(set(r.names(domain="alpha")), {"a", "c"})
        self.assertEqual(r.names(domain="beta"), ["b"])

    def test_contains(self):
        r = PrimitiveRegistry()
        r.register("x", lambda v: v, domain="test")
        self.assertIn("x", r)
        self.assertNotIn("y", r)

    def test_len(self):
        r = PrimitiveRegistry()
        self.assertEqual(len(r), 0)
        r.register("a", lambda x: x, domain="d")
        self.assertEqual(len(r), 1)

    def test_register_many(self):
        r = PrimitiveRegistry()
        r.register_many({"f": lambda x: x, "g": lambda x: -x}, domain="d")
        self.assertEqual(len(r), 2)

    def test_global_registry_has_math(self):
        math_ops = registry.names(domain="math")
        self.assertIn("sin", math_ops)
        self.assertIn("sq", math_ops)
        self.assertGreaterEqual(len(math_ops), 15)

    def test_summary_runs(self):
        s = registry.summary()
        self.assertIn("[math]", s)


class TestNode(unittest.TestCase):

    def _prims(self):
        return {n: registry.get(n) for n in registry.names(domain="math")}

    def test_leaf_variable_eval(self):
        leaf = make_leaf_var(0)
        self.assertEqual(leaf.eval([42.0], {}), 42.0)

    def test_leaf_const_eval(self):
        leaf = make_leaf_const(3.14)
        self.assertAlmostEqual(leaf.eval([], {}), 3.14, places=9)

    def test_node_eval_sin(self):
        node = make_node("sin", [make_leaf_var(0)])
        p = self._prims()
        result = node.eval([math.pi / 2], p)
        self.assertAlmostEqual(result, 1.0, places=9)

    def test_nested_eval(self):
        tree = make_node("sin", [make_node("sq", [make_leaf_var(0)])])
        p = self._prims()
        self.assertAlmostEqual(tree.eval([1.0], p), math.sin(1.0), places=9)

    def test_size(self):
        leaf = make_leaf_var(0)
        self.assertEqual(leaf.size(), 1)
        node = make_node("sin", [leaf])
        self.assertEqual(node.size(), 2)
        deeper = make_node("cos", [node])
        self.assertEqual(deeper.size(), 3)

    def test_str_leaf_var(self):
        self.assertEqual(str(make_leaf_var(0)), "x")
        self.assertEqual(str(make_leaf_var(1)), "y")

    def test_str_node(self):
        tree = make_node("sin", [make_leaf_var(0)])
        self.assertEqual(str(tree), "sin(x)")

    def test_clone_independence(self):
        tree = make_node("sin", [make_leaf_var(0)])
        clone = tree.clone()
        clone.children[0].var_idx = 1
        self.assertEqual(tree.children[0].var_idx, 0)

    def test_all_subtrees_count(self):
        tree = make_node("sin", [make_node("sq", [make_leaf_var(0)])])
        subtrees = tree.all_subtrees()
        self.assertEqual(len(subtrees), 3)

    def test_fingerprint_scalar(self):
        tree = make_node("sq", [make_leaf_var(0)])
        p = self._prims()
        test_inputs = [[1.0], [2.0], [3.0]]
        fp = tree.fingerprint(test_inputs, p)
        self.assertEqual(fp, (1.0, 4.0, 9.0))

    def test_unknown_primitive_raises(self):
        tree = make_node("unknown_op", [make_leaf_var(0)])
        with self.assertRaises(KeyError):
            tree.eval([1.0], {})


class TestMutations(unittest.TestCase):

    def _ops(self):
        return registry.names(domain="math")

    def test_random_tree_is_node(self):
        tree = random_tree(self._ops(), n_vars=1, max_depth=3, rng=random.Random(0))
        self.assertIsInstance(tree, Node)

    def test_random_tree_depth_zero_is_leaf(self):
        tree = random_tree(self._ops(), n_vars=1, max_depth=0, rng=random.Random(0))
        self.assertIsNone(tree.op)

    def test_mutate_returns_new_tree(self):
        rng = random.Random(42)
        tree = make_node("sin", [make_leaf_var(0)])
        mutated = mutate(tree, self._ops(), n_vars=1, rng=rng)
        self.assertIsInstance(mutated, Node)
        self.assertEqual(str(tree), "sin(x)")

    def test_crossover_returns_node(self):
        rng = random.Random(7)
        a = make_node("sin", [make_leaf_var(0)])
        b = make_node("cos", [make_leaf_var(0)])
        result = crossover(a, b, rng)
        self.assertIsInstance(result, Node)


class TestBeamSearch(unittest.TestCase):

    def test_finds_identity(self):
        xs = [float(i) for i in range(-5, 6)]
        ys = list(xs)
        prims = {n: registry.get(n) for n in registry.names(domain="math")}

        def fitness(tree):
            preds = [tree.eval([x], prims) for x in xs]
            mse = sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(ys)
            return mse + 0.001 * tree.size()

        cfg = SearchConfig(beam_size=10, offspring=20, generations=50,
                           verbose=False, seed=0)
        result = BeamSearch(fitness, registry.names(domain="math"), n_vars=1, config=cfg).run()
        self.assertLess(result.best_fitness, 0.1)

    def test_history_populated(self):
        xs, ys = [1.0, 2.0, 3.0], [1.0, 4.0, 9.0]
        prims = {n: registry.get(n) for n in registry.names(domain="math")}

        def fitness(tree):
            try:
                preds = [tree.eval([x], prims) for x in xs]
                return sum((p-y)**2 for p, y in zip(preds, ys))/len(ys) + 0.01*tree.size()
            except Exception:
                return 1e9

        cfg = SearchConfig(beam_size=5, offspring=10, generations=5, verbose=False, seed=0)
        result = BeamSearch(fitness, registry.names(domain="math"), n_vars=1, config=cfg).run()
        self.assertGreater(len(result.history), 0)
        self.assertIn("gen", result.history[0])

    def test_result_fields(self):
        cfg = SearchConfig(beam_size=5, offspring=5, generations=3, verbose=False, seed=0)
        result = BeamSearch(lambda t: float(t.size()),
                            registry.names(domain="math"), n_vars=1, config=cfg).run()
        self.assertIsInstance(result.best_tree, Node)
        self.assertIsInstance(result.best_fitness, float)
        self.assertGreaterEqual(result.elapsed_s, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
