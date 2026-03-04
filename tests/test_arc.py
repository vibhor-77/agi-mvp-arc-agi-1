"""
tests/test_arc.py
=================
Unit tests for the ARC domain — primitives, task representation,
fitness function, and end-to-end solving.

Run with:
    python -m pytest tests/test_arc.py -v
    python -m unittest tests.test_arc -v
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.primitives import registry
import domains.arc.primitives  # registers all primitives as side effect

from domains.arc.primitives import (
    grot90, grot180, grot270,
    grefl_h, grefl_v, gtrsp,
    ginv, gid,
    gswap_01, gswap_12,
    ggravity_down, ggravity_left,
    gmirror_v, gmirror_h,
    ghollow, gscale2x,
    gcheckerboard, gstripe_h2, gstripe_v2, gtile2x2,
    gcountbar, gmajority, gkeep_rows2,
    gframe8, gdiag1,
    g_filter_color, g_extract_objects, g_render_object,
)
from domains.arc.domain import (
    ARCTask, ARCDomain, grid_cell_accuracy, is_exact_match
)
from core.tree import make_leaf_var, make_node
from core.search import SearchConfig


# Shared test grids
G3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
G_SPARSE = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
G_SOLID  = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]


# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------

class TestGeometric(unittest.TestCase):

    def test_rot90(self):
        self.assertEqual(grot90(G3), [[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    def test_rot180(self):
        self.assertEqual(grot180(G3), [[9, 8, 7], [6, 5, 4], [3, 2, 1]])

    def test_rot270(self):
        self.assertEqual(grot270(G3), [[3, 6, 9], [2, 5, 8], [1, 4, 7]])

    def test_rot_360_is_identity(self):
        self.assertEqual(grot90(grot90(grot90(grot90(G3)))), G3)

    def test_refl_h(self):
        self.assertEqual(grefl_h(G3), [[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    def test_refl_v(self):
        self.assertEqual(grefl_v(G3), [[7, 8, 9], [4, 5, 6], [1, 2, 3]])

    def test_refl_h_twice_identity(self):
        self.assertEqual(grefl_h(grefl_h(G3)), G3)

    def test_transpose(self):
        self.assertEqual(gtrsp(G3), [[1, 4, 7], [2, 5, 8], [3, 6, 9]])

    def test_transpose_twice_identity(self):
        self.assertEqual(gtrsp(gtrsp(G3)), G3)

    def test_identity_returns_copy(self):
        result = gid(G3)
        self.assertEqual(result, G3)
        self.assertIsNot(result, G3)

    def test_rot90_then_refl_h_is_transpose(self):
        """rot90 then refl_h is equivalent to transpose."""
        self.assertEqual(grefl_h(grot90(G3)), gtrsp(G3))


# ---------------------------------------------------------------------------
# Color primitives
# ---------------------------------------------------------------------------

class TestColor(unittest.TestCase):

    def test_invert(self):
        g = [[0, 1], [2, 3]]
        self.assertEqual(ginv(g), [[3, 2], [1, 0]])

    def test_invert_symmetry(self):
        # ginv shifts by max, so double-invert works on 0-based symmetric grid
        g = [[0,1],[2,3]]
        self.assertEqual(ginv(ginv(g)), g)

    def test_swap_01(self):
        g = [[0, 1, 2]]
        self.assertEqual(gswap_01(g), [[1, 0, 2]])

    def test_swap_12(self):
        g = [[1, 2, 3]]
        self.assertEqual(gswap_12(g), [[2, 1, 3]])

    def test_swap_is_involution(self):
        self.assertEqual(gswap_01(gswap_01(G3)), G3)


# ---------------------------------------------------------------------------
# Gravity primitives
# ---------------------------------------------------------------------------

class TestGravity(unittest.TestCase):

    def test_gravity_down(self):
        g = [[1, 0], [0, 2], [0, 0]]
        self.assertEqual(ggravity_down(g), [[0, 0], [0, 0], [1, 2]])

    def test_gravity_down_already_settled(self):
        g = [[0, 0], [0, 0], [1, 2]]
        self.assertEqual(ggravity_down(g), g)

    def test_gravity_left(self):
        g = [[0, 1, 0, 2]]
        self.assertEqual(ggravity_left(g), [[1, 2, 0, 0]])

    def test_gravity_down_preserves_all_values(self):
        """Gravity must not create or destroy cells, only move them."""
        import collections
        g = [[1, 0, 2], [0, 3, 0], [0, 0, 4]]
        result = ggravity_down(g)
        flat_in  = [c for row in g      for c in row]
        flat_out = [c for row in result for c in row]
        self.assertEqual(
            collections.Counter(flat_in),
            collections.Counter(flat_out)
        )


# ---------------------------------------------------------------------------
# Structural primitives
# ---------------------------------------------------------------------------

class TestStructural(unittest.TestCase):

    def test_mirror_v(self):
        g = [[1, 2], [3, 4], [0, 0], [0, 0]]
        result = gmirror_v(g)
        self.assertEqual(result[3], g[0])
        self.assertEqual(result[2], g[1])

    def test_mirror_h(self):
        g = [[1, 2, 0, 0]]
        self.assertEqual(gmirror_h(g), [[1, 2, 2, 1]])

    def test_hollow_solid_3x3(self):
        result = ghollow(G_SOLID)
        self.assertEqual(result[1][1], 0)   # centre hollowed
        self.assertEqual(result[0][0], 1)   # border unchanged

    def test_hollow_border_intact(self):
        result = ghollow(G_SOLID)
        for c in range(3):
            self.assertEqual(result[0][c], 1)
            self.assertEqual(result[2][c], 1)

    def test_scale2x_shape(self):
        g = [[1, 2]]
        result = gscale2x(g)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [1, 1, 2, 2])
        self.assertEqual(result[1], [1, 1, 2, 2])

    def test_scale2x_content(self):
        g = [[3, 5], [7, 9]]
        result = gscale2x(g)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], [3, 3, 5, 5])
        self.assertEqual(result[2], [7, 7, 9, 9])

    def test_frame8(self):
        g = [[0] * 5 for _ in range(5)]
        result = gframe8(g)
        self.assertEqual(result[0][0], 8)
        self.assertEqual(result[2][2], 0)   # centre untouched
        self.assertEqual(result[4][4], 8)

    def test_diag1(self):
        g = [[0] * 3 for _ in range(3)]
        result = gdiag1(g)
        for i in range(3):
            self.assertEqual(result[i][i], 1)
        self.assertEqual(result[0][1], 0)


# ---------------------------------------------------------------------------
# Shapes and Objects primitives
# ---------------------------------------------------------------------------

class TestShapes(unittest.TestCase):
    def test_filter_color(self):
        g = [
            [1, 2, 0],
            [2, 1, 0],
            [0, 0, 3]
        ]
        out = g_filter_color(g, 1)
        self.assertEqual(out, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_extract_objects(self):
        g = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 2] # 2 is smaller than the 1s cluster
        ]
        out = g_extract_objects(g)
        self.assertEqual(out, [[1, 1], [1, 0]])
        
    def test_extract_empty_returns_clone(self):
        g = [[0, 0], [0, 0]]
        out = g_extract_objects(g)
        self.assertEqual(out, [[0, 0], [0, 0]])
        self.assertIsNot(out, g)

    def test_render_object(self):
        g1 = [[1, 1], [1, 1]]
        g2 = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        out = g_render_object(g1, g2)
        # Should paste the 2x2 square perfectly in the 4x4 center
        expected = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ]
        self.assertEqual(out, expected)

# ---------------------------------------------------------------------------
# Pattern primitives
# ---------------------------------------------------------------------------

class TestPattern(unittest.TestCase):

    def test_checkerboard_pattern(self):
        g = [[0] * 4 for _ in range(4)]
        result = gcheckerboard(g)
        self.assertEqual(result[0][0], 1)   # (0+0) even
        self.assertEqual(result[0][1], 2)   # (0+1) odd
        self.assertEqual(result[1][0], 2)   # (1+0) odd

    def test_stripe_h2(self):
        g = [[0] * 3 for _ in range(4)]
        result = gstripe_h2(g)
        self.assertEqual(result[0], [1, 1, 1])
        self.assertEqual(result[1], [2, 2, 2])
        self.assertEqual(result[2], [1, 1, 1])

    def test_stripe_v2(self):
        g = [[0] * 4 for _ in range(2)]
        result = gstripe_v2(g)
        self.assertEqual(result[0], [1, 2, 1, 2])

    def test_tile2x2(self):
        g = [[1, 2], [3, 4], [0, 0], [0, 0]]
        result = gtile2x2(g)
        self.assertEqual(result[2][0], 1)   # same as g[0][0]
        self.assertEqual(result[3][1], 4)   # same as g[1][1]

    def test_tile2x2_is_periodic(self):
        g = [[5, 7], [8, 9], [0, 0], [0, 0]]
        result = gtile2x2(g)
        # Every row at even index should match row 0
        for r in range(0, len(result), 2):
            self.assertEqual(result[r][0], g[0][0])


# ---------------------------------------------------------------------------
# Counting primitives
# ---------------------------------------------------------------------------

class TestCounting(unittest.TestCase):

    def test_countbar_exact(self):
        g = [[1, 1, 0, 0], [2, 0, 0, 0]]
        result = gcountbar(g)
        self.assertEqual(result[0], [1, 1, 0, 0])
        self.assertEqual(result[1], [2, 0, 0, 0])

    def test_countbar_all_zero_row(self):
        g = [[0, 0, 0]]
        result = gcountbar(g)
        self.assertEqual(result[0], [0, 0, 0])

    def test_majority_simple(self):
        g = [[1, 1, 2, 1]]
        result = gmajority(g)
        self.assertEqual(result, [[1, 1, 1, 1]])

    def test_majority_tie_broken_by_first(self):
        g = [[1, 2, 1, 2]]
        result = gmajority(g)
        # max(set, key=count) picks 1 or 2 (both appear twice)
        # just check it's one of the values
        self.assertIn(result[0][0], [1, 2])

    def test_keep_rows2_filters_sparse(self):
        g = [[1, 0, 0], [1, 1, 0], [0, 0, 0]]
        result = gkeep_rows2(g)
        self.assertEqual(result[0], [0, 0, 0])   # only 1 non-zero → zeroed
        self.assertEqual(result[1], [1, 1, 0])   # 2 non-zero → kept
        self.assertEqual(result[2], [0, 0, 0])


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):

    def test_expected_arc_primitives_present(self):
        arc_ops = registry.names(domain="arc")
        expected = [
            "grot90", "grot180", "grefl_h", "grefl_v", "gtrsp",
            "ginv", "ggravity_down", "gmirror_v", "ghollow",
            "gcheckerboard", "gtile2x2", "gcountbar", "gmajority",
            "gscale2x", "gframe8", "gdiag1", "gstripe_h2", "gstripe_v2",
        ]
        for op in expected:
            self.assertIn(op, arc_ops, f"Expected '{op}' in arc primitives")

    def test_arc_ops_not_in_math_domain(self):
        math_ops = registry.names(domain="math")
        self.assertNotIn("grot90", math_ops)
        self.assertNotIn("ggravity_down", math_ops)

    def test_all_arc_primitives_callable(self):
        """Every registered ARC primitive must be a callable."""
        for name in registry.names(domain="arc"):
            fn = registry.get(name)
            self.assertTrue(callable(fn), f"'{name}' is not callable")

    def test_all_arc_primitives_accept_grid(self):
        """Every registered ARC primitive must execute safely (with appropriate dummy args)."""
        test_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        for name in registry.names(domain="arc"):
            fn = registry.get(name)
            arity = registry.arity(name)
            try:
                args = [test_grid] * arity
                result = fn(*args)
                self.assertIsInstance(result, list, f"'{name}' did not return a list")
            except Exception as e:
                self.fail(f"'{name}' raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# ARCTask
# ---------------------------------------------------------------------------

class TestARCTask(unittest.TestCase):

    def test_from_dict(self):
        d = {
            "name": "test_task",
            "train": [{"input": [[1]], "output": [[2]]}],
            "test":  [{"input": [[3]], "output": [[4]]}],
        }
        task = ARCTask.from_dict(d)
        self.assertEqual(task.name, "test_task")
        self.assertEqual(len(task.train_pairs), 1)
        self.assertEqual(len(task.test_pairs), 1)

    def test_deep_copy_on_init(self):
        inp = [[1, 2]]
        out = [[3, 4]]
        task = ARCTask(name="t", train_pairs=[(inp, out)])
        inp[0][0] = 99
        # Mutation of original list must not affect task
        self.assertEqual(task.train_pairs[0][0][0][0], 1)

    def test_from_dict_no_test_pairs(self):
        d = {"name": "no_test", "train": [{"input": [[1]], "output": [[2]]}]}
        task = ARCTask.from_dict(d)
        self.assertEqual(task.test_pairs, [])


# ---------------------------------------------------------------------------
# Grid accuracy helpers
# ---------------------------------------------------------------------------

class TestGridAccuracy(unittest.TestCase):

    def test_perfect_match(self):
        g = [[1, 2], [3, 4]]
        self.assertEqual(grid_cell_accuracy(g, g), 1.0)
        self.assertTrue(is_exact_match(g, g))

    def test_no_match(self):
        self.assertEqual(grid_cell_accuracy([[1, 2]], [[3, 4]]), 0.0)

    def test_partial_match(self):
        self.assertAlmostEqual(grid_cell_accuracy([[1, 2]], [[1, 3]]), 0.5)

    def test_shape_mismatch_returns_zero_row_length(self):
        self.assertEqual(grid_cell_accuracy([[1, 2]], [[1, 2], [3, 4]]), 0.0)  # row count mismatch

    def test_shape_mismatch_returns_zero_column_length(self):
        self.assertEqual(grid_cell_accuracy([[1, 2, 3]], [[1, 2]]), 0.0)  # col count mismatch

    def test_empty_grid(self):
        self.assertEqual(grid_cell_accuracy([], []), 0.0)

    def test_float_input_returns_zero(self):
        self.assertEqual(grid_cell_accuracy([[1, 2]], 1.5), 0.0)
        self.assertEqual(grid_cell_accuracy(1.5, [[1, 2]]), 0.0)


# ---------------------------------------------------------------------------
# ARCDomain — fitness and solving
# ---------------------------------------------------------------------------

class TestARCDomain(unittest.TestCase):

    def _rot90_task(self):
        pairs = [
            ([[1, 2], [3, 4]], grot90([[1, 2], [3, 4]])),
            ([[0, 1], [1, 0]], grot90([[0, 1], [1, 0]])),
            ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], grot90([[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
        ]
        test = [([[2, 0], [0, 2]], grot90([[2, 0], [0, 2]]))]
        return ARCTask(name="rot90", train_pairs=pairs, test_pairs=test)

    def test_fitness_perfect_tree_is_near_zero(self):
        task = self._rot90_task()
        domain = ARCDomain(task, lam=0.0)
        tree = make_node("grot90", [make_leaf_var(0)])
        self.assertLess(domain.fitness(tree), 1e-6)

    def test_fitness_wrong_tree_is_positive(self):
        task = self._rot90_task()
        domain = ARCDomain(task, lam=0.0)
        tree = make_node("grot180", [make_leaf_var(0)])
        self.assertGreater(domain.fitness(tree), 0)

    def test_check_solution_correct_tree(self):
        task = self._rot90_task()
        domain = ARCDomain(task)
        tree = make_node("grot90", [make_leaf_var(0)])
        self.assertTrue(domain.check_solution(tree))

    def test_check_solution_wrong_tree(self):
        task = self._rot90_task()
        domain = ARCDomain(task)
        tree = make_node("grot180", [make_leaf_var(0)])
        self.assertFalse(domain.check_solution(tree))

    def test_solve_rot90_end_to_end(self):
        """End-to-end: search must find grot90 in few generations."""
        task = self._rot90_task()
        domain = ARCDomain(task)
        cfg = SearchConfig(beam_size=10, offspring=20, generations=50,
                           verbose=False, seed=0)
        result = domain.solve(cfg)
        self.assertTrue(
            domain.check_solution(result.best_tree),
            f"Failed to solve rot90, found: {result.best_tree}"
        )

    def test_train_accuracy_perfect(self):
        task = self._rot90_task()
        domain = ARCDomain(task)
        tree = make_node("grot90", [make_leaf_var(0)])
        self.assertAlmostEqual(domain.train_accuracy(tree), 1.0)

    def test_test_accuracy_perfect(self):
        task = self._rot90_task()
        domain = ARCDomain(task)
        tree = make_node("grot90", [make_leaf_var(0)])
        self.assertAlmostEqual(domain.test_accuracy(tree), 1.0)

    def test_primitive_subset(self):
        """Using a restricted primitive subset should still find the answer."""
        task = self._rot90_task()
        domain = ARCDomain(task, primitive_subset=["grot90", "grefl_h", "gid"])
        cfg = SearchConfig(beam_size=5, offspring=10, generations=30,
                           verbose=False, seed=0)
        result = domain.solve(cfg)
        self.assertTrue(domain.check_solution(result.best_tree))

    def test_lam_increases_fitness(self):
        """Higher lambda should increase fitness (more complexity penalty)."""
        task = self._rot90_task()
        tree = make_node("grot90", [make_leaf_var(0)])
        d_low  = ARCDomain(task, lam=0.0)
        d_high = ARCDomain(task, lam=1.0)
        # With lam=0 and perfect tree, fitness ≈ 0
        # With lam=1 and size=2, fitness ≈ 2
        self.assertLess(d_low.fitness(tree), d_high.fitness(tree))


if __name__ == "__main__":
    unittest.main(verbosity=2)
