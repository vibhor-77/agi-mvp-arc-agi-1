#!/usr/bin/env python3
"""
tests/test_new_primitives.py
============================
Unit tests for the new ARC primitives added in the expressivity expansion.
Tests: flood fill, any-color extraction, parametric recoloring, object count predicates,
       grid XOR/diff, downscaling, and extended color replacements.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from domains.arc.primitives import (
    g_flood_fill,
    g_extract_objects_any,
    g_fg_to_most_common,
    g_fg_to_least_common,
    g_unique_color_per_obj,
    g_has_1_object,
    g_has_2_objects,
    g_has_gt2_objects,
    g_xor,
    g_diff,
    g_downscale_2x,
    g_downscale_3x,
    g_replace_2_with_3,
    g_replace_3_with_1,
    g_replace_3_with_2,
    g_replace_0_with_1,
    g_replace_0_with_2,
    g_replace_nonzero_with_1,
)


class TestFloodFill(unittest.TestCase):
    def test_basic_hole(self):
        """A 3x3 grid with a hollow shape should have its interior filled."""
        g = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = g_flood_fill(g)
        self.assertEqual(result, [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])

    def test_no_holes(self):
        """An open shape touching the border shouldn't fill anything."""
        g = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        # The center 0 is connected to the border via corners? No, only 4-connected.
        # The center 0 at (1,1) is NOT connected to border zeros via 4-connectivity.
        # Border zeros at (0,0), (0,2), (2,0), (2,2) are border connected.
        # (1,1) is surrounded by 1s → it IS enclosed!
        result = g_flood_fill(g)
        self.assertEqual(result[1][1], 1)

    def test_fully_empty(self):
        """All zeros, fully border-connected → no fill."""
        g = [[0, 0], [0, 0]]
        result = g_flood_fill(g)
        self.assertEqual(result, [[0, 0], [0, 0]])


class TestExtractObjectsAny(unittest.TestCase):
    def test_multicolor_blob(self):
        """Two adjacent cells of different colors should be one object."""
        g = [
            [0, 0, 0],
            [0, 1, 2],
            [0, 0, 0],
        ]
        result = g_extract_objects_any(g)
        self.assertEqual(result, [[1, 2]])

    def test_two_separate_blobs(self):
        """Should return the larger blob."""
        g = [
            [1, 0, 2],
            [1, 0, 0],
            [1, 0, 0],
        ]
        result = g_extract_objects_any(g)
        # Largest blob is the 3-pixel vertical bar of 1s
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 1)


class TestParametricRecoloring(unittest.TestCase):
    def test_most_common(self):
        g = [
            [1, 2, 1],
            [0, 1, 0],
        ]
        result = g_fg_to_most_common(g)
        # Most common = 1 (appears 3 times)
        self.assertEqual(result, [
            [1, 1, 1],
            [0, 1, 0],
        ])

    def test_least_common(self):
        g = [
            [1, 2, 1],
            [0, 1, 0],
        ]
        result = g_fg_to_least_common(g)
        # Least common = 2 (appears 1 time)
        self.assertEqual(result, [
            [2, 2, 2],
            [0, 2, 0],
        ])

    def test_unique_color_per_obj(self):
        g = [
            [1, 0, 3],
            [1, 0, 3],
        ]
        result = g_unique_color_per_obj(g)
        # First object gets color 1, second gets color 2
        self.assertEqual(result[0][0], 1)  # first object
        self.assertEqual(result[0][2], 2)  # second object
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][2], 2)


class TestObjectCountPredicates(unittest.TestCase):
    def test_single_object(self):
        g = [
            [0, 1, 0],
            [0, 1, 0],
        ]
        result = g_has_1_object(g)
        # Should return the grid (truthy)
        self.assertTrue(any(c != 0 for row in result for c in row))

    def test_two_objects(self):
        g = [
            [1, 0, 2],
            [0, 0, 0],
        ]
        result = g_has_2_objects(g)
        self.assertTrue(any(c != 0 for row in result for c in row))

    def test_single_fails_two(self):
        g = [
            [0, 1, 0],
            [0, 1, 0],
        ]
        result = g_has_2_objects(g)
        # Should be all zeros
        self.assertFalse(any(c != 0 for row in result for c in row))

    def test_gt2_objects(self):
        g = [
            [1, 0, 2, 0, 3],
        ]
        result = g_has_gt2_objects(g)
        self.assertTrue(any(c != 0 for row in result for c in row))


class TestXorDiff(unittest.TestCase):
    def test_xor_basic(self):
        g1 = [[1, 0], [0, 2]]
        g2 = [[0, 3], [0, 2]]
        result = g_xor(g1, g2)
        # (0,0): v1=1, v2=0 → 1
        # (0,1): v1=0, v2=3 → 3
        # (1,0): both 0 → 0
        # (1,1): v1=2, v2=2 → both non-zero → 0
        self.assertEqual(result, [[1, 3], [0, 0]])

    def test_diff_basic(self):
        g1 = [[1, 2], [3, 4]]
        g2 = [[0, 1], [1, 0]]
        result = g_diff(g1, g2)
        # Keep g1 where g2 is 0
        self.assertEqual(result, [[1, 0], [0, 4]])


class TestDownscaling(unittest.TestCase):
    def test_2x_downscale(self):
        g = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 0, 0],
            [3, 3, 0, 0],
        ]
        result = g_downscale_2x(g)
        self.assertEqual(result, [[1, 2], [3, 0]])

    def test_3x_downscale(self):
        g = [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 0, 0, 0],
            [3, 3, 3, 0, 0, 0],
            [3, 3, 3, 0, 0, 0],
        ]
        result = g_downscale_3x(g)
        self.assertEqual(result, [[1, 2], [3, 0]])


class TestColorReplacements(unittest.TestCase):
    def test_replace_2_with_3(self):
        g = [[1, 2, 3]]
        result = g_replace_2_with_3(g)
        self.assertEqual(result, [[1, 3, 3]])

    def test_replace_3_with_1(self):
        g = [[1, 2, 3]]
        result = g_replace_3_with_1(g)
        self.assertEqual(result, [[1, 2, 1]])

    def test_replace_0_with_1(self):
        g = [[0, 1, 0]]
        result = g_replace_0_with_1(g)
        self.assertEqual(result, [[1, 1, 1]])

    def test_replace_nonzero_with_1(self):
        g = [[0, 3, 5, 0]]
        result = g_replace_nonzero_with_1(g)
        self.assertEqual(result, [[0, 1, 1, 0]])


class TestRegistryIntegration(unittest.TestCase):
    def test_all_new_primitives_registered(self):
        """Verify all new primitives are accessible in the global registry."""
        from core.primitives import registry
        
        new_names = [
            "g_flood_fill", "g_extract_objects_any",
            "g_fg_to_most_common", "g_fg_to_least_common", "g_unique_color_per_obj",
            "g_has_1_object", "g_has_2_objects", "g_has_gt2_objects",
            "g_xor", "g_diff",
            "g_downscale_2x", "g_downscale_3x",
            "g_replace_2_with_3", "g_replace_3_with_1", "g_replace_3_with_2",
            "g_replace_0_with_1", "g_replace_0_with_2", "g_replace_nonzero_with_1",
        ]
        arc_names = registry.names(domain="arc")
        for name in new_names:
            self.assertIn(name, arc_names, f"'{name}' not found in arc registry")

    def test_arities_correct(self):
        """Binary ops should have arity 2, unary ops arity 1."""
        from core.primitives import registry
        
        self.assertEqual(registry.arity("g_xor"), 2)
        self.assertEqual(registry.arity("g_diff"), 2)
        self.assertEqual(registry.arity("g_flood_fill"), 1)
        self.assertEqual(registry.arity("g_downscale_2x"), 1)
        self.assertEqual(registry.arity("g_has_1_object"), 1)

    def test_total_ops_count(self):
        """Verify we now have significantly more ops than before."""
        from core.primitives import registry
        total = len(registry.names(domain="arc"))
        # We had ~150 before, should now have ~170+
        self.assertGreaterEqual(total, 165, f"Expected ≥165 arc ops, got {total}")


if __name__ == "__main__":
    unittest.main()
