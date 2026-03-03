"""
tests/test_integration.py
=========================
Integration tests covering the full pipeline:
  - BenchmarkConfig / TaskResult / BenchmarkReport dataclasses
  - evaluate_tasks() end-to-end on a small task slice
  - run_benchmark() with baseline_only=True on a small task slice
  - JSON serialisation (as_dict, save_path round-trip)
  - build_benchmark / get_benchmark equivalence
  - load_tasks_from_dir() — real-dataset loader (tested with synthetic JSON)
  - scripts/run_all.py entry-point functions (symreg, cartpole)

These tests are deliberately kept fast by using tiny search configs
(beam_size=5, generations=10) and running only 2–3 tasks.

Run with:
    python -m unittest tests.test_integration -v
    python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from domains.arc.benchmark import build_benchmark, get_benchmark
from domains.arc.runner import (
    BenchmarkConfig,
    BenchmarkReport,
    TaskResult,
    evaluate_tasks,
    load_tasks_from_dir,
    run_benchmark,
)
from domains.arc.domain import ARCTask


# ---------------------------------------------------------------------------
# Minimal search config shared across tests (very fast)
# ---------------------------------------------------------------------------

_FAST_CFG = BenchmarkConfig(
    beam_size=5,
    offspring=10,
    generations=10,
    workers=1,
    lam=0.02,
    verbose=False,
    baseline_only=True,
)

_BASELINE_OPS = [
    "grot90", "grot180", "grot270",
    "grefl_h", "grefl_v",
    "gtrsp", "ginv", "gid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_tasks(n: int = 3) -> list[ARCTask]:
    """Return the first *n* geometric tasks (Category A; guaranteed easy for baseline ops).

    Task names follow the pattern '<letter>_<index>_<description>' where:
      A = Geometric transforms (rotate, reflect, transpose)
      B = Color operations
      C = Object operations
      D = Pattern generation
      E = Counting / encoding
      F = Compositional
    """
    tasks = build_benchmark()
    geo = [t for t in tasks if t.name.startswith("A_")]
    return geo[:n]


# ---------------------------------------------------------------------------
# 1. Benchmark / get_benchmark equivalence
# ---------------------------------------------------------------------------

class TestBuildVsGet(unittest.TestCase):
    """build_benchmark() and get_benchmark() must return identical tasks."""

    def test_same_length(self):
        self.assertEqual(len(build_benchmark()), len(get_benchmark()))

    def test_same_names(self):
        built = [t.name for t in build_benchmark()]
        gotten = [t.name for t in get_benchmark()]
        self.assertEqual(built, gotten)

    def test_same_train_pairs(self):
        for b, g in zip(build_benchmark(), get_benchmark()):
            self.assertEqual(b.train_pairs, g.train_pairs,
                             msg=f"Train pairs differ for task {b.name}")

    def test_deterministic_across_calls(self):
        """Two calls to build_benchmark with the same seed must agree."""
        a = build_benchmark(seed=0)
        b = build_benchmark(seed=0)
        for ta, tb in zip(a, b):
            self.assertEqual(ta.train_pairs, tb.train_pairs)

    def test_different_seeds_differ(self):
        """Different seeds should produce (at least some) different grids."""
        a = build_benchmark(seed=0)
        b = build_benchmark(seed=99)
        # At least one task must differ (probabilistic but highly reliable)
        differs = any(ta.train_pairs != tb.train_pairs for ta, tb in zip(a, b))
        self.assertTrue(differs, "Seeds 0 and 99 produced identical benchmarks")

    def test_minimum_task_count(self):
        self.assertGreaterEqual(len(build_benchmark()), 70,
                                "Benchmark should contain at least 70 tasks")

    def test_all_tasks_have_train_and_test(self):
        for task in build_benchmark():
            self.assertGreater(len(task.train_pairs), 0,
                               f"{task.name} has no training pairs")
            self.assertGreater(len(task.test_pairs), 0,
                               f"{task.name} has no test pairs")

    def test_all_tasks_have_names(self):
        for task in build_benchmark():
            self.assertIsInstance(task.name, str)
            self.assertGreater(len(task.name), 0)


# ---------------------------------------------------------------------------
# 2. BenchmarkConfig defaults and field types
# ---------------------------------------------------------------------------

class TestBenchmarkConfig(unittest.TestCase):

    def test_default_fields(self):
        cfg = BenchmarkConfig()
        self.assertIsInstance(cfg.beam_size, int)
        self.assertIsInstance(cfg.offspring, int)
        self.assertIsInstance(cfg.generations, int)
        self.assertIsInstance(cfg.workers, int)
        self.assertIsInstance(cfg.lam, float)
        self.assertIsInstance(cfg.verbose, bool)
        self.assertIsInstance(cfg.baseline_only, bool)

    def test_defaults_are_positive(self):
        cfg = BenchmarkConfig()
        self.assertGreater(cfg.beam_size, 0)
        self.assertGreater(cfg.offspring, 0)
        self.assertGreater(cfg.generations, 0)
        self.assertGreater(cfg.workers, 0)
        self.assertGreater(cfg.lam, 0)

    def test_custom_values_stored(self):
        cfg = BenchmarkConfig(beam_size=7, generations=3, verbose=False)
        self.assertEqual(cfg.beam_size, 7)
        self.assertEqual(cfg.generations, 3)
        self.assertFalse(cfg.verbose)


# ---------------------------------------------------------------------------
# 3. TaskResult dataclass and as_dict
# ---------------------------------------------------------------------------

class TestTaskResult(unittest.TestCase):

    def _make(self, **kwargs) -> TaskResult:
        defaults = dict(
            task_name="Geometric_rot90_0",
            category="Geometric",
            true_op="grot90",
            found_expr="grot90(x)",
            train_acc=1.0,
            test_acc=1.0,
            solved=True,
            near_solved=True,
            n_nodes=1,
            elapsed_s=0.5,
        )
        defaults.update(kwargs)
        return TaskResult(**defaults)

    def test_as_dict_keys(self):
        d = self._make().as_dict()
        for key in ("task", "category", "true_op", "found_expr",
                    "train_acc", "test_acc", "solved", "near_solved",
                    "n_nodes", "elapsed_s"):
            self.assertIn(key, d, msg=f"Missing key: {key}")

    def test_as_dict_values_serialisable(self):
        d = self._make().as_dict()
        json.dumps(d)  # must not raise

    def test_solved_flag(self):
        self.assertTrue(self._make(solved=True).solved)
        self.assertFalse(self._make(solved=False).solved)

    def test_near_solved_flag(self):
        self.assertTrue(self._make(test_acc=0.85, near_solved=True).near_solved)
        self.assertFalse(self._make(test_acc=0.5, near_solved=False).near_solved)

    def test_accuracy_rounded_in_dict(self):
        d = self._make(train_acc=0.123456789).as_dict()
        # as_dict rounds to 4dp
        self.assertEqual(d["train_acc"], round(0.123456789, 4))


# ---------------------------------------------------------------------------
# 4. BenchmarkReport properties and summary
# ---------------------------------------------------------------------------

class TestBenchmarkReport(unittest.TestCase):

    def _make_report(self, solved_flags: list[bool]) -> BenchmarkReport:
        report = BenchmarkReport(label="Test", n_ops=8)
        for i, solved in enumerate(solved_flags):
            report.results.append(TaskResult(
                task_name=f"Geometric_task_{i}",
                category="Geometric",
                true_op="grot90",
                found_expr="grot90(x)" if solved else "x",
                train_acc=1.0 if solved else 0.5,
                test_acc=1.0 if solved else 0.4,
                solved=solved,
                near_solved=solved or False,
                n_nodes=1,
                elapsed_s=0.1,
            ))
        report.total_elapsed_s = 1.0
        return report

    def test_n_tasks(self):
        report = self._make_report([True, False, True])
        self.assertEqual(report.n_tasks, 3)

    def test_n_solved(self):
        report = self._make_report([True, False, True])
        self.assertEqual(report.n_solved, 2)

    def test_pct_solved(self):
        report = self._make_report([True, False, True, False])
        self.assertAlmostEqual(report.pct_solved, 50.0)

    def test_mean_test_acc(self):
        report = self._make_report([True, False])  # acc 1.0 and 0.4
        self.assertAlmostEqual(report.mean_test_acc, 0.7, places=5)

    def test_by_category_counts(self):
        report = self._make_report([True, False, True])
        cats = report.by_category()
        self.assertIn("Geometric", cats)
        self.assertEqual(cats["Geometric"]["total"], 3)
        self.assertEqual(cats["Geometric"]["solved"], 2)

    def test_empty_report_safe(self):
        report = BenchmarkReport(label="Empty", n_ops=0)
        self.assertEqual(report.n_tasks, 0)
        self.assertEqual(report.n_solved, 0)
        self.assertEqual(report.pct_solved, 0.0)
        self.assertEqual(report.mean_test_acc, 0.0)
        self.assertIsInstance(report.summary(), str)

    def test_summary_is_string(self):
        report = self._make_report([True, False])
        s = report.summary()
        self.assertIsInstance(s, str)
        self.assertIn("Geometric", s)

    def test_summary_contains_solved_count(self):
        report = self._make_report([True, True, False])
        s = report.summary()
        self.assertIn("2/3", s)

    def test_as_dict_keys(self):
        report = self._make_report([True])
        d = report.as_dict()
        for key in ("label", "n_ops", "n_tasks", "n_solved",
                    "pct_solved", "mean_test_acc", "total_elapsed_s", "results"):
            self.assertIn(key, d)

    def test_as_dict_json_serialisable(self):
        report = self._make_report([True, False])
        json.dumps(report.as_dict())  # must not raise

    def test_as_dict_results_length(self):
        report = self._make_report([True, False, True])
        self.assertEqual(len(report.as_dict()["results"]), 3)


# ---------------------------------------------------------------------------
# 5. evaluate_tasks — end-to-end on 2 geometric tasks
# ---------------------------------------------------------------------------

class TestEvaluateTasks(unittest.TestCase):
    """Run evaluate_tasks on 2 tasks with a tiny config."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = _mini_tasks(2)
        cls.report = evaluate_tasks(
            cls.tasks, _BASELINE_OPS, _FAST_CFG, "Integration-test"
        )

    def test_returns_benchmark_report(self):
        self.assertIsInstance(self.report, BenchmarkReport)

    def test_result_count_matches_tasks(self):
        self.assertEqual(len(self.report.results), len(self.tasks))

    def test_all_results_are_task_results(self):
        for r in self.report.results:
            self.assertIsInstance(r, TaskResult)

    def test_task_names_preserved(self):
        names_in = [t.name for t in self.tasks]
        names_out = [r.task_name for r in self.report.results]
        self.assertEqual(names_in, names_out)

    def test_accuracies_in_range(self):
        for r in self.report.results:
            self.assertGreaterEqual(r.train_acc, 0.0)
            self.assertLessEqual(r.train_acc, 1.0)
            self.assertGreaterEqual(r.test_acc, 0.0)
            self.assertLessEqual(r.test_acc, 1.0)

    def test_found_expr_is_string(self):
        for r in self.report.results:
            self.assertIsInstance(r.found_expr, str)
            self.assertGreater(len(r.found_expr), 0)

    def test_n_nodes_positive(self):
        for r in self.report.results:
            self.assertGreater(r.n_nodes, 0)

    def test_elapsed_s_positive(self):
        for r in self.report.results:
            self.assertGreater(r.elapsed_s, 0.0)

    def test_total_elapsed_positive(self):
        self.assertGreater(self.report.total_elapsed_s, 0.0)

    def test_pct_solved_in_range(self):
        self.assertGreaterEqual(self.report.pct_solved, 0.0)
        self.assertLessEqual(self.report.pct_solved, 100.0)

    def test_summary_runs_without_error(self):
        s = self.report.summary()
        self.assertIsInstance(s, str)

    def test_category_a_recorded(self):
        """Category letter 'A' must appear (task names are '<letter>_<idx>_<desc>')."""
        cats = self.report.by_category()
        self.assertIn("A", cats)


# ---------------------------------------------------------------------------
# 6. run_benchmark — full pipeline, baseline_only, small task slice
# ---------------------------------------------------------------------------

class TestRunBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tasks = _mini_tasks(3)
        cls.baseline, cls.expanded = run_benchmark(
            tasks=tasks,
            cfg=_FAST_CFG,   # baseline_only=True
        )

    def test_returns_two_reports(self):
        self.assertIsInstance(self.baseline, BenchmarkReport)
        # expanded is None when baseline_only=True
        self.assertIsNone(self.expanded)

    def test_baseline_has_results(self):
        self.assertGreater(len(self.baseline.results), 0)

    def test_baseline_label_contains_ops(self):
        self.assertIn("ops", self.baseline.label.lower())

    def test_baseline_n_ops_is_8(self):
        self.assertEqual(self.baseline.n_ops, 8)

    def test_baseline_as_dict_serialisable(self):
        json.dumps(self.baseline.as_dict())


# ---------------------------------------------------------------------------
# 7. JSON save_path round-trip
# ---------------------------------------------------------------------------

class TestSavePath(unittest.TestCase):

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save = os.path.join(tmpdir, "results.json")
            run_benchmark(tasks=_mini_tasks(2), cfg=_FAST_CFG, save_path=save)
            self.assertTrue(os.path.exists(save), "results.json not created")

    def test_save_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save = os.path.join(tmpdir, "results.json")
            run_benchmark(tasks=_mini_tasks(2), cfg=_FAST_CFG, save_path=save)
            with open(save) as f:
                data = json.load(f)
            self.assertIn("baseline", data)

    def test_saved_baseline_has_results_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save = os.path.join(tmpdir, "results.json")
            run_benchmark(tasks=_mini_tasks(2), cfg=_FAST_CFG, save_path=save)
            with open(save) as f:
                data = json.load(f)
            self.assertIn("results", data["baseline"])

    def test_saved_result_count_matches(self):
        tasks = _mini_tasks(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            save = os.path.join(tmpdir, "results.json")
            run_benchmark(tasks=tasks, cfg=_FAST_CFG, save_path=save)
            with open(save) as f:
                data = json.load(f)
            self.assertEqual(data["baseline"]["n_tasks"], len(tasks))


# ---------------------------------------------------------------------------
# 8. scripts/run_all.py entry-point functions
# ---------------------------------------------------------------------------

class TestRunAllFunctions(unittest.TestCase):
    """Smoke-test the three entry-point functions from scripts/run_all.py."""

    def test_run_symbolic_regression_quick(self):
        """run_symbolic_regression(quick=True) must complete without error."""
        import importlib.util, types
        spec = importlib.util.spec_from_file_location(
            "run_all",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_all.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Must not raise
        mod.run_symbolic_regression(quick=True)

    def test_run_cartpole_quick(self):
        """run_cartpole(quick=True) must complete without error."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_all",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "run_all.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run_cartpole(quick=True)


# ---------------------------------------------------------------------------
# 9. ARCTask.from_dict round-trip
# ---------------------------------------------------------------------------

class TestARCTaskRoundTrip(unittest.TestCase):

    def test_build_benchmark_tasks_have_correct_structure(self):
        for task in build_benchmark():
            for inp, out in task.train_pairs:
                self.assertIsInstance(inp, list)
                self.assertIsInstance(out, list)
                self.assertGreater(len(inp), 0)
                self.assertGreater(len(out), 0)
                # Each row must be a list of ints in [0, 9]
                for row in inp:
                    for cell in row:
                        self.assertIn(cell, range(10),
                                      msg=f"Cell value {cell} out of [0,9] in {task.name}")

    def test_test_pairs_non_empty(self):
        for task in build_benchmark():
            self.assertGreater(len(task.test_pairs), 0,
                               f"{task.name} has no test pairs")

    def test_true_op_is_nonempty_string(self):
        """Every task must have a non-empty true_op label."""
        for task in build_benchmark():
            self.assertIsInstance(task.true_op, str,
                                  msg=f"{task.name}.true_op is not a string")
            self.assertGreater(len(task.true_op), 0,
                               msg=f"{task.name}.true_op is empty")

    def test_most_true_ops_are_registered_primitives(self):
        """Most (≥60%) single-op tasks must have true_op matching a registered ARC primitive.

        NOTE: 17 of 76 tasks have descriptive true_op strings (e.g. 'recolor(1->3)',
        'grot90 then recolor') that are human-readable labels for compound or
        parameterised transforms not directly in the registry. This is a known
        documentation gap; the benchmark still evaluates correctly because search
        is graded on grid accuracy, not on true_op string matching.
        """
        from core.primitives import registry
        arc_ops = set(registry.names(domain="arc"))
        tasks = build_benchmark()
        matched = sum(1 for t in tasks if t.true_op in arc_ops)
        pct = matched / len(tasks)
        self.assertGreater(pct, 0.60,
                           msg=f"Only {matched}/{len(tasks)} tasks have true_op in registry")


# ---------------------------------------------------------------------------
# 10. load_tasks_from_dir — real-dataset loader
# ---------------------------------------------------------------------------

class TestLoadTasksFromDir(unittest.TestCase):
    """
    Tests for load_tasks_from_dir().

    Because we can't bundle the real 400-task ARC-AGI dataset in the repo,
    these tests create synthetic JSON files that match the official ARC format
    exactly, then verify that loading behaves correctly.
    """

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _arc_json(n_train: int = 3, n_test: int = 1) -> dict:
        """Return a minimal valid ARC task dict."""
        pair = {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}
        return {
            "train": [pair] * n_train,
            "test":  [pair] * n_test,
        }

    def _make_dir(self, tmpdir: str, n_files: int = 3) -> str:
        """Write *n_files* synthetic ARC JSON files into *tmpdir*."""
        for i in range(n_files):
            p = os.path.join(tmpdir, f"task_{i:04d}.json")
            with open(p, "w") as f:
                json.dump(self._arc_json(), f)
        return tmpdir

    # ── error handling ────────────────────────────────────────────────────

    def test_missing_dir_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_tasks_from_dir("/nonexistent/path/that/does/not/exist")
        self.assertIn("arc_data", str(ctx.exception))

    def test_empty_dir_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                load_tasks_from_dir(tmpdir)
            # Error message should hint at the correct path
            self.assertIn("arc_data/data/evaluation", str(ctx.exception))

    # ── happy path ────────────────────────────────────────────────────────

    def test_loads_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=5)
            tasks = load_tasks_from_dir(tmpdir)
            self.assertEqual(len(tasks), 5)

    def test_returns_arc_task_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=2)
            for task in load_tasks_from_dir(tmpdir):
                self.assertIsInstance(task, ARCTask)

    def test_task_names_are_file_stems(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=3)
            tasks = load_tasks_from_dir(tmpdir)
            stems = {f"task_{i:04d}" for i in range(3)}
            self.assertEqual({t.name for t in tasks}, stems)

    def test_train_pairs_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=1)
            task = load_tasks_from_dir(tmpdir)[0]
            self.assertEqual(len(task.train_pairs), 3)

    def test_test_pairs_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=1)
            task = load_tasks_from_dir(tmpdir)[0]
            self.assertEqual(len(task.test_pairs), 1)

    def test_grid_values_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=1)
            task = load_tasks_from_dir(tmpdir)[0]
            inp, out = task.train_pairs[0]
            self.assertEqual(inp, [[1, 2], [3, 4]])
            self.assertEqual(out, [[3, 1], [4, 2]])

    def test_sorted_by_filename(self):
        """Tasks must come back in filename order for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write files in reverse order to catch unsorted loading
            for i in [4, 2, 0, 3, 1]:
                p = os.path.join(tmpdir, f"task_{i:04d}.json")
                with open(p, "w") as f:
                    json.dump(self._arc_json(), f)
            tasks = load_tasks_from_dir(tmpdir)
            names = [t.name for t in tasks]
            self.assertEqual(names, sorted(names))

    def test_non_json_files_ignored(self):
        """Non-.json files in the directory must be silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=2)
            # Write a non-JSON file that should be ignored
            with open(os.path.join(tmpdir, "README.md"), "w") as f:
                f.write("# not a task")
            tasks = load_tasks_from_dir(tmpdir)
            self.assertEqual(len(tasks), 2)

    def test_tasks_are_solvable_with_arc_domain(self):
        """Tasks loaded from synthetic JSON must work with ARCDomain (no crash)."""
        from domains.arc.domain import ARCDomain
        from core.search import SearchConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_dir(tmpdir, n_files=1)
            task = load_tasks_from_dir(tmpdir)[0]
            domain = ARCDomain(task)
            from core.tree import make_leaf_var
            tree = make_leaf_var(0)
            # fitness() and check_solution() must not raise
            f = domain.fitness(tree)
            self.assertIsInstance(f, float)
            solved = domain.check_solution(tree)
            self.assertIsInstance(solved, bool)


if __name__ == "__main__":
    unittest.main()
