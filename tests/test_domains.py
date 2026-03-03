"""
tests/test_domains.py
=====================
Unit tests for symbolic regression and CartPole domains.

Run with:
    python -m pytest tests/test_domains.py -v
    python -m unittest tests.test_domains -v
"""
import math
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from domains.symbolic_reg.domain import SymbolicRegressionDomain
from domains.cartpole.domain import (
    CartPoleDomain, CartPoleState, step_physics, run_episode,
    THETA_LIMIT, X_LIMIT, MAX_STEPS,
)
from core.search import SearchConfig
from core.tree import make_leaf_var, make_node, make_leaf_const


# ---------------------------------------------------------------------------
# SymbolicRegressionDomain
# ---------------------------------------------------------------------------

class TestSymbolicRegressionDomain(unittest.TestCase):

    def _linear_domain(self):
        """y = x — trivially solvable."""
        xs = [float(i) for i in range(-5, 6)]
        return SymbolicRegressionDomain(xs, xs, lam=0.0)

    def test_fitness_identity_is_zero(self):
        domain = self._linear_domain()
        tree = make_leaf_var(0)
        self.assertAlmostEqual(domain.fitness(tree), 0.0, places=12)

    def test_fitness_wrong_is_positive(self):
        xs = [float(i) for i in range(-5, 6)]
        ys = [2.0 * x for x in xs]
        domain = SymbolicRegressionDomain(xs, ys, lam=0.0)
        tree = make_leaf_var(0)   # predicts y=x, truth is y=2x
        self.assertGreater(domain.fitness(tree), 0)

    def test_fitness_constant_offset_mse(self):
        xs = [1.0, 2.0, 3.0]
        ys = [2.0, 3.0, 4.0]   # y = x + 1
        domain = SymbolicRegressionDomain(xs, ys, lam=0.0)
        tree = make_leaf_var(0)  # predicts y = x → MSE = 1.0
        self.assertAlmostEqual(domain.fitness(tree), 1.0, places=9)

    def test_predict(self):
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 1.0, 4.0]
        domain = SymbolicRegressionDomain(xs, ys, lam=0.0)
        tree = make_node("sq", make_leaf_var(0))
        preds = domain.predict(tree)
        for p, y in zip(preds, ys):
            self.assertAlmostEqual(p, y, places=9)

    def test_from_function_factory(self):
        domain = SymbolicRegressionDomain.from_function(
            lambda x: x ** 2,
            x_range=(-2.0, 2.0),
            n_points=20,
        )
        self.assertEqual(len(domain._ys), 20)
        self.assertAlmostEqual(domain._ys[0], (-2.0) ** 2, places=9)

    def test_solve_identity_fast(self):
        xs = [float(i) for i in range(-5, 6)]
        domain = SymbolicRegressionDomain(xs, xs, lam=0.0)
        cfg = SearchConfig(beam_size=5, offspring=10, generations=30,
                           verbose=False, seed=0)
        result = domain.solve(cfg)
        self.assertLess(result.best_fitness, 0.1)

    def test_n_vars_univariate(self):
        domain = SymbolicRegressionDomain([1.0, 2.0], [1.0, 4.0])
        self.assertEqual(domain.n_vars(), 1)

    def test_n_vars_multivariate(self):
        domain = SymbolicRegressionDomain([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
        self.assertEqual(domain.n_vars(), 2)

    def test_multivariate_fitness_no_crash(self):
        xs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ys = [3.0, 7.0, 11.0]
        domain = SymbolicRegressionDomain(xs, ys, lam=0.0)
        tree = make_leaf_var(0)
        f = domain.fitness(tree)
        self.assertIsInstance(f, float)


# ---------------------------------------------------------------------------
# CartPole physics
# ---------------------------------------------------------------------------

class TestCartPolePhysics(unittest.TestCase):

    def test_step_returns_new_state(self):
        s = CartPoleState(0.0, 0.0, 0.05, 0.0)
        s2 = step_physics(s, 10.0)
        self.assertIsInstance(s2, CartPoleState)
        self.assertIsNot(s2, s)

    def test_positive_force_moves_cart_right(self):
        s = CartPoleState(0.0, 0.0, 0.0, 0.0)
        s2 = step_physics(s, 10.0)
        self.assertGreater(s2.x_dot, s.x_dot)

    def test_upright_zero_force_stays_upright(self):
        s = CartPoleState(0.0, 0.0, 0.0, 0.0)
        s2 = step_physics(s, 0.0)
        self.assertLess(abs(s2.theta), 0.01)

    def test_is_terminal_x_limit(self):
        s = CartPoleState(x=X_LIMIT + 0.1)
        self.assertTrue(s.is_terminal())

    def test_is_terminal_theta_limit(self):
        s = CartPoleState(theta=THETA_LIMIT + 0.01)
        self.assertTrue(s.is_terminal())

    def test_is_not_terminal_at_origin(self):
        s = CartPoleState()
        self.assertFalse(s.is_terminal())

    def test_as_list(self):
        s = CartPoleState(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(s.as_list(), [1.0, 2.0, 3.0, 4.0])

    def test_negative_x_limit_is_also_terminal(self):
        s = CartPoleState(x=-(X_LIMIT + 0.1))
        self.assertTrue(s.is_terminal())


# ---------------------------------------------------------------------------
# CartPole episodes
# ---------------------------------------------------------------------------

class TestCartPoleEpisodes(unittest.TestCase):

    def test_run_episode_returns_list(self):
        traj = run_episode(lambda s: 0.0, seed=0)
        self.assertIsInstance(traj, list)
        self.assertGreaterEqual(len(traj), 1)

    def test_run_episode_states_are_lists(self):
        traj = run_episode(lambda s: 0.0, seed=0)
        self.assertEqual(len(traj[0].as_list()), 4)  # [x, x_dot, theta, theta_dot]

    def test_theta_dot_policy_survives_long(self):
        """
        Policy: push in direction of angular velocity.
        Should survive substantially more steps than random.
        """
        def policy(state):
            return state[3]  # theta_dot

        steps = [len(run_episode(policy, seed=s)) for s in range(5)]
        mean_steps = sum(steps) / len(steps)
        self.assertGreater(mean_steps, 100,
                           f"theta_dot policy too weak: mean={mean_steps:.1f} steps")


# ---------------------------------------------------------------------------
# CartPoleDomain
# ---------------------------------------------------------------------------

class TestCartPoleDomain(unittest.TestCase):

    def test_fitness_is_negative(self):
        """More steps survived → more negative fitness."""
        domain = CartPoleDomain(n_episodes=3)
        tree = make_leaf_var(3)   # theta_dot policy
        f = domain.fitness(tree)
        self.assertLess(f, 0)

    def test_better_policy_lower_fitness(self):
        """theta_dot policy must beat constant-zero policy."""
        seeds = list(range(5))
        domain = CartPoleDomain(n_episodes=5, seeds=seeds)
        f_zero = domain.fitness(make_leaf_const(0.0))
        f_good = domain.fitness(make_leaf_var(3))
        self.assertLess(f_good, f_zero,
                        "theta_dot policy should score lower (better) than zero policy")

    def test_n_vars_is_4(self):
        self.assertEqual(CartPoleDomain().n_vars(), 4)

    def test_demonstrate_returns_states(self):
        domain = CartPoleDomain()
        traj = domain.demonstrate(make_leaf_var(3), seed=0)
        self.assertIsInstance(traj, list)
        self.assertTrue(all(isinstance(s, CartPoleState) for s in traj))

    def test_solve_smoke(self):
        """Smoke test: search completes without crashing."""
        domain = CartPoleDomain(n_episodes=3, seeds=[0, 1, 2])
        cfg = SearchConfig(beam_size=5, offspring=8, generations=5,
                           verbose=False, seed=42)
        result = domain.solve(cfg)
        self.assertLess(result.best_fitness, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
