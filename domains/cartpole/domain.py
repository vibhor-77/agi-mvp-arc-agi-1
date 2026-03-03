"""
domains/cartpole/domain.py
==========================
Symbolic Reinforcement Learning on CartPole.

The goal is to discover a *symbolic* control policy:
    force = f(x, ẋ, θ, θ̇)

where f is an expression tree.  The same beam search that finds
sin(x²)+2x for regression here finds a policy like "θ_dot" or
"aθ + bθ_dot" that keeps the pole balanced.

CartPole physics are implemented from scratch — no gym dependency.
This makes the code self-contained and the physics transparent.

Physical model (standard inverted pendulum)
-------------------------------------------
    State : [x, x_dot, theta, theta_dot]
    Action: force ∈ {-FORCE_MAG, +FORCE_MAG}  (sign comes from tree output)
    Episode terminates when |x| > 2.4 m or |θ| > 12°
    Success: survive 200 steps

Variable indices for expression trees
--------------------------------------
    0 → x         (cart position)
    1 → x_dot     (cart velocity)
    2 → theta     (pole angle, radians)
    3 → theta_dot (pole angular velocity)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.domain import Domain
from core.tree import Node
from core.search import SearchConfig, SearchResult
from core.primitives import registry


# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------

GRAVITY    = 9.8
MASS_CART  = 1.0
MASS_POLE  = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
HALF_POLE  = 0.5           # half the pole length (metres)
POLEMASS_LENGTH = MASS_POLE * HALF_POLE
FORCE_MAG  = 10.0
TAU        = 0.02          # seconds per step
THETA_LIMIT= 12 * math.pi / 180   # ≈ 0.2094 rad
X_LIMIT    = 2.4           # metres
MAX_STEPS  = 200


# ---------------------------------------------------------------------------
# Physics simulation
# ---------------------------------------------------------------------------

@dataclass
class CartPoleState:
    x:         float = 0.0
    x_dot:     float = 0.0
    theta:     float = 0.0
    theta_dot: float = 0.0

    def as_list(self) -> list[float]:
        return [self.x, self.x_dot, self.theta, self.theta_dot]

    def is_terminal(self) -> bool:
        return abs(self.x) > X_LIMIT or abs(self.theta) > THETA_LIMIT


def step_physics(state: CartPoleState, force: float) -> CartPoleState:
    """
    Advance the simulation one step using Euler integration of the
    standard inverted-pendulum equations of motion.

    Parameters
    ----------
    state : CartPoleState
    force : float
        Applied force (positive = right, negative = left).

    Returns
    -------
    CartPoleState
        New state after one timestep.
    """
    x, x_dot, theta, theta_dot = state.x, state.x_dot, state.theta, state.theta_dot
    cos_th = math.cos(theta)
    sin_th = math.sin(theta)

    # Equations of motion
    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sin_th) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_th - cos_th * temp) / (
        HALF_POLE * (4.0 / 3.0 - MASS_POLE * cos_th ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_th / TOTAL_MASS

    return CartPoleState(
        x         = x         + TAU * x_dot,
        x_dot     = x_dot     + TAU * x_acc,
        theta     = theta     + TAU * theta_dot,
        theta_dot = theta_dot + TAU * theta_acc,
    )


def run_episode(
    policy_fn,
    seed: int = 0,
    max_steps: int = MAX_STEPS,
) -> list[CartPoleState]:
    """
    Simulate one CartPole episode using *policy_fn* as the controller.

    Parameters
    ----------
    policy_fn : Callable[[list[float]], float]
        Maps state [x, ẋ, θ, θ̇] to a force value.
        The sign determines direction; magnitude is clamped to FORCE_MAG.
    seed : int
        RNG seed for initial state perturbation.
    max_steps : int
        Episode length cap.

    Returns
    -------
    list[CartPoleState]
        All states visited, including the terminal state.
    """
    import random
    rng = random.Random(seed)

    # Small random initial perturbation (standard gym initialisation)
    state = CartPoleState(
        x         = rng.uniform(-0.05, 0.05),
        x_dot     = rng.uniform(-0.05, 0.05),
        theta     = rng.uniform(-0.05, 0.05),
        theta_dot = rng.uniform(-0.05, 0.05),
    )
    trajectory = [state]

    for _ in range(max_steps - 1):
        if state.is_terminal():
            break
        raw_force = policy_fn(state.as_list())
        force = math.copysign(FORCE_MAG, raw_force)  # clamp to ±FORCE_MAG
        state = step_physics(state, force)
        trajectory.append(state)

    return trajectory


# ---------------------------------------------------------------------------
# CartPole domain
# ---------------------------------------------------------------------------

class CartPoleDomain(Domain):
    """
    Discover a symbolic control policy for CartPole.

    Fitness = − mean(steps survived) across *n_episodes* episodes.
    (Negative because beam search minimises fitness.)

    Parameters
    ----------
    n_episodes : int
        Number of episodes used to evaluate each policy.
        More episodes → more stable evaluation, but slower.
    seeds : list[int] | None
        Episode seeds.  Auto-generated from 0..n_episodes if None.
    extra_ops : list[str] | None
        Primitive names.  Defaults to math primitives.
    """

    def __init__(
        self,
        n_episodes: int = 20,
        seeds: list[int] | None = None,
        extra_ops: list[str] | None = None,
    ) -> None:
        self.n_episodes = n_episodes
        self._seeds = seeds or list(range(n_episodes))
        ops = extra_ops if extra_ops is not None else registry.names(domain="math")
        self._op_list = ops
        self._primitives = {n: registry.get(n) for n in ops}

    def primitive_names(self) -> list[str]:
        return self._op_list

    def n_vars(self) -> int:
        # [x, x_dot, theta, theta_dot]
        return 4

    def fitness(self, tree: Node) -> float:
        """
        Evaluate *tree* as a control policy.

        Returns − mean_steps (so lower fitness = longer survival).
        """
        def policy(state: list[float]) -> float:
            return float(tree.eval(state, self._primitives))

        total_steps = 0
        for seed in self._seeds:
            traj = run_episode(policy, seed=seed)
            total_steps += len(traj)

        mean_steps = total_steps / max(len(self._seeds), 1)
        return -mean_steps   # negate: minimise → maximise steps

    def description(self) -> str:
        return f"CartPole (symbolic RL, {self.n_episodes} episodes)"

    def on_result(self, result: SearchResult) -> None:
        mean_steps = -result.best_fitness
        print(
            f"  Best policy     : {result.best_tree}\n"
            f"  Mean survival   : {mean_steps:.1f} / {MAX_STEPS} steps\n"
            f"  Tree size       : {result.best_tree.size()} nodes"
        )

    # ------------------------------------------------------------------ #
    # Demo helpers                                                         #
    # ------------------------------------------------------------------ #

    def demonstrate(self, tree: Node, seed: int = 0) -> list[CartPoleState]:
        """Run one full episode with *tree* and return the trajectory."""
        def policy(state: list[float]) -> float:
            return float(tree.eval(state, self._primitives))
        return run_episode(policy, seed=seed)
