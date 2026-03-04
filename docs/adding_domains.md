# Adding New Problem Domains

The framework is designed so that the beam search engine, tree structure, and
primitive registry are **completely domain-agnostic**. Adding a new domain
(ARC-AGI-2, Zork, NetHack, MuJoCo, ...) requires implementing exactly one
abstract class: `Domain`.

---

## The Domain interface

```python
# core/domain.py
class Domain(ABC):

    @abstractmethod
    def primitive_names(self) -> list[str]:
        """Which primitives from the registry to use."""

    @abstractmethod
    def fitness(self, tree: Node) -> float:
        """Score a candidate tree. Lower is better. 0 = perfect."""

    @abstractmethod
    def n_vars(self) -> int:
        """Number of input variables a leaf can reference."""
```

That's the entire required interface. Everything else is optional.

---

## Worked example: ARC-AGI-2

ARC-AGI-2 uses the same grid format as ARC-AGI-1. Adapting is trivial:

```python
# domains/arc2/domain.py
from domains.arc.domain import ARCDomain, ARCTask

class ARC2Domain(ARCDomain):
    """
    ARC-AGI-2 domain.

    Uses the same primitives and fitness as ARC-AGI-1.
    Override fitness() if ARC-AGI-2 requires different scoring.
    """
    pass


# Load ARC-AGI-2 tasks (download from arcprize.org)
import json, pathlib
tasks = []
for p in pathlib.Path("arc2_data").glob("*.json"):
    d = json.loads(p.read_text())
    d["name"] = p.stem
    tasks.append(ARCTask.from_dict(d))

# Solve with the same expanded primitive set
domain = ARC2Domain(tasks[0])
result = domain.solve()
```

---

## Worked example: ARC-AGI-3 (interactive)

ARC-AGI-3 tasks are game-like: the system must explore a grid world.
The fitness function runs an episode instead of comparing static grids:

```python
# domains/arc3/domain.py
from core.domain import Domain
from core.tree import Node
from core.primitives import registry
import domains.arc.primitives  # reuse grid ops

class ARC3Domain(Domain):
    """
    ARC-AGI-3 interactive domain.

    The system controls an agent that takes actions (up/down/left/right/pick/drop).
    Fitness = -score after N steps in the task environment.
    """

    def __init__(self, task_env, n_steps: int = 50):
        self.env = task_env
        self.n_steps = n_steps
        import domains.arc3.primitives  # register action primitives
        self._ops = registry.names(domain="arc3")
        self._prims = {n: registry.get(n) for n in self._ops}

    def primitive_names(self) -> list[str]:
        # Mix of grid-reading ops and action ops
        return self._ops

    def fitness(self, tree: Node) -> float:
        """Run an episode, return -mean_score (lower = better)."""
        state = self.env.reset()
        total_reward = 0.0
        for _ in range(self.n_steps):
            action = tree.eval([state], self._prims)
            state, reward, done = self.env.step(action)
            total_reward += reward
            if done:
                break
        return -total_reward   # negate because search minimises

    def n_vars(self) -> int:
        return 1   # one input: the current state grid
```

---

## Worked example: Zork / text adventure

```python
# domains/zork/domain.py
from core.domain import Domain
from core.tree import Node
from core.primitives import registry
import domains.zork.primitives   # go_north, pick_up, examine, ...

class ZorkDomain(Domain):
    """
    Symbolic policy for a Zork-style text adventure.

    State is a feature vector extracted from the game description:
      [score, room_id, inventory_weight, turns_left, ...]

    A tree like go_north(is_dark(x)) reads: "go north if it's dark".
    """

    def __init__(self, n_episodes: int = 5, max_steps: int = 100):
        self.n_episodes = n_episodes
        self.max_steps  = max_steps
        self._ops = registry.names(domain="zork")
        self._prims = {n: registry.get(n) for n in self._ops}

    def primitive_names(self) -> list[str]:
        return self._ops

    def fitness(self, tree: Node) -> float:
        scores = []
        for seed in range(self.n_episodes):
            env = make_zork_env(seed=seed)
            state = env.reset()
            total = 0.0
            for _ in range(self.max_steps):
                action = tree.eval([state.as_list()], self._prims)
                state, reward, done = env.step(action)
                total += reward
                if done: break
            scores.append(total)
        return -sum(scores) / len(scores)   # minimise negative score

    def n_vars(self) -> int:
        return len(ZORK_STATE_FEATURES)
```

---

## Checklist for a new domain

1. **Create `domains/<name>/`** with `__init__.py` and `domain.py`
2. **Define primitives** in `domains/<name>/primitives.py`, register them
   with `registry.register(..., domain="<name>")`
3. **Subclass `Domain`**, implement the three abstract methods
4. **Write tests** in `tests/test_<name>.py`
5. **Add an evaluation script** akin to `evaluate_agi.py` to run the test set autonomously.

---

## Domain design tips

### Fitness function

The fitness function is everything. It should:

- **Be fast.** It's called `beam_size × offspring × generations` times per task.
  For slow environments (CartPole, Zork), cache states, use shorter episodes for
  the search and longer ones for final evaluation.

- **Be differentiating.** If 90% of random trees get the same fitness, the search
  can't make progress. A partial-credit fitness (cell accuracy, partial score) is
  usually better than binary solved/unsolved.

- **Include MDL.** Add `λ × tree.size()` to prefer compact solutions:
  ```python
  return main_error + 0.02 * tree.size()
  ```

### Primitives

- **Unary only.** Each primitive takes exactly one value and returns one value.
  Binary operations can be built at the tree level by having two child subtrees,
  but this isn't currently supported — keep ops unary for simplicity.

- **Domain-specific.** Don't add Zork actions to the ARC domain. Use
  `domain="zork"` in `registry.register(...)` to keep namespaces clean.

- **Start small.** Add 10–20 obviously useful primitives and let the search
  compose them. It's better to have 20 correct ops than 200 redundant ones.

### Number of variables

`n_vars()` controls how many leaf variable indices are valid. For a domain
with state `[x, y, health, ammo]`, return `4` and trees can reference
`x` (var 0), `y` (var 1), etc.
