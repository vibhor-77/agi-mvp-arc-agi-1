# agi-mvp-arc-agi-1

**MDL-guided symbolic search for ARC-AGI and beyond.**

A clean, extensible framework that discovers transformation rules from examples using
Minimum Description Length (MDL) beam search over expression trees. Originally built
for ARC-AGI-1, designed to generalise to ARC-AGI-2, ARC-AGI-3, Zork, NetHack, and any
problem where "understanding" means finding the shortest program that explains the data.

---

## Quick start

```bash
# Clone and enter
git clone https://github.com/YOUR_USERNAME/agi-mvp-arc-agi-1.git
cd agi-mvp-arc-agi-1

# No external dependencies — uses only Python stdlib + numpy (optional)
python -m domains.arc.runner --quick        # ~30 s demo
python -m domains.arc.runner               # full benchmark (~3 min)

# Run all tests
python -m unittest discover tests/ -v
```

---

## Results

| System | Ops | Solved | Score |
|--------|-----|--------|-------|
| Baseline (geometric only) | 8 | 20/76 | 26% |
| Expanded DSL | 89 | 58/76 | **76%** |

Per-category breakdown (expanded DSL):

| Category | Score |
|----------|-------|
| Geometric (rotate, reflect, transpose) | 12/12 (100%) |
| Color (swap, fill, gravity) | 9/12 (75%) |
| Object (frame, mirror, hollow, scale) | 17/20 (85%) |
| Pattern (checkerboard, stripe, tile) | 6/12 (50%) |
| Counting (bar chart, majority, filter) | 7/8 (88%) |
| Compositional (2+ ops) | 7/12 (58%) |

---

## Architecture

```
agi-mvp-arc-agi-1/
├── core/                        # Domain-agnostic engine
│   ├── primitives.py            # PrimitiveRegistry — central op store
│   ├── tree.py                  # Expression tree (Node, mutate, crossover)
│   ├── search.py                # Beam search engine (BeamSearch, SearchConfig)
│   └── domain.py                # Abstract Domain base class
│
├── domains/                     # One subpackage per problem domain
│   ├── arc/
│   │   ├── primitives.py        # 89 grid transformation ops (Grid → Grid)
│   │   ├── domain.py            # ARCDomain, ARCTask, fitness function
│   │   ├── benchmark.py         # 76-task representative benchmark
│   │   └── runner.py            # CLI benchmark runner
│   ├── symbolic_reg/
│   │   └── domain.py            # Symbolic regression (y = f(x))
│   └── cartpole/
│       └── domain.py            # Symbolic RL on CartPole
│
├── tests/
│   ├── test_core.py             # 40+ tests: registry, tree, search
│   ├── test_arc.py              # 40+ tests: all primitive categories
│   └── test_domains.py         # 30+ tests: symreg, CartPole
│
├── docs/
│   ├── adding_primitives.md     # How to add new ARC operations
│   ├── adding_domains.md        # How to add new problem domains
│   └── theory.md                # MDL, Kolmogorov complexity, DreamCoder
│
└── scripts/
    └── run_all.py               # Run all three domains, save results
```

### Design principles

1. **One interface to rule them all.** Every domain implements three methods:
   `primitive_names()`, `fitness(tree)`, `n_vars()`. The search engine is identical
   across ARC, symbolic regression, CartPole, and any future domain.

2. **MDL fitness.** `fitness = error + λ × tree_size`. Smaller = better. This is
   Kolmogorov complexity in practice: prefer the shortest program that explains the data.

3. **Zero-import primitive addition.** To add a new ARC op, write a function and call
   `registry.register(...)`. The search picks it up automatically. No core changes needed.

4. **Expression trees, not neural networks.** Every solution is interpretable. You can
   read exactly what rule the system discovered.

---

## Adding new primitives

See [`docs/adding_primitives.md`](docs/adding_primitives.md).

Short version — add to `domains/arc/primitives.py`:

```python
def my_new_op(g: Grid) -> Grid:
    """One-line description of what it does."""
    # ... transform g ...
    return result

registry.register("my_new_op", my_new_op, domain="arc",
                  description="One-line description of what it does.")
```

That's it. The beam search will use it automatically.

---

## Adding a new domain

See [`docs/adding_domains.md`](docs/adding_domains.md).

```python
# domains/my_domain/domain.py
from core.domain import Domain
from core.primitives import registry

class MyDomain(Domain):

    def primitive_names(self) -> list[str]:
        return registry.names(domain="my_domain")

    def fitness(self, tree) -> float:
        # Lower is better. This is the only domain-specific logic.
        ...

    def n_vars(self) -> int:
        return 1  # number of input variables

# Then just:
domain = MyDomain()
result = domain.solve()
print(result.best_tree)
```

Planned domains: `arc2` (ARC-AGI-2), `nethack`, `zork`, `minigrid`.

---

## How to load real ARC-AGI-1 data

```python
import json
import pathlib
from domains.arc.domain import ARCTask, ARCDomain
from domains.arc.runner import BenchmarkConfig, evaluate_tasks
from core.primitives import registry
import domains.arc.primitives  # register ops

# Download the dataset from github.com/fchollet/ARC-AGI
tasks = []
for p in pathlib.Path("arc_data/evaluation").glob("*.json"):
    d = json.loads(p.read_text())
    d["name"] = p.stem
    tasks.append(ARCTask.from_dict(d))

cfg = BenchmarkConfig(generations=100, beam_size=20)
expanded_ops = registry.names(domain="arc")
report = evaluate_tasks(tasks, expanded_ops, cfg, "ARC-AGI-1 Eval")
print(report.summary())
```

---

## Theory

The system is built on three ideas:

**MDL (Minimum Description Length):** The best model minimises
`description_length(model) + description_length(data | model)`.
In our case: `tree_size + MSE`. A smaller tree that fits the data is always preferred.

**Library learning (DreamCoder):** After solving a corpus, we extract frequent subtrees
and promote them as named primitives. The compression ratio (nodes/problem) decreases
over rounds — this is the empirical signature of intelligence growth.

**Beam search over expression trees:** Maintains K elite candidates across generations.
Each candidate is mutated/crossed over, scored by MDL, and the best K survive.

See [`docs/theory.md`](docs/theory.md) for the full derivation.

---

## Citation / related work

- Chollet, F. (2019). On the Measure of Intelligence. [ARC paper]
- Ellis et al. (2021). DreamCoder: Bootstrapping inductive program synthesis. PLDI.
- Schmidt & Lipson (2009). Distilling free-form natural laws from experimental data. Science.
- ARC Prize: arcprize.org

---

## License

MIT
