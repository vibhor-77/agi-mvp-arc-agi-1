# Next Steps

*Prioritised roadmap for this project — from immediate fixes to longer-term research.*

---

## Immediate (1–2 days)

### 1. Download and run on real ARC-AGI-1 data

The benchmark in `domains/arc/benchmark.py` is programmatically generated
(76 tasks). To test against the real 400-task eval set:

```bash
# Download the dataset (real tasks live under data/evaluation/ and data/training/)
git clone https://github.com/fchollet/ARC-AGI arc_data

# Smoke test: 10 real tasks, fast settings
python -m domains.arc.runner --data arc_data/data/evaluation --quick --tasks 10

# Full 400-task eval run (expect several hours at default settings)
python -m domains.arc.runner --data arc_data/data/evaluation --workers 4 --save results_real.json
```

Or use the convenience script:

```bash
python run_real_arc.py --quick --tasks 10    # smoke test
python run_real_arc.py                       # full run
```

The `--data` flag tells the runner to load from real JSON files instead of the
programmatic benchmark. Without it, the runner always uses the built-in 76 tasks.

Expected result: ~3% on real tasks with the current 89-op DSL (only tasks expressible
as a single unary primitive are solvable). Most real ARC tasks require object
segmentation, conditional logic, or multi-op composition not yet in the framework.

### 2. Tune beam search hyperparameters

The current defaults (beam=20, offspring=50, generations=100) were chosen for
the programmatic benchmark. Real ARC tasks are harder — try:

```python
cfg = BenchmarkConfig(
    beam_size   = 30,     # wider beam
    offspring   = 80,
    generations = 200,    # more search budget per task
    workers     = 4,      # parallel if you have cores
)
```

### 3. Add SymPy simplification to tree scoring

Currently `grot90(grot90(grot90(grot90(x))))` and `gid(x)` have different sizes
but identical semantics. Adding SymPy canonicalisation would fix double-counting:

```python
# In domains/arc/domain.py, fitness()
import sympy
# ... after finding a solution, simplify before reporting tree.size()
```

---

## Short term (1–2 weeks)

### 4. Object segmentation primitive

This is the single biggest gap. ~25% of ARC-AGI-1 tasks require reasoning about
*individual objects* (connected components), not whole-grid transforms.

```python
def gsegment(g: Grid) -> Grid:
    """
    Label connected components (4-connected) with unique colors 1..N.
    Background (0) stays 0.
    Returns the labeled grid.
    """
    # BFS/DFS connected component labeling
    ...

registry.register("gsegment", gsegment, domain="arc",
                  description="Label connected components")
```

### 5. Parametric recoloring

Many ARC tasks require "map color X to color Y based on context" — not a fixed
swap. A parametric version would take the most-frequent color in the input and
remap it to the least-frequent, etc.

```python
def grecolor_max_to_min(g: Grid) -> Grid:
    """Remap the most-frequent foreground color to the least-frequent."""
    ...

def grecolor_by_count(g: Grid) -> Grid:
    """Each color's new value = its rank by frequency (1=rarest, N=most common)."""
    ...
```

### 6. Conditional primitives

A meaningful fraction of ARC tasks apply a rule conditionally (e.g., "if cell is
on the border, apply A; otherwise apply B"). This needs a new node type:

```python
class ConditionalNode:
    """if predicate(x) then branch_a(x) else branch_b(x)"""
```

This would require extending `core/tree.py` to support ternary nodes.

### 7. Library learning pass on solved tasks

After a benchmark run, extract frequently-occurring subtrees from solved tasks
and promote them to new primitives (DreamCoder-style):

```python
# scripts/library_learning.py
from collections import Counter

def extract_subtrees(results: list[TaskResult]) -> Counter:
    """Count all subtrees across all solved tasks."""
    counts = Counter()
    for r in results:
        if r.solved:
            for subtree in r.best_tree.all_subtrees():
                if subtree.size() > 1:
                    counts[str(subtree)] += 1
    return counts
```

If subtree T appears in K solved tasks and size(T) > 1, defining T as a
named primitive saves K × (size(T) - 1) nodes total. This is the compression
ratio metric.

---

## Medium term (1–3 months)

### 8. ARC-AGI-2 domain

ARC-AGI-2 uses the same JSON format but is significantly harder (contextual
reasoning, multi-rule interaction). Adapting is trivial:

```python
# domains/arc2/domain.py — 5 lines
from domains.arc.domain import ARCDomain
class ARC2Domain(ARCDomain):
    pass   # same primitives, same fitness, just harder tasks
```

The interesting work is understanding *why* the same primitives fail on ARC-AGI-2
tasks and adding the missing ones.

### 9. Refinement loops

The top ARC Prize 2025 systems all use refinement loops. The basic idea:

1. Run beam search → find candidate program P
2. Execute P on the task → check output
3. Feed the error back as a new fitness signal
4. Repeat

This maps exactly onto your four-pillar framework. The search is the
exploration; the feedback loop is the refinement; MDL is the approximability
criterion.

```python
class RefinementLoop:
    def __init__(self, domain, max_rounds=5):
        self.domain = domain
        self.max_rounds = max_rounds

    def solve(self):
        program = None
        for round in range(self.max_rounds):
            result = self.domain.solve(seed=round)
            if self.domain.check_solution(result.best_tree):
                return result
            # Update fitness based on partial solution quality
            self.domain.update_fitness_from_partial(result.best_tree)
        return result
```

### 10. LLM-guided primitive generation

The state of the art (SOAR, ARChitects) uses LLMs to generate candidate
programs. A hybrid approach:

1. LLM generates Python snippets for new grid primitives based on task examples
2. Snippets are parsed, safety-checked, and registered into the registry
3. Beam search uses the expanded vocabulary

This would let the system "invent" new primitives on the fly for tasks that
can't be solved with existing vocabulary.

---

## Research directions (3+ months)

### 11. Measure compression ratio as an intelligence metric

After each library learning round, compute:

```
compression_ratio = total_nodes_in_solutions / n_solved_tasks
```

If this ratio decreases monotonically over rounds, that's empirical evidence of
the system "getting smarter" — it encodes the same solutions in fewer symbols.
This could be a publishable result.

### 12. Zork / text adventure domain

```python
# domains/zork/domain.py
class ZorkDomain(Domain):
    """
    State: feature vector extracted from game text
    Action: one of go_north/south/east/west, pick_up, drop, examine, ...
    Fitness: -mean_score after N steps
    """
```

The interesting question: can beam search discover a compact symbolic policy
that generalises across Zork rooms? This tests whether the MDL framework scales
to sequential decision problems with natural language state.

### 13. Multimodal primitives

For video/robotics, define primitives over continuous state:

- `detect_edge(frame)` → binary mask
- `track_object(frame, mask)` → bounding box
- `estimate_velocity(frame_t, frame_t1)` → flow field

The same MDL beam search then discovers compact causal models of physical scenes.

### 14. Publish results

A paper comparing pure symbolic (this system) vs. LLM-guided symbolic vs.
test-time-training approaches on ARC-AGI-1 would be timely. The key contribution:
showing that a clean MDL-driven primitive vocabulary, with library learning,
can close a significant fraction of the gap to LLM-based systems while being
fully interpretable.

Target venues: NeurIPS 2026 workshop, ICLR 2027.

---

## Known limitations to address

| Limitation | Impact | Fix |
|------------|--------|-----|
| No object segmentation | ~25% of ARC tasks unsolvable | Add `gsegment` |
| No parametric recoloring | ~15% of ARC tasks | Add `grecolor_*` variants |
| No conditional logic | ~10% of ARC tasks | Extend tree to ternary nodes |
| Unary-only primitives | Can't express binary relations | Add BinaryNode type |
| Fixed vocabulary | Can't adapt to new task types | Library learning + LLM |
| No test-time adaptation | Hard cap on performance | Refinement loops |
| Programmatic benchmark | Doesn't measure real ARC performance | Download real data |

---

## Performance targets

| Milestone | Target | What's needed |
|-----------|--------|---------------|
| Current (programmatic benchmark) | 76% | Done ✓ |
| Real ARC-AGI-1 (current DSL, single-op tasks) | ~3% | Done ✓ (13/400 training tasks) |
| Real ARC-AGI-1 (+ segmentation + multi-op) | ~15–20% | `gsegment` + binary/ternary nodes |
| Real ARC-AGI-1 (+ conditional logic) | ~30–40% | ConditionalNode, parametric recolor |
| Real ARC-AGI-1 (+ refinement loops) | ~50%+ | Refinement loop infra |
| Real ARC-AGI-1 (+ LLM-guided prims) | ~60%+ | LLM integration |
| ARC-AGI-2 | TBD | Understand task distribution |
