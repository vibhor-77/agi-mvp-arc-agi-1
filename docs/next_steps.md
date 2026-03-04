# Next Steps

*Prioritized roadmap for this project — from immediate fixes to longer-term research.*

---

## Immediate (1–2 days)

### 1. Scaling Wake-Sleep Compute Cycles
The MVP has officially transitioned from programmatic toy tasks to real ARC-AGI data. We must now let the engine natively expand its abstraction dictionary.

**Step A (The Wake-Sleep Abstraction Gathering):**
```bash
python3 train_wake_sleep.py
```
This loops the 400 training tasks leveraging the native Sweet Spot defaults (`10x100`, 8 cores). Over 5 epochs, the `PrimitiveLibrary` will natively invent macro-primitives out of solved ASTs and compress the logic space into `arc_library.json`.

**Step B (The Out-of-Sample Evaluation):**
```bash
python3 evaluate_agi.py
```
This script acts identically to the hidden ARC evaluation constraint. It forces the solver to load `arc_library.json` and attempt the 400 *evaluation* tasks.

### 2. Shattering the 10% Accuracy Ceiling (The Expressivity Gap)
Currently, our `grid_cell_accuracy` rigorously denies false "partial credit" for mismatched grids. This mathematically anchors our current engine's capability at ~10% accuracy. Why? Because pure `BeamSearch` hits combinatorial explosion at depth 5-8. If a task requires 15 distinct primitive manipulations (a loop, a conditional, and a translation), it is mathematically unreachable under current constraints.
To break this ceiling, we *must* introduce **Native Turing-Complete Mechanisms**. The runtime must be upgraded to support dynamic algorithmic abstractions like unbounded `while` loops, iterators, and graph traversals. By expanding the symbolic engine's literal grammar depth, the Library Learning algorithm can map these structures without relying on fuzzy LLM synthesis.

### 3. Conditional & Looping Primitives
A massive fraction of ARC tasks apply a rule conditionally (e.g., "if cell is on the border, apply A; otherwise apply B") or require unknown loop counts. This needs a new node type in `core/tree.py` to support true Turing-complete ternary conditional rendering `if predicate(x) then branch_a(x) else branch_b(x)`.

---

## Short term (1–2 weeks)

### 4. Add SymPy canonicalization to tree scoring
Currently `grot90(grot90(grot90(grot90(x))))` and `gid(x)` have different sizes but identical semantics. Adding SymPy canonicalization to the semantic hashing engine would fix double-counting during evaluation.

### 5. Expanded Object segmentation primitives
We recently added basic `g_extract_objects` connected-component extraction. We need to expand this to handle diagonal connectivity and explicit background extraction.

### 4. Parametric Recoloring
Many ARC tasks require "map color X to color Y based on context" — not a fixed swap like `gswap_12`. A parametric version would take the most-frequent color in the input and remap it to the least-frequent.

### 5. Conditional primitives
A meaningful fraction of ARC tasks apply a rule conditionally (e.g., "if cell is on the border, apply A; otherwise apply B"). This needs a new node type in `core/tree.py` to support ternary conditional rendering operations `if predicate(x) then branch_a(x) else branch_b(x)`.

---

## Medium term (1–3 months)

### 6. ARC-AGI-2 Domain
ARC-AGI-2 uses the same JSON format but is significantly harder (contextual reasoning, multi-rule interaction). Testing against ARC-AGI-2 will prove whether the Disjoint Crossover pooling and Lexicase engines scale to harder tasks.

### 7. Core Object & Graph Reasoning
ARC requires understanding "objects" (contiguous blocks of pixels) that move independently. Our grid-level primitives (`grot90`) manipulate the universally whole image. We need map-reduce style logic that partitions a grid into an Object Graph, translates an inner bounding box, and reconstructs the canvas.

---

## Research directions (3+ months)

### 8. Measure Compression Ratio as an Intelligence Metric
After each library learning epoch, if `compression_ratio = total_nodes_in_solutions / n_solved_tasks` decreases monotonically over rounds, that is empirical rigorous evidence of the system "getting smarter" — encoding the same solutions in fewer symbols.

### 9. Zork / Text Adventure Domain
```python
# domains/zork/domain.py
class ZorkDomain(Domain):
    """
    State: feature vector extracted from game text
    Action: one of go_north/south/east/west, pick_up, drop, examine, ...
    Fitness: -mean_score after N steps
    """
```
Can the identical `BeamSearch` engine discover a compact symbolic policy that generalizes across Zork rooms? This tests whether the MDL framework scales to continuous decision problems.
