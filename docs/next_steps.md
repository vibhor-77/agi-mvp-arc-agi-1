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

### 2. ~~Shattering the 10% Accuracy Ceiling (The Expressivity Gap)~~ ✅ COMPLETED
The DSL has been expanded from 150 to **168 operations** with flood fill, parametric recoloring, object count predicates, grid XOR/diff, downscaling, and more. Combined with Turing-complete `g_if`/`g_while`, the solver now has the vocabulary needed to crack significantly more tasks. Current solve rate is at **~24%** and climbing.

### 3. ~~Conditional & Looping Primitives~~ ✅ COMPLETED
`g_if` (ternary branching) and `g_while` (bounded loops) are fully implemented with lazy evaluation in `core/tree.py`. Object count predicates (`g_has_1_object`, `g_has_2_objects`, `g_has_gt2_objects`) give the conditional branching meaningful predicate inputs.

---

## Short term (1–2 weeks)

### 4. Add SymPy canonicalization to tree scoring
Currently `grot90(grot90(grot90(grot90(x))))` and `gid(x)` have different sizes but identical semantics. Adding SymPy canonicalization to the semantic hashing engine would fix double-counting during evaluation.

### 5. ~~Expanded Object segmentation primitives~~ ✅ COMPLETED
`g_extract_objects_any` now handles any-color connectivity. `gmap_largest_cc` uses 8-connected components. `g_unique_color_per_obj` assigns sequential colors to distinct objects.

### 4. ~~Parametric Recoloring~~ ✅ COMPLETED
`g_fg_to_most_common` and `g_fg_to_least_common` dynamically remap foreground colors.

### 5. ~~Conditional primitives~~ ✅ COMPLETED
`g_if` with lazy evaluation, plus object count predicates for branching.

---

## Medium term (1–3 months)

### 6. ARC-AGI-2 Domain
ARC-AGI-2 uses the same JSON format but is significantly harder (contextual reasoning, multi-rule interaction). Testing against ARC-AGI-2 will prove whether the Disjoint Crossover pooling and Lexicase engines scale to harder tasks.

### 7. Core Object & Graph Reasoning
ARC requires understanding "objects" (contiguous blocks of pixels) that move independently. Our grid-level primitives (`grot90`) manipulate the universally whole image. We need map-reduce style logic that partitions a grid into an Object Graph, translates an inner bounding box, and reconstructs the canvas.

--- [ ] **Search Hyperparameter Tuning** (IN PROGRESS)
    - [x] Sweep combinations of Beam Size, Offspring, and Generations.
    - [x] Define "Return on Compute" (ROC) metric to find the efficiency sweet spot.
    - [ ] Apply winner to the 400-task full training run.

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
