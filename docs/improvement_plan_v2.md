# Improvement Plan: Next Training Run

The current run is at **24.1% solve rate (34/141 tasks)** on Epoch 1. Our goal is to push past 30%+ on the next run by expanding DSL expressivity and search efficiency.

## Analysis of Current Bottlenecks

Looking at the 107 **unsolved** tasks, the patterns that repeatedly fail fall into these categories:

### Category A: Missing Primitives (Expressivity Gap)
1. **Flood Fill** — Many ARC tasks require pixel-level flood fills (filling enclosed regions). We have `gdilate` but not a proper `g_flood_fill`.
2. **Multi-color connected components** — `_get_all_objects` only groups same-color neighbors. Tasks where objects span multiple colors need **any-nonzero** connectivity.
3. **Parametric Recoloring** — `g_replace_1_with_2` is hardcoded. Tasks need dynamic "most_frequent → least_frequent" or "foreground → specific" mappings.
4. **Object Count Predicates** — Tasks that ask "if there are N objects, do X" require predicate primitives.
5. **XOR / Difference** — Binary combinator that computes the symmetric difference of two grids.
6. **Downscaling** — We have `g_scale_2x` and `g_scale_3x` but no inverse `g_downscale_2x` for tasks where the output is smaller.

### Category B: Search Efficiency
1. **Crossover Rate** — Currently crossover vs mutation is governed by a fixed `mutation_rate=0.5`. More aggressive crossover could combine partial solutions.
2. **Progressive Deepening** — Start with `generations=30` for easy tasks, save budget for hard ones.

### Category C: Library Learning
1. **Transition Matrix Sparsity** — The learned matrix from prior runs has only 14 entries. Sub-tree extraction `min_tasks=2` is too strict for a single epoch on small subsets.

---

## Tier 1: High-Impact DSL Expansions (implementing now)

### 1.1 Flood Fill Primitive
```python
g_flood_fill(g: Grid) -> Grid
```
Fills all enclosed zero-regions (regions of `0` not connected to the grid border) with the surrounding color. Critical for "fill the hole" tasks.

### 1.2 Any-Color Connected Components  
```python
g_extract_objects_any(g: Grid) -> Grid
```
Extract the largest connected component treating ALL non-zero pixels as connected (not just same-color). Unlocks multi-colored object isolation.

### 1.3 Parametric Recoloring Suite
```python
g_fg_to_most_common(g: Grid) -> Grid     # all non-zero → most frequent non-zero color
g_fg_to_least_common(g: Grid) -> Grid    # all non-zero → least frequent non-zero color  
g_unique_color_per_obj(g: Grid) -> Grid  # assign distinct sequential colors to each object
```

### 1.4 Count / Size Predicate Primitives
```python
g_has_1_object(g: Grid) -> Grid   # returns g if exactly 1 object, else zeros
g_has_2_objects(g: Grid) -> Grid   # returns g if exactly 2 objects, else zeros
g_has_gt2_objects(g: Grid) -> Grid # returns g if >2 objects, else zeros
```
These give the `g_if` conditional branching actual meaningful predicates to work with.

### 1.5 Grid Difference / XOR
```python
g_xor(g1: Grid, g2: Grid) -> Grid  # symmetric difference: non-zero where grids differ
g_diff(g1: Grid, g2: Grid) -> Grid  # g1 minus g2: keep g1 pixels where g2 is zero
```

### 1.6 Downscaling
```python
g_downscale_2x(g: Grid) -> Grid  # majority-vote 2x2 blocks → 1 pixel
g_downscale_3x(g: Grid) -> Grid  # majority-vote 3x3 blocks → 1 pixel
```

## Tier 2: Search Engine Optimizations

### 2.1 Min-Tasks Threshold Auto-tuning in Library Learning
When fewer tasks are solved (<10), drop `min_tasks=1` to allow any reusable subtree to be promoted. The current threshold of `min_tasks=2` is too strict for the first epoch.

## Verification

All new primitives will have unit tests validating correct behavior on simple grids. The existing test suite must continue passing.
