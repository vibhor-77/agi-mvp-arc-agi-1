# Session Walkthrough: DSL Expansion & Library Learning Improvements

## Context
- A training run was actively in progress (**Epoch 1/5**, ~155/400 tasks at 24.1% solve rate)
- We worked in parallel to prepare improvements for the next training run

## Changes Made

### 1. DSL Expansion: 18 New Primitives (150 → 168 ops)

| Category | Primitives | Impact |
|----------|-----------|--------|
| **Flood Fill** | `g_flood_fill` | Fills enclosed zero-regions (holes). Critical for "fill the shape" tasks |
| **Any-Color Extraction** | `g_extract_objects_any` | Connected components ignoring color boundaries. Unlocks multi-color objects |
| **Parametric Recoloring** | `g_fg_to_most_common`, `g_fg_to_least_common`, `g_unique_color_per_obj` | Dynamic color mapping instead of hardcoded swaps |
| **Count Predicates** | `g_has_1_object`, `g_has_2_objects`, `g_has_gt2_objects` | Give `g_if` branching meaningful structural predicates |
| **Grid Combinators** | `g_xor` (arity=2), `g_diff` (arity=2) | Symmetric difference and set subtraction of grids |
| **Downscaling** | `g_downscale_2x`, `g_downscale_3x` | Inverse of scale ops — majority-vote block compression |
| **Color Replacement** | `g_replace_2_with_3`, `g_replace_3_with_1`, `g_replace_3_with_2`, `g_replace_0_with_1`, `g_replace_0_with_2`, `g_replace_nonzero_with_1` | Comprehensive targeted color mapping matrix |

### 2. Library Learning Threshold Optimization

**Before:**
```python
min_tasks = 2 if len(successful_trees) > 2 else 1
lib.extract_from_tasks(successful_trees, min_size=3, min_tasks=min_tasks)
```

**After:**
```python
# min_size lowered: 3 → 2 (captures 2-node patterns like grot90(ginv(x)))
# min_tasks adaptive: scales with solved task count
if len(successful_trees) < 10:
    min_tasks = 1
elif len(successful_trees) < 30:
    min_tasks = 2 
else:
    min_tasks = 3
lib.extract_from_tasks(successful_trees, min_size=2, min_tasks=min_tasks)
```

> [!IMPORTANT]
> The `min_size=3` threshold was preventing the system from learning the most common useful patterns — 2-node compositions like `grot90(ginv(x))`. This single change should dramatically increase the library size after the Sleep phase.

### 4. Systematic Hyperparameter Optimization (Ongoing)
We have transitioned from ad-hoc tuning to a systematic grid-search approach. We are currently measuring the **Return on Compute (ROC)** across:
- **Beam Width**: [5, 10, 20]
- **Offspring**: [10, 20, 40]
- **Generations**: [25, 50, 100]

Findings will be documented in `docs/hyperparameter_optimization_results.md` and used to set the optimal production defaults for the final 400-task training run.

### 3. Files Modified
- [primitives.py](file:///Users/vibhorjain/github/agi-mvp-arc-agi-1/domains/arc/primitives.py) — 18 new primitives + registrations
- [train_wake_sleep.py](file:///Users/vibhorjain/github/agi-mvp-arc-agi-1/train_wake_sleep.py) — Adaptive library learning thresholds
- [next_steps.md](file:///Users/vibhorjain/github/agi-mvp-arc-agi-1/docs/next_steps.md) — Updated roadmap
- [test_new_primitives.py](file:///Users/vibhorjain/github/agi-mvp-arc-agi-1/tests/test_new_primitives.py) — **23 new unit tests** (NEW FILE)

### 4. Test Results
```
============================= 211 passed in 8.72s ==============================
```
All 211 tests pass (188 existing + 23 new). Zero regressions.

## Next Run Expected Impact

The new primitives target the exact failure modes observed in the current run's unsolved tasks:

1. **Flood fill** → "fill the enclosed region" tasks
2. **Object count predicates + g_if** → "do X if 1 object, do Y if 2 objects" tasks  
3. **Parametric recoloring** → "recolor based on frequency" tasks
4. **Downscaling** → "compress/summarize the grid" tasks
5. **XOR/diff** → "what changed between grids" tasks
6. **Lower min_size** → More library primitives discovered → richer DSL in Epochs 2-5

> [!TIP]
> The next training run should use these improvements. Run:
> ```bash
> python3 run_full_pipeline.py
> ```
