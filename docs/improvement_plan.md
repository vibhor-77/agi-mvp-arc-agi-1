# Improvement Plan: Next Training Run

The current run is at **24.1% solve rate (34/141 tasks)** on Epoch 1. Our goal is to push past 30%+ on the next run by expanding DSL expressivity and search efficiency.

### Category A: DSL Expressivity
1. **Fractal & Kronecker Primitives** — **Success.** Added `g_fractal_self` and `g_tile_self` to handle self-similar scaling and nesting logic.
2. **Object Discovery** — Current primitives like `g_extract_objects` are baseline. Future: multi-slot memory for object manipulation.

### Category B: Search & Exploration
1. **Soft-Success Learning** — **Success.** Sleep phase now extracts abstractions from both solved AND near-solved (>=90%) tasks.
2. **Increased Budget** — Planned: Bump evaluation beam width for near-miss tasks.

### Category C: Library Learning
1. **Wake-Sleep Marathon** — **Success.** COMPLETED a 400-task, 5-epoch training loop.
   - Initial solve rate: 9.25% (37/400 Tasks).
   - Final solve rate: **12.5% (50/400 Tasks)**.
   - Library growth: 150 abstractions learned, with 36 being actively reused to solve complex generalization tasks.
2. **Robust Multi-processing** — **Success.** Implemented manual `mp.Process` management with `p.kill()` for stragglers. Run 2,000 evaluations with zero hangs.

---

## Progress Report (Implementing Now)

### Infrastructure: Robust Parallel Search
- **Manual Process Management**: Replaced `ProcessPoolExecutor` with manual `multiprocessing.Process` management.
- **Hard Timeouts**: Implemented `p.kill()` for stragglers on the parent side, ensuring macOS processes actually terminate.
- **Communication Pipe**: Used `mp.Pipe` for reliable result retrieval from worker to parent.

---

### 1.1 Flood Fill / Interior Coloring (DONE)
- `g_frame_each_pixel`: Solves pixel-isolation tasks.
- `g_fill_rects_by_size`: Solves binary-size categorization tasks.
- `g_color_interior_by_area`: Solves parity-based interior coloring tasks.

### 1.2 Parametric Recoloring Suite (DONE)
- `g_fg_to_most_common`: Replaces foreground with the dominant color.
- `g_fg_to_least_common`: Replaces foreground with the rarest color.
- `g_unique_color_per_obj`: Color each component distinctly.

## Tier 2: Search Engine Optimizations

### 2.1 Min-Tasks Threshold Auto-tuning in Library Learning
When fewer tasks are solved (<10), drop `min_tasks=1` to allow any reusable subtree to be promoted. The current threshold of `min_tasks=2` is too strict for the first epoch.

## Verification

All new primitives will have unit tests validating correct behavior on simple grids. The existing test suite must continue passing.
