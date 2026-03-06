# Improvement Plan: Next Training Run

The current run is at **24.1% solve rate (34/141 tasks)** on Epoch 1. Our goal is to push past 30%+ on the next run by expanding DSL expressivity and search efficiency.

### Category C: Library Learning
1. **Transition Matrix Sparsity** — Fixed by setting `min_tasks=1` and `min_size=3` in `agi.py`, allowing for self-abstraction and better primitive reuse across epochs.

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
