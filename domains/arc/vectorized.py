import numpy as np
from numba import njit
from typing import List, Any

Grid = List[List[int]]

def _to_np(g: Grid) -> np.ndarray:
    return np.array(g, dtype=np.int8)

def _from_np(a: np.ndarray) -> Grid:
    return a.tolist()

# ── Optimized Repeat Primitives ─────────────────────────────────────────────

def g_repeat_v_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (2, 1)))

def g_repeat_h_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (1, 2)))

def g_repeat_v3_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (3, 1)))

def g_repeat_h3_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (1, 3)))

def g_repeat_2x2_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (2, 2)))

def g_repeat_3x3_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.tile(a, (3, 3)))

# ── Optimized Scaling ────────────────────────────────────────────────────────

def g_scale_2x_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.repeat(np.repeat(a, 2, axis=0), 2, axis=1))

def g_scale_3x_vec(g: Grid) -> Grid:
    a = _to_np(g)
    return _from_np(np.repeat(np.repeat(a, 3, axis=0), 3, axis=1))

# ── Optimized Fractal Primitives ────────────────────────────────────────────

def g_fractal_inflate_vec(g: Grid) -> Grid:
    a = _to_np(g)
    mask = (a != 0).astype(np.int8)
    return _from_np(np.kron(mask, a))

def g_fractal_self_vec(g: Grid) -> Grid:
    a = _to_np(g)
    R, C = a.shape
    if R * R > 32 or C * C > 32: return g # Safety guard
    mask = (a != 0).astype(np.int8)
    return _from_np(np.kron(mask, a))

# ── Optimized Tiling ────────────────────────────────────────────────────────

def g_tile_self_vec(g: Grid) -> Grid:
    a = _to_np(g)
    R, C = a.shape
    if R == 0 or C == 0: return g
    reps_r = (30 + R - 1) // R
    reps_c = (30 + C - 1) // C
    tiled = np.tile(a, (reps_r, reps_c))
    return _from_np(tiled[:30, :30])

# ── Numba Accelerated BFS/Labeling ──────────────────────────────────────────

@njit
def _numba_label_kernel(grid, labels):
    R, C = grid.shape
    next_label = 1
    # We use a simple stack-based DFS inside Numba for performance
    stack = np.zeros((R * C, 2), dtype=np.int32)
    
    for r in range(R):
        for c in range(C):
            if grid[r, c] != 0 and labels[r, c] == 0:
                labels[r, c] = next_label
                top = 0
                stack[top, 0] = r
                stack[top, 1] = c
                top += 1
                
                while top > 0:
                    top -= 1
                    curr_r = stack[top, 0]
                    curr_c = stack[top, 1]
                    
                    # 4-connectivity
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < R and 0 <= nc < C:
                            if grid[nr, nc] != 0 and labels[nr, nc] == 0:
                                labels[nr, nc] = next_label
                                stack[top, 0] = nr
                                stack[top, 1] = nc
                                top += 1
                next_label += 1
    return next_label - 1

def g_flood_fill_vec(g: Grid) -> Grid:
    a = _to_np(g)
    R, C = a.shape
    if R == 0 or C == 0: return g
    
    zero_mask = (a == 0).astype(np.int8)
    labels = np.zeros((R, C), dtype=np.int32)
    _numba_label_kernel(zero_mask, labels)
    
    border_labels = set()
    border_labels.add(0) # background
    for r in range(R):
        border_labels.add(labels[r, 0])
        border_labels.add(labels[r, C-1])
    for c in range(C):
        border_labels.add(labels[0, c])
        border_labels.add(labels[R-1, c])
    
    out = a.copy()
    for r in range(R):
        for c in range(C):
            lbl = labels[r, c]
            if lbl > 0 and lbl not in border_labels:
                # Fill with a neighbor color if available
                fill_color = 1
                found = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and a[nr, nc] != 0:
                        fill_color = a[nr, nc]
                        found = True
                        break
                out[r, c] = fill_color
    return _from_np(out)
