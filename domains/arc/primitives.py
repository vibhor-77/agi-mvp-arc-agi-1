"""
domains/arc/primitives.py
=========================
All grid transformation primitives for ARC-AGI tasks.

Organisation
------------
Primitives are grouped into six categories.  All are *unary*:
Grid -> Grid.  They are registered into the module-level
``core.primitives.registry`` under domain="arc".

Categories
----------
  GEOMETRIC   — rotate, reflect, transpose
  COLOR       — swap, fill, invert, mod
  GRAVITY     — cells fall toward an edge
  SORT        — sort rows/cols
  STRUCTURAL  — frame, mirror, hollow, diagonal, scale
  PATTERN     — checkerboard, stripes, tile
  COUNTING    — bar-chart encoding, row filter, majority

Adding New Primitives
---------------------
Add a function below and register it at the bottom of the file:

    def my_new_op(g: Grid) -> Grid:
        \"\"\"Description of what it does.\"\"\"
        ...

    registry.register("my_new_op", my_new_op, domain="arc",
                      description="Description of what it does.")

The beam search will pick it up automatically the next time you call
``registry.names(domain="arc")``.

ARC color convention
--------------------
ARC uses integers 0–9. 0 is conventionally background.
All primitives treat 0 as background unless stated otherwise.
"""
from __future__ import annotations

import copy
from typing import Any, Callable
import numpy as np
from core.primitives import registry

# Type alias for readability
Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone(g: Grid) -> Grid:
    # Optimized nested list slice is ~10-20x faster than deepcopy
    return [row[:] for row in g]


def _rows(g: Grid) -> int:
    return len(g)


def _cols(g: Grid) -> int:
    return len(g[0]) if g else 0


# ---------------------------------------------------------------------------
# JIT Optimization Infrastructure
# ---------------------------------------------------------------------------

def _to_numpy_grid(value: Any) -> Any:
    """Convert Python list grids to compact ndarray; leave non-grids untouched."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list) and value and isinstance(value[0], list):
        return np.asarray(value, dtype=np.int16)
    return value


def _to_python_grid(value: Any) -> Any:
    """Convert ndarray outputs back to plain Python grids for stable APIs/tests."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _safe_grid_op(fn: Callable) -> Callable:
    """
    Wrap a grid op so it returns an unchanged clone of the first argument 
    on any error. Optimized for native NumPy execution.
    """
    arg_annotations = list(getattr(fn, "__annotations__", {}).values())
    wants_numpy = any("ndarray" in str(a) for a in arg_annotations)

    def _wrapped(*args, **kwargs):
        normalized_args = tuple(_to_numpy_grid(a) for a in args) if wants_numpy else args
        try:
            res = fn(*normalized_args, **kwargs)
            return _to_python_grid(res)
        except Exception:
            # Retry with original argument types for list-native implementations.
            try:
                res = fn(*args, **kwargs)
                return _to_python_grid(res)
            except Exception:
                pass
            # Fallback to a clone of the original input
            if args and isinstance(args[0], np.ndarray):
                return args[0].tolist()
            return _clone(args[0]) if args else []
    _wrapped.__name__ = fn.__name__
    setattr(_wrapped, "_arc_numpy_safe", True)
    return _wrapped


# ---------------------------------------------------------------------------
# GEOMETRIC (JIT ACCELERATED)
# ---------------------------------------------------------------------------

@_safe_grid_op
def grot90(g: np.ndarray) -> np.ndarray:
    """Rotate 90° clockwise."""
    return np.rot90(g, k=-1)

@_safe_grid_op
def grot180(g: np.ndarray) -> np.ndarray:
    """Rotate 180°."""
    return np.rot90(g, k=2)

@_safe_grid_op
def grot270(g: np.ndarray) -> np.ndarray:
    """Rotate 270° clockwise."""
    return np.rot90(g, k=1)

@_safe_grid_op
def grefl_h(g: np.ndarray) -> np.ndarray:
    """Reflect horizontally (flip left-right)."""
    return np.fliplr(g)

@_safe_grid_op
def grefl_v(np_g: np.ndarray) -> np.ndarray:
    """Reflect vertically (flip top-bottom)."""
    return np.flipud(np_g)

@_safe_grid_op
def gtrsp(g: np.ndarray) -> np.ndarray:
    """Transpose (swap rows and columns)."""
    return np.transpose(g)

@_safe_grid_op
def ganti_trsp(g: np.ndarray) -> np.ndarray:
    """Anti-transpose (reflect across the anti-diagonal)."""
    return np.transpose(np.flipud(np.fliplr(g)))


# ---------------------------------------------------------------------------
# COLOR (JIT ACCELERATED)
# ---------------------------------------------------------------------------

@_safe_grid_op
def ginv(g: np.ndarray) -> np.ndarray:
    """Invert colors: c → max_color - c."""
    if g.size == 0: return g
    m = np.max(g)
    return m - g

@_safe_grid_op
def gswap(g: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Helper for swapping two colors."""
    res = g.copy()
    mask1 = (g == c1)
    mask2 = (g == c2)
    res[mask1] = c2
    res[mask2] = c1
    return res

@_safe_grid_op
def gswap_01(g: np.ndarray) -> np.ndarray: return gswap(g, 0, 1)
@_safe_grid_op
def gswap_12(g: np.ndarray) -> np.ndarray: return gswap(g, 1, 2)
@_safe_grid_op
def gswap_23(g: np.ndarray) -> np.ndarray: return gswap(g, 2, 3)
@_safe_grid_op
def gswap_03(g: np.ndarray) -> np.ndarray: return gswap(g, 0, 3)
@_safe_grid_op
def gswap_13(g: np.ndarray) -> np.ndarray: return gswap(g, 1, 3)
@_safe_grid_op
def gswap_02(g: np.ndarray) -> np.ndarray: return gswap(g, 0, 2)

@_safe_grid_op
def gfill_bg(g: np.ndarray) -> np.ndarray:
    """Replace background (0) with color 5."""
    res = g.copy()
    res[g == 0] = 5
    return res

@_safe_grid_op
def gzero_bg(g: np.ndarray) -> np.ndarray:
    """Replace background (0) with color 9."""
    res = g.copy()
    res[g == 0] = 9
    return res

@_safe_grid_op
def gclear_nonbg(g: np.ndarray) -> np.ndarray:
    """Set all non-zero cells to 0."""
    return np.zeros_like(g)

@_safe_grid_op
def gmod2(g: np.ndarray) -> np.ndarray: return g % 2
@_safe_grid_op
def gmod3(g: np.ndarray) -> np.ndarray: return g % 3

@_safe_grid_op
def gmax_color(g: np.ndarray) -> np.ndarray:
    """Fill entire grid with the maximum color value present."""
    if g.size == 0: return g
    return np.full_like(g, np.max(g))

@_safe_grid_op
def gmin_color(g: np.ndarray) -> np.ndarray:
    """Fill entire grid with the minimum color value present."""
    if g.size == 0: return g
    return np.full_like(g, np.min(g))

@_safe_grid_op
def gid(g: np.ndarray) -> np.ndarray: return g.copy()


# ---------------------------------------------------------------------------
# GRAVITY  (non-zero cells fall toward an edge)
# ---------------------------------------------------------------------------

def ggravity_down(g: Grid) -> Grid:
    """Non-zero cells fall to the bottom of their column."""
    rows, cols = _rows(g), _cols(g)
    result = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        col = [g[r][c] for r in range(rows) if g[r][c] != 0]
        for i, v in enumerate(reversed(col)):
            result[rows - 1 - i][c] = v
    return result


def ggravity_up(g: Grid) -> Grid:
    """Non-zero cells float to the top of their column."""
    rows, cols = _rows(g), _cols(g)
    result = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        col = [g[r][c] for r in range(rows) if g[r][c] != 0]
        for i, v in enumerate(col):
            result[i][c] = v
    return result


def ggravity_right(g: Grid) -> Grid:
    """Non-zero cells slide to the right of their row."""
    result = []
    for row in g:
        non_z = [c for c in row if c != 0]
        zeros = [0] * (len(row) - len(non_z))
        result.append(zeros + non_z)
    return result


def ggravity_left(g: Grid) -> Grid:
    """Non-zero cells slide to the left of their row."""
    result = []
    for row in g:
        non_z = [c for c in row if c != 0]
        zeros = [0] * (len(row) - len(non_z))
        result.append(non_z + zeros)
    return result


@_safe_grid_op
def g_overlay(g1: Grid, g2: Grid) -> Grid:
    """Overlays non-zero pixels from g1 onto g2. If sizes differ, aligns to top-left."""
    out = _clone(g2)
    R1, C1 = len(g1), len(g1[0])
    R2, C2 = len(out), len(out[0])
    
    for r in range(min(R1, R2)):
        for c in range(min(C1, C2)):
            if g1[r][c] != 0:
                out[r][c] = g1[r][c]
    return out


# ---------------------------------------------------------------------------
# SORT
# ---------------------------------------------------------------------------

def gsort_rows_asc(g: Grid) -> Grid:
    """Sort each row in ascending order."""
    return [sorted(row) for row in g]


def gsort_rows_desc(g: Grid) -> Grid:
    """Sort each row in descending order."""
    return [sorted(row, reverse=True) for row in g]


def gsort_cols_asc(g: Grid) -> Grid:
    """Sort each column in ascending order."""
    t = gtrsp(g)
    t = [sorted(row) for row in t]
    return gtrsp(t)


def gsort_cols_desc(g: Grid) -> Grid:
    """Sort each column in descending order."""
    t = gtrsp(g)
    t = [sorted(row, reverse=True) for row in t]
    return gtrsp(t)


# ---------------------------------------------------------------------------
# STRUCTURAL
# ---------------------------------------------------------------------------

def _frame(g: Grid, color: int) -> Grid:
    rows, cols = _rows(g), _cols(g)
    if rows < 2 or cols < 2:
        return _clone(g)
    g2 = _clone(g)
    for c in range(cols):
        g2[0][c] = color
        g2[rows - 1][c] = color
    for r in range(rows):
        g2[r][0] = color
        g2[r][cols - 1] = color
    return g2


def gframe1(g: Grid) -> Grid:
    """Draw a border of color 1 around the grid."""
    return _frame(g, 1)


def gframe2(g: Grid) -> Grid:
    """Draw a border of color 2 around the grid."""
    return _frame(g, 2)


def gframe5(g: Grid) -> Grid:
    """Draw a border of color 5 around the grid."""
    return _frame(g, 5)


def gframe8(g: Grid) -> Grid:
    """Draw a border of color 8 around the grid."""
    return _frame(g, 8)


def gframe9(g: Grid) -> Grid:
    """Draw a border of color 9 around the grid."""
    return _frame(g, 9)


def gmirror_v(g: Grid) -> Grid:
    """Mirror vertically: bottom half becomes a copy of the top half."""
    rows = _rows(g)
    half = rows // 2
    result = _clone(g)
    for r in range(half):
        result[rows - 1 - r] = list(g[r])
    return result


def gmirror_h(g: Grid) -> Grid:
    """Mirror horizontally: right half becomes a copy of the left half."""
    result = []
    for row in g:
        half = len(row) // 2
        result.append(row[:half] + row[:half][::-1])
    return result


def ghollow(g: Grid) -> Grid:
    """
    Hollow out solid objects: interior cells whose four cardinal neighbours
    are all the same non-zero color become 0.
    """
    rows, cols = _rows(g), _cols(g)
    result = _clone(g)
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            v = g[r][c]
            if (v != 0
                    and g[r - 1][c] == v
                    and g[r + 1][c] == v
                    and g[r][c - 1] == v
                    and g[r][c + 1] == v):
                result[r][c] = 0
    return result


def gdiag1(g: Grid) -> Grid:
    """Overwrite the main diagonal with color 1."""
    result = _clone(g)
    n = min(_rows(g), _cols(g))
    for i in range(n):
        result[i][i] = 1
    return result


def gdiag9(g: Grid) -> Grid:
    """Overwrite the main diagonal with color 9."""
    result = _clone(g)
    n = min(_rows(g), _cols(g))
    for i in range(n):
        result[i][i] = 9
    return result


def ganti_diag1(g: Grid) -> Grid:
    """Overwrite the anti-diagonal with color 1."""
    result = _clone(g)
    rows, cols = _rows(g), _cols(g)
    n = min(rows, cols)
    for i in range(n):
        result[i][cols - 1 - i] = 1
    return result


def gscale2x(g: Grid) -> Grid:
    """Scale each cell to a 2×2 block (doubles grid dimensions)."""
    result = []
    for row in g:
        new_row = [c for c in row for _ in range(2)]
        result.append(new_row)
        result.append(list(new_row))
    return result


def gcrop_border(g: Grid) -> Grid:
    """Remove the outermost ring of cells (inverse of gframe*)."""
    rows, cols = _rows(g), _cols(g)
    if rows <= 2 or cols <= 2:
        return _clone(g)
    return [row[1:-1] for row in g[1:-1]]


def gpad1(g: Grid) -> Grid:
    """Add a ring of zeros around the grid."""
    rows, cols = _rows(g), _cols(g)
    top_bot = [[0] * (cols + 2)]
    result = top_bot + [[0] + row + [0] for row in g] + top_bot
    return result


# ---------------------------------------------------------------------------
# BINARY COMBINATORS
# ---------------------------------------------------------------------------

def goverlay(g1: Grid, g2: Grid) -> Grid:
    """Overlay g2 on top of g1. Non-zero cells in g2 overwrite g1."""
    if not g1 or not g2 or len(g1) != len(g2) or len(g1[0]) != len(g2[0]):
        return _clone(g1)
    
    result = _clone(g1)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if g2[r][c] != 0:
                result[r][c] = g2[r][c]
    return result

def ghstack(g1: Grid, g2: Grid) -> Grid:
    """Stack g1 and g2 horizontally."""
    if not g1 or not g2 or len(g1) != len(g2):
        return _clone(g1)
    
    return [row1 + row2 for row1, row2 in zip(g1, g2)]

def gvstack(g1: Grid, g2: Grid) -> Grid:
    """Stack g1 and g2 vertically."""
    if not g1 or not g2 or len(g1[0]) != len(g2[0]):
        return _clone(g1)
    
    return _clone(g1) + _clone(g2)

def gmask(g1: Grid, mask_g: Grid) -> Grid:
    """Keep cells from g1 where mask_g is non-zero, else set to 0."""
    if not g1 or not mask_g or len(g1) != len(mask_g) or len(g1[0]) != len(mask_g[0]):
        return _clone(g1)
    
    result = _clone(g1)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if mask_g[r][c] == 0:
                result[r][c] = 0
    return result


# ---------------------------------------------------------------------------
# PATTERN
# ---------------------------------------------------------------------------

def gcheckerboard(g: Grid) -> Grid:
    """Fill with a 2-color checkerboard: (r+c) even → 1, odd → 2."""
    rows, cols = _rows(g), _cols(g)
    return [[(1 if (r + c) % 2 == 0 else 2) for c in range(cols)]
            for r in range(rows)]


def gcheckerboard03(g: Grid) -> Grid:
    """Checkerboard in colors 0 and 3."""
    rows, cols = _rows(g), _cols(g)
    return [[(0 if (r + c) % 2 == 0 else 3) for c in range(cols)]
            for r in range(rows)]


def gstripe_h2(g: Grid) -> Grid:
    """Alternating horizontal stripes: even rows → 1, odd rows → 2."""
    rows, cols = _rows(g), _cols(g)
    return [[(1 if r % 2 == 0 else 2)] * cols for r in range(rows)]


def gstripe_v2(g: Grid) -> Grid:
    """Alternating vertical stripes: even cols → 1, odd cols → 2."""
    rows, cols = _rows(g), _cols(g)
    return [[(1 if c % 2 == 0 else 2) for c in range(cols)]
            for _ in range(rows)]


def gstripe_h3(g: Grid) -> Grid:
    """Horizontal stripes cycling through colors 1, 2, 3."""
    rows, cols = _rows(g), _cols(g)
    return [[(r % 3 + 1)] * cols for r in range(rows)]


def gstripe_v3(g: Grid) -> Grid:
    """Vertical stripes cycling through colors 1, 2, 3."""
    rows, cols = _rows(g), _cols(g)
    return [[(c % 3 + 1) for c in range(cols)] for _ in range(rows)]


def gtile2x2(g: Grid) -> Grid:
    """Tile the top-left 2×2 quadrant to fill the entire grid."""
    rows, cols = _rows(g), _cols(g)
    if rows < 2 or cols < 2:
        return _clone(g)
    return [[g[r % 2][c % 2] for c in range(cols)] for r in range(rows)]


def gtile_top_row(g: Grid) -> Grid:
    """Tile the top row to fill every row."""
    if not g:
        return _clone(g)
    return [list(g[0]) for _ in range(_rows(g))]


def gtile_left_col(g: Grid) -> Grid:
    """Tile the left column to fill every column."""
    rows, cols = _rows(g), _cols(g)
    return [[g[r][0]] * cols for r in range(rows)]


# ---------------------------------------------------------------------------
# COUNTING / STRUCTURAL ENCODING
# ---------------------------------------------------------------------------

def gcountbar(g: Grid) -> Grid:
    """
    Per-row bar chart encoding.

    For each row:
      - Count non-zero cells (= N)
      - Take the first non-zero color (= C)
      - Output row: first N cells are C, rest are 0
    """
    result = []
    for row in g:
        non_z = [c for c in row if c != 0]
        count = len(non_z)
        color = non_z[0] if non_z else 0
        result.append([color] * count + [0] * (len(row) - count))
    return result


def gmajority(g: Grid) -> Grid:
    """
    Per-row majority color fill.

    Each row is replaced by its majority (most common) non-zero color,
    broadcast across the entire row.
    """
    result = []
    for row in g:
        non_z = [c for c in row if c != 0]
        if non_z:
            majority = max(set(non_z), key=non_z.count)
            result.append([majority] * len(row))
        else:
            result.append(list(row))
    return result


def gkeep_rows2(g: Grid) -> Grid:
    """Zero out rows with fewer than 2 non-zero cells."""
    result = []
    for row in g:
        if sum(1 for c in row if c != 0) >= 2:
            result.append(list(row))
        else:
            result.append([0] * len(row))
    return result


def gkeep_rows3(g: Grid) -> Grid:
    """Zero out rows with fewer than 3 non-zero cells."""
    result = []
    for row in g:
        if sum(1 for c in row if c != 0) >= 3:
            result.append(list(row))
        else:
            result.append([0] * len(row))
    return result


def gkeep_rows4(g: Grid) -> Grid:
    """Zero out rows with fewer than 4 non-zero cells."""
    result = []
    for row in g:
        if sum(1 for c in row if c != 0) >= 4:
            result.append(list(row))
        else:
            result.append([0] * len(row))
    return result


def gcol_majority(g: Grid) -> Grid:
    """Per-column majority color fill (transpose of gmajority)."""
    return gtrsp(gmajority(gtrsp(g)))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
# fmt: off

_ARC_PRIMITIVES: dict[str, tuple[object, str]] = {
    # Geometric
    "grot90":       (grot90,       "Rotate 90° clockwise"),
    "grot180":      (grot180,      "Rotate 180°"),
    "grot270":      (grot270,      "Rotate 270° clockwise"),
    "grefl_h":      (grefl_h,      "Reflect left-right"),
    "grefl_v":      (grefl_v,      "Reflect top-bottom"),
    "gtrsp":        (gtrsp,        "Transpose rows/cols"),
    "ganti_trsp":   (ganti_trsp,   "Anti-transpose"),
    # Color
    "ginv":         (ginv,         "Invert colors (c → max-c)"),
    "gswap_01":     (gswap_01,     "Swap colors 0 and 1"),
    "gswap_02":     (gswap_02,     "Swap colors 0 and 2"),
    "gswap_03":     (gswap_03,     "Swap colors 0 and 3"),
    "gswap_12":     (gswap_12,     "Swap colors 1 and 2"),
    "gswap_13":     (gswap_13,     "Swap colors 1 and 3"),
    "gswap_23":     (gswap_23,     "Swap colors 2 and 3"),
    "gfill_bg":     (gfill_bg,     "Replace background(0) with 5"),
    "gzero_bg":     (gzero_bg,     "Replace background(0) with 9"),
    "gclear_nonbg": (gclear_nonbg, "Clear all non-background cells"),
    "gmod2":        (gmod2,        "c → c mod 2"),
    "gmod3":        (gmod3,        "c → c mod 3"),
    "gmax_color":   (gmax_color,   "Fill grid with max color present"),
    "gmin_color":   (gmin_color,   "Fill grid with min color present"),
    "gid":          (gid,          "Identity (no-op copy)"),
    # Gravity
    "ggravity_down":  (ggravity_down,  "Non-zero cells fall downward"),
    "ggravity_up":    (ggravity_up,    "Non-zero cells float upward"),
    "ggravity_right": (ggravity_right, "Non-zero cells slide right"),
    "ggravity_left":  (ggravity_left,  "Non-zero cells slide left"),
    # Sort
    "gsort_rows_asc":  (gsort_rows_asc,  "Sort each row ascending"),
    "gsort_rows_desc": (gsort_rows_desc, "Sort each row descending"),
    "gsort_cols_asc":  (gsort_cols_asc,  "Sort each column ascending"),
    "gsort_cols_desc": (gsort_cols_desc, "Sort each column descending"),
    # Structural
    "gframe1":      (gframe1,      "Border of color 1"),
    "gframe2":      (gframe2,      "Border of color 2"),
    "gframe5":      (gframe5,      "Border of color 5"),
    "gframe8":      (gframe8,      "Border of color 8"),
    "gframe9":      (gframe9,      "Border of color 9"),
    "gmirror_v":    (gmirror_v,    "Mirror: bottom = flipped top"),
    "gmirror_h":    (gmirror_h,    "Mirror: right = flipped left"),
    "ghollow":      (ghollow,      "Hollow out solid objects"),
    "gdiag1":       (gdiag1,       "Overwrite main diagonal with 1"),
    "gdiag9":       (gdiag9,       "Overwrite main diagonal with 9"),
    "ganti_diag1":  (ganti_diag1,  "Overwrite anti-diagonal with 1"),
    "gscale2x":     (gscale2x,     "Scale grid 2× (each cell → 2×2 block)"),
    "gcrop_border": (gcrop_border, "Remove outermost ring of cells"),
    "gpad1":        (gpad1,        "Add ring of zeros around grid"),
    # Pattern
    "gcheckerboard":    (gcheckerboard,    "Checkerboard in colors 1,2"),
    "gcheckerboard03":  (gcheckerboard03,  "Checkerboard in colors 0,3"),
    "gstripe_h2":       (gstripe_h2,       "Horizontal stripes colors 1,2"),
    "gstripe_v2":       (gstripe_v2,       "Vertical stripes colors 1,2"),
    "gstripe_h3":       (gstripe_h3,       "Horizontal stripes cycling 1,2,3"),
    "gstripe_v3":       (gstripe_v3,       "Vertical stripes cycling 1,2,3"),
    "gtile2x2":         (gtile2x2,         "Tile top-left 2×2 to fill grid"),
    "gtile_top_row":    (gtile_top_row,    "Tile top row to fill all rows"),
    "gtile_left_col":   (gtile_left_col,   "Tile left column to fill all cols"),
    # Counting
    "gcountbar":    (gcountbar,    "Bar-chart encode row counts"),
    "gmajority":    (gmajority,    "Fill each row with majority color"),
    "gcol_majority":(gcol_majority,"Fill each col with majority color"),
    "gkeep_rows2":  (gkeep_rows2,  "Zero rows with <2 non-zero cells"),
    "gkeep_rows3":  (gkeep_rows3,  "Zero rows with <3 non-zero cells"),
    "gkeep_rows4":  (gkeep_rows4,  "Zero rows with <4 non-zero cells"),
}

_ARC_BINARY_PRIMITIVES: dict[str, tuple[object, str]] = {
    # Combinators
    "goverlay":     (goverlay,     "Overlay obj2 on obj1 (obj2 non-zero overwrites)"),
    "ghstack":      (ghstack,      "Horizontally stack obj1 and obj2"),
    "gvstack":      (gvstack,      "Vertically stack obj1 and obj2"),
    "gmask":        (gmask,        "Mask obj1 using obj2's non-zero pixels"),
}
# fmt: on

for _name, (_fn, _desc) in _ARC_PRIMITIVES.items():
    registry.register(_name, _fn, domain="arc", description=_desc, arity=1, overwrite=True)  # type: ignore[arg-type]

for _name, (_fn, _desc) in _ARC_BINARY_PRIMITIVES.items():
    registry.register(_name, _fn, domain="arc", description=_desc, arity=2, overwrite=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ADDITIONAL PRIMITIVES
# (added for improved ARC-AGI-1 coverage and ARC-AGI-2/3 readiness)
# ---------------------------------------------------------------------------

# ── More color swaps (covers all same-magnitude pairs 0-9) ────────────────

def gswap_04(g: Grid) -> Grid:
    """Swap colors 0 and 4."""
    def s(c): return 4 if c == 0 else (0 if c == 4 else c)
    return [[s(c) for c in row] for row in g]


def gswap_05(g: Grid) -> Grid:
    """Swap colors 0 and 5."""
    def s(c): return 5 if c == 0 else (0 if c == 5 else c)
    return [[s(c) for c in row] for row in g]


def gswap_14(g: Grid) -> Grid:
    """Swap colors 1 and 4."""
    def s(c): return 4 if c == 1 else (1 if c == 4 else c)
    return [[s(c) for c in row] for row in g]


def gswap_24(g: Grid) -> Grid:
    """Swap colors 2 and 4."""
    def s(c): return 4 if c == 2 else (2 if c == 4 else c)
    return [[s(c) for c in row] for row in g]


# ── Structural: crop/pad variants ────────────────────────────────────────────

def gcrop_top(g: Grid) -> Grid:
    """Remove the top row."""
    return g[1:] if len(g) > 1 else _clone(g)


def gcrop_bottom(g: Grid) -> Grid:
    """Remove the bottom row."""
    return g[:-1] if len(g) > 1 else _clone(g)


def gcrop_left(g: Grid) -> Grid:
    """Remove the leftmost column."""
    return [row[1:] for row in g] if g and len(g[0]) > 1 else _clone(g)


def gcrop_right(g: Grid) -> Grid:
    """Remove the rightmost column."""
    return [row[:-1] for row in g] if g and len(g[0]) > 1 else _clone(g)


# ── Structural: fill operations ───────────────────────────────────────────────

def gfill_ones(g: Grid) -> Grid:
    """Fill the entire grid with 1 (useful for masking)."""
    rows, cols = _rows(g), _cols(g)
    return [[1] * cols for _ in range(rows)]


def gfill_zeros(g: Grid) -> Grid:
    """Fill the entire grid with 0 (clear)."""
    rows, cols = _rows(g), _cols(g)
    return [[0] * cols for _ in range(rows)]


# ── Color: modular arithmetic variants ───────────────────────────────────────

def gmod4(g: Grid) -> Grid:
    """Replace each color c with c % 4."""
    return [[c % 4 for c in row] for row in g]


def gmod5(g: Grid) -> Grid:
    """Replace each color c with c % 5."""
    return [[c % 5 for c in row] for row in g]


def gadd1_mod10(g: Grid) -> Grid:
    """Increment each color by 1 (mod 10), cycling 9 → 0."""
    return [[(c + 1) % 10 for c in row] for row in g]


def gsub1_mod10(g: Grid) -> Grid:
    """Decrement each color by 1 (mod 10), cycling 0 → 9."""
    return [[(c - 1) % 10 for c in row] for row in g]


# ── Structural: diagonal fill variants ───────────────────────────────────────

def gdiag2(g: Grid) -> Grid:
    """Overwrite the main diagonal with color 2."""
    result = _clone(g)
    n = min(_rows(g), _cols(g))
    for i in range(n):
        result[i][i] = 2
    return result


def gdiag5(g: Grid) -> Grid:
    """Overwrite the main diagonal with color 5."""
    result = _clone(g)
    n = min(_rows(g), _cols(g))
    for i in range(n):
        result[i][i] = 5
    return result


def ganti_diag2(g: Grid) -> Grid:
    """Overwrite the anti-diagonal with color 2."""
    result = _clone(g)
    rows, cols = _rows(g), _cols(g)
    n = min(rows, cols)
    for i in range(n):
        result[i][cols - 1 - i] = 2
    return result


# ── Pattern: more frame colors ────────────────────────────────────────────────

def gframe3(g: Grid) -> Grid:
    """Draw a border of color 3 around the grid."""
    return _frame(g, 3)


def gframe4(g: Grid) -> Grid:
    """Draw a border of color 4 around the grid."""
    return _frame(g, 4)


def gframe6(g: Grid) -> Grid:
    """Draw a border of color 6 around the grid."""
    return _frame(g, 6)


def gframe7(g: Grid) -> Grid:
    """Draw a border of color 7 around the grid."""
    return _frame(g, 7)


# ── Counting: column-oriented variants ───────────────────────────────────────

def gcountbar_cols(g: Grid) -> Grid:
    """Per-column bar chart encoding (transpose of gcountbar)."""
    return gtrsp(gcountbar(gtrsp(g)))


def gkeep_cols2(g: Grid) -> Grid:
    """Zero out columns with fewer than 2 non-zero cells."""
    rows, cols = _rows(g), _cols(g)
    result = [list(row) for row in g]
    for c in range(cols):
        col_vals = [g[r][c] for r in range(rows)]
        if sum(1 for v in col_vals if v != 0) < 2:
            for r in range(rows):
                result[r][c] = 0
    return result


def gkeep_cols3(g: Grid) -> Grid:
    """Zero out columns with fewer than 3 non-zero cells."""
    rows, cols = _rows(g), _cols(g)
    result = [list(row) for row in g]
    for c in range(cols):
        col_vals = [g[r][c] for r in range(rows)]
        if sum(1 for v in col_vals if v != 0) < 3:
            for r in range(rows):
                result[r][c] = 0
    return result


# ── Pattern: checkerboard variants ────────────────────────────────────────────

def gcheckerboard14(g: Grid) -> Grid:
    """Checkerboard in colors 1 and 4."""
    rows, cols = _rows(g), _cols(g)
    return [[(1 if (r + c) % 2 == 0 else 4) for c in range(cols)]
            for r in range(rows)]


def gcheckerboard25(g: Grid) -> Grid:
    """Checkerboard in colors 2 and 5."""
    rows, cols = _rows(g), _cols(g)
    return [[(2 if (r + c) % 2 == 0 else 5) for c in range(cols)]
            for r in range(rows)]


# ── Structural: dilation / erosion ────────────────────────────────────────────

def gdilate(g: Grid) -> Grid:
    """
    Binary dilation: any cell adjacent (4-connected) to a non-zero cell
    inherits its non-zero value.  Expands objects outward by one cell.
    Background (0) cells that touch foreground become the foreground color.
    """
    rows, cols = _rows(g), _cols(g)
    result = _clone(g)
    for r in range(rows):
        for c in range(cols):
            if g[r][c] == 0:
                # Check 4-connected neighbours
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] != 0:
                        result[r][c] = g[nr][nc]
                        break
    return result


def gerode(g: Grid) -> Grid:
    """
    Binary erosion: a non-zero cell is set to 0 if any of its 4-connected
    neighbours is 0.  Shrinks objects inward by one cell.
    """
    rows, cols = _rows(g), _cols(g)
    result = _clone(g)
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols) or g[nr][nc] == 0:
                        result[r][c] = 0
                        break
    return result


# ── Structural: connected-component utilities ─────────────────────────────────

def gborder_only(g: Grid) -> Grid:
    """
    Keep only cells on the outermost border; set interior to 0.
    Complement of gcrop_border: this keeps the border, not the interior.
    """
    rows, cols = _rows(g), _cols(g)
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                result[r][c] = g[r][c]
    return result


def ginterior_only(g: Grid) -> Grid:
    """
    Keep only the interior cells (not on the border); set border to 0.
    Complement of gborder_only.
    """
    rows, cols = _rows(g), _cols(g)
    if rows <= 2 or cols <= 2:
        return [[0] * cols for _ in range(rows)]
    result = [[0] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            result[r][c] = g[r][c]
    return result


# Register the new primitives
# ---------------------------------------------------------------------------
# SHAPES & OBJECTS
# ---------------------------------------------------------------------------

@_safe_grid_op
def g_filter_color(g: Grid, color: int) -> Grid:
    """
    Returns a blank grid where only cells matching `color` are preserved.
    """
    R, C = len(g), len(g[0])
    out = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if g[r][c] == color:
                out[r][c] = color
    return out

@_safe_grid_op
def g_extract_objects(g: Grid) -> Grid:
    """
    Finds the largest 4-way connected non-zero object, extracts it into the smallest bounding box,
    and returns that extracted grid. If no objects exist, returns the original grid.
    """
    R, C = len(g), len(g[0])
    visited = set()
    objects = []

    def get_neighbors(r, c, color):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited:
                if g[nr][nc] == color:
                    yield nr, nc

    for r in range(R):
        for c in range(C):
            col = g[r][c]
            if col != 0 and (r, c) not in visited:
                # BFS to find the object
                obj_cells = [(r, c)]
                visited.add((r, c))
                queue = [(r, c)]
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    for nr, nc in get_neighbors(curr_r, curr_c, col):
                        visited.add((nr, nc))
                        obj_cells.append((nr, nc))
                        queue.append((nr, nc))
                objects.append(obj_cells)

    if not objects:
        return _clone(g)

    # Find the largest object by cell count
    largest_obj = max(objects, key=len)

    # Compute bounding box
    min_r = min(r for r, c in largest_obj)
    max_r = max(r for r, c in largest_obj)
    min_c = min(c for r, c in largest_obj)
    max_c = max(c for r, c in largest_obj)

    # Extract bounding box
    box_R = max_r - min_r + 1
    box_C = max_c - min_c + 1
    out = [[0] * box_C for _ in range(box_R)]
    
    for r, c in largest_obj:
        out[r - min_r][c - min_c] = g[r][c]
        
    return out

# ---------------------------------------------------------------------------
# KEEP (Shape Topological Filters)
# ---------------------------------------------------------------------------

def _filter_objects(g: Grid, predicate: Callable[[list[tuple[int, int]], Grid], bool]) -> Grid:
    """Isolates all objects, checks them against `predicate`, and returns a grid containing only the passing objects."""
    objs = _get_all_objects(g)
    if not objs: return _clone(g)
    
    # Needs to see the raw cells relative to the grid
    R, C = len(g), len(g[0])
    out = [[0] * C for _ in range(R)]
    
    # For relative ranking (like largest/smallest), we evaluated the raw list
    remaining_objs = getattr(predicate, 'filter_list', lambda x: [o for o in x if predicate(o[3], o[2])])([ (min_r, min_c, box, cells) for min_r, min_c, box, cells in objs ])
    
    for (min_r, min_c, box, cells) in remaining_objs:
        tR, tC = len(box), len(box[0])
        for tr in range(tR):
            for tc in range(tC):
                if box[tr][tc] != 0:
                    rr = min_r + tr
                    cc = min_c + tc
                    if 0 <= rr < R and 0 <= cc < C:
                        out[rr][cc] = box[tr][tc]
    return out

@_safe_grid_op
def gkeep_square(g: Grid) -> Grid:
    """Keep only objects whose bounding box is 100% solid (no internal 0s)."""
    return _filter_objects(g, lambda cells, box: all(cell != 0 for row in box for cell in row))

@_safe_grid_op
def gkeep_hollow(g: Grid) -> Grid:
    """Keep only objects that contain at least one enclosed empty space."""
    def is_hollow(cells, box):
        # A simple approximation: if the # of non-zero pixels is less than the box area, and it's not simply a diagonal or L-shape
        R, C = len(box), len(box[0])
        if R <= 2 or C <= 2: return False
        
        # Check if any interior 0 pixel is fully surrounded
        for r in range(1, R-1):
            for c in range(1, C-1):
                if box[r][c] == 0:
                    # simplistic check: do we have non-zeros bounding this somehow? (A true flooded BFS would be better, but this works for ARC basics)
                    if box[r-1][c] != 0 and box[r+1][c] != 0 and box[r][c-1] != 0 and box[r][c+1] != 0:
                        return True
        return False
    return _filter_objects(g, is_hollow)

@_safe_grid_op
def gkeep_solid(g: Grid) -> Grid:
    """Keep only objects that are not hollow."""
    def is_hollow_local(cells, box):
        R, C = len(box), len(box[0])
        if R <= 2 or C <= 2: return False
        for r in range(1, R-1):
            for c in range(1, C-1):
                if box[r][c] == 0:
                    if box[r-1][c] != 0 and box[r+1][c] != 0 and box[r][c-1] != 0 and box[r][c+1] != 0:
                        return True
        return False
    return _filter_objects(g, lambda cells, box: not is_hollow_local(cells, box))

@_safe_grid_op
def gkeep_largest(g: Grid) -> Grid:
    """Keep only the single largest object."""
    def filter_largest(obj_list):
        if not obj_list: return []
        largest = max(obj_list, key=lambda x: len(x[3]))
        return [largest]
    predicate = lambda cells, box: True
    predicate.filter_list = filter_largest
    return _filter_objects(g, predicate)

@_safe_grid_op
def gkeep_smallest(g: Grid) -> Grid:
    """Keep only the single smallest object."""
    def filter_smallest(obj_list):
        if not obj_list: return []
        smallest = min(obj_list, key=lambda x: len(x[3]))
        return [smallest]
    predicate = lambda cells, box: True
    predicate.filter_list = filter_smallest
    return _filter_objects(g, predicate)

@_safe_grid_op
def gkeep_symmetric_v(g: Grid) -> Grid:
    """Keep only objects that are vertically symmetric within their own bounds."""
    def is_sym_v(cells, box):
        R, C = len(box), len(box[0])
        for r in range(R):
            for c in range(C // 2):
                if box[r][c] != box[r][C - 1 - c]:
                    return False
        return True
    return _filter_objects(g, is_sym_v)

@_safe_grid_op
def gkeep_symmetric_h(g: Grid) -> Grid:
    """Keep only objects that are horizontally symmetric within their own bounds."""
    def is_sym_h(cells, box):
        R, C = len(box), len(box[0])
        for r in range(R // 2):
            for c in range(C):
                if box[r][c] != box[R - 1 - r][c]:
                    return False
        return True
    return _filter_objects(g, is_sym_h)

def _make_gkeep_color(color_int: int) -> Callable:
    @_safe_grid_op
    def _gkeep_c(g: Grid) -> Grid:
        return _filter_objects(g, lambda cells, box: box[0][0] == color_int or any(c == color_int for row in box for c in row if c != 0))
    # Give it a nice name for registry display
    _gkeep_c.__name__ = f"gkeep_color{color_int}"
    return _gkeep_c

gkeep_color1 = _make_gkeep_color(1)
gkeep_color2 = _make_gkeep_color(2)
gkeep_color3 = _make_gkeep_color(3)
gkeep_color4 = _make_gkeep_color(4)
gkeep_color5 = _make_gkeep_color(5)
gkeep_color6 = _make_gkeep_color(6)
gkeep_color7 = _make_gkeep_color(7)
gkeep_color8 = _make_gkeep_color(8)
gkeep_color9 = _make_gkeep_color(9)


# ---------------------------------------------------------------------------
# MAP (Higher-Order Object Transforms)
# ---------------------------------------------------------------------------

def _get_all_objects(g: Grid) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Helper to extract all objects with their bounding box offsets."""
    R, C = len(g), len(g[0])
    visited = set()
    objects = []
    
    def get_neighbors(r, c, color):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited:
                if g[nr][nc] == color:
                    yield nr, nc

    for r in range(R):
        for c in range(C):
            col = g[r][c]
            if col != 0 and (r, c) not in visited:
                obj_cells = [(r, c)]
                visited.add((r, c))
                queue = [(r, c)]
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    for nr, nc in get_neighbors(curr_r, curr_c, col):
                        visited.add((nr, nc))
                        obj_cells.append((nr, nc))
                        queue.append((nr, nc))
                
                # Bounding box
                min_r = min(cr for cr, cc in obj_cells)
                max_r = max(cr for cr, cc in obj_cells)
                min_c = min(cc for cr, cc in obj_cells)
                max_c = max(cc for cr, cc in obj_cells)
                
                # Extract Box Grid
                box_R = max_r - min_r + 1
                box_C = max_c - min_c + 1
                box = [[0] * box_C for _ in range(box_R)]
                for cr, cc in obj_cells:
                    box[cr - min_r][cc - min_c] = g[cr][cc]
                    
                objects.append((min_r, min_c, box, obj_cells))
                
    return objects

def _apply_gmap(g: Grid, transform_fn: Callable) -> Grid:
    """Isolate objects natively, transform their boxes, and composite back onto void."""
    objs = _get_all_objects(g)
    if not objs:
        return _clone(g)
        
    # Return to original dimensions. We paste modified objects back into blank.
    R, C = len(g), len(g[0])
    out = [[0] * C for _ in range(R)]
    
    for (min_r, min_c, box, cells) in objs:
        transformed = transform_fn(box)
        tR = len(transformed)
        tC = len(transformed[0])
        # Paste back to approximated original bounding anchor
        for tr in range(tR):
            for tc in range(tC):
                if transformed[tr][tc] != 0:
                    rr = min_r + tr
                    cc = min_c + tc
                    if 0 <= rr < R and 0 <= cc < C:
                        out[rr][cc] = transformed[tr][tc]
    return out


@_safe_grid_op
def gmap_rot90(g: Grid) -> Grid:
    """Extract all objects, rotate them 90 deg clockwise in place, overlay back onto bg."""
    return _apply_gmap(g, grot90)

@_safe_grid_op
def gmap_rot180(g: Grid) -> Grid:
    """Extract all objects, rotate them 180 deg in place, overlay back onto bg."""
    return _apply_gmap(g, grot180)

@_safe_grid_op
def gmap_reflect_h(g: Grid) -> Grid:
    """Extract all objects, reflect them horizontally in place, overlay back onto bg."""
    return _apply_gmap(g, greflect_h)

@_safe_grid_op
def gmap_reflect_v(g: Grid) -> Grid:
    """Extract all objects, reflect them vertically in place, overlay back onto bg."""
    return _apply_gmap(g, greflect_v)

@_safe_grid_op
def gmap_fill_color(g: Grid) -> Grid:
    """Fills the interior bounding box of every object uniformly with its dominant color."""
    def fill_box(box):
        if not box or not box[0]: return box
        # dominant color of box
        freq = {}
        for row in box:
            for cell in row:
                if cell != 0:
                    freq[cell] = freq.get(cell, 0) + 1
        color = max(freq.items(), key=lambda x: x[1])[0] if freq else 1
        return [[color]*len(box[0]) for _ in range(len(box))]
        
    return _apply_gmap(g, fill_box)

@_safe_grid_op
def g_render_object(g1: Grid, g2: Grid) -> Grid:
    """
    g1 is the extracted small object. g2 is the large background grid.
    This primitive attempts to paste g1 into the center of g2.
    """
    r1, c1 = len(g1), len(g1[0])
    r2, c2 = len(g2), len(g2[0])
    
    if r1 > r2 or c1 > c2:
        return _clone(g2) # Doesn't fit
        
    out = _clone(g2)
    start_r = (r2 - r1) // 2
    start_c = (c2 - c1) // 2
    
    for r in range(r1):
        for c in range(c1):
            if g1[r][c] != 0:
                out[start_r + r][start_c + c] = g1[r][c]
                
    return out


# ── Compression & Gravity ──────────────────────────────────────────────────────

@_safe_grid_op
def g_remove_empty_rows(g: Grid) -> Grid:
    """Removes any row that is completely filled with 0s (Gravity Compression)."""
    return _clone([r for r in g if any(c != 0 for c in r)])

@_safe_grid_op
def g_remove_empty_cols(g: Grid) -> Grid:
    """Removes any column that is completely filled with 0s (Gravity Compression)."""
    if not g or not g[0]: return _clone(g)
    cols = [c for c in range(len(g[0])) if any(g[r][c] != 0 for r in range(len(g)))]
    if not cols: return [[]]
    return [[r[c] for c in cols] for r in g]

# ── Halving ────────────────────────────────────────────────────────────────────

@_safe_grid_op
def ghalf_top(g: Grid) -> Grid:
    """Returns the top half of the grid."""
    return _clone(g[:max(1, len(g)//2)])

@_safe_grid_op
def ghalf_bottom(g: Grid) -> Grid:
    """Returns the bottom half of the grid."""
    return _clone(g[len(g)//2:])

@_safe_grid_op
def ghalf_left(g: Grid) -> Grid:
    """Returns the left half of the grid."""
    if not g or not g[0]: return _clone(g)
    mid = max(1, len(g[0])//2)
    return [[c for c in r[:mid]] for r in g]

@_safe_grid_op
def ghalf_right(g: Grid) -> Grid:
    """Returns the right half of the grid."""
    if not g or not g[0]: return _clone(g)
    mid = len(g[0])//2
    return [[c for c in r[mid:]] for r in g]

# ── Scaling & Inflation ────────────────────────────────────────────────────────


@_safe_grid_op
def g_scale_2x(g: Grid) -> Grid:
    """Scale the grid by a factor of 2."""
    return [[g[r//2][c//2] for c in range(len(g[0])*2)] for r in range(len(g)*2)]


@_safe_grid_op
def g_scale_3x(g: Grid) -> Grid:
    """Scale the grid by a factor of 3."""
    return [[g[r//3][c//3] for c in range(len(g[0])*3)] for r in range(len(g)*3)]


@_safe_grid_op
def g_fractal_inflate(g: Grid) -> Grid:
    """Fractal expansion: replace each non-zero pixel with a copy of the entire grid."""
    R, C = len(g), len(g[0])
    out = [[0] * (C * C) for _ in range(R * R)]
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0:
                for ir in range(R):
                    for ic in range(C):
                        out[r * R + ir][c * C + ic] = g[ir][ic]
    return out



# ── Shift & Align ──────────────────────────────────────────────────────────────

@_safe_grid_op
def g_shift_up(g: Grid) -> Grid:
    """Shift all pixels up by 1, padding with 0 at the bottom."""
    R, C = len(g), len(g[0])
    out = [[0]*C for _ in range(R)]
    for r in range(1, R):
        for c in range(C):
            out[r-1][c] = g[r][c]
    return out

@_safe_grid_op
def g_shift_down(g: Grid) -> Grid:
    """Shift all pixels down by 1, padding with 0 at the top."""
    R, C = len(g), len(g[0])
    out = [[0]*C for _ in range(R)]
    for r in range(R-1):
        for c in range(C):
            out[r+1][c] = g[r][c]
    return out

@_safe_grid_op
def g_shift_left(g: Grid) -> Grid:
    """Shift all pixels left by 1, padding with 0 on the right."""
    R, C = len(g), len(g[0])
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(1, C):
            out[r][c-1] = g[r][c]
    return out

@_safe_grid_op
def g_shift_right(g: Grid) -> Grid:
    """Shift all pixels right by 1, padding with 0 on the left."""
    R, C = len(g), len(g[0])
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C-1):
            out[r][c+1] = g[r][c]
    return out

@_safe_grid_op
def g_align_up(g: Grid) -> Grid:
    """Shift all non-zero pixels up as a rigid body until hitting the top wall."""
    R, C = len(g), len(g[0])
    min_r = R
    for r in range(R):
        if any(c != 0 for c in g[r]):
            min_r = r
            break
    if min_r == R or min_r == 0:
        return _clone(g)
    
    out = [[0]*C for _ in range(R)]
    for r in range(min_r, R):
        for c in range(C):
            out[r - min_r][c] = g[r][c]
    return out

@_safe_grid_op
def g_align_down(g: Grid) -> Grid:
    """Shift all non-zero pixels down as a rigid body until hitting the bottom wall."""
    R, C = len(g), len(g[0])
    max_r = -1
    for r in range(R - 1, -1, -1):
        if any(c != 0 for c in g[r]):
            max_r = r
            break
    if max_r == -1 or max_r == R - 1:
        return _clone(g)
    
    shift = (R - 1) - max_r
    out = [[0]*C for _ in range(R)]
    for r in range(max_r, -1, -1):
        for c in range(C):
            out[r + shift][c] = g[r][c]
    return out

@_safe_grid_op
def g_align_left(g: Grid) -> Grid:
    """Shift all non-zero pixels left as a rigid body until hitting the left wall."""
    R, C = len(g), len(g[0])
    min_c = C
    for c in range(C):
        if any(g[r][c] != 0 for r in range(R)):
            min_c = c
            break
    if min_c == C or min_c == 0:
        return _clone(g)
    
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(min_c, C):
            out[r][c - min_c] = g[r][c]
    return out

# ── Computer Vision Geometry Primitives ────────────────────────────────────────────

@_safe_grid_op
def gmap_largest_cc(g: Grid) -> Grid:
    """Keep only the largest 8-connected component in the geometry."""
    R, C = len(g), len(g[0])
    visited = set()
    components = []
    
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                comp = []
                q = [(r, c)]
                color = g[r][c]
                visited.add((r, c))
                head = 0
                while head < len(q):
                    cr, cc = q[head]
                    head += 1
                    comp.append((cr, cc, color))
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append((len(comp), comp))
                
    if not components:
        return _clone(g)
        
    components.sort(key=lambda x: x[0], reverse=True)
    largest = components[0][1]
    
    out = [[0]*C for _ in range(R)]
    for r, c, color in largest:
        out[r][c] = color
    return out

@_safe_grid_op
def gmap_bounding_boxes(g: Grid) -> Grid:
    """Wrap a solid color bounding block around each disjoint 8-connected component."""
    R, C = len(g), len(g[0])
    visited = set()
    out = _clone(g)
    
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                color = g[r][c]
                q = [(r, c)]
                visited.add((r, c))
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                head = 0
                while head < len(q):
                    cr, cc = q[head]
                    head += 1
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                            min_r, max_r = min(min_r, nr), max(max_r, nr)
                            min_c, max_c = min(min_c, nc), max(max_c, nc)
                            
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        out[rr][cc] = color
                        
    return out

@_safe_grid_op
def g_align_right(g: Grid) -> Grid:
    """Shift all non-zero pixels right as a rigid body until hitting the right wall."""
    R, C = len(g), len(g[0])
    max_c = -1
    for c in range(C - 1, -1, -1):
        if any(g[r][c] != 0 for r in range(R)):
            max_c = c
            break
    if max_c == -1 or max_c == C - 1:
        return _clone(g)
    
    shift = (C - 1) - max_c
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(max_c, -1, -1):
            out[r][c + shift] = g[r][c]
    return out

@_safe_grid_op
def g_center_h(g: Grid) -> Grid:
    """Center all non-zero pixels horizontally."""
    R, C = len(g), len(g[0])
    cols = [c for c in range(C) if any(g[r][c] != 0 for r in range(R))]
    if not cols:
        return _clone(g)
    
    min_c, max_c = cols[0], cols[-1]
    width = max_c - min_c + 1
    target_c = (C - width) // 2
    shift = target_c - min_c
    
    if shift == 0:
        return _clone(g)
        
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and 0 <= c + shift < C:
                out[r][c + shift] = g[r][c]
    return out

@_safe_grid_op
def g_center_v(g: Grid) -> Grid:
    """Center all non-zero pixels vertically."""
    R, C = len(g), len(g[0])
    rows = [r for r in range(R) if any(c != 0 for c in g[r])]
    if not rows:
        return _clone(g)
        
    min_r, max_r = rows[0], rows[-1]
    height = max_r - min_r + 1
    target_r = (R - height) // 2
    shift = target_r - min_r
    
    if shift == 0:
        return _clone(g)
        
    out = [[0]*C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and 0 <= r + shift < R:
                out[r + shift][c] = g[r][c]
    return out

# ── Sequence & Extrapolation ───────────────────────────────────────────────────

@_safe_grid_op
def g_repeat_v(g: Grid) -> Grid:
    """Repeat the entire grid vertically, appending a copy to itself."""
    return [list(r) for r in g] + [list(r) for r in g]

@_safe_grid_op
def g_repeat_h(g: Grid) -> Grid:
    """Repeat the entire grid horizontally, appending a copy to itself."""
    return [list(r) + list(r) for r in g]

@_safe_grid_op
def g_repeat_v3(g: Grid) -> Grid:
    """Repeat the entire grid vertically 3 times."""
    return [list(r) for r in g] * 3

@_safe_grid_op
def g_repeat_h3(g: Grid) -> Grid:
    """Repeat the entire grid horizontally 3 times."""
    return [list(r) * 3 for r in g]

@_safe_grid_op
def g_repeat_2x2(g: Grid) -> Grid:
    """Tile the entire grid 2x2 times (doubles dimensions)."""
    return [list(r) * 2 for r in g] * 2

@_safe_grid_op
def g_repeat_3x3(g: Grid) -> Grid:
    """Tile the entire grid 3x3 times (triples dimensions)."""
    return [list(r) * 3 for r in g] * 3

# ── Target Color Replacements ──────────────────────────────────────────────────

def g_replace_1_with_2(g: Grid) -> Grid:
    """Transforms all 1s into 2s."""
    return [[2 if c == 1 else c for c in row] for row in g]

def g_replace_2_with_1(g: Grid) -> Grid:
    """Transforms all 2s into 1s."""
    return [[1 if c == 2 else c for c in row] for row in g]

def g_replace_1_with_3(g: Grid) -> Grid:
    """Transforms all 1s into 3s."""
    return [[3 if c == 1 else c for c in row] for row in g]

# ── Flood Fill ──────────────────────────────────────────────────────────────────

@_safe_grid_op
def g_flood_fill(g: Grid) -> Grid:
    """
    Fill all enclosed zero-regions (holes) with the surrounding non-zero color.
    A zero-region is 'enclosed' if it does not touch the grid border.
    Critical for 'fill the hole' ARC tasks.
    """
    R, C = len(g), len(g[0])
    if R == 0 or C == 0:
        return _clone(g)
    
    # 1) Find all zero-regions connected to the border (these are NOT holes)
    border_connected = set()
    queue = []
    for r in range(R):
        for c in range(C):
            if g[r][c] == 0 and (r == 0 or r == R-1 or c == 0 or c == C-1):
                if (r, c) not in border_connected:
                    border_connected.add((r, c))
                    queue.append((r, c))
    
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == 0 and (nr, nc) not in border_connected:
                border_connected.add((nr, nc))
                queue.append((nr, nc))
    
    # 2) Any zero cell NOT border-connected is a hole → fill with neighboring color
    out = _clone(g)
    for r in range(R):
        for c in range(C):
            if out[r][c] == 0 and (r, c) not in border_connected:
                # Find nearest non-zero neighbor color
                fill_color = 1
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < R and 0 <= nc < C and g[nr][nc] != 0:
                        fill_color = g[nr][nc]
                        break
                out[r][c] = fill_color
    return out


# ── Any-Color Connected Component Extraction ────────────────────────────────

@_safe_grid_op
def g_extract_objects_any(g: Grid) -> Grid:
    """
    Extract the largest connected component treating ALL non-zero pixels as connected
    (not restricted to same-color). Returns the bounding box of the largest blob.
    Unlocks multi-colored object isolation.
    """
    R, C = len(g), len(g[0])
    visited = set()
    objects = []
    
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                obj_cells = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    obj_cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] != 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                objects.append(obj_cells)
    
    if not objects:
        return _clone(g)
    
    largest = max(objects, key=len)
    min_r = min(r for r, c in largest)
    max_r = max(r for r, c in largest)
    min_c = min(c for r, c in largest)
    max_c = max(c for r, c in largest)
    
    box_R = max_r - min_r + 1
    box_C = max_c - min_c + 1
    out = [[0] * box_C for _ in range(box_R)]
    for r, c in largest:
        out[r - min_r][c - min_c] = g[r][c]
    return out


# ── Parametric Recoloring ──────────────────────────────────────────────────────

@_safe_grid_op
def g_fg_to_most_common(g: Grid) -> Grid:
    """Map ALL non-zero cells to the single most-frequent non-zero color."""
    freq = {}
    for row in g:
        for c in row:
            if c != 0:
                freq[c] = freq.get(c, 0) + 1
    if not freq:
        return _clone(g)
    most_common = max(freq, key=freq.get)
    return [[most_common if c != 0 else 0 for c in row] for row in g]

@_safe_grid_op
def g_fg_to_least_common(g: Grid) -> Grid:
    """Map ALL non-zero cells to the single least-frequent non-zero color."""
    freq = {}
    for row in g:
        for c in row:
            if c != 0:
                freq[c] = freq.get(c, 0) + 1
    if not freq:
        return _clone(g)
    least_common = min(freq, key=freq.get)
    return [[least_common if c != 0 else 0 for c in row] for row in g]

@_safe_grid_op
def g_unique_color_per_obj(g: Grid) -> Grid:
    """Assign sequential colors (1, 2, 3, ...) to each distinct connected object."""
    R, C = len(g), len(g[0])
    visited = set()
    out = [[0] * C for _ in range(R)]
    color_idx = 0
    
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                color_idx += 1
                assign_color = color_idx % 10 if color_idx % 10 != 0 else 1
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    out[cr][cc] = assign_color
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] != 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return out


# ── Object Count Predicates (for g_if branching) ───────────────────────────────

def _count_objects(g: Grid) -> int:
    """Count disjoint 4-connected non-zero components."""
    R, C = len(g), len(g[0]) if g else 0
    visited = set()
    count = 0
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                count += 1
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] != 0 and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
    return count

@_safe_grid_op
def g_has_1_object(g: Grid) -> Grid:
    """Returns a copy of g if exactly 1 connected object exists, else a zeroed grid (for g_if predicates)."""
    if _count_objects(g) == 1:
        return _clone(g)
    return [[0] * len(g[0]) for _ in range(len(g))]

@_safe_grid_op
def g_has_2_objects(g: Grid) -> Grid:
    """Returns a copy of g if exactly 2 connected objects exist, else a zeroed grid."""
    if _count_objects(g) == 2:
        return _clone(g)
    return [[0] * len(g[0]) for _ in range(len(g))]

@_safe_grid_op
def g_has_gt2_objects(g: Grid) -> Grid:
    """Returns a copy of g if more than 2 connected objects exist, else a zeroed grid."""
    if _count_objects(g) > 2:
        return _clone(g)
    return [[0] * len(g[0]) for _ in range(len(g))]


# ── Grid XOR / Difference ──────────────────────────────────────────────────────

@_safe_grid_op
def g_xor(g1: Grid, g2: Grid) -> Grid:
    """Symmetric difference: non-zero in output where exactly one input is non-zero."""
    R1, C1 = len(g1), len(g1[0])
    R2, C2 = len(g2), len(g2[0])
    R, C = max(R1, R2), max(C1, C2)
    out = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            v1 = g1[r][c] if r < R1 and c < C1 else 0
            v2 = g2[r][c] if r < R2 and c < C2 else 0
            if v1 != 0 and v2 == 0:
                out[r][c] = v1
            elif v1 == 0 and v2 != 0:
                out[r][c] = v2
    return out

@_safe_grid_op
def g_diff(g1: Grid, g2: Grid) -> Grid:
    """Set difference: keep g1 pixels only where g2 is zero. Useful for subtraction."""
    R1, C1 = len(g1), len(g1[0])
    R2, C2 = len(g2), len(g2[0])
    R, C = min(R1, R2), min(C1, C2)
    out = _clone(g1)
    for r in range(R):
        for c in range(C):
            if r < R2 and c < C2 and g2[r][c] != 0:
                out[r][c] = 0
    return out


# ── Downscaling ───────────────────────────────────────────────────────────────

@_safe_grid_op
def g_downscale_2x(g: Grid) -> Grid:
    """Majority-vote downscale: each 2x2 block → 1 pixel (uses most common non-zero color, or 0)."""
    R, C = len(g), len(g[0])
    oR, oC = R // 2, C // 2
    if oR == 0 or oC == 0:
        return _clone(g)
    out = [[0] * oC for _ in range(oR)]
    for r in range(oR):
        for c in range(oC):
            freq = {}
            for dr in range(2):
                for dc in range(2):
                    sr, sc = r*2+dr, c*2+dc
                    if sr < R and sc < C and g[sr][sc] != 0:
                        v = g[sr][sc]
                        freq[v] = freq.get(v, 0) + 1
            out[r][c] = max(freq, key=freq.get) if freq else 0
    return out

@_safe_grid_op
def g_downscale_3x(g: Grid) -> Grid:
    """Majority-vote downscale: each 3x3 block → 1 pixel."""
    R, C = len(g), len(g[0])
    oR, oC = R // 3, C // 3
    if oR == 0 or oC == 0:
        return _clone(g)
    out = [[0] * oC for _ in range(oR)]
    for r in range(oR):
        for c in range(oC):
            freq = {}
            for dr in range(3):
                for dc in range(3):
                    sr, sc = r*3+dr, c*3+dc
                    if sr < R and sc < C and g[sr][sc] != 0:
                        v = g[sr][sc]
                        freq[v] = freq.get(v, 0) + 1
            out[r][c] = max(freq, key=freq.get) if freq else 0
    return out


# ── Extended Color Replacements ────────────────────────────────────────────────

def _make_replace(src: int, dst: int):
    """Factory for targeted color replacement primitives."""
    @_safe_grid_op
    def _repl(g: Grid) -> Grid:
        return [[dst if c == src else c for c in row] for row in g]
    _repl.__name__ = f"g_replace_{src}_with_{dst}"
    return _repl

g_replace_2_with_3 = _make_replace(2, 3)
g_replace_3_with_1 = _make_replace(3, 1)
g_replace_3_with_2 = _make_replace(3, 2)
g_replace_0_with_1 = _make_replace(0, 1)
g_replace_0_with_2 = _make_replace(0, 2)
g_replace_nonzero_with_1 = lambda g: [[1 if c != 0 else 0 for c in row] for row in g]
g_replace_nonzero_with_1.__name__ = "g_replace_nonzero_with_1"


# ── Logical Branching ────────────────────────────────────────────────────────
def g_if(cond: Grid, true_branch: Grid, false_branch: Grid) -> Grid:
    """
    Ternary logical branching.
    If the cond grid contains ANY non-zero pixels (is 'truthy'), 
    evaluates to the true_branch grid.
    Otherwise (an empty grid or all 0s), evaluates to the false_branch grid.
    """
    has_pixels = any(val != 0 for row in cond for val in row)
    return _clone(true_branch) if has_pixels else _clone(false_branch)

def g_while(cond: Grid, body_branch: Grid) -> Grid:
    """
    Dummy python function to satisfy python Arity pytest validation.
    The true recurrent logic is physically intercepted and dynamically evaluated inside core.tree.Node.eval()
    """
    return _clone(body_branch)

_NEW_ARC_PRIMITIVES: dict[str, tuple[object, str]] = {
    # LOGIC (True Turing Completeness)
    "g_if":             (g_if, "If cond has any non-zero pixels, return true_branch; else return false_branch"),
    "g_while":          (g_while, "Recurrent unrolled loop: repeatedly applies body_branch to x until cond(x) is False"),
    
    # More color swaps
    "gswap_04":         (gswap_04,         "Swap colors 0 and 4"),
    "gswap_05":         (gswap_05,         "Swap colors 0 and 5"),
    "gswap_14":         (gswap_14,         "Swap colors 1 and 4"),
    "gswap_24":         (gswap_24,         "Swap colors 2 and 4"),
    # Crop variants
    "gcrop_top":        (gcrop_top,        "Remove top row"),
    "gcrop_bottom":     (gcrop_bottom,     "Remove bottom row"),
    "gcrop_left":       (gcrop_left,       "Remove leftmost column"),
    "gcrop_right":      (gcrop_right,      "Remove rightmost column"),
    # Fill
    "gfill_ones":       (gfill_ones,       "Fill grid with 1"),
    "gfill_zeros":      (gfill_zeros,      "Fill grid with 0"),
    # Modular arithmetic
    "gmod4":            (gmod4,            "c → c mod 4"),
    "gmod5":            (gmod5,            "c → c mod 5"),
    "gadd1_mod10":      (gadd1_mod10,      "c → (c+1) mod 10"),
    "gsub1_mod10":      (gsub1_mod10,      "c → (c-1) mod 10"),
    # Diagonal variants
    "gdiag2":           (gdiag2,           "Overwrite main diagonal with 2"),
    "gdiag5":           (gdiag5,           "Overwrite main diagonal with 5"),
    "ganti_diag2":      (ganti_diag2,      "Overwrite anti-diagonal with 2"),
    # More frames
    "gframe3":          (gframe3,          "Border of color 3"),
    "gframe4":          (gframe4,          "Border of color 4"),
    "gframe6":          (gframe6,          "Border of color 6"),
    "gframe7":          (gframe7,          "Border of color 7"),
    # Column counting
    "gcountbar_cols":   (gcountbar_cols,   "Bar-chart encode column counts"),
    "gkeep_cols2":      (gkeep_cols2,      "Zero cols with <2 non-zero cells"),
    "gkeep_cols3":      (gkeep_cols3,      "Zero cols with <3 non-zero cells"),
    # Checkerboard variants
    "gcheckerboard14":  (gcheckerboard14,  "Checkerboard in colors 1,4"),
    "gcheckerboard25":  (gcheckerboard25,  "Checkerboard in colors 2,5"),
    # Morphological
    "gdilate":          (gdilate,          "Morphological dilation (expand objects)"),
    "gerode":           (gerode,           "Morphological erosion (shrink objects)"),
    # Border/interior split
    "gborder_only":     (gborder_only,     "Keep only border cells"),
    "ginterior_only":   (ginterior_only,   "Keep only interior cells"),
    # Compression & Gravity
    "g_remove_empty_rows": (g_remove_empty_rows, "Removes all empty rows"),
    "g_remove_empty_cols": (g_remove_empty_cols, "Removes all empty columns"),
    # Halving
    "ghalf_top":        (ghalf_top,        "Extract top half of grid"),
    "ghalf_bottom":     (ghalf_bottom,     "Extract bottom half of grid"),
    "ghalf_left":       (ghalf_left,       "Extract left half of grid"),
    "ghalf_right":      (ghalf_right,      "Extract right half of grid"),
    # Shifts
    "g_shift_up":       (g_shift_up,       "Shift all pixels up 1 space"),
    "g_shift_down":     (g_shift_down,     "Shift all pixels down 1 space"),
    "g_shift_left":     (g_shift_left,     "Shift all pixels left 1 space"),
    "g_shift_right":    (g_shift_right,    "Shift all pixels right 1 space"),
    # Alignments
    "g_align_up":       (g_align_up,       "Align non-zero pixels to top wall"),
    "g_align_down":     (g_align_down,     "Align non-zero pixels to bottom wall"),
    "g_align_left":     (g_align_left,     "Align non-zero pixels to left wall"),
    "g_align_right":    (g_align_right,    "Align non-zero pixels to right wall"),
    "g_center_h":       (g_center_h,       "Center non-zero pixels horizontally"),
    "g_center_v":       (g_center_v,       "Center non-zero pixels vertically"),
    # Sequence & Extrapolation
    "g_repeat_v":       (g_repeat_v,       "Repeat grid vertically"),
    "g_repeat_h":       (g_repeat_h,       "Repeat grid horizontally"),
    "g_repeat_v3":      (g_repeat_v3,      "Repeat grid vertically 3 times"),
    "g_repeat_h3":      (g_repeat_h3,      "Repeat grid horizontally 3 times"),
    "g_repeat_2x2":     (g_repeat_2x2,     "Tile grid 2x2 times"),
    "g_repeat_3x3":     (g_repeat_3x3,     "Tile grid 3x3 times"),
    # Color Target Replacements
    "g_replace_1_with_2": (g_replace_1_with_2, "Transforms color 1 to 2"),
    "g_replace_2_with_1": (g_replace_2_with_1, "Transforms color 2 to 1"),
    "g_replace_1_with_3": (g_replace_1_with_3, "Transforms color 1 to 3"),
    # Scaling & Inflation
    "g_fractal_inflate": (g_fractal_inflate, "Fractal expansion: tile grid into its own non-zero pixels"),
    "g_scale_2x":       (g_scale_2x,       "Scale up the grid 2x"),
    "g_scale_3x":       (g_scale_3x,       "Scale up the grid 3x"),
    # Objects
    "g_filter_color":   (g_filter_color,   "Extracts mask of a single color"),
    "g_extract_objects": (g_extract_objects, "Extracts largest grid object to bounding box"),
    "g_render_object":  (g_render_object,  "Pastes a small object into the center of a background grid"),
    # MAP (Higher-Order)
    "gmap_rot90":       (gmap_rot90, "Rotate all isolated shapes 90° clockwise in place"),
    "gmap_rot180":      (gmap_rot180, "Rotate all isolated shapes 180° in place"),
    "gmap_reflect_h":   (gmap_reflect_h, "Reflect all isolated shapes horizontally in place"),
    "gmap_reflect_v":   (gmap_reflect_v, "Reflect all isolated shapes vertically in place"),
    "gmap_fill_color":  (gmap_fill_color, "Fill isolated shape bounding boxes with solid dominant color"),
    # KEEP (Shape Topological Filters)
    "gkeep_square":     (gkeep_square, "Keep only objects whose bounding box is 100% solid"),
    "gkeep_hollow":     (gkeep_hollow, "Keep only objects that enclose a hollow frame"),
    "gkeep_solid":      (gkeep_solid, "Keep only objects that are not hollow"),
    "gkeep_largest":    (gkeep_largest, "Keep only the single largest object"),
    "gkeep_smallest":   (gkeep_smallest, "Keep only the single smallest object"),
    "gkeep_symmetric_v": (gkeep_symmetric_v, "Keep only vertically symmetric shapes"),
    "gkeep_symmetric_h": (gkeep_symmetric_h, "Keep only horizontally symmetric shapes"),
    "gkeep_color1":     (gkeep_color1, "Keep only objects containing color 1"),
    "gkeep_color2":     (gkeep_color2, "Keep only objects containing color 2"),
    "gkeep_color3":     (gkeep_color3, "Keep only objects containing color 3"),
    "gkeep_color4":     (gkeep_color4, "Keep only objects containing color 4"),
    "gkeep_color5":     (gkeep_color5, "Keep only objects containing color 5"),
    "gkeep_color6":     (gkeep_color6, "Keep only objects containing color 6"),
    "gkeep_color7":     (gkeep_color7, "Keep only objects containing color 7"),
    "gkeep_color8":     (gkeep_color8, "Keep only objects containing color 8"),
    "gkeep_color9":     (gkeep_color9, "Keep only objects containing color 9"),
    # COMPOSITORS
    "g_overlay":        (g_overlay, "Overlay grid 1's non-zero pixels onto grid 2"),
    
    # Computer Vision / Object Detection
    "gmap_largest_cc":     (gmap_largest_cc, "Keep only the largest 8-connected topological component"),
    "gmap_bounding_boxes": (gmap_bounding_boxes, "Wrap a solid bounding box around each independent shape component"),
    
    # ── NEW: Flood Fill ──
    "g_flood_fill":     (g_flood_fill, "Fill enclosed zero-regions (holes) with surrounding color"),
    
    # ── NEW: Any-Color Object Extraction ──
    "g_extract_objects_any": (g_extract_objects_any, "Extract largest connected component (any non-zero as connected)"),
    
    # ── NEW: Parametric Recoloring ──
    "g_fg_to_most_common":   (g_fg_to_most_common,   "Map all non-zero cells to the most frequent non-zero color"),
    "g_fg_to_least_common":  (g_fg_to_least_common,  "Map all non-zero cells to the least frequent non-zero color"),
    "g_unique_color_per_obj": (g_unique_color_per_obj, "Assign sequential colors (1,2,3,...) to each distinct object"),
    
    # ── NEW: Object Count Predicates (for g_if branching) ──
    "g_has_1_object":   (g_has_1_object,   "Returns g if exactly 1 object, else zeros (predicate)"),
    "g_has_2_objects":   (g_has_2_objects,   "Returns g if exactly 2 objects, else zeros (predicate)"),
    "g_has_gt2_objects": (g_has_gt2_objects, "Returns g if >2 objects, else zeros (predicate)"),
    
    # ── NEW: Grid XOR / Difference ──
    "g_xor":  (g_xor,  "Symmetric difference: non-zero where exactly one input is non-zero"),
    "g_diff": (g_diff, "Set difference: keep g1 pixels where g2 is zero"),
    
    # ── NEW: Downscaling ──
    "g_downscale_2x": (g_downscale_2x, "Majority-vote downscale: 2x2 blocks → 1 pixel"),
    "g_downscale_3x": (g_downscale_3x, "Majority-vote downscale: 3x3 blocks → 1 pixel"),
    
    # ── NEW: Extended Color Replacements ──
    "g_replace_2_with_3":      (g_replace_2_with_3,      "Transforms color 2 to 3"),
    "g_replace_3_with_1":      (g_replace_3_with_1,      "Transforms color 3 to 1"),
    "g_replace_3_with_2":      (g_replace_3_with_2,      "Transforms color 3 to 2"),
    "g_replace_0_with_1":      (g_replace_0_with_1,      "Transforms color 0 (bg) to 1"),
    "g_replace_0_with_2":      (g_replace_0_with_2,      "Transforms color 0 (bg) to 2"),
    "g_replace_nonzero_with_1": (g_replace_nonzero_with_1, "Reduces all non-zero to binary mask of 1s"),
}

_BINARY_NEW_OPS = {"g_overlay", "g_render_object", "g_filter_color", "g_xor", "g_diff"}

for _name, (_fn, _desc) in _NEW_ARC_PRIMITIVES.items():
    if _name == "g_if":
        _arity = 3
    elif _name == "g_while":
        _arity = 2
    elif _name in _BINARY_NEW_OPS:
        _arity = 2
    else:
        _arity = 1
    registry.register(_name, _fn, domain="arc", description=_desc, arity=_arity, overwrite=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OBJECT-LEVEL PRIMITIVES  (batch 3 — targeting observed failure modes)
# ---------------------------------------------------------------------------

def g_scale_by_color(g: Grid) -> Grid:
    """
    Scale each pixel into a block whose size equals its color value.
    Each cell at (r, c) with value v creates a v×v filled block placed at
    pixel position (r * max_v, c * max_v), so blocks never overlap.
    Zero cells leave their region empty.
    """
    flat = [v for row in g for v in row if v != 0]
    if not flat:
        return _clone(g)
    max_v = max(flat)
    if max_v == 0:
        return _clone(g)
    rows, cols = _rows(g), _cols(g)
    out_rows, out_cols = rows * max_v, cols * max_v
    if out_rows > 30 or out_cols > 30:
        return _clone(g)
    result = [[0] * out_cols for _ in range(out_rows)]
    for r in range(rows):
        for c in range(cols):
            v = g[r][c]
            if v == 0:
                continue
            for dr in range(v):
                for dc in range(v):
                    result[r * max_v + dr][c * max_v + dc] = v
    return result


def g_frame_each_pixel(g: Grid) -> Grid:
    """
    For each isolated non-zero pixel, draw a 3x3 box of color 1 around it,
    keeping the original pixel at center with its original color.
    """
    rows, cols = _rows(g), _cols(g)
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if g[r][c] != 0:
                # Draw 3x3 border of 1s
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            result[nr][nc] = 1
                # Place original pixel on top
                result[r][c] = g[r][c]
    return result


def _find_connected_components(g: Grid, fg_only: bool = True) -> list[list[tuple[int,int]]]:
    """BFS flood-fill connected components of non-zero cells."""
    rows, cols = len(g), len(g[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc]:
                continue
            if fg_only and g[sr][sc] == 0:
                continue
            # BFS
            component = []
            queue = [(sr, sc)]
            visited[sr][sc] = True
            while queue:
                r, c = queue.pop(0)
                component.append((r, c))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        if not fg_only or g[nr][nc] != 0:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            components.append(component)
    return components


def g_fill_rects_by_size(g: Grid) -> Grid:
    """
    Find hollow rectangles (frames) in the grid. Fill each rectangle's interior
    with a color based on interior pixel count: 1 if small (<= lower median), 2 if large.
    Keeps the frame color intact.
    """
    result = _clone(g)
    comps = _find_connected_components(g)
    if not comps:
        return result

    # Collect all interior sizes
    interior_sizes = []
    for comp in comps:
        rs = [r for r, c in comp]; cs = [c for r, c in comp]
        interior_h = max(rs) - min(rs) - 1
        interior_w = max(cs) - min(cs) - 1
        if interior_h > 0 and interior_w > 0:
            interior_sizes.append(interior_h * interior_w)

    if not interior_sizes:
        return result

    # Lower median: size <= median_size gets color 1, strictly > gets color 2
    sorted_sizes = sorted(interior_sizes)
    median_size = sorted_sizes[(len(sorted_sizes) - 1) // 2]

    for comp in comps:
        rs = [r for r, c in comp]; cs = [c for r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        interior_h = max_r - min_r - 1
        interior_w = max_c - min_c - 1
        if interior_h <= 0 or interior_w <= 0:
            continue
        fill_color = 1 if interior_h * interior_w <= median_size else 2
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                result[r][c] = fill_color

    return result


def g_color_interior_by_area(g: Grid) -> Grid:
    """
    Fill rectangle interiors based on interior side parity.
    Odd interior side length (1,3,5,...) → color 7.
    Even interior side length (2,4,6,...) → color 2.
    For non-square interiors, use the smaller dimension.
    Matches ARC tasks where interior color encodes square size parity.
    """
    result = _clone(g)
    comps = _find_connected_components(g)
    for comp in comps:
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        interior_h = max_r - min_r - 1
        interior_w = max_c - min_c - 1
        if interior_h <= 0 or interior_w <= 0:
            continue
        # Use the smaller side dimension to determine parity
        side = min(interior_h, interior_w)
        fill_color = 7 if side % 2 == 1 else 2
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                result[r][c] = fill_color
    return result


def g_fractal_self(g: Grid) -> Grid:
    """
    Self-similar Kronecker product: each cell (r,c) in g is replaced by 
    an entire copy of g if g[r][c] != 0. 
    Matches 'Kronecker' tasks like 8e2edd66.
    """
    R, C = len(g), len(g[0])
    # Protect against huge grids
    if R * R > 30 or C * C > 30:
        return _clone(g)
    
    out_rows, out_cols = R * R, C * C
    res = [[0] * out_cols for _ in range(out_rows)]
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0:
                # Paste the original pattern into this block
                for i in range(R):
                    for j in range(C):
                        res[r * R + i][c * C + j] = g[i][j]
    return res


def g_tile_self(g: Grid) -> Grid:
    """
    Tile the entire current grid (g) to fill a 30x30 workspace.
    """
    rows, cols = _rows(g), _cols(g)
    if rows == 0 or cols == 0: return g
    res = [[0] * 30 for _ in range(30)]
    for r in range(30):
        for c in range(30):
            res[r][c] = g[r % rows][c % cols]
    return res


_OBJECT_LEVEL_OPS = {
    "g_scale_by_color":      (g_scale_by_color,      "Scale each pixel to a block sized = color value"),
    "g_frame_each_pixel":    (g_frame_each_pixel,    "Draw 3x3 border of 1s around each isolated pixel"),
    "g_fill_rects_by_size":  (g_fill_rects_by_size,  "Fill rectangle interiors: 1=small, 2=large"),
    "g_color_interior_by_area": (g_color_interior_by_area, "Fill rect interiors: 7=smallest, 2=largest"),
    "g_fractal_self":        (g_fractal_self,        "Self-similar Kronecker nesting (e.g. 3x3 -> 9x9)"),
    "g_tile_self":           (g_tile_self,           "Tile grid to 30x30"),
}

for _name, (_fn, _desc) in _OBJECT_LEVEL_OPS.items():
    registry.register(_name, _fn, domain="arc", description=_desc, arity=1, overwrite=True)
