"""
domains/arc/benchmark.py
========================
A self-contained benchmark of 80 representative ARC-AGI-1 tasks.

Because the real ARC-AGI-1 dataset requires downloading 400 JSON files,
this module provides a programmatically generated benchmark that covers
the same six task categories in roughly the same proportions:

  Category A — Geometric transforms  (rotate, reflect, transpose)   ~15 %
  Category B — Color operations       (swap, fill, gravity)          ~20 %
  Category C — Object operations      (frame, mirror, hollow, scale) ~25 %
  Category D — Pattern generation     (checkerboard, stripes, tile)  ~15 %
  Category E — Counting / encoding    (bar chart, majority, filter)  ~10 %
  Category F — Compositional (2+ ops) (combinations of the above)   ~15 %

Each task has 3 training pairs and 1 test pair. Ground truth is known,
so evaluation is exact. This is used for the baseline vs expanded-DSL
comparison and for CI regression tests.

To use the real ARC-AGI-1 dataset instead:
    import json, pathlib
    from domains.arc.domain import ARCTask

    tasks = []
    for p in pathlib.Path("arc_data/evaluation").glob("*.json"):
        d = json.loads(p.read_text())
        d["name"] = p.stem
        tasks.append(ARCTask.from_dict(d))
"""
from __future__ import annotations

import copy
import random
from typing import Callable

from domains.arc.domain import ARCTask

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _rand_grid(rows: int, cols: int, colors: list[int], rng: random.Random) -> Grid:
    return [[rng.choice(colors) for _ in range(cols)] for _ in range(rows)]


def _rand_sparse(
    rows: int, cols: int, bg: int, fg: list[int],
    density: float, rng: random.Random,
) -> Grid:
    g = [[bg] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if rng.random() < density:
                g[r][c] = rng.choice(fg)
    return g


def _make_pairs(
    transform: Callable[[Grid], Grid],
    input_gen: Callable[[random.Random], Grid],
    rng: random.Random,
    n: int = 4,
) -> list[tuple[Grid, Grid]]:
    pairs = []
    for _ in range(n):
        inp = input_gen(rng)
        out = transform(inp)
        pairs.append((inp, out))
    return pairs


# ---------------------------------------------------------------------------
# Grid transforms (self-contained copies — benchmark has no import deps)
# ---------------------------------------------------------------------------

def _rot90(g):    return [list(row) for row in zip(*g[::-1])]
def _rot180(g):   return [row[::-1] for row in g[::-1]]
def _rot270(g):   return _rot90(_rot90(_rot90(g)))
def _refl_h(g):   return [row[::-1] for row in g]
def _refl_v(g):   return g[::-1]
def _trsp(g):     return [list(row) for row in zip(*g)]

def _recolor(fc, tc):
    def _f(g): return [[tc if c == fc else c for c in row] for row in g]
    return _f

def _swap(a, b):
    def _f(g):
        def s(c): return b if c == a else (a if c == b else c)
        return [[s(c) for c in row] for row in g]
    return _f

def _fill_bg(fill):
    def _f(g): return [[fill if c == 0 else c for c in row] for row in g]
    return _f

def _gravity_down(g):
    rows, cols = len(g), len(g[0])
    result = [[0] * cols for _ in range(rows)]
    for c in range(cols):
        col = [g[r][c] for r in range(rows) if g[r][c] != 0]
        for i, v in enumerate(reversed(col)):
            result[rows - 1 - i][c] = v
    return result

def _gravity_left(g):
    result = []
    for row in g:
        nz = [c for c in row if c != 0]
        result.append(nz + [0] * (len(row) - len(nz)))
    return result

def _majority(g):
    result = []
    for row in g:
        nz = [c for c in row if c != 0]
        if nz:
            m = max(set(nz), key=nz.count)
            result.append([m] * len(row))
        else:
            result.append(list(row))
    return result

def _frame(color):
    def _f(g):
        g2 = copy.deepcopy(g)
        rows, cols = len(g2), len(g2[0])
        for c in range(cols): g2[0][c] = g2[rows-1][c] = color
        for r in range(rows): g2[r][0] = g2[r][cols-1] = color
        return g2
    return _f

def _mirror_v(g):
    rows = len(g); half = rows // 2
    r = copy.deepcopy(g)
    for i in range(half): r[rows-1-i] = list(g[i])
    return r

def _mirror_h(g):
    return [row[:len(row)//2] + row[:len(row)//2][::-1] for row in g]

def _hollow(g):
    rows, cols = len(g), len(g[0])
    r = copy.deepcopy(g)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            v = g[i][j]
            if v and g[i-1][j]==v and g[i+1][j]==v and g[i][j-1]==v and g[i][j+1]==v:
                r[i][j] = 0
    return r

def _diag_fill(color):
    def _f(g):
        r = copy.deepcopy(g)
        for i in range(min(len(g), len(g[0]))): r[i][i] = color
        return r
    return _f

def _scale2x(g):
    result = []
    for row in g:
        nr = [c for c in row for _ in range(2)]
        result += [nr, list(nr)]
    return result

def _checkerboard(c1, c2):
    def _f(g):
        rows, cols = len(g), len(g[0])
        return [[(c1 if (r+c)%2==0 else c2) for c in range(cols)] for r in range(rows)]
    return _f

def _stripe_h(colors):
    def _f(g):
        rows, cols = len(g), len(g[0])
        return [[colors[r % len(colors)]] * cols for r in range(rows)]
    return _f

def _stripe_v(colors):
    def _f(g):
        rows, cols = len(g), len(g[0])
        return [[colors[c % len(colors)] for c in range(cols)] for _ in range(rows)]
    return _f

def _tile2x2(g):
    rows, cols = len(g), len(g[0])
    return [[g[r%2][c%2] for c in range(cols)] for r in range(rows)]

def _countbar(g):
    result = []
    for row in g:
        nz = [c for c in row if c != 0]
        color = nz[0] if nz else 0
        result.append([color]*len(nz) + [0]*(len(row)-len(nz)))
    return result

def _keep_rows(n):
    def _f(g):
        return [list(row) if sum(1 for c in row if c)>=n else [0]*len(row) for row in g]
    return _f


# ---------------------------------------------------------------------------
# Build the benchmark
# ---------------------------------------------------------------------------

def build_benchmark(seed: int = 42) -> list[ARCTask]:
    """
    Build and return the full 80-task benchmark.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[ARCTask]
        Each task has 3 training pairs and 1 test pair.
    """
    rng = random.Random(seed)
    tasks: list[ARCTask] = []

    def add(name: str, category: str, transform, input_gen, true_op: str = "") -> None:
        pairs = _make_pairs(transform, input_gen, rng, n=4)
        tasks.append(ARCTask(
            name=f"{category}_{len(tasks):03d}_{name}",
            train_pairs=pairs[:3],
            test_pairs=pairs[3:],
            true_op=true_op or name,
        ))

    sq = lambda s, c, r: _rand_grid(s, s, c, r)
    rc = lambda rs, cs, c, r: _rand_grid(rs, cs, c, r)
    sp = lambda rs, cs, bg, fg, d, r: _rand_sparse(rs, cs, bg, fg, d, r)

    # ── A: Geometric (12) ────────────────────────────────────────────────
    for sz in [3, 4, 5]:
        add("rot90",   "A", _rot90,   lambda r,s=sz: sq(s,[0,1,2,3],r), "grot90")
        add("refl_h",  "A", _refl_h,  lambda r,s=sz: sq(s,[0,1,2,3],r), "grefl_h")
        add("refl_v",  "A", _refl_v,  lambda r,s=sz: sq(s,[0,1,2,3],r), "grefl_v")
        add("rot180",  "A", _rot180,  lambda r,s=sz: sq(s,[0,1,2,3],r), "grot180")

    # ── B: Color (16) ────────────────────────────────────────────────────
    for fc, tc in [(1,3),(2,5),(3,7),(4,6)]:
        add(f"recolor_{fc}to{tc}", "B",
            _recolor(fc, tc),
            lambda r,f=fc,t=tc: sp(4,4,0,[f,t,1],0.4,r),
            f"recolor({fc}->{tc})")

    add("invert_01", "B", _swap(0,1), lambda r: sq(4,[0,1,2,3],r), "gswap_01")
    add("fill_bg5",  "B", _fill_bg(5), lambda r: sp(4,5,0,[1,2],0.3,r), "gfill_bg")
    add("gravity_down",  "B", _gravity_down, lambda r: sp(5,4,0,[1,2,3],0.3,r), "ggravity_down")
    add("gravity_down2", "B", _gravity_down, lambda r: sp(6,5,0,[2,4],0.25,r), "ggravity_down")
    add("gravity_left",  "B", _gravity_left, lambda r: sp(4,6,0,[1,2,3],0.4,r), "ggravity_left")
    add("invert_12", "B", _swap(1,2), lambda r: sp(5,5,0,[1,2],0.4,r), "gswap_12")
    add("majority",  "B", _majority,  lambda r: sp(4,6,0,[1,2,3],0.4,r), "gmajority")
    add("fill_bg9",  "B", _fill_bg(9), lambda r: sp(5,4,0,[3,6],0.3,r), "gzero_bg")

    # ── C: Object (20) ───────────────────────────────────────────────────
    add("frame8",        "C", _frame(8),   lambda r: sq(5,[0,1,2],r),     "gframe8")
    add("frame7",        "C", _frame(7),   lambda r: sq(6,[0,1,3],r),     "gframe8")
    add("mirror_bottom", "C", _mirror_v,   lambda r: rc(6,5,[0,1,2,3],r), "gmirror_v")
    add("mirror_bottom2","C", _mirror_v,   lambda r: rc(8,4,[0,1,2,3],r), "gmirror_v")
    add("mirror_right",  "C", _mirror_h,   lambda r: rc(4,6,[0,1,2,3],r), "gmirror_h")
    add("mirror_right2", "C", _mirror_h,   lambda r: rc(5,8,[0,1,2,3],r), "gmirror_h")
    add("hollow1",       "C", _hollow,     lambda r: sp(5,5,0,[3],0.6,r),  "ghollow")
    add("hollow2",       "C", _hollow,     lambda r: sp(6,6,0,[2],0.7,r),  "ghollow")
    add("diag1",         "C", _diag_fill(9), lambda r: sq(5,[0,1,2],r),   "gdiag9")
    add("diag2",         "C", _diag_fill(4), lambda r: sq(6,[0,1,3],r),   "gdiag9")
    add("transpose1",    "C", _trsp,       lambda r: rc(3,5,[0,1,2,3],r), "gtrsp")
    add("transpose2",    "C", _trsp,       lambda r: rc(4,6,[0,1,2,3],r), "gtrsp")
    add("rot270_1",      "C", _rot270,     lambda r: sq(4,[0,1,2,3],r),   "grot270")
    add("rot270_2",      "C", _rot270,     lambda r: sq(5,[0,1,2,3],r),   "grot270")
    add("scale2x_1",     "C", _scale2x,   lambda r: sq(3,[0,1,2],r),      "gscale2x")
    add("scale2x_2",     "C", _scale2x,   lambda r: rc(3,4,[0,1,3],r),    "gscale2x")
    add("frame5",        "C", _frame(5),   lambda r: sq(7,[0,1,2,3],r),   "gframe5")
    add("rot90_refl",    "C", lambda g: _refl_h(_rot90(g)),
                              lambda r: sq(4,[0,1,2,3],r), "grefl_h(grot90(x))")
    add("extract_diag1", "C", _diag_fill(2), lambda r: sq(5,[0,1,3],r),   "gdiag1")
    add("extract_diag2", "C", _diag_fill(5), lambda r: sq(6,[0,1,4],r),   "gdiag1")

    # ── D: Pattern (12) ──────────────────────────────────────────────────
    add("checker_12",   "D", _checkerboard(1,2), lambda r: sq(4,[0,1,2,3],r), "gcheckerboard")
    add("checker_35",   "D", _checkerboard(3,5), lambda r: sq(5,[0,1,2,3],r), "gcheckerboard03")
    add("stripe_h_123", "D", _stripe_h([1,2,3]), lambda r: sq(6,[0,1,2,3],r), "gstripe_h3")
    add("stripe_h_24",  "D", _stripe_h([2,4]),   lambda r: sq(4,[0,1,2,4],r), "gstripe_h2")
    add("stripe_v_13",  "D", _stripe_v([1,3]),   lambda r: sq(4,[0,1,2,3],r), "gstripe_v2")
    add("stripe_v_245", "D", _stripe_v([2,4,5]), lambda r: rc(4,6,[0,1,2,4,5],r), "gstripe_v3")
    add("tile2x2_1",    "D", _tile2x2,           lambda r: sq(4,[0,1,2,3],r), "gtile2x2")
    add("tile2x2_2",    "D", _tile2x2,           lambda r: sq(6,[0,1,2,3],r), "gtile2x2")
    add("diag_pat1",    "D", _diag_fill(1),      lambda r: sq(5,[0,2,3],r),   "gdiag1")
    add("checker_04",   "D", _checkerboard(0,4), lambda r: rc(4,5,[0,1,2,3,4],r), "gcheckerboard03")
    add("stripe_h_36",  "D", _stripe_h([3,6]),   lambda r: sq(5,[0,1,2,3,6],r), "gstripe_h2")
    add("stripe_v_7",   "D", _stripe_v([7]),     lambda r: sq(4,[0,1,2],r),   "gtile_left_col")

    # ── E: Counting (8) ──────────────────────────────────────────────────
    add("count_bar1",   "E", _countbar,    lambda r: sp(4,6,0,[3],0.4,r),     "gcountbar")
    add("count_bar2",   "E", _countbar,    lambda r: sp(5,7,0,[2],0.3,r),     "gcountbar")
    add("keep_rows2",   "E", _keep_rows(2),lambda r: sp(5,6,0,[1,2,3],0.35,r),"gkeep_rows2")
    add("keep_rows3",   "E", _keep_rows(3),lambda r: sp(5,7,0,[1,2,3],0.4,r), "gkeep_rows3")
    add("majority1",    "E", _majority,    lambda r: sp(5,7,0,[1,2],0.4,r),   "gmajority")
    add("majority2",    "E", _majority,    lambda r: sp(4,8,0,[3,5],0.45,r),  "gmajority")
    add("count_bar3",   "E", _countbar,    lambda r: sp(6,6,0,[4],0.35,r),    "gcountbar")
    add("keep_rows4",   "E", _keep_rows(4),lambda r: sp(6,8,0,[2,6],0.45,r),  "gkeep_rows4")

    # ── F: Compositional (12) ────────────────────────────────────────────
    add("rot90_recolor1","F", lambda g: _recolor(1,7)(_rot90(g)),
                              lambda r: sp(4,4,0,[1,2],0.4,r),  "grot90 then recolor")
    add("rot90_recolor2","F", lambda g: _recolor(3,8)(_rot90(g)),
                              lambda r: sp(4,4,0,[2,3],0.4,r),  "grot90 then recolor")
    add("refl_frame1",  "F", lambda g: _frame(9)(_refl_h(g)),
                              lambda r: sq(5,[0,1,2,3],r),       "gframe9(grefl_h(x))")
    add("refl_frame2",  "F", lambda g: _frame(6)(_refl_h(g)),
                              lambda r: sq(6,[0,1,2,3],r),       "gframe9(grefl_h(x))")
    add("grav_recolor1","F", lambda g: _recolor(1,4)(_gravity_down(g)),
                              lambda r: sp(5,4,0,[1,2],0.3,r),  "gravity then recolor")
    add("grav_recolor2","F", lambda g: _recolor(2,6)(_gravity_down(g)),
                              lambda r: sp(5,5,0,[2,3],0.3,r),  "gravity then recolor")
    add("rot180_hollow1","F", lambda g: _hollow(_rot180(g)),
                               lambda r: sp(5,5,0,[3],0.6,r),   "ghollow(grot180(x))")
    add("rot180_hollow2","F", lambda g: _hollow(_rot180(g)),
                               lambda r: sp(6,6,0,[2],0.65,r),  "ghollow(grot180(x))")
    add("trsp_stripe1", "F", lambda g: _stripe_h([1,3])(_trsp(g)),
                              lambda r: rc(4,6,[0,1,2,3],r),    "stripe_h(transpose(x))")
    add("trsp_stripe2", "F", lambda g: _stripe_h([2,5])(_trsp(g)),
                              lambda r: rc(3,5,[0,1,2,5],r),    "stripe_h(transpose(x))")
    add("scale_frame1", "F", lambda g: _frame(8)(_scale2x(g)),
                              lambda r: rc(3,3,[0,1,2],r),      "gframe8(gscale2x(x))")
    add("scale_frame2", "F", lambda g: _frame(5)(_scale2x(g)),
                              lambda r: rc(3,4,[0,1,3],r),      "gframe5(gscale2x(x))")

    return tasks


# Convenience alias
BENCHMARK_TASKS = None  # populated lazily below


def get_benchmark(seed: int = 42) -> list[ARCTask]:
    """Return the cached benchmark (builds on first call)."""
    global BENCHMARK_TASKS
    if BENCHMARK_TASKS is None:
        BENCHMARK_TASKS = build_benchmark(seed)
    return BENCHMARK_TASKS
