# Adding New Primitives

Every ARC transformation primitive is a **unary function** `Grid → Grid`.
Adding one takes about 5 lines and requires no changes to core/.

## Step 1: Define the function

Open `domains/arc/primitives.py` and add your function in the appropriate
category section:

```python
def gmy_op(g: Grid) -> Grid:
    """One-line description: what does it transform?"""
    # Implement your transformation
    result = _clone(g)
    # ... modify result ...
    return result
```

**Conventions:**
- Prefix with `g` (for "grid")
- Treat `0` as background unless your op is specifically about background
- Never mutate the input `g` — always `_clone(g)` first
- Handle edge cases (empty grid, 1×1 grid) gracefully
- Add a docstring; it appears in `registry.summary()` and docs

## Step 2: Register it

At the bottom of `domains/arc/primitives.py`, add to `_ARC_PRIMITIVES` dict
**or** call `registry.register()` directly:

```python
# Option A: add to the dict (preferred for batches)
_ARC_PRIMITIVES["gmy_op"] = (gmy_op, "One-line description")

# Option B: register directly (preferred for isolated additions)
registry.register("gmy_op", gmy_op, domain="arc",
                  description="One-line description")
```

That's it. The beam search will include `gmy_op` in its vocabulary the
next time you call `registry.names(domain="arc")`.

## Step 3: Write a test

Open `tests/test_arc.py` and add a test case for your primitive:

```python
class TestMyNewPrimitive(unittest.TestCase):

    def test_gmy_op_basic(self):
        g = [[1, 2], [3, 4]]
        expected = ...  # what gmy_op(g) should produce
        self.assertEqual(gmy_op(g), expected)

    def test_gmy_op_identity_case(self):
        """If there's a natural identity (e.g., applying twice = identity)."""
        g = [[1, 2], [3, 4]]
        self.assertEqual(gmy_op(gmy_op(g)), g)

    def test_gmy_op_accepts_3x3(self):
        """Always test the most common ARC grid size."""
        g = [[i * 3 + j for j in range(3)] for i in range(3)]
        result = gmy_op(g)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
```

Run tests: `python -m unittest tests.test_arc -v`

## ARC color conventions

ARC uses integers 0–9:

| Color | Convention |
|-------|-----------|
| 0 | Background (conventionally black) |
| 1 | Blue |
| 2 | Red |
| 3 | Green |
| 4 | Yellow |
| 5 | Grey |
| 6 | Pink/Magenta |
| 7 | Orange |
| 8 | Azure |
| 9 | Maroon |

Most primitives should treat 0 as background. Document any deviation clearly.

## Primitive naming conventions

| Prefix | Category | Examples |
|--------|----------|---------|
| `grot` | Rotation | `grot90`, `grot180`, `grot270` |
| `grefl` | Reflection | `grefl_h`, `grefl_v` |
| `gswap` | Color swap | `gswap_01`, `gswap_23` |
| `ggravity` | Gravity | `ggravity_down`, `ggravity_left` |
| `gframe` | Border | `gframe1` through `gframe9` |
| `gmirror` | Mirror | `gmirror_h`, `gmirror_v` |
| `gstripe` | Stripe pattern | `gstripe_h2`, `gstripe_v3` |
| `gcountbar` | Count encoding | `gcountbar`, `gcountbar_cols` |

## Tips for good primitives

1. **One operation per primitive.** `grot90_then_frame` is not a primitive —
   the search will compose `grot90` and `gframe8` itself.

2. **Think in categories.** Are you adding a geometric op? A color op?
   A structural op? Group it with its category.

3. **Check for redundancy.** `grefl_h(grot90(g))` == `gtrsp(g)`, so
   you don't need a separate `gtranspose_h` primitive.

4. **Edge cases first.** What happens with a 1×1 grid? An empty grid?
   Add a `_safe_grid_op` wrapper if you're unsure:

   ```python
   gmy_op = _safe_grid_op(lambda g: ...)
   ```

5. **Correctness > performance.** These run millions of times per benchmark.
   Keep the hot path simple. Avoid deep copies unless necessary.
