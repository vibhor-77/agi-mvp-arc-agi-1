# Theoretical Foundations

## Core Claim

Intelligence is compression. A system is intelligent to the degree that it can find
compact descriptions of its observations — discovering patterns that generalise.

This framework operationalises that claim: find the shortest program that transforms
every input into the correct output.

---

## Minimum Description Length (MDL)

Given data D and a model (program) M, the MDL principle selects the M that minimises:

    L(M) + L(D | M)

where L denotes description length in bits.

In our implementation:

    fitness(tree, data) = mean_error(tree, data) + λ · tree.size()

- `mean_error` estimates L(D | M): how well the model explains the data
- `tree.size()` estimates L(M): the complexity of the model itself
- `λ` trades them off (typical value: 0.02)

This is equivalent to regularised empirical risk minimisation, but with an
explicit Occam's razor interpretation: simpler programs are preferred.

---

## Kolmogorov Complexity

The ideal (but uncomputable) measure is Kolmogorov complexity K(x):
the length of the shortest program that outputs x.

MDL is the computable approximation: instead of searching over all programs,
we search over expression trees built from a fixed primitive vocabulary.

As the vocabulary grows (library learning), the effective Kolmogorov
complexity of solvable tasks decreases — the same information can be expressed
in fewer nodes.

---

## Expression Trees

Every candidate solution is a tree where:
- Leaves are input variables (the grid, state vector, ...)
- Internal nodes are primitive functions from the registered vocabulary
- All primitives are unary: `f(child) → value`

Example for ARC: `gframe8(gscale2x(x))` means "scale the grid 2× then add a border of color 8".

This representation is:
- **Interpretable:** You can read what the rule is
- **Composable:** Complex rules emerge from simple parts
- **Searchable:** Mutations preserve syntactic validity

---

## Beam Search

At each generation:

1. Every beam member spawns `offspring` children via mutation or crossover
2. All candidates are scored by `fitness(tree)`
3. The best `beam_size` unique trees (by string) survive
4. Repeat until convergence or `generations` exceeded

Mutations include:
- **Subtree replacement:** Replace a random subtree with a new random one
- **Op wrap:** Insert a new op at the root
- **Constant tweak:** Slightly perturb numeric constants
- **Crossover:** Splice a subtree from one parent into another

The search is stochastic and benefits from larger `beam_size` for harder problems.

---

## Library Learning (DreamCoder connection)

After solving a corpus of tasks, frequent subtrees can be extracted and promoted
to first-class primitives. This is the core idea behind DreamCoder (Ellis et al., 2021).

In the compression framework:
- If subtree T appears in K solutions, defining T as a primitive saves K · (size(T)-1) nodes
- The compression ratio (total nodes / tasks solved) decreases over rounds
- This decrease is the empirical signature of the system "learning to learn"

Current implementation: manual vocabulary expansion.
Future: automated library learning via frequent-subtree mining.

---

## ARC-AGI Performance Analysis

### Why 76% and not 100%?

The remaining 24% of benchmark tasks require operations not yet in the primitive vocabulary:

1. **Parametric recoloring** (map color X → Y based on spatial context)
   - Requires the system to infer a mapping from the examples, not just a fixed swap
2. **Object segmentation** (find connected components, count distinct objects)
   - Requires a graph traversal primitive, not in current DSL
3. **Conditional logic** (if object size > threshold then ...)
   - Requires a predicate + branch primitive
4. **Relative spatial relationships** (above, adjacent, inside)
   - Requires a primitive that reasons about object positions

### Realistic extrapolation to ARC-AGI-1 public eval

The 400-task ARC-AGI-1 eval has the same category distribution as our benchmark.
Expected performance with the current 89-op DSL:

- Geometric tasks: ~90–100% (our DSL has all geometric ops)
- Color tasks: ~50–70% (parametric recoloring is the main gap)
- Object tasks: ~40–60% (segmentation is missing)
- Pattern tasks: ~40–60% (stripe parameters vary)
- Counting tasks: ~60–80% (counting without object boundaries)
- Compositional: ~30–50% (compounds of gaps compound)

**Overall estimate: 20–35% on the real ARC-AGI-1 eval.**

This places us in the range of purpose-built symbolic systems:
- Vector Symbolic Algebra: 10.8% train, 3% eval (2023)
- DreamCoder-style systems: 15–25%
- LLM-guided DSL + test-time training: 40–55% (2024 ARC Prize winners)

---

## References

- Chollet, F. (2019). *On the Measure of Intelligence.* arXiv:1911.01547
- Ellis, K. et al. (2021). *DreamCoder: Bootstrapping Inductive Program Synthesis.* PLDI 2021.
- Rissanen, J. (1978). *Modeling by Shortest Data Description.* Automatica.
- Schmidt, M. & Lipson, H. (2009). *Distilling free-form natural laws from experimental data.* Science.
- Grünwald, P. (2007). *The Minimum Description Length Principle.* MIT Press.
