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

### Performance Bottlenecks and Primitive Expansion

While the baseline primitive vocabulary excels at core transformations, solving real ARC-AGI-1 tasks natively requires the Wake-Sleep cycle (`train_wake_sleep.py`) to discover and promote operations the core vocabulary lacks:

1. **Parametric recoloring** (map color X → Y based on spatial context)
   - Requires generating generic variables mapping unknown source/target colors.
2. **Object segmentation** (find connected components, count distinct objects)
   - Addressed initially by `g_extract_objects`, though diagonal connectivity is still lacking.
3. **Conditional logic** (if object size > threshold then ...)
   - Requires the AST to dynamically compose ternary `if-else` blocks for context execution.
4. **Relative spatial relationships** (above, adjacent, inside)
   - Requires primitives that iterate over bounding boxes and reason mathematically.

### Extrapolating to the Real ARC-AGI-1 Public Eval

Our baseline architectural run achieved ~3-5% solve success when fully zero-shot on the real training set using only the atomic operations. However, through aggressive **Library Learning**, we systematically bypass the single-primitive limitation, dynamically adding macro-primitives that encode multi-step domain knowledge into simple `Node` objects.

By fusing **Semantic Hashing**, **Lexicase Selection**, and a deep search budget (`beam=10`, `generations=100`), the objective is to push the solver natively past the 20-30% mark, demonstrating true out-of-sample composition before engaging any LLM agents.

This goal isolates pure algorithmic machine learning against other structural engines in the field:
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
