# Architecture & Testing Philosophy 🏗️

The goal of this repository is to build a scalable Minimum Viable Product (MVP) of an Artificial General Intelligence (AGI) solver, leaning exclusively on explicit algorithmic structures rather than opaque Large Language Model (LLM) weights.

Because the ARC-AGI dataset is essentially a benchmark of *program synthesis*, this MVP relies on advanced functional composition.

## 1. The N-ary AST Engine

For the system to evaluate hundreds of distinct logical operations across varying grid sizes, the underlying mathematical architecture must be completely domain-agnostic. 

I implemented a universal **Abstract Syntax Tree (AST)** `Node` class in `core/tree.py`.
Originally, this solver utilized entirely unary nodes (one parent, one child), but I've successfully refactored the engine to be infinitely recursive **N-ary**. Trees can now support multiple geometric grid inputs for a single operator (e.g. `goverlay(grid_A, grid_B)`).

This structure treats trees as immutable values during `BeamSearch`, and supports dynamic arities, `random_tree` generation, and complex `mutate` logic (sub-tree replacement, structural wrapping).

## 2. Testing Constraints & Philosophy

Every single primitive introduced to the AGI framework must be flawlessly predictable. 

*   `tests/test_arc.py` evaluates every geometric mapping function to ensure perfectly clean boundaries and deterministic responses.
*   The tests verify that elements like Gravity shifting, Inversion limits, Matrix Rotation, and Color swapping don't artificially duplicate data or break under strict unit tests.
*   This also tests that every single dynamic `Node` correctly maps to the physical grid output expected by the solver (Grid Cell Accuracy rating = 1.0).

## 3. Paradigm Upgrades: (SR + LL)

The AGI reasoning core is a fusion of two powerful paradigms: **Symbolic Regression (SR)** and **Library Learning (LL)**.

To maximize performance within the N-ary AST tree, I engineered three monumental algorithmic upgrades into the `BeamSearch` logic:

### A. Semantic Hashing (Anti-Aliasing)
The search space of functional programming is infinitely redundant (e.g., `grot90(grot90(x))` vs `grot180(x)`). 
Using **Semantic Hashing**, the system hashes the output of every generated AST against the training data. If a new candidate output perfectly matches the output of a smaller, pre-existing tree, the new candidate is instantly culled. This completely eliminates redundant computation and expands the effective capacity of the `BeamSearch`.

### B. Lexicase Selection
Usually, AI models judge fitness based on global average accuracy. 
By wiring **Lexicase Selection** into the deduping sorting arrays, the system now tracks individual case errors. If an AST mathematically solves even *one* singular edge case perfectly, it receives an artificial priority boost, preventing the engine from discarding uniquely valuable sub-abstractions!

Eventually, this model will hit a hard logical ceiling (around 10-15% accuracy on ARC) because `BeamSearch` alone cannot efficiently discover Deep Logic (like `while` loops, or complex `if/else` pixel mapping) before combinatorial explosion kills the search queue. 

To break past this ceiling, the engine requires actual Object-Oriented segmentation and LLM-synthesized Python primitives (see `next_steps.md`).

## 4. The Deep Learning Paradigm: Weight Persistence (`--model`)

Rather than starting from scratch every time, the AGI engine treats its learned Library exactly like a Neural Network treats its Weight parameters. The engine accepts a `--model <filepath.json>` argument, which natively stores two things:

1. **The Learned Abstractions (Macro-ASTs):** The literal mathematical subtrees (e.g., `lib_op_12 = grot90(goverlay(x, y))`) that were proven to solve tasks in previous Epochs.
2. **The Generative Priors:** The Markov transition matrices specifying $P(\text{child} \mid \text{parent})$, ensuring the Search Engine generates culturally-aware mutations on unseen tasks rather than blind noise.
