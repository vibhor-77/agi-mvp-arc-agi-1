# Strategy Roadmap & True AGI Vision 🌟

The ultimate goal of this repository is to create a robust, generalized intelligence pipeline capable of learning abstractions and applying them procedurally across wildly different domains without human-engineered hardcoded weights or deep neural network black-box parameters.

## The Overarching Strategy: Train vs. Eval

The strategy for achieving AGI on multi-domain environments (like ARC or Zork) directly mirrors the architecture from [agi-mvp-no-noise](https://github.com/vibhor-77/agi-mvp-no-noise), where the system solved for mathematical curves `y = f(x)` using simple functional composing blocks like `sin`, `cos`, and `square`. However, the transformations here are not purely mathematical—they manipulate objects, colors, and abstract concepts.

Every ARC-AGI task contains a **Training Set** (where the input and output are known) and an **Evaluation Set** (where only the input is known).

### 1. The Training Phase (Building the Arsenal)
During training, the AGI engine examines the known `(input, output)` pairs.
We start with a baseline set of "common sense" atomic primitives (e.g., `rotate`, `flip`, `fill_color`). 
Using **Symbolic Regression (SR)** and **Library Learning (LL)** techniques, the system searches to find the sequence of primitives that maps the input grid to the output grid.
As it solves training examples, it dynamically:
* Discovers common individual operations.
* Identifies groups of operations used frequently.
* Creates new, single-node composite operations (Library Primitives) that shortcut complex logic.
* Learns **shape, pattern, and object identification** primitives to isolate blocks on which operations will be performed.

### 2. The Evaluation Phase (Solving the Unknown)
Once the training phase has successfully expanded the primitive library (the DSL) with these optimized, domain-specific higher-level functions, the engine tackles the Evaluation Set.
It repeats the Symbolic Regression search, combining the newly invented primitives. Governed by Minimum Description Length (MDL) and Occam's Razor, the engine finds the *simplest possible transformation* that explains the training examples, and formally applies that transformation to the unknown evaluation input to solve it.

## Expected Performance & Current Limits

Currently, strictly using the baseline geometry and color DSL (89 operations), the AGI solver natively cracks **~3%** of the true ARC-AGI dataset tasks with just a single primitive operation. 
*The computational ceiling is not the Search Engine, but the expressiveness of the starting DSL.*

To improve accuracy on the real ARC-AGI benchmark, the `train_wake_sleep.py` pipeline must run exclusively on the training datasets to invent the specific higher-level primitives (like connected-object segmentation or color-conditional logic) required to boost the 3% baseline to >50%.

## 🎯 Next Steps & Future Work

### 1. Robust AST Serialization
Currently, the `PrimitiveLibrary` can save its abstractions to `library.json`, but `PrimitiveLibrary.load()` is a sub-optimal stub.
To truly scale the Wake-Sleep cycle across persistent deployments, we need a massive, dynamic AST string parser capable of perfectly rebuilding `Node` hierarchies from raw text strings natively stored on disk.
Once implemented, the AGI can train for days, save its semantic library, and resume instantly from where it left off.

### 2. Generative Curriculum Learning
Rather than feeding random ARC tasks to the solver during the Wake phase, the framework needs a "Curriculum" module.
This module should conceptually isolate and present the simplest geometric tasks first (e.g., solid color fills, single-object gravity), expanding up to relational abstractions (e.g., hollow out the red squares but gravity-shift the blue ones).
By building a Curriculum pipeline, the Generative Prior matrix won't be poisoned by chaotic high-noise abstractions from complex tasks the model physically couldn't solve yet.

### 3. Programmatic LLM Sub-Agents
While this MVP intentionally avoids leaning entirely on LLMs, integrating an LLM capable of generating Python natively could push this architecture to the absolute limit.
Instead of relying on a hardcoded Domain DSL (e.g. `gswap_01`, `grot90`), a "dreaming" LLM sub-agent could be prompted during the Sleep phase to write *new* raw Python primitives dynamically based on generalized puzzle logic.
If an LLM writes `def g_fill_bounds(grid):...` and it compiles successfully, the AGI engine could instantly absorb it into the `PrimitiveRegistry` as a base variable, bypassing the manual DSL engineering bottleneck!
