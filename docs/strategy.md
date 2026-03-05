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

Currently, strictly using the baseline geometry and color DSL (89 operations), the AGI solver natively cracks **~10%** of the true ARC-AGI dataset tasks with just a single primitive operation and basic sequencing. 
*The computational ceiling is not the Search Engine, but the expressiveness of the starting DSL.*

To improve accuracy on the real ARC-AGI benchmark, we proved that breaking out of linear transformation pipelines requires **Turing-Equivalent Conditionals**. By building a Zork text-adventure Domain mapped to the exact same `BeamSearch` evolutionary engine, we observed the engine spontaneously generate logical `if/then/else` branching (e.g. `z_if(z_is_locked(x), z_act_unlock(x), z_act_north(x))`). 

Porting this architecture back into ARC (`g_if(gkeep_hollow(x), gmap_fill(x), grot90(x))`) instantly unlocked non-linear algorithmic topologies. The `train_wake_sleep.py` pipeline must run exclusively on the training datasets with these conditional branches to invent the specific higher-level macro-primitives required to boost the baseline to >50%.

## 🎯 Next Steps & Future Work

### 1. Robust AST Serialization (✅ COMPLETED)
Previously we thought `PrimitiveLibrary.load()` was a stub, but the dynamic `Node.parse()` Python reconstruction natively builds back complete string hierachies like `g_repeat_2x2(g_if(x, 1, 0))` right from JSON arrays perfectly. Models scale persistently.

### 2. Generative Curriculum Learning (✅ COMPLETED)
Rather than randomly feeding chaotic 30x30 ARC noise to the solver during the Wake phase, the Wake-Sleep `tasks` are now intrinsically sorted by grid pixel complexity dimension. The engine isolates and learns core abstractions on 3x3 simple geometry first, completely cleaning the Generative Prior Markov Transition Matrix from poisoning.

### 3. Programmatic LLM Sub-Agents
While this MVP intentionally avoids leaning entirely on LLMs, integrating an LLM capable of generating Python natively could push this architecture to the absolute limit.
Instead of relying on a hardcoded Domain DSL (e.g. `gswap_01`, `grot90`), a "dreaming" LLM sub-agent could be prompted during the Sleep phase to write *new* raw Python primitives dynamically based on generalized puzzle logic.
If an LLM writes `def g_fill_bounds(grid):...` and it compiles successfully, the AGI engine could instantly absorb it into the `PrimitiveRegistry` as a base variable, bypassing the manual DSL engineering bottleneck!
