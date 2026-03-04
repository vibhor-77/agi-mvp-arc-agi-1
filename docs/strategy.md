# Strategy Roadmap & True AGI Vision 🌟

The ultimate goal of this repository is to create a robust, generalized intelligence pipeline capable of learning abstractions and applying them procedurally across wildly different domains without human-engineered hardcoded weights or deep neural network black-box parameters.

## The Overarching Strategy

To conquer ARC-AGI, the solver depends on **Library Learning** (LL) extracting knowledge during a "Sleep" state from tasks dynamically solved via **Symbolic Regression** (SR) during a "Wake" state.

The strategy thus far has been to expand the underlying exploration capacity of `BeamSearch` so that the generative tree can mathematically express as many distinct transformation sequences as possible within a computationally practical timeframe.

By combining Semantic Hashing (pruning redundancies), Lexicase Selection (preserving niche edge-cases), and DreamCoder Generative Priors (focusing on probabilistically advantageous shapes), the system is learning how to *shortcut* simple problems and spend its heavy compute solving harder challenges.

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
