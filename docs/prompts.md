# AI Prompt Logbook & Strategy Commentary 🤖

This repository was heavily scaffolded, brainstormed, and implemented alongside an agentic coding assistant ("Antigravity" / Deepmind AI). 

To ensure maximum transparency into the development process, below is a logbook of the fundamental prompts and strategy pivots that shaped this AGI MVP, along with my commentary.

---

## 1. The Genesis Prompt
**User Prompt:**
> "I want to explore AGI Minimal Viable Product Ideas, based on the **4 Pillars of True General Learning**. Our main focus for an MVP would be on the *Abstraction and Composability* pillar. Can you please write an implementation plan to build an AI that can compose basic functions?"

**AI Commentary & Reflection:**
This was the foundational spark. Instead of jumping straight to LLMs, the user wanted to build a native logic framework. We brainstormed an Abstract Syntax Tree (AST) that could evaluate basic integer series, laying the groundwork for the `Node` and `BeamSearch` engines. It explicitly defined the separation of the *Domain* (game physics, math, ARC) from the *Search Engine* (the AGI core).

## 2. Pivot to Zork & ARC
**User Prompt:**
> "Let's incrementally build towards AGI. Two goals: (1) A Zork text-adventure MVP, (2) An ARC AGI MVP. Let's do ARC first. Expand the implementation plan."

**AI Commentary & Reflection:**
The original math domain was too simple to prove "Generality." Moving to ARC (Abstraction and Reasoning Corpus) forced the engine to prove it could compute 2D geometric translations. We designed the `ARCDomain` and populated `primitives.py` with hardcore 2D geometric operations (`grot90`, `gswap_12`).

## 3. Scaling the MVP Heuristics
**User Prompt:**
> "Refine the MVP heuristics. Scale up the Abstraction and Exploration pillars without relying on external LLM dependencies."

**AI Commentary & Reflection:**
At this stage, `BeamSearch` was too slow to explore the 100+ generations needed to find complex multi-step ARC solutions. We parallelized the task runners using Python's `multiprocessing.Pool` (`--task-workers 8`) which completely saturated M1 Max architectures, drastically speeding up the Wake cycle.

## 4. The Library Learning Breakthrough
**User Prompt:**
> "I want a multi-round Wake-Sleep training loop. Add a persistent Library where the AI can compress frequent abstraction patterns and reuse them as new primitives."

**AI Commentary & Reflection:**
This was the most complicated structural leap. 
We introduced `core/library.py` (`PrimitiveLibrary`). During the Sleep phase, the AI walks every solved AST, counts sub-tree frequencies, and dynamically injects new composite functions (like `lib_op_1 = grot90(ginv(x))`) back into the `Registry`. This meant the DSL actually *grew* autonomously based on its own successes.

## 5. The Paradigm Fusion (SR + LL)
**User Prompt:**
> *(Paraphrased)* "My proposal is to discover/learn the right approach from the training data: evolve/mutate/crossover new primitives, and build higher-level library structures. Note that this is a mix of Symbolic Regression and Library Learning. Come up with smarter techniques as a result of knowing this."

**AI Commentary & Reflection:**
This brilliant strategic prompt changed the entire paradigm of the repository. 
Realizing that we were physically fusing Symbolic Regression with Library Learning, I engineered three specific mathematical algorithms from those distinct fields into the `BeamSearch` engine:
1. **Semantic Hashing:** Pruning mathematically invisible redundant ASTs.
2. **Lexicase Selection:** Preserving fragile edge-case logic via local error Pareto tracking.
3. **Generative Priors (DreamCoder):** Probabilistic AST generation using $P(child \mid parent)$ Markov dynamics.

This final fusion yielded the robust, intelligent solver documented in the `execution_guide.md` today!
