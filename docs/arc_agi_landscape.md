# ARC-AGI Landscape — State of the Art (March 2026)

*Research context for understanding where this project sits relative to the field.*

---

## What is ARC-AGI?

Created by François Chollet in 2019, ARC (Abstraction and Reasoning Corpus)
is a benchmark designed to measure **fluid intelligence** — the ability to
generalise from limited examples to novel, never-before-seen tasks.

Each task provides 3–5 input/output grid pairs as demonstrations, plus 1–3
test inputs. Systems must infer the underlying rule and apply it correctly
to the test inputs. Tasks are solvable by any adult human with no special
training; they require visual reasoning, not memorisation.

**Key design constraint:** you cannot memorise your way to a high score.
Every eval task is novel. A system that achieves 85% must demonstrate
genuine generalisation.

---

## Timeline

| Year | Event | Score |
|------|-------|-------|
| 2019 | Chollet introduces ARC in "On the Measure of Intelligence" | — |
| 2020 | First Kaggle ARC Challenge. Winner: "ice cuber" with handcrafted DSL | 21% |
| 2022–2023 | Various DSL + program synthesis approaches | 10–25% |
| 2024 | ARC Prize 2024 launched. OpenAI o3-preview demonstrated | ~53% private |
| 2025 | ARC Prize 2025. NVARC wins with LLM + test-time training | ~55% |
| 2025 | SOAR: self-improving evolutionary program synthesis | ~52% |
| 2025 | ARC-AGI-2 released (harder version) | ~24% best |
| 2026 | This project (Symbolic Regression + Library Learning | Wake-Sleep scaling |
---

## Approaches that work

### 1. Handcrafted DSL + program synthesis (2020–2023)
The first successful approach: define a domain-specific language of grid
operations, then enumerate or search for programs. This is closest to what
we're doing. Ceiling: ~25% without refinement loops.

### 2. LLM-guided code generation (2023–2024)
Use an LLM (GPT-4, Claude, etc.) to generate Python code that transforms
grids. Filter by test execution. Fine-tune on successful solutions. Ceiling:
~35–40% without test-time adaptation.

### 3. Test-time training (2024–2025)
Train model weights specifically on each task's demonstration pairs before
predicting the test output. This is the top-performing approach. Systems:
ARChitects, NVARC, Tiny Recursive Model. Ceiling: ~55% on ARC-AGI-1.

### 4. Evolutionary program synthesis + LLM (2025)
SOAR: fine-tune an LLM on its own search traces. The model learns to write
better programs by observing what worked. ~52% on ARC-AGI-1.

### 5. Neural cellular automata (2025)
Self-organizing models trained per-task using gradient descent. Competitive
efficiency, interpretable, but moderate solve rates.

### 6. Vector Symbolic Algebra (2025)
Combines perceptual heuristics (System 1) with symbolic program inference
(System 2). Interpretable but limited: 10.8% training, 3.0% eval.

---

## Why this project is positioned where it is

Our approach is closest to **handcrafted DSL + program synthesis**, but with:
1. A principled MDL fitness instead of ad hoc scoring
2. A clean extensible primitive registry (89 ops vs. typical 20–40)
3. Morphological operations (dilate, erode) not in most DSLs
4. Library learning capability (automatic primitive promotion)

The primary ceiling is the same as all DSL approaches: tasks requiring
*contextual reasoning* (what does this symbol mean relative to others?)
can't be expressed as fixed unary operations.

The path to higher scores requires either:
1. **More expressive primitives** (segmentation, parametric recoloring)
2. **Refinement loops** (feedback from partial solutions)
3. **LLM-guided primitive generation** (dynamic vocabulary expansion)

All three are natural extensions of this framework — unlike test-time
training, which requires a fundamentally different architecture.

---

## ARC-AGI-2 (released 2025)

ARC-AGI-2 is designed to resist the approaches that cracked ARC-AGI-1.
Key differences:

- **Semantic symbol interpretation**: symbols must be understood in context,
  not just by visual pattern
- **Multi-rule interaction**: tasks require applying multiple rules simultaneously
- **Contextual selection**: rules applied differently based on context

Current best score: ~24% (NVARC, 2025). Pure symbolic approaches achieve
single-digit percentages.

Our starting 89-op DSL solves a foundational baseline. However, by deploying Wake-Sleep **Library Learning**, the engine dynamically expands its vocabulary to cover missing concepts. The gap reveals exactly which foundational primitives must be provided (e.g., connected-component segmentation) before the LL engine can compound them into higher-order contextual logic.

---

## What "solving" ARC-AGI would prove

Chollet's definition: a system that achieves 85% on ARC-AGI demonstrates
**binary fluid intelligence** — genuine generalisation from examples, not
memorisation.

No current system is close. The gap between 55% and 85% represents the
difference between "can apply known techniques to novel instances" and
"can infer fundamentally new abstractions from scratch."

This project's bet: the gap can be closed by making the primitive vocabulary
large enough and adding refinement loops. The MDL criterion is the right
objective function. The search is the right mechanism. The missing piece is
vocabulary breadth and contextual reasoning.

---

## Useful resources

- [ARC Prize](https://arcprize.org) — official competition and leaderboard
- [ARC-AGI dataset](https://github.com/fchollet/ARC-AGI) — 400 eval tasks
- [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) — harder version
- [Chollet 2019 paper](https://arxiv.org/abs/1911.01547) — "On the Measure of Intelligence"
- [SOAR paper](https://arxiv.org/abs/2507.07456) — self-improving evolutionary synthesis
- [DreamCoder paper](https://arxiv.org/abs/2006.08381) — library learning
- [ARC-DSL](https://github.com/michaelrosenbaum/arc-dsl) — reference DSL
