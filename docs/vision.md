# The Vision — AGI via MDL-Guided Primitive Discovery

*Vibhor Jain's original research vision, as articulated across multiple sessions,
with Claude's analysis and connecting literature.*

---

## The Core Claim

> "Even text, audio, images, video and robotics can be framed mathematically as a
> composition of primitive functions which we then smartly compound using the
> explore-exploit tradeoff. And that this is similar to how human knowledge and
> superpower compounds over centuries and millennia."
>
> — Vibhor Jain, March 2026

This is a strong, falsifiable claim with deep theoretical grounding. It connects to:

- **Kolmogorov complexity**: the true measure of information is the length of the
  shortest program that produces it
- **MDL (Minimum Description Length)**: the best model minimises
  `L(model) + L(data | model)` — i.e. prefer compact descriptions that fit the data
- **DreamCoder** (Ellis et al., 2021): bootstraps a primitive library by solving
  tasks and then distilling recurring subtrees into new named primitives
- **Program synthesis**: the field of automatically writing programs from
  input/output examples — what this system does

---

## The Four Pillars

Vibhor's framework identifies four properties of general intelligence:

### 1. Feedback loops
The system must learn from its own outputs. In this framework:
- The fitness function is the feedback signal
- Library learning closes the loop: solutions become new primitives

### 2. Approximability
The system must trade off accuracy against complexity:
- MDL fitness: `error + λ·size`
- λ controls the accuracy/simplicity tradeoff
- This is Occam's Razor operationalised

### 3. Abstraction and composability
The system must build complex solutions from simple parts:
- Expression trees are the composable structure
- New primitives abstract over recurring patterns
- Composition depth can be arbitrary

### 4. Exploration
The system must search beyond known solutions:
- Beam search with mutation = exploration strategy
- Library learning = exploitation of discovered structure
- UCB operator selection (planned) = intelligent exploration bias

---

## Why This Approach is Distinctive

Most current AI is strong on approximability (deep learning approximates any
function) and feedback (gradient descent is an excellent feedback mechanism),
but **weak on abstraction and exploration**:

- Neural networks don't produce interpretable abstractions
- Gradient descent is local and doesn't explore the full hypothesis space
- There's no mechanism for discovered knowledge to compound

This system inverts those weaknesses:
- Every solution is an interpretable symbolic expression
- Beam search explores globally (no gradient required)
- Library learning compounds discovered primitives over time

---

## The Compounding Hypothesis

The key insight is that intelligence compounds through primitive accumulation.
Consider how mathematics works:

- Round 1: discover counting (natural numbers)
- Round 2: discover addition (composes counting)
- Round 3: discover multiplication (composes addition)
- Round N: discover calculus, quantum mechanics, ...

Each round's primitives are the building blocks for the next. The same
mechanism, applied to grid transformations (or natural language, or robotics),
should produce increasingly powerful solutions from increasingly compact
descriptions.

**Measurable prediction:** After K rounds of library learning, the compression
ratio (total nodes in solutions / tasks solved) should decrease monotonically.
This is what was observed in Session 2: ratio dropped from 8.8 → 5.5 (37%)
in just two rounds on 10 functions.

---

## Connection to Chollet's "On the Measure of Intelligence"

François Chollet (ARC-AGI creator) defines intelligence as:

> "Skill-acquisition efficiency: the rate at which a system can acquire new
> skills, relative to prior experience and the amount of information available."

This is almost exactly Vibhor's four-pillar framework:
- Skill acquisition = search + fitness (pillars 1 + 2)
- Efficiency = MDL preference for compact solutions (pillar 3)
- Prior experience compounding = library learning (pillar 4)

The ARC benchmark was specifically designed to measure this property.
It's no coincidence that it's the natural target for this framework.

---

## What Succeeds and What Doesn't (Honest Assessment)

### Where the framework clearly works

| Problem | Why it works |
|---------|-------------|
| Symbolic regression | Single variable, continuous primitives, clean MDL fitness |
| Grid geometric transforms | Discrete, composable, exhaustive vocabulary possible |
| CartPole policy | State vector inputs, simple physics, reward = fitness |
| Any "find the rule" problem | MDL fitness is exactly right for this class |

### Where the framework currently struggles

| Problem | Why it struggles | Fix needed |
|---------|-----------------|-----------|
| ARC tasks with object reasoning | Segmentation not a primitive | Add `gsegment` |
| ARC tasks with contextual recoloring | Parametric ops needed | Add `grecolor_*` |
| Natural language (Zork) | Primitive vocabulary undefined | NL parsing layer |
| Long-horizon sequential tasks | No memory primitive | Stateful nodes |
| Real-time robotics | Continuous control, latency | C extension, GPU eval |

### The honest scorecard vs. ARC-AGI state of the art

| System | ARC-AGI-1 % | Approach |
|--------|-------------|---------|
| This system (projected) | 20–35% | MDL beam search, 89 ops |
| Vector Symbolic Algebra | 3% eval | Purpose-built ARC DSL |
| DreamCoder-style DSL | 15–25% | Library learning |
| SOAR (LLM + evolution) | ~52% | Self-improving LLM loop |
| ARChitects (best 2025) | ~55% | LLM + test-time training |
| Human baseline | ~84% | — |

The gap from 35% to 55% is closed by refinement loops and LLM-guided primitive
generation. Both are natural extensions of this framework.

---

## The Longer Vision

If the compounding hypothesis is correct, then:

1. **ARC-AGI-1** → demonstrate that pure symbolic MDL can match LLM-based
   approaches with far less compute and full interpretability
2. **ARC-AGI-2** → harder tasks reveal which primitives are still missing;
   library learning fills the gaps
3. **Zork** → sequential symbolic RL; demonstrate that the same framework
   generalises to language and long-horizon tasks
4. **Robotics** → primitive vocabulary over sensor readings; discover compact
   control laws
5. **General intelligence** → a system that can acquire the vocabulary for any
   new domain from examples, guided by MDL

This is an ambitious but not absurd programme. The theoretical case is strong.
The empirical case is being built here, one domain at a time.
