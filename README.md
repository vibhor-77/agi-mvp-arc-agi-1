# Universal ARC-AGI Solver 🧠🧩

An exploratory general-purpose Artificial General Intelligence (AGI) solver framework structured around the *Four Pillars of True General Learning*:
1. Feedback Loops
2. Abstraction & Composability
3. Exploration
4. Approximability

This project fuses the paradigms of **Symbolic Regression (SR)** and **Library Learning (LL)** to conquer the complex multi-domain reasoning benchmark: ARC-AGI.

---

## 📖 Documentation Directory

To keep the repository clean and easy to read, the documentation is divided into the following categories:

### 1. Core Guides
1. **[Quickstart & Execution Guide](docs/execution_guide.md)**
   * How to install dependencies, run the solver frameworks, and evaluate the full benchmark. Expected results and baseline execution targets.
2. **[Architecture & Philosophy](docs/architecture.md)**
   * Explanation of the N-ary AST abstraction engine, code structure, testing strategy, and paradigm fusion (Semantic Hashing, Lexicase Selection, Generative Priors).
3. **[Model & Report Guide](docs/model_guide.md)**
   * Instructions for reading the explicitly generated JSON Model Primitives (`arc_library.json`) and understanding the Markdown Introspection Evaluation Reports.

### 2. Developmental Logs
3. **[Prompts Log](docs/prompts.md)**
   * A continuous log of LLM meta-prompts, agent instructions, and user requests used to autonomously generate and steer this codebase.
4. **[Session Log](docs/session_log.md)**
   * A historical record of iterative development sprints, bugs encountered, and breakthroughs.
5. **[Next Steps](docs/next_steps.md)**
   * A living document tracking immediate todo items and short-term roadmap objectives.

### 3. Theory and Strategy
6. **[Theory](docs/theory.md)**
   * The fundamental mathematical and conceptual underpinning behind the Universal Solver.
7. **[Vision](docs/vision.md)**
   * The long-term aspirational goals for extending these principles towards robust AGI.
8. **[Strategy](docs/strategy.md)**
   * Tactical approaches to bridging the gap between current MVP capabilities and the vision.
9. **[ARC-AGI Landscape](docs/arc_agi_landscape.md)**
   * Analysis of the broader ARC-AGI ecosystem, existing SOTA solutions (like Jeremy Berman's techniques), and where this pure-symbolic approach fits.

### 4. Extension Manuals
10. **[Adding Domains](docs/adding_domains.md)**
    * Instructions on how to plug in entirely new environments (e.g., Zork, Math) alongside ARC.
44. **[Adding Primitives](docs/adding_primitives.md)**
    * Guide on writing new Python functions and registering them into the DSL for Wake-Sleep extraction.

---

## ✅ Verification & Compliance
As of March 2026, the AGI Sandbox executes completely deterministically (no fuzzy LLM logic). The codebase guarantees execution safety and mathematical validity through:
- **100% Integration Test Pass Rate (186/186 tests)**
- **86% Total Line/Branch Coverage** spanning the Symbolic Tree Compilers, Multiprocessing Execution loops, and Lexicase fitness functions.
- Fully regressed against `Float TypeError` mappings, Object Array `Dimension Mismatch` loopholes, and Zombie Multiprocessing threads.
