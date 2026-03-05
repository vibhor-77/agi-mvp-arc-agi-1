# Universal ARC-AGI Solver 🧠🧩

An exploratory general-purpose Artificial General Intelligence (AGI) solver framework structured around the *Four Pillars of True General Learning*:
1. **Feedback Loops**: Recursive wake-sleep iteration.
2. **Abstraction & Composability**: DSL expansion via sub-tree extraction.
3. **Exploration**: Multi-objective beam search via Lexicase selection.
4. **Approximability**: MDL-based MDL complexity priors.

This project fuses **Symbolic Regression (SR)** and **Library Learning (LL)** to conquer the ARC-AGI benchmark through pure-symbolic discovery.

---

## ⚡ Fast-Track: Reproduction

To launch a full production training pass using the scientifically verified **"Sweet Spot"** (the most efficient hyperparameter balance identified via ROC_math):

```bash
# Optimized Configuration: Beam 10 | Offspring 20 | Gen 25
python3 train_wake_sleep.py --task-workers 8 --epochs 5
```

📊 **[See "Sweet Spot" Optimization Results](docs/hyperparameter_optimization_results.md)**  
🔬 **[How we Tune: The ROC_math Guide](docs/hyperparameter_tuning_guide.md)**

---

## 📖 Documentation Hub

The documentation is organized by functional role. Use this map to navigate the repository:

### 🔬 1. Research & Performance
*   **[Sweet Spot Results](docs/hyperparameter_optimization_results.md)**: Finalized search parameters for ARC-AGI.
*   **[Hyperparameter Tuning Guide](docs/hyperparameter_tuning_guide.md)**: Instructions for running sweeps and calculating Return-on-Compute.
*   **[ARC-AGI Landscape](docs/arc_agi_landscape.md)**: Competitive analysis of solver architectures.

### 🏠 2. Core Engine Manuals
*   **[Quickstart & Execution Guide](docs/execution_guide.md)**: Detailed install and run instructions.
*   **[Architecture & Philosophy](docs/architecture.md)**: Deep dive into the N-ary AST engine and search logic.
*   **[Model & Report Guide](docs/model_guide.md)**: How to interpret learned library JSONs and HTML Introspection reports.

### 🧠 3. Theory & Vision
*   **[The Theory of Universal Solving](docs/theory.md)**: Mathematical foundations.
*   **[Vision for AGI](docs/vision.md)**: Aspirational long-term roadmap.
*   **[Execution Strategy](docs/strategy.md)**: Tactical plan for scaling the prototype.

### 🛠️ 4. Developer & Maintenance
*   **[Adding Primitives](docs/adding_primitives.md)**: Expanding the DSL with new Python functions.
*   **[Adding Domains](docs/adding_domains.md)**: Moving beyond ARC (e.g. into game environments).
*   **[Session Log](docs/session_log.md)**: Historical record of major architectural shifts and bugs.
*   **[Prompt Engineering Log](docs/prompts.md)**: Meta-prompts used to evolve the agentic codebase.
*   **[Next Steps](docs/next_steps.md)**: Live TODO board.

---

## ✅ Verification & Compliance
As of March 2026, the AGI Sandbox executes completely deterministically. 
- **100% Integration Test Pass Rate (211/211 tests)**
- **88.2% Total Line/Branch Coverage** across the Tree Compiler and Search Engine.
- **Hardware Agnostic**: Benchmarked via noise-free Program Evaluation metrics ($ROC_{math}$).
