
An exploratory AGI solver for the Abstraction and Reasoning Corpus (ARC). Built on a robust symbolic reasoning engine that utilizes the **"Wake-Sleep" cycle** to meta-learn reusable library operators across training tasks.

---

## 📖 Documentation & Research Logs

| Category | Key Documents |
| :--- | :--- |
| **🚀 Results** | [Training Marathon (400 Tasks)](docs/training_marathon_results.md) • [Improvement Plan](docs/improvement_plan.md) |
| **🏗️ Deep Dives** | [Architecture Overview](docs/architecture.md) • [Execution Guide](docs/execution_guide.md) |
| **🧬 Methodology** | [Walkthrough](docs/walkthrough.md) • [Hyperparameter Optimization](docs/hyperparameter_optimization_report.md) |
| **🔭 Vision** | [Project Strategy](docs/strategy.md) • [Research Theory](docs/theory.md) |

---

## ⚡ Quickstart

To launch a full training pass on the ARC-AGI dataset:

```bash
# 1. Start Training (Wake-Sleep)
python agi.py train --epochs 5 --task-workers 8

# 2. Run Evaluation on separate set
python agi.py eval --model models/latest_model.json

# 3. Or run the full pipeline in one go
python agi.py pipeline --train-tasks 100 --eval-tasks 100
```

---

## 🏗️ Architecture

The system is split into three core specialized layers:

### 1. The Core Engine (`core/`)
*   **Tree Compiler (`tree.py`)**: A purely symbolic N-ary AST representation for programmatic logic. Supports serialization, functional evaluation, and semantic fingerprinting.
*   **Search Engine (`search.py`)**: Implements a high-throughput Beam Search guided by $ROC_{math}$ (Return-on-Compute). Uses Lexicase selection to maintain diverse "stepping stone" hypotheses.
*   **Library Learner (`library.py`)**: The "Sleep" phase. It scans successful solutions, extracts frequent sub-structures, and promotes them to first-class DSL primitives.

### 2. ARC Domain Layer (`domains/arc/`)
*   **ARC Domain (`domain.py`)**: Implements the logic for checking 2D spatial grid solutions, calculating MDL complexity, and ranking candidates.
*   **Benchmark Runner (`runner.py`)**: A parallelized, high-performance executor with live "Scoreboard" reporting and HTML introspection.
*   **Primitives (`primitives.py`)**: The initial "Seed" DSL (geometric transforms, color ops, etc.).

### 3. Unified Interface (`agi.py`)
A single, professional CLI for orchestrating training and evaluation runs.

---

## 📊 Scoreboard & Introspection

During execution, the solver displays a live scoreboard:
```text
  ┌ scoreboard [Epoch 1/5] ─
  │ ✓ solved=26 (16.1%)  ⚠️ near=67  ✗ unsolved=135
  │ → active=8  ⏳ pending=231  done=163/400  success=16.1%
  │ TIME:  elapsed=122.3s (Throughput: 0.75s/task | Latency Avg: 5.8s)
  │ WORK:  speedup=7.77x (97.1% core) | STRAGGLER: 122.3s, 5.3k evals
  │ EVALS: total=863.9k (7.1k/s | Per-Task Avg: 5.3k)
```

Detailed HTML reports are generated in `reports/`, allowing you to visually inspect failure modes and the evolution of the solver's logic.

---

## ✅ Principles of AGI Verification

1.  **Determinism**: Every run is deterministic given a fixed seed.
2.  **Noise Mitigation**: Metrics are normalized by total program evaluations ($ROC_{math}$) to avoid hardware bias.
3.  **No LLM Dependency**: Pure symbolic discovery. No external APIs or pre-trained models.
4.  **Full Coverage**: 100% integration test pass rate on the core tree compiler.
