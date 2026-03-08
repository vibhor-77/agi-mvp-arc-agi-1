
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

## ⚙️ Installation

### Option 1: Conda (Recommended for Apple Silicon / M1 / M2 / M3)
If you are on an M-series Mac or using Anaconda/Miniconda:

```bash
# 1. Create and activate a new environment
conda create -n agi-arc python=3.10 -y
conda activate agi-arc

# 2. Install high-performance dependencies
conda install numba numpy -y
pip install -r requirements.txt
```

### Option 2: Virtual Environment (venv)
```bash
# 1. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Performance Optimization (JIT Acceleration)
The ARC solver is optimized for high-throughput symbolic search on modern hardware:
*   **Vectorized Object Engine**: Object extraction (BFS/DFS) and connected component labeling are fully offloaded to **Numba-parallel** kernels, bypassing Python's iteration tax.
*   **10,000+ Evals/Sec**: Core matching kernels and primitives are JIT-compiled, pushing core-normalized throughput past **10k evals/sec** on M-series hardware.
*   **NumPy Native**: The entire evaluation pipeline resides in machine-native memory, ensuring bit-perfect parity with the ARC specification at scale.
*   **Parallel Execution**: Non-daemonic multiprocessing enables scaled search across all available CPU cores with minimal overhead.

---

## 🚀 Quickstart

To launch a full training pass on the ARC-AGI dataset:

```bash
# 1. Verify Installation (pytest)
pytest -q

# 2. Start Training (Wake-Sleep)
python agi.py train --epochs 5 --task-workers 0

# 3. Run Evaluation on separate set
python agi.py eval --model models/latest_model.json

# 3. Or run the full pipeline in one go
python agi.py pipeline --train-tasks 100 --eval-tasks 100 --task-workers 0
```

---

## 🛡️ Resource Guard

Use auto worker selection to maximize throughput while avoiding swap:

```bash
python agi.py train \
  --task-workers 0 \
  --reserve-mem-gb 12 \
  --mem-per-task-worker-gb 3 \
  --cpu-reserve 2
```

Notes:
* `--task-workers 0` enables CPU+RAM-aware worker capping.
* `--capture-traces` is off by default because traces are memory-heavy.
* Tune `--reserve-mem-gb` upward if your system starts swapping.

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

### 3. Task-Adaptive Meta-Learning (`domains/arc/domain.py`)
*   **Feature Extraction**: Analyzes input grids for geometric signatures: Symmetry (H/V), Object Density, Color Entropy, and Resizing trends.
*   **Dynamic Biasing**: Automatically adjusts the Search Transition Matrix to prioritize relevant primitive families (e.g., boosting **Geometric** for symmetric tasks, **Collective** for high-object tasks).
*   **Cumulative Culture**: Maintains a global pool of successful program ASTs. Successful solutions are "seeded" into future tasks, enabling **Zero-Shot Transfer** and compound learning.

### 4. Reward Shaping & Structural Scorer
*   **Multi-Factor SSIM**: Replaces binary bit-matching with a weighted similarity score that rewards Dimension Match (20%), Color Palette Overlap (20%), Non-Zero Density (10%), and exact Pixel Alignment (50%).
*   **Smooth Landscape**: This scoring allows the beam search to "feel" its way toward bit-perfection by rewarding near-misses and structural alignment.

### 5. Unified Interface (`agi.py`)
A single, professional CLI for orchestrating training and evaluation runs.

---

## 📦 Collective Object Engine (DSL+)

The system includes a specialized **G-Family** of primitives for multi-object manipulation without explicit loops:
*   **`g_rainbow`**: Sequentially recolors objects (1, 2, 3...) based on their discovery order.
*   **`g_stack_v` / `g_stack_h`**: Rigid body gravity simulation (vertical and horizontal).
*   **`g_sort_h`**: Orders objects horizontally based on pixel area (MDL-efficient sorting).
*   **`g_connect_pixels_to_rect`**: Projects isolated markers to the nearest aligned object border.
*   **`g_recolor_isolated_to_nearest`**: Sophisticated noise reduction for "floating" pixel artifacts.
*   **`g_place_like`**: Context-aware spatial anchoring (placing objects relative to a reference grid).

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
