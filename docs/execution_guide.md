# Quickstart & Execution Guide 🚀

This repository tests true AGI paradigms against the deeply complex ARC-AGI dataset and the CartPole control interface natively without external LLM dependencies.

## 1. Setup & Installation

You need Python 3.10+ installed on your system.
This solver natively runs primarily on standard-library Python, but you'll need `pytest` to run the test suite.

```bash
git clone https://github.com/vibhor-77/agi-mvp-arc-agi-1.git
cd agi-mvp-arc-agi-1
```

To run the "real" ARC-AGI dataset, you'll need to clone the official JSON grid files into the directory:

```bash
git clone https://github.com/fchollet/ARC-AGI arc_data
```

## 2. Verify the Test Suite

Before running the heavy benchmarks, verify that the core engine and its geometric abstractions are functioning exactly as intended. I've designed a massive unit testing suite that evaluates the physical constraints of `ARCDomain`, the complexity limits of `BeamSearch` and `PrimitiveLibrary`, and the logic of our N-ary `Node` syntax tree framework.

```bash
python3 -m unittest discover tests -v
# OR
pytest tests/
```

*Expected Output: >170 tests passing perfectly in ~2 seconds.*

## 3. The Training Pipeline (Wake-Sleep)

The crown jewel of this architecture is the cyclic `train_wake_sleep.py` script.
This script utilizes an end-to-end evolutionary loop specifically over the **Training Dataset**:
1. **WAKE Phase:** Attempt batches of ARC tasks using the current AST search limits.
2. **SLEEP Phase:** Extract common functional sub-trees from solved tasks and compress them into new reusable primitives (`lib_op_X`), computing a sequence generative prior $P(\text{child} \mid \text{parent})$.
3. **Repeat:** The expanded DSL tackles harder training tasks in subsequent epochs.

By default, running the script with no arguments runs a deep search across all 400 training tasks using the optimized parameters:
```bash
python3 train_wake_sleep.py
```

## 4. The Evaluation Pipeline

Once the `arc_library.json` file has been fully saturated with learned shape-extraction and logic primitives from the Training set, we test the system's true intelligence on the Unseen Evaluation set.

The `evaluate_agi.py` script is explicitly hardcoded to lock out training data and evaluate against `arc_data/data/evaluation`. It applies the exact logic patterns discovered mathematically in the training pipeline:

```bash
python3 evaluate_agi.py
```

This enforces a strict generalization benchmark devoid of hardcoded logic.

## 5. The Master Pipeline Orchestrator

Rather than running the training and evaluation engines manually, you can use the unified sequence script `run_full_pipeline.py` to continuously process Wake-Sleep and pipe the dynamically generated model straight into the Evaluation sandbox seamlessly.

```bash
python3 run_full_pipeline.py \
    --train-tasks 400 \
    --eval-tasks 400 \
    --epochs 5 \
    --task-workers 8
```

## 6. The "Sweet Spot" (Advanced Parameters)

You do not need to pass any arguments to run the engine. However, the default parameters were chosen through rigorous empirical "Sweet Spot Optimization" using the `scripts/sweep_hyperparams.py` matrix on real ARC data.

- **`--beam-size 10` & `--generations 100`**: We discovered that expanding the beam beyond 10 did not yield *harder* problem solutions, but instead exponentially slowed down processing. The 10x100 tree search hits the algorithmic ceiling in under 20 seconds, representing the optimal complexity-accuracy trade-off.
- **`--task-workers 8`**: By parallelizing exactly 8 tasks globally (using macOS `ProcessPoolExecutor`), the engine fully saturates the M1 Max performance cores without bottlenecking. Memory payload leakage is strictly contained.

When overriding parameters manually, the execution shape is:
```bash
python3 train_wake_sleep.py \
    --tasks 400 \
    --epochs 5 \
    --beam-size 10 \
    --generations 100 \
    --workers 1 \
    --task-workers 8 \
    --model models/arc-agi-1_<timestamp>.json \
    --report reports/train_<timestamp>.md \
    --seed 42
```
*(Tip: Keeping `--workers 1` ensures internal search tree logging remains clean on your terminal shell.)*

### Specialized Artifact Parameters 
The evaluation pipeline defaults to writing outputs cleanly, but you have the power to define exactly where these abstraction layers and evaluation heuristics are saved.

- **`--model <filepath>`** (*Default: LATEST*)
  This controls where the mathematical abstractions and transition states discovered during the `train_wake_sleep.py` loops are incrementally written or loaded from. By default, `train_wake_sleep.py` deposits timestamped files into `models/`. When executing `evaluate_agi.py`, it automatically sweeps the `models/` folder and dynamically loads the most recently created compilation.
- **`--report <filepath>`** (*Default: reports/train_<timestamp>.md / reports/eval_<timestamp>.md*)
  Our Introspection diagnostics natively generate comprehensive Markdown files categorized by failures (Dimension Mismatch, Pixel Mismatch). **Zero-dependency Javascript `.html` wrappers** are also generated concurrently, so you can dynamically view the grids within your web browser safely.
- **`--task-ids <string>`** (*Default: None*)
  Both `train_wake_sleep.py` and `evaluate_agi.py` accept a comma-separated array of specific string IDs (e.g. `--task-ids 007bbfb7,025d127b`) if you demand exact targeting over massive datasets. *Note: When using `run_full_pipeline.py`, this is parameterized cleanly into `--train-task-ids` and `--eval-task-ids` to stop evaluation sets crashing on training UUIDs.*
- **`--seed <integer>`** (*Default: None*)
  If you need 100% deterministic reproducibility for scientific logging across multiple machines, lock the `BeamSearch` evolutionary mutations globally via an exact seed integers.

## 7. Current Performance Metrics
As of Phase 5 testing (with Deep 35-Depth Search, Dynamic Alignment, and Sequence Extrapolation):
- **Empirical Baseline Accuracy:** 20.0% perfect mathematically solved (over a sub-sample of 5 unseen architectures).
- **Mean Pixel Accuracy:** ~93.0% on training iterations (driven by new geometric primitives like `g_center_h` and object alignment logic).
