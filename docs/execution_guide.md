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

## 3. Running the Base Evaluator

To execute the solver against the baseline `8` operation subset versus the full `93` expanded AGI primitives subset, run the benchmark orchestrator.
The `--quick` flag limits the tree generations and beam size heavily to test if the orchestrator runs without crashing.
The `--tasks` flag limits the number of JSON tasks the orchestrator evaluates.

```bash
python3 run_real_arc.py --quick --tasks 8 --workers 1
```

**M1 Max Optimization:**
If you want to unleash the full processing power of your MacBook Pro without blocking on macOS ThreadPool limitations, utilize the `--task-workers` threaded flag to evaluate multiple AST tasks physically simultaneously:
```bash
python3 run_real_arc.py --task-workers 8
```

## 4. Running the Wake-Sleep Paradigm Loop

The crown jewel of this architecture is the cyclic `train_wake_sleep.py` script.
This script utilizes an end-to-end evolutionary loop:
1. **WAKE Phase:** Attempt batches of ARC tasks.
2. **SLEEP Phase:** Extract common functional sub-trees from solved tasks and compress them into new reusable primitives (`lib_op_X`), computing a sequence generative prior $P(\text{child} \mid \text{parent})$.
3. **Repeat:** The expanded DSL tackles harder tasks in subsequent epochs with weighted semantic guidance.

```bash
python3 train_wake_sleep.py --tasks 10 --epochs 3 --task-workers 8
```
