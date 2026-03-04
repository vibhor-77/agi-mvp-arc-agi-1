# Model & Report Visualization Guide

The Universal Symbolic AGI solver shifts away from traditional opaque neural networks and instead builds a transparent, mathematically verifiable **Primitive Library** (acting as your model file) and compiles **Markdown Introspection Reports** (acting as your evaluation metrics).

This document explains what these files are, how to read them, and how to verify the AGI engine's cognitive progress.

---

## 1. The Model File (`models/arc-agi-1_<timestamp>.json`)

The "model" of this engine is not a massive matrix of unintelligible float variables. It is an explicit library of functional abstractions natively discovered by the algorithm. 

By default, executing `train_wake_sleep.py` saves this file incrementally to uniquely timestamped documents inside the `models/` folder.

### How to read the `.json` file:
The JSON model file is structured into two main dictionaries:
1. `"library"`: The learned primitive abstractions.
2. `"transitions"`: The Markov Generative Priors.

**Example `"library"` structure:**
```json
"library": {
  "lib_op_512": {
    "expr": "grot90(goverlay(X0, X1))",
    "arity": 2
  }
}
```
**Interpretation:** The model natively discovered that rotating a grid 90 degrees after instantly overlaying it with another grid was mathematically useful across multiple different ARC tasks. It compressed that entire deep sub-tree into a single macro command called `lib_op_512`.

**Example `"transitions"` structure:**
```json
"transitions": {
  "g_extract_color": {
    "g_invert": 0.05,
    "grot90": 0.85
  }
}
```
**Interpretation:** The engine empirically proved that executing `grot90` immediately after extracting a specific color is historically 85% likely to lead to a solution. Next time the `BeamSearch` queue runs, it will statistically spawn AST sequences matching this pattern first.

---

## 2. Introspection Reports 

45: Both the train and evaluation scripts now feature native Markdown Auto-Generators that output explicitly *why* tasks failed.
46: 
47: By default, the training script generates `reports/train_<timestamp>.md` (spanning all 5 Epochs), and the evaluation script generates `reports/eval_<timestamp>.md`.
48: Zero-dependency Javascript `.html` wrappers are generated uniformly alongside them for seamless GUI browser rendering.

### How to understand the `Introspection` outputs:
During the `BeamSearch`, if a task is unsolved, the engine records the final and best AST (by fitness score) and logs the failure category:

- **Dimension Mismatch**: `Expected 4x4, Actual 6x6`
  *(The AI learned rules to manipulate the colors, but failed to crop the boundary).*
- **Pixel Mismatch**: `3/16 pixels incorrect` 
  *(The AI brilliantly solved the structure, but failed on a few edge-case details. Usually indicates it is missing an `if/else` control flow mechanism).*
- **Capacity Exhausted**: `No valid logical AST discovered within generation limits`
  *(The required combination of functions is so mathematically complex that it would take billions of guesses. Pure proof that we need more macro rules in the library).*

---

## 3. Quickstart Execution Commands

The repository is pre-tuned with strict empirical "Sweet Spot" constraints. You do not need to configure complex hyperparameters manually.

### Phase 1: Train (Wake-Sleep)
This loop executes the full abstraction-gathering cycle across the 400 ARC training grids using 8 OS CPU cores. It iterates for 5 Epochs, building hierarchical abstraction logic.
```bash
python3 train_wake_sleep.py
```
**What happens behind the scenes:**
- The engine compiles and saves `models/arc-agi-1_<timestamp>.json` continually.
- The engine writes the Epoch reports to `reports/train_<timestamp>.md`.
- No parameters required. It will default to 10-width beam paths across 100 deep-generations per grid.

### Phase 2: Blind Evaluation
This validates your generated models purely out-of-sample on unseen arrays.
```bash
python3 evaluate_agi.py
```
**What happens behind the scenes:**
- The script dynamically sweeps the `models/` directory for the most recently modified `.json` file and assumes it is the target architecture.
- It parses the 400 separate unseen Evaluation tasks.
- It writes the final comprehensive Introspection strings natively to `reports/eval_<timestamp>.md`.

### Phase 3: The Unified Pipeline Engine
If you don't want to run the training and evaluation steps manually, you can use the unified sequence script to orchestrate the entire end-to-end extraction natively:
```bash
python3 run_full_pipeline.py --train-tasks 400 --eval-tasks 400
```
**What happens behind the scenes:**
- It securely executes `train_wake_sleep.py` and passes the resulting absolute timestamp ID down structurally into `evaluate_agi.py` to prevent race conditions across parallel invocations.

### (Optional) Reproducibility
If you need to identically reproduce a test bench report for research logging, provide a deterministic kernel seed to lock the uniform mutations:
```bash
python3 evaluate_agi.py --seed 42
```
