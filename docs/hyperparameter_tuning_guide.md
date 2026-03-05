# AGI Hyperparameter Tuning Guide

This guide explains how to use the systematic optimization tools in the AGI Sandbox to find the most efficient search parameters for a given DSL configuration.

## The Search Philosophy: Return on Compute (ROC)

In symbolic reasoning, there is a literal "exponential explosion" of the search space as trees get deeper. Therefore, we do not simply search for the highest solve rate; we search for the highest **Return on Compute (ROC)**.

**Formula:**
$$ROC = \frac{SolveRate \times 100}{Time_{minutes}}$$

This metric prevents us from over-investing CPU time in marginal improvements that could be better handled by **Library Learning** (the Sleep phase).

---

## How to Run the Parallel Sweep

The sweep script independently tests combinations of `beam_size`, `offspring`, and `generations`. It parallelizes the combinations across all available CPU cores to saturate the system.

### Basic Usage
```bash
# Run the sweep and capture results in a log
python3 -u scripts/sweep_hyperparams.py | tee logs/hyperparameter_experiment.log
```

### Advanced: Customizing the Sweep Space
If you want to test different ranges, edit the following lists in `scripts/sweep_hyperparams.py`:
```python
beam_sizes = [5, 10, 20]
offspring_counts = [10, 20, 40]
gen_depths = [25, 50, 100]
```

### Advanced: Scaling the Task Subset
The sweep currently targets the first **5 tasks** of the training set for speed. For a more rigorous (but slower) sweep, increase the slice:
```python
tasks = load_tasks_from_dir("arc_data/data/training")[:10]
```

---

## Interpreting Results

The script outputs a live table and saves a detailed JSON record to `docs/hyperparameter_sweep_data.json`.

1.  **Solve Gradient**: If `Solves` stays constant while `Time` increases, you have passed the point of diminishing returns for that parameter.
2.  **Breadth vs. Depth**: If a wide search (High Offspring/Beam, Low Gen) has a higher ROC than a deep search (Low Breadth, High Gen), the domain favors "short-path" discoveries.
3.  **The Sweet Spot**: The "Winner" is the configuration with the highest ROC. Use these parameters in your `run_full_pipeline.py` or `train_wake_sleep.py` calls.

---

## Production Integration

Once the sweep identifies a winner, apply it to the main training loop:

```bash
python3 train_wake_sleep.py \
    --beam-size [WINNER_BEAM] \
    --offspring [WINNER_OFFSPRING] \
    --generations [WINNER_GEN]
```
