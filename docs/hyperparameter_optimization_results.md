# AGI Search Optimization: The "Sweet Spot" Experiment Results

## Objective
To identify the most efficient hyperparameter configuration for the ARC-AGI solver. Efficiency is defined as the **Return on Compute (ROC)**.

## Methodology
- **Dataset**: 5 representative ARC Training tasks (loading order from training directory).
- **Space**: 168 Primitives.
- **Workers**: 10 (Full parallelization of tasks).

## Results Table
| Beam | Offspring | Gen | Solves | Mean Acc | Time (s) | ROC Score |
|------|-----------|-----|--------|----------|----------|-----------|
| 5    | 10        | 25  | 0      | 0.774    | 2.8      | 0.00      |
| 5    | 10        | 50  | 0      | 0.745    | 4.2      | 0.00      |
| 5    | 10        | 100 | 1      | 0.807    | 7.1      | 78.18     |
| 5    | 20        | 25  | 0      | 0.774    | 7.5      | 0.00      |
| 5    | 20        | 50  | 1      | 0.807    | 13.4     | 42.73     |
| 5    | 20        | 100 | 1      | 0.807    | 19.2     | 30.30     |
| 5    | 40        | 25  | 1      | 0.920    | 9.6      | 58.56     |
| [WAITING FOR BEAM 10 DATA] | | | | | | |

## Initial Observations
1.  **Exploration > Depth**: At Beam 5, doubling the offspring (10 -> 20) at Gen 50 solved the same task as Gen 100 with lower offspring. Higher breadth finds solutions faster.
2.  **Early "ROC" Leader**: `Beam 5 | Offspring 10 | Gen 100` currently holds the highest ROC (78.18), but this is likely because it's the "leanest" settings that crack a task.
3.  **Accuracy Plateaus**: Mean accuracy stays around 0.80-0.92, suggesting that for hard tasks, more search alone won't help without more specialized primitives or learned abstractions.
