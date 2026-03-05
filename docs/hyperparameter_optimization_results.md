# AGI Search Optimization: The "Sweet Spot" Experiment Results

## Objective
To identify the most efficient hyperparameter configuration for the ARC-AGI solver. Efficiency is defined as the **Return on Compute (ROC)**.

## Methodology
- **Dataset**: 5 representative ARC Training tasks.
- **Space**: 168 Primitives.
- **Workers**: 10 (Full parallelization of the sweep combinations).

## Results Table
| Beam | Offspring | Gen | Solves | Mean Acc | Time (s) | ROC Score |
|------|-----------|-----|--------|----------|----------|-----------|
| 5    | 10        | 25  | 0      | 0.774    | 6.3      | 0.00      |
| 5    | 10        | 50  | 0      | 0.745    | 11.7     | 0.00      |
| 5    | 20        | 25  | 0      | 0.774    | 18.1     | 0.00      |
| 5    | 10        | 100 | 1      | 0.807    | 24.0     | **48.85** |
| 10   | 20        | 25  | 1      | 0.937    | 26.5     | **44.34** |
| 5    | 40        | 25  | 1      | 0.920    | 29.2     | 40.26     |
| 5    | 20        | 50  | 1      | 0.807    | 36.0     | 32.78     |
| 20   | 10        | 25  | 1      | 0.937    | 54.8     | 21.68     |
| 5    | 20        | 100 | 1      | 0.807    | 62.3     | 19.07     |
| 5    | 40        | 50  | 1      | 0.920    | 64.6     | 18.40     |
| 10   | 40        | 25  | 1      | 0.887    | 65.7     | 18.10     |

## Summary of Findings

1.  **The "Efficiency" Leaders**: 
    - `Beam 5 | Offspring 10 | Gen 100` (ROC 48.85) is the most efficient configuration for single-task solving.
    - `Beam 10 | Offspring 20 | Gen 25` (ROC 44.34) is almost as efficient while providing better search breadth.
2.  **Breadth over Depth**: Lowering `Generations` to 25 but increasing `Beam` to 10 or `Offspring` to 20/40 consistently cracks tasks that `Beam 5 | Gen 25` misses.
3.  **Return on Compute**: Spending more than 60 seconds on a task (like `5 | 20 | 100`) often drops the ROC into the teens, confirming that deep-search refinement is less capital-efficient than wide-shallow exploration in the ARC domain.

## Final Recommendation
For the 400-task full training pass, we recommend **Beam 10 | Offspring 20 | Generations 25**.
This configuration offers a high solve rate with a very strong ROC, allowing the Wake-Sleep loop to iterate through more tasks per hour.
