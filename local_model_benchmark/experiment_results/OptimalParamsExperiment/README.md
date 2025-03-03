
# OptimalParamsExperiment

**Description:** The goal is to find the optimal parameters for the given LocalModel model.

## Base model hyperparameters:
```
Input size:         19
Batch size:         1

Hidden size:        512
Sequence length:    0.0001
Layers:             6

Learning rate:      0.0001
Epochs:             10

Bidirectional:      False

```
# Base model evaluation
Displays the performance for every feature predicted of the `Base Model`.

![Base model evaluation](./plots/base_model_eval.png)

## Optimal model hyperparameters:
```
Input size:         19
Batch size:         1

Hidden size:        512
Sequence length:    0.0001
Layers:             3

Learning rate:      0.0001
Epochs:             10

Bidirectional:      False

```
# Optimal model evaluation
Displays the performance for every feature predicted of the `Optimal Model`.

![Optimal model evaluation](./plots/optimal_model_eval.png)

