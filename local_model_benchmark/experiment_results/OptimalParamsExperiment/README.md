
# OptimalParamsExperiment

**Description:** The goal is to find the optimal parameters for the given LocalModel model.

# Base model evaluation
Hyperparameters:
```
Input size:         19
Batch size:         1

Hidden size:        128
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             10

Bidirectional:      False
```
## Base model predicted vs reference values
Displays the performance for every feature predicted of the `Base Model`.

![Base model predicted vs reference values](./plots/base_model_eval.png)

# Optimal model evaluation
Hyperparameters:
```
Input size:         19
Batch size:         1

Hidden size:        512
Sequence length:    4
Layers:             3

Learning rate:      0.0001
Epochs:             10

Bidirectional:      False
```
## Optimal model predicted vs reference values
Displays the performance for every feature predicted of the `Optimal Model`.

![Optimal model predicted vs reference values](./plots/optimal_model_eval.png)

# Compare metric results

Base model:
{'mae': 12079.820915318696,
 'mse': 2604285059.360079,
 'r2': -33.558703393503315,
 'rmse': 12590.539022166504}

Optimal model:
{'mae': 13079.038928012296,
 'mse': 3067017069.3189445,
 'r2': -96.77961744184172,
 'rmse': 13542.745215822733}
