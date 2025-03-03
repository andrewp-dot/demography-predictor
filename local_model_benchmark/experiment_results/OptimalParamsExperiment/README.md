
# OptimalParamsExperiment

**Description:** The goal is to find the optimal parameters for the given LocalModel model.

# Base model evaluation
Hyperparameters:
```
Input size:         19
Batch size:         1

Hidden size:        231
Sequence length:    10
Layers:             5

Learning rate:      0.008718063369524982
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

Hidden size:        277
Sequence length:    14
Layers:             4

Learning rate:      0.008718063369524982
Epochs:             10

Bidirectional:      False
```
## Optimal model predicted vs reference values
Displays the performance for every feature predicted of the `Optimal Model`.

![Optimal model predicted vs reference values](./plots/optimal_model_eval.png)

# Compare metric results

Base model:
{'mae': 12172.931254521078,
 'mse': 2638474189.728765,
 'r2': -31.176810774180474,
 'rmse': 12677.22373346006}

Optimal model:
{'mae': 6521.000963137108,
 'mse': 882496245.760047,
 'r2': -12.048249423368086,
 'rmse': 7435.632998746361}
