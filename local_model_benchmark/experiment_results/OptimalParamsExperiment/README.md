
# OptimalParamsExperiment

**Description:** Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. Each model is trained just to predict 1 feauture from all features.

## Hyperparameters
```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             20

Bidirectional:      False
```
# Base model evaluation
Hyperparameters:
```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             20

Bidirectional:      False
```

## Base model predicted vs reference values
Displays the performance for every feature predicted of the `Base Model`.

![Base model predicted vs reference values](./plots/base_model_eval.png)

# Optimal model evaluation

## Hyperparameters
```
Input size:         16
Batch size:         1

Hidden size:        1747
Sequence length:    10
Layers:             3

Learning rate:      5.883446307204169e-05
Epochs:             20

Bidirectional:      False
```

## Optimal model predicted vs reference values
Displays the performance for every feature predicted of the `Optimal Model`.

![Optimal model predicted vs reference values](./plots/optimal_model_eval.png)

# Compare metric results

Base model:
{'mae': 9421.125407766645,
 'mse': 1484320573.7399616,
 'r2': -11.417041799465022,
 'rmse': 10371.3786139561}

Optimal model:
{'mae': 10593.934058979394,
 'mse': 2020882912.3074396,
 'r2': -9.766559135249262,
 'rmse': 11396.100754484884}

