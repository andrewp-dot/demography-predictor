
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
Epochs:             1

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
Epochs:             1

Bidirectional:      False
```

## Base model predicted vs reference values
Displays the performance for every feature predicted of the `Base Model`.

![Base model predicted vs reference values](./plots/base_model_eval.png)

