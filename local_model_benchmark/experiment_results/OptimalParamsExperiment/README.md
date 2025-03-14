
# OptimalParamsExperiment

**Description:** The goal is to find the optimal parameters for the given BaseLSTM model.

# Base model evaluation
Hyperparameters:
```
Input size:         1
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
Hyperparameters:
```
Input size:         1
Batch size:         1

Hidden size:        959
Sequence length:    10
Layers:             1

Learning rate:      0.008404695959103137
Epochs:             20

Bidirectional:      False
```


## Optimal model predicted vs reference values
Displays the performance for every feature predicted of the `Optimal Model`.

![Optimal model predicted vs reference values](./plots/optimal_model_eval.png)

# Compare metric results

Base model:
{'mae': 1.8177286783854545,
 'mse': 3.42414460331214,
 'r2': -0.17399243542132026,
 'rmse': 1.850444433997449}

Optimal model:
{'mae': 5.789184570312462,
 'mse': 36.43132465581051,
 'r2': -11.490739881992338,
 'rmse': 6.035836698901861}

