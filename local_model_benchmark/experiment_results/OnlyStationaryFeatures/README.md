
# OnlyStationaryFeatures

**Description:** Train and evaluate model on single state data using only stationary features.

## Hyperparameters

```
Input size:         14
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             10

Bidirectional:      False
```

## Features
```
year
fertility rate, total
arable land
birth rate, crude
gdp growth
death rate, crude
population ages 15-64
population ages 0-14
agricultural land
population ages 65 and above
rural population
rural population growth
urban population
population growth
```


## Loss graph


![Loss graph](./plots/loss.png)


## Prediction of Czechia by the training data


![Prediction of Czechia by the training data](./plots/evaluation.png)

# Metric result
{'mae': 1.947342423652564,
 'mse': 11.342403564432024,
 'r2': -29.84284275862847,
 'rmse': 2.1629936609733944}
