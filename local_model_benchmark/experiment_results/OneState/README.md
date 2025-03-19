
# OneState

**Description:** Train and evaluate model on single state data.

## Hyperparameters
```
Input size:         16
Batch size:         1

Hidden size:        128
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
population, total
net migration
arable land
birth rate, crude
gdp growth
death rate, crude
agricultural land
rural population
rural population growth
age dependency ratio
urban population
population growth
adolescent fertility rate
life expectancy at birth, total
```
## Loss graph


![Loss graph](./plots/loss.png)


## Prediction of Czechia by the training data


![Prediction of Czechia by the training data](./plots/evaluation.png)

# Metric result
{'mae': 13728.871346101718,
 'mse': 2832384243.20279,
 'r2': -34.26208806894042,
 'rmse': 14375.692214473955}
