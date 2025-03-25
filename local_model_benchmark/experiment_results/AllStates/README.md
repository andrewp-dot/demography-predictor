
# AllStates

**Description:** Train and evaluate model on whole dataset.

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


## Evaluation of the model - state: Czechia


![Evaluation of the model - state: Czechia](./plots/evaluation_czechia.png)

# Metric result
{'mae': 19484711.574640106,
 'mse': 7494895401017509.0,
 'r2': -1815750.8932965673,
 'rmse': 21675393.035403267}
