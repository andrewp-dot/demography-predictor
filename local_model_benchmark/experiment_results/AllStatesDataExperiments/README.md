
# AllStatesDataExperiments

**Description:** Train and evaluate model on whole dataset.

## Hyperparameters
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
population ages 15-64
population ages 0-14
agricultural land
population ages 65 and above
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
{'mae': 1367861.523275283,
 'mse': 35766567040986.16,
 'r2': -12923.462468859938,
 'rmse': 1392789.204320912}
