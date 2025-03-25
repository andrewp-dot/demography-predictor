
# AllStatesWithoutHighErrorFeatures

**Description:** Train and evaluate model on whole dataset excluding the high errorneous features.

## Hyperparameters
```
Input size:         14
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
{'mae': 5.641811257960231,
 'mse': 67.32442509373665,
 'r2': -6018.214294423909,
 'rmse': 5.834995847268258}
