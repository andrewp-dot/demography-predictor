
# RNNvsStatisticalMethods

**Description:** Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. ARIMA model is trained just to predict 1 feauture from all features and LSTM predict all features at once.

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

## Hyperparameters

```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             2

Learning rate:      0.0001
Epochs:             30

Bidirectional:      False
```

