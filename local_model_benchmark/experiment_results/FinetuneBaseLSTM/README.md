
# FinetuneBaseLSTM

**Description:** Finetunes the base LSTM model.

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
```# Base model parameters

## Hyperparameters
```
Input size:         16
Batch size:         32

Hidden size:        256
Sequence length:    10
Layers:             3

Learning rate:      0.0001
Epochs:             3

Bidirectional:      False
```
# Finetunable model parameters

## Hyperparameters
```
Input size:         16
Batch size:         1

Hidden size:        256
Sequence length:    10
Layers:             1

Learning rate:      0.0001
Epochs:             30

Bidirectional:      False
```

## Finetuned model predictions - Czechia
Finetuned model predictions.

![Finetuned model predictions - Czechia](./plots/finetuned_model_Czechia_predictions.png)

