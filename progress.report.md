# Progress report

**Basic Terminology:**

*Local model* - predictive model, predicts the future values of all features (feautures **does NOT** include `country name`)

*Global model* - based on whole dataset, future feature values (features **DOES** include `country name`) returns the value of target feauture (population total, age distribution, gender distribution...)

## What is done
1. Global model and feature experiments using `pycaret`
2. Local Model experiments:
    - Data experiments
        - Train LSTM model on whole dataset 
        - Train LSTM model using just one State 
        - Train LSTM with only stationary features (% features etc.)
        - Use only stationary features
        - Compare LSTM model and statistical model(s)
    - Model experiments
        - Parameter optimization using `optuna` library
3. Statistical model implementation (ARIMA)

Bonus:
1. Simple CLI for `lbenchmark`. For more info run `poetry run cli lbenchmark --help`

## Work in progress (WIP)
1. Global model implementation (Note: implemented, not integrated)
2. Demography predictor implementation (both, local and global)


## TODO:

0. !!!! FIX THE INTERFACE FOR THE EXPERIMENTS TO BE COMPATIBLE WITH EVERY MODEL YOU CREATE, OR DESIGN THE EXPERIMENTS IN THAT WAY, THAT YOU CAN COMPARE LOCAL MODELS !!! (objective: find a way how to compare different LSTM models).

1. Statistical model implementation (Grey model)
2. Experiments
    - experiment with exclusion of covid years
    - use just part of the dataset (only states with same geolocation, wealth level...)
    - Different Neural network architectures (not a just a pure LSTM)
    - Finetune experiment -> use all states and finetune layer using the state 
    - (BONUS) LSTM and NN as a global model instead of local 
3. Write experiment results -> method comparision, advantages, disadvantages
4. Predictions using the population from Lakmoos AI. –> find a call, data adapter for Clones -> Add to Poli's calendar

## Bonus
1. Model inference (`API`)
2. Maybe Docker appliaction?