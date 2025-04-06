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
        
        <!-- New -->
        - experiment with exclusion of covid years
        - Finetune experiment -> use all states and finetune layer using the state 
        - use just part of the dataset (only states with same geolocation, wealth level...) -> by wealth country richness
    - Model experiments
        - Parameter optimization using `optuna` library
        - Compare LSTM model and statistical model(s)
3. Statistical model implementation (ARIMA)
4. Global model implementation (XGBoost)
5. Demography predictor implementation (both, local and global) -> relatively poor performance, for poulation total it is useless

<!-- New -->
6. Write experiment results -> method comparision, advantages, disadvantages (ARIMA, LSTM experiment)
    - fix experiment, use one neural network and N ARIMAS for comparision
    - print the table with header: `feature_name` | `best_model`
7. The full model evaluation
8. Demography predictor training script for multiple models:
    - base LSTM as local model
    - finetunable LSTM with custom data (single state, multiple states data)
    - choose from possible targets

<!-- The newest -->
9. Model comparation script:
    - Better model evaluation saving... (Maybe saved model short description? Loss graph, evaluation -> ovearll metrics, by state metrics..) -> model evaluation comparison
10. Ensemble model (local model, which predicts each feature indepentetly)
    - pure LSTM (LSTM for each feature)
    - pure ARIMA (ARIMA model for state for each feature)


Bonus:
1. Simple CLI for `lbenchmark`. For more info run `poetry run cli lbenchmark --help`
2. Model inference (`API`) + simple client (can be used as `cookbook` for implementing request client)
3. `API` endpoint for Lakmoos integration demo (distribution curve - generrator input?)

## Work in progress (WIP)
1. Statistical model implementation (Grey model)
2. Experiments:
    - None


## TODO:
1. Experiments
    - Different Neural network architectures (not a just a pure LSTM)
    - (BONUS) LSTM and NN as a global model instead of local 
    - (fix) features for experiments -> use all or mostly the same experiments
    - Try Forest of LSTMs (each LSTM for each feature? combined with ARIMA (by better feature performance?))

2. Try predict population total using the population growth
    

## TO DISCUSS:
1. Low accuracy of my model. To try:
    - better data preprocessing (logaritmic standardization for features like `net migration` and `age dependency ratio`)
    - improve training of the model:
        - bidirectional LSTM
        - train on prediction multiple future values at once (instead of just `X(t + 1)` predict `X(t + 1), X(t + 2), X(t + 3) ...` )
2. What if everything fails?
3. Structure of my bachelor thesis:
    - Theory
    - Data preprocessing (Mainly this)
    - Method selection (Mainly this -> 2 types -> RNN method selection and Target variables method selection)
    - Model design (Do I need to rewrite it?)
    - Experiments (I have some, do I need to write new ones, or just use existing and change method selection?)

    

## Bonus
1. Maybe Docker appliaction?