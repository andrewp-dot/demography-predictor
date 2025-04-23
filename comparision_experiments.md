# This is just temporary - here you can find the experiments designs. More existing experiments will be melted into the 1.


## Experiment 1: Train using different data

**Goal:** Find the best dataset (or subset) for training the LSTM network for the predicting single state feature development.

Models:
    - base LSTM (OneState)
    - base LSTM (StatesCategories)
    - base LSTM (AllStates, AllStatesWithoutHighErrorFeatures)

## Experiment 2: Predicting all features at once vs model for each feature separately

**Goal:** Is it better to compare are features separately or together at once? 

Models:
    - BasLSTM
    - BaseRNN * NumberOfFeatures (RNNvsStatisticalMethodsSingleFeature)

## Experiment 3: Compare different model types

**Goal:**: Find the answer for *hypothesis*: The adding adional layers for trained for specific tasks improves the model accuracy.

Models:
    - base LSTM -> best model from model 1
    - finetunable LSTM with custom data single state (FineTuneLSTMExp)
    - finetunable LSTM with custom data multiple states  (states groups) (FineTuneLSTMExp)

## Experiment 4: Parameter tuning?

**Goal**: Improve accuracy of the model by using the optimal paremeters.

**Goal:**
    - best model from previous experiments (3,2,1)
    - optuna (LSTMOptimalParameters)


## Experiment 5: LSTM vs Statistical models

**Goal:** LSTM has better accuracy then the statistical models.

Models:
    - BaseRNN (RNNvsStatisticalMethods) (or best model from previous experiments)
    - LocalARIMA
    - GM (not implemented yet)

