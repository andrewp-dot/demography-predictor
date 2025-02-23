"""
In this file are experiments with local model. 
"""

# Standard libraries
import pprint
import logging
from config import setup_logging
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.local_model.preprocessing import StateDataLoader
from src.local_model.model import LSTMHyperparameters, LocalModel, EvaluateLSTM

logger = logging.getLogger("benchmark")

# TODO: Define experiments

## Maybe here define base experiment function


# Data based experiments


## 1. Use data for just a single state
def single_state_data_experiment() -> None:
    STATE = "Czechia"
    state_loader = StateDataLoader(STATE)

    # Single state dataframe
    state_df = state_loader.load_data()

    # Exclude country name
    state_df = state_df.drop(columns=["country name"])

    # Get features
    FEATURES = [col.lower() for col in state_df.columns]

    single_state_params = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=128,
        sequence_length=10,
        learning_rate=0.0001,
        epochs=10,
        batch_size=1,
        num_layers=3,
    )
    single_state_rnn = LocalModel(single_state_params)

    # Preprocess data
    scaled_state_df, state_scaler = state_loader.scale_data(
        data=state_df, scaler=MinMaxScaler()
    )
    state_train, state_test = state_loader.split_data(scaled_state_df)

    # Get the training input and target sequences
    train_in_seqs, train_target_seqs = state_loader.preprocess_training_data(
        data=state_train,
        sequence_len=single_state_params.sequence_length,
        features=FEATURES,
    )

    train_batches, target_batches = state_loader.create_batches(
        batch_size=single_state_params.batch_size,
        input_sequences=train_in_seqs,
        target_sequences=train_target_seqs,
    )

    # Train model
    single_state_rnn.train_model(
        batch_inputs=train_batches, batch_targets=target_batches
    )

    # Get stats
    stats = single_state_rnn.training_stats
    fig = stats.create_plot()

    # Save training stats or plot it

    # Evaluate model
    unscaled_eval_input, unscaled_eval_output = state_loader.split_data(state_df)

    single_state_rnn_evaluation = EvaluateLSTM(single_state_rnn)
    single_state_rnn_evaluation.eval(
        unscaled_eval_input,
        unscaled_eval_output,
        features=FEATURES,
        scaler=state_scaler,
    )

    # Get evaluation metrics
    formated_metrics = pprint.pformat(single_state_rnn_evaluation.metrics)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiment
    single_state_data_experiment()

## 2. Use data for all states (whole dataset)
## 3. Use data with categories (divide states to categories by GDP in the last year, by geolocation, ...)
## 4. Devide data for aligned sequences (% values - 0 - 100) and for absolute values, which can rise (population, total, ...)

# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal
# 2. Compare prediction using whole state data and the last few records of data
# 3. Predict parameters for different years (e.g. to 2030, 2040, ... )
# 4. Compare model with statistical methods (ARIMA, GM)


# Use different scaler(s)
# 1. Robust scalers, MinMax, Logaritmic transformation

# Odstranit outliers?
# Zaokrúhlenie dát, čo s nimi?

# Spájanie modelov:
# Stacking? - priemerovanie vysledkov viacerych modelov subezne
# Boosting? - Ada boost (les neuroniek? :D) , XGBoost

# Try pycaret
