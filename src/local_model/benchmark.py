"""
In this file are experiments with local model. 
"""

# Standard libraries
import pprint
import logging
from config import setup_logging
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
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
        batch_size=1,  # Edit this for faster training
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
    # formated_metrics = pprint.pformat(single_state_rnn_evaluation.metrics)

    print(single_state_rnn_evaluation.per_target_metrics.head())
    print("-" * 100)
    print(single_state_rnn_evaluation.overall_metrics.head())


## 2. Use data for all states (whole dataset)
def whole_dataset_experiment() -> None:
    # Load whole dataset
    states_loader = StatesDataLoader()

    all_states = states_loader.load_all_states()

    # Get only numerical features
    FEATURES = [
        col.lower()  # Lower to ensure key compatibility
        for col in all_states["Czechia"].select_dtypes(include="number").columns
    ]

    # Get hyperparameters for training
    all_state_state_params = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=128,
        sequence_length=10,
        learning_rate=0.0001,
        epochs=10,
        batch_size=1,
        num_layers=3,
    )

    # TODO: Maybe you an write this to all in one function

    # Split data
    states_train_data, states_test_data = states_loader.split_data(
        states_dict=all_states, sequence_len=all_state_state_params.sequence_length
    )

    # Scale data
    scaled_train_data, all_states_scaler = states_loader.scale_data(
        states_train_data, scaler=MinMaxScaler()
    )

    # Create input and target sequences
    train_input_sequences, train_target_sequences = (
        states_loader.create_train_sequences(
            states_data=scaled_train_data,
            sequence_len=all_state_state_params.sequence_length,
            features=FEATURES,
        )
    )

    # Create input and target batches for faster training
    train_input_batches, train_target_batches = states_loader.create_train_batches(
        input_sequences=train_input_sequences,
        target_sequences=train_target_sequences,
        batch_size=all_state_state_params.batch_size,
    )

    # Train rnn
    all_states_rnn = LocalModel(all_state_state_params)

    all_states_rnn.train_model(
        batch_inputs=train_input_batches, batch_targets=train_target_batches
    )

    # Get stats
    stats = all_states_rnn.training_stats
    fig = stats.create_plot()

    # Save training stats or plot it

    # Evaluate model
    # unscaled_eval_input, unscaled_eval_output = state_loader.split_data(state_df)

    single_state_rnn_evaluation = EvaluateLSTM(all_states_rnn)
    single_state_rnn_evaluation.eval(
        states_train_data,
        states_test_data,
        features=FEATURES,
        scaler=all_states_scaler,
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiment
    # single_state_data_experiment()
    whole_dataset_experiment()


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
