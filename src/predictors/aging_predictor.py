# Standard library imports
import pandas as pd
from torch import nn
import logging
from typing import Union, List
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import save_model
from src.predictors.predictor_base import DemographyPredictor
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel
from src.local_model.finetunable_model import FineTunableLSTM
from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.state_preprocessing import StateDataLoader

logger = logging.getLogger("demograpy_predictor")


def create_local_model(features: List[str]):

    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Define local model
    hyperparameters = LSTMHyperparameters(
        input_size=len(features),
        hidden_size=512,
        sequence_length=13,
        learning_rate=0.0001,
        epochs=30,
        batch_size=16,  # Do not change this if you do not want to experience segfault
        num_layers=4,
    )

    # Train local model
    local_model = BaseLSTM(hyperparameters=hyperparameters, features=features)

    # Create train and test data dict
    train_data_dict, test_data_dict = states_loader.split_data(
        states_dict=state_dfs, sequence_len=hyperparameters.sequence_length
    )

    # Create batches
    input_train_batches, target_train_batches, fitted_scaler = (
        states_loader.preprocess_train_data_batches(
            states_train_data_dict=train_data_dict,
            hyperparameters=hyperparameters,
            features=features,
        )
    )

    # Set fitted scaler
    local_model.set_scaler(scaler=fitted_scaler)

    local_model.train_model(
        batch_inputs=input_train_batches,
        batch_targets=target_train_batches,
        display_nth_epoch=1,
        loss_function=nn.HuberLoss(),
    )

    return local_model


def create_finetunable_model(features: List[str], state: str):

    local_model = create_local_model(features=features)
    BASE_HYPERPARAMETERS = local_model.hyperparameters

    # Maybe finetune model
    finetunable_hyperparameters = LSTMHyperparameters(
        input_size=BASE_HYPERPARAMETERS.input_size,
        hidden_size=256,
        sequence_length=BASE_HYPERPARAMETERS.sequence_length,
        learning_rate=0.001,
        epochs=50,
        batch_size=1,  # Do not change this if you do not want to experience segfault
        num_layers=1,
    )

    finetunable_model = FineTunableLSTM(
        base_model=local_model, hyperparameters=finetunable_hyperparameters
    )

    # Preprocess state data
    state_loader = StateDataLoader(state=state)

    czech_data = state_loader.load_data()

    train_df, test_df = state_loader.split_data(data=czech_data, split_rate=0.8)

    train_input_batches, target_input_batches, _ = (
        state_loader.preprocess_training_data_batches(
            train_data_df=train_df,
            hyperparameters=finetunable_hyperparameters,
            features=local_model.FEATURES,
            scaler=local_model.SCALER,
        )
    )

    finetunable_model.train_model(
        batch_inputs=train_input_batches,
        batch_targets=target_input_batches,
        display_nth_epoch=1,
        loss_function=nn.HuberLoss(),
    )

    return finetunable_model


def predictor_v1(targets: List[str]) -> DemographyPredictor:

    # Train global model
    ## Load data
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    ## Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    ## Targets
    ALL_TARGETS = [
        "population, total",
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    # Features
    EXCLUDE_FEATURES = []

    EXCLUDE_FOR_LOCAL = ["year", "country name"]

    GLOBAL_FEATURES: List[str] = [
        col
        for col in whole_dataset_df.columns
        if col not in ALL_TARGETS
        and col not in EXCLUDE_FEATURES  # Exclude targets and some features
    ]

    LOCAL_FEATURES = [
        feature for feature in GLOBAL_FEATURES if feature not in EXCLUDE_FOR_LOCAL
    ]

    # Maybe use get model instead of manual training
    local_model = create_local_model(features=LOCAL_FEATURES)
    # local_model = create_finetunable_model(features=LOCAL_FEATURES, state="Czechia")

    # Define global model
    global_model: GlobalModel = GlobalModel(
        model=XGBRegressor(),
        features=GLOBAL_FEATURES,
        targets=targets,
        scaler=MinMaxScaler(),
    )

    X_train, X_test, y_train, y_test = global_model.create_train_test_data(
        data=whole_dataset_df[GLOBAL_FEATURES + targets],
        split_size=0.8,
    )

    global_model.train(X_train=X_train, y_train=y_train)

    # Create predictor
    predictor = DemographyPredictor(
        "predictor_v1", local_model=local_model, global_model=global_model
    )

    return predictor


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Create predictor
    targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]
    # targets: List[str] = ["population, total"]

    pred = predictor_v1(targets=targets)

    # Save model
    STATE = "Czechia"
    save_model(model=pred, name=f"aging_{STATE}.pkl")

    # Print predictor
    print(pred)

    # Czech data
    state_loader = StateDataLoader(state=STATE)

    czech_data = state_loader.load_data()

    train_df, test_df = state_loader.split_data(data=czech_data, split_rate=0.8)
    target_year = test_df["year"].max()

    # Validation data
    evaluation = EvaluateModel(model=pred)

    # Get data from every state
    states_data_loader = StatesDataLoader()
    all_states_df = states_data_loader.load_all_states()

    train_dict, test_dict = states_data_loader.split_data(
        states_dict=all_states_df,
        sequence_len=pred.local_model.hyperparameters.sequence_length,
    )

    evaluation.eval_for_every_state(
        X_test_states=train_dict,
        y_test_states=test_dict,
    )

    print(evaluation.all_states_evaluation)
    exit()

    YEARS = test_df["year"].values

    test_y = test_df[targets]

    print(test_y.head())

    # Get last year of predictions
    last_year = train_df["year"].max()

    predictions_df = pred.predict(input_data=train_df, target_year=target_year)

    print(predictions_df.head())

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=len(targets), ncols=1, figsize=(10, 8))

    if len(targets) == 1:
        axes = [axes]

    for i, target in enumerate(targets):
        axes[i].plot(YEARS, test_y[target], label="Reference data")
        axes[i].plot(YEARS, predictions_df[target], label="Predicted data")

        axes[i].set_title(target)
        axes[i].legend()
        axes[i].grid()

    fig.tight_layout()
    fig.savefig(fname="test.png")

    predictions_df = pred.predict(input_data=train_df, target_year=2050)

    print(predictions_df)
