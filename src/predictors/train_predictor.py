# Standard library imports
import logging
import pandas as pd
from typing import List, Literal, Union, Dict
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import save_model
from src.predictors.predictor_base import (
    DemographyPredictor,
    create_local_model,
    create_finetunable_model,
)
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel

from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.state_preprocessing import StateDataLoader

# Get logger and settings
logger = logging.getLogger("demography_predictor")
settings = Config()


# Note: this constatns may be configurable in the config.py file
# Get all possible target features to exclude them from the input
ALL_POSSIBLE_TARGETS = settings.ALL_POSSIBLE_TARGET_FEATURES

# Features which are excluded for also for local and global model
EXCLUDE_FEATURES = []

# No need to predict the future values for these features
EXCLUDE_FOR_LOCAL = ["year", "country name"]


# TODO:
# Create predictor with BaseLSTM
# Create predictor with FineTunableLSTM for One state
# Create predictor with FineTunableLSTM for a group of states


def predictor_base_lstm(targets: List[str]) -> DemographyPredictor:
    """
    Demography predictor using BaseLSTM as the local model and XGBRegressor as global model

    Args:
        targets (List[str]): The list of target features.

    Returns:
        out: DemographyPredictor: The created model for predicting target demographic parameter(s).
    """

    # Train local model
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    # Get global features
    GLOBAL_FEATURES: List[str] = [
        col
        for col in whole_dataset_df.columns
        if col not in ALL_POSSIBLE_TARGETS
        and col not in EXCLUDE_FEATURES  # Exclude targets and some features
    ]

    # Get local features
    LOCAL_FEATURES = [
        feature for feature in GLOBAL_FEATURES if feature not in EXCLUDE_FOR_LOCAL
    ]

    # Create or load base model
    base_model_hyperparameters = LSTMHyperparameters(
        input_size=len(LOCAL_FEATURES),
        hidden_size=512,
        sequence_length=10,
        learning_rate=0.0001,
        epochs=30,
        batch_size=16,  # Do not change this if you do not want to experience segfault
        num_layers=3,
    )

    # Maybe use get model instead of manual training
    local_model = create_local_model(
        features=LOCAL_FEATURES, hyperparameters=base_model_hyperparameters
    )

    # Define global model
    global_model: GlobalModel = GlobalModel(
        model=XGBRegressor(),
        features=GLOBAL_FEATURES,
        targets=targets,
        scaler=MinMaxScaler(),
    )

    # X_train, X_test, y_train, y_test = global_model.create_train_test_timeseries(
    #     data=whole_dataset_df[GLOBAL_FEATURES + targets],
    #     split_size=0.8,
    # )

    X_train, X_test, y_train, y_test = global_model.create_train_test_timeseries(
        states_dfs=state_dfs,
        states_loader=states_loader,
        split_size=0.8,
    )

    global_model.train(X_train=X_train, y_train=y_train)

    # Create predictor
    predictor = DemographyPredictor(
        "predictor_v1", local_model=local_model, global_model=global_model
    )

    return predictor


def predictor_finetuned(
    targets: List[str], finetuning_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
) -> DemographyPredictor:

    # Train local model
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    # Targets
    ALL_TARGETS = settings.ALL_POSSIBLE_TARGET_FEATURES

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

    # Create or load base model
    base_model_hyperparameters = LSTMHyperparameters(
        input_size=len(LOCAL_FEATURES),
        hidden_size=512,
        sequence_length=10,
        learning_rate=0.0001,
        epochs=30,
        batch_size=16,  # Do not change this if you do not want to experience segfault
        num_layers=3,
    )

    # Maybe use get model instead of manual training
    local_model = create_local_model(
        features=LOCAL_FEATURES, hyperparameters=base_model_hyperparameters
    )

    # Get finetunable model model
    finetunable_hyperparameters = LSTMHyperparameters(
        input_size=base_model_hyperparameters.input_size,
        hidden_size=256,
        sequence_length=base_model_hyperparameters.sequence_length,
        learning_rate=0.0001,
        epochs=50,
        batch_size=1,  # Do not change this if you do not want to experience segfault
        num_layers=1,
    )

    finetunable_local_model = create_finetunable_model(
        base_model=local_model,
        hyperparameters=finetunable_hyperparameters,
        finetuning_data=finetuning_data,
    )

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
        "predictor_v2", local_model=finetunable_local_model, global_model=global_model
    )

    return predictor


def train(
    model_name: str,
    model_type: Literal["base", "finetunable"],
    targets: List[str],
    finetuning_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]] | None = None,
) -> DemographyPredictor:
    """
    Trains and saves the predictor by the given parameters.

    Args:
        model_name (str): Name which is used to save model.
        model_type (Literal[&quot;base&quot;, &quot;finetunable-single-state&quot;, &quot;inetunable-state-group&quot;): _description_
        targets (List[str]): List of target features

    Raises:
        ValueError: If the prediction model has value 'None'.

    Returns:
        out: DemographyPredictor: The created model for predicting target demographic parameter(s).
    """

    # Setup logging
    setup_logging()

    # Select predictor
    pred = None
    if "base" == model_type:
        pred = predictor_base_lstm(targets=targets)
    elif "finetunable" == model_type:
        if finetuning_data is None:
            raise ValueError(
                f"Model is type is set to '{model_type}', but no data for finetuning provided!"
            )
        pred = predictor_finetuned(targets=targets, finetuning_data=finetuning_data)

    # Log predictor
    logger.info(f"Trained predictor: {pred}")

    if pred is None:
        raise ValueError("The dmography prediction model cannot have the value 'None'.")

    # Adjust model name and save model
    model_name = model_name if model_name.endswith(".pkl") else f"{model_name}.pkl"
    save_model(model=pred, name=model_name)

    return pred


def evalulate_for_single_state(
    predictor: DemographyPredictor, eval_state_name: str
) -> None:
    # Czech data
    state_loader = StateDataLoader(state=eval_state_name)

    czech_data = state_loader.load_data()

    train_df, test_df = state_loader.split_data(data=czech_data, split_rate=0.8)

    # Evaluate model
    evaluation = EvaluateModel(model=predictor)
    evaluation.eval(test_X=train_df, test_y=test_df)

    # Print the metrics
    print(evaluation.overall_metrics)


def evalulate_for_every_state(predictor: DemographyPredictor) -> None:
    states_data_loader = StatesDataLoader()
    all_states_df = states_data_loader.load_all_states()

    train_dict, test_dict = states_data_loader.split_data(
        states_dict=all_states_df,
        sequence_len=predictor.local_model.hyperparameters.sequence_length,
    )

    # Evaluate model
    evaluation = EvaluateModel(model=predictor)
    evaluation.eval_for_every_state(
        X_test_states=train_dict,
        y_test_states=test_dict,
    )

    print(evaluation.all_states_evaluation)


# def predict_and_plot(predictor: DemographyPredictor, test_X: pd.DataFrame, test_y) -> None:
#     test_y = test_df[targets]


#     print(test_y.head())

#     # Get last year of predictions
#     last_year = train_df["year"].max()
#     YEARS = range(last_year)

#     predictions_df = pred.predict(input_data=train_df, target_year=target_year)

#     print(predictions_df.head())

#     import matplotlib.pyplot as plt

#     fig, axes = plt.subplots(nrows=len(targets), ncols=1, figsize=(10, 8))

#     if len(targets) == 1:
#         axes = [axes]

#     for i, target in enumerate(targets):
#         axes[i].plot(YEARS, test_y[target], label="Reference data")
#         axes[i].plot(YEARS, predictions_df[target], label="Predicted data")

#         axes[i].set_title(target)
#         axes[i].legend()
#         axes[i].grid()

#     fig.tight_layout()
#     fig.savefig(fname="test.png")

#     predictions_df = pred.predict(input_data=train_df, target_year=2050)

#     print(predictions_df)


if __name__ == "__main__":
    # TODO: set this up to cli?

    # Setup logging
    setup_logging()

    # Create predictor - more details are displayed in the functions predictor_vX, where X is int number
    aging_targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]
    gender_targets: List[str] = ["population, female", "population, male"]
    population_total_targets: List[str] = [
        "population, total"
    ]  # Predicting this is really useless using this method

    # Get the finetuning data
    states_loader = StatesDataLoader()

    # Define the group of states
    STATES_GROUP: List[str] = [
        "Australia",
        "Austria",
        "Bahamas, The",
        "Bahrain",
        "Belgium",
        "Brunei Darussalam",
        "Canada",
        "Cyprus",
        "Czechia",
        "Denmark",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Hong Kong SAR, China",
        "Iceland",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Korea, Rep.",
        "Kuwait",
        "Latvia",
        "Lithuania",
        "Luxembourg",
        "Malta",
        "Netherlands",
        "New Zealand",
        "Norway",
        "Oman",
        "Poland",
        "Portugal",
        "Qatar",
        "Saudi Arabia",
        # "Singapore",
        "Slovak Republic",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
    ]
    group_states_dict = states_loader.load_states(states=STATES_GROUP)

    # Single state finetuning
    state_loader = StateDataLoader(state="Czechia")
    state_data_df = state_loader.load_data()

    # Train the model
    pred = train(
        model_name="aging_Czechia_model.pkl",
        model_type="finetunable",
        targets=aging_targets,
        finetuning_data=state_data_df,
    )

    # Evaluate the model
    evalulate_for_every_state(predictor=pred)
