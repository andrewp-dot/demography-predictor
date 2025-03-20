# Standard library imports
import pandas as pd
import logging
from typing import Union, List
from xgboost import XGBRegressor

# Custom imports
from src.utils.log import setup_logging
from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.state_preprocessing import StateDataLoader


# TODO: implement different models
# TODO: RNN + XBOOST? -> model v1
# TODO: ARIMA + XGBOOST -> model v2

logger = logging.getLogger("demograpy_predictor")


class DemographyPredictor:

    def __init__(
        self,
        name: str,
        local_model: Union[BaseLSTM, LocalARIMA],
        global_model: Union[GlobalModel],
    ):
        # Define name of the model -> name is used to versionning etc
        self.name: str = name

        # Define architecture: local model -> global model
        self.local_model: Union[BaseLSTM, LocalARIMA, FineTunableLSTM] = local_model
        self.global_model: Union[GlobalModel] = global_model

    def __repr__(self) -> str:
        return f"{self.name} ({type(self.local_model).__name__} -> {type(self.global_model.model).__name__})"

    def predict(self, input_data: pd.DataFrame, target_year: int):

        # Preprocess input data -> scale data
        data = input_data.copy()

        # Feattures
        FEAUTRES = self.local_model.FEATURES

        # Get scaled data
        scaled_data = self.local_model.scaler.transform(data[FEAUTRES])
        scaled_data_df = pd.DataFrame(scaled_data, columns=FEAUTRES, index=data.index)

        # Get last year of predictions
        last_year = input_data["year"].max()

        # Predict features
        feature_predictions = self.local_model.predict(
            input_data=scaled_data_df, last_year=last_year, target_year=target_year
        )

        feature_predictions_df = pd.DataFrame(feature_predictions, columns=FEAUTRES)

        # Predict target variable using global model
        final_predictions = self.global_model.predict_human_readable(
            data=feature_predictions_df
        )

        print(final_predictions)


def create_local_model(features: List[str]):

    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    # Define local model
    hyperparameters = LSTMHyperparameters(
        input_size=len(features),
        hidden_size=256,
        sequence_length=15,
        learning_rate=0.0001,
        epochs=30,
        batch_size=32,  # Do not change this if you do not want to experience segfault
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
        # display_nth_epoch=1,
    )

    return local_model


def predictor_v1() -> DemographyPredictor:

    # Train global model
    ## Load data
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

    ## Merge data to single dataframe
    whole_dataset_df = states_loader.merge_states(state_dfs=state_dfs)

    ## Targets
    ALL_TARGETS = [
        "population, total" "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]
    # targets: List[str] = ["population, total"]
    targets: List[str] = [
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
    ]

    # Features
    FEATURES: List[str] = [
        col for col in whole_dataset_df.columns if col not in ALL_TARGETS
    ]

    LOCAL_FEATURES = [feature for feature in FEATURES if feature != "country name"]

    local_model = create_local_model(features=LOCAL_FEATURES)

    # Define global model
    global_model: GlobalModel = GlobalModel(
        model=XGBRegressor(),
        features=FEATURES,
        targets=targets,
        scaler=local_model.scaler,
    )

    X_train, X_test, y_train, y_test = global_model.create_train_test_data(
        data=whole_dataset_df, split_size=0.8, fitted_scaler=local_model.scaler
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
    pred = predictor_v1()

    # Print predictor
    print(pred)

    # Czech data
    state_loader = StateDataLoader(state="Czechia")

    czech_data = state_loader.load_data()

    pred.predict(input_data=czech_data, target_year=2030)
