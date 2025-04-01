# Standard library imports
from torch import nn
import pandas as pd
import logging
from typing import Union, List, Dict

# Custom imports
from src.local_model.base import LSTMHyperparameters
from src.local_model.model import BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.global_model.model import GlobalModel
from src.local_model.statistical_models import LocalARIMA

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.state_preprocessing import StateDataLoader

# Get pre-configured logger
logger = logging.getLogger("demography_predictor")


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

        self.FEATURES: str = global_model.FEATURES
        self.TARGETS: str = global_model.TARGETS

    def __repr__(self) -> str:
        return f"{self.name} ({type(self.local_model).__name__} -> {type(self.global_model.model).__name__})"

    def predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        # Preprocess input data -> scale data
        data = input_data.copy()

        # Features
        LOCAL_FEAUTRES = self.local_model.FEATURES

        # Predict features
        feature_predictions_df = self.local_model.predict(
            input_data=data[LOCAL_FEAUTRES],
            last_year=last_year,
            target_year=target_year,
        )

        # Add missing features
        GLOBAL_FEATURES = self.global_model.FEATURES

        adjust_for_gm_features_df = feature_predictions_df.copy()
        # Add country name
        adjust_for_gm_features_df["country name"] = data["country name"][0]
        adjust_for_gm_features_df["country name"] = adjust_for_gm_features_df[
            "country name"
        ].astype("category")

        # Add year
        adjust_for_gm_features_df["year"] = pd.Series(
            list(range(last_year, target_year + 1))
        )

        if set(adjust_for_gm_features_df.columns) != set(GLOBAL_FEATURES):
            raise ValueError(
                f"Missing features for global model: {set(adjust_for_gm_features_df.columns) ^ set(GLOBAL_FEATURES)}"
            )

        preprocessed_for_gm = self.global_model.preprocess_data(
            data=adjust_for_gm_features_df
        )

        # Predict target variable using global model
        final_predictions_df = self.global_model.predict_human_readable(
            data=preprocessed_for_gm[GLOBAL_FEATURES]
        )

        # Add years to final predictions
        PREDICTED_YEARS = list(range(last_year + 1, target_year + 1))
        predicted_years_df = pd.DataFrame({"years": PREDICTED_YEARS})

        years_final_predictions_df = pd.concat(
            [predicted_years_df, final_predictions_df], axis=1
        )

        return years_final_predictions_df


def create_local_model(features: List[str], hyperparameters: LSTMHyperparameters):

    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()

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


def create_finetunable_model(
    base_model: BaseLSTM,
    hyperparameters: LSTMHyperparameters,
    finetuning_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
):

    # Create finetunable model
    finetunable_model = FineTunableLSTM(
        base_model=base_model, hyperparameters=hyperparameters
    )

    # TODO: add data for finetuning..

    # By the type preprocess data
    # If the finetuning data are in the format of state: datframe pair -> maybe use just isinstance(dict)
    if isinstance(finetuning_data, dict) and all(
        isinstance(k, str) and isinstance(v, pd.DataFrame)
        for k, v in finetuning_data.items()
    ):
        states_loader = StatesDataLoader()
        training_data = states_loader.load_states(states=list(finetuning_data.keys()))

        train_input_batches, target_input_batches, _ = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=training_data,
                hyperparameters=hyperparameters,
                features=base_model.FEATURES,
            )
        )

    elif isinstance(finetuning_data, pd.DataFrame):

        # Get state
        if not ("country name" in finetuning_data.columns):
            raise ValueError(
                "No country name provided in a single state data finetuning."
            )

        if finetuning_data["country name"].nunique() != 1:
            raise ValueError(
                "There are multiple countries in the dataframe. For state group finetuning please use Dict[str, pd.DataFrame] format."
            )

        # Preprocess state data
        state_loader = StateDataLoader(
            state=""
        )  # Do not need to load the data here, this instance is used just for data preprocessing

        train_df, test_df = state_loader.split_data(
            data=finetuning_data, split_rate=0.8
        )

        train_input_batches, target_input_batches, _ = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_df,
                hyperparameters=hyperparameters,
                features=base_model.FEATURES,
                scaler=base_model.SCALER,
            )
        )

    # Finetune model based on given data
    finetunable_model.train_model(
        batch_inputs=train_input_batches,
        batch_targets=target_input_batches,
        display_nth_epoch=1,
        loss_function=nn.HuberLoss(),
    )

    return finetunable_model
