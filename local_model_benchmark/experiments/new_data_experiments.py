# Standard library imports
import logging
import pandas as pd
from typing import List, Dict

from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from local_model_benchmark.config import (
    LocalModelBenchmarkSettings,
    get_core_parameters,
)

# from src.utils.save_model import save_experiment_model, get_experiment_model
from src.base import CustomModelBase
from src.compare_models.compare import compare_models_by_states

from local_model_benchmark.experiments.base_experiment import BaseExperiment
from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.evaluation import EvaluateModel

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


settings = LocalModelBenchmarkSettings()
logger = logging.getLogger("benchmark")


class DataUsedForTraining(BaseExperiment):
    """
    Trains models using dfferent data:
    - Model trained using specific state data.
    - Model trained using group of states (e.g. by wealth) data.
    - Model trained using all available states data.

    Models are compared by evaluation on the specific (chosen) state data used for training for the first model.
    """

    FEATURES: List[str] = [
        col.lower()
        for col in [
            "year",
            "Fertility rate, total",
            # "Population, total",
            "Net migration",
            "Arable land",
            "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            "Rural population",
            "Rural population growth",
            "Age dependency ratio",
            "Urban population",
            "Population growth",
            "Adolescent fertility rate",
            "Life expectancy at birth, total",
        ]
    ]

    BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES)
    )

    MULTISTATE_BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES), batch_size=16
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_by_single_state(
        self,
        state: str,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> BaseLSTM:

        single_state_loader = StateDataLoader(state=state)
        model_single_state = BaseLSTM(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS, features=self.FEATURES
        )

        # Preprocess data
        state_data = single_state_loader.load_data()

        train_df, _ = single_state_loader.split_data(
            data=state_data, split_rate=split_rate
        )
        train_batches, target_batches, scaler = (
            single_state_loader.preprocess_training_data_batches(
                train_data_df=train_df,
                hyperparameters=model_single_state.hyperparameters,
                features=model_single_state.FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Set used scaler
        model_single_state.set_scaler(scaler=scaler)

        model_single_state.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=display_nth_epoch,
        )

        return model_single_state

    def __train_group_of_states(
        self,
        states: List[str],
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> CustomModelBase:

        model_group_states = BaseLSTM(
            hyperparameters=self.MULTISTATE_BASE_LSTM_HYPERPARAMETERS,
            features=self.FEATURES,
        )

        # Preprocess data
        states_data_dict = states_loader.load_states(states=states)

        train_states_dict, _ = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=model_group_states.hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        train_batches, target_batches, scaler = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=train_states_dict,
                hyperparameters=model_group_states.hyperparameters,
                features=model_group_states.FEATURES,
            )
        )

        # Set fitted scaler
        model_group_states.set_scaler(scaler=scaler)

        model_group_states.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=display_nth_epoch,
        )
        return model_group_states

    def __train_all_states(
        self,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> CustomModelBase:

        model_all_states = BaseLSTM(
            hyperparameters=self.MULTISTATE_BASE_LSTM_HYPERPARAMETERS,
            features=self.FEATURES,
        )

        # Preprocess data
        states_data_dict = states_loader.load_all_states()

        states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=model_all_states.hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        train_states_dict, _ = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=model_all_states.hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        train_batches, target_batches, scaler = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=train_states_dict,
                hyperparameters=model_all_states.hyperparameters,
                features=model_all_states.FEATURES,
            )
        )

        # Set fitted scaler
        model_all_states.set_scaler(scaler=scaler)

        model_all_states.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=display_nth_epoch,
        )
        return model_all_states

    def run(self, state: str, state_group: List[str], split_rate: float = 0.8):

        # TODO:
        # 1. figure out what to do with this readme notes etc.
        # 2. plot predictions?
        # Create readme
        self.create_readme()

        COMPARATION_MODELS_DICT: Dict[str, BaseLSTM] = {}

        # Train model using one state
        COMPARATION_MODELS_DICT["single_state_model"] = self.__train_by_single_state(
            state=state,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using group of states
        # Get one loader for multiple states to save memmory
        states_loader = StatesDataLoader()
        COMPARATION_MODELS_DICT["group_states_model"] = self.__train_group_of_states(
            states=state_group,
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using  all states data
        COMPARATION_MODELS_DICT["all_states_model"] = self.__train_all_states(
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Evaluate models - per-target-performance
        per_target_metrics_df = compare_models_by_states(
            models=COMPARATION_MODELS_DICT, states=[state], by="per-features"
        )
        overall_metrics_df = compare_models_by_states(
            models=COMPARATION_MODELS_DICT, states=[state], by="overall-metrics"
        )

        # Print results to the readme
        self.readme_add_section(
            title="## Per target metrics - model comparision",
            text=f"```\n{per_target_metrics_df}\n```\n\n",
        )

        self.readme_add_section(
            title="## Overall metrics - model comparision",
            text=f"```\n{overall_metrics_df}\n```\n\n",
        )


if __name__ == "__main__":

    # Setup logging
    setup_logging()

    # States divided to this categories by GPT
    RICH: List[str] = [
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
        "Singapore",
        "Slovak Republic",
        "Slovenia",
        "Spain",
        "Sweden",
        "Switzerland",
        "United Arab Emirates",
        "United Kingdom",
        "United States",
    ]

    exp = DataUsedForTraining(
        description="Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.",
    )

    exp.run(state="Czechia", state_group=RICH)
