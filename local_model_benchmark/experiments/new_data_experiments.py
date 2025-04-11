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
from src.compare_models.compare import ModelComparator

from local_model_benchmark.experiments.base_experiment import BaseExperiment
from src.train_scripts.train_local_models import (
    train_base_lstm,
    train_finetunable_model,
    train_finetunable_model_from_scratch,
)
from src.pipeline import LocalModelPipeline
from src.local_model.model import LSTMHyperparameters, BaseLSTM
from src.evaluation import EvaluateModel

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.data_transformer import DataTransformer

settings = LocalModelBenchmarkSettings()
logger = logging.getLogger("benchmark")


# TODO:
# More experiments
# 1. Feature selection experiment?
# 2. Compare group experiment using this?
# 3. To exclude states? ... find out which population or what makes the problem.


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
        input_size=len(FEATURES), batch_size=16, epochs=10
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_by_single_state(
        self,
        state: str,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        state_data = states_loader.load_states(states=[state])

        base_model_pipeline = train_base_lstm(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=state_data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_group_of_states(
        self,
        states: List[str],
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_states(states=states)

        base_model_pipeline = train_base_lstm(
            hyperparameters=self.MULTISTATE_BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_all_states(
        self,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_lstm(
            hyperparameters=self.MULTISTATE_BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def run(self, state: str, state_group: List[str], split_rate: float = 0.8):

        # TODO:
        # 1. figure out what to do with this readme notes etc.
        # 2. plot predictions?
        # Create readme
        self.create_readme()

        COMPARATION_MODELS_DICT: Dict[str, BaseLSTM] = {}
        TRANSFORMERS_MODELS_DICT: Dict[str, DataTransformer] = {}

        # Create model trained on a single state
        single_state_model_pipeline = self.__train_by_single_state(
            state=state,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using one state
        COMPARATION_MODELS_DICT["single_state_model"] = (
            single_state_model_pipeline.model
        )
        TRANSFORMERS_MODELS_DICT["single_state_model"] = (
            single_state_model_pipeline.transformer
        )

        # Train model using group of states
        # Get one loader for multiple states to save memmory
        group_states_model_pipeline = self.__train_group_of_states(
            states=state_group,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        COMPARATION_MODELS_DICT["group_states_model"] = (
            group_states_model_pipeline.model
        )
        TRANSFORMERS_MODELS_DICT["group_states_model"] = (
            group_states_model_pipeline.transformer
        )

        # Train model using  all states data
        all_states_model_pipeline = self.__train_all_states(
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        COMPARATION_MODELS_DICT["all_states_model"] = all_states_model_pipeline.model
        TRANSFORMERS_MODELS_DICT["all_states_model"] = (
            all_states_model_pipeline.transformer
        )

        # Compare models
        comparator = ModelComparator()
        # Evaluate models - per-target-performance
        per_target_metrics_df = comparator.compare_models_by_states(
            models=COMPARATION_MODELS_DICT,
            transformers=TRANSFORMERS_MODELS_DICT,
            states=[state],
            by="per-features",
        )
        overall_metrics_df = comparator.compare_models_by_states(
            models=COMPARATION_MODELS_DICT,
            transformers=TRANSFORMERS_MODELS_DICT,
            states=[state],
            by="overall-metrics",
        )

        comparaison_plots = comparator.create_comparision_plots()

        # Save and display the state plot
        self.save_plot(
            fig_name="state_prediction_comparions.png", figure=comparaison_plots[state]
        )

        self.readme_add_plot(
            plot_name="## Model comparision prediction plot",
            plot_description="In the next feagure you can see each model predictions compared to each other and the reference data.",
            fig_name="state_prediction_comparions.png",
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
