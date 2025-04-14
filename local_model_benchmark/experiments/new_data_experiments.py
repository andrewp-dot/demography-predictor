# Standard library imports
import logging
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Custom imports
from src.utils.log import setup_logging
from local_model_benchmark.config import (
    LocalModelBenchmarkSettings,
    get_core_parameters,
)

# from src.utils.save_model import save_experiment_model, get_experiment_model
from src.base import CustomModelBase
from src.compare_models.compare import ModelComparator

from src.state_groups import StatesGroups, StatesByGeolocation, StatesByWealth

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


# Load the dataset (replace 'your_dataset.csv' with the actual file path)
HIGHLY_CORRELATED_COLUMNS: List[str] = [
    "life expectancy at birth, total",
    "age dependency ratio",
    "rural population",
    "birth rate, crude",
    "adolescent fertility rate",
]


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
            # "year",
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
        # if col.lower() not in HIGHLY_CORRELATED_COLUMNS
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

        # Create model trained on a single state
        single_state_model_pipeline = self.__train_by_single_state(
            state=state,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using one state
        COMPARATION_MODELS_DICT["single_state_model"] = single_state_model_pipeline

        # Train model using group of states
        # Get one loader for multiple states to save memmory
        group_states_model_pipeline = self.__train_group_of_states(
            states=state_group,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        COMPARATION_MODELS_DICT["group_states_model"] = group_states_model_pipeline

        # Train model using  all states data
        all_states_model_pipeline = self.__train_all_states(
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        COMPARATION_MODELS_DICT["all_states_model"] = all_states_model_pipeline

        # Compare models
        comparator = ModelComparator()
        # Evaluate models - per-target-performance
        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=COMPARATION_MODELS_DICT,
            states=[state],
            by="per-features",
        )
        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=COMPARATION_MODELS_DICT,
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


class FeatureSelectionExperiment(BaseExperiment):
    """
    Feature selection experiment.
    """

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self):
        pass


# TODO:
# 1. Create expeirment for the groups (or to find the best subset of data)
# 2. Subset of data validaion:
#   # 1. Create a subset of data
#   # 2. From the subset of data create training and testing data (randomly select some % of given states from the group, and valiedate the model on the rest of the data)


class StatesByGroup(BaseExperiment):
    """
    Experiment for the states by wealth.
    """

    FEATURES: List[str] = [
        col.lower()
        for col in [
            # "year",
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
        if col.lower() not in HIGHLY_CORRELATED_COLUMNS
    ]

    BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES),
        batch_size=16,
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

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
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_models_for_each_group(
        self, state_groups: StatesGroups, split_rate: float = 0.8
    ) -> Dict[str, Tuple[LocalModelPipeline, List[str]]]:
        """
        Trains model for each group of states.

        Args:
            state_groups (StatesGroups): The state groups to train the models on.
            split_rate (float, optional): Split rate of data for training and validation. Defaults to 0.8.

        Returns:
            out: Dict[str, Tuple[LocalModelPipeline, List[str]]]: The key is the group name and the value is a tuple of the model and the states which will be used for validation.
        """

        # Load data and setup variables
        group_states_dict: Dict[str, List[str]] = state_groups.model_dump()

        GROUP_MODEL_VALIDATION_DATA: Dict[str, Tuple[LocalModelPipeline, List[str]]] = (
            {}
        )

        for group, states in group_states_dict.items():
            # Train model for each group

            train_states, test_states = train_test_split(
                states, test_size=(1 - split_rate), random_state=42
            )

            group_model = self.__train_group_of_states(
                states=train_states,
                split_rate=0.8,
            )

            GROUP_MODEL_VALIDATION_DATA[group] = (group_model, test_states)

        return GROUP_MODEL_VALIDATION_DATA

    def run(self, state_groups: StatesGroups, split_rate: float = 0.8):

        # Create readme
        self.create_readme()

        # Train base model
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_lstm(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Create group models vs base model tuples

        GROUP_MODELS: Dict[str, Tuple[LocalModelPipeline, List[str]]] = (
            self.__train_models_for_each_group(state_groups=state_groups)
        )

        # Compare models
        comparator = ModelComparator()

        COMPARATION_MODELS_DICT: Dict[str, LocalModelPipeline] = {}

        for group, (group_model, validation_states) in GROUP_MODELS.items():
            # Add group model to the comparision
            COMPARATION_MODELS_DICT[group] = group_model
            COMPARATION_MODELS_DICT["base_model"] = base_model_pipeline

            # Evaluate models - per-target-performance
            per_target_metrics_df = comparator.compare_models_by_states(
                pipelines=COMPARATION_MODELS_DICT,
                states=validation_states,
                by="per-features",
            )
            overall_metrics_df = comparator.compare_models_by_states(
                pipelines=COMPARATION_MODELS_DICT,
                states=validation_states,
                by="overall-metrics",
            )


def main():
    # Setup logging
    setup_logging()

    exp = DataUsedForTraining(
        description="Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.",
    )

    STATE: str = "Czechia"
    GROUPS_BY_WEALTH = StatesByWealth()
    GROUPS_BY_GEOLOCATION = StatesByGeolocation()

    SELECTED_GROUP: List[str] = GROUPS_BY_WEALTH.get_states_corresponding_group(
        state=STATE
    )

    # Get the group of the selected state
    exp.run(state=STATE, state_group=SELECTED_GROUP, split_rate=0.8)


if __name__ == "__main__":
    main()
