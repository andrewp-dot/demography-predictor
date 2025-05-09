# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import logging
import pandas as pd
from typing import List, Dict, Tuple
import random

from sklearn.model_selection import train_test_split

import pprint

# Custom imports
from src.utils.log import setup_logging

from src.utils.constants import (
    get_core_hyperparameters,
    basic_features,
    highly_correlated_features,
)

from src.compare_models.compare import ModelComparator

from src.state_groups import StatesGroups, StatesByGeolocation, StatesByWealth

from model_experiments.base_experiment import BaseExperiment
from src.train_scripts.train_feature_models import (
    train_base_rnn,
)
from src.pipeline import FeatureModelPipeline
from src.feature_model.model import RNNHyperparameters, BaseRNN

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("benchmark")


class DataUsedForTraining(BaseExperiment):
    """
    Trains models using dfferent data:
    - Model trained using specific state data.
    - Model trained using group of states (e.g. by wealth) data.
    - Model trained using all available states data.

    Models are compared by evaluation on the specific (chosen) state data used for training for the first model.
    """

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES)
    )

    MULTISTATE_BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES), batch_size=16, epochs=10
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_by_single_state(
        self,
        name: str,
        state: str,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        state_data = states_loader.load_states(states=[state])

        base_model_pipeline = train_base_rnn(
            name=name,
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=state_data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_group_of_states(
        self,
        name: str,
        states: List[str],
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_states(states=states)

        base_model_pipeline = train_base_rnn(
            name=name,
            hyperparameters=self.MULTISTATE_BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_all_states(
        self,
        name: str,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        # Load data
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_rnn(
            name=name,
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

        COMPARATION_MODELS_DICT: Dict[str, BaseRNN] = {}

        # Create model trained on a single state
        single_state_model_pipeline = self.__train_by_single_state(
            name="single-model",
            state=state,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using one state
        COMPARATION_MODELS_DICT["single_state_model"] = single_state_model_pipeline

        # Train model using group of states
        # Get one loader for multiple states to save memmory
        group_states_model_pipeline = self.__train_group_of_states(
            name="group-model",
            states=state_group,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        COMPARATION_MODELS_DICT["group_states_model"] = group_states_model_pipeline

        # Train model using  all states data
        all_states_model_pipeline = self.__train_all_states(
            name="all-states-model",
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
            plot_name="Model comparision prediction plot",
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


# TODO:
# 1. Create expeirment for the groups (or to find the best subset of data)
# 2. Subset of data validaion:
#   # 1. Create a subset of data
#   # 2. From the subset of data create training and testing data (randomly select some % of given states from the group, and valiedate the model on the rest of the data)


class StatesByGroup(BaseExperiment):
    """
    Experiment for the states by wealth.
    """

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        batch_size=16,
    )

    def __init__(self, description: str, name: str = ""):
        super().__init__(
            name=f"{self.__class__.__name__}_{name}", description=description
        )

    def __train_group_of_states(
        self,
        states: List[str],
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> Tuple[FeatureModelPipeline, List[str]]:
        """
        Trains model using the group of states. Returns the model and the states which was used for training.

        Args:
            states (List[str]): The list of states which you want to train the model on.
            split_rate (float): Split rate for data training and validation.
            display_nth_epoch (int, optional): Display every nth epoch. Defaults to 10.

        Returns:
            out: Tuple[FeatureModelPipeline, List[str]]: model, training_states_list
        """

        # Load data
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_states(states=states)

        all_states: List[str] = list(states_data_dict.keys())

        # Filter states with too low records
        to_remove_states: List[str] = []
        for state, df in states_data_dict.items():
            if len(df) <= self.BASE_LSTM_HYPERPARAMETERS.sequence_length:
                logger.warning(
                    f"State {state} has too low records ({len(df)}). Removing from training data."
                )
                to_remove_states.append(state)

        for state in to_remove_states:
            del states_data_dict[state]
            all_states.remove(state)

        base_model_pipeline = train_base_rnn(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline, all_states

    def __train_models_for_each_group(
        self, state_groups: StatesGroups, split_rate: float = 0.8
    ) -> Dict[str, Tuple[FeatureModelPipeline, List[str]]]:
        """
        Trains model for each group of states.

        Args:
            state_groups (StatesGroups): The state groups to train the models on.
            split_rate (float, optional): Split rate of data for training and validation. Defaults to 0.8.

        Returns:
            out: Dict[str, Tuple[FeatureModelPipeline, List[str]]]: The key is the group name and the value is a tuple of the model and the states which will be used for validation.
        """

        # Load data and setup variables
        group_states_dict: Dict[str, List[str]] = state_groups.model_dump()

        GROUP_MODEL_VALIDATION_DATA: Dict[
            str, Tuple[FeatureModelPipeline, List[str]]
        ] = {}

        # Train model for each group
        for group, states in group_states_dict.items():

            group_model, training_states = self.__train_group_of_states(
                states=states,
                split_rate=0.8,
            )

            train_states, test_states = train_test_split(
                training_states, test_size=(1 - split_rate), random_state=42
            )

            GROUP_MODEL_VALIDATION_DATA[group] = (group_model, test_states)

        return GROUP_MODEL_VALIDATION_DATA

    def run(self, state_groups: StatesGroups, split_rate: float = 0.8):

        # Create readme
        self.create_readme()

        # Train base model
        states_loader = StatesDataLoader()
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_rnn(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Create group models vs base model tuples

        GROUP_MODELS: Dict[str, Tuple[FeatureModelPipeline, List[str]]] = (
            self.__train_models_for_each_group(state_groups=state_groups)
        )

        COMPARATION_MODELS_DICT: Dict[str, FeatureModelPipeline] = {}

        for group, (group_model, validation_states) in GROUP_MODELS.items():

            logger.info(f"Comparing the group: {group} ... ")
            # Compare models
            comparator = ModelComparator()

            self.readme_add_section(text="", title=f"## {group} model")

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

            # Choose random state for the comparision
            comparison_plots = comparator.create_comparision_plots()

            random_state_index: int = random.randint(0, len(validation_states) - 1)
            random_state: str = validation_states[random_state_index]

            # Save and display the state plot
            FIG_NAME: str = f"{group}_{random_state}_prediction_comparions.png".replace(
                " ", "_"
            ).replace(",", "")
            self.save_plot(
                fig_name=FIG_NAME,
                figure=comparison_plots[random_state],
            )

            self.readme_add_plot(
                plot_name="Model comparision prediction plot",
                plot_description="In the next feagure you can see each model predictions compared to each other and the reference data.",
                fig_name=FIG_NAME,
            )

            # Print results to the readme
            self.readme_add_section(
                title="## Per target metrics - model comparision",
                text=f"```\n{per_target_metrics_df.sort_values(by='state')}\n```\n\n",
            )

            self.readme_add_section(
                title="## Overall metrics - model comparision",
                text=f"```\n{overall_metrics_df.sort_values(by='state')}\n```\n\n",
            )

            # Reset the dictionary
            COMPARATION_MODELS_DICT = {}


# Try this on whole pipeline?


# TODO: Make this work for all pipelines -> use better evaluation method
class FeatureSelectionExperiment(BaseExperiment):

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    BASE_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        len(FEATURES),
        batch_size=32,
        hidden_size=256,
        num_layers=2,
    )

    def __init__(self, description: str):
        super().__init__(self.__class__.__name__, description)

    # TODO:
    # Maybe be optimized removing more then 1 feature at the time, or random selection for combination of features, iteration limits etc.
    def output_bad_performing_features(
        self,
        all_features: List[str],
        min_features: int = 5,
        batch_size: int = 32,
        epochs: int = 10,
    ) -> List[str]:
        """
        Returns excluded features from the model.

        Args:
            all_features (List[str]): List of all available features.
            min_features (int, optional): Minimum number of selected features. Defaults to 5.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            epochs (int, optional): _description_. Defaults to 10.


        Returns:
            out: List[str]: Features which appears to be bad performing.
        """

        bad_features = []

        while len(all_features) > min_features:
            rank_features = {}

            self.readme_add_section(
                text=f"```\n{pprint.pformat(all_features)}\n```\n\n",
                title="# All features",
            )

            for excluded_feature in all_features:
                current_features = [f for f in all_features if f != excluded_feature]

                # Train model
                hyperparams = get_core_hyperparameters(
                    input_size=len(current_features),
                    batch_size=batch_size,
                    epochs=epochs,
                )
                data = StatesDataLoader().load_all_states()

                pipeline = train_base_rnn(
                    name="base-lstm",
                    hyperparameters=hyperparams,
                    data=data,
                    features=current_features,
                    split_rate=0.8,
                )

                val_loss = pipeline.training_stats.validation_loss[-1]
                rank_features[excluded_feature] = val_loss

                # Save the combination of features

            rank_features = dict(
                sorted(rank_features.items(), key=lambda item: item[1])
            )

            self.readme_add_section(
                text=f"```\n{pprint.pformat(rank_features, sort_dicts=False)}\n```\n\n",
                title="## Validation loss per excluded feature",
            )

            logger.info(f"Ranked features: {rank_features}")

            # Get the best performance of the model with the worst feature excluded
            worst_feature = min(rank_features, key=rank_features.get)
            all_features.remove(worst_feature)
            bad_features.append(worst_feature)

        return bad_features

    def run(self):

        # Create readme
        self.create_readme()

        bad_performing_features = self.output_bad_performing_features(
            all_features=self.FEATURES
        )

        # Format the bad performing features
        bad_performing_features_formated = pprint.pformat(bad_performing_features)
        self.readme_add_section(
            text=f"```\n{bad_performing_features_formated}\n```\n\n",
            title="## Bad performing features",
        )


class StatesSubsetExperiment(BaseExperiment):

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES)
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    GEOLOCATION_GROUPS: Dict[str, List[str]] = {
        "europe": StatesByGeolocation().europe,
        "asia": StatesByGeolocation().asia,
        "africa": StatesByGeolocation().africa,
        "north_america": StatesByGeolocation().north_america,
        "south_america": StatesByGeolocation().south_america,
        "oceania": StatesByGeolocation().oceania,
    }

    WEALTH_GROUPS: Dict[str, List[str]] = {
        "high_income": StatesByWealth().high_income,
        "upper_middle_income": StatesByWealth().upper_middle_income,
        "lower_middle_income": StatesByWealth().lower_middle_income,
        "low_income": StatesByWealth().low_income,
    }

    HIDDEN_SIZE_TO_TRY: List[int] = [32, 64, 128, 256, 512]

    def __train_models_for_group(
        self,
        group_name: str,
        group_data: Dict[str, pd.DataFrame],
        split_rate: float = 0.8,
    ):

        all_states_dict = group_data

        # Train models with different
        TO_COMPARE_MODELS: Dict[str, FeatureModelPipeline] = {}
        for hidden_size in self.HIDDEN_SIZE_TO_TRY:

            MODEL_NAME = f"lstm-{hidden_size}"
            TO_COMPARE_MODELS[MODEL_NAME] = train_base_rnn(
                name=MODEL_NAME,
                features=self.FEATURES,
                hyperparameters=get_core_hyperparameters(
                    input_size=len(self.FEATURES),
                    batch_size=16,
                    hidden_size=hidden_size,
                ),
                data=all_states_dict,
                split_rate=split_rate,
                display_nth_epoch=10,
            )

        comparator = ModelComparator()

        # Evaluate on the group states
        EVAL_STATES = list(all_states_dict.keys())

        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=EVAL_STATES, by="overall-one-metric"
        )

        self.readme_add_section(
            title=f"## Overall metrics for {group_name}  - model comparision",
            text=f"```\n{overall_metrics_df}\n```\n\n",
        )

    def run(
        self,
        split_rate: float = 0.8,
    ):
        # Create readme
        self.create_readme()

        loader = StatesDataLoader()

        ALL_GROUPS: Dict[str, List[str]] = {
            **self.WEALTH_GROUPS,
            **self.GEOLOCATION_GROUPS,
        }

        # For every group tran the models and see which models you can use for which groups
        for group_name, states in ALL_GROUPS.items():

            states = loader.load_states(states=states)

            # Trains and evaluates the model
            self.__train_models_for_group(
                group_name=group_name, group_data=states, split_rate=split_rate
            )


def main():

    # Setup logging
    setup_logging()

    exp_data = DataUsedForTraining(
        description="Trains base LSTM models using data in 3 categories: single state data, group of states (e.g. by wealth divided states) and with all available states data.",
    )
    exp_wealth_groups = StatesByGroup(
        name="wealth", description="States by the given state groups."
    )
    exp_geolocation_groups = StatesByGroup(
        name="geolocation", description="States by the given state groups."
    )

    exp_feature_selection = FeatureSelectionExperiment(
        description="Feature selection experiment. It will output the features which are not performing well."
    )

    exp_states_subset = StatesSubsetExperiment(
        description="Trains the model from the simpliest to more complex ones by hidden size for each group and reveals which models performs best for each group."
    )

    STATE: str = "Czechia"
    GROUPS_BY_WEALTH = StatesByWealth()
    GROUPS_BY_GEOLOCATION = StatesByGeolocation()

    SELECTED_GROUP: List[str] = GROUPS_BY_WEALTH.get_states_corresponding_group(
        state=STATE
    )

    # Get the group of the selected state
    # exp_data.run(state=STATE, state_group=SELECTED_GROUP, split_rate=0.8)

    # exp_wealth_groups.run(
    #     state_groups=GROUPS_BY_WEALTH,
    #     split_rate=0.8,
    # )

    # exp_geolocation_groups.run(
    #     state_groups=GROUPS_BY_GEOLOCATION,
    #     split_rate=0.8,
    # )

    exp_feature_selection.run()

    # exp_states_subset.run()


if __name__ == "__main__":
    main()
