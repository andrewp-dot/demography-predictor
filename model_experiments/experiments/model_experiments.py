# Standard library imports
import logging
import pandas as pd
from typing import List, Dict, Union, Type

from torch import nn
from itertools import product

# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    aging_targets,
)

from model_experiments.config import (
    FeatureModelBenchmarkSettings,
)

from src.utils.constants import get_core_hyperparameters

# from src.utils.save_model import save_experiment_model, get_experiment_model
from src.state_groups import StatesByGeolocation, StatesByWealth

from src.pipeline import FeatureModelPipeline
from src.train_scripts.train_feature_models import (
    train_base_rnn,
    train_finetunable_model,
    train_finetunable_model_from_scratch,
    train_ensemble_model,
    train_arima_ensemble_model,
)

from src.compare_models.compare import ModelComparator

from model_experiments.base_experiment import BaseExperiment
from src.feature_model.model import RNNHyperparameters, BaseRNN

from src.feature_model.ensemble_model import (
    PureEnsembleModel,
    # train_models_for_ensemble_model,
    # train_arima_models_for_ensemble_model,
)

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


settings = FeatureModelBenchmarkSettings()
logger = logging.getLogger("benchmark")


# TODO: Fix ensemble model
class FineTunedModels(BaseExperiment):
    """
    Experiment compares following models:
    - BaseRNN
    - FineTunableLSTM - single state
    - FineTunableLSTM - group of states
    """

    FEATURES: List[str] = basic_features()

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        batch_size=16,
        hidden_size=256,
    )

    FINETUNE_MODELS_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        hidden_size=124,
        batch_size=1,
        num_layers=1,
        epochs=BASE_LSTM_HYPERPARAMETERS.epochs * 3,
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_base_rnn_model(
        self,
        name: str,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        states_data_dict = states_loader.load_all_states()

        train_states_dict, _ = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=self.BASE_LSTM_HYPERPARAMETERS.sequence_length,
            split_rate=split_rate,
        )

        lstm_pipeline = train_base_rnn(
            name=name,
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            features=self.FEATURES,
            data=train_states_dict,
            display_nth_epoch=display_nth_epoch,
        )

        return lstm_pipeline

    def __train_finetuned_lstm_model(
        self,
        name: str,
        base_pipeline: FeatureModelPipeline,
        states_loader: StatesDataLoader,
        states: List[str],
        split_rate: float = 0.8,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        # Load states data and get training data
        states_data_dict = states_loader.load_states(states=states)

        fintunable_pipeline = train_finetunable_model(
            name=name,
            base_model_pipeline=base_pipeline,
            finetunable_model_hyperparameters=self.FINETUNE_MODELS_HYPERPARAMETERS,
            finetunable_model_data=states_data_dict,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return fintunable_pipeline

    def run(self, state: str, state_group: List[str], split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[BaseRNN, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_rnn_model(
            name="base-lstm",
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Finetune base lstm using single state
        TO_COMPARE_MODELS["single-state-finetuned"] = self.__train_finetuned_lstm_model(
            name="single-state-finetuned",
            base_pipeline=TO_COMPARE_MODELS["base-lstm"],
            states_loader=states_loader,
            states=[state],
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Finetune base lstm using group of states state
        TO_COMPARE_MODELS["group-states-finetuned"] = self.__train_finetuned_lstm_model(
            name="group-states-finetuned",
            base_pipeline=TO_COMPARE_MODELS["base-lstm"],
            states=state_group,
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Evaluate models - per-target-performance
        comparator = ModelComparator()
        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=[state], by="overall-metrics"
        )

        # Print results to the readme
        self.readme_add_section(
            title="## Per target metrics - model comparision",
            text=f"```\n{per_target_metrics_df.sort_values(by=['state','target'])}\n```\n\n",
        )

        self.readme_add_section(
            title="## Overall metrics - model comparision",
            text=f"```\n{overall_metrics_df}\n```\n\n",
        )


class CompareWithStatisticalModels(BaseExperiment):
    """
    In this experiment the statistical models (ARIMA(1,1,1) and GM(1,1) models) are compared with BaseRNN model.
    """

    FEATURES: List[str] = [""]

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        batch_size=16,
        hidden_size=256,
        epochs=30,
        num_layers=2,
    )

    FINETUNE_MODELS_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES),
        hidden_size=124,
        batch_size=1,
        num_layers=1,
        epochs=BASE_LSTM_HYPERPARAMETERS.epochs * 3,
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_base_rnn_model(
        self,
        name: str,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> FeatureModelPipeline:

        # Preprocess data
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_rnn(
            name=name,
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=states_data_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return base_model_pipeline

    def __train_ensemble_model(
        self, split_rate: float, display_nth_epoch=10
    ) -> FeatureModelPipeline:

        loader = StatesDataLoader()
        all_states_dict = loader.load_all_states()

        # Train ARIMA instead
        pipeline = train_ensemble_model(
            name="ensemble-model",
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=all_states_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return pipeline

    def __train_arima_ensemble_model(
        self, name: str, split_rate: float, state: str
    ) -> PureEnsembleModel:
        ensemble_model = train_arima_ensemble_model(
            name=name,
            features=self.FEATURES,
            state=state,
            split_rate=split_rate,
        )

        return ensemble_model

    def run(self, state: str, split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[FeatureModelPipeline, PureEnsembleModel]] = (
            {}
        )

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_rnn_model(
            name="base-lstm",
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        TO_COMPARE_MODELS["ensemble-lstm"] = self.__train_ensemble_model(
            split_rate=split_rate, display_nth_epoch=1
        )

        TO_COMPARE_MODELS["ensemble-arima"] = self.__train_arima_ensemble_model(
            name="ensemble-arima", split_rate=split_rate, state=state
        )

        # Evaluate models - per-target-performance
        comparator = ModelComparator()
        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=[state], by="overall-metrics"
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


class DifferentHiddenAndNumOfLayers(BaseExperiment):

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    HIDDEN_SIZE_TO_TRY: List[int] = [
        32,
        64,
        128,
        256,
        512,
    ]

    NUMBER_OF_LAYERS: List[int] = [1, 2, 3]

    MODEL_TYPES: Dict[str, Type] = {
        "GRU": nn.GRU,
        "RNN": nn.RNN,
        "LSTM": nn.LSTM,
    }

    BASE_LSTM_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES), batch_size=16
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self, split_rate: float = 0.8, evaluation_states: List[str] = None):
        # Create readme
        self.create_readme()

        # Load data
        loader = StatesDataLoader()
        all_states_dict = loader.load_all_states()

        # Train models with different
        TO_COMPARE_MODELS: Dict[str, FeatureModelPipeline] = {}
        for (name, rnn_type), num_layers, hidden_size in product(
            self.MODEL_TYPES.items(), self.NUMBER_OF_LAYERS, self.HIDDEN_SIZE_TO_TRY
        ):

            MODEL_NAME = f"{name}_{num_layers}_{hidden_size}"
            TO_COMPARE_MODELS[MODEL_NAME] = train_base_rnn(
                name=MODEL_NAME,
                features=self.FEATURES,
                hyperparameters=get_core_hyperparameters(
                    input_size=len(self.FEATURES),
                    batch_size=16,
                    hidden_size=hidden_size,
                    epochs=30,
                    num_layers=num_layers,
                ),
                data=all_states_dict,
                split_rate=split_rate,
                display_nth_epoch=5,
                rnn_type=rnn_type,
            )

        comparator = ModelComparator()

        EVALUATION_STATES = evaluation_states
        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=EVALUATION_STATES, by="per-targets"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS,
            states=EVALUATION_STATES,
            by="overall-one-metric",
        )

        # Print results to the readme
        # Add per metric rankings
        # Print all dataframe
        self.readme_add_section(
            title="## Per target metrics - model comparision",
            text=f"```\n{per_target_metrics_df.sort_values(by=['model', 'state', 'target'])}\n```\n\n",
        )

        # Extract the model type first
        overall_metrics_df["model_type"] = overall_metrics_df["model"].str.extract(
            r"^([A-Z]+)", expand=False
        )

        # Sort the df by the model type
        overall_metrics_df_sorted = overall_metrics_df.sort_values(
            by=["model_type", "mse"], ascending=[True, True]
        ).reset_index(drop=True)

        # Drop the model type
        overall_metrics_df_sorted.drop(columns=["model_type"], inplace=True)

        self.readme_add_section(
            title="## Overall metrics - model comparision",
            text=f"```\n{overall_metrics_df_sorted}\n```\n\n",
        )

        # Print best model hidden size:

        #
        # self.readme_add_section(
        #     title="## Best model hidden size for each type",
        #     text=f"```\n{overall_metrics_df.loc[overall_metrics_df.groupby('model')[BEST_BY_METRIC].idxmin()].reset_index(drop=True)}\n```\n\n",
        # )

        # Print honduras plot
        # fig = comparator.create_state_comparison_plot(state="Honduras")

        # self.save_plot(fig_name="honduras_predictions.png", figure=fig)
        # self.readme_add_plot(
        #     plot_name="Honduras predictions",
        #     plot_description="Honduras feature predictions.",
        #     fig_name="honduras_predictions.png",
        # )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # States divided to this categories by GPT
    STATE: str = "Czechia"
    SELECTED_GROUP: List[str] = StatesByWealth().get_states_corresponding_group(
        state=STATE
    )

    # exp_1 = FeaturePredictionSeparatelyVSAtOnce(
    #     description="Compares single LSTM model vs LSTM for every feature."
    # )
    # exp_1.run(state="Czechia", split_rate=0.8)

    # exp_2 = FineTunedModels(
    #     description="See if finetuning the model helps the model to be more accurate."
    # )

    # exp_2.run(state="Czechia", state_group=SELECTED_GROUP, split_rate=0.8)

    # 3 are nto runnable due to broken (incompatibile) PureEnsembleModel with pipeline creation.
    # exp_3 = CompareWithStatisticalModels(
    #     description="Compares BaseRNN with statistical models and BaseRNN for single feature prediction."
    # )
    # exp_3.run(state="Czechia", split_rate=0.8)

    # Runnable
    exp_4 = DifferentHiddenAndNumOfLayers(
        description="Try to train BaseRNN models with different layers.",
    )
    exp_4.run(split_rate=0.8)
    # exp_4.run(split_rate=0.8, evaluation_states=[STATE])
