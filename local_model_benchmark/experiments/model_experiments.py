# Standard library imports
import logging
import pandas as pd
from typing import List, Dict, Union

from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from local_model_benchmark.config import (
    LocalModelBenchmarkSettings,
    get_core_parameters,
)

# from src.utils.save_model import save_experiment_model, get_experiment_model
from src.base import CustomModelBase
from src.state_groups import StatesByGeolocation, StatesByWealth

from src.pipeline import LocalModelPipeline
from src.train_scripts.train_local_models import (
    train_base_lstm,
    train_finetunable_model,
    train_ensemble_model,
    train_arima_ensemble_model,
)

from src.compare_models.compare import ModelComparator

from local_model_benchmark.experiments.base_experiment import BaseExperiment
from src.local_model.model import LSTMHyperparameters, BaseLSTM

from src.evaluation import EvaluateModel
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import (
    PureEnsembleModel,
    # train_models_for_ensemble_model,
    train_arima_models_for_ensemble_model,
)

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


settings = LocalModelBenchmarkSettings()
logger = logging.getLogger("benchmark")


# TODO:
# make this work again -> make this compatible with pipelines?


class FeaturePredictionSeparatelyVSAtOnce(BaseExperiment):
    """
    Compares performance of 2 models:
    - BaseLSTM (input_size=N_FEATURES)
    - Multiple BaseLSTM (input_size=1, number of models=N_FEATURES)

    Models are compared by evaluation on the specific (chosen) states data.
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
        input_size=len(FEATURES), batch_size=16, hidden_size=256
    )

    ENSEMBLE_MODELS_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=1, batch_size=16, hidden_size=64
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_base_lstm_model(
        self,
        name: str,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        # Preprocess data
        states_data_dict = states_loader.load_all_states()

        # The train_base_lstm function splits the data automatically
        lstm_pipeline = train_base_lstm(
            name=name,
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            features=self.FEATURES,
            data=states_data_dict,
            display_nth_epoch=display_nth_epoch,
            split_rate=split_rate,
        )

        return lstm_pipeline

    def __train_ensemble_model(
        self, split_rate: float, display_nth_epoch=10
    ) -> PureEnsembleModel:

        loader = StatesDataLoader()
        all_states_dict = loader.load_all_states()

        ensemble_model = train_ensemble_model(
            name="ensemble-model",
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=all_states_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return ensemble_model

    def run(self, state: str, split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[BaseLSTM, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_lstm_model(
            name="base-lstm",
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train ensemble model
        TO_COMPARE_MODELS["ensemble-model"] = self.__train_ensemble_model(
            split_rate=split_rate, display_nth_epoch=1
        )

        # Evaluate models - per-target-performance
        comparator = ModelComparator()
        per_target_metrics_df = comparator.compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="overall-metrics"
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


class FineTunedModels(BaseExperiment):
    """
    Experiment compares following models:
    - BaseLSTM
    - FineTunableLSTM - single state
    - FineTunableLSTM - group of states
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
        input_size=len(FEATURES), batch_size=16, hidden_size=256
    )

    FINETUNE_MODELS_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES),
        hidden_size=124,
        batch_size=1,
        num_layers=1,
        epochs=BASE_LSTM_HYPERPARAMETERS.epochs * 3,
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_base_lstm_model(
        self,
        name: str,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        states_data_dict = states_loader.load_all_states()

        train_states_dict, _ = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=self.BASE_LSTM_HYPERPARAMETERS.sequence_length,
            split_rate=split_rate,
        )

        lstm_pipeline = train_base_lstm(
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
        base_pipeline: LocalModelPipeline,
        states_loader: StatesDataLoader,
        states: List[str],
        split_rate: float = 0.8,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

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

        TO_COMPARE_MODELS: Dict[str, Union[BaseLSTM, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_lstm_model(
            name="base-lstm",
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Finetune base lstm using single state
        TO_COMPARE_MODELS["single-state-finetuned"] = self.__train_finetuned_lstm_model(
            base_model=TO_COMPARE_MODELS["base-lstm"],
            states_loader=states_loader,
            states=[state],
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Finetune base lstm using group of states state
        TO_COMPARE_MODELS["group-states-finetuned"] = self.__train_finetuned_lstm_model(
            base_model=TO_COMPARE_MODELS["base-lstm"],
            states=state_group,
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Evaluate models - per-target-performance
        comparator = ModelComparator()
        per_target_metrics_df = comparator.compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="overall-metrics"
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


class CompareWithStatisticalModels(BaseExperiment):
    """
    In this experiment the statistical models (ARIMA(1,1,1) and GM(1,1) models) are compared with BaseLSTM model.
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
        input_size=len(FEATURES), batch_size=16, hidden_size=256
    )

    FINETUNE_MODELS_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES),
        hidden_size=124,
        batch_size=1,
        num_layers=1,
        epochs=BASE_LSTM_HYPERPARAMETERS.epochs * 3,
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __train_base_lstm_model(
        self,
        name: str,
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> LocalModelPipeline:

        # Preprocess data
        states_data_dict = states_loader.load_all_states()

        base_model_pipeline = train_base_lstm(
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
    ) -> PureEnsembleModel:

        loader = StatesDataLoader()
        all_states_dict = loader.load_all_states()

        # Train ARIMA instead
        ensemble_model = train_ensemble_model(
            name="ensemble-model",
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
            data=all_states_dict,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        return ensemble_model

    def __train_arima_ensemble_model(
        self, split_rate: float, state: str
    ) -> PureEnsembleModel:
        ensemble_model = train_arima_ensemble_model(
            features=self.FEATURES, state=state, split_rate=split_rate
        )

        return ensemble_model

    def run(self, state: str, split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[LocalModelPipeline, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_lstm_model(
            name="base-lstm",
            features=self.FEATURES,
            states_loader=states_loader,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        TO_COMPARE_MODELS["ensemble-lstm"] = self.__train_ensemble_model(
            split_rate=split_rate, display_nth_epoch=1
        )

        # TODO: statistical models in here

        # TODO: train ARIMA
        TO_COMPARE_MODELS["ensemble-arima"] = self.__train_arima_ensemble_model(
            split_rate=split_rate, state=state
        )

        # TODO: train GM model

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


class DifferentHiddenLayers(BaseExperiment):

    FEATURES = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            # "population, total",
            # "Net migration",
            "Arable land",
            # "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            # "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
        ]
    ]

    HIDDEN_SIZE_TO_TRY: List[int] = [32, 64, 128, 256, 512]

    BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES), batch_size=16
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self, split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        EVALUATION_STATES: List[str] = ["Czechia", "United States", "Honduras"]

        # Load data
        loader = StatesDataLoader()
        all_states_dict = loader.load_all_states()

        # Train models with different
        TO_COMPARE_MODELS: Dict[str, LocalModelPipeline] = {}
        for hidden_size in self.HIDDEN_SIZE_TO_TRY:

            MODEL_NAME = f"lstm-{hidden_size}"
            TO_COMPARE_MODELS[MODEL_NAME] = train_base_lstm(
                name=MODEL_NAME,
                features=self.FEATURES,
                hyperparameters=get_core_parameters(
                    input_size=len(self.FEATURES),
                    batch_size=16,
                    hidden_size=hidden_size,
                ),
                data=all_states_dict,
                split_rate=split_rate,
                display_nth_epoch=10,
            )

        comparator = ModelComparator()

        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=EVALUATION_STATES, by="per-features"
        )
        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_MODELS, states=EVALUATION_STATES, by="overall-metrics"
        )

        # Print results to the readme
        self.readme_add_section(
            title="## Per target metrics - model comparision",
            text=f"```\n{per_target_metrics_df.sort_values(by=['state'])}\n```\n\n",
        )

        self.readme_add_section(
            title="## Overall metrics - model comparision",
            text=f"```\n{overall_metrics_df.sort_values(by=['state'])}\n```\n\n",
        )

        # Print honduras plot
        fig = comparator.create_state_comparison_plot(state="Honduras")

        self.save_plot(fig_name="honduras_predictions.png", figure=fig)
        self.readme_add_plot(
            plot_name="Honduras predictions",
            plot_description="Honduras feature predictions.",
            fig_name="honduras_predictions.png",
        )


class DifferentArchitecturesComparision(BaseExperiment):

    FEATURES = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            # "population, total",
            "Net migration",
            "Arable land",
            # "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            # "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
        ]
    ]

    BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters(
        input_size=len(FEATURES), batch_size=16
    )

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)
        raise NotImplementedError(
            "Comparision using different architecture not implemented yet."
        )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # States divided to this categories by GPT
    STATE: str = "Czechia"
    SELECTED_GROUP: List[str] = StatesByWealth().get_states_corresponding_group(
        state=STATE
    )

    failing_experiments = []

    # TODO: Try this if it is runnable
    # try:
    exp_1 = FeaturePredictionSeparatelyVSAtOnce(
        description="Compares single LSTM model vs LSTM for every feature."
    )
    exp_1.run(state="Czechia", split_rate=0.8)
    # except Exception as e:
    #     logger.error(f"exp 1: {e}")
    #     failing_experiments.append("exp ")
    # TODO: Try this if it is runnable

    exit()

    try:
        exp_2 = FineTunedModels(
            description="See if finetuning the model helps the model to be more accurate."
        )

        exp_2.run(state="Czechia", state_group=SELECTED_GROUP, split_rate=0.8)
    except Exception as e:
        logger.error(f"exp 2: {e}")
        failing_experiments.append("exp 2")

    # TODO: Try this if it is runnable

    try:
        exp_3 = CompareWithStatisticalModels(
            description="Compares BaseLSTM with statistical models and BaseLSTM for single feature prediction."
        )
        exp_3.run(state="Czechia", split_rate=0.8)
    except Exception as e:
        logger.error(f"exp 3: {e}")
        failing_experiments.append("exp 3")

    # Runnable
    # exp_4 = DifferentHiddenLayers(
    #     description="Try to train BaseLSTM models with different layers."
    # )
    # exp_4.run(split_rate=0.8)

    print(failing_experiments)
