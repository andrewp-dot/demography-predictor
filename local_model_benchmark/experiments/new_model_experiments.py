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
from src.compare_models.compare import compare_models_by_states

from local_model_benchmark.experiments.base_experiment import BaseExperiment
from src.local_model.model import LSTMHyperparameters, BaseLSTM

from src.evaluation import EvaluateModel
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import (
    PureEnsembleModel,
    train_models_for_ensemble_model,
)

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


settings = LocalModelBenchmarkSettings()
logger = logging.getLogger("benchmark")


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
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> BaseLSTM:

        model_all_states = BaseLSTM(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
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

    def __train_ensemble_model(
        self, split_rate: float, display_nth_epoch=10
    ) -> PureEnsembleModel:

        feature_models = train_models_for_ensemble_model(
            features=self.FEATURES,
            hyperaparameters=self.BASE_LSTM_HYPERPARAMETERS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
        )

        ensemble_model = PureEnsembleModel(feature_models=feature_models)

        return ensemble_model

    def run(self, state: str, split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[BaseLSTM, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_lstm_model(
            states_loader=states_loader, split_rate=split_rate, display_nth_epoch=1
        )

        # Train ensemble model
        TO_COMPARE_MODELS["ensemble-model"] = self.__train_ensemble_model(
            split_rate=split_rate, display_nth_epoch=1
        )

        # Evaluate models - per-target-performance
        per_target_metrics_df = compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = compare_models_by_states(
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
        states_loader: StatesDataLoader,
        split_rate: float,
        display_nth_epoch: int = 10,
    ) -> BaseLSTM:

        model_all_states = BaseLSTM(
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS,
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

    def __train_finetuned_lstm_model(
        self,
        base_model: BaseLSTM,
        states_loader: StatesDataLoader,
        states: List[str],
        split_rate: float = 0.8,
        display_nth_epoch: int = 10,
    ) -> FineTunableLSTM:

        # Load states data and get training data
        states_data_dict = states_loader.load_states(states=states)

        train_dict, _ = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=base_model.hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        train_batches, target_batches, scaler = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=train_dict,
                hyperparameters=self.FINETUNE_MODELS_HYPERPARAMETERS,
                features=base_model.FEATURES,
            )
        )

        # Create and train finetunable model
        finetunable_model = FineTunableLSTM(
            base_model=base_model, hyperparameters=self.FINETUNE_MODELS_HYPERPARAMETERS
        )

        # Set scaler for sure
        finetunable_model.set_scaler(scaler=scaler)

        finetunable_model.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=display_nth_epoch,
        )

        return finetunable_model

    def run(self, state: str, state_group: List[str], split_rate: float = 0.8):
        # Create readme
        self.create_readme()

        TO_COMPARE_MODELS: Dict[str, Union[BaseLSTM, PureEnsembleModel]] = {}

        # Get data loader
        states_loader: StatesDataLoader = StatesDataLoader()

        # Train base lstm
        TO_COMPARE_MODELS["base-lstm"] = self.__train_base_lstm_model(
            states_loader=states_loader, split_rate=split_rate, display_nth_epoch=1
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
        per_target_metrics_df = compare_models_by_states(
            models=TO_COMPARE_MODELS, states=[state], by="per-features"
        )
        overall_metrics_df = compare_models_by_states(
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

    # exp_1 = FeaturePredictionSeparatelyVSAtOnce(
    #     description="Compares single LSTM model vs LSTM for every feature."
    # )
    # exp_1.run(state="Czechia", split_rate=0.8)

    exp_2 = FineTunedModels(
        description="See if finetuning the model helps the model to be more accurate."
    )

    exp_2.run(state="Czechia", state_group=RICH, split_rate=0.8)
