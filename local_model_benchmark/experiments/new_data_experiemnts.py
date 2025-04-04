# Standard library imports
import logging
import pandas as pd
from typing import List

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

    BASE_LSTM_HYPERPARAMETERS: LSTMHyperparameters = get_core_parameters()
    FEATURES: List[str] = []

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
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS, features=self.FEATURES
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
            hyperparameters=self.BASE_LSTM_HYPERPARAMETERS, features=self.FEATURES
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

    def run(self, state: str, state_group: List[str], split_rate: float):

        # TODO: figure out what to do with this readme notes etc.
        # Create readme
        self.create_readme()
        # self.readme_add_params()
        # self.readme_add_features()

        # Train model using one state
        trained_on_single_state_data = self.__train_by_single_state(
            state=state,
            split_rate=split_rate,
            display_nth_epoch=1,
        )

        # Train model using group of states
        # Get one loader for multiple states to save memmory
        states_loader = StatesDataLoader()
        trained_on_group_of_states = self.__train_group_of_states(
            states=state_group, states_loader=states_loader, split_rate=split_rate
        )

        # Train model using  all states data
        trained_on_all_states = self.__train_all_states(
            states_loader=states_loader, split_rate=split_rate
        )

        # Compare model performance
        single_state_model_evaluation = EvaluateModel(
            model=trained_on_single_state_data
        )
        group_state_model_evaluation = EvaluateModel(model=trained_on_group_of_states)
        all_states_model_evaluation = EvaluateModel(model=trained_on_all_states)

        # Prepare evaluation data
        eval_state_loader = StateDataLoader(state=state)
        eval_state_df = eval_state_loader.load_data()
        test_X, test_y = eval_state_loader.split_data(
            data=eval_state_df, split_rate=split_rate
        )

        # Evaluate models - overall performance
        single_state_eval_df = single_state_model_evaluation.eval(
            test_X=test_X, test_y=test_y
        )
        group_state_eval_df = group_state_model_evaluation.eval(
            test_X=test_X, test_y=test_y
        )
        all_states_eval_df = all_states_model_evaluation.eval(
            test_X=test_X, test_y=test_y
        )

        # Evaluate models - per-target-performance
        single_state_model_evaluation.eval_per_target(test_X=test_X, test_y=test_y)
        group_state_model_evaluation.eval_per_target(test_X=test_X, test_y=test_y)
        all_states_model_evaluation.eval_per_target(test_X=test_X, test_y=test_y)
