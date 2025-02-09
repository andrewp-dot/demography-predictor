# Standart library imports
from typing import Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import pandas as pd
import torch
from torch import nn

# Custom modules imports
from config import Config

# Set the seed for reproducibility
torch.manual_seed(42)

# TODO:
# 0. Implement the dataset class -> scale data
# 1. Implement the model
# 2. Implement the training loop
# 3. Implement the evaluation loop
# 4. Implement the prediction loop
# 5. Implement the save and load functions


class StateDataLoader:

    def __init__(self, state: str):
        self.state = state

    def load_data(self) -> pd.DataFrame:
        """
        Load the state data
        :return: pd.DataFrame
        """
        data = pd.read_csv(f"{Config.states_data_dir}/{self.state}.csv")
        return data

    def scale_data(
        self,
        data: pd.DataFrame,
        scaler: Union[RobustScaler, StandardScaler, MinMaxScaler],
    ) -> pd.DataFrame:
        """Scales the data using the specified scaler.

        Args:
            data (pd.DataFrame): input data
            scaler (Union[RobustScaler, StandardScaler, MinMaxScaler]): scaler to use

        Returns:
            pd.DataFrame: scaled data
        """
        raise NotImplementedError("scale_data method is not implemented yet.")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Scales and transforms the data for the specified format (3D tensor): `(batch_size,time_steps,input_features)`, where:
        - batch_size: the number of samples processed in one forward/backward pass (how many samples the network sees before it updates itself)
        - time_steps or sequence_length: number of time steps
        - input_features: number of input features

        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        raise NotImplementedError("preprocess_data method is not implemented yet.")


class LSTMHyperparameters:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
    ):
        self.input_size = input_size  # number of input features
        self.hidden_size = hidden_size  # number of hidden units in the LSTM layer
        self.sequence_length = sequence_length  # length of the input sequence, i.e., how many time steps the model will look back
        self.learning_rate = (
            learning_rate  # how much the model is learning from the data
        )
        self.epochs = epochs  # number of times the entire dataset is passed forward and backward through the neural network
        self.batch_size = (
            batch_size  # number of samples processed in one forward/backward pass
        )


class LocalModel(nn.Module):

    def __init__(self, hyperparameters: LSTMHyperparameters):
        super(LocalModel, self).__init__()

        self.hyperparameters: LSTMHyperparameters = hyperparameters

        # 3 layer model:
        # 1. LSTM layer
        lstm1 = nn.LSTMCell(
            input_size=hyperparameters.input_size,
            hidden_size=hyperparameters.hidden_size,
        )

        # 2. LSTM layer
        lstm2 = nn.LSTMCell(
            input_size=hyperparameters.hidden_size,
            hidden_size=hyperparameters.hidden_size,
        )

        # 3. LSTM layer
        lstm3 = nn.LSTMCell(
            input_size=hyperparameters.hidden_size,
            hidden_size=hyperparameters.hidden_size,
        )

        # 4. Linear layer - output layer
        linear = nn.Linear(
            in_features=hyperparameters.hidden_size,
            out_features=hyperparameters.input_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # define outputs
        outputs = []
        n_samples = x.size(0)  # or batch_size

        raise NotImplementedError("forward method is not implemented yet.")

    def train(self):
        raise NotImplementedError("train method is not implemented yet.")


if __name__ == "__main__":
    pass
