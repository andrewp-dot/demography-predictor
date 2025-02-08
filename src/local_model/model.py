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
        - batch_size: the number of samples processed in one forward/backward pass
        - time_steps or sequence_length: number of time steps
        - input_features: number of input features

        :param data: pd.DataFrame
        :return: pd.DataFrame
        """
        raise NotImplementedError("preprocess_data method is not implemented yet.")


class LocalModel(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(LocalModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

    def train(self):
        raise NotImplementedError("train method is not implemented yet.")


if __name__ == "__main__":
    pass
