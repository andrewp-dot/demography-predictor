# Standard libraries imports
import pandas as pd
from typing import Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Custom libraries imports
from config import Config


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
