# Standard libraries imports
import pandas as pd
import torch
from typing import Union, Tuple
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Custom libraries imports
from config import Config


config = Config()


class StateDataLoader:

    def __init__(self, state: str):
        self.state = state

    def load_data(self) -> pd.DataFrame:
        """
        Load the state data
        :return: pd.DataFrame
        """
        data = pd.read_csv(f"{config.states_data_dir}/{self.state}.csv")

        # Lowercase columns
        data.columns = data.columns.str.lower()
        return data

    def scale_data(
        self,
        data: pd.DataFrame,
        scaler: Union[RobustScaler, StandardScaler, MinMaxScaler],
    ) -> pd.DataFrame:
        """
        Scales the data using the specified scaler.

        :param data: pd.DataFrame: input data
        :param scaler: Union[RobustScaler, StandardScaler, MinMaxScaler]: scaler to use

        Returns:
            pd.DataFrame: scaled data
        """
        raise NotImplementedError("scale_data method is not implemented yet.")

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train and test sets.

        :param data: pd.DataFrame
        :return: Tuple[pd.DataFrame, pd.DataFrame]: train and test data
        """

        # Sort data by year
        data.sort_values(by="year", inplace=True)

        # Split data
        split_rate = 0.8
        train_data = data.iloc[: int(split_rate * len(data))]
        test_data = data.iloc[int(split_rate * len(data)) :]

        return train_data, test_data

    def preprocess_data(
        self,
        data: pd.DataFrame,
        batch_size: int,
        sequence_len: int,
        features: list[str],
    ) -> pd.DataFrame:
        """
        Scales and transforms the data for the specified format (3D tensor): `(batch_size,time_steps,input_features)`, where:
        - batch_size: the number of samples processed in one forward/backward pass (how many samples the network sees before it updates itself)
        - time_steps or sequence_length: number of time steps
        - input_features: number of input features

        :param data: pd.DataFrame
        :return: pd.DataFrame
        """

        # Copy data to avoid modifying the original data
        current_data = data.copy()

        # Select features
        current_data = current_data[features]

        # Get data using rolling window
        input_sequences = []
        target_sequences = []
        number_of_samples = current_data.shape[0] - sequence_len
        for i in range(number_of_samples):

            # Get the input sequence
            input_sequences.append(
                # Converting to a PyTorch tensor
                torch.tensor(
                    current_data.iloc[i : i + sequence_len].values, dtype=torch.float32
                )
            )

            # Get tha target sequence output
            target_sequences.append(
                # Converting to a PyTorch tensor
                torch.tensor(current_data.iloc[i + sequence_len].values)
            )

        # TODO: use batch_size parameter

        # Return
        for input, target in zip(input_sequences, target_sequences):
            print(f"Input: {input}, Target: {target}")

        return input_sequences, target_sequences


if __name__ == "__main__":

    # Load czech data
    czech_data_loader = StateDataLoader("Czechia")
    czech_data = czech_data_loader.load_data()

    train, test = czech_data_loader.split_data(czech_data)

    print("-" * 100)
    print("Train data tail:")
    print(train.tail())

    print("-" * 100)
    print("Test data head:")
    print(test.head())

    # Preprocess data
    print("-" * 100)
    print("Preprocess data:")
    train = czech_data_loader.preprocess_data(
        train, 18, 5, ["year", "population, total"]
    )
