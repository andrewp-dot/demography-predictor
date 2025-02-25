# Standard libraries imports
import pandas as pd
import torch
from typing import Union, Tuple, List
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import logging

# Custom libraries imports
from config import Config, setup_logging

logger = logging.getLogger("data_preprocessing")

config = Config()


class StateDataLoader:

    def __init__(self, state: str):
        self.state = state

    def load_data(self) -> pd.DataFrame:
        """
        Load the state data. Renames the columns all to lower.

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
    ) -> Tuple[pd.DataFrame, Union[RobustScaler, StandardScaler, MinMaxScaler]]:
        """
        Scales the data using the specified scaler.

        :param data: pd.DataFrame: input data
        :param scaler: Union[RobustScaler, StandardScaler, MinMaxScaler]: scaler to use

        :return: Tuple[pd.DataFrame, Union[RobustScaler, StandardScaler, MinMaxScaler]]: scaled dataframe and fitted scaler for data unscaling
        """

        # Copy data to avoid inplace edits
        to_scale_data = data.copy()

        # Scale data
        scaled_data = scaler.fit_transform(to_scale_data)

        # Create dataframe from scaled data
        scaled_data_df = pd.DataFrame(scaled_data, columns=to_scale_data.columns)
        return scaled_data_df, scaler

    def split_data(
        self, data: pd.DataFrame, split_rate: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into train and test sets. Data are expected to be sorted by year in ascending order.

        :param data: pd.DataFrame
        :return: Tuple[pd.DataFrame, pd.DataFrame]: train and test data
        """

        # Sort data by year
        # data.sort_values(by="year", inplace=True)

        # Split data
        train_data = data.iloc[: int(split_rate * len(data))]
        test_data = data.iloc[int(split_rate * len(data)) :]

        return train_data, test_data

    def create_batches(
        self,
        batch_size: int,
        input_sequences: torch.Tensor,
        target_sequences: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Groups sequences to batches.

        :param batch_size: int: the number of samples processed in one forward/backward pass (how many samples the network sees before it updates itself)
        :param input_sequences: torch.Tensor: sequences to create a input batches
        :param target_sequences: torch.Tensor: sequences to create a target batches. Defaults to None.

        :return: Tuple[torch.Tensor, torch.Tensor]: input_batches, target_batches. If target_sequences are not specified returns (input_batches, None)
        """

        # Reshape
        input_batches = input_sequences.view(
            -1, batch_size, input_sequences.shape[1], input_sequences.shape[2]
        )  # (num_batches, batch_size, sequence_len, num_features)

        if target_sequences is not None:
            target_batches = target_sequences.view(
                -1, batch_size, target_sequences.shape[1]
            )  # (num_batches, batch_size, num_features)

            return input_batches, target_batches

        return input_batches, None

    def preprocess_training_data(
        self,
        data: pd.DataFrame,
        sequence_len: int,
        features: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the data for the specified format (3D tensor): `(batch_size,time_steps,input_features)`, where:
        - time_steps or sequence_length: number of time steps
        - input_features: number of input features

        :param data: pd.DataFrame: input data
        :param sequence_len: int: sequence length of the input/target sequences
        :param features: List[str]: list of features to use
        :return: Tuple[torch.Tensor, torch.Tensor]: input_sequences, target_sequences
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
                torch.tensor(
                    current_data.iloc[i + sequence_len].values, dtype=torch.float32
                )
            )

        # Return
        for input, target in zip(input_sequences, target_sequences):
            logger.debug(f"Input: {input.shape}, Target: {target.shape}")

        return torch.stack(input_sequences), torch.stack(target_sequences)

    def preprocess_data(
        self,
        data: pd.DataFrame,
        sequence_len: int,
        features: List[str],
    ) -> torch.Tensor:
        """
        Transforms the data for the specified format (3D tensor): `(batch_size,time_steps,input_features)`, where:
        - time_steps or sequence_length: number of time steps
        - input_features: number of input features

        **Note: creates only input sequences from the provided data.**

        :param data: pd.DataFrame: input data
        :param sequence_len: int: sequence length of the input/target sequences
        :param freatures: List[str]: list of features to use
        :return: Tuple[torch.Tensor, torch.Tensor]: input_sequences
        """
        # Copy data to avoid modifying the original data
        current_data = data.copy()

        # Select features
        current_data = current_data[features]

        # Get data using rolling window
        input_sequences = []

        # + 1 in order to get also the last sample
        number_of_samples = current_data.shape[0] - sequence_len + 1
        for i in range(number_of_samples):

            # Get the input sequence
            input_sequences.append(
                # Converting to a PyTorch tensor
                torch.tensor(
                    current_data.iloc[i : i + sequence_len].values, dtype=torch.float32
                )
            )

        return torch.stack(input_sequences)


if __name__ == "__main__":

    # Set up logging
    setup_logging()

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
    input_sequences, target_sequences = czech_data_loader.preprocess_training_data(
        train, 5, ["year", "population, total"]
    )

    input_batches, target_batches = czech_data_loader.create_batches(
        6, input_sequences=input_sequences, target_sequences=target_sequences
    )

    print("-" * 100)
    print("Batches: ")

    # Input batches
    print("-" * 100)
    print("Input:")
    print(input_batches.shape)

    # Output batches
    print("-" * 100)
    print("Target:")
    print(target_batches.shape)
