# Standard libraries imports
import pandas as pd
import torch
from typing import Union, Tuple, List
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import logging

# Custom libraries imports
from config import Config
from src.utils.log import setup_logging
from src.local_model.base import LSTMHyperparameters

logger = logging.getLogger("data_preprocessing")

config = Config()


class StateDataLoader:

    def __init__(self, state: str):
        self.state = state

    def load_data(self, exclude_covid_years: bool = False) -> pd.DataFrame:
        """
        Load the state data. Renames the columns all to lower.

        :return: pd.DataFrame
        """
        data = pd.read_csv(f"{config.states_data_dir}/{self.state}.csv")

        # Lowercase columns in case they are not
        data.columns = data.columns.str.lower()

        if exclude_covid_years:
            # Covid started about 17th November in 2019. The impact of covid 19 is expected on years 2020 and higher.
            data = data.loc[data["year"] < 2020]

        return data

    def scale_data(
        self,
        data: pd.DataFrame,
        features: List[str],
        scaler: Union[RobustScaler, StandardScaler, MinMaxScaler],
    ) -> Tuple[pd.DataFrame, Union[RobustScaler, StandardScaler, MinMaxScaler]]:
        """
        Scales the data using the specified scaler.

        :param data: pd.DataFrame: input data
        :param scaler: Union[RobustScaler, StandardScaler, MinMaxScaler]: scaler to use

        :return: Tuple[pd.DataFrame, Union[RobustScaler, StandardScaler, MinMaxScaler]]: scaled dataframe and fitted scaler for data unscaling
        """

        # Set features constant
        FEATURES = features

        # Copy data to avoid inplace edits
        to_scale_data = data.copy()

        # Scale data
        scaled_data = scaler.fit_transform(to_scale_data[FEATURES])

        # Create dataframe from scaled data
        scaled_data_df = pd.DataFrame(scaled_data, columns=FEATURES)
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
        Groups sequences to batches. (Format: (num_batches, batch_size, sequence_len, num_features) ).

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

    def get_last_year(self, data: pd.DataFrame) -> int:
        """
        Get the last record in the dataframme.

        Args:
            data (pd.DataFrame): Dataframe which should contain the year.

        Returns:
            out: int: Last year of the record.
        """

        # Verify if the 'year' is in the columns
        if "year" not in data.columns:
            raise ValueError("The given dataframe does not contain the 'year' column!")

        # Retrieve the last year from data
        last_year = int(data["year"].sort_values(ascending=True).iloc[-1])

        return last_year

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

        Args:
            data (pd.DataFrame): Scaled input data.
            sequence_len (int): Sequence length of the input/target sequences.
            features (List[str]): List of features to use.

        Returns:
            out: Tuple[torch.Tensor, torch.Tensor]: input_sequences, target_sequences.
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
        input_sequences, target_sequences = torch.stack(input_sequences), torch.stack(
            target_sequences
        )

        logger.debug(
            f"[Preprocessing train data]: input sequences shape: {input_sequences.shape}, target sequences shape: {target_sequences.shape}"
        )

        return input_sequences, target_sequences

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

        Args:
            data (pd.DataFrame): Scaled input data.
            sequence_len (int): Sequence length of the input sequences.
            features (List[str]): List of features to use.

        Returns:
            out: torch.Tensor: Stacked input sequences.
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

    def preprocess_data_batches(
        self,
        data: pd.DataFrame,
        hyperparameters: LSTMHyperparameters,
        features: List[str],
        scaler: Union[MinMaxScaler, RobustScaler, StandardScaler],
    ) -> torch.Tensor:

        # Get features
        FEATURES = features

        # Scale data
        scaled_data, _ = self.scale_data(data, features=FEATURES, scaler=scaler)

        # Create input sequences
        input_sequences = self.preprocess_data(
            data=scaled_data,
            sequence_len=hyperparameters.sequence_length,
            features=FEATURES,
        )

        # Create batches
        input_batches, _ = self.create_batches(
            batch_size=hyperparameters.batch_size, input_sequences=input_sequences
        )

        return input_batches

    def preprocess_training_data_batches(
        self,
        train_data_df: pd.DataFrame,
        hyperparameters: LSTMHyperparameters,
        features: List[str],
        scaler: Union[MinMaxScaler, RobustScaler, StandardScaler],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Union[MinMaxScaler, RobustScaler, StandardScaler]
    ]:
        """
        Converts training data to format for training. From the unscaled training data to batches.

        Args:
            train_data_df (pd.DataFrame): Unscaled training data.
            state_loader (StateDataLoader): Loader for the state.
            hyperparameters (LSTMHyperparameters): Hyperparameters used for training the model.
            features (List[str]): Features used in model.

        Returns:
            out: Tuple[torch.Tensor, torch.Tensor, Union[MinMaxScaler, RobustScaler, StandardScaler]]: train input batches, train target batches,
            fitted scaler used for training data scaling.
        """

        # Get features
        FEATURES = features

        # Scale data
        scaled_train_data, state_scaler = self.scale_data(
            train_data_df, features=FEATURES, scaler=scaler
        )

        # Create input and target sequences
        train_input_sequences, train_target_sequences = self.preprocess_training_data(
            data=scaled_train_data,
            sequence_len=hyperparameters.sequence_length,
            features=FEATURES,
        )

        # Create input and target batches for faster training
        train_input_batches, train_target_batches = self.create_batches(
            batch_size=hyperparameters.batch_size,
            input_sequences=train_input_sequences,
            target_sequences=train_target_sequences,
        )

        # Return training batches, target batches and fitted scaler
        return train_input_batches, train_target_batches, state_scaler


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
