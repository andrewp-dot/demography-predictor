# Standard library imports
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Callable, Union
from sklearn.preprocessing import MinMaxScaler
from src.base import LSTMHyperparameters

import torch
from sklearn.preprocessing import LabelEncoder

# Custom imports
from src.preprocessors.state_preprocessing import StateDataLoader


# TODO:
# 1. Save this as preprocessor for data
# 2. Preprocess categorical data
# 3. Understand the data -> plot the dataset etc.


class DataTransformer:

    # Categorical columns
    CATEGORICAL_COLUMNS: List[str] = ["country name"]

    # Division of features by types
    ABSOLUTE_COLUMNS: List[str] = [
        # Features
        "year",
        "fertility rate, total",
        "birth rate, crude",
        "adolescent fertility rate",
        "death rate, crude",
        "life expectancy at birth, total",
    ]

    PERCENTUAL_COLUMNS: List[str] = [
        # Features
        "population growth",
        "arable land",
        "gdp growth",
        "agricultural land",
        "rural population",
        "rural population growth",
        "urban population",
        "age dependency ratio",
        # Targets
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
        "population, female",
        "population, male",
    ]

    SPECIAL_COLUMNS: Dict[str, Callable] = {
        "net migration": "transform_net_migration",
        "population, total": "transform_population_total",
    }

    def __init__(self):
        self.SCALER: MinMaxScaler | None = None
        self.TARGET_SCALER: MinMaxScaler | None = None
        self.__unused_states: List[str] = []
        self.LABEL_ENCODERS: Dict[str, LabelEncoder] | None = None

    def get_unused_states(self) -> List[str]:
        return self.__unused_states

    def transform_categorical_columns(
        self,
        data: pd.DataFrame,
        inverse: bool = False,
    ):

        # Copy dataframe to avoid direct modification
        to_encode_data = data.copy()

        # Filter out supported categorical columns
        categorical_columns = list(set(self.CATEGORICAL_COLUMNS) & set(data.columns))
        to_encode_data = to_encode_data[categorical_columns]

        # Create new label encoders if the label encoders is None and there are some columns
        if self.LABEL_ENCODERS is None and categorical_columns:
            new_label_encoders: Dict[str, LabelEncoder] = {}
            for col in categorical_columns:
                le = LabelEncoder()
                to_encode_data[col] = le.fit_transform(to_encode_data[col])
                new_label_encoders[col] = (
                    le  # Save encoders if you want to inverse later
                )

            # Save the encoders dict
            self.LABEL_ENCODERS = new_label_encoders
            return to_encode_data

        # If the label encoders are created
        for col in categorical_columns:

            # For inverse transformation
            if inverse:
                to_encode_data[col] = self.LABEL_ENCODERS[col].inverse_transform(
                    to_encode_data[col]
                )
            else:
                to_encode_data[col] = self.LABEL_ENCODERS[col].transform(
                    to_encode_data[col]
                )

        return to_encode_data[categorical_columns]

    def transform_percentual_columns(
        self, data: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:

        to_scale_data = data.copy()

        # Get available percentual columns
        percentual_columns = list(set(self.PERCENTUAL_COLUMNS) & set(data.columns))

        if inverse:
            return to_scale_data[percentual_columns].apply(lambda x: x * 100)

        return to_scale_data[percentual_columns].apply(lambda x: x / 100)

    def transform_net_migration(
        self, net_migration_data: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:
        to_scale_data = net_migration_data.copy()

        # Decode
        if inverse:
            return to_scale_data.apply(lambda col: np.sign(col) * np.expm1(np.abs(col)))
        # Encode
        return to_scale_data.apply(lambda col: np.sign(col) * np.log1p(np.abs(col)))

    def transform_population_total(
        self, population_total_data: pd.DataFrame, inverse: bool = False
    ):
        to_scale_data = population_total_data.copy()

        # Decode
        if inverse:
            return to_scale_data.apply(lambda col: np.sign(col) * np.expm1(np.abs(col)))
        # Encode
        return to_scale_data.apply(lambda col: np.sign(col) * np.log1p(np.abs(col)))

    def transform_data(
        self, data: pd.DataFrame, columns: List[str], inverse: bool = False
    ) -> pd.DataFrame:

        # Transforms data using specified transformation
        to_transform_data = data.copy()

        # Maintain the original column order
        FEATURES = columns

        # Process categorical columns
        categorical_df = self.transform_categorical_columns(
            data=to_transform_data[columns], inverse=inverse
        )

        # Process percentual columns - normalize from 0 - 100 to range 0 - 1
        percentual_df = self.transform_percentual_columns(
            data=to_transform_data[columns], inverse=inverse
        )

        # Get available absolute columns - leave as is, just use min max scaling
        absolute_columns = list(set(self.ABSOLUTE_COLUMNS) & set(columns))
        absolute_columns_df = to_transform_data[absolute_columns]

        # Process special columns
        special_columns_dfs: List[pd.DataFrame] = []
        for col, func_name in self.SPECIAL_COLUMNS.items():
            if col in columns:

                # Dynamically call the transformation methods
                transform_func = getattr(self, func_name)
                special_columns_dfs.append(
                    transform_func(to_transform_data[col], inverse=inverse)
                )

        # Merge transformed columns
        transformed_data_df = pd.concat(
            [categorical_df, absolute_columns_df, percentual_df, *special_columns_dfs],
            axis=1,
        )

        # Maintain the column order
        transformed_data_df = transformed_data_df[FEATURES]

        return transformed_data_df

    def __scale_and_fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        columns: List[str],
        scaler: MinMaxScaler,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        """
        This method is scaling for the model training. Fit specified scaler on the training data.

        Args:
            training_data (pd.DataFrame): Training dataframe (X values).
            validation_data (pd.DataFrame): Validation dataframe (y values).
            columns (List[str]): The columns for scaling.
            scaler (MinMaxScaler): Scaler to be fitted on training data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: scaled_training_data, scaled_validation_data, fitted_scaler
        """
        # Transforms raw data, fits the given scaler
        # Should be used on training_data

        ORIGINAL_COLUMNS = training_data.columns
        # ORIGINAL_DATA = pd.concat([training_data, validation_data], axis=0)
        ORIGINAL_DATA_TRAINING_DATA = training_data.copy()
        ORIGINAL_DATA_VALIDATION_DATA = validation_data.copy()

        # Transform data
        transformed_training_df = self.transform_data(
            data=training_data, columns=columns, inverse=False
        )
        transformed_validation_df = self.transform_data(
            data=validation_data, columns=columns
        )

        # Fit data on the training data
        scaler.fit(transformed_training_df)

        # Scale data
        scaled_training_data = scaler.transform(transformed_training_df)
        scaled_validation_data = scaler.transform(transformed_validation_df)

        scaled_training_data_df = pd.DataFrame(
            scaled_training_data, columns=transformed_training_df.columns
        )
        scaled_validation_data_df = pd.DataFrame(
            scaled_validation_data, columns=transformed_validation_df.columns
        )

        # scaled_data_df = pd.DataFrame(scaled_data, columns=merged_data.columns)

        # Reconstruct the original dataframes
        non_transformed_features = [f for f in ORIGINAL_COLUMNS if not f in columns]

        # Get the t
        scaled_training_data_df = pd.concat(
            [
                ORIGINAL_DATA_TRAINING_DATA[non_transformed_features].reset_index(
                    drop=True
                ),
                scaled_training_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        scaled_validation_data_df = pd.concat(
            [
                ORIGINAL_DATA_VALIDATION_DATA[non_transformed_features].reset_index(
                    drop=True
                ),
                scaled_validation_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return (
            scaled_training_data_df[ORIGINAL_COLUMNS],
            scaled_validation_data_df[ORIGINAL_COLUMNS],
            scaler,
        )

    def scale_and_fit(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        features: List[str],
        targets: List[str] | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method is scaling for the model training. Fit specified scaler on the training data.

        Args:
            training_data (pd.DataFrame): Training dataframe (X values).
            validation_data (pd.DataFrame): Validation dataframe (y values).
            columns (List[str]): The columns for scaling.
            scaler (MinMaxScaler): Scaler to be fitted on training data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: scaled_training_data, scaled_validation_data
        """

        FEATURES: List[str] = features
        if targets:
            FEATURES: List[str] = [f for f in features if not f in targets]
            TARGETS: List[str] = targets

        # Scale and fit feature data
        training_feature_data, validation_feature_data, feature_scaler = (
            self.__scale_and_fit(
                training_data=training_data,
                validation_data=validation_data,
                columns=FEATURES,
                scaler=MinMaxScaler(),
            )
        )
        # print(training_feature_data.columns)
        # print(training_feature_data)

        # Scale and fit targets
        if targets:
            scaled_training_data, scaled_validation_data, target_scaler = (
                self.__scale_and_fit(
                    training_data=training_feature_data,
                    validation_data=validation_feature_data,
                    columns=TARGETS,
                    scaler=MinMaxScaler(),
                )
            )
            self.TARGET_SCALER = target_scaler
        else:
            scaled_training_data = training_feature_data
            scaled_validation_data = validation_feature_data

        # Save scalers
        self.SCALER = feature_scaler

        return scaled_training_data, scaled_validation_data

    def scale_data(
        self, data: pd.DataFrame, features: List[str], targets: List[str] | None = None
    ) -> pd.DataFrame:

        if self.SCALER is None:
            raise ValueError(
                "The scaler isnt fitted yet. Please use scale_and_fit first."
            )

        to_scale_data = data.copy()
        ORIGINAL_COLUMNS = data.columns

        # Maintain the original column order
        FEATURES = features
        TARGETS = targets

        if not targets:
            TARGETS = []

        FEATURES = [f for f in features if not f in TARGETS]

        to_scale_feature_data = to_scale_data[FEATURES]

        # Transform feature data
        transformed_feature_data_df = self.transform_data(
            data=to_scale_feature_data, columns=FEATURES
        )
        scaled_feature_data = self.SCALER.transform(transformed_feature_data_df)
        scaled_feature_data_df = pd.DataFrame(
            scaled_feature_data, columns=transformed_feature_data_df.columns
        )

        scaled_data_df = scaled_feature_data_df

        if TARGETS:

            to_scale_target_data = to_scale_data[TARGETS]

            transformed_target_data_df = self.transform_data(
                data=to_scale_target_data, columns=TARGETS
            )
            scaled_target_data = self.TARGET_SCALER.transform(
                transformed_target_data_df
            )
            scaled_target_data_df = pd.DataFrame(
                scaled_target_data, columns=transformed_target_data_df.columns
            )

            # Update scaled data if there is also a target scaling
            scaled_data_df = pd.concat([scaled_data_df, scaled_target_data_df], axis=1)

        # Reconstruct the original dataframe
        non_transformed_features = [
            f for f in ORIGINAL_COLUMNS if not f in FEATURES and f not in TARGETS
        ]

        scaled_data_df = pd.concat(
            [
                data[non_transformed_features].reset_index(drop=True),
                scaled_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return scaled_data_df[ORIGINAL_COLUMNS]

    def unscale_data(
        self,
        data: pd.DataFrame,
        targets: List[str],
    ) -> pd.DataFrame:

        if self.TARGET_SCALER is None:
            raise ValueError(
                "The scaler isnt fitted yet. Please use scale_and_fit first."
            )

        # Check the fitted columns and specified columns compatibility?
        to_unscale_data = data.copy()

        ORIGINAL_COLUMNS = data.columns

        # Maintain the original column order
        FEATURES = targets
        to_unscale_data = to_unscale_data[FEATURES]

        # Unsale data
        unscaled_data = self.TARGET_SCALER.inverse_transform(to_unscale_data)
        unscaled_data_df = pd.DataFrame(unscaled_data, columns=targets)

        # Inverse transform data
        reverse_transformed_data_df = self.transform_data(
            data=unscaled_data_df, columns=targets, inverse=True
        )

        INTEGER_COLUMNS = ["year"]  # Add more if needed

        for col in INTEGER_COLUMNS:
            if col in reverse_transformed_data_df.columns:
                reverse_transformed_data_df[col] = (
                    reverse_transformed_data_df[col].round().astype(int)
                )

        # Reconstruct the original dataframe
        non_transformed_features = [f for f in ORIGINAL_COLUMNS if f not in FEATURES]

        reverse_transformed_data_df = pd.concat(
            [
                data[non_transformed_features].reset_index(drop=True),
                reverse_transformed_data_df.reset_index(drop=True),
            ],
            axis=1,
        )

        return reverse_transformed_data_df[ORIGINAL_COLUMNS]

    def create_sequences(
        self, input_data: pd.DataFrame, columns: List[str], sequence_len: int
    ) -> torch.Tensor:
        # Creates sequences by rolling window from any input data

        # Copy data to avoid modifying the original data
        current_data = input_data.copy()

        # Select features
        current_data = current_data[columns]

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

    def create_input_batches(
        self, input_sequences: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        # Reshapes input sequences to create batches from any input sequences

        if len(input_sequences.shape) != 3:
            raise ValueError(
                "Input sequences must have shape (num_samples, sequence_len, feature_num)"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        num_samples, sequence_len, feature_num = input_sequences.shape

        if batch_size > num_samples:
            raise ValueError(
                "batch_size cannot be larger than the number of available samples"
            )

        # Calculate the number of batches and use trimming for correct reshaping
        num_batches = num_samples // batch_size
        trimmed_size = num_batches * batch_size  # Only keep full batches

        # Get only sequences which can craete full batch
        trimmed_sequences = input_sequences[:trimmed_size]

        # Reshape to (num_batches, batch_size, sequence_len, feature_num)
        # Use view for effiecency
        input_batches = trimmed_sequences.view(
            num_batches, batch_size, sequence_len, feature_num
        )

        return input_batches

    def create_target_batches(
        self, target_sequences: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        if len(target_sequences.shape) != 2:
            raise ValueError(
                "Target sequences must have shape (num_samples, num_features)"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        num_samples, num_features = (
            target_sequences.shape
        )  # In shape (batch_size, num_features)

        if batch_size > num_samples:
            raise ValueError(
                "batch_size cannot be larger than the number of available samples"
            )

        # Calculate the number of batches and use trimming for correct reshaping
        num_batches = num_samples // batch_size
        trimmed_size = num_batches * batch_size  # Only keep full batches

        # Trim tensor to match full batches
        trimmed_sequences = target_sequences[:trimmed_size]

        # Reshape to (num_batches, batch_size, num_features)
        target_batches = trimmed_sequences.reshape(
            num_batches, batch_size, num_features
        )

        return target_batches

    def create_train_test_sequences(
        self,
        data: pd.DataFrame,
        sequence_len: int,
        features: List[str],
        targets: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From the states data creates the input sequences and the desired output of the sequnce.

        Args:
            data (pd.DataFrame): _description_
            sequence_len (int): _description_
            features (List[str]): _description_

        Returns:
            out: Tuple[torch.Tensor, torch.Tensor]: Input sequence, expected output of the sequence
        """

        # Copy data to avoid modifying the original data
        current_data = data.copy()

        # Select features
        current_input_data = current_data[features]
        current_target_data = current_data[features]

        if not targets is None:
            current_target_data = current_data[targets]

        # Get data using rolling window
        input_sequences = []
        sequence_output = []

        # + 1 in order to get also the last sample
        number_of_samples = current_data.shape[0] - sequence_len
        for i in range(number_of_samples):

            # Get the input sequence
            input_sequences.append(
                # Converting to a PyTorch tensor
                torch.tensor(
                    current_input_data.iloc[i : i + sequence_len].values,
                    dtype=torch.float32,
                )
            )

            # Get the expected output of the sequence
            sequence_output.append(
                torch.tensor(
                    current_target_data.iloc[i + sequence_len].values,
                    dtype=torch.float32,
                )
            )

        return torch.stack(input_sequences), torch.stack(sequence_output)

    def create_train_test_data_batches(
        self,
        data: pd.DataFrame,
        hyperparameters: LSTMHyperparameters,
        features: List[str],
        split_rate: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform the data to training set and validation set.

        Args:
            data (pd.DataFrame): Pre-scaled data for training and validation.
            hyperparameters (LSTMHyperparameters): Hyperparameters of the model.
            features (List[str]): Selected features from the data.

        Returns:
            out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: train_inputs, train_targets, val_inputs, val_targets
        """
        # Adjust the data
        df = data.copy()
        FEATURES = features

        # Create sequences and batches from the full training data
        input_sequences, sequences_outputs = self.create_train_test_sequences(
            data=df, sequence_len=hyperparameters.sequence_length, features=FEATURES
        )

        # Convert to numpy
        input_sequences = np.array(input_sequences)
        sequences_outputs = np.array(sequences_outputs)

        # Compute split index
        total_samples = len(input_sequences)
        split_idx = int(split_rate * total_samples)

        # Split into training and validation
        train_inputs = input_sequences[:split_idx]
        train_targets = sequences_outputs[:split_idx]
        val_inputs = input_sequences[split_idx:]
        val_targets = sequences_outputs[split_idx:]

        # Convert to torch tensors
        train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
        train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
        val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
        val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)

        # Batch inputs
        train_inputs_tensor = self.create_input_batches(
            batch_size=hyperparameters.batch_size,
            input_sequences=train_inputs_tensor,
        )

        val_inputs_tensor = self.create_input_batches(
            batch_size=hyperparameters.batch_size,
            input_sequences=val_inputs_tensor,
        )

        train_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=train_targets_tensor,
        )

        val_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=val_targets_tensor,
        )

        return (
            train_inputs_tensor,
            train_targets_tensor,
            val_inputs_tensor,
            val_targets_tensor,
        )

    # Create train test data from the states dict
    def create_train_test_multiple_states_batches(
        self,
        data: Dict[str, pd.DataFrame],
        hyperparameters: LSTMHyperparameters,
        features: List[str],
        split_rate: float = 0.8,
        targets: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        SEQUENCE_LEN: int = hyperparameters.sequence_length

        # Training data
        train_inputs_all = []
        train_targets_all = []

        # Validation data
        test_inputs_all = []
        test_targets_all = []

        # Split each state
        for state_name, state_data in data.items():

            # Check if training data have enough records
            if int(len(state_data) * split_rate) <= SEQUENCE_LEN:
                self.__unused_states.append(state_name)
                continue

            input_sequences, sequences_outputs = self.create_train_test_sequences(
                data=state_data,
                sequence_len=SEQUENCE_LEN,
                features=features,
                targets=targets,
            )

            # Parse them to test and validation
            # Convert to numpy
            input_sequences = np.array(input_sequences)
            sequences_outputs = np.array(sequences_outputs)

            # Compute split index
            total_samples = len(input_sequences)
            split_idx = int(split_rate * total_samples)

            # Split into training and validation
            train_inputs = input_sequences[:split_idx]
            train_targets = sequences_outputs[:split_idx]
            val_inputs = input_sequences[split_idx:]
            val_targets = sequences_outputs[split_idx:]

            # Convert to torch tensors
            train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
            train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
            val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
            val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)

            # Save all train inputs and targets
            train_inputs_all.append(train_inputs_tensor)
            train_targets_all.append(train_targets_tensor)

            # Save all test inputs and targets
            test_inputs_all.append(val_inputs_tensor)
            test_targets_all.append(val_targets_tensor)

        # Convert inputs to tensors
        train_inputs_all = torch.cat(train_inputs_all)
        train_targets_all = torch.cat(train_targets_all)

        test_inputs_all = torch.cat(test_inputs_all)
        test_targets_all = torch.cat(test_targets_all)

        # Batch train and validation inputs
        train_inputs_tensor = self.create_input_batches(
            batch_size=hyperparameters.batch_size,
            input_sequences=train_inputs_all,
        )

        val_inputs_tensor = self.create_input_batches(
            batch_size=hyperparameters.batch_size,
            input_sequences=test_inputs_all,
        )

        # Batch train and validation targets
        train_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=train_targets_all,
        )

        val_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=test_targets_all,
        )

        return (
            train_inputs_tensor,
            train_targets_tensor,
            val_inputs_tensor,
            val_targets_tensor,
        )


def main():
    STATE = "Czechia"

    # Load data
    loader = StateDataLoader(state=STATE)
    transformer = DataTransformer()

    data = loader.load_data()

    EXCLUDE_COLUMNS = ["country name", "year"]
    COLUMNS = [col for col in data.columns if col not in EXCLUDE_COLUMNS]

    # Split data
    train_df, test_df = loader.split_data(data=data)

    print("Original train data:")
    print(train_df.head())
    print()

    # Scale training data
    scaled_trainig_data, fitted_scaler = transformer.scale_and_fit(
        training_data=train_df, columns=COLUMNS, scaler=MinMaxScaler()
    )

    print("Scaled train data:")
    print(scaled_trainig_data.head())
    print()

    # Unscale data
    unsacled_training_data = transformer.unscale_data(
        data=scaled_trainig_data, columns=COLUMNS
    )

    print("Unscaled train data:")
    print(unsacled_training_data.head())
    print()

    print()
    print("-" * 100)
    print()

    # Scale test data using fitted scaler
    print("Original test data:")
    print(test_df.head())
    print()

    scaled_test_data = transformer.scale_data(data=test_df, columns=COLUMNS)
    print("Scaled test data:")
    print(scaled_test_data.head())
    print()

    unscaled_test_data = transformer.unscale_data(
        data=scaled_test_data, columns=COLUMNS
    )
    print("Unscaled test data:")
    print(unscaled_test_data.head())
    print()


if __name__ == "__main__":
    main()
