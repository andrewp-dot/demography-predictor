# Standard library imports
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


import torch

# Custom imports
from src.base import RNNHyperparameters


class RNNTrainingDataPreprocessor:

    def __init__(self):
        # Get the unused states for fitting
        self.__unused_states: List[str] = []

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
        self, target_sequences: torch.Tensor, batch_size: int, future_steps: int = 1
    ) -> torch.Tensor:

        if len(target_sequences.shape) != 3:
            raise ValueError(
                "Target sequences must have shape (num_samples, timesteps, num_features)"
            )

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        num_samples, timesteps, num_features = (
            target_sequences.shape
        )  # In shape (num_samples, timesteps, num_features)

        if batch_size > num_samples:
            raise ValueError(
                "batch_size cannot be larger than the number of available samples"
            )

        if timesteps < future_steps:
            raise ValueError(
                "The number of timesteps must be greater than or equal to future_steps"
            )

        # Calculate how many windows we can extract from each sequence
        usable_timesteps = timesteps - future_steps + 1

        # Prepare target windows
        target_windows = []
        for i in range(usable_timesteps):
            future_window = target_sequences[
                :, i : i + future_steps, :
            ]  # (num_samples, future_steps, num_features)
            target_windows.append(
                future_window.unsqueeze(1)
            )  # (num_samples, 1, future_steps, num_features)

        target_windows = torch.cat(
            target_windows, dim=1
        )  # (num_samples, usable_timesteps, future_steps, num_features)

        # Now flatten num_samples and usable_timesteps into one dimension
        all_samples = target_windows.reshape(
            -1, future_steps, num_features
        )  # (num_samples * usable_timesteps, future_steps, num_features)

        # Trim to have full batches
        total_samples = all_samples.shape[0]
        num_batches = total_samples // batch_size
        trimmed_size = num_batches * batch_size

        all_samples = all_samples[:trimmed_size]

        # Reshape into batches
        target_batches = all_samples.reshape(
            num_batches, batch_size, future_steps, num_features
        )

        return target_batches

    def create_train_test_sequences(
        self,
        data: pd.DataFrame,
        sequence_len: int,
        future_steps: int,
        features: List[str],
        targets: List[str] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        From the states data creates the input sequences and the desired output of the sequnce.

        Args:
            data (pd.DataFrame): _description_
            sequence_len (int): _description_
            future_steps (int): _description_
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
        number_of_samples = current_data.shape[0] - sequence_len - future_steps + 1
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
            next_value_idx = i + sequence_len

            # Shape (timesteps, features)
            output = torch.tensor(
                current_target_data.iloc[
                    next_value_idx : next_value_idx + future_steps
                ].values,
                dtype=torch.float32,
            )

            # Reshape to match multiple timestep predictions - shape (timesteps * features vector)
            # output = output.reshape(-1)

            sequence_output.append(output)

        return torch.stack(input_sequences), torch.stack(sequence_output)

    def create_train_test_data_batches(
        self,
        data: pd.DataFrame,
        hyperparameters: RNNHyperparameters,
        features: List[str],
        targets: Optional[List[str]] = None,
        split_rate: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform the data to training set and validation set.

        Args:
            data (pd.DataFrame): Pre-scaled data for training and validation.
            hyperparameters (RNNHyperparameters): Hyperparameters of the model.
            features (List[str]): Selected features from the data.

        Returns:
            out: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: train_inputs, train_targets, val_inputs, val_targets
        """
        # Adjust the data
        df = data.copy()
        FEATURES = features

        TARGETS = targets

        # Create sequences and batches from the full training data
        input_sequences, sequences_outputs = self.create_train_test_sequences(
            data=df,
            sequence_len=hyperparameters.sequence_length,
            future_steps=hyperparameters.future_step_predict,
            features=FEATURES,
            targets=TARGETS,
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
            future_steps=hyperparameters.future_step_predict,
        )

        val_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=val_targets_tensor,
            future_steps=hyperparameters.future_step_predict,
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
        hyperparameters: RNNHyperparameters,
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
                future_steps=hyperparameters.future_step_predict,
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
            future_steps=hyperparameters.future_step_predict,
        )

        val_targets_tensor = self.create_target_batches(
            batch_size=hyperparameters.batch_size,
            target_sequences=test_targets_all,
            future_steps=hyperparameters.future_step_predict,
        )

        return (
            train_inputs_tensor,
            train_targets_tensor,
            val_inputs_tensor,
            val_targets_tensor,
        )
