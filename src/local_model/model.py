import pandas as pd
from typing import Tuple, Dict, List, Callable, Union
import pprint
import logging

import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# Custom imports
from src.local_model.base import (
    CustomModelBase,
    LSTMHyperparameters,
    EvaluateModel,
    TrainingStats,
)
from src.preprocessors.state_preprocessing import StateDataLoader
from config import setup_logging


logger = logging.getLogger("local_model")

# TODO: explain detaching?
# You can detach the hidden state here if you need to avoid backprop through the entire sequence
# h_n = h_n.detach()
# c_n = c_n.detach()


class LocalModel(CustomModelBase):

    def __init__(self, hyperparameters: LSTMHyperparameters):
        super(LocalModel, self).__init__()

        self.hyperparameters: LSTMHyperparameters = hyperparameters

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3 layer model:
        self.lstm = nn.LSTM(
            input_size=hyperparameters.input_size,
            # Define the number of hidden units in the LSTM layer
            hidden_size=hyperparameters.hidden_size,
            # Define number of layers of the neural LSTM network
            num_layers=hyperparameters.num_layers,
            # 'batch_first' indicates that the input is in the format (batch_size, sequence_length, input_features)
            # not in the format (sequence_length, batch_size, input_features)
            batch_first=True,
        )

        # Out layer: Linear layer
        self.linear = nn.Linear(
            in_features=hyperparameters.hidden_size,
            out_features=hyperparameters.input_size,
        )

        # Get stats
        self.training_stats = TrainingStats()

    def __initialize_hidden_states(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden states for the LSTM layers.

        Args:
            n_samples (int): number of samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden states: h_t, c_t
        """
        h_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        )

        c_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        )

        return h_t, c_t

    def forward(
        self,
        x: torch.Tensor,
        # hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:

        # Get the size of the batch
        batch_size = x.size(0)

        # If no hidden state is passed, initialize it
        h_t, c_t = self.__initialize_hidden_states(batch_size)

        # Forward propagate through LSTM
        out, (h_n, c_n) = self.lstm(x, (h_t, c_t))  # In here 'out' is a tensor

        # Use the output from the last time step
        out = self.linear(out[:, -1, :])  # Using the last time step's output

        # Return both output and the updated hidden state (for the next batch)
        # return out, (h_n, c_n)
        return out

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ):
        """
        Trains model using batched input sequences and batched target sequences.


        :param batch_inputs: torch.Tensor: batches of input sequences
        :param batch_targets: torch.Tensor: batches of target sequences
        :param display_nth_epoch: int: First and every nth epoch is displayed
        """
        # Put the model to the device
        self.to(device=self.device)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyperparameters.learning_rate
        )

        # Define the training loop
        num_epochs = self.hyperparameters.epochs

        # Setup epochs for stats
        self.training_stats.epochs = list(range(num_epochs))

        # Training loop
        for epoch in range(num_epochs):

            # Init epoch loss
            epoch_loss = 0

            for batch_input, batch_target in zip(batch_inputs, batch_targets):

                # Put the targets to the device
                batch_input, batch_target = batch_input.to(
                    self.device
                ), batch_target.to(self.device)

                # Forward pass
                outputs = self(batch_input)

                # Compute loss
                loss = criterion(outputs, batch_target)
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Computes gradients
                optimizer.step()  # Update weights and biases

            epoch_loss /= len(batch_inputs)
            self.training_stats.losses.append(epoch_loss)

            if not epoch % display_nth_epoch:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
    ) -> torch.Tensor:
        """
        Predicts values using past data. 2 cycles: 1st to gain context for all past data. 2nd is used to generate predictions.

        :param input_data: pd.DataFrame: input data
        :param last_year: int: the last known year used in order to compute the number of new data iterations
        :param target_year: int: predict data to year

        :return torch.Tensor: prediction tensor
        """

        to_predict_years_num = (
            target_year - last_year
        )  # To include also the target year

        logger.info(f"Last recorder year: {last_year}")
        logger.info(f"To predict years: {to_predict_years_num}")

        # Put the model into the evaluation mode
        self.eval()

        input_sequence = torch.tensor(data=input_data.values, dtype=torch.float32)

        logger.debug(f"Input sequence: {input_sequence.shape}")

        # Move input to the appropriate device
        input_sequence.to(self.device)

        num_timesteps, input_size = input_sequence.shape
        sequence_length = self.hyperparameters.sequence_length

        # Array of predicions of previous values
        predictions = []

        # Predictions for new years (from last to target_year)
        new_predictions = []
        with torch.no_grad():
            # Use past data for further context
            for i in range(num_timesteps - sequence_length + 1):

                # Slide over the sequence
                window = input_sequence[i : i + sequence_length]  # Extract window
                window = window.unsqueeze(
                    0
                )  # Add batch dimension: (1, sequence_length, input_size)

                # Forward pass
                pred = self(window)

                predictions.append(pred.cpu())

            current_window = input_sequence[-sequence_length:].unsqueeze(0)

            # Predict new data using autoregression
            for _ in range(to_predict_years_num):
                logger.debug(f"Current input window: {current_window}")

                # Forward pass
                pred = self(current_window)  # Shape: (1, output_size)
                pred_value = pred.squeeze(0)  # Remove batch dim
                logger.debug(f"New prediction value: {pred_value}")

                predictions.append(pred.cpu())  # Store new prediction
                new_predictions.append(pred.cpu())

                # Shift the window by removing the first value and adding the new prediction
                current_window = torch.cat(
                    (current_window[:, 1:, :], pred.unsqueeze(0)), dim=1
                )

        # Combine with years
        for year, pred in zip(range(last_year + 1, target_year + 1), new_predictions):
            logger.debug(f"{year}: {pred}")

        new_predictions_tensor = torch.cat(new_predictions, dim=0)
        return new_predictions_tensor


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    # Setup logging
    setup_logging()

    FEATURES = [
        # "year",
        # "Fertility rate, total",
        # "Population, total",
        "Net migration",
        # "Arable land",
        # "Birth rate, crude",
        # "GDP growth",
        # "Death rate, crude",
        # "Agricultural land",
        # "Rural population",
        # "Rural population growth",
        # "Age dependency ratio",
        # "Urban population",
        # "Population growth",
        # "Adolescent fertility rate",
        # "Life expectancy at birth, total",
    ]

    FEATURES = [col.lower() for col in FEATURES]

    hyperparameters = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=2048,
        sequence_length=15,
        learning_rate=0.0001,
        epochs=30,
        batch_size=1,
        num_layers=4,
    )
    rnn = LocalModel(hyperparameters)

    # Load data
    czech_loader = StateDataLoader("Czechia")

    czech_data = czech_loader.load_data()

    # Exclude country name
    czech_data = czech_data.drop(columns=["country name"])

    print(czech_data[FEATURES])

    # Scale data
    scaled_cz_data, cz_scaler = czech_loader.scale_data(
        czech_data, features=FEATURES, scaler=MinMaxScaler()
    )

    print(scaled_cz_data.head())

    train, test = czech_loader.split_data(scaled_cz_data)

    # Get input/output sequences from train data
    input_sequences, target_sequences = czech_loader.preprocess_training_data(
        train,
        hyperparameters.sequence_length,
        features=FEATURES,
    )

    # Get input/output batches of sequences
    batch_inputs, batch_targets = czech_loader.create_batches(
        batch_size=hyperparameters.batch_size,
        input_sequences=input_sequences,
        target_sequences=target_sequences,
    )

    print("-" * 100)
    logger.info(f"Batch inputs shape: {batch_inputs.shape}")
    logger.info(f"Batch targets shape: {batch_targets.shape}")

    # Train model
    rnn.train_model(
        batch_inputs=batch_inputs, batch_targets=batch_targets, display_nth_epoch=1
    )

    # Evaluate model
    model_evaluation = EvaluateModel(rnn)

    # Split the raw data
    val_X, val_y = czech_loader.split_data(czech_data)

    model_evaluation.eval(
        test_X=val_X,
        test_y=val_y,
        features=FEATURES,
        scaler=cz_scaler,
    )

    logger.info(
        f"Model evaluation by per target metrics: \n{model_evaluation.per_target_metrics}"
    )
    logger.info(
        f"Model evaluation by overall metrics: \n{model_evaluation.overall_metrics}"
    )

    fig = model_evaluation.plot_predictions()

    import matplotlib.pyplot as plt

    plt.show()
