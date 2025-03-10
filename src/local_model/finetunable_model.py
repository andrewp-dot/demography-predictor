# Standard library imports
import pandas as pd
import logging
import torch
from torch import nn

# Custom imports
from src.local_model.base import LSTMHyperparameters, TrainingStats
from src.local_model.model_base import BaseLSTM

logger = logging.getLogger(name="finetuneable_local_model")

# NOTE:
# What to do with hidden state initialization?
# 1.    If you want to maintain the hidden state acros multiple inputs and outputs -> you should initialize the hidden state once.
#       (Maybe add a resetting function).
#
# 2.    If you do want to have fresh start, initialize hidden state every time
# 3.    If you dont initiliaze hidden state, pytorch does it automatically


class FineTunableLSTM(BaseLSTM):

    def __init__(self, base_model: BaseLSTM, hyperparameters: LSTMHyperparameters):
        super(FineTunableLSTM, self).__init__(hyperparameters)

        # Get the hyperparameters and device
        self.hyperparameters: LSTMHyperparameters = hyperparameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained LSTM layers
        self.base_lstm = base_model.lstm

        # Freeze pretrained LSTM layers
        for param in self.base_lstm.parameters():
            param.requires_grad = False

        # Add new LSTM layers for fine-tuning
        fine_tune_hidden_size = hyperparameters.hidden_size

        self.new_lstm = nn.LSTM(
            input_size=base_model.lstm.hidden_size,
            hidden_size=fine_tune_hidden_size,
            num_layers=hyperparameters.num_layers,
            batch_first=True,
        )

        # New fine-tunable output layer
        self.fc = nn.Linear(fine_tune_hidden_size, hyperparameters.input_size)

        # Get training stats
        self.training_stasts: TrainingStats = TrainingStats()

        # Initialize the hidden states
        self.h_0 = None
        self.c_0 = None

    def __initialize_hidden_states(self, batch_size: int) -> None:

        if self.h_0 is None and self.c_0 is None:

            # Initialize hidden state and cell state for both: base model layers and finetunable layers
            self.h_0 = torch.zeros(
                self.base_lstm.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            )
            self.c_0 = torch.zeros(
                self.base_lstm.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            )

    def __reset_hidden_states(self, batch_size: int) -> None:
        self.h_0 = None
        self.c_0 = None

    def forward(self, x: torch.Tensor):

        # Forward pass for the base model, skip gradiend calculation -> freeze base model lstm layers
        with torch.no_grad():
            old_lstm_out, _ = self.base_lstm(
                x,
                (
                    self.h_0[: self.base_lstm.num_layers],
                    self.c_0[: self.base_lstm.num_layers],
                ),
            )

        # Get the output of new lstm
        new_lstm_out, _ = self.new_lstm(
            old_lstm_out,
            (
                self.h_0[self.base_lstm.num_layers :],
                self.c_0[self.base_lstm.num_layers :],
            ),
        )

        # Get last time step
        last_time_step_out = new_lstm_out[:, -1, :]

        return self.fc(last_time_step_out)

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ):
        # Put the model to the device
        self.to(device=self.device)

        # Get loss function
        criterion = nn.MSELoss()

        # Get optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hyperparameters.learning_rate,
        )

        # Training loop
        num_epochs = self.hyperparameters.epochs
        for epoch in range(num_epochs):

            # Initialize epoch loss
            epoch_loss = 0

            # Train for every batch
            for batch_input, batch_target in zip(batch_inputs, batch_targets):

                # Get the batch input
                batch_input, batch_target = batch_input.to(
                    device=self.device
                ), batch_target.to(device=self.device)

                outputs = self(batch_input)

                loss = criterion(outputs, batch_target)
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average loss in this epoch
            epoch_loss /= len(batch_inputs)

            # Add epoch and epoch loss to training stats
            self.training_stasts.losses.append(epoch_loss)
            self.training_stasts.epochs.append(epoch)

            # Display average loss
            if not epoch % display_nth_epoch:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
