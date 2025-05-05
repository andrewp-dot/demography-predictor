# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import pandas as pd
import logging
import torch
from torch import nn
from typing import Tuple, Dict, List, Union


# Custom imports
from src.base import RNNHyperparameters, TrainingStats, CustomModelBase

from src.feature_model.model import BaseRNN


logger = logging.getLogger(name="finetuneable_local_model")


class FineTunableLSTM(CustomModelBase):

    def __init__(
        self,
        base_model: BaseRNN,
        hyperparameters: RNNHyperparameters,
    ):
        super(FineTunableLSTM, self).__init__(
            features=base_model.FEATURES,
            targets=base_model.TARGETS,
            hyperparameters=hyperparameters,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Load pretrained LSTM layers
        self.base_model = base_model
        self.base_lstm = base_model.rnn

        # Freeze pretrained LSTM layers
        for param in self.base_model.rnn.parameters():
            param.requires_grad = False

        # Add new LSTM layers for fine-tuning
        fine_tune_hidden_size = hyperparameters.hidden_size

        logger.debug(f"Base lstm hidden size: {base_model.rnn.hidden_size}")
        logger.debug(f"New lstm hidden size: {fine_tune_hidden_size}")

        # Hidden layer for transforming the hidden size of the base model to hidden size of the new lstm model
        self.hidden_transform = nn.Linear(
            base_model.rnn.hidden_size, fine_tune_hidden_size
        )

        self.new_lstm = nn.LSTM(
            input_size=fine_tune_hidden_size,
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
        self.base_h_0 = None
        self.base_c_0 = None

    def __initialize_hidden_states(
        self,
        batch_size: int,
        base_h_0: torch.Tensor | None = None,
        base_c_0: torch.Tensor | None = None,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Initiliaze hidden state for base model
        if base_h_0 is None:
            base_h_0 = torch.zeros(
                self.base_model.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.base_model.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

        # Initiliaze cell state
        if base_c_0 is None:
            base_c_0 = torch.zeros(
                self.base_model.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.base_model.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

        # Initiliaze hidden state for finetunable layer
        if h_0 is None:
            # Initialize hidden state and cell state for both: base model layers and finetunable layers
            h_0 = torch.zeros(
                self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

        # Initiliaze cell state for the finetunable layer
        if c_0 is None:
            c_0 = torch.zeros(
                self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

        # Return states
        return base_h_0, base_c_0, h_0, c_0

    def __reset_hidden_state(self) -> None:
        self.base_h_0, self.base_c_0 = None, None
        self.h_0, self.c_0 = None, None

    def __update_hidden_state(
        self,
        base_h_0: torch.Tensor,
        base_c_0: torch.Tensor,
        h_0: torch.Tensor,
        c_0: torch.Tensor,
        keep_context: bool,
    ) -> None:

        # For keeping the state
        if keep_context:
            self.base_h_0, self.base_c_0 = base_h_0, base_c_0
            self.h_0, self.c_0 = h_0, c_0
            return

        # Reset the state
        self.__reset_hidden_state()

    def forward(
        self,
        x: torch.Tensor,
        base_h_0: torch.Tensor | None = None,
        base_c_0: torch.Tensor | None = None,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ):

        # Ensure contiguous memory for faster execution
        self.base_lstm.flatten_parameters()
        self.new_lstm.flatten_parameters()

        # Initialize hidden states
        self.base_h_0, self.base_c_0, self.h_0, self.c_0 = (
            self.__initialize_hidden_states(
                batch_size=x.size(0),
                base_h_0=base_h_0,
                base_c_0=base_c_0,
                h_0=h_0,
                c_0=c_0,
            )
        )

        logger.debug(
            f"Shapes: {self.base_h_0.shape}, {self.base_c_0.shape}, {self.h_0.shape}, {self.c_0.shape}"
        )

        # Move inputs to the correct device
        x = x.to(self.device)
        self.base_h_0 = self.base_h_0.to(self.device)
        self.base_c_0 = self.base_c_0.to(self.device)
        self.h_0 = self.h_0.to(self.device)
        self.c_0 = self.c_0.to(self.device)

        # Forward pass for the frozen base LSTM
        with torch.no_grad():
            base_lstm_out, (base_h_n, base_c_n) = self.base_lstm(
                x, (self.base_h_0, self.base_c_0)
            )

        # Move base output to device
        base_lstm_out = base_lstm_out.to(self.device)

        # Transform base LSTM output (NOT hidden state)
        transformed_out = self.hidden_transform(base_lstm_out)
        transformed_out = transformed_out.to(self.device)

        # Forward pass through fine-tunable LSTM
        new_lstm_out, (new_h_n, new_c_n) = self.new_lstm(
            transformed_out, (self.h_0, self.c_0)
        )

        # Move to device
        new_lstm_out = new_lstm_out.to(self.device)

        # Extract last time step
        last_time_step_out = new_lstm_out[:, -1, :]

        # Final prediction
        return self.fc(last_time_step_out), (base_h_n, base_c_n), (new_h_n, new_c_n)

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_validation_inputs: torch.Tensor | None = None,
        batch_validation_targets: torch.Tensor | None = None,
        display_nth_epoch: int = 10,
        loss_function: Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss] = None,
    ):
        # TODO: trains the base LSTM model if it is not trained

        # Put the model to the device
        self.to(device=self.device)

        # Get loss function
        criterion = loss_function
        if loss_function is None:
            criterion = nn.MSELoss()

        # Get optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hyperparameters.learning_rate,
        )

        training_stats: Dict[str, List[int | float]] = {
            "epochs": [],
            "training_loss": [],
            "validation_loss": [],
        }

        num_epochs = self.hyperparameters.epochs
        training_stats["epochs"] = list(range(num_epochs))

        GET_VALIDATION_CURVE: bool = (
            not batch_validation_inputs is None and not batch_validation_targets is None
        )

        # Training loop
        for epoch in range(num_epochs):

            # Initialize epoch loss
            epoch_loss = 0.0

            # Train for every batch
            for batch_input, batch_target in zip(batch_inputs, batch_targets):

                # Get the batch input
                batch_input, batch_target = batch_input.to(
                    device=self.device
                ), batch_target.to(device=self.device)

                # Forward pass
                outputs, (base_h_0, base_c_0), (h_0, c_0) = self(
                    batch_input, self.base_h_0, self.base_c_0, self.h_0, self.c_0
                )

                loss = criterion(outputs, batch_target)
                epoch_loss += loss.cpu().item()

                # Update or reset the hidden state
                self.__update_hidden_state(
                    base_h_0=base_h_0,
                    base_c_0=base_c_0,
                    h_0=h_0,
                    c_0=c_0,
                    keep_context=False,
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate average loss in this epoch
            epoch_loss /= len(batch_inputs)

            # Add epoch and epoch loss to training stats
            training_stats["training_loss"].append(epoch_loss)

            # Add validation loss
            if GET_VALIDATION_CURVE:
                validation_epoch_loss = 0.0

                with torch.no_grad():

                    for batch_input, batch_target in zip(
                        batch_validation_inputs, batch_validation_targets
                    ):

                        # Put the targets to the device
                        batch_input, batch_target = batch_input.to(
                            device=self.device
                        ), batch_target.to(device=self.device)

                        # Forward pass
                        outputs, (base_h_0, base_c_0), (h_0, c_0) = self(
                            batch_input,
                            self.base_h_0,
                            self.base_c_0,
                            self.h_0,
                            self.c_0,
                        )

                        # Compute loss
                        loss = criterion(outputs, batch_target)

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"Loss is NaN/Inf at epoch {epoch}")
                            raise ValueError(
                                "Loss became NaN or Inf, stopping training!"
                            )

                        # loss.to(
                        #     device=self.device
                        # )  # Use this to prevent errors, see if this will work on azure

                        validation_epoch_loss += loss.cpu().item()

                # Save the stats
                validation_epoch_loss /= len(batch_validation_inputs)
                training_stats["validation_loss"].append(validation_epoch_loss)

            # Display average loss
            if not epoch % display_nth_epoch or epoch == (num_epochs - 1):
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation loss: {validation_epoch_loss:.4f}"
                )

            # Reset hidden states after epoch
            self.__reset_hidden_state()

        return training_stats

    # TODO: predict stateless or statefull... initiliazie hidden state and reset after every forward pass or not
    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
        keep_context: bool = True,
    ) -> torch.Tensor:
        """
        Based on input data predicts the values given input data.

        Args:
            input_data (pd.DataFrame): Preprocessed (scaled) input data.
            last_year (int): The year of the last record in the input data.
            target_year (int): The target year to which predictions are generated.
            keep_context (bool, optional): Experimental. If set to True, use past data to get more context. Defaults to True.

        Returns:
            out: torch.Tensor: Prediction tensor.
        """

        to_predict_years_num = (
            target_year - last_year
        )  # To include also the target year

        logger.info(f"Last recorded year: {last_year}")
        logger.info(f"Target year: {target_year}")
        logger.info(f"To predict years: {to_predict_years_num}")

        # Put the model into the evaluation mode
        self.eval()

        # Scale data
        # scaled_input_data = self.SCALER.transform(input_data[self.FEATURES])

        # scaled_input_data_df = pd.DataFrame(scaled_input_data)

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
                window = window.unsqueeze(0).to(
                    self.device
                )  # Add batch dimension: (1, sequence_length, input_size)

                # Put the input to the device
                window.to(device=self.device)

                # Forward pass
                pred, (base_h_0, base_c_0), (h_0, c_0) = self(
                    window, self.base_h_0, self.base_c_0, self.h_0, self.c_0
                )

                # Reset or upate hiden state based on keep_context value
                self.__update_hidden_state(
                    base_h_0=base_h_0,
                    base_c_0=base_c_0,
                    h_0=h_0,
                    c_0=c_0,
                    keep_context=False,
                )

                predictions.append(pred.cpu())

            current_window = input_sequence[-sequence_length:].unsqueeze(0)

            # Predict new data using autoregression
            for _ in range(to_predict_years_num):
                logger.debug(f"Current input window: {current_window}")

                # Forward pass
                pred, (base_h_0, base_c_0), (h_0, c_0) = self(
                    current_window, self.base_h_0, self.base_c_0, self.h_0, self.c_0
                )  # Shape: (1, output_size)

                self.__update_hidden_state(
                    base_h_0=base_h_0,
                    base_c_0=base_c_0,
                    h_0=h_0,
                    c_0=c_0,
                    keep_context=False,
                )

                pred_value = pred.squeeze(0)  # Remove batch dim
                logger.debug(f"New prediction value: {pred_value}")

                predictions.append(pred.cpu())  # Store new prediction
                new_predictions.append(pred.cpu())

                # Shift the window by removing the first value and adding the new prediction
                current_window = torch.cat(
                    (
                        current_window[:, 1:, :].to(device=self.device),
                        pred.unsqueeze(0),
                    ),
                    dim=1,
                )

        # Reset the hidden state for next prediction
        self.__reset_hidden_state()

        # Combine with years
        for year, pred in zip(range(last_year + 1, target_year + 1), new_predictions):
            logger.debug(f"{year}: {pred}")

        new_predictions_tensor = torch.cat(new_predictions, dim=0)

        return new_predictions_tensor
