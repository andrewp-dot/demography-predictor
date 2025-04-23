import pandas as pd
from typing import Tuple, Dict, List, Union

import logging

import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.utils.save_model import save_model

from src.base import RNNHyperparameters, TrainingStats, CustomModelBase

# from src.evaluation import EvaluateModel

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.data_transformer import DataTransformer

logger = logging.getLogger("local_model")


class GRUNet(CustomModelBase):

    def __init__(
        self,
        hyperparameters: RNNHyperparameters,
        features: List[str],
        targets: List[str] | None = None,
    ):
        super(GRUNet, self).__init__(
            features=features,
            targets=(targets if targets else features),
            hyperparameters=hyperparameters,
            scaler=None,
        )

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3 layer model:
        self.lstm = nn.GRU(
            input_size=hyperparameters.input_size,
            # Define the number of hidden units in the LSTM layer
            hidden_size=hyperparameters.hidden_size,
            # Define number of layers of the neural LSTM network
            num_layers=hyperparameters.num_layers,
            # 'batch_first' indicates that the input is in the format (batch_size, sequence_length, input_features)
            # not in the format (sequence_length, batch_size, input_features)
            batch_first=True,
        )

        # # Add some other layers
        # self.linear_1 = nn.Linear(
        #     in_features=hyperparameters.hidden_size,
        #     out_features=256,
        # )
        # self.relu_1 = nn.ReLU()  # Adds non-linearity

        # self.linear_2 = nn.Linear(
        #     in_features=256,
        #     out_features=128,
        # )
        # self.relu_2 = nn.ReLU()  # Adds non-linearity

        # Out layer: Linear layer
        self.linear = nn.Linear(
            in_features=hyperparameters.hidden_size,
            out_features=(
                hyperparameters.future_step_predict * hyperparameters.output_size
            ),  # Future steps * number of features to predict
        )

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
        ).to(device=self.device)

        c_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        ).to(device=self.device)

        return h_t, c_t

    def forward(
        self,
        x: torch.Tensor,
        # hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        self.lstm.flatten_parameters()  # # Fix for contiguous memory issue

        # Get the size of the batch
        batch_size = x.size(0)

        # If no hidden state is passed, initialize it
        h_t, c_t = self.__initialize_hidden_states(batch_size)

        # Forward propagate through LSTM
        out, (h_n, c_n) = self.lstm(x, (h_t, c_t))  # In here 'out' is a tensor

        # Use the output from the last time step -> use fully connected layers
        out = out[:, -1, :]

        # out = self.linear_1(out)  # Using the last time step's output
        # out = self.relu_1(out)

        # out = self.linear_2(out)
        # out = self.relu_2(out)

        out = self.linear(out)  # Using the last time step's output

        # Return both output and the updated hidden state (for the next batch)
        # return out, (h_n, c_n)

        # Transform prediction to the correct format
        # out = out.view(
        #     -1,
        #     self.hyperparameters.future_step_predict,
        #     self.hyperparameters.output_size,
        # )

        out = out.view(
            batch_size,
            self.hyperparameters.future_step_predict,
            self.hyperparameters.output_size,
        )

        return out

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_validation_inputs: torch.Tensor | None = None,
        batch_validation_targets: torch.Tensor | None = None,
        display_nth_epoch: int = 10,
        loss_function: Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss] = None,
    ) -> Dict[str, List[int | float]]:
        """
        Trains model using batched input sequences and batched target sequences.

        Args:
            batch_inputs (torch.Tensor): Batches of input sequences.
            batch_targets (torch.Tensor): Batches of target sequences
            display_nth_epoch (int, optional): Display every nth epochs. Always displays first and last epoch. Defaults to 10.
            loss_function (Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss], optional): _description_. Defaults to None.
            training_stats (TrainingStats | None, optional): Training stats object to track the training and validation loss, If None, no records are tracked. Defaults to None.

        Raises:
            ValueError: If the loss is None or Infinity.

        Returns:
            out: Dict[str, List[int | float]]: Training statistics. contains list of epochs, training and validation loss curves.
        """

        torch.autograd.set_detect_anomaly(True)

        # Get loss function
        criterion = loss_function
        if loss_function is None:
            criterion = nn.MSELoss()

        # Put the model to the device
        self.to(device=self.device)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyperparameters.learning_rate
        )

        # Define the training stats
        training_stats: Dict[str, List[int | float]] = {
            "epochs": [],
            "training_loss": [],
            "validation_loss": [],
        }

        # Define the training loop
        num_epochs = self.hyperparameters.epochs
        training_stats["epochs"] = list(range(num_epochs))

        # Set the flag for gettting the validation loss
        GET_VALIDATION_CURVE: bool = (
            not batch_validation_inputs is None and not batch_validation_targets is None
        )

        # Training loop
        for epoch in range(num_epochs):

            # Init epoch loss
            epoch_loss = 0.0
            for batch_input, batch_target in zip(batch_inputs, batch_targets):

                # Put the targets to the device
                batch_input, batch_target = batch_input.to(
                    device=self.device
                ), batch_target.to(device=self.device)

                # Forward pass
                outputs = self(batch_input)

                # Compute loss
                loss = criterion(outputs, batch_target)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Loss is NaN/Inf at epoch {epoch}")
                    raise ValueError("Loss became NaN or Inf, stopping training!")

                loss.to(
                    device=self.device
                )  # Use this to prevent errors, see if this will work on azure

                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Computes gradients
                optimizer.step()  # Update weights and biases

            epoch_loss /= len(batch_inputs)
            training_stats["training_loss"].append(epoch_loss)

            # Get validation loss if available
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
                        outputs = self(batch_input)

                        # Compute loss
                        loss = criterion(outputs, batch_target)

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"Loss is NaN/Inf at epoch {epoch}")
                            raise ValueError(
                                "Loss became NaN or Inf, stopping training!"
                            )

                        loss.to(
                            device=self.device
                        )  # Use this to prevent errors, see if this will work on azure

                        validation_epoch_loss += loss.item()

                # Save the stats
                validation_epoch_loss /= len(batch_validation_inputs)
                training_stats["validation_loss"].append(validation_epoch_loss)

            # Display loss
            if not epoch % display_nth_epoch or epoch == (
                num_epochs - 1
            ):  # Display first, nth epoch and last
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation loss: {validation_epoch_loss:.4f}"
                )

        return training_stats

    def shap_predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Single prediction for shap

        with torch.no_grad():
            pred = self(input_tensor)

        return pred

    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
    ) -> torch.Tensor:
        """
         Predicts values using past data. 2 cycles: 1st to gain context for all past data. 2nd is used to generate predictions.

        Args:
            input_data (torch.Tensor): Preprocessed (scaled) input data.
            last_year (int): The last year of all records in the data.
            target_year (int): The request year to get predictions to.

        Returns:
            out: torch.Tensor: Generated predictions.
        """

        to_predict_years_num = (
            target_year - last_year
        )  # To include also the target year

        logger.info(f"Last recorded year: {last_year}")
        logger.info(f"Target year: {target_year}")
        logger.info(f"To predict years: {to_predict_years_num}")

        # Put the model into the evaluation mode
        self.eval()

        logger.debug(f"Input sequence: {input_data.shape}")

        # Move input to the appropriate device
        input_sequence = torch.tensor(data=input_data.values, dtype=torch.float32)

        input_sequence.to(self.device)
        self.to(device=self.device)

        sequence_length = self.hyperparameters.sequence_length

        PREDICTED_STEPS: int = self.hyperparameters.future_step_predict

        # Array of predicions of previous values
        predictions = []

        with torch.no_grad():

            current_window = (
                input_sequence[-sequence_length:].unsqueeze(0).to(device=self.device)
            )

            # Calculate number of iterations
            num_of_iterations = (
                to_predict_years_num + PREDICTED_STEPS - 1
            ) // PREDICTED_STEPS
            for _ in range(num_of_iterations):
                logger.debug(f"Current input window: {current_window}")

                # Forward pass
                pred = self(current_window)  # Shape: (1, future_steps, output_size)

                pred_values = pred.squeeze(0)  # (future_steps, output_size)

                # Store new predictions
                predictions.append(pred_values.cpu())

                # Shift the window by removing the first value and adding the new prediction
                current_window = torch.cat(
                    (current_window[:, PREDICTED_STEPS:, :], pred_values.unsqueeze(0)),
                    dim=1,
                )

        new_predictions_tensor = torch.cat(predictions, dim=0)

        # Get only prediction to the target year
        new_predictions_tensor = new_predictions_tensor[:to_predict_years_num]

        return new_predictions_tensor
