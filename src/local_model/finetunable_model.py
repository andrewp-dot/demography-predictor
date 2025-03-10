# Standard library imports
import pandas as pd
import logging
import torch
from torch import nn
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from config import setup_logging
from src.local_model.base import LSTMHyperparameters, TrainingStats, EvaluateModel
from src.local_model.model_base import BaseLSTM
from src.local_model.model import LocalModel

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger(name="finetuneable_local_model")

# NOTE:
# What to do with hidden state initialization?
# 1.    If you want to maintain the hidden state acros multiple inputs and outputs -> you should initialize the hidden state once.
#       (Maybe add a resetting function).
#
# 2.    If you do want to have fresh start, initialize hidden state every time
# 3.    If you dont initiliaze hidden state, pytorch does it automatically

# TODO:
# Where to put resseting hidden state function?


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

    def __initialize_hidden_states(
        self,
        batch_size: int,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Initiliaze hidden state
        if h_0 is None:
            # Initialize hidden state and cell state for both: base model layers and finetunable layers
            h_0 = torch.zeros(
                self.base_lstm.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            )

        # Initiliaze cell state
        if c_0 is None:
            c_0 = torch.zeros(
                self.base_lstm.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            )

        # Return states
        return h_0, c_0

    def __reset_hidden_state(self) -> None:
        self.h_0, self.c_0 = None, None

    def __update_hidden_state(
        self, h_0: torch.Tensor, c_0: torch.Tensor, keep_context: bool
    ) -> None:

        # For keeping the state
        if keep_context:
            self.h_0, self.c_0 = h_0, c_0
            return

        # Reset the state
        self.__reset_hidden_state()

    def forward(
        self,
        x: torch.Tensor,
        h_0: torch.Tensor | None = None,
        c_0: torch.Tensor | None = None,
    ):
        # Note: does not reset hidden states

        # Initialize hidden state
        h_0, c_0 = self.__initialize_hidden_states(
            batch_size=x.size(0), h_0=h_0, c_0=c_0
        )

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

        return self.fc(last_time_step_out), (h_0, c_0)

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

                # Forward pass
                outputs, (h_0, c_0) = self(batch_input, self.h_0, self.c_0)

                loss = criterion(outputs, batch_target)
                epoch_loss += loss.item()

                # Update or reset the hidden state
                self.__update_hidden_state(h_0=h_0, c_0=c_0, keep_context=False)

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

    # TODO: predict stateless or statefull... initiliazie hidden state and reset after every forward pass or not
    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
        keep_context: bool = True,
    ):

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
                pred, (h_0, c_0) = self(window, self.h_0, self.c_0)

                # Reset or upate hiden state based on keep_context value
                self.__update_hidden_state(h_0=h_0, c_0=c_0, keep_context=keep_context)

                predictions.append(pred.cpu())

            current_window = input_sequence[-sequence_length:].unsqueeze(0)

            # Predict new data using autoregression
            for _ in range(to_predict_years_num):
                logger.debug(f"Current input window: {current_window}")

                # Forward pass
                pred, (h_0, c_0) = self(
                    current_window, self.h_0, self.c_0
                )  # Shape: (1, output_size)

                self.__update_hidden_state(h_0=h_0, c_0=c_0, keep_context=keep_context)

                pred_value = pred.squeeze(0)  # Remove batch dim
                logger.debug(f"New prediction value: {pred_value}")

                predictions.append(pred.cpu())  # Store new prediction
                new_predictions.append(pred.cpu())

                # Shift the window by removing the first value and adding the new prediction
                current_window = torch.cat(
                    (current_window[:, 1:, :], pred.unsqueeze(0)), dim=1
                )

        # Reset the hidden state for next prediction
        self.__reset_hidden_state()

        # Combine with years
        for year, pred in zip(range(last_year + 1, target_year + 1), new_predictions):
            logger.debug(f"{year}: {pred}")

        new_predictions_tensor = torch.cat(new_predictions, dim=0)
        return new_predictions_tensor


if __name__ == "__main__":
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
