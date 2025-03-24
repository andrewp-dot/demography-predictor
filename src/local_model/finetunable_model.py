# Standard library imports
import pandas as pd
import logging
import torch
from torch import nn
from typing import Tuple, List, Union
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model
from src.local_model.base import LSTMHyperparameters, TrainingStats, EvaluateModel
from src.local_model.base import CustomModelBase
from src.local_model.model import BaseLSTM

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
# 1. Where to put resseting hidden state function?
# 2. Add adapter layer for change the size shape of finetunable layer


class FineTunableLSTM(CustomModelBase):

    def __init__(
        self,
        base_model: BaseLSTM,
        hyperparameters: LSTMHyperparameters,
    ):
        super(FineTunableLSTM, self).__init__(
            features=self.base_model.FEATURES,
            targets=self.base_model.TARGETS,
            hyperparameters=hyperparameters,
            scaler=self.base_model.SCALER,
        )

        # Get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained LSTM layers
        self.base_model = base_model
        self.base_lstm = base_model.lstm

        # Freeze pretrained LSTM layers
        for param in self.base_model.lstm.parameters():
            param.requires_grad = False

        # Add new LSTM layers for fine-tuning
        # TODO: add layer to get different number of neurons
        fine_tune_hidden_size = hyperparameters.hidden_size

        # Hidden layer for transforming the hidden size of the base model to hidden size of the new lstm model
        self.hidden_transform = nn.Linear(
            base_model.lstm.hidden_size, fine_tune_hidden_size
        )

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
                self.base_model.hyperparameters.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

        # Initiliaze cell state
        if c_0 is None:
            c_0 = torch.zeros(
                self.base_model.hyperparameters.num_layers
                + self.hyperparameters.num_layers
                * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
                batch_size,
                self.hyperparameters.hidden_size,
                dtype=torch.float32,
            ).to(self.device)

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

        # Initialize hidden state
        self.h_0, self.c_0 = self.__initialize_hidden_states(
            batch_size=x.size(0), h_0=h_0, c_0=c_0
        )

        # Move hidden states to device
        x = x.to(device=self.device)
        self.h_0 = self.h_0.to(self.device)
        self.c_0 = self.c_0.to(self.device)

        # Forward pass for the base model, skip gradiend calculation -> freeze base model lstm layers
        with torch.no_grad():

            old_lstm_out, _ = self.base_lstm(
                x,
                (
                    self.h_0[: self.base_lstm.num_layers],
                    self.c_0[: self.base_lstm.num_layers],
                ),
            )

        # Put the out to the device
        old_lstm_out.to(device=self.device)

        # Get the output of new lstm
        new_lstm_out, _ = self.new_lstm(
            old_lstm_out,
            (
                self.h_0[self.base_lstm.num_layers :],
                self.c_0[self.base_lstm.num_layers :],
            ),
        )

        # Get the output of new lstm
        new_lstm_out.to(device=self.device)

        # Get last time step
        last_time_step_out = new_lstm_out[:, -1, :]

        return self.fc(last_time_step_out), (h_0, c_0)

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
        loss_function: Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss] = None,
    ):
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
                window = window.unsqueeze(0).to(
                    self.device
                )  # Add batch dimension: (1, sequence_length, input_size)

                # Put the input to the device
                window.to(device=self.device)

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


def train_base_model(
    hyperparameters: LSTMHyperparameters,
    features: List[str],
    evaluation_state_name: str,
) -> BaseLSTM:

    MODEL_NAME = f"base_model_{hyperparameters.hidden_size}.pkl"

    # Set features const
    FEATURES = features

    try:
        model: BaseLSTM = get_model(MODEL_NAME)
        logger.info(f"Base model already exist. Model name: {MODEL_NAME}")

        # TODO: change this
        if model.FEATURES != FEATURES:
            raise ValueError(
                "The Base model has incompatibile features with the one you want"
            )

        return model
    except ValueError as e:
        logger.info(f"{str(e)}. Training new model.")

    # Load data
    all_states_loader = StatesDataLoader()
    all_states_dict = all_states_loader.load_all_states()

    # Get training and test data
    train_data_dict, test_data_dict = all_states_loader.split_data(
        states_dict=all_states_dict,
        sequence_len=hyperparameters.sequence_length,
        split_rate=0.8,
    )

    # Preprocess training data
    trainig_batches, target_batches, base_fitted_scaler = (
        all_states_loader.preprocess_train_data_batches(
            states_train_data_dict=train_data_dict,
            hyperparameters=hyperparameters,
            features=FEATURES,
        )
    )

    # Create model
    base_model = BaseLSTM(hyperparameters=hyperparameters, features=FEATURES)
    base_model.set_scaler(scaler=base_fitted_scaler)

    # Train model using whole dataset
    logger.info("Training base model...")
    base_model.train_model(
        batch_inputs=trainig_batches,
        batch_targets=target_batches,
        display_nth_epoch=1,
    )

    # Get trainig stats
    loss_plot_figure = base_model.training_stats.create_plot()

    # Evaluate base model
    logger.info("Evaluating base model...")
    model_evaluation = EvaluateModel(base_model)
    model_evaluation.eval(
        test_X=train_data_dict[evaluation_state_name],
        test_y=test_data_dict[evaluation_state_name],
        features=FEATURES,
        scaler=base_fitted_scaler,
    )

    # Get predictions plot
    base_predictions_plot = model_evaluation.plot_predictions()

    # Print evaluation metrics
    logger.info(
        f"[BaseModel]: Overall evaluation metrics:\n{model_evaluation.overall_metrics}\n"
    )
    logger.info(
        f"[BaseModel]: Per features evaluation metrics:\n{model_evaluation.per_target_metrics}\n"
    )

    return base_model


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    FEATURES = [
        # "year",
        "Fertility rate, total",
        # "Population, total",
        # "Net migration",
        "Arable land",
        "Birth rate, crude",
        "GDP growth",
        "Death rate, crude",
        "Agricultural land",
        "Rural population",
        "Rural population growth",
        "Age dependency ratio",
        "Urban population",
        "Population growth",
        "Adolescent fertility rate",
        "Life expectancy at birth, total",
    ]

    FEATURES = [col.lower() for col in FEATURES]
    EVLAUATION_STATE_NAME = "Croatia"

    hyperparameters = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=512,
        sequence_length=12,
        learning_rate=0.0001,
        epochs=40,
        batch_size=16,
        num_layers=3,
    )

    # Create and train base model
    base_model = train_base_model(
        hyperparameters=hyperparameters,
        features=FEATURES,
        evaluation_state_name=EVLAUATION_STATE_NAME,
    )

    # Save base model
    save_model(base_model, f"base_model_{base_model.hyperparameters.hidden_size}.pkl")

    # Create finetunable model
    finetunable_hyperparameters = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=hyperparameters.hidden_size,  # Yet the base model hidden size and finetunable layer hidden size has to be the same
        sequence_length=hyperparameters.sequence_length,
        learning_rate=0.0001,
        epochs=50,
        batch_size=1,
        num_layers=2,
    )

    finetunable_local_model = FineTunableLSTM(
        base_model=base_model, hyperparameters=finetunable_hyperparameters
    )

    # Load data
    state_loader = StateDataLoader(EVLAUATION_STATE_NAME)
    state_data_df = state_loader.load_data()

    # Split state data
    train_state_data_df, test_state_data_df = state_loader.split_data(
        data=state_data_df, split_rate=0.8
    )

    # Use the same scaler as the base model
    trainig_batches, target_batches, _ = state_loader.preprocess_training_data_batches(
        train_data_df=train_state_data_df,
        hyperparameters=finetunable_hyperparameters,
        features=FEATURES,
        scaler=base_model.scaler,
    )

    logger.info("Finetuning model...")
    finetunable_local_model.train_model(
        batch_inputs=trainig_batches, batch_targets=target_batches, display_nth_epoch=1
    )

    # Evaluate finetunable model
    logger.info("Evaluating finetunable model...")
    finetunable_model_evaluation = EvaluateModel(finetunable_local_model)
    finetunable_model_evaluation.eval(
        test_X=train_state_data_df,
        test_y=test_state_data_df,
        features=FEATURES,
        scaler=base_model.scaler,
    )

    logger.info(
        f"[FinetunabelModel]: Overall metrics:\n{finetunable_model_evaluation.overall_metrics}\n"
    )
    logger.info(
        f"[FinetunabelModel]: Per feature metrics:\n{finetunable_model_evaluation.per_target_metrics}\n"
    )

    fig = finetunable_model_evaluation.plot_predictions()
    fig.show()

    # Save model
    save_model(
        finetunable_local_model, f"finetunable_model_{EVLAUATION_STATE_NAME}.pkl"
    )
