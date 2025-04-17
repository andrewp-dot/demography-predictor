import pandas as pd
from typing import Tuple, Dict, List, Callable, Union
import pprint
import logging

import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler


# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.utils.save_model import save_model

from src.base import LSTMHyperparameters, TrainingStats, CustomModelBase

# from src.evaluation import EvaluateModel
from src.state_groups import StatesByWealth

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.data_transformer import DataTransformer


logger = logging.getLogger("local_model")


class ExpLSTM(CustomModelBase):

    def __init__(
        self,
        hyperparameters: LSTMHyperparameters,
        features: List[str],
        targets: List[str] | None = None,
    ):
        super(ExpLSTM, self).__init__(
            features=features,
            targets=(targets if targets else features),
            hyperparameters=hyperparameters,
            scaler=None,
        )

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

        # Dropout layer (e.g., 0.3 dropout rate)
        self.dropout = nn.Dropout(p=0.3)

        # Additional linear layers
        self.fc1 = nn.Linear(
            hyperparameters.hidden_size, hyperparameters.hidden_size // 2
        )
        self.relu = nn.ReLU()

        # Out layer: Linear layer
        self.linear = nn.Linear(
            in_features=hyperparameters.hidden_size // 2,
            out_features=hyperparameters.output_size,
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

        out = self.dropout(out)

        out = self.fc1(out)
        out = self.relu(out)

        out = self.linear(out)  # Using the last time step's output

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

    def shap_predict(self, input_tensor: pd.DataFrame) -> torch.Tensor:
        # Single prediction for shap
        with torch.no_grad():
            pred = self(input_tensor)

        return pred

        # Predict the future values
        # new_input_data: pd.DataFrame = intial_data
        # for year in range(last_year + 1, target_year + 1):

        #     # you need to append it to a sequence
        #     next_year_targets = self.predict(input_data=intial_data)

        #     # Create a dataframe from it
        #     next_year_targts_df = pd.DataFrame(next_year_targets, columns=self.TARGETS)

        #     # Add the predictions to the known values
        #     new_input_data = pd.concat()

    # TODO:
    # Rewrite this for prediction -> input (scaled data - tensor?) -> output (scaled data -> tensor?)
    # Make prediction method in pipeline
    def predict(
        self,
        input_data: pd.DataFrame,
    ) -> torch.Tensor:
        """
         Predicts values using past data. 2 cycles: 1st to gain context for all past data. 2nd is used to generate predictions.

        Args:
            input_data (torch.Tensor): Preprocessed (scaled) input data.

        Returns:
            out: torch.Tensor: Next year targets predictions.
        """

        # Put the model into the evaluation mode
        self.eval()

        logger.debug(f"Input sequence: {input_data.shape}")

        # Move input to the appropriate device
        input_sequence = torch.tensor(data=input_data.values, dtype=torch.float32)

        input_sequence.to(self.device)
        self.to(device=self.device)

        sequence_length = self.hyperparameters.sequence_length

        pred_value = None
        with torch.no_grad():

            current_window = (
                input_sequence[-sequence_length:].unsqueeze(0).to(device=self.device)
            )

            # Predict new data using autoregression
            # for _ in range(to_predict_years_num):
            logger.debug(f"Current input window: {current_window}")

            # Forward pass
            pred = self(current_window)  # Shape: (1, output_size)
            pred_value = pred.squeeze(0)  # Remove batch dim
            logger.debug(f"New prediction value: {pred_value}")

            pred_value = pred.cpu()
            # predictions.append(pred.cpu())  # Store new prediction

        return pred_value


def main(save_plots: bool = True, to_save_model: bool = False, epochs: int = 50):
    # Setup logging
    setup_logging()

    FEATURES = [
        col.lower()
        for col in [
            "year",
            "Fertility rate, total",
            # "Population, total",
            "Net migration",
            "Arable land",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            "Rural population growth",
            "Urban population",
            "Population growth",
            # Features with high correlation
            # "Birth rate, crude",
            # "Rural population",
            # "Age dependency ratio",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
        ]
    ]

    # FEATURES: List[str] = [
    #     "year",
    #     "fertility rate, total",
    #     "arable land",
    #     "gdp growth",
    #     "death rate, crude",
    #     "agricultural land",
    #     "rural population growth",
    #     "urban population",
    #     "population growth",
    # ]

    TARGETS: List[str] = [
        # "population, total",
        # Aging targets
        "population ages 15-64",
        "population ages 0-14",
        "population ages 65 and above",
        # Gender targets
        # "population, female",
        # "population, male",
    ]

    WHOLE_MODEL_FEATURES: List[str] = FEATURES + TARGETS

    TARGETS = None

    # Setup model
    hyperparameters = get_core_hyperparameters(
        input_size=len(WHOLE_MODEL_FEATURES),
        epochs=epochs,
        batch_size=32,
        # output_size=len(TARGETS),  # TODO: make this more robust
        output_size=len(WHOLE_MODEL_FEATURES),  # TODO: make this more robust
    )
    rnn = ExpLSTM(hyperparameters, features=WHOLE_MODEL_FEATURES, targets=TARGETS)

    # Load data
    states_loader = StatesDataLoader()
    states_data_dict = states_loader.load_all_states()

    # states_data_dict = states_loader.load_states(states=StatesByWealth().high_income)

    # Get training data and validation data
    train_data_dict, test_data_dict = states_loader.split_data(
        states_dict=states_data_dict, sequence_len=hyperparameters.sequence_length
    )
    train_states_df = states_loader.merge_states(train_data_dict)
    test_states_df = states_loader.merge_states(test_data_dict)

    # Transform data
    transformer = DataTransformer()

    # This is useless maybe -> just for fitting the scaler on training data after transformation in datatransforme
    scaled_training_data, scaled_test_data = transformer.scale_and_fit(
        training_data=train_states_df,
        validation_data=test_states_df,
        features=FEATURES,
        targets=TARGETS,
    )

    # Create a dictionary from it
    scaled_states_dict = states_loader.parse_states(scaled_training_data)

    batch_inputs, batch_targets, batch_validation_inputs, batch_validation_targets = (
        transformer.create_train_test_multiple_states_batches(
            data=scaled_states_dict,
            hyperparameters=hyperparameters,
            features=WHOLE_MODEL_FEATURES,
            targets=TARGETS,
        )
    )

    print("-" * 100)
    logger.info(f"Batch inputs shape: {batch_inputs.shape}")
    logger.info(f"Batch targets shape: {batch_targets.shape}")

    # Train model
    stats = rnn.train_model(
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        batch_validation_inputs=batch_validation_inputs,
        batch_validation_targets=batch_validation_targets,
        display_nth_epoch=1,
    )

    training_stats = TrainingStats.from_dict(stats_dict=stats)

    if save_plots:
        fig = training_stats.create_plot()
        fig.savefig(f"BaseLSTM_training_stats_{hyperparameters.epochs}_epochs.png")

    if to_save_model:
        save_model(name=f"ExpLSTM_pop_total.pkl", model=rnn)
        save_model(name=f"ExpLSTM__pop_total_transformer.pkl", model=transformer)


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    main(save_plots=True, to_save_model=True, epochs=10)
