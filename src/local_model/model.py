# Standard library imports
import pandas as pd
from typing import Tuple, Dict, List, Callable
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
import logging
import pprint

# Custom imports
from preprocessors.preprocessing import StateDataLoader
from config import setup_logging


logger = logging.getLogger("local_model")


# Set the seed for reproducibility
torch.manual_seed(42)

# TODO:
# Implement training stats (plotting etc.)


class LSTMHyperparameters:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        num_layers: int,
        bidirectional: bool = False,
    ):
        self.input_size = input_size  # Number of input features
        self.hidden_size = hidden_size  # Number of hidden units in the LSTM layer
        self.sequence_length = sequence_length  # Length of the input sequence, i.e., how many time steps the model will look back
        self.learning_rate = (
            learning_rate  # How much the model is learning from the data
        )
        self.epochs = epochs  # Number of times the entire dataset is passed forward and backward through the neural network
        self.batch_size = (
            batch_size  # Number of samples processed in one forward/backward pass
        )
        self.num_layers = num_layers  # Number of LSTM layers
        self.bidirectional = bidirectional


class TrainingStats:

    def __init__(self):
        self.losses = []
        # self.mse = []
        self.epochs = []

    def create_plot(self) -> Figure:
        """
        Creates a figure of training statistics.

        :returns: Figure: figure of training statistics.
        """

        # TODO: change the plot function, to plot multiple metrcis

        # Define figure parameters
        fig = plt.figure(figsize=(10, 5))

        # Name the axis
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Plot the graph(s)
        plt.plot(self.epochs, self.losses)

        # Show the plot
        # plt.legend()
        plt.grid()

        return fig

    def plot(self) -> None:
        fig: Figure = self.create_plot()
        fig.show()


class LocalModel(nn.Module):

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
        # if hidden_state is None:
        h_t, c_t = self.__initialize_hidden_states(batch_size)
        # else:
        #     # If hidden state is passed, use it
        #     h_t, c_t = hidden_state

        # Forward propagate through LSTM
        out, (h_n, c_n) = self.lstm(x, (h_t, c_t))  # In here 'out' is a tensor

        # You can detach the hidden state here if you need to avoid backprop through the entire sequence
        # h_n = h_n.detach()
        # c_n = c_n.detach()

        # Use the output from the last time step
        out = self.linear(out[:, -1, :])  # Using the last time step's output

        # Return both output and the updated hidden state (for the next batch)
        # return out, (h_n, c_n)
        return out

    def train_model(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor):
        """
        Trains model using batched input sequences and batched target sequences.


        :param batch_inputs: torch.Tensor: batches of input sequences
        :param batch_targets: torch.Tensor: batches of target sequences
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
                # logger.debug(f"[Training loop] input: {batch_input.shape}")
                # logger.debug(f"[Training loop] target: {batch_target.shape}")

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(batch_inputs)
            self.training_stats.losses.append(epoch_loss)

            if not epoch % 10:
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
                # logger.debug(
                #     f"Input sequence: {input_sequence[i : i + sequence_length]}"
                # )
                window = input_sequence[i : i + sequence_length]  # Extract window
                window = window.unsqueeze(
                    0
                )  # Add batch dimension: (1, sequence_length, input_size)

                print("-" * 100)
                print("Input:")
                pprint.pprint(window)

                pred = self(window)  # Forward pass

                print("Output:")
                pprint.pprint(pred.cpu())
                print("-" * 100)

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

        prediction_tensor = torch.cat(predictions, dim=0)  # Combine all predictions
        logger.info(f"Prediction tensor: {prediction_tensor}")

        # Combine with years
        for year, pred in zip(range(last_year + 1, target_year + 1), new_predictions):
            logger.debug(f"{year}: {pred}")

        new_predictions_tensor = torch.cat(new_predictions, dim=0)
        return new_predictions_tensor


class EvaluateLSTM:

    def __init__(self, model: nn.Module):
        self.model: LocalModel = model
        self.predicted: pd.DataFrame | None = None
        self.reference_values: pd.DataFrame | None = None

        # Define metric dataframes
        self.per_target_metrics: pd.DataFrame | None = None
        self.overall_metrics: pd.DataFrame | None = None

    def __get_metric(
        self,
        metric: Callable,
        features: List[str],
        metric_key: str = "",
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Computes and saves given metric.


        :param metric: callable: Function for metric computation.
        :param metric_key: (str, optional): Key of the metric which can be accasible in `EvaluateLSTM.metrics` dict (`{metric_key}`, `{metric_key}_per_target` if per target is available).
            If not given, the name of the function is used. Defaults to "".

        :returns: Tuple[pd.DataFrame, pd.DataFrame | None]: metric value for all targets, metric values for each target separately.
        """

        # Adjust metric key if not given
        metric_key = metric_key or metric.__name__
        FEATURES = features

        # Initialize metric values
        overall_metric_df = None
        separate_target_metric_values_df = None

        if metric != r2_score:
            # Compute metric for each target separately
            metric_per_target_values = metric(
                self.reference_values, self.predicted, multioutput="raw_values"
            )

            # Get metric values for separate targets
            metric_per_target_values_dict = {
                "feature": FEATURES,
                metric_key: metric_per_target_values,
            }

            separate_target_metric_values_df = pd.DataFrame(
                metric_per_target_values_dict
            )

        # Average MAE across all targets
        average_metric_value = metric(
            self.reference_values, self.predicted, multioutput="uniform_average"
        )

        # Get overall metric dataframe
        overall_metric_df = pd.DataFrame(
            {"metric": metric_key, "value": [average_metric_value]}
        )

        return overall_metric_df, separate_target_metric_values_df

    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        features: list[str],
        scaler: MinMaxScaler | RobustScaler | StandardScaler,
    ) -> None:
        """Evaluates model perforemance based on known and unknown sequences.

        :param test_X: pd.DataFrame: unscaled validation input data (known sequences)
        :param test_y: pd.DataFrame: unscaled validation target data (unknown sequences)
        :param features: list[str]: input features list
        :param scaler: (MinMaxScaler | RobustScaler | StandardScaler): scaler to scale data
        """

        # Set features as a constant
        FEATURES = features

        # Get the last year and get the number of years
        X_years = test_X[["year"]]
        last_year = int(X_years.iloc[-1].item())

        # Get the prediction year
        y_target_years = test_y[["year"]]
        target_year = int(y_target_years.iloc[-1].item())

        # Scale data
        test_X = scaler.transform(test_X)
        test_y = scaler.transform(test_y)

        # Transform data back to dataframe
        test_X = pd.DataFrame(test_X, columns=FEATURES)
        test_y = pd.DataFrame(test_y, columns=FEATURES)

        # Generate predictions
        logger.debug(f"[Eval]: predicting values from {last_year} to {target_year}...")
        predictions = self.model.predict(
            input_data=test_X[FEATURES],
            last_year=last_year,
            target_year=target_year,
        )

        # Save predictions
        predictions_df = pd.DataFrame(predictions, columns=test_y.columns)
        self.predicted = scaler.inverse_transform(predictions_df)

        # Save true values
        self.reference_values = scaler.inverse_transform(test_y)

        logger.debug(f"[Eval]: predictions shape: {predictions.shape}")

        # Get the real value of the predicions
        # denormalized_predictions = scaler.inverse_transform(predictions)

        # Create dataframe from predictions the real value of the predicions
        # denormalized_predicions_df = pd.DataFrame(
        #     denormalized_predictions, columns=test_X.columns
        # )

        # Get MAE
        overall_mae_df, mae_per_target_df = self.__get_metric(
            mean_absolute_error, FEATURES, "mae"
        )

        # Get MSE
        overall_mse_df, mse_per_target_df = self.__get_metric(
            mean_squared_error, FEATURES, "mse"
        )

        # Get RMSE
        overall_rmse_df, rmse_per_target_df = self.__get_metric(
            root_mean_squared_error, FEATURES, "rmse"
        )

        # Get R^2
        overall_r2_df, _ = self.__get_metric(r2_score, FEATURES, "r2")

        # Create per target dataframe
        mae_mse_df = pd.merge(
            left=mae_per_target_df, right=mse_per_target_df, on="feature"
        )
        self.per_target_metrics = pd.merge(
            left=mae_mse_df, right=rmse_per_target_df, on="feature"
        )

        # Get overall dataframe
        self.overall_metrics = pd.concat(
            [overall_mae_df, overall_mse_df, overall_rmse_df, overall_r2_df],
            axis=0,
        )


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    # Setup logging
    setup_logging()

    FEATURES = [
        "year",
        # "Fertility rate, total",
        "Population, total",
        # "Net migration",
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
        hidden_size=128,
        sequence_length=10,
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
        czech_data[FEATURES], scaler=MinMaxScaler()
    )

    print(scaled_cz_data.head())

    train, test = czech_loader.split_data(scaled_cz_data)

    # Get input/output sequences from train data
    input_sequences, target_sequences = czech_loader.preprocess_training_data(
        train, hyperparameters.sequence_length, features=FEATURES
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
    rnn.train_model(batch_inputs=batch_inputs, batch_targets=batch_targets)

    # Evaluate model
    model_evaluation = EvaluateLSTM(rnn)

    # Split the raw data
    val_X, val_y = czech_loader.split_data(czech_data[FEATURES])

    model_evaluation.eval(
        test_X=val_X, test_y=val_y, features=FEATURES, scaler=cz_scaler
    )

    logger.info(
        f"Model evaluation by per target metrics: \n{model_evaluation.per_target_metrics}"
    )
    logger.info(
        f"Model evaluation by overall metrics: \n{model_evaluation.overall_metrics}"
    )

    exit(1)

    # Get the last year and get the number of years
    years = czech_data[["year"]]
    last_year = int(years.iloc[-1].item())
    target_year = 2100

    predictions = rnn.predict(
        input_data=scaled_cz_data[FEATURES],
        last_year=last_year,
        target_year=target_year,
    )

    predictions_df = pd.DataFrame(predictions, columns=FEATURES)
    denormalized_predicions = cz_scaler.inverse_transform(predictions)

    denormalized_predicions_df = pd.DataFrame(denormalized_predicions, columns=FEATURES)

    if FEATURES == ["year"]:
        print("-" * 100)
        print("Predictions:")

        # Just print predictions in fancier way
        first_predicted_year = int(
            years.iloc[0 + hyperparameters.sequence_length].item()
        )
        for index, year in enumerate(
            range(first_predicted_year, target_year + 1)
        ):  # + 1 in order to include also the target year
            print(f"{year}: {denormalized_predicions_df.iloc[index,0]}")

    # Get train stats
    stats = rnn.training_stats

    print("-" * 100)
    print("Losses:")
    pprint.pprint(stats.losses)

    stats.plot()
