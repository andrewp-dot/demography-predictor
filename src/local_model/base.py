# Standard library imports
import pandas as pd
from abc import abstractmethod
from typing import Tuple, Dict, List, Callable, Union
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
from config import setup_logging


logger = logging.getLogger("local_model")


# Set the seed for reproducibility
torch.manual_seed(42)


class LSTMHyperparameters:

    def __init__(
        self,
        input_size: int,
        # output_size: int,
        hidden_size: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        num_layers: int,
        bidirectional: bool = False,
    ):
        """
        Define parameters for LSTM networks

        Args:
            input_size (int): Defines the number of input features.
            output_size (int): Defines the number of output features.
            hidden_size (int): Defines the number of neurons in a layer.
            sequence_length (int): Length of the processing sequnece (number of past samples using for predicition).
            learning_rate (float): Defines how much does the model learn (step in gradient descend).
            epochs (int): Number of epochs to train the nerual network.
            batch_size (int): Number of samples used to update the weights in the neural network. Bigger batch size for faster training and better generalization.
            num_layers (int): Number of LSTM layers (or LSTM combined layers in neura networks). In case of FineTunable networks, defines the number of finetunable layers.
            bidirectional (bool, optional): If you can go also forward and backward (for gatharing context from the past and also from future). Defaults to False.
        """
        self.input_size = input_size  # Number of input features
        # self.output_size = output_size  # Number of output features
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

    def __repr__(self) -> str:

        repr_string = f"""
Input size:         {self.input_size}
Batch size:         {self.batch_size}

Hidden size:        {self.hidden_size}
Sequence length:    {self.sequence_length}
Layers:             {self.num_layers}

Learning rate:      {self.learning_rate}
Epochs:             {self.epochs}

Bidirectional:      {self.bidirectional}
"""
        return repr_string


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


class BaseEvaluation:

    def __init__(self, *args, **kwargs):
        self.predicted: pd.DataFrame | None = None
        self.reference_values: pd.DataFrame | None = None
        self.predicted_years: List[int] | None = None

        # Define metric dataframes
        self.per_target_metrics: pd.DataFrame | None = None
        self.overall_metrics: pd.DataFrame | None = None

    def get_overall_metric(
        self,
        metric: Callable,
        metric_key: str = "",
    ) -> pd.DataFrame:
        """
        Computes and saves given metric.

        Args:
            metric (Callable): Function for metric computation.
            features (List[str]): Key of the metric which can be accasible in `EvaluateModel.metrics` dict (`{metric_key}`, `{metric_key}_per_target` if per target is available).
                If not given, the name of the function is used. Defaults to "".
            metric_key (str, optional): _description_. Defaults to "".

        Returns:
            out: Tuple[pd.DataFrame, pd.DataFrame | None]: metric value for all targets, metric values for each target separately.
        """

        # Adjust metric key if not given
        metric_key = metric_key or metric.__name__

        # Initialize metric values
        overall_metric_df = None

        # Average MAE across all targets
        average_metric_value = metric(
            self.reference_values, self.predicted, multioutput="uniform_average"
        )

        # Get overall metric dataframe
        overall_metric_df = pd.DataFrame(
            {"metric": metric_key, "value": [average_metric_value]}
        )

        return overall_metric_df

    def get_overall_metrics(self) -> None:
        # Get MAE
        overall_mae_df = self.get_overall_metric(mean_absolute_error, "mae")

        # Get MSE
        overall_mse_df = self.get_overall_metric(mean_squared_error, "mse")

        # Get RMSE
        overall_rmse_df = self.get_overall_metric(root_mean_squared_error, "rmse")

        # Get R^2
        overall_r2_df = self.get_overall_metric(r2_score, "r2")

        # Get overall dataframe
        self.overall_metrics = pd.concat(
            [overall_mae_df, overall_mse_df, overall_rmse_df, overall_r2_df],
            axis=0,
            ignore_index=True,
        )

    def get_feautre_specific_metric(
        self,
        metric: Callable,
        features: List[str],
        metric_key: str = "",
    ) -> pd.DataFrame:
        """
        Computes and saves given metric.

        Args:
            metric (Callable): Function for metric computation.
            features (List[str]): Key of the metric which can be accasible in `EvaluateModel.metrics` dict (`{metric_key}`, `{metric_key}_per_target` if per target is available).
                If not given, the name of the function is used. Defaults to "".
            metric_key (str, optional): _description_. Defaults to "".

        Returns:
            out: pd.DataFrame: metric value for all targets, metric values for each target separately.
        """

        # Adjust metric key if not given
        metric_key = metric_key or metric.__name__
        FEATURES = features

        # Initialize metric values
        separate_target_metric_values_df = None

        # TODO: fix this for r2
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

        return separate_target_metric_values_df

    def get_feature_specific_metrics(self, features: List[str]) -> None:
        """
        Creates and saves evaluation dataframe for every predicted features. Saves as a property 'per_target_metrics' of the class.

        Args:
            features (List[str]): List of features used for evaluation.
        """

        # Set features constant
        FEATURES = features

        # Get MAE
        mae_per_target_df = self.get_feautre_specific_metric(
            mean_absolute_error, FEATURES, "mae"
        )

        # Get MSE
        mse_per_target_df = self.get_feautre_specific_metric(
            mean_squared_error, FEATURES, "mse"
        )

        # Get RMSE
        rmse_per_target_df = self.get_feautre_specific_metric(
            root_mean_squared_error, FEATURES, "rmse"
        )

        # Get R^2
        # _ = self.get_feautre_specific_metric(r2_score, FEATURES, "r2")

        # Create per target dataframe
        mae_mse_df = pd.merge(
            left=mae_per_target_df, right=mse_per_target_df, on="feature"
        )
        self.per_target_metrics = pd.merge(
            left=mae_mse_df, right=rmse_per_target_df, on="feature"
        )

    def to_readable_dict(self) -> Dict[str, float]:
        """
        Converts the overall metrics dataframe to readable dict.

        Returns:
            out: Dict[str, float]: Readable dict, the metric names are the keys and the values are the corresponding metric value.
        """

        # Get the overall metrics
        df = self.overall_metrics

        # Convert the metrics dataframe
        result = dict(zip(df["metric"], df["value"]))

        return result

    @abstractmethod
    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        features: list[str],
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    def plot_predictions(self) -> Figure:

        # Get years
        YEARS = self.predicted_years

        # Get the feature data to create plots
        FEATURES = list(self.predicted.columns)
        N_FEATURES = len(FEATURES)

        # Create a figure with N rows and 1 column
        fig, axes = plt.subplots(N_FEATURES, 1, figsize=(8, 2 * N_FEATURES))

        # Ensure axes is always iterable
        if N_FEATURES == 1:
            axes = [axes]  # Convert to list for consistent indexing

        # Plotting in each subplot
        for index, feature in zip(range(N_FEATURES), FEATURES):
            # Plot reference values
            axes[index].plot(
                YEARS,
                self.reference_values[feature],
                label=f"Reference values",
                color="b",
            )

            # Plot predicted values
            axes[index].plot(
                YEARS,
                self.predicted[feature],
                label=f"Predicted",
                color="r",
            )
            axes[index].set_title(f"{feature}")
            axes[index].set_xlabel("Years")
            axes[index].set_ylabel("Value")
            axes[index].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.grid()
        plt.legend()

        return fig


class EvaluateModel(BaseEvaluation):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model: CustomModelBase = model

    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        features: list[str],
        scaler: Union[MinMaxScaler | RobustScaler | StandardScaler],
    ) -> None:
        """
        Evaluates model perforemance based on known and unknown sequences of 1 state data. Needs the year column in order to compute number of prediction iterations.

        Args:
            test_X (pd.DataFrame,): Unscaled validation input data (known sequences)
            test_y (pd.DataFrame,): Unscaled validation target data (unknown sequences)
            features (list[str]): Features names used to predict data.
            scaler (Union[MinMaxScaler  |  RobustScaler  |  StandardScaler]): Scaler to scale data
            from_year (int): Last record's year in test_X data.
            to_year (int): Last record's of the 'test_y' data.
        """

        # Set features as a constant
        FEATURES = features

        # Get the last year and get the number of years
        X_years = test_X[["year"]]
        last_year = int(X_years.iloc[-1].item())

        # # Get the prediction year
        y_target_years = test_y[["year"]]
        target_year = int(y_target_years.iloc[-1].item())

        # Exctract the features seen in fit time
        test_X = test_X[FEATURES]
        test_y = test_y[FEATURES]

        # Scale data
        test_X = scaler.transform(test_X)
        test_y = scaler.transform(test_y)

        # Transform data back to dataframe
        test_X = pd.DataFrame(test_X, columns=FEATURES)
        test_y = pd.DataFrame(test_y, columns=FEATURES)

        # Generate predictions
        self.predicted_years = list(
            range(last_year + 1, target_year + 1)
        )  # values are predicted from next year after last year and for the target year (+ 1 to include target year)
        logger.debug(f"[Eval]: predicting values from {last_year} to {target_year}...")

        predictions = self.model.predict(
            input_data=test_X,
            last_year=last_year,
            target_year=target_year,
        )

        # Save predictions
        predictions_df = pd.DataFrame(predictions, columns=test_y.columns)
        self.predicted = pd.DataFrame(
            scaler.inverse_transform(predictions_df), columns=test_y.columns
        )

        # Save true values
        self.reference_values = pd.DataFrame(
            scaler.inverse_transform(test_y), columns=test_y.columns
        )

        logger.debug(f"[Eval]: predictions shape: {predictions.shape}")

        # Get metrics
        self.get_feature_specific_metrics(features=FEATURES)
        self.get_overall_metrics()

    def is_new_better(self, new_model_evaluation: "EvaluateModel") -> bool:
        """
        Compares 2 model evaluations.

        Args:
            new_model_evaluation (EvaluateModel): _description_

        Returns:
            out: bool: True if the new model evaluation is better according to metrics. False, if the old model is better or has same performance as the new one.
        """

        # Votes sums
        votes: List[bool] = []

        # Compare by error metrics -> the lower, the better
        # error_metrics: List[str] = ["mae", "mse", "rmse"]
        error_metrics: List[str] = []

        for metric in error_metrics:

            # Old model metric
            current_model_metric_value = self.overall_metrics.loc[
                self.overall_metrics["metric"] == metric, "value"
            ].values[0]

            # New model metric
            new_model_metric_value = new_model_evaluation.overall_metrics.loc[
                new_model_evaluation.overall_metrics["metric"] == metric, "value"
            ].values[0]

            # Add vote if the new model has lower error metric
            votes.append(current_model_metric_value > new_model_metric_value)

        # Compare by r2 -> the higher, the better
        current_model_r2_score = self.overall_metrics.loc[
            self.overall_metrics["metric"] == "r2", "value"
        ].values[0]

        new_model_r2_score = new_model_evaluation.overall_metrics.loc[
            new_model_evaluation.overall_metrics["metric"] == "r2", "value"
        ].values[0]

        # Add vote if new model has greater r2 score
        votes.append(current_model_r2_score < new_model_r2_score)

        # The model is better if it outperforms the model in more then half metrics
        return sum(votes) / len(votes) > 0.5


class CustomModelBase(nn.Module):
    """
    Defines the interface for the neural networks models in this project. From this

    Raises:
        NotImplementedError: If forward method is not implemented yet.
        NotImplementedError: If train_model method is not implemented yet.
        NotImplementedError: If predict method is not implemented yet.
    """

    def __init__(self, hyperparameters: LSTMHyperparameters, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hyperparameters: LSTMHyperparameters = hyperparameters

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method for your model is not implemented!")

    @abstractmethod
    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ) -> None:
        raise NotImplementedError("Train function for your model is not implemented!")

    @abstractmethod
    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Predict function for your model is not implemented!")
