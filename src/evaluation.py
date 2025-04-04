# Standard library imports
import pandas as pd
from abc import abstractmethod
from typing import Dict, List, Callable
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import logging

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)

# Custom imports
from src.utils.log import setup_logging
from src.base import CustomModelBase


logger = logging.getLogger("evaluation")


class BaseEvaluation:

    def __init__(self, *args, **kwargs):
        self.predicted: pd.DataFrame | None = None
        self.reference_values: pd.DataFrame | None = None
        self.predicted_years: List[int] | None = None

        # Define metric dataframes
        self.per_target_metrics: pd.DataFrame | None = None
        # self.overall_metrics: pd.DataFrame | None = None

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

        # Average metric across all targets
        average_metric_value = metric(
            self.reference_values, self.predicted, multioutput="uniform_average"
        )

        # Get overall metric dataframe
        overall_metric_df = pd.DataFrame(
            {"metric": metric_key, "value": [average_metric_value]}
        )

        return overall_metric_df

    def get_overall_metrics(self) -> None:
        """
        Computes overall metrics for the model and saves it to the `overall_metrics` object property.
        """

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

    def get_metric_value(self, evaluation_df: pd.DataFrame, metric: str) -> float:
        """
        From the evaluation dict in format of the overall metrics dataframe (columns: metric | value) extracts the value of the given metric.

        Args:
            evaluation_df (pd.DataFrame): Evaluation dataframe compatibile with the overall metrics dataframe.
            metric (str): Metric which you want to extract.

        Returns:
            out: float: The value of the given metric.
        """
        return evaluation_df.loc[evaluation_df["metric"] == metric, "value"].values[0]

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

    def plot_single_feautre_prediction(self, feature: str) -> Figure:
        """
        Creates figure for a specified feature prediction.

        Returns:
            out: Figure: The plot of the reference / predicted values for a single feature.
        """
        # Get years
        YEARS = self.predicted_years

        fig = plt.figure(figsize=(8, 2))

        plt.plot(
            YEARS,
            self.reference_values[feature],
            label=f"Reference values",
            color="b",
        )

        plt.plot(
            YEARS,
            self.predicted[feature],
            label=f"Predicted",
            color="r",
        )

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.grid()
        plt.legend()

        return fig

    def plot_predictions(self) -> Figure:
        """
        Creates figure for a all features evaluation.

        Returns:
            out: Figure: The plot of the reference / predicted values for all features.
        """

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
        self.all_states_evaluation: pd.DataFrame = None

    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
    ):
        # Get features and targets
        FEATURES = self.model.FEATURES
        TARGETS = self.model.TARGETS

        # Get the last year and get the number of years
        X_years = test_X[["year"]]
        last_year = int(X_years.iloc[-1].item())

        # # Get the prediction year
        y_target_years = test_y[["year"]]
        target_year = int(y_target_years.iloc[-1].item())

        # Get predicted years
        self.predicted_years = list(
            range(last_year + 1, target_year + 1)
        )  # values are predicted from next year after last year and for the target year (+ 1 to include target year)
        logger.debug(f"[Eval]: predicting values from {last_year} to {target_year}...")

        # Adjust features
        test_X = test_X[FEATURES]
        test_y = test_y[TARGETS]

        # Get predictions
        predictions_df = self.model.predict(
            input_data=test_X,
            last_year=last_year,
            target_year=target_year,
        )
        self.predicted = predictions_df[self.model.TARGETS]

        logger.critical(self.predicted)

        self.reference_values = test_y

        # Get metrics for time series model
        if FEATURES == TARGETS:
            self.get_feature_specific_metrics(features=FEATURES)

        # Get overall metrics for model
        self.get_overall_metrics()

    def eval_for_every_state(
        self,
        X_test_states: Dict[str, pd.DataFrame],
        y_test_states: Dict[str, pd.DataFrame],
    ) -> None:

        # Create empty dataframe
        # state | mae | mse | rmse | r2
        # ...
        columns = ["state", "mae", "mse", "rmse", "r2"]
        all_evaluation_df: pd.DataFrame = pd.DataFrame(columns=columns)

        new_rows = []

        for state in X_test_states.keys():

            # Get X_test and y_test for state
            test_X = X_test_states[state]
            test_y = y_test_states[state]

            # Run evaluation (Maybe create new evaluation?)
            current_state_evaluation = EvaluateModel(self.model)
            current_state_evaluation.eval(test_X=test_X, test_y=test_y)

            # Get the evaluation state
            m = current_state_evaluation.overall_metrics

            # Get metrics
            new_row = pd.DataFrame(
                [
                    {
                        "state": state,
                        "mae": self.get_metric_value(m, "mae"),
                        "mse": self.get_metric_value(m, "mse"),
                        "rmse": self.get_metric_value(m, "rmse"),
                        "r2": self.get_metric_value(m, "r2"),
                    }
                ]
            )
            new_rows.append(new_row)

        # Add evaluation to the states
        all_evaluation_df = pd.concat(new_rows, ignore_index=True)

        # Save the evaluation
        self.all_states_evaluation = all_evaluation_df

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


if __name__ == "__main__":
    # Setup logging
    setup_logging()
