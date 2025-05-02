# Standard library imports
import pandas as pd
from abc import abstractmethod
from typing import Dict, List, Callable, Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import logging
import numpy as np


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


# Custom imports
from src.utils.log import setup_logging

from src.pipeline import FeatureModelPipeline, TargetModelPipeline, PredictorPipeline


logger = logging.getLogger("model_compare")


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

        # Average metric across all targets
        average_metric_value = metric(
            self.reference_values, self.predicted, multioutput="uniform_average"
        )

        # Get overall metric dataframe
        overall_metric_df = pd.DataFrame({metric_key: [average_metric_value]})

        return overall_metric_df

    def get_overall_metrics(self) -> pd.DataFrame:
        """
        Computes overall metrics for the model.

        Returns:
            out: pd.DataFrame: The dataframe for overall (all targets included) metrics.
        """

        # Get MAE
        overall_mae_df = self.get_overall_metric(mean_absolute_error, "mae")

        # Get MSE
        overall_mse_df = self.get_overall_metric(mean_squared_error, "mse")

        # Get RMSE
        overall_rmse_df = self.get_overall_metric(root_mean_squared_error, "rmse")

        # Get MAPE
        overall_mape_df = self.get_overall_metric(
            mean_absolute_percentage_error, "mape"
        )

        # Get R^2
        overall_r2_df = self.get_overall_metric(r2_score, "r2")

        # Get overall dataframe
        return pd.concat(
            [
                overall_mae_df,
                overall_mse_df,
                overall_rmse_df,
                overall_mape_df,
                overall_r2_df,
            ],
            axis=1,
        )

    def get_single_target_specific_metric(
        self,
        metric: Callable,
        targets: List[str],
        metric_key: str = "",
    ) -> pd.DataFrame:
        """
        Computes and saves given metric.

        Args:
            metric (Callable): Function for metric computation.
            targets (List[str]): List of predicted targets.
            metric_key (str, optional): Key of the metric which can be accasible in `EvaluateModel.metrics` dict (`{metric_key}`, `{metric_key}_per_target` if per target is available).
                If not given, the name of the function is used.. Defaults to "".

        Returns:
            out: pd.DataFrame: metric value for all targets, metric values for each target separately.
        """

        # Adjust metric key if not given
        metric_key = metric_key or metric.__name__
        TARGETS = targets

        # Initialize metric values
        separate_target_metric_values_df = None

        if metric == r2_score:
            # Compute r2 for each target separately
            metric_per_target_values = [
                r2_score(self.reference_values.iloc[:, i], self.predicted.iloc[:, i])
                for i in range(self.reference_values.shape[1])
            ]

        else:
            # Compute metric for each target separately
            metric_per_target_values = metric(
                self.reference_values, self.predicted, multioutput="raw_values"
            )

        # Get metric values for separate targets
        metric_per_target_values_dict = {
            "target": TARGETS,
            metric_key: metric_per_target_values,
        }

        separate_target_metric_values_df = pd.DataFrame(metric_per_target_values_dict)

        return separate_target_metric_values_df

    # TODO: get rarget specific metrics overall
    def get_target_specific_metrics(self, targets: List[str]) -> pd.DataFrame:
        """
        Creates and saves evaluation dataframe for every predicted targets.

        Args:
            targets (List[str]): List of features used for evaluation.

        Returns:
            out: pd.DataFrame: The dataframe with evaluation metrics per target.
        """

        # Set features constant
        TARGETS = targets

        # Get MAE
        mae_per_target_df = self.get_single_target_specific_metric(
            mean_absolute_error, TARGETS, "mae"
        )

        # Get MSE
        mse_per_target_df = self.get_single_target_specific_metric(
            mean_squared_error, TARGETS, "mse"
        )

        # Get RMSE
        rmse_per_target_df = self.get_single_target_specific_metric(
            root_mean_squared_error, TARGETS, "rmse"
        )

        # Get MAPE
        mape_per_target_df = self.get_single_target_specific_metric(
            mean_absolute_percentage_error, TARGETS, "mape"
        )

        # Get R^2
        r2_per_target_df = self.get_single_target_specific_metric(
            r2_score, TARGETS, "r2"
        )

        # Create per target dataframe
        to_merge_dfs: List[pd.DataFrame] = [
            mae_per_target_df,
            mse_per_target_df,
            rmse_per_target_df,
            mape_per_target_df,
            r2_per_target_df,
        ]

        # Merge dataframes
        per_target_metrics_df = to_merge_dfs[0]
        for df in to_merge_dfs[1:]:
            per_target_metrics_df = pd.merge(
                left=per_target_metrics_df, right=df, on="target"
            )

        return per_target_metrics_df

    def get_last_and_target_year(
        self, test_X: pd.DataFrame, test_y: pd.DataFrame
    ) -> Tuple[int, int]:
        """
        If possible extracts the last and target years from the given (validation) data.

        Args:
            test_X (pd.DataFrame): Validation input data.
            test_y (pd.DataFrame): Validation output data.

        Raises:
            ValueError: If the "year" column is missing in `test_X` data.
            ValueError: If the "year" column is missing in `test_y` data.

        Returns:
            Tuple[int, int]: last_year, target_year
        """

        if "year" not in test_X.columns:
            raise ValueError("Could not find out the last year for input data.")

        # Get the last year and get the number of years
        X_years = test_X[["year"]]
        last_year = int(X_years.iloc[-1].item())

        if "year" not in test_y.columns:
            raise ValueError("Could not find out the target year for test data.")

        # # Get the prediction year
        y_target_years = test_y[["year"]]
        target_year = int(y_target_years.iloc[-1].item())

        return last_year, target_year

    @abstractmethod
    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
        *args,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    def plot_single_target_prediction(self, target: str) -> Figure:
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
            self.reference_values[target],
            label=f"Reference values",
            color="b",
        )

        plt.plot(
            YEARS,
            self.predicted[target],
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
            axes[index].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.legend()

        return fig


class EvaluateModel(BaseEvaluation):

    def __init__(
        self,
        pipeline: Union[
            "FeatureModelPipeline", "TargetModelPipeline", "PredictorPipeline"
        ],
    ):
        super().__init__()
        # self.transformer: DataTransformer = pipeline.transformer
        # self.model: CustomModelBase = pipeline.model

        self.pipeline: Union[
            "FeatureModelPipeline", "TargetModelPipeline", "PredictorPipeline"
        ] = pipeline

        # Reference values for every state
        # {"Czechia": {"predicted"..., "reference": ...}, "United States": ...}
        self.multiple_states_evaluations: Dict[str, Dict[str, pd.DataFrame]] = {}

    def __get_refference_and_predicted_data(
        self, test_X: pd.DataFrame, test_y: pd.DataFrame
    ) -> None:
        # Get features and targets

        if isinstance(self.pipeline, FeatureModelPipeline) or isinstance(
            self.pipeline, TargetModelPipeline
        ):
            # Get features and targets
            FEATURES = self.pipeline.model.FEATURES
            TARGETS = self.pipeline.model.TARGETS

        elif isinstance(self.pipeline, PredictorPipeline):
            # Get features and targets
            FEATURES = self.pipeline.global_model_pipeline.model.FEATURES
            TARGETS = self.pipeline.global_model_pipeline.model.TARGETS

        # Get year data
        last_year, target_year = self.get_last_and_target_year(
            test_X=test_X, test_y=test_y
        )

        # Adjust features
        # X_test = test_X[FEATURES]
        y_test = test_y[TARGETS]

        # Get predicted years
        self.predicted_years = list(
            range(last_year + 1, target_year + 1)
        )  # values are predicted from next year after last year and for the target year (+ 1 to include target year)
        logger.debug(f"[Eval]: predicting values from {last_year} to {target_year}...")

        # Get predictions
        if isinstance(self.pipeline, FeatureModelPipeline):
            input_data = test_X

            predictions_df = self.pipeline.predict(
                input_data=input_data,
                last_year=last_year,
                target_year=target_year,
                state=str(test_X["country_name"].iloc[0]),
            )

        elif isinstance(self.pipeline, TargetModelPipeline):

            # This is ground truth evaluation, so the last year and target year are the same
            input_data = test_X

            # Get true feature values for testing ground truth
            feature_df = pd.concat([test_X[FEATURES], test_y[FEATURES]], axis=0)
            previous_targets_df = test_X[TARGETS]

            # Pad previous_targets_df to match the length
            pad_len = len(feature_df) - len(previous_targets_df)

            # Delete the actual feature values
            padding = pd.DataFrame(
                np.nan, columns=previous_targets_df.columns, index=range(pad_len)
            )

            previous_targets_padded = pd.concat(
                [previous_targets_df, padding], ignore_index=True
            )

            # Get the final input data
            final_input = pd.concat(
                [feature_df.reset_index(drop=True), previous_targets_padded], axis=1
            )

            predictions_df = self.pipeline.predict(
                input_data=final_input,
                last_year=last_year,
                target_year=target_year,
                state=str(test_X["country_name"].iloc[0]),
            )

        elif isinstance(self.pipeline, PredictorPipeline):
            predictions_df = self.pipeline.predict(
                input_data=test_X, target_year=target_year
            )

        self.predicted = predictions_df[TARGETS]

        self.reference_values = y_test

    def eval(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Basic evaluation function. Evaluates for all targets together.

        Args:
            test_X (pd.DataFrame): Test input data.
            test_y (pd.DataFrame): Test output data.

         Returns:
            out: pd.DataFrame: Evaluation dataframe with overall metrics.

        """

        # Make predictions and save reference data
        self.__get_refference_and_predicted_data(test_X=test_X, test_y=test_y)

        # Get overall metrics for model
        return self.get_overall_metrics()

    def eval_per_target(
        self,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Basic evaluation function. Evaluates accuracy per each target separately.

        Args:
            test_X (pd.DataFrame): Test input data.
            test_y (pd.DataFrame): Test output data.

        Returns:
            out: pd.DataFrame: Evaluation dataframe with metrics per target.

        """
        # Make predictions and save reference data
        self.__get_refference_and_predicted_data(test_X=test_X, test_y=test_y)

        # Get metrics for per target evaluation
        if isinstance(self.pipeline, FeatureModelPipeline):
            TARGETS = self.pipeline.model.TARGETS

        elif isinstance(self.pipeline, TargetModelPipeline):
            TARGETS = self.pipeline.model.TARGETS

        elif isinstance(self.pipeline, PredictorPipeline):
            TARGETS = self.pipeline.global_model_pipeline.model.TARGETS

        else:
            raise ValueError(
                f"Not supported type of pipeline '{self.pipeline.__class__.__name__}'."
            )
        return self.get_target_specific_metrics(targets=TARGETS)

    def eval_for_every_state(
        self,
        X_test_states: Dict[str, pd.DataFrame],
        y_test_states: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Overall evaluation metrics per state data.

        Args:
            X_test_states (Dict[str, pd.DataFrame]): The `key` is the state name and `value` is the current state input data.
            y_test_states (Dict[str, pd.DataFrame]): The `key` is the state name and `value` is the current state validation data.

        Returns:
            out: pd.DataFrame: Evaluation dataframe for every state.
        """

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

            state_overall_metrics = self.eval(test_X=test_X, test_y=test_y)

            # Save reference and target values for each state if there is multiple state evaluation
            self.multiple_states_evaluations[state] = {
                "reference": self.reference_values,
                "predicted": self.predicted,
                "years": self.predicted_years,
            }

            state_overall_metrics["state"] = state

            new_rows.append(state_overall_metrics)

        # Add evaluation to the states
        all_evaluation_df = pd.concat(new_rows, ignore_index=True)

        # Save the evaluation
        return all_evaluation_df

    def eval_for_every_state_overall(
        self,
        X_test_states: Dict[str, pd.DataFrame],
        y_test_states: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:

        # Need this function for exclusion of the features in order to improve performance for the PredictorPipeline
        # Evalutas for every state but does not save the metrics separately, but overall

        per_state_metric_df = self.eval_for_every_state(
            X_test_states=X_test_states, y_test_states=y_test_states
        )

        state_lens: Dict[str, int] = {}
        states_len_sum: int = 0
        for state, df in X_test_states.items():
            state_lens[state] = df.shape[0]
            states_len_sum += df.shape[0]

        # After computet sum, adjust the weights
        state_weights: Dict[str, float] = {
            state: value / states_len_sum for state, value in state_lens.items()
        }

        # Use weighted average for the overall metrics
        metrics = ["mae", "mse", "rmse", "mape", "r2"]
        overall_metrics: Dict[str, float] = {metric: 0.0 for metric in metrics}
        for state in state_weights.keys():
            # Get the state metrics
            state_metrics = per_state_metric_df[per_state_metric_df["state"] == state]

            # Get the weights
            weight = state_weights[state]

            # Adjust the metrics
            for metric in metrics:
                overall_metrics[metric] += state_metrics[metric].item() * weight

        # Adjust the metrics by division of all states len
        for metric in metrics:
            overall_metrics[metric] = [overall_metrics[metric]]

        return pd.DataFrame(overall_metrics)
