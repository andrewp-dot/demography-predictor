# Standard library imports
import os
import pandas as pd
import logging
from typing import List, Dict, Union
from torch import nn

import matplotlib.pyplot as plt

# Import tested tree models
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# Custom imports
from src.utils.log import setup_logging
from src.utils.constants import get_core_hyperparameters
from src.utils.constants import (
    basic_features,
    highly_correlated_features,
    aging_targets,
)

from src.pipeline import GlobalModelPipeline
from src.train_scripts.train_global_models import (
    train_global_model_tree,
    train_global_rnn,
    train_global_arima_ensemble,
)

from src.base import TrainingStats

# TODO: change this -> add exogeneopus variables to the ARIMA model -> convert to global model
# from src.train_scripts.train_local_models import train_arima_ensemble_model
from src.global_model.model import XGBoostTuneParams

from model_experiments.base_experiment import BaseExperiment
from src.base import RNNHyperparameters

from src.compare_models.compare import ModelComparator

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("benchmark")


class SecondModelSelection(BaseExperiment):
    """
    Question: How to evaluate this? GROUND TRUTH testing? For now YES.
    """

    FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

    TARGETS: List[str] = aging_targets()

    BASE_RNN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
        input_size=len(FEATURES + TARGETS),
        hidden_size=256,
        batch_size=16,
        output_size=len(TARGETS),
        epochs=30,
    )

    XGBOOST_TUNE_PARAMETERS: XGBoostTuneParams = XGBoostTuneParams(
        n_estimators=[200, 400],
        learning_rate=[0.01, 0.05, 0.1],
        max_depth=[3, 5, 7],
        subsample=[0.8, 1.0],
        colsample_bytree=[0.8, 1.0],
    )

    # EVALUATION_STATES: List[str] = ["Czechia", "Honduras", "United States"]
    EVALUATION_STATES: List[str] = None  # If None, select all

    # If empty select all states
    # EVALUATION_STATES: List[str] = []

    SAVE_MODEL_DIR: str = os.path.abspath(
        os.path.join(".", "model_selection", "trained_models")
    )

    MODEL_NAMES: List[str] = [
        "ensemble-arimas",
        "simple-rnn",
        "base-lstm",
        "base-gru",
        "xgboost",
        "rf",
        "lightgbm",
    ]

    # Need to save this to save their training stats for plot
    RNN_NAMES = ["simple-rnn", "base-gru", "base-lstm"]

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

        self.rnn_training_stats: Dict[str, TrainingStats] = {}

    def __get_models(self, model_names: List[str]) -> Dict[str, GlobalModelPipeline]:

        # Try to get model
        pipelines: Dict[str, GlobalModelPipeline] = {}
        for name in model_names:
            pipelines[name] = GlobalModelPipeline.get_pipeline(
                name=name, custom_dir=self.SAVE_MODEL_DIR
            )

        return pipelines

    def __train_models(
        self,
        data: Dict[str, pd.DataFrame],
        split_rate: float = 0.8,
        display_nth_epoch: int = 1,
        force_retrain: bool = False,
    ) -> Dict[str, GlobalModelPipeline]:

        # Try to get the models
        # Train ARIMA models for states
        TO_COMPARE_PIPELINES: Dict[str, GlobalModelPipeline] = {}

        if not force_retrain:
            try:
                return self.__get_models(model_names=self.MODEL_NAMES)
            except Exception as e:
                logger.info(f"Models not found. Reatraining all models ({e}).")

        TO_COMPARE_PIPELINES["ensemble-arimas"] = train_global_arima_ensemble(
            name="ensemble-arimas",
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            p=1,
            d=1,  # ARMA model - no need to integrate percentual data.
            q=1,
        )
        TO_COMPARE_PIPELINES["ensemble-arimas"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Train classic rnn
        logger.info("Training simple rnn...")
        TO_COMPARE_PIPELINES["simple-rnn"] = train_global_rnn(
            name="simple-rnn",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.RNN,
        )
        TO_COMPARE_PIPELINES["simple-rnn"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train lstm
        logger.info("Training base lstm...")
        TO_COMPARE_PIPELINES["base-lstm"] = train_global_rnn(
            name="base-lstm",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.LSTM,
        )
        TO_COMPARE_PIPELINES["base-lstm"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train gru
        logger.info("Training base gru...")
        TO_COMPARE_PIPELINES["base-gru"] = train_global_rnn(
            name="base-gru",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.GRU,
        )
        TO_COMPARE_PIPELINES["base-gru"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Save training stats for rnns
        for name in self.RNN_NAMES:
            self.rnn_training_stats[name] = TrainingStats.from_dict(
                stats_dict=TO_COMPARE_PIPELINES[name].model.training_stats
            )

            fig = self.rnn_training_stats[name].create_plot()
            plt.savefig(os.path.join(self.SAVE_MODEL_DIR, name, "loss.png"))

        # Train xgboost
        logger.info("Training xgboost...")
        TO_COMPARE_PIPELINES["xgboost"] = train_global_model_tree(
            name="xgboost",
            tree_model=XGBRegressor(objective="reg:squarederror", random_state=42),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            tune_parameters=self.XGBOOST_TUNE_PARAMETERS,
        )
        TO_COMPARE_PIPELINES["xgboost"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train randomforest
        logger.info("Training random forest...")
        TO_COMPARE_PIPELINES["rf"] = train_global_model_tree(
            name="rf",
            tree_model=RandomForestRegressor(n_estimators=100, random_state=42),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            tune_parameters=self.XGBOOST_TUNE_PARAMETERS,
        )
        TO_COMPARE_PIPELINES["rf"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        logger.info("Training lightgbm...")
        TO_COMPARE_PIPELINES["lightgbm"] = train_global_model_tree(
            name="lightgbm",
            tree_model=LGBMRegressor(
                n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42
            ),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            tune_parameters=self.XGBOOST_TUNE_PARAMETERS,
        )
        TO_COMPARE_PIPELINES["lightgbm"].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        return TO_COMPARE_PIPELINES

    def run(self, split_rate: float = 0.8, force_retrain: bool = False) -> None:

        # Create readme
        self.create_readme()
        DISPLAY_NTH_EPOCH = 1

        # Load data
        loader = StatesDataLoader()
        states_data_dict = loader.load_all_states()

        # Get or train models
        TO_COMPARE_PIPELINES: Dict[str, GlobalModelPipeline] = self.__train_models(
            data=states_data_dict,
            force_retrain=force_retrain,
            split_rate=split_rate,
            display_nth_epoch=DISPLAY_NTH_EPOCH,
        )

        # TODO: figure out how to display the results -
        # 0. Display only best and worst states for each model?
        # 1. per target per state?
        # 2. per target per overall?
        # 3. per state overall?

        comparator = ModelComparator()

        # If empty select all states
        if self.EVALUATION_STATES:
            EVALUATION_STATES = self.EVALUATION_STATES
        else:
            EVALUATION_STATES = list(states_data_dict.keys())

        per_target_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_PIPELINES, states=EVALUATION_STATES, by="per-targets"
        )

        overall_metrics_per_state_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_PIPELINES,
            states=EVALUATION_STATES,
            by="overall-metrics",
        )

        overall_metrics_df = comparator.compare_models_by_states(
            pipelines=TO_COMPARE_PIPELINES,
            states=EVALUATION_STATES,
            by="overall-one-metric",
        )

        # Print bast performing and worst performing states for each method
        for model in self.MODEL_NAMES:
            model_state_metrics_df = overall_metrics_per_state_df[
                overall_metrics_per_state_df["model"] == model
            ].sort_values(by=["mse", "r2"], ascending=[True, False])

            # Top 5 best states
            model_state_metrics_df.head(5)

            # Top 5 worst states
            model_state_metrics_df.tail(5)

            # Write section
            self.readme_add_section(
                title=f"## Model {model} - top states",
                text=f"```\n{model_state_metrics_df.head()}\n```\n\n",
            )

            self.readme_add_section(
                title=f"## Model {model} - worst states",
                text=f"```\n{model_state_metrics_df.tail()}\n```\n\n",
            )

        # Print the rankings for each model target predictions
        # Remove 'state' column

        # Remove 'state' column as we don't need it anymore
        per_target_metrics_df = per_target_metrics_df.drop(columns=["state"])

        # # Adjust this to use correct weight
        # def weighted_mean(group, metric, weight):
        #     return (group[metric] * group[weight]).sum() / group[weight].sum()

        # # Calculate weighted mean for mae, mse, rmse, and r2 using 'rank' as the weight
        # per_target_metrics_df_mean = per_target_metrics_df.groupby(
        #     ["target", "model"], as_index=False
        # ).apply(
        #     lambda group: pd.Series(
        #         {
        #             "mae": weighted_mean(group, "mae", "rank"),
        #             "mse": weighted_mean(group, "mse", "rank"),
        #             "rmse": weighted_mean(group, "rmse", "rank"),
        #             "r2": weighted_mean(group, "r2", "rank"),
        #         }
        #     )
        # )

        per_target_metrics_df_mean = per_target_metrics_df.groupby(
            ["target", "model"], as_index=False
        ).mean()

        # Sort by a specific metric (e.g., mae) and reset the index
        per_target_metrics_df_sorted = per_target_metrics_df_mean.sort_values(
            by=["target", "mse", "r2"], ascending=[True, True, False]
        ).reset_index(drop=True)

        # Add rank column based on sorted values
        per_target_metrics_df_sorted["rank"] = (
            per_target_metrics_df_sorted["mae"].rank(method="first").astype(int)
        )

        self.readme_add_section(
            title="## Per target metrics - model comparision",
            text=f"```\n{per_target_metrics_df_sorted}\n```\n\n",
        )

        # Overall model selection
        self.readme_add_section(
            title="## Overall metrics - model comparision",
            text=f"```\n{overall_metrics_df}\n```\n\n",
        )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    exp = SecondModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data."
    )
    exp.run(split_rate=0.8)
