# Standard library imports
import os
import json
import pandas as pd
import logging
from typing import List, Dict, Literal, Optional
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
    population_total_targets,
    gender_distribution_targets,
)

from src.pipeline import TargetModelPipeline
from src.train_scripts.train_target_models import (
    train_global_model_tree,
    train_global_rnn,
    train_global_arima_ensemble,
)

from src.base import TrainingStats

from src.target_model.model import XGBoostTuneParams

from model_experiments.base_experiment import BaseExperiment
from src.base import RNNHyperparameters

from src.compare_models.compare import ModelComparator

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("benchmark")


class SecondModelSelection(BaseExperiment):
    """
    Question: How to evaluate this? GROUND TRUTH testing? For now YES.
    """

    SAVE_MODEL_DIR: str = os.path.abspath(
        os.path.join(".", "model_selection", "trained_models")
    )

    MODEL_NAMES: List[str] = [
        "ARIMA",
        "ARIMAX",
        "RNN",
        "LSTM",
        "GRU",
        "XGBoost",
        "random_forest",
        "LightGBM",
    ]

    # Need to save this to save their training stats for plot
    RNN_NAMES = ["RNN", "GRU", "LSTM"]

    # Need this to save the parameters used for comparision
    TREE_NAMES = ["XGBoost", "random_forest", "LightGBM"]

    def __init__(
        self,
        description: str,
        target_group_prefix: Literal["pop_total", "aging", "gender_dist"],
    ):
        super().__init__(name=self.__class__.__name__, description=description)
        self.FEATURES: List[str] = basic_features(exclude=highly_correlated_features())

        # Add this just in order
        self.TARGET_GROUP_PREFIX: str = target_group_prefix

        self.TARGETS_BY_PREFIX: Dict[str, callable] = {
            "pop_total": population_total_targets,
            "aging": aging_targets,
            "gender_dist": gender_distribution_targets,
        }

        # Get targets by experiment
        self.TARGETS: List[str] = self.TARGETS_BY_PREFIX[self.TARGET_GROUP_PREFIX]()

        self.BASE_RNN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
            input_size=len(self.FEATURES + self.TARGETS),
            hidden_size=64,
            batch_size=16,
            output_size=len(self.TARGETS),
            epochs=30,
        )

        # This is used to tune x
        self.XGBOOST_TUNE_PARAMETERS: XGBoostTuneParams = XGBoostTuneParams(
            n_estimators=[200, 400],
            learning_rate=[0.01, 0.05, 0.1],
            max_depth=[3, 5, 7],
            subsample=[0.8, 1.0],
            colsample_bytree=[0.8, 1.0],
        )

        # If empty select all states
        self.EVALUATION_STATES: List[str] = None  # If None, select all

        # Training stats of trained neural networks
        self.rnn_training_stats: Dict[str, TrainingStats] = {}

        # Parameters of compared trees
        self.tree_params: Dict[str, dict] = {}

    def __get_models(self, model_names: List[str]) -> Dict[str, TargetModelPipeline]:

        # Try to get model
        pipelines: Dict[str, TargetModelPipeline] = {}
        for name in model_names:
            pipelines[name] = TargetModelPipeline.get_pipeline(
                name=f"{self.TARGET_GROUP_PREFIX}_{name}",
                custom_dir=self.SAVE_MODEL_DIR,
            )

        return pipelines

    def __get_tree_params(
        self, to_compare_pipelines: Dict[str, TargetModelPipeline]
    ) -> None:
        def make_json_serializable(d: dict):
            serializable_dict = {}
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool, type(None), list, dict)):
                    serializable_dict[k] = v
                else:
                    serializable_dict[k] = str(v)  # fallback: convert to string
            return serializable_dict

        for name in self.TREE_NAMES:
            # Get model hyperparameters and save them do local variable object of the class
            raw_params = to_compare_pipelines[name].model.model.get_params()

            self.tree_params[name] = make_json_serializable(raw_params)

            # Path of the model
            tree_model_params_path = os.path.join(
                self.SAVE_MODEL_DIR, name, "params.json"
            )

            with open(tree_model_params_path, "w") as f:
                json.dump(self.tree_params[name], fp=f, indent=4)

    def __plot_rnn_model_losses(
        self, TO_COMPARE_PIPELINES: Dict[str, TargetModelPipeline]
    ):
        fig, ax = plt.subplots(
            nrows=len(self.RNN_NAMES), ncols=1, figsize=(10, 5 * len(self.RNN_NAMES))
        )

        # If only one model, make ax a list of one element for consistency
        if len(self.RNN_NAMES) == 1:
            ax = [ax]

        # Save training stats for RNNs
        for index, name in enumerate(self.RNN_NAMES):
            stats = TrainingStats.from_dict(
                stats_dict=TO_COMPARE_PIPELINES[name].model.training_stats
            )

            # Set labels and title
            ax[index].set_xlabel("Epocha")  # Corrected "Epocha" to "Epoch"
            ax[index].set_ylabel("Strata")
            ax[index].set_title(name)
            ax[index].grid(True)

            # Plot training loss for the model
            ax[index].plot(stats.epochs, stats.training_loss, label="Tréningová strata")

            ax[index].plot(
                stats.epochs,
                stats.validation_loss,
                label="Validačná strata",
            )

            ax[index].legend()
            ax[index].grid(True)

        fig.tight_layout()

        # Save the figure
        self.save_plot(fig_name=f"{self.TARGET_GROUP_PREFIX}_rnn_loss.png", figure=fig)

    def __train_models(
        self,
        data: Dict[str, pd.DataFrame],
        split_rate: float = 0.8,
        display_nth_epoch: int = 1,
        force_retrain: bool = False,
    ) -> Dict[str, TargetModelPipeline]:

        # Try to get the models
        # Train ARIMA models for states
        TO_COMPARE_PIPELINES: Dict[str, TargetModelPipeline] = {}

        if not force_retrain:
            try:
                return self.__get_models(model_names=self.MODEL_NAMES)
            except Exception as e:
                logger.info(f"Models not found. Reatraining all models ({e}).")

        # Train ensemble ARIMAX model - ARIMAX model for each target for each state
        name = f"{self.TARGET_GROUP_PREFIX}_ARIMAX"
        TO_COMPARE_PIPELINES[name] = train_global_arima_ensemble(
            name=name,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            p=1,
            d=1,  # ARMA model - no need to integrate percentual data.
            q=1,
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        name = f"{self.TARGET_GROUP_PREFIX}_ARIMA"
        TO_COMPARE_PIPELINES[name] = train_global_arima_ensemble(
            name=name,
            data=data,
            features=[],
            targets=self.TARGETS,
            split_rate=split_rate,
            p=1,
            d=1,  # ARMA model - no need to integrate percentual data.
            q=1,
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train classic rnn
        logger.info("Training simple rnn...")
        name = f"{self.TARGET_GROUP_PREFIX}_RNN"
        TO_COMPARE_PIPELINES[name] = train_global_rnn(
            name=name,
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.RNN,
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train lstm
        logger.info("Training base lstm...")
        name = f"{self.TARGET_GROUP_PREFIX}_LSTM"
        TO_COMPARE_PIPELINES[name] = train_global_rnn(
            name=name,
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.LSTM,
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train gru
        logger.info("Training base gru...")
        name = f"{self.TARGET_GROUP_PREFIX}_GRU"
        TO_COMPARE_PIPELINES[name] = train_global_rnn(
            name=name,
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.GRU,
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Save pipeline
        self.__plot_rnn_model_losses(TO_COMPARE_PIPELINES=TO_COMPARE_PIPELINES)

        # Train xgboost
        logger.info("Training xgboost...")
        name = f"{self.TARGET_GROUP_PREFIX}_XGBoost"
        TO_COMPARE_PIPELINES[name] = train_global_model_tree(
            name=name,
            tree_model=XGBRegressor(
                n_estimators=100, objective="reg:squarederror", random_state=42
            ),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            xgb_tune_parameters=None,  # Do not tune parameters
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Train randomforest
        logger.info("Training random forest...")
        name = f"{self.TARGET_GROUP_PREFIX}_random_forest"
        TO_COMPARE_PIPELINES[name] = train_global_model_tree(
            name=name,
            tree_model=RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                min_samples_leaf=20,
            ),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            xgb_tune_parameters=None,  # Nothing to tune in here
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        logger.info("Training lightgbm...")
        name = f"{self.TARGET_GROUP_PREFIX}_LightGBM"
        TO_COMPARE_PIPELINES[name] = train_global_model_tree(
            name=name,
            tree_model=LGBMRegressor(
                n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42
            ),
            states_data=data,
            features=self.FEATURES,
            targets=self.TARGETS,
            sequence_len=self.BASE_RNN_HYPERPARAMETERS.sequence_length,
            xgb_tune_parameters=None,  # Nothing to tune in here
        )
        TO_COMPARE_PIPELINES[name].save_pipeline(custom_dir=self.SAVE_MODEL_DIR)

        # Save tree params
        self.__get_tree_params(to_compare_pipelines=TO_COMPARE_PIPELINES)

        return TO_COMPARE_PIPELINES

    def run(
        self,
        split_rate: float = 0.8,
        force_retrain: bool = False,
        evaluation_states: Optional[List[str]] = None,
    ) -> None:

        # Create readme
        self.create_readme(readme_name=f"{self.TARGET_GROUP_PREFIX}_README.md")
        DISPLAY_NTH_EPOCH = 1

        # Load data
        loader = StatesDataLoader()
        states_data_dict = loader.load_all_states()

        # Get or train models
        TO_COMPARE_PIPELINES: Dict[str, TargetModelPipeline] = self.__train_models(
            data=states_data_dict,
            force_retrain=force_retrain,
            split_rate=split_rate,
            display_nth_epoch=DISPLAY_NTH_EPOCH,
        )

        comparator = ModelComparator()

        # If empty select all states
        if evaluation_states:
            EVALUATION_STATES = evaluation_states
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
            ].sort_values(by=["r2", "mse"], ascending=[False, True])

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

        # Remove 'state' column as we don't need it anymore
        per_target_metrics_df = per_target_metrics_df.drop(columns=["state"])

        per_target_metrics_df_mean = per_target_metrics_df.groupby(
            ["target", "model"], as_index=False
        ).mean()

        # Sort by a specific metric (e.g., mae) and reset the index
        per_target_metrics_df_sorted = per_target_metrics_df_mean.sort_values(
            by=["target", "r2", "mse"], ascending=[True, False, True]
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

    # Experiment for aging
    exp_aging = SecondModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="aging",
    )

    exp_aging.run(split_rate=0.8, force_retrain=True)

    # Experiment for population total
    exp_pop_total = SecondModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="pop_total",
    )
    exp_pop_total.run(split_rate=0.8, force_retrain=True)

    # Experiment for gender distribution
    exp_gender_dist = SecondModelSelection(
        description="Compares models to predict the target variable(s) using past data and future known (ground truth) data.",
        target_group_prefix="gender_dist",
    )

    exp_gender_dist.run(split_rate=0.8, force_retrain=True)
