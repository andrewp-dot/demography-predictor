# Standard library imports
import os
import pandas as pd
import logging
from typing import List, Dict, Literal, Optional
from torch import nn

import matplotlib.pyplot as plt


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

from src.pipeline import FeatureModelPipeline
from src.train_scripts.train_feature_models import (
    train_base_rnn,
    train_ensemble_model,
    train_arima_ensemble_all_states,
)

from src.base import TrainingStats

from model_experiments.base_experiment import BaseExperiment
from src.base import RNNHyperparameters

from src.compare_models.compare import ModelComparator

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


logger = logging.getLogger("benchmark")


class FirstModelExperiment(BaseExperiment):

    SAVE_MODEL_DIR: str = os.path.abspath(
        os.path.join(".", "model_selection", "trained_models")
    )

    MODEL_NAMES: List[str] = [
        "feature_ARIMA",
        "feature_RNN",
        "feature_GRU",
        "feature_LSTM",
        "feature_LSTM_NN",
    ]

    # Need to save this to save their training stats for plot
    RNN_NAMES = ["feature_RNN", "feature_GRU", "feature_LSTM", "feature_LSTM_NN"]

    def __init__(
        self,
        description: str,
    ):
        super().__init__(name=self.__class__.__name__, description=description)
        self.FEATURES: List[str] = basic_features(
            exclude=[
                *highly_correlated_features(),
                "year",
            ]  # Need to exlucde year because of ARIMA
        )

        # Get targets by experiment
        self.TARGETS: List[str] = self.FEATURES

        self.BASE_RNN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=256,
            batch_size=16,
            output_size=len(self.TARGETS),
            epochs=30,
        )

        self.RNN_NN_HYPERPARAMETERS: RNNHyperparameters = get_core_hyperparameters(
            input_size=len(self.FEATURES),
            hidden_size=256,
            batch_size=16,
            output_size=len(self.TARGETS),
            epochs=30,
            num_layers=2,
        )

        # EVALUATION_STATES: List[str] = ["Czechia", "Honduras", "United States"]
        # If empty select all states
        self.EVALUATION_STATES: List[str] = None  # If None, select all

        self.rnn_training_stats: Dict[str, TrainingStats] = {}

    def get_models(self, model_names: List[str]) -> Dict[str, FeatureModelPipeline]:

        # Try to get model
        pipelines: Dict[str, FeatureModelPipeline] = {}
        for name in model_names:
            pipelines[name] = FeatureModelPipeline.get_pipeline(
                name=name, custom_dir=self.SAVE_MODEL_DIR
            )

        return pipelines

    def __plot_rnn_model_losses(
        self, TO_COMPARE_PIPELINES: Dict[str, FeatureModelPipeline]
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

        fig.tight_layout()

        # Save the figure
        self.save_plot(fig_name=f"rnn_loss.png", figure=fig)

    def __train_models(
        self,
        data: Dict[str, pd.DataFrame],
        split_rate: float = 0.8,
        display_nth_epoch: int = 1,
        force_retrain: bool = False,
        evaluation_states: Optional[List[str]] = None,
    ) -> Dict[str, FeatureModelPipeline]:
        TO_COMPARE_PIPELINES: Dict[str, FeatureModelPipeline] = {}

        if not force_retrain:
            try:
                return self.get_models(model_names=self.MODEL_NAMES)
            except Exception as e:
                logger.info(f"Models not found. Reatraining all models ({e}).")

        logger.info("Training feature_ARIMA...")
        if evaluation_states:
            arima_data = {state: data[state] for state in evaluation_states}
        else:
            arima_data = data

        TO_COMPARE_PIPELINES["feature_ARIMA"] = train_arima_ensemble_all_states(
            name="feature_ARIMA",
            data=arima_data,
            features=self.FEATURES,
            split_rate=split_rate,
        )
        TO_COMPARE_PIPELINES["feature_ARIMA"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Create simple rnn
        logger.info("Training feature_RNN...")
        TO_COMPARE_PIPELINES["feature_RNN"] = train_base_rnn(
            name="feature_RNN",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.RNN,
        )
        TO_COMPARE_PIPELINES["feature_RNN"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Train lstm
        logger.info("Training feature_LSTM...")
        TO_COMPARE_PIPELINES["feature_LSTM"] = train_base_rnn(
            name="feature_LSTM",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.LSTM,
        )
        TO_COMPARE_PIPELINES["feature_LSTM"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Train gru
        logger.info("Training feature_GRU...")
        TO_COMPARE_PIPELINES["feature_GRU"] = train_base_rnn(
            name="feature_GRU",
            hyperparameters=self.BASE_RNN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.GRU,
        )
        TO_COMPARE_PIPELINES["feature_GRU"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Hybrid LSTM
        logger.info("Training LSTM + NN...")
        TO_COMPARE_PIPELINES["feature_LSTM_NN"] = train_base_rnn(
            name="feature_LSTM_NN",
            hyperparameters=self.RNN_NN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.LSTM,
            additional_bpnn=[32],
        )
        TO_COMPARE_PIPELINES["feature_LSTM_NN"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        logger.info("Training RNN + NN...")
        TO_COMPARE_PIPELINES["feature_RNN_NN"] = train_base_rnn(
            name="feature_RNN_NN",
            hyperparameters=self.RNN_NN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.RNN,
            additional_bpnn=[32],
        )
        TO_COMPARE_PIPELINES["feature_RNN_NN"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        logger.info("Training GRU + NN...")
        TO_COMPARE_PIPELINES["feature_RNN_NN"] = train_base_rnn(
            name="feature_RNN_NN",
            hyperparameters=self.RNN_NN_HYPERPARAMETERS,
            data=data,
            features=self.FEATURES,
            split_rate=split_rate,
            display_nth_epoch=display_nth_epoch,
            rnn_type=nn.GRU,
            additional_bpnn=[32],
        )
        TO_COMPARE_PIPELINES["feature_GRU_NN"].save_pipeline(
            custom_dir=self.SAVE_MODEL_DIR
        )

        # Plot loss for all rnns or nns
        self.__plot_rnn_model_losses(TO_COMPARE_PIPELINES=TO_COMPARE_PIPELINES)

        return TO_COMPARE_PIPELINES

    def run(
        self,
        split_rate: float = 0.8,
        force_reatrain: bool = False,
        evaluation_states: Optional[List[str]] = None,
    ) -> None:

        # Setup readme
        self.create_readme()

        # Load data
        loader = StatesDataLoader()
        states_data_dict = loader.load_all_states()

        TO_COMPARE_PIPELINES = self.__train_models(
            data=states_data_dict,
            split_rate=split_rate,
            force_retrain=force_reatrain,
            evaluation_states=evaluation_states,
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

            # Write section
            self.readme_add_section(
                title=f"## Model {model} - top states",
                text=f"```\n{model_state_metrics_df.head(3)}\n```\n\n",
            )

            self.readme_add_section(
                title=f"## Model {model} - worst states",
                text=f"```\n{model_state_metrics_df.tail(3)}\n```\n\n",
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


class ARIMAvsUnivariateRNN(BaseExperiment):
    pass


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    exp = FirstModelExperiment(
        description="Compare models for predicting all features which are used for target predictions."
    )

    # Run for all
    exp.run(force_reatrain=True)

    # STATE = "Brunei Darussalam"
    # exp.run(evaluation_states=["Brazil", "Qatar", STATE], force_reatrain=True)

    # models = exp.get_models(model_names=["feature_ARIMA"])

    # arima_model = models["feature_ARIMA"]

    # states_loader = StatesDataLoader()
    # quatar_dict = states_loader.load_states(states=[STATE])

    # input_data = quatar_dict[STATE]

    # prediction_df = arima_model.predict(
    #     input_data=input_data.iloc[: int(len(input_data) * 0.2)],
    #     last_year=2010,
    #     target_year=2045,
    #     state=STATE,
    # )

    # print(prediction_df)
