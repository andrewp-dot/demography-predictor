# Standard library imports
import pandas as pd
import logging

from typing import List, Dict, Union, Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Custom imports
from config import Config
from src.utils.log import setup_logging
from src.utils.save_model import get_multiple_models
from src.utils.constants import get_core_hyperparameters, translate_target

from src.evaluation import EvaluateModel
from src.feature_model.ensemble_model import PureEnsembleModel

from src.target_model.model import TargetModelTree
from src.statistical_models.multistate_wrapper import StatisticalMultistateWrapper

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.pipeline import FeatureModelPipeline, TargetModelPipeline, PredictorPipeline


logger = logging.getLogger("model_compare")

settings = Config()


class ModelComparator:

    def __init__(self):

        # Save model evaluations
        self.eval_states: List[str] = []
        self.model_evaluations: Dict[str, EvaluateModel] = {}

    def rank_models(self, evaluation_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        From the evaluation dictionaries create a ranking dataframe. If there are missing metrics in the dataframes, it returns an empty datframe.

        Args:
            evaluation_dfs (List[pd.DataFrame]): Evaluation dataframes for model.

        Returns:
            out: pd.DataFrame: Ranking dataframe.
        """

        # Concatenate all model evaluations into a single DataFrame
        df_all = pd.concat(evaluation_dfs, ignore_index=True)

        # Define ranking function
        def rank_models(df: pd.DataFrame) -> pd.DataFrame:
            """Ranks models based on best metric scores."""
            df["rank"] = (
                df["mae"].rank(ascending=True)  # Lower MAE is better
                + df["mse"].rank(ascending=True)  # Lower MSE is better
                + df["rmse"].rank(ascending=True)  # Lower RMSE is better
                + (
                    df["r2"].rank(ascending=False) if "r2" in df.columns else 0
                )  # Higher R² is better
            )
            return df.sort_values("rank", ascending=True)

        # Rank models
        try:
            df_ranked = rank_models(df_all)
        except KeyError as e:
            logger.error(f"KeyError: Failed to compare models: {str(e)}.")
            return pd.DataFrame({})

        return df_ranked

    # Compare models predictors
    # TODO:
    # 1. adjust this for pipeliens
    # 2. add support for predictor pipeline
    def compare_models_by_states(
        self,
        pipelines: Dict[
            str, Union[FeatureModelPipeline, TargetModelPipeline, PredictorPipeline]
        ],
        states: List[str] | None = None,
        by: Literal[
            "overall-metrics", "overall-one-metric", "per-targets"
        ] = "overall-metrics",
    ) -> pd.DataFrame:
        """
        Compares model performance by metrics. Returns ranking dataframe.

        Args:
            models (List[str]): _description_
            states (List[str] | None, optional): _description_. Defaults to None.
            by (Literal[&quot;overall-metrics&quot;, &quot;per-features&quot;, optional): Overall-metrics -> compares model performance by the overall metrics for state.
                "per-features" is options mainly for LSTM predictor models, gives information about model performance per feature. Defaults to "overall-metrics".

        Raises:
            ValueError: If the models are not comparable.

        Returns:
            out: pd.DataFrame: Ranking dataframe.
        """

        # Check if there is something to compare
        if len(pipelines) <= 1:
            logger.warning("No models to compare.")
            return {}

        # Save eval states
        self.eval_states = states

        # Check if the models has same features and targets
        # first_model_features: List[str] = to_compare_models[models[0]].FEATURES
        model_names: List[str] = list(pipelines.keys())

        first_model_targets: List[str] = pipelines[model_names[0]].TARGETS

        for model_name in model_names[1:]:
            # Get next model
            model = pipelines[model_name]

            # Check if they have the same targets
            if model.TARGETS != first_model_targets:
                raise ValueError(
                    f"The model '{model_name}' has different targets then the first model ({set(model.TARGETS).symmetric_difference(set(first_model_targets))})"
                )

        # Get evaluation for all states
        model_evaluations: Dict[str, pd.DataFrame] = {}

        # Get evaluation data
        states_loaders = StatesDataLoader()

        # Get the data by given states or all
        if states is None:
            states_data_dict = states_loaders.load_all_states()
        else:
            states_data_dict = states_loaders.load_states(states=states)

        # Getnerate evaluation for every model
        for model_name, pipeline in pipelines.items():
            # Preprocess data for the model - supports different sequence length, by type

            logger.info(f"Evaluating model: {model_name}")

            # Adjust hyperparameters by the model type
            if isinstance(pipeline, PredictorPipeline):
                if isinstance(pipeline.local_model_pipeline.model, PureEnsembleModel):
                    model_sequence_len = get_core_hyperparameters(
                        input_size=1
                    ).sequence_length
                else:
                    model_sequence_len = pipeline.local_model_pipeline.sequence_length
            elif isinstance(pipeline.model, PureEnsembleModel) or isinstance(
                pipeline.model, StatisticalMultistateWrapper
            ):
                model_sequence_len = get_core_hyperparameters(
                    input_size=1
                ).sequence_length

            elif isinstance(pipeline.model, TargetModelTree):
                model_sequence_len = pipeline.model.sequence_len
            else:
                model_sequence_len = pipeline.model.hyperparameters.sequence_length

            train_data_dict, test_data_dict = states_loaders.split_data(
                states_dict=states_data_dict,
                sequence_len=model_sequence_len,
            )

            model_evaluation = EvaluateModel(
                pipeline=pipeline,
            )

            # Save the model evaluation refference
            self.model_evaluations[model_name] = model_evaluation

            # Save the evaluation
            if "overall-metrics" == by:
                model_evaluations[model_name] = model_evaluation.eval_for_every_state(
                    X_test_states=train_data_dict, y_test_states=test_data_dict
                )

            # TODO: do this by for per target
            elif "overall-one-metric" == by:
                model_evaluations[model_name] = (
                    model_evaluation.eval_for_every_state_overall(
                        X_test_states=train_data_dict, y_test_states=test_data_dict
                    )
                )

            elif "per-targets" == by:

                states_evaluation_for_model: pd.DataFrame | None = None
                for state in train_data_dict.keys():

                    # Get per target per state evaluation
                    model_evaluation.eval(
                        test_X=train_data_dict[state], test_y=test_data_dict[state]
                    )
                    per_target_metrics = model_evaluation.get_target_specific_metrics(
                        model.TARGETS
                    )
                    per_target_metrics["state"] = state

                    # Init dataframe if none
                    if states_evaluation_for_model is None:
                        states_evaluation_for_model = per_target_metrics

                    # Concat dataframe if exists
                    else:
                        states_evaluation_for_model = pd.concat(
                            [
                                states_evaluation_for_model,
                                per_target_metrics,
                            ],
                            axis=0,
                        )

                # Save all states
                model_evaluations[model_name] = states_evaluation_for_model

        # Compare evaluations
        # Convert evaluations (Dict[str, pd.DataFrame]) into a single DataFrame
        df_list = []
        for model, df in model_evaluations.items():
            df["model"] = model  # Add model name to DataFrame
            df_list.append(df)

        # Rank models
        df_ranked = self.rank_models(df_list)

        return df_ranked

    def create_state_comparison_plot(
        self,
        state: str,
        model_names: Optional[List[str]] = None,
        label_font_size: Optional[int] = None,
        legend_font_size: Optional[int] = None,
        value_unit: Optional[str] = "%",
    ) -> Figure:
        """
        Create comparison plot for the given state. The plot contains the reference values and predicted values for all models.

        Args:
            state (str): The predicted state.
            model_names (Optional[List[str]], optional): If specified, display these models in the prediction plot. If None, use all compared models. Defaults to None.
            lang (Literal[&quot;en&quot;, &quot;sk&quot;], optional): Choose language for plotted images. Defaults to "en".

        Raises:
            ValueError: If there are no model evaluations to create state comparison plot.

        Returns:
            out: Figure: The create comparision plot for the given state (with it's reference values).
        """

        LANG = settings.plot_description_language
        LABELS: Dict[str, Dict[str, str]] = {
            "en": {
                "reference_values": "Reference values",
                "predicted_values": "Predicted",
                "yaxis": f"Value ({value_unit})",
                "xaxis": "Year",
            },
            "sk": {
                "reference_values": "Referenčné hodnoty",
                "predicted_values": "Predikcie",
                "yaxis": f"Hodnota ({value_unit})",
                "xaxis": "Rok",
            },
        }

        # Plot refference data
        if not self.model_evaluations:
            raise ValueError(
                f"No model evaluations to create state comparison plot. Compare some models first."
            )

        if model_names:
            plot_model_names = model_names
        else:
            plot_model_names = list(self.model_evaluations.keys())

        # Get the first model
        first_model_name = plot_model_names[0]

        # Get the refference data for the state
        TARGETS = self.model_evaluations[first_model_name].pipeline.TARGETS
        N_TARGETS = len(TARGETS)

        # Get evalution for the first model for the state to get refference data info
        first_model_evaluation: EvaluateModel = self.model_evaluations[first_model_name]

        first_model_state_evaluation: Dict[str, pd.DataFrame] = (
            first_model_evaluation.multiple_states_evaluations[state]
        )

        # HERE IS THE PROBLEM -> IN THE YEARS COMPUTING OR SOMETHING
        YEARS = first_model_state_evaluation["years"]
        reference_data = first_model_state_evaluation["reference"]

        # Create a figure with N rows and 1 column
        fig, axes = plt.subplots(N_TARGETS, 1, figsize=(8, 2 * N_TARGETS))

        if N_TARGETS == 1:
            axes = [axes]

        # Plotting in each subplot
        for index, target in zip(range(N_TARGETS), TARGETS):

            # Plot reference values
            axes[index].plot(
                YEARS,
                reference_data[target],
                label=LABELS[LANG]["reference_values"],
                color="r",
            )

            # Fore every model evaluation
            for model_name in plot_model_names:

                # Get evaluation
                evaluation = self.model_evaluations[model_name]

                # Plot predicted values

                axes[index].plot(
                    # evaluation.multiple_states_evaluations[state]["years"],
                    YEARS,
                    evaluation.multiple_states_evaluations[state]["predicted"][target],
                    label=f"{LABELS[LANG]['predicted_values']} - {model_name}",
                )

            # Set the axis
            axes[index].set_title(f"{translate_target(target, to_capitalize=True)}")
            axes[index].set_xlabel(LABELS[LANG]["xaxis"], fontsize=label_font_size)
            axes[index].set_ylabel(f"{LABELS[LANG]['yaxis']}", fontsize=label_font_size)
            axes[index].grid()

            # Update fontsize for legend
            legend = axes[index].legend()

            if legend and legend_font_size is not None:
                for text in legend.get_texts():
                    text.set_fontsize(legend_font_size)

        # Add some space
        fig.subplots_adjust(hspace=1)
        fig.tight_layout()
        return fig

    def create_comparision_plots(self) -> Dict[str, Figure]:
        """
        Creates dictionary of evaluation plot for every state and every model on every target prediction.

        Returns:
            out: Dict[str, Figure]: The key is the 'state' and the figure is the comparison plot of compared models. The plot of the reference / predicted values for all models and their targets.
        """

        state_plots: Dict[str, Figure] = {}

        for state in self.eval_states:
            state_plots[state] = self.create_state_comparison_plot(state=state)

        # return state_plots
        return state_plots


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Demography predictor models
    aging_comparation_model_names: List[str] = [
        "aging_base_model.pkl",
        "aging_ensemble_arima_Czechia.pkl",
        "aging_ensemble_lstm.pkl",
        "aging_finetunable_Czechia_model.pkl",
    ]

    # Local predictor models
    local_predictor_model_names: List[str] = [
        "test_base_model.pkl",
        "test_finetunable_model_Croatia.pkl",
    ]

    # Get models
    to_compare_models = get_multiple_models(names=aging_comparation_model_names)

    COMPARATION_STATES = ["Czechia", "Afghanistan", "United States", "Croatia"]

    model_comparator = ModelComparator()

    # # Example 1: comparing local predictor models using per feature metrics
    # ranked_models = model_comparator.compare_models_by_states(
    #     models=to_compare_models,
    #     states=COMPARATION_STATES,
    #     by="per-features",
    # )

    # print(
    #     ranked_models[
    #         (ranked_models["state"] == "Croatia")
    #         # & (ranked_models["target"].isin(["fertility rate, total", "arable land"]))
    #     ]
    # )

    # # Example 2: comparing demographic predictor by specified states
    ranked_models = model_comparator.compare_models_by_states(
        models=to_compare_models,
        states=COMPARATION_STATES,
        by="overall-metrics",
    )

    # Display
    for state in COMPARATION_STATES:
        print(ranked_models[ranked_models["state"] == state])
