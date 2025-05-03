# Standard library imports
from typing import List, Optional, Dict, Tuple
import pandas as pd
import os

import matplotlib.pyplot as plt


# Custom imports
from config import Config

from src.state_groups import StatesByWealth, StatesByGeolocation
from src.utils.log import setup_logging
from src.pipeline import PredictorPipeline
from src.statistical_models.multistate_wrapper import StatisticalMultistateWrapper

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.evaluation import EvaluateModel

from model_experiments.base_experiment import BaseExperiment
from src.shap_explainer.print_predictions import create_prediction_plots
from src.shap_explainer.explain import explain_best_worst_states


settings = Config()


def create_evaluation_data(
    pipeline: PredictorPipeline, states: Optional[List[str]] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    # Load data
    loader = StatesDataLoader()

    if states:
        test_data_dict = loader.load_states(states=states)
    else:
        test_data_dict = loader.load_all_states()

    if isinstance(pipeline.local_model_pipeline.model, StatisticalMultistateWrapper):
        sequence_len = 10
    else:
        sequence_len = pipeline.local_model_pipeline.sequence_length

    X_test_states, y_test_states = loader.split_data(
        test_data_dict,
        sequence_len=sequence_len,
    )

    return X_test_states, y_test_states


class EvalAllStates(BaseExperiment):

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self, pipeline: PredictorPipeline):

        # Create readme
        self.create_readme(readme_name=f"{pipeline.name}_README.md")

        X_test_states, y_test_states = create_evaluation_data(pipeline=pipeline)

        evaluation = EvaluateModel(pipeline=pipeline)

        # Eval for every state
        eval_df = evaluation.eval_for_every_state(
            X_test_states=X_test_states, y_test_states=y_test_states
        )

        ovearall_eval_df = evaluation.eval_for_every_state_overall(
            X_test_states=X_test_states, y_test_states=y_test_states
        )

        # Save to json file
        with open(os.path.join(self.plot_dir, "all_states_evaluation.json"), "w") as f:

            # Sort
            eval_df.sort_values(
                by=["rmse", "r2"], ascending=[True, False], inplace=True
            )
            eval_df.to_json(f, indent=4, orient="records")

        # Print top 5 states and the worst 5 states

        self.readme_add_section(
            title=f"Overall performance:",
            text=f"\n```{ovearall_eval_df}\n```\n\n",
        )

        top_states_df = eval_df.head(5)
        self.readme_add_section(
            title="Best performance states:", text=f"\n```{top_states_df}\n```\n\n"
        )

        worst_states_df = eval_df.tail(5)
        self.readme_add_section(
            title=f"Worst performance for states:",
            text=f"\n```{worst_states_df}\n```\n\n",
        )

        # Create shap explanation for
        explain_best_worst_states(
            pipeline=pipeline,
            custom_dir=self.plot_dir,
        )


class EvalGroupStates(BaseExperiment):

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def __evaluate_for_income_groups(self, pipeline: PredictorPipeline):

        STATES_BY_WEALTH = StatesByWealth()
        groups = STATES_BY_WEALTH.model_fields

        self.readme_add_section(
            title=f"# Wealth groups",
            text="",
        )

        evaluation = EvaluateModel(pipeline=pipeline)

        all_group_dfs: List[str] = []
        for group in groups:

            states = getattr(STATES_BY_WEALTH, group)

            X_test_dict, y_test_dict = create_evaluation_data(
                pipeline=pipeline, states=states
            )
            group_metric_df = evaluation.eval_for_every_state_overall(
                X_test_states=X_test_dict, y_test_states=y_test_dict
            )

            # Get the group and concat
            group_metric_df["group"] = group
            all_group_dfs.append(group_metric_df)

        all_groups_df = pd.concat(all_group_dfs, axis=0)
        all_groups_df.sort_values(by=["rmse"], ascending=[True], inplace=True)

        self.readme_add_section(
            title=f"## All income groups evaluation:",
            text=f"\n```{all_groups_df}\n```\n\n",
        )

    def __evaluate_for_geolocation_groups(self, pipeline: PredictorPipeline):

        STATES_BY_GEOLOCATION = StatesByGeolocation()
        groups = STATES_BY_GEOLOCATION.model_fields

        evaluation = EvaluateModel(pipeline=pipeline)

        self.readme_add_section(
            title=f"# Geolocoation groups",
            text="",
        )

        all_group_dfs: List[str] = []
        for group in groups:

            states = getattr(STATES_BY_GEOLOCATION, group)

            X_test_dict, y_test_dict = create_evaluation_data(
                pipeline=pipeline, states=states
            )
            group_metric_df = evaluation.eval_for_every_state_overall(
                X_test_states=X_test_dict, y_test_states=y_test_dict
            )

            # Get the group and concat
            group_metric_df["group"] = group
            all_group_dfs.append(group_metric_df)

        all_groups_df = pd.concat(all_group_dfs, axis=0)
        all_groups_df.sort_values(by=["rmse"], ascending=[True], inplace=True)

        self.readme_add_section(
            title=f"## All geolocation groups evaluation:",
            text=f"\n```{all_groups_df}\n```\n\n",
        )

    def run(self, pipeline: PredictorPipeline):

        # Create readme
        self.create_readme(readme_name=f"{pipeline.name}_README.md")

        # Evaluate for income groups
        self.__evaluate_for_income_groups(pipeline=pipeline)

        # Evaluate for geolocation groups
        self.__evaluate_for_geolocation_groups(pipeline=pipeline)


class EvalConvergenceExperiment(BaseExperiment):

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def print_prediction_plot(
        self, input_data: pd.DataFrame, prediction_df: pd.DataFrame, fig_name: str
    ):
        EXTRACT_VALUES = prediction_df.columns
        TARGET_COLUMNS = [col for col in prediction_df.columns if col != "year"]

        previous_target_data = input_data[EXTRACT_VALUES]

        complete_df = pd.concat([previous_target_data, prediction_df], axis=0)

        print(complete_df)

        # Plot
        fig, axes = plt.subplots(
            len(TARGET_COLUMNS), 1, figsize=(10, 5 * len(TARGET_COLUMNS))
        )

        if len(EXTRACT_VALUES) == 1:
            axes = [axes]

        for i, target in enumerate(TARGET_COLUMNS):
            ax = axes[i]

            #
            ax.plot(
                complete_df["year"], complete_df[target], color="b", label="Známe dáta"
            )
            ax.plot(
                prediction_df["year"],
                prediction_df[target],
                color="r",
                label="Predpovedané dáta",
            )

            ax.set_title(f"Predpovedané dáta pre {target}")
            ax.set_xlabel("Rok")
            ax.set_ylabel(target)
            ax.legend()

        self.save_plot(fig_name=fig_name, figure=fig)

    def run(self, pipeline: PredictorPipeline, states: List[str], target_year: int):
        # Create readme
        self.create_readme(readme_name=f"{pipeline.name}_README.md")

        # Load the test data
        loader = StatesDataLoader()

        states_data_dict = loader.load_states(states=states)

        for state, input_data in states_data_dict.items():
            prediction_df = pipeline.predict(
                input_data=input_data, target_year=target_year
            )
            self.print_prediction_plot(
                input_data=input_data,
                prediction_df=prediction_df,
                fig_name=f"{state}.png",
            )


def run_all(pipeline_name: str):
    pipeline = PredictorPipeline.get_pipeline(pipeline_name)

    exp = EvalAllStates(
        description="Evaluates pipeline for all available data of available populations."
    )

    exp_eval_groups = EvalGroupStates(description="Evalutes pipeline for groups.")

    convergence_exp = EvalConvergenceExperiment(
        description="Tries to predict data to specified years. Need to see where it converages."
    )

    # Run experiments
    exp.run(pipeline=pipeline)
    exp_eval_groups.run(pipeline=pipeline)
    convergence_exp.run(
        pipeline=pipeline, states=["Czechia", "United States"], target_year=2050
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    run_all()
