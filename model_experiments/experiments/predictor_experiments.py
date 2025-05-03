# Standard library imports
from typing import List, Optional, Dict, Tuple
import pandas as pd
import os


# Custom imports
from config import Config

from src.utils.constants import (
    basic_features,
    aging_targets,
    gender_distribution_targets,
)
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
        self.create_readme()

        X_test_states, y_test_states = create_evaluation_data(pipeline=pipeline)

        evaluation = EvaluateModel(pipeline=pipeline)

        # Eval for every state

        eval_df = evaluation.eval_for_every_state(
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

        for group in groups:

            states = getattr(STATES_BY_WEALTH, group)

            X_test_dict, y_test_dict = create_evaluation_data(
                pipeline=pipeline, states=states
            )
            group_metric_df = evaluation.eval_for_every_state_overall(
                X_test_states=X_test_dict, y_test_states=y_test_dict
            )

            self.readme_add_section(
                title=f"## Group {group} evaluation:",
                text=f"\n```{group_metric_df}\n```\n\n",
            )

    def __evaluate_for_geolocation_groups(self, pipeline: PredictorPipeline):

        STATES_BY_GEOLOCATION = StatesByGeolocation()
        groups = STATES_BY_GEOLOCATION.model_fields

        evaluation = EvaluateModel(pipeline=pipeline)

        self.readme_add_section(
            title=f"# Geolocoation groups",
            text="",
        )

        for group in groups:

            states = getattr(STATES_BY_GEOLOCATION, group)

            X_test_dict, y_test_dict = create_evaluation_data(
                pipeline=pipeline, states=states
            )
            group_metric_df = evaluation.eval_for_every_state_overall(
                X_test_states=X_test_dict, y_test_states=y_test_dict
            )

            self.readme_add_section(
                title=f"## Group {group} evaluation:",
                text=f"\n```{group_metric_df}\n```\n\n",
            )

    def run(self, pipeline: PredictorPipeline):

        # Create readme
        self.create_readme()

        # Evaluate for income groups
        self.__evaluate_for_income_groups(pipeline=pipeline)

        # Evaluate for geolocation groups
        self.__evaluate_for_geolocation_groups(pipeline=pipeline)

        # Create shap explanation for
        # explain(
        #     pipeline=pipeline,
        #     states=[top_states_df.iloc[0]["state"], top_states_df.iloc[-1]["state"]],
        #     custom_dir=self.plot_dir,
        # )


class EvalConvergenceExperiment(BaseExperiment):

    def __init__(self, description: str):
        super().__init__(name=self.__class__.__name__, description=description)

    def run(self, pipeline: PredictorPipeline, states: List[str]):

        # Create readme
        self.create_readme()

        # Get the model

        loader = StatesDataLoader()

        X_test_dict, y_test_dict = create_evaluation_data(
            pipeline=pipeline, states=states
        )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    pipeline_name = "age-predictor"
    pipeline = PredictorPipeline.get_pipeline(pipeline_name)

    exp = EvalAllStates(
        description="Evaluates pipeline for all available data of available populations."
    )

    exp_eval_groups = EvalGroupStates(description="Evalutes pipeline for groups.")

    exp.run(pipeline=pipeline)
    # exp_eval_groups.run(pipeline=pipeline)
