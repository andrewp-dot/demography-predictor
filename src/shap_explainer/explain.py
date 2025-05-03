# Standard library imports
import os
import logging

from typing import Union, List

import matplotlib.pyplot as plt


# Custom imports
from src.utils.log import setup_logging
from src.pipeline import FeatureModelPipeline, TargetModelPipeline, PredictorPipeline
from src.shap_explainer.explainers import LSTMExplainer, TargetModelExplainer
from src.shap_explainer.print_predictions import create_prediction_plots

from src.evaluation import EvaluateModel
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

# TODO: create logger
logger = logging.getLogger("explain")


def setup_save_dir(save_path: str):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


def explain_lstm(pipeline: FeatureModelPipeline, save_path: str, states: List[str]):
    setup_save_dir(save_path=save_path)

    explainer = LSTMExplainer(pipeline=pipeline)

    # Explain for every state
    for state in states:

        print(f"Creating plots for state {state}...")

        # Get results
        sequences = explainer.create_sequences(state=state)

        shap_values = explainer.get_shap_values(input_sequences=sequences)

        state_save_path = os.path.join(save_path, state)
        os.makedirs(state_save_path, exist_ok=True)

        # Get feature importance and save the plot
        explainer.get_feature_importance(
            shap_values=shap_values, save_path=state_save_path
        )

        # Get plots
        explainer.get_force_plot(
            save_path=state_save_path,
            shap_values=shap_values,
            input_sequences=sequences,
            sample_idx=0,
            time_step=0,
        )

        explainer.get_summary_plot(
            save_path=state_save_path,
            shap_values=shap_values,
            input_x=sequences,
            sample_idx=0,
            target_index=0,
        )

        explainer.get_waterfall_plot(
            save_path=state_save_path,
            shap_values=shap_values,
            input_x=sequences,
            sample_idx=0,
            target_index=0,
        )

        print(f"Done!")
        print()


def explain_tree(pipeline: TargetModelPipeline, save_path: str, states: List[str]):
    setup_save_dir(save_path=save_path)

    explainer = TargetModelExplainer(pipeline=pipeline)

    for state in states:

        print(f"Creating plots for state {state}...")

        # Get results
        input_data = explainer.create_inputs(state=state)

        shap_values = explainer.get_shap_values(X=input_data)

        # print(shap_values)

        state_save_path = os.path.join(save_path, state)
        os.makedirs(state_save_path, exist_ok=True)

        explainer.get_feature_importance(
            shap_values=shap_values, save_path=state_save_path
        )

        explainer.get_force_plot(shap_values=shap_values, save_path=state_save_path)

        explainer.get_summary_plot(shap_values=shap_values, save_path=state_save_path)

        explainer.get_waterfall_plot(shap_values=shap_values, save_path=state_save_path)

        print("Done!")
        print()


def explain(
    pipeline: Union[FeatureModelPipeline, TargetModelPipeline, PredictorPipeline],
    states: List[str],
):

    # Save the explenation with plots to folder
    SAVE_PATH = os.path.join(".", "explanations", pipeline.name)

    # Get the pipeline type
    if isinstance(pipeline, FeatureModelPipeline):
        # Explain base lstm
        explain_lstm(pipeline=pipeline, save_path=SAVE_PATH, states=states)

    elif isinstance(pipeline, TargetModelPipeline):
        # Explain XGB -> tree model explainer
        explain_tree(pipeline=pipeline, save_path=SAVE_PATH, states=states)

    elif isinstance(pipeline, PredictorPipeline):
        # Explain base lstm  and explain XGB -> tree model explainer

        explain_lstm(
            states=states,
            pipeline=pipeline.local_model_pipeline,
            save_path=os.path.join(SAVE_PATH, "lstm"),
        )
        explain_tree(
            states=states,
            pipeline=pipeline.global_model_pipeline,
            save_path=os.path.join(SAVE_PATH, "gm"),
        )
    else:
        raise ValueError(
            f"The explanation for pipeline of type '{type(pipeline)}' is not implemented yet!"
        )


def explain_best_worst_states(
    pipeline: Union[FeatureModelPipeline, TargetModelPipeline, PredictorPipeline],
    metric: str = "rmse",
):

    evaluation = EvaluateModel(pipeline=pipeline)

    loader = StatesDataLoader()
    all_states_data_dict = loader.load_all_states()

    if isinstance(pipeline, FeatureModelPipeline):
        sequence_len = pipeline.model.hyperparameters.sequence_length
    elif isinstance(pipeline, TargetModelPipeline):
        sequence_len = 10
    elif isinstance(pipeline, PredictorPipeline):
        sequence_len = (
            pipeline.local_model_pipeline.model.hyperparameters.sequence_length
        )

    X_test_states, y_test_states = loader.split_data(
        states_dict=all_states_data_dict, sequence_len=sequence_len
    )

    one_metric = evaluation.eval_for_every_state_overall(
        X_test_states=X_test_states, y_test_states=y_test_states
    )
    logger.info(f"\n{one_metric}")

    # Eval for all states per targets
    states_per_target_dict = evaluation.eval_for_every_state_per_target(
        X_test_states=X_test_states, y_test_states=y_test_states
    )

    logger.info(
        f"\n{states_per_target_dict[states_per_target_dict['state'] == 'Czechia']}"
    )

    every_state_evaluation_df = evaluation.eval_for_every_state(
        X_test_states=X_test_states, y_test_states=y_test_states
    )
    every_state_evaluation_df.sort_values(by=[metric], inplace=True)

    SAVE_PATH = os.path.join(".", "explanations", pipeline.name)

    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join(SAVE_PATH, "state_performance_sorted.json"), "w") as f:
        every_state_evaluation_df.to_json(f, indent=4, orient="records")

    best_state = every_state_evaluation_df.iloc[0]["state"]
    worst_state = every_state_evaluation_df.iloc[-1]["state"]

    logger.info(f"Best state: {best_state}, Worst state: {worst_state}")
    explain(pipeline, states=[best_state, worst_state])

    state_plots = create_prediction_plots(
        pipeline=pipeline, states=[best_state, worst_state]
    )

    # Save it to dir
    for state, plot in state_plots.items():
        path = os.path.join(SAVE_PATH, f"{state}_predictions.png")
        plt.figure(plot.number)
        plt.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Get pipeline
    # TODO: maybe do some arguments in here?
    # PIPELINE_NAME = "lstm_features_only"
    # pipeline = FeatureModelPipeline.get_pipeline(name=PIPELINE_NAME)

    PIPELINE_NAME = "test-target-model-tree"
    # pipeline = PredictorPipeline.get_pipeline(name=PIPELINE_NAME)
    pipeline = TargetModelPipeline.get_pipeline(name=PIPELINE_NAME)

    explain_best_worst_states(pipeline=pipeline)
