# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import os
import logging

from typing import Union, List, Optional, Literal
import matplotlib.pyplot as plt


# Custom imports
from src.utils.log import setup_logging
from src.pipeline import FeatureModelPipeline, TargetModelPipeline, PredictorPipeline
from src.shap_explainer.explainers import LSTMExplainer, TargetModelExplainer
from src.shap_explainer.print_predictions import create_prediction_plots
from src.feature_model.model import BaseRNN


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
    custom_dir: Optional[str] = None,
):

    # Save the explenation with plots to folder
    if custom_dir:
        SAVE_PATH = os.path.join(custom_dir, pipeline.name)
    else:
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

        if isinstance(pipeline.local_model_pipeline.model, BaseRNN):
            explain_lstm(
                states=states,
                pipeline=pipeline.local_model_pipeline,
                save_path=os.path.join(SAVE_PATH, "lstm"),
            )
        explain_tree(
            states=states,
            pipeline=pipeline.global_model_pipeline,
            save_path=os.path.join(SAVE_PATH, "target-model"),
        )
    else:
        raise ValueError(
            f"The explanation for pipeline of type '{type(pipeline)}' is not implemented yet!"
        )


def explain_best_worst_states(
    pipeline: Union[FeatureModelPipeline, TargetModelPipeline, PredictorPipeline],
    metric: str = "rmse",
    custom_dir: Optional[str] = None,
    additional_states: Optional[List[str]] = None,
):

    # Create evaluation object and load data
    evaluation = EvaluateModel(pipeline=pipeline)

    loader = StatesDataLoader()
    all_states_data_dict = loader.load_all_states()

    if isinstance(pipeline, FeatureModelPipeline):
        sequence_len = pipeline.sequence_length
    elif isinstance(pipeline, TargetModelPipeline):
        sequence_len = pipeline.model.sequence_len
    elif isinstance(pipeline, PredictorPipeline):

        # If it is BaseRNN just get sequence len form that
        sequence_len = max(
            pipeline.local_model_pipeline.sequence_length,
            pipeline.global_model_pipeline.model.sequence_len,
        )

    X_test_states, y_test_states = loader.split_data(
        states_dict=all_states_data_dict, sequence_len=sequence_len
    )

    # Evaluate for every state with overall metric
    one_metric = evaluation.eval_for_every_state_overall(
        X_test_states=X_test_states, y_test_states=y_test_states
    )
    logger.info(f"\n{one_metric}")

    # Evaluate for every state with separate metric for every state
    every_state_evaluation_df = evaluation.eval_for_every_state(
        X_test_states=X_test_states, y_test_states=y_test_states
    )
    every_state_evaluation_df.sort_values(by=[metric], inplace=True)

    # Save gathered evaluations
    SAVE_PATH = os.path.join(".", "explanations", pipeline.name)
    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join(SAVE_PATH, "state_performance_sorted.json"), "w") as f:
        every_state_evaluation_df.to_json(f, indent=4, orient="records")

    # Print best and worst state (or additional state). Create predictions, explain the model using SHAP plots.
    best_state = every_state_evaluation_df.iloc[0]["state"]
    worst_state = every_state_evaluation_df.iloc[-1]["state"]

    logger.info(f"Best state: {best_state}, Worst state: {worst_state}")

    # Prevent 'None' unpacking error
    additional_states = additional_states if additional_states else []

    explain(
        pipeline,
        states=[best_state, worst_state, *additional_states],
        custom_dir=custom_dir,
    )

    state_plots = create_prediction_plots(
        pipeline=pipeline, states=[best_state, worst_state, *additional_states]
    )

    # Save it to dir
    for state, plot in state_plots.items():
        path = os.path.join(SAVE_PATH, f"{state}_predictions.png")
        plt.figure(plot.number)
        plt.savefig(path, bbox_inches="tight")


def explain_cli(
    pipeline_type: Literal["feature", "target", "full-predictor"],
    name: str,
    is_experimental: bool,
    core_metric: Literal["mae", "mse", "rmse", "r2", "mape"] = "rmse",
    additional_states: Optional[List[str]] = None,
) -> Union[FeatureModelPipeline, TargetModelPipeline, PredictorPipeline]:

    if "feature" == pipeline_type:
        load_type = FeatureModelPipeline
    elif "target" == pipeline_type:
        load_type = TargetModelPipeline
    elif "full-predictor" == PredictorPipeline:
        load_type = PredictorPipeline
    else:
        raise ValueError(f"Unsupported type of pipeline. ({pipeline_type})")

    pipeline = load_type.get_pipeline(name=name, experimental=is_experimental)
    explain_best_worst_states(
        pipeline=pipeline, metric=core_metric, additional_states=additional_states
    )


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Example arguments
    PIPELINE_NAME = "test-target-model-tree"
    PIPELINE_TYPE = "target"
    IS_EXPERIMENTAL = False
    CORE_METRIC = "rmse"
    ADDITIONAL_STATES = ["Czechia"]

    explain_cli(
        pipeline_type=PIPELINE_TYPE,
        name=PIPELINE_NAME,
        is_experimental=IS_EXPERIMENTAL,
        core_metric=CORE_METRIC,
        additional_states=ADDITIONAL_STATES,
    )
