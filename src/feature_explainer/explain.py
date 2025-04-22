# TODO:
# 1. get pipeline - (LocalModelPipeline, GlobalModelPipeline) or (PredictorPipeline)
# 2. get the data (state) as an inputs
#   2.1 figure out where to store results
# 3. explain features
# 4. create plots -> top 5 best performing, top 5 worst performing

# Note:
# model_test.py


# Standard library imports
import os
import logging

from typing import Union, List


# Custom imports
from src.utils.log import setup_logging
from src.pipeline import LocalModelPipeline, GlobalModelPipeline, PredictorPipeline
from src.feature_explainer.explainers import LSTMExplainer, GlobalModelExplainer

# TODO: create logger
logger = logging.getLogger("feature_engineering")


def setup_save_dir(save_path: str):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


def explain_lstm(pipeline: LocalModelPipeline, save_path: str, states: List[str]):
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

        # explainer.get_waterfall_plot(
        #     shap_values=shap_values,
        #     input_x=sequences,
        #     sample_idx=0,
        #     target_index=0,
        # )

        print(f"Done!")
        print()


def explain_tree(pipeline: GlobalModelPipeline, save_path: str, states: List[str]):
    setup_save_dir(save_path=save_path)
    raise NotImplementedError("")


def main(
    pipeline: Union[LocalModelPipeline, GlobalModelPipeline, PredictorPipeline],
    states: List[str],
):

    # Save the explenation with plots to folder
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "explanations", pipeline.name)

    # Get the pipeline type
    if isinstance(pipeline, LocalModelPipeline):
        # Explain base lstm
        explain_lstm(pipeline=pipeline, save_path=SAVE_PATH, states=states)

    elif isinstance(pipeline, GlobalModelPipeline):
        # Explain XGB -> tree model explainer
        explain_tree(pipeline=pipeline, save_path=SAVE_PATH)

    elif isinstance(pipeline, PredictorPipeline):
        # Explain base lstm  and explain XGB -> tree model explainer
        explain_lstm(pipeline=pipeline.local_model_pipeline, save_path=SAVE_PATH)
        explain_tree(pipeline=pipeline.global_model_pipeline, save_path=SAVE_PATH)
    else:
        raise ValueError(
            f"The explanation for pipeline of type '{type(pipeline)}' is not implemented yet!"
        )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Get pipeline
    PIPELINE_NAME = "lstm_features_only"
    pipeline = LocalModelPipeline.get_pipeline(name=PIPELINE_NAME)

    # Run explainer for the pipeline
    main(pipeline, states=["Czechia", "Honduras"])
