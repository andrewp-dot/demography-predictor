# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import os
from typing import List, Dict, Union
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Cutsom imports
from src.utils.log import setup_logging
from src.evaluation import EvaluateModel
from src.pipeline import FeatureModelPipeline, TargetModelPipeline, PredictorPipeline
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def create_prediction_plots(
    pipeline: Union[FeatureModelPipeline | TargetModelPipeline | PredictorPipeline],
    states: List[str],
) -> Dict[str, Figure]:

    # Load data
    loader = StatesDataLoader()
    to_plot_states_dict = loader.load_states(states=states)

    if isinstance(pipeline, FeatureModelPipeline):

        sequence_len = pipeline.model.hyperparameters.sequence_length
    elif isinstance(pipeline, TargetModelPipeline):
        sequence_len = pipeline.model.sequence_len

    elif isinstance(pipeline, PredictorPipeline):
        sequence_len = max(
            pipeline.local_model_pipeline.sequence_length,
            pipeline.global_model_pipeline.model.sequence_len,
        )

    # Split data

    # TODO: add support for features steps
    X_test_dict, y_test_dict = loader.split_data(
        to_plot_states_dict, sequence_len=sequence_len
    )

    # Save the evaluation predictions to dir
    plots: Dict[str, Figure] = {}
    evaluation = EvaluateModel(pipeline=pipeline)

    for state in states:

        evaluation.eval(test_X=X_test_dict[state], test_y=y_test_dict[state])
        plots[state] = evaluation.plot_predictions()

    return plots


def save_plots(pipeline_name: str, plot_dict: Dict[str, Figure], save_dir: str) -> None:

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    for state, fig in plot_dict.items():
        file_path = os.path.join(
            save_dir, f"{pipeline_name}_{state}.png"
        )  # Save each figure as a PNG

        # Save the figure
        fig.savefig(file_path)
        plt.close(fig)


if __name__ == "__main__":

    # Setup logging
    setup_logging()

    TO_PLOT_STATES: List[str] = ["Czechia", "Honduras"]

    # PIPELINE = FeatureModelPipeline.get_pipeline(name="lstm_features_only")
    # PIPELINE = TargetModelPipeline.get_pipeline(name="test_gm")
    PIPELINE = PredictorPipeline.get_pipeline(name="test_predictor")

    BEST_PERFORMING_STATES: List[str] = [
        "Guatemala",
        "St. Vincent and the Grenadines",
        "Nepal",
        "Venezuela, RB",
        "Philippines",
    ]
    WORST_PERFORMING_STATES: List[str] = [
        "Egypt, Arab Rep.",
        "Chile",
        "Vanuatu",
        "Senegal",
        "Solomon Islands",
    ]

    plot_dict = create_prediction_plots(
        pipeline=PIPELINE, states=BEST_PERFORMING_STATES
    )
    save_plots(
        pipeline_name=PIPELINE.name, plot_dict=plot_dict, save_dir="./imgs/good-states"
    )

    plot_dict = create_prediction_plots(
        pipeline=PIPELINE, states=WORST_PERFORMING_STATES
    )
    save_plots(
        pipeline_name=PIPELINE.name, plot_dict=plot_dict, save_dir="./imgs/bad-states"
    )
