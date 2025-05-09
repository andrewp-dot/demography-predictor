# Copyright (c) 2025 Adrián Ponechal
# Licensed under the MIT License

# Standard library imports
import os
import pandas as pd
import click
from typing import Literal, Optional, List, Dict
import matplotlib.pyplot as plt


# Custom imports
from config import Config
from src.utils.log import setup_logging


from model_experiments.main import (
    print_available_experiments,
    run_all_experiments,
    run_experiments,
)

from src.train_scripts.train_predictors import (
    train_aging_predictor,
    train_gender_dist_predictor,
)

## Explain command
from src.shap_explainer.explain import explain_cli

## Compare predictions command
from src.preprocessors.state_preprocessing import StateDataLoader


## Compare command
from src.compare_models.compare import ModelComparator
from src.pipeline import PredictorPipeline


## Predictor-experiments command
from model_experiments.experiments.predictor_experiments import run_all


## Model selection experiments
from model_experiments.main import (
    run_feature_model_selection,
    run_target_model_selection,
)


settings = Config()


@click.group()
def cli():
    """
    Defines the main cli group
    """
    setup_logging()
    pass


@cli.group()
def model_experiments():
    """
    Defines the model experiment group. Used to run experiments on the models.
    """
    setup_logging()
    pass


@cli.command()
@click.option(
    "--name",
    type=str,
    help="The name of the required (trained) pipeline to explain.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["feature", "target", "full-predictor"], case_sensitive=False),
    help="The type of the required (trained) pipeline to explain.",
    required=True,
)
@click.option(
    "--experimental",
    is_flag=True,
    help="If set to true, loads the pipeline from the trained_modes/experimental_models folder.",
    default=False,
)
@click.option(
    "--core-metric",
    type=click.Choice(["mae", "mse", "rmse", "r2", "mape"]),
    help="All states will be sorted by this metric.",
    default="rmse",
)
@click.option(
    "--additional-states",
    type=str,
    help='List of additional states included in model explaination. List states as a comma separated strings string, e.g. --additional-states "Czechia, United States"',
    default=None,
)
def explain(
    name: str,
    type: Literal["feature", "target", "full-predictor"],
    experimental: bool,
    core_metric: Literal["mae", "mse", "rmse", "r2", "mape"],
    additional_states: str,
):
    """
    Explains the model using SHAP. The model is loaded from the trained_models folder.

    Args:
        name (str): Name of the model to explain
        type (Literal[&quot;feature&quot;, &quot;target&quot;, &quot;full&quot;): Type of the model (pipeline).
        experimental (bool): If set to true, loads the pipeline from the trained_modes/experimental_models folder.
        core_metric (Literal[&quot;mae&quot;, &quot;mse&quot;, &quot;rmse&quot;, &quot;r2&quot;, &quot;mape&quot;]): Core metric to sort the evaluation by.
        additional_states (str): States to include in the explanation.
    """

    # Parse states
    additional_states_list = (
        [state.strip() for state in additional_states.split(",")]
        if additional_states
        else []
    )

    explain_cli(
        pipeline_type=type,
        name=name,
        is_experimental=experimental,
        core_metric=core_metric,
        additional_states=additional_states_list,
    )


@cli.command()
@click.option(
    "--models",
    type=str,
    help='List of comma separated models to compare. as a comma separated strings string, e.g. --models "model_1, model_2"',
    required=True,
)
@click.option(
    "--state",
    type=str,
    help="For prediction future.",
    required=True,
)
@click.option(
    "--target-year",
    type=int,
    help="Get the target year for prediction.",
    default=2035,
)
@click.option(
    "--plot-prefix",
    type=str,
    help="If set, the plot will be saved with this prefix.",
    default="",
)
def compare_predictions(models: str, state: str, target_year: int, plot_prefix: str):
    """
    Compares predictions of the models. If models are not specified, all models are compared.
    """

    # Get models
    MODEL_NAMES = [model.strip() for model in models.split(",")]

    to_compare_models: Dict[str, PredictorPipeline] = {
        name: PredictorPipeline.get_pipeline(name=name) for name in MODEL_NAMES
    }

    # Get state input data
    loader = StateDataLoader(state=state)

    input_data = loader.load_data()

    # Create predictions for model
    TARGETS = None
    model_predictions: Dict[str, pd.DataFrame] = {}
    for name, model in to_compare_models.items():
        prediction_df = model.predict(
            input_data=input_data,
            target_year=target_year,
        )
        model_predictions[name] = prediction_df

        # Get targets
        if not TARGETS:
            TARGETS = model.TARGETS

    # Create plots
    fig, axes = plt.subplots(
        nrows=len(TARGETS),
        ncols=1,
        figsize=(10, 5 * len(TARGETS)),
        sharex=True,
    )

    # Labels
    LANG = settings.plot_description_language
    LABELS = {
        "en": {
            "prediction_for": "Prediction for",
            "year": "Year",
            "value": "Value",
        },
        "sk": {
            "prediction_for": "Predikcia",
            "year": "Rok",
            "value": "Hodnota",
        },
    }

    # Plot the predictions
    for i, target in enumerate(TARGETS):
        axes[i].set_title(f"{LABELS[LANG]['prediction_for']} {target}")
        axes[i].set_xlabel({LABELS[LANG]["year"]})
        axes[i].set_ylabel({LABELS[LANG]["value"]})

        for name, prediction_df in model_predictions.items():
            target_df = prediction_df[["year", target]]
            axes[i].plot(
                target_df["year"],
                target_df[target],
                label=name,
            )

        axes[i].legend()
        axes[i].grid()
        axes[i].set_xticks(prediction_df["year"].unique())
        axes[i].set_xticklabels(prediction_df["year"].unique(), rotation=45)

    # Save the plot
    plt.tight_layout()
    if plot_prefix:
        plot_prefix += "_"

    plt.savefig(os.path.join("imgs", "comparision_plots", f"{plot_prefix}{state}.png"))


@cli.command()
@click.option(
    "--models",
    type=str,
    help='List of comma separated models to compare. as a comma separated strings string, e.g. --models "model_1, model_2"',
    required=True,
)
@click.option(
    "--states",
    type=str,
    help='List of comma separated states to use for evaluation (or plotting) as a comma separated strings string, e.g. --states "Czechia, Honduras"',
    required=True,
)
@click.option(
    "--show-plots",
    is_flag=True,
    help="If specified, plots will be shown.",
    required=False,
    default=False,
)
@click.option(
    "--by",
    type=click.Choice(
        ["overall-metrics", "overall-one-metric", "per-targets"],
        case_sensitive=False,
    ),
    help="By which metric to compare models. Default is 'overall-metrics'.",
    default="overall-metrics",
)
def compare_predictors(
    models: str,
    states: str,
    show_plots: bool,
    by: Literal["overall-metrics", "overall-one-metric", "per-targets"],
):
    """
    Compares models. If models are not specified, all models are compared.
    """

    # Get models
    MODEL_NAMES = [model.strip() for model in models.split(",")]

    to_compare_models: Dict[str, PredictorPipeline] = {
        name: PredictorPipeline.get_pipeline(name=name) for name in MODEL_NAMES
    }

    COMPARATION_STATES = [state.strip() for state in states.split(",")]

    # Create comparator
    model_comparator = ModelComparator()

    # Compare models dict
    ranked_models = model_comparator.compare_models_by_states(
        pipelines=to_compare_models,
        states=COMPARATION_STATES,
        by=by,
    )

    # Display better models for state
    for state in COMPARATION_STATES:
        print(ranked_models[ranked_models["state"] == state])

    # Show plots
    if show_plots:
        for state in COMPARATION_STATES:
            model_comparator.create_state_comparison_plot(
                state=state,
                model_names=MODEL_NAMES,
            )
            # plt.show()
            plt.savefig(os.path.join("imgs", "comparision_plots", f"{state}.png"))


# Subgroup for the experiment group
# poetry run cli model_experiments --all
# poetry run cli model_experiments --experiments "name 1"  "name 2"
# poetry run cli model_experiments --list


@model_experiments.command()
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed descriptions of available experiments.",
)
def list(detailed):
    """
    Lists experiments. If detailed, lists experiments with their descriptions
    """

    print_available_experiments(with_description=detailed)


@model_experiments.command()
@click.option(
    "--all",
    is_flag=True,
    help="Runs all experiments with default parameters. Defualt parameters are defined in the config file.",
)
@click.option(
    "--state",
    type=str,
    help="Specifies the state to run experiment or evaluation on.",
    default="Czechia",
)
@click.option(
    "--split-rate",
    type=float,
    help="Size of the training set. For example 0.8 is equal to 80% of the dataset is used for training.",
    default=0.8,
)
@click.argument(
    "experiments",
    nargs=-1,
)
def run(all, experiments, state, split_rate):
    """
    Runs experiments based on given options.
    """
    setup_logging()

    if all:
        run_all_experiments(state=state, split_rate=split_rate)

    elif experiments:
        run_experiments(experiments=experiments, state=state, split_rate=split_rate)


@cli.group()
def train():
    """
    Defines training group, used to train models.
    """
    setup_logging()
    pass


@train.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor. It is also a model key.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["LSTM", "ARIMA"], case_sensitive=False),
    help="Type of the model. You can choose from options: LSTM, ARIMA",
    default="ARIMA",
)
@click.option(
    "--wealth-groups",
    type=str,
    help='List of comma separated wealth groups. as a comma separated strings string, e.g. --wealth-groups "high_income, lower_middle_income"',
)
@click.option(
    "--geolocation-groups",
    type=str,
    help='List of comma separated geolocation groups. as a comma separated strings string, e.g. --states "europe, north america"',
)
@click.option(
    "--states",
    type=str,
    help='List of included in training states as a comma separated strings string, e.g. --states "Czechia, United States"',
)
@click.option(
    "--modify-for-target-model",
    is_flag=True,
    help="If specified, modifed data will be used to train the target prediction model. By deafult, data are adjusted for feature target model",
)
def age_predictor(
    name: str,
    type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[str],
    geolocation_groups: Optional[str],
    states: Optional[str],
    modify_for_target_model: bool,
):

    # Parse data arguments
    if states:
        states: List[str] = [s.strip() for s in states.split(",")]
    if wealth_groups:
        wealth_groups: List[str] = [s.strip() for s in wealth_groups.split(",")]
    if geolocation_groups:
        geolocation_groups: List[str] = [
            s.strip() for s in geolocation_groups.split(",")
        ]

    train_aging_predictor(
        name=name,
        model_type=type,
        states=states,
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        modify_for_target_model=modify_for_target_model,
    )


@train.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor. It is also a model key.",
    required=True,
)
@click.option(
    "--type",
    type=click.Choice(["LSTM", "ARIMA"], case_sensitive=False),
    help="Type of the model. You can choose from options: LSTM, ARIMA",
    default="ARIMA",
)
@click.option(
    "--wealth-groups",
    type=str,
    help='List of comma separated wealth groups. as a comma separated strings string, e.g. --wealth-groups "high_income, lower_middle_income"',
)
@click.option(
    "--geolocation-groups",
    type=str,
    help='List of comma separated geolocation groups. as a comma separated strings string, e.g. --states "europe, north america"',
)
@click.option(
    "--states",
    type=str,
    help='List of included in training states as a comma separated strings string, e.g. --states "Czechia, United States"',
)
@click.option(
    "--modify-for-target-model",
    is_flag=True,
    help="If specified, modifed data will be used to train the target prediction model. By deafult, data are adjusted for feature target model",
)
def gender_dist_predictor(
    name: str,
    type: Literal["LSTM", "ARIMA"],
    wealth_groups: Optional[str],
    geolocation_groups: Optional[str],
    states: Optional[str],
    modify_for_target_model: bool,
):

    # Parse data arguments
    if states:
        states: List[str] = [s.strip() for s in states.split(",")]
    if wealth_groups:
        wealth_groups: List[str] = [s.strip() for s in wealth_groups.split(",")]
    if geolocation_groups:
        geolocation_groups: List[str] = [
            s.strip() for s in geolocation_groups.split(",")
        ]

    train_gender_dist_predictor(
        name=name,
        model_type=type,
        states=states,
        wealth_groups=wealth_groups,
        geolocation_groups=geolocation_groups,
        modify_for_target_model=modify_for_target_model,
    )


@cli.group()
def predictor_experiments():
    """
    The predictor validation/evaluation experiments.
    """
    setup_logging()
    pass


@predictor_experiments.command()
@click.option(
    "--name",
    type=str,
    help="Name of the predictor to run predictor experiments with.",
    required=True,
)
def run(name: str):
    run_all(pipeline_name=name)


@cli.group()
def model_selection():
    """
    The model selection experiments.
    """
    setup_logging()
    pass


@model_selection.command()
@click.option(
    "--split-rate",
    type=float,
    help="Size of the training set. For example 0.8 is equal to 80% of the dataset is used for training.",
    default=0.8,
)
@click.option(
    "--force-retrain",
    is_flag=True,
    help="If set, the model will be retrained even if it already exists.",
)
@click.option(
    "--only-rnn-retrain",
    is_flag=True,
    help="If set, only the RNN model will be retrained.",
)
@click.option(
    "--evaluation-states",
    type=str,
    help='List of comma separated states to use for evaluation (or plotting) as a comma separated strings string, e.g. --evaluation-states "Czechia, Honduras"',
)
@click.option(
    "--core-metric",
    type=click.Choice(["mae", "rmse", "mape"]),
    default="rmse",
    help="Core metric to sort the evaluation by.",
)
def feature_model(
    split_rate: float = 0.8,
    force_retrain: bool = False,
    only_rnn_retrain: bool = False,
    evaluation_states: Optional[str] = None,
    core_metric: Literal["mae", "rmse", "mape"] = "rmse",
):
    """
    Runs the feature model selection experiments.
    """

    evaluation_states_list = None
    if evaluation_states:
        evaluation_states_list = [
            state.strip() for state in evaluation_states.split(",")
        ]

    # Run the feature model selection
    run_feature_model_selection(
        split_rate=split_rate,
        force_retrain=force_retrain,
        only_rnn_retrain=only_rnn_retrain,
        evaluation_states=evaluation_states_list,
        core_metric=core_metric,
    )


@model_selection.command()
@click.option(
    "--split-rate",
    type=float,
    help="Size of the training set. For example 0.8 is equal to 80% of the dataset is used for training.",
    default=0.8,
)
@click.option(
    "--force-retrain",
    is_flag=True,
    help="If set, the model will be retrained even if it already exists.",
)
@click.option(
    "--only-rnn-retrain",
    is_flag=True,
    help="If set, only the RNN model will be retrained.",
)
@click.option(
    "--evaluation-states",
    type=str,
    help='List of comma separated states to use for evaluation (or plotting) as a comma separated strings string, e.g. --evaluation-states "Czechia, Honduras"',
)
@click.option(
    "--exp-type",
    type=click.Choice(["aging", "pop_total", "gender_dist"]),
    default="aging",
    help="Type of the experiment to run. Default is 'aging'.",
)
@click.option(
    "--core-metric",
    type=click.Choice(["mae", "rmse", "mape"]),
    default="rmse",
    help="Core metric to sort the evaluation by.",
)
def target_model(
    split_rate: float = 0.8,
    force_retrain: bool = False,
    only_rnn_retrain: bool = False,
    evaluation_states: Optional[str] = None,
    exp_type: Literal["aging", "pop_total", "gender_dist"] = "aging",
    core_metric: Literal["mae", "rmse", "mape"] = "rmse",
):
    """
    Runs the target model selection experiments.
    """

    evaluation_states_list = None
    if evaluation_states:
        evaluation_states_list = [
            state.strip() for state in evaluation_states.split(",")
        ]

    # Run the target model selection
    run_target_model_selection(
        split_rate=split_rate,
        force_retrain=force_retrain,
        only_rnn_retrain=only_rnn_retrain,
        evaluation_states=evaluation_states_list,
        exp_type=exp_type,
        core_metric=core_metric,
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run cli
    cli()
