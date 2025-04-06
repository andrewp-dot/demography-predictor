# Standard library imports
import pandas as pd
import logging

from typing import List, Dict, Union, Literal

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_model, get_multiple_models
from src.utils.constants import get_core_hyperparameters

from src.base import CustomModelBase
from src.evaluation import EvaluateModel
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.ensemble_model import PureEnsembleModel

from src.predictors.predictor_base import DemographyPredictor

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

logger = logging.getLogger("model_compare")


def rank_models(evaluation_dfs: List[pd.DataFrame]) -> pd.DataFrame:
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
def compare_models_by_states(
    models: Dict[str, Union[DemographyPredictor, CustomModelBase]],
    states: List[str] | None = None,
    by: Literal["overall-metrics", "per-features"] = "overall-metrics",
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
    if len(models) <= 1:
        logger.warning("No models to compare.")
        return {}

    # Check if the models has same features and targets
    # first_model_features: List[str] = to_compare_models[models[0]].FEATURES
    model_names: List[str] = list(models.keys())
    first_model_targets: List[str] = models[model_names[0]].TARGETS

    for model_name in model_names[1:]:
        # Get next model
        model = models[model_name]

        # Check if they have the same targets
        if models[model_name].TARGETS != first_model_targets:
            raise ValueError(
                f"The model '{model_name}' has different features then the first model"
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
    for model_name, model in models.items():
        # Preprocess data for the model - suppoorts different sequence length, by type

        # Adjust hyperparameters by the model type
        if isinstance(model, DemographyPredictor):
            if isinstance(model.local_model, PureEnsembleModel):
                model_sequence_len = get_core_hyperparameters(
                    input_size=1
                ).sequence_length
            else:
                model_sequence_len = model.local_model.hyperparameters.sequence_length
        elif isinstance(model, CustomModelBase):
            model_sequence_len = model.hyperparameters.sequence_length

        train_data_dict, test_data_dict = states_loaders.split_data(
            states_dict=states_data_dict,
            sequence_len=model_sequence_len,
        )

        model_evaluation = EvaluateModel(model=model)

        # Save the evaluation
        if "overall-metrics" == by:
            model_evaluations[model_name] = model_evaluation.eval_for_every_state(
                X_test_states=train_data_dict, y_test_states=test_data_dict
            )

        elif "per-features" == by:

            states_evaluation_for_model: pd.DataFrame | None = None
            for state in train_data_dict.keys():

                # Get per target per state evaluation
                model_evaluation.eval(
                    test_X=train_data_dict[state], test_y=test_data_dict[state]
                )
                per_target_metrics = model_evaluation.get_target_specific_metrics(
                    model.FEATURES
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
    df_ranked = rank_models(df_list)

    return df_ranked


# Compare models by features
def compare_models_by_features(models: List[str]) -> pd.DataFrame:
    # Check if there is something to compare
    if len(models) <= 1:
        logger.warning("No models to compare.")
        return {}

    # For every model get evaluation
    to_compare_models: Dict[str, Union[DemographyPredictor, CustomModelBase]] = {
        model_name: get_model(model_name) for model_name in models
    }

    # Check if the models has same features and targets
    # first_model_features: List[str] = to_compare_models[models[0]].FEATURES
    first_model_targets: List[str] = to_compare_models[models[0]].TARGETS

    for model_name in models[1:]:
        # Get next model
        model = to_compare_models[model_name]

        # Check if they have the same targets
        if model.TARGETS != first_model_targets:
            raise ValueError(
                f"The model '{model_name}' has different features then the first model"
            )

    # Get evaluation for all states
    model_evaluations: Dict[str, pd.DataFrame] = {}


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

    # # Example 1: comparing local predictor models using per feature metrics
    # ranked_models = compare_models_by_states(
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
    ranked_models = compare_models_by_states(
        models=to_compare_models,
        states=COMPARATION_STATES,
        by="overall-metrics",
    )

    # Display
    for state in COMPARATION_STATES:
        print(ranked_models[ranked_models["state"] == state])
