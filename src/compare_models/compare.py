# Standard library imports
import pandas as pd
import logging

from typing import List, Dict, Union

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_model

from src.local_model.base import EvaluateModel, CustomModelBase
from src.local_model.finetunable_model import FineTunableLSTM

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
            + (df["r2"].rank(ascending=False))  # Higher R² is better
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
    models: List[str], states: List[str] | None = None
) -> pd.DataFrame:

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

    # Get evaluation data
    states_loaders = StatesDataLoader()

    # Get the data by given states or all
    if states is None:
        states_data_dict = states_loaders.load_all_states()
    else:
        states_data_dict = states_loaders.load_states(states=states)

    # Getnerate evaluation for every model
    for model_name, model in to_compare_models.items():
        # Preprocess data for the model - suppoorts different sequence length, by type

        # Adjust hyperparameters by the model type
        if isinstance(model, DemographyPredictor):
            model_sequence_len = model.local_model.hyperparameters.sequence_length
        elif isinstance(model, CustomModelBase):
            model_sequence_len = model.hyperparameters.sequence_length

        train_data_dict, test_data_dict = states_loaders.split_data(
            states_dict=states_data_dict,
            sequence_len=model_sequence_len,
        )

        model_evaluation = EvaluateModel(model=model)
        model_evaluation.eval_for_every_state(
            X_test_states=train_data_dict, y_test_states=test_data_dict
        )

        # Save the evaluation
        model_evaluations[model_name] = model_evaluation.all_states_evaluation

        # TODO: do this for the rest of this 2 evaluations
        # model_evaluation.eval(
        #     test_X=train_data_dict["Czechia"], test_y=test_data_dict["Czechia"]
        # )
        # model_evaluation.get_overall_metrics()
        # print(model_evaluation.overall_metrics.head())
        # model_evaluation.get_feature_specific_metrics(model.FEATURES)
        # print(model_evaluation.per_target_metrics.head())

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
    aging_comparation_models: List[str] = [
        "aging_Czechia.pkl",
        "aging_Czechia_model.pkl",
        "aging_rich_group_model.pkl",
        "aging_model.pkl",
    ]

    # TODO:
    # Train and try to evaluate local predictor models
    # Local predictor models
    local_predictor_models: List[str] = [
        "test_base_model.pkl",
        "test_finetunable_model_Croatia.pkl",
    ]

    # Get models
    ranked_models = compare_models_by_states(
        models=local_predictor_models,
        states=["Czechia", "Afghanistan", "United States", "Croatia"],
    )

    # print(ranked_models[ranked_models["state"] == "Czechia"])
    print(ranked_models)
