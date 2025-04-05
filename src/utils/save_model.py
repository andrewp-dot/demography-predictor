import os
import joblib
from typing import Any, List, Dict

from config import Config


settings = Config()


def get_model(name: str) -> Any:
    """
    Get the model object from the specified directory in the config file.

    Args:
        name (str): Name of the model (including .pkl suffix).

    Raises:
        ValueError: If the model does not exist, raises an error.

    Returns:
        out: Any: the trained model object.
    """
    MODEL_PATH = os.path.join(settings.trained_models_dir, f"{name}")

    if not os.path.isfile(MODEL_PATH):
        raise ValueError(f"The specified model '{name}' does not exist!")

    return joblib.load(MODEL_PATH)


def get_multiple_models(names: List[str]) -> Dict[str, Any]:
    """
    Get the model objects from the specified directory in the config file.

    Args:
        names (List[str]): List of the model names to get.

    Returns:
        out: Dict[str, Any]: Dictionary of the multiple models. The `key` is the model name and `value` is the model object.
    """
    # For every model get evaluation
    loaded_models: Dict[str, Any] = {
        model_name: get_model(model_name) for model_name in names
    }

    return loaded_models


def save_model(model, name: str) -> None:
    """
    Saves the trained model to the specified directory in the config file.

    Args:
        name (str): Name of the model you want to save (including .pkl suffix).
    """

    MODEL_PATH = os.path.join(settings.trained_models_dir, f"{name}")

    joblib.dump(model, MODEL_PATH)


def get_experiment_model(name: str) -> None:
    """
    Get the model object from the 'experimental_models' subdirectory of the directory specified in the config file.

    Args:
        name (str): Name of the model (including .pkl suffix).

    Raises:
        ValueError: If the model does not exist, raises an error.

    Returns:
        out: Any: the trained model object.
    """
    EXPERIMENTAL_MODELS_DIR = os.path.abspath(
        os.path.join(settings.trained_models_dir, "experimental_models")
    )

    MODEL_PATH = os.path.join(EXPERIMENTAL_MODELS_DIR, f"{name}")

    if not os.path.isfile(MODEL_PATH):
        raise ValueError(f"The specified model '{name}' does not exist!")

    return joblib.load(MODEL_PATH)


def save_experiment_model(model, name: str) -> None:
    """
    Saves the trained model to the 'experimental_models' subdirectory of the directory specified in the config file.

    Args:
        name (str): Name of the model you want to save (including .pkl suffix).
    """

    EXPERIMENTAL_MODELS_DIR = os.path.abspath(
        os.path.join(settings.trained_models_dir, "experimental_models")
    )

    MODEL_PATH = os.path.join(EXPERIMENTAL_MODELS_DIR, f"{name}")

    # If the experimental directory does not exist, create it
    if not os.path.isdir(EXPERIMENTAL_MODELS_DIR):
        os.makedirs(EXPERIMENTAL_MODELS_DIR)

    joblib.dump(model, MODEL_PATH)
