# Standard library imports
import pandas as pd
import logging
from typing import Dict, Union, List

# Custom library imports
from src.utils.log import setup_logging
from src.utils.save_model import save_model, get_model

from src.base import LSTMHyperparameters
from src.evaluation import EvaluateModel

from src.local_model.model import BaseLSTM
from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.statistical_models import LocalARIMA

from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

# Get logger
logger = logging.getLogger("local_model")


# TODO: add hyperparameters or something...
class PureEnsembleModel:

    def __init__(
        self, feature_models: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]]
    ):
        # Get features
        self.FEATURES: List[str] = list(feature_models.keys())

        # Targets are the same
        self.TARGETS: List[str] = self.FEATURES

        self.model: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]] = (
            feature_models
        )

    def predict(
        self, input_data: pd.DataFrame, last_year: int, target_year: int
    ) -> pd.DataFrame:

        # Preprocess data
        years_to_predict = range(last_year + 1, target_year + 1)

        timestep_predictions: pd.DataFrame | None = None

        # Predict each feature for every year
        for feature in self.FEATURES:

            current_model = self.model[feature]
            # By type
            if isinstance(current_model, LocalARIMA):
                feature_prediction_df = current_model.predict(
                    data=input_data, steps=len(years_to_predict)
                )

            if isinstance(current_model, BaseLSTM) or isinstance(
                current_model, FineTunableLSTM
            ):
                # Predict featue for X years
                feature_prediction_df = current_model.predict(
                    input_data=input_data,
                    last_year=last_year,
                    target_year=target_year,
                )

            # Rows - years
            # Columns -> features
            if timestep_predictions is None:
                timestep_predictions = feature_prediction_df
            else:
                # Concat by columns
                timestep_predictions = pd.concat(
                    [timestep_predictions, feature_prediction_df], axis=1
                )

        # return prediction_df
        return timestep_predictions


def train_models_for_ensemble_model(
    features: List[str],
    hyperaparameters: LSTMHyperparameters,
    split_rate: float = 0.8,
    display_nth_epoch: int = 10,
) -> Dict[str, Union[BaseLSTM, FineTunableLSTM]]:

    # Load data for training
    states_loader = StatesDataLoader()
    states_data_dict = states_loader.load_all_states()

    # Ensure the input / output size will be 1
    ADJUSTED_PARAMS = hyperaparameters
    ADJUSTED_PARAMS.input_size = 1  # Predict 1 target at the time

    # Split data
    train_dict, _ = states_loader.split_data(
        states_dict=states_data_dict,
        sequence_len=ADJUSTED_PARAMS.sequence_length,
        split_rate=split_rate,
    )

    # Train and save models
    trained_models: Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]] = {}
    for feature in features:

        logger.info(f"Training for predicting feature: {feature}")

        # Set feature as a target
        target = feature

        # Preprocess data for feature
        input_batches, target_batches, scaler = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=train_dict,
                hyperparameters=ADJUSTED_PARAMS,
                features=[target],
            )
        )

        # Create RNN
        rnn = BaseLSTM(hyperparameters=ADJUSTED_PARAMS, features=[target])

        # Set scaler
        rnn.set_scaler(scaler=scaler)

        # Train model
        rnn.train_model(
            batch_inputs=input_batches,
            batch_targets=target_batches,
            display_nth_epoch=display_nth_epoch,
        )

        trained_models[target] = rnn

    return trained_models


def train_arima_models_for_ensemble_model(
    features: List[str], state: str, split_rate: float = 0.8
) -> Dict[str, LocalARIMA]:

    # Load state data
    state_loader = StateDataLoader(state=state)
    state_data = state_loader.load_data()

    # Split data
    train_df, _ = state_loader.split_data(data=state_data, split_rate=split_rate)

    # Train and save models
    trained_models: Dict[str, LocalARIMA] = {}
    for feature in features:

        # Create ARIMA
        arima = LocalARIMA(p=1, d=1, q=1, features=[], target=feature, index="year")

        # Train model
        arima.train_model(data=train_df)

        # Save trained model
        trained_models[feature] = arima

    return trained_models


def train_model(sequence_length: int) -> PureEnsembleModel:
    MODEL_NAME: str = "ensemble_model.pkl"

    try:
        return get_model(name=MODEL_NAME)
    except Exception as e:
        logger.warning(f"Excpetion: {str(e)}")
        logger.warning(f"Training new ensemble model: {MODEL_NAME}")

    FEATURES: List = [
        col.lower()
        for col in [
            # "year",
            "Fertility rate, total",
            # "Population, total",
            # "Net migration",
            "Arable land",
            "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            "Rural population",
            "Rural population growth",
            "Age dependency ratio",
            "Urban population",
            "Population growth",
            "Adolescent fertility rate",
            "Life expectancy at birth, total",
        ]
    ]

    HYPERPARAMETERS = LSTMHyperparameters(
        input_size=1,
        hidden_size=256,
        sequence_length=sequence_length,
        learning_rate=0.0001,
        epochs=10,
        batch_size=16,
        num_layers=3,
        bidirectional=False,
    )

    trained_models = train_models_for_ensemble_model(
        features=FEATURES, hyperaparameters=HYPERPARAMETERS
    )

    # Create pure ensemble model
    em = PureEnsembleModel(feature_models=trained_models)

    # Save model
    save_model(model=em, name=MODEL_NAME)

    # Load states data
    state_loader = StateDataLoader(state="Finland")
    state_data = state_loader.load_data()

    last_year = (
        state_data.sort_values(by="year", ascending=True)["year"].iloc[-1].item()
    )

    # Try to predict something
    pred_df = em.predict(input_data=state_data, last_year=last_year, target_year=2035)

    # Test print
    print(pred_df)
    print(f"\nShape of the df: {pred_df.shape}")

    return em


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    SEQUENCE_LENGTH = 10
    # TODO:
    # 1. evaluate predictions           DONE
    # 2. try using some arima models and evalute (train_arima_models_for_ensemble_model)...
    # 3. Support creating ensemble model using dict with types or predefined models to train
    em = train_model(sequence_length=SEQUENCE_LENGTH)

    # Eval model
    # Load states data

    evaluation_states: List[str] = ["Czechia", "Finland", "Croatia", "United States"]

    # state_loader = StateDataLoader(state="Finland")
    # state_data = state_loader.load_data()

    states_loader = StatesDataLoader()
    eval_states_data = states_loader.load_states(states=evaluation_states)

    state_train_data_dict, state_test_data_dict = states_loader.split_data(
        states_dict=eval_states_data, sequence_len=SEQUENCE_LENGTH
    )
    # train_df, test_df = state_loader.split_data(data=state_data)

    em_evaluation = EvaluateModel(model=em)
    # em_evaluation.eval(test_X=train_df, test_y=test_df)
    em_evaluation.eval_for_every_state(
        X_test_states=state_train_data_dict, y_test_states=state_test_data_dict
    )

    print(em_evaluation.overall_metrics)
    print(em_evaluation.per_target_metrics)
    print(em_evaluation.all_states_evaluation)

    # Dict[str, Union[LocalARIMA, BaseLSTM, FineTunableLSTM]] = {
    #     "feature": BaseLSTM(hyperaparameters=LSTMHyperparameters(...), features=["feature"])
    # }


#     [{'best_model': 'BaseLSTM', 'feature': 'fertility rate, total'},
#  {'best_model': 'BaseLSTM', 'feature': 'population, total'},
#  {'best_model': 'ARIMA', 'feature': 'net migration'},
#  {'best_model': 'BaseLSTM', 'feature': 'arable land'},
#  {'best_model': 'BaseLSTM', 'feature': 'birth rate, crude'},
#  {'best_model': 'ARIMA', 'feature': 'gdp growth'},
#  {'best_model': 'ARIMA', 'feature': 'death rate, crude'},
#  {'best_model': 'BaseLSTM', 'feature': 'agricultural land'},
#  {'best_model': 'ARIMA', 'feature': 'rural population'},
#  {'best_model': 'ARIMA', 'feature': 'rural population growth'},
#  {'best_model': 'BaseLSTM', 'feature': 'age dependency ratio'},
#  {'best_model': 'ARIMA', 'feature': 'urban population'},
#  {'best_model': 'BaseLSTM', 'feature': 'population growth'},
#  {'best_model': 'BaseLSTM', 'feature': 'adolescent fertility rate'},
#  {'best_model': 'BaseLSTM', 'feature': 'life expectancy at birth, total'}]
