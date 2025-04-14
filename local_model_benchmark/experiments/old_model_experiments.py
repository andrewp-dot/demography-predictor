# Standard libraries
import os
import pandas as pd
import pprint
import logging
from typing import List, Dict, Literal, Tuple, Union

import copy
import optuna


from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_experiment_model, save_experiment_model

from local_model_benchmark.utils import pre_train_model_if_needed
from local_model_benchmark.config import get_core_parameters
from local_model_benchmark.experiments.base_experiment import BaseExperiment, Experiment

from src.local_model.finetunable_model import FineTunableLSTM
from src.local_model.statistical_models import LocalARIMA, EvaluateARIMA
from src.preprocessors.state_preprocessing import StateDataLoader
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.local_model.model import LSTMHyperparameters, BaseLSTM, EvaluateModel


# Setup logger
logger = logging.getLogger("benchmark")


# Model input based eperiments:
# 1. Compare performance of LSTM networks with different neurons in layers, try to find optimal (optimization algorithm?)
class OptimalParamsExperiment(BaseExperiment):

    # TODO: evaluation -> use more then r2 score?
    def __init__(
        self,
        model: BaseLSTM,
        name: str,
        description: str,
        features: List[str],
        hidden_size_range: Tuple[int, int],
        sequence_length_range: Tuple[int, int],
        num_layers_range: Tuple[int, int],
        learning_rate_range: Tuple[int, int],
    ):
        super().__init__(model, name, description, features)

        # Add ranges for parameters
        self.hidden_size_range: Tuple[int, int] = self.__min_max_tuple(
            hidden_size_range
        )
        self.sequence_length_range: Tuple[int, int] = self.__min_max_tuple(
            sequence_length_range
        )
        self.num_layers_range: Tuple[int, int] = self.__min_max_tuple(num_layers_range)
        self.learning_rate_range: Tuple[int, int] = self.__min_max_tuple(
            learning_rate_range
        )

    def __min_max_tuple(self, tup: Tuple[int, int]) -> Tuple[int, int]:
        """
        Adjust tuple to format (min, max). If the both numbers are equal,

        Args:
            tup (Tuple): _description_

        Returns:
            Tuple: _description_
        """
        # if the max value is first
        if tup[0] > tup[1]:
            return tup[1], tup[0]

        return tup

    def adjust_hidden_size(
        self, base_parameters: LSTMHyperparameters, hidden_size: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.hidden_size = hidden_size
        return new_params

    def adjust_sequence_len(
        self, base_parameters: LSTMHyperparameters, sequence_length: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.sequence_length = sequence_length
        return new_params

    def adjust_num_layers(
        self, base_parameters: LSTMHyperparameters, num_layers: int
    ) -> LSTMHyperparameters:

        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.num_layers = num_layers
        return new_params

    def adjust_learning_rate(
        self, base_parameters: LSTMHyperparameters, learning_rate: float
    ):
        # Create a copy of original parameters
        new_params = copy.deepcopy(base_parameters)
        new_params.learning_rate = learning_rate
        return new_params

    def find_optimal_hyperparams(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        base_params: LSTMHyperparameters,
        state_loader: StateDataLoader,
        features: List[str],
    ):

        def objective(trial: optuna.Trial):

            # Get ranges from settings
            hidden_size = trial.suggest_int(
                "hidden_size",
                self.hidden_size_range[0],
                self.hidden_size_range[1],
            )
            sequence_length = trial.suggest_int(
                "sequence_length",
                self.sequence_length_range[0],
                self.sequence_length_range[1],
            )
            num_layers = trial.suggest_int(
                "num_layers",
                self.num_layers_range[0],
                self.num_layers_range[1],
            )
            learning_rate = trial.suggest_float(
                "learning_rate",
                self.learning_rate_range[0],
                self.learning_rate_range[1],
            )

            # Set hyperparameters
            NEW_HYPERPARAMS = LSTMHyperparameters(
                input_size=base_params.input_size,
                # Set new
                hidden_size=hidden_size,
                sequence_length=sequence_length,
                learning_rate=learning_rate,
                num_layers=num_layers,
                # Get old
                epochs=base_params.epochs,
                batch_size=base_params.batch_size,
                bidirectional=base_params.bidirectional,
            )

            # Preprocess data
            train_batches, target_batches, state_scaler = (
                state_loader.preprocess_training_data_batches(
                    train_data_df=train_df,
                    hyperparameters=NEW_HYPERPARAMS,
                    features=features,
                    scaler=MinMaxScaler(),
                )
            )

            # Train model
            rnn = BaseLSTM(base_params, features=self.FEATURES)

            rnn.set_scaler(state_scaler)

            rnn.train_model(batch_inputs=train_batches, batch_targets=target_batches)

            # Evaluate model
            rnn_evaluation = EvaluateModel(rnn)
            rnn_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
            )

            # Return a score (maximize R²)
            return rnn_evaluation.overall_metrics.loc[
                rnn_evaluation.overall_metrics["metric"] == "r2", "value"
            ].values[0]

        # Run Bayesian Optimization
        study = optuna.create_study(direction="maximize")  # Maximize R² score
        study.optimize(objective, n_trials=20)

        # Best parameters
        print("Best parameters:", study.best_params)

        return study.best_params

    def run(self, state: str, split_rate: float):

        self.create_readme()

        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Drop country name
        state_df.drop(columns=["country name"], inplace=True)

        # Get features
        FEATURES = self.FEATURES

        # Split data
        train_data_df, test_data_df = state_loader.split_data(
            state_df, split_rate=split_rate
        )

        best_params_dict = self.find_optimal_hyperparams(
            train_df=train_data_df,
            test_df=test_data_df,
            base_params=self.model.hyperparameters,
            state_loader=state_loader,
            features=FEATURES,
        )

        # Train and evaluate base model
        base_train_batches, base_target_batches, base_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_data_df,
                hyperparameters=self.model.hyperparameters,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        self.model.set_scaler(base_scaler)

        self.model.train_model(
            batch_inputs=base_train_batches,
            batch_targets=base_target_batches,
            display_nth_epoch=2,
        )

        base_model_evaluation = EvaluateModel(self.model)
        base_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
        )

        # Plot and save base model plot
        base_fig = base_model_evaluation.plot_predictions()

        # Add params
        self.readme_add_params()

        self.readme_add_section(
            title="# Base model evaluation",
            text=f"Hyperparameters:\n```{str(self.model.hyperparameters)}```",
        )

        self.save_plot(fig_name="base_model_eval.png", figure=base_fig)
        self.readme_add_plot(
            plot_name="Base model predicted vs reference values",
            plot_description="Displays the performance for every feature predicted of the `Base Model`.",
            fig_name="base_model_eval.png",
        )

        # Train and evaluate parametricaly adjusted model
        # Rewrite the parameters to optimal
        OPTIMAL_PAREMETRS = self.adjust_hidden_size(
            self.model.hyperparameters, best_params_dict["hidden_size"]
        )
        OPTIMAL_PAREMETRS = self.adjust_sequence_len(
            OPTIMAL_PAREMETRS, best_params_dict["sequence_length"]
        )
        OPTIMAL_PAREMETRS = self.adjust_num_layers(
            OPTIMAL_PAREMETRS, best_params_dict["num_layers"]
        )
        OPTIMAL_PAREMETRS = self.adjust_learning_rate(
            OPTIMAL_PAREMETRS, best_params_dict["learning_rate"]
        )

        # Preprocess data
        optimal_train_batches, optimal_target_batches, optimal_scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_data_df,
                hyperparameters=OPTIMAL_PAREMETRS,
                features=FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Train and evaluate the adjusted parameters model
        optimal_model = BaseLSTM(hyperparameters=OPTIMAL_PAREMETRS, features=FEATURES)

        optimal_model.set_scaler(optimal_scaler)

        optimal_model.train_model(
            batch_inputs=optimal_train_batches,
            batch_targets=optimal_target_batches,
            display_nth_epoch=2,
        )

        optimal_model_evaluation = EvaluateModel(optimal_model)
        optimal_model_evaluation.eval(
            test_X=train_data_df,
            test_y=test_data_df,
        )

        # Plot and save base model plot
        optimal_fig = optimal_model_evaluation.plot_predictions()

        self.save_plot(fig_name="optimal_model_eval.png", figure=optimal_fig)

        # Write optimal model evaluation
        self.readme_add_section(
            title="# Optimal model evaluation",
            text="",
        )
        self.readme_add_params(custom_params=optimal_model.hyperparameters)

        self.readme_add_plot(
            plot_name="Optimal model predicted vs reference values",
            plot_description="Displays the performance for every feature predicted of the `Optimal Model`.",
            fig_name="optimal_model_eval.png",
        )

        # Save the results
        formatted_base_model_evaluation: str = pprint.pformat(
            base_model_evaluation.to_readable_dict()
        )
        formatted_optimal_model_evaluation: str = pprint.pformat(
            optimal_model_evaluation.to_readable_dict()
        )
        compare_models_by_metric: str = f"""
Base model:
{formatted_base_model_evaluation}

Optimal model:
{formatted_optimal_model_evaluation}
"""

        self.readme_add_section(
            title="# Compare metric results", text=compare_models_by_metric
        )


# 2. Compare model with statistical methods (ARIMA, GM)
# 2.1. VAR, SARIMA, ARIMA * 19?
class CompareLSTMARIMASingleFeatureExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float) -> None:
        # Create readme
        self.create_readme()

        # Adjust parameters for RNN and print them into the README.md
        BASE_HYPERPARAMS = get_core_parameters(input_size=len(self.FEATURES))

        ADJUSTED_PARAMS = copy.deepcopy(BASE_HYPERPARAMS)
        ADJUSTED_PARAMS.input_size = (
            1  # Set the input size to 1, beacause there is only 1 feature
        )

        self.readme_add_section(
            title="# LSTM model parameters",
            text=f"Hyperparameters:\n```{str(ADJUSTED_PARAMS)}```",
        )

        # Load data
        STATE = state
        state_loader = StateDataLoader(STATE)

        state_df = state_loader.load_data()

        # Split data
        train_df, test_df = state_loader.split_data(
            data=state_df, split_rate=split_rate
        )

        # Exclude "year" from features
        features = self.FEATURES
        features.remove("year")

        # feature: best_model dict
        feature_bestmodel_dict: Dict[str, str] = {}

        # For every features create ARIMA and LSTM model
        for feature in features:

            logger.info(f"[{self.name}]: ### Current target: {feature} ###")

            # Set feature as a target
            target = feature

            # Create new section in README.md
            self.readme_add_section(
                title=f"# LSTM & ARIMA Comparision: Feature: {target}",
                text=f"Comparision of LSTM and ARIMA model of predicting feature {target}. State: {state}",
            )

            # Create ARIMA
            arima = LocalARIMA(1, 1, 1, features=[], target=target, index="year")
            arima.train_model(train_df)

            # Evaluate ARIMA
            arima_evaluation = EvaluateARIMA(arima=arima)
            arima_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
                features=[],
                target=target,
                index="year",
            )

            # Save ARIMA evaluation
            arima_predictions_fig = arima_evaluation.plot_predictions()
            arima_fig_name = f"arima_evaluation_{target}.png".replace(" ", "_")

            self.save_plot(fig_name=arima_fig_name, figure=arima_predictions_fig)
            self.readme_add_plot(
                plot_name="Arima evaluation",
                plot_description="",
                fig_name=arima_fig_name,
            )

            # Create RNN
            rnn = BaseLSTM(hyperparameters=ADJUSTED_PARAMS, features=[target])

            # Preprocess data
            input_batches, target_batches, scaler = (
                state_loader.preprocess_training_data_batches(
                    train_data_df=train_df,
                    hyperparameters=ADJUSTED_PARAMS,
                    features=[target],
                    scaler=MinMaxScaler(),
                )
            )

            rnn.set_scaler(scaler=scaler)

            # Train model
            rnn.train_model(
                batch_inputs=input_batches,
                batch_targets=target_batches,
                display_nth_epoch=1,
            )

            # Evaluate model
            rnn_evaluation = EvaluateModel(model=rnn)

            # TODO: what to do with this?
            rnn_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
            )

            # Save LSTM evaluation
            rnn_fig = rnn_evaluation.plot_predictions()
            rnn_fig_name = f"lstm_evaluation_{target}.png".replace(" ", "_")

            self.save_plot(fig_name=rnn_fig_name, figure=rnn_fig)
            self.readme_add_plot(
                plot_name=f"RNN evaluation - {target}",
                plot_description="",
                fig_name=rnn_fig_name,
            )

            # Print metrics to file
            formatted_rnn_eval = pprint.pformat(rnn_evaluation.to_readable_dict())
            formatted_arima_eval = pprint.pformat(arima_evaluation.to_readable_dict())

            self.readme_add_section(
                title="### Overall metrics (ARIMA)",
                text=f"```\n{formatted_arima_eval}\n```\n",
            )
            self.readme_add_section(
                title="### Overall metrics (RNN)",
                text=f"```\n{formatted_rnn_eval}\n```\n",
            )

            # Compare
            if rnn_evaluation.is_new_better(new_model_evaluation=arima_evaluation):
                feature_bestmodel_dict[target] = "ARIMA"
            else:
                feature_bestmodel_dict[target] = str(
                    type(self.model).__name__
                )  # Model class name

        # Create comparatation dataframe
        comparation_df: pd.DataFrame = pd.DataFrame(
            {
                "feature": list(feature_bestmodel_dict.keys()),
                "best_model": list(feature_bestmodel_dict.values()),
            }
        )

        formatted_comparation_dict = pprint.pformat(
            comparation_df.to_dict(orient="records")
        )

        self.readme_add_section(
            title="# Feature comparation dict",
            text=f"```\n{formatted_comparation_dict}\n```\n",
        )


class CompareLSTMARIMAExperiment(BaseExperiment):

    def __train_lstm(self, split_rate: float):
        states_loader = StatesDataLoader()

        states_data_dict = states_loader.load_all_states()

        # Split data
        train_dict, test_dict = states_loader.split_data(
            states_dict=states_data_dict,
            sequence_len=self.model.hyperparameters.sequence_length,
            split_rate=split_rate,
        )

        # Preprocess data
        train_batches, target_batches, scaler = (
            states_loader.preprocess_train_data_batches(
                states_train_data_dict=train_dict,
                hyperparameters=self.model.hyperparameters,
                features=self.FEATURES,
            )
        )

        # Set scaler to the model
        self.model.set_scaler(scaler)

        # Train model
        self.model.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=1,
        )

    def __finetune_lstm(self, state: str, split_rate: float):
        state_loader = StateDataLoader(state=state)

        states_df = state_loader.load_data()

        # Split data
        train_df, test_df = state_loader.split_data(
            data=states_df,
            split_rate=split_rate,
        )

        # Preprocess data
        train_batches, target_batches, scaler = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_df,
                hyperparameters=self.model.hyperparameters,
                features=self.FEATURES,
                scaler=MinMaxScaler(),
            )
        )

        # Set scaler to the model
        self.model.set_scaler(scaler)

        # Train model
        self.model.train_model(
            batch_inputs=train_batches,
            batch_targets=target_batches,
            display_nth_epoch=1,
        )

    def __train_and_eval_ARIMA_models(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:

        arima_evaluation_dict: Dict[str, pd.DataFrame] = {}

        ARIMA_FEATURES = self.FEATURES
        ARIMA_FEATURES.remove("year")
        for feature in ARIMA_FEATURES:

            logger.info(f"[{self.name}]: Current target: {feature}")

            # Set feature as a target
            target = feature

            # Get the feature to predict
            train_df = train_df
            test_df = test_df

            # Create ARIMA
            arima = LocalARIMA(1, 1, 1, features=[], target=target, index="year")
            arima.train_model(train_df)

            # Evaluate ARIMA
            arima_evaluation = EvaluateARIMA(arima=arima)
            arima_evaluation.eval(
                test_X=train_df,
                test_y=test_df,
                features=[],
                target=target,
                index="year",
            )

            # Save overall evaluation
            arima_evaluation_dict[target] = arima_evaluation.overall_metrics

            # Save ARIMA evaluation
            arima_predictions_fig = arima_evaluation.plot_single_feautre_prediction(
                feature=target
            )
            arima_fig_name = f"arima_evaluation_{target}.png".replace(" ", "_")

            self.save_plot(fig_name=arima_fig_name, figure=arima_predictions_fig)
            self.readme_add_plot(
                plot_name=f"Arima evaluation: {target}",
                plot_description="",
                fig_name=arima_fig_name,
            )

        return arima_evaluation_dict

    def __compare_results_by_error_metrics(
        self, lstm_evaluation_df: pd.DataFrame, arima_evaluation_df: pd.DataFrame
    ) -> pd.DataFrame:

        # Define error metrics
        ERROR_METRICS = ["mae", "mse", "rmse"]

        # Compare the same features
        lstm_features = lstm_evaluation_df["feature"].unique()
        arima_features = arima_evaluation_df["feature"].unique()

        COMMON_FEATURES = list(set(lstm_features) & set(arima_features))

        # feature: best model
        result_dict: Dict[str, str] = {}

        for feature in COMMON_FEATURES:

            # Extracts the records in this format: {'feature': value, 'mae': value, 'mse': value, 'rmse': value}
            lstm_feature_dict = (
                lstm_evaluation_df.loc[lstm_evaluation_df["feature"] == feature]
                .squeeze()
                .to_dict()
            )

            arima_feature_dict = (
                arima_evaluation_df.loc[arima_evaluation_df["feature"] == feature]
                .squeeze()
                .to_dict()
            )

            MODEL_KEY = str(type(self.model).__name__)
            votes_dict: Dict[str, str] = {
                MODEL_KEY: 0,
                "ARIMA": 0,
            }

            # Compare by metrics
            for metric in ERROR_METRICS:
                if lstm_feature_dict[metric] < arima_feature_dict[metric]:
                    votes_dict[MODEL_KEY] += 1
                if lstm_feature_dict[metric] > arima_feature_dict[metric]:
                    votes_dict["ARIMA"] += 1

            if votes_dict[MODEL_KEY] > votes_dict["ARIMA"]:
                result_dict[feature] = MODEL_KEY
            elif votes_dict[MODEL_KEY] < votes_dict["ARIMA"]:
                result_dict[feature] = "ARIMA"
            else:
                result_dict[feature] = ",".join(votes_dict.keys())

        # Convert result dict to dataframe
        to_result_df_dict: Dict[str, List[str]] = {
            "feature": list(result_dict.keys()),
            "best_model": list(result_dict.values()),
        }

        return pd.DataFrame(to_result_df_dict)

    def run(self, state: str, split_rate: float) -> None:
        # Create readme
        self.create_readme()

        # Add features add LSTM hyperparametes
        self.readme_add_features()
        self.readme_add_params()

        # Train the lstm
        if isinstance(self.model, FineTunableLSTM):
            self.__finetune_lstm(state=state, split_rate=split_rate)
        else:
            self.__train_lstm(split_rate=split_rate)

        # Preprocess data for ARIMA and LSTM evaluation
        state_loader = StateDataLoader(state=state)
        state_df = state_loader.load_data()

        arima_train_df, arima_test_df = state_loader.split_data(
            data=state_df, split_rate=split_rate
        )

        # Evaluate model
        lstm_evaluation = EvaluateModel(model=self.model)
        lstm_evaluation.eval(test_X=arima_train_df, test_y=arima_test_df)

        # Train ARIMA models
        arima_evaluation_dict = self.__train_and_eval_ARIMA_models(
            train_df=arima_train_df, test_df=arima_test_df
        )

        # Convert ARIMA evaluation dict to the similiar per feature dataframe
        arima_comparable_df = pd.DataFrame(
            columns=lstm_evaluation.per_target_metrics.columns
        )

        for feature, df in arima_evaluation_dict.items():

            # Create new row data frame
            new_row_dict = {
                "feature": feature,
                "mae": lstm_evaluation.get_metric_value(df, "mae"),
                "mse": lstm_evaluation.get_metric_value(df, "mse"),
                "rmse": lstm_evaluation.get_metric_value(df, "rmse"),
            }

            arima_comparable_df.loc[len(arima_comparable_df)] = new_row_dict

        # Write the metrics to the readme
        lstm_comaparable_df = lstm_evaluation.per_target_metrics

        # Compare results
        reusults = self.__compare_results_by_error_metrics(
            lstm_evaluation_df=lstm_comaparable_df,
            arima_evaluation_df=arima_comparable_df,
        )

        lstm_pretty_dict = pprint.pformat(lstm_comaparable_df.to_dict(orient="records"))
        arima_pretty_dict = pprint.pformat(
            arima_comparable_df.to_dict(orient="records")
        )

        # Write the metrics to the readme
        self.readme_add_section(
            title="## LSTM per feature dict metrics",
            text=f"\n```\n{lstm_pretty_dict}\n```\n\n",
        )

        self.readme_add_section(
            title="## ARIMA per feature dict metrics",
            text=f"\n```\n{arima_pretty_dict}\n```\n\n",
        )

        # Write the comparision resuls
        results_pretty_dict = pprint.pformat(reusults.to_dict(orient="records"))
        self.readme_add_section(
            title="# LSTM ARIMA comparision results",
            text=f"\n```\n{results_pretty_dict}\n```\n\n",
        )


# 3. Finetune model for the given state
class FineTuneExperiment(BaseExperiment):

    def run(self, state: str, split_rate: float):
        # Create readme
        self.create_readme()
        self.readme_add_features()

        EVLAUATION_STATE_NAME = state

        # Create and train base model
        base_model: BaseLSTM = self.model.base_model

        # Save the model params
        self.readme_add_section(title="# Base model parameters", text="")
        self.readme_add_params(custom_params=base_model.hyperparameters)

        # Save base model
        save_experiment_model(
            base_model,
            f"base_model_{base_model.hyperparameters.hidden_size}.pkl",
        )

        # Write the finetunable model hyperparameters
        self.readme_add_section(title="# Finetunable model parameters", text="")
        self.readme_add_params()

        # Load data
        state_loader = StateDataLoader(EVLAUATION_STATE_NAME)
        state_data_df = state_loader.load_data()

        # Split state data
        train_state_data_df, test_state_data_df = state_loader.split_data(
            data=state_data_df, split_rate=split_rate
        )

        # Use the same scaler as the base model
        trainig_batches, target_batches, _ = (
            state_loader.preprocess_training_data_batches(
                train_data_df=train_state_data_df,
                hyperparameters=self.model.hyperparameters,
                features=self.FEATURES,
                scaler=base_model.scaler,
            )
        )

        logger.info("Finetuning model...")
        self.model.train_model(
            batch_inputs=trainig_batches,
            batch_targets=target_batches,
            display_nth_epoch=1,
        )

        # Evaluate finetunable model
        logger.info("Evaluating finetunable model...")
        finetunable_model_evaluation = EvaluateModel(self.model)
        finetunable_model_evaluation.eval(
            test_X=train_state_data_df,
            test_y=test_state_data_df,
        )

        logger.info(
            f"[FinetunabelModel]: Overall metrics:\n{finetunable_model_evaluation.overall_metrics}\n"
        )
        logger.info(
            f"[FinetunabelModel]: Per feature metrics:\n{finetunable_model_evaluation.per_target_metrics}\n"
        )

        # Save the predictions
        fig = finetunable_model_evaluation.plot_predictions()

        fig_name = f"finetuned_model_{EVLAUATION_STATE_NAME}_predictions.png"
        self.save_plot(fig_name=fig_name, figure=fig)
        self.readme_add_plot(
            plot_name=f"Finetuned model predictions - {EVLAUATION_STATE_NAME}",
            plot_description="Finetuned model predictions.",
            fig_name=fig_name,
        )

        # Save model
        save_experiment_model(
            self.model, f"finetunable_model_{EVLAUATION_STATE_NAME}.pkl"
        )


class LSTMOptimalParameters(Experiment):

    NAME = "OptimalParamsExperiment"
    DESCRIPTION = (
        "The goal is to find the optimal parameters for the given BaseLSTM model."
    )

    def __init__(self):

        self.name = self.NAME

        self.FEATURES = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
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

        hyperparameters = get_core_parameters(input_size=len(self.FEATURES))

        self.model = BaseLSTM(hyperparameters=hyperparameters, features=self.FEATURES)

        self.exp = OptimalParamsExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
            hidden_size_range=(128, 2048),
            sequence_length_range=(10, 15),
            num_layers_range=(1, 5),
            learning_rate_range=(1e-5, 1e-2),
        )

    def run(self, state="Czechia", split_rate=0.8):
        return self.exp.run(state=state, split_rate=split_rate)


class RNNvsStatisticalMethodsSingleFeature(Experiment):
    NAME = "RNNvsStatisticalMethodsSingleFeature"
    DESCRIPTION = "Uses just 1 states data. Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. Each model is trained just to predict 1 feauture from all features."

    def __init__(self):

        self.name = self.NAME

        self.FEATURES = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
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

        hyperparameters = get_core_parameters(input_size=len(self.FEATURES))

        self.model = BaseLSTM(hyperparameters=hyperparameters, features=self.FEATURES)

        self.exp = CompareLSTMARIMASingleFeatureExperiment(
            model=self.model,
            name=self.NAME,
            description=self.DESCRIPTION,
            features=self.FEATURES,
        )

    def run(self, state="Czechia", split_rate=0.8):
        return self.exp.run(state=state, split_rate=split_rate)


class RNNvsStatisticalMethods(Experiment):

    NAME = "RNNvsStatisticalMethods"
    DESCRIPTION = "Compares the performance of LSTM model with the statistical ARIMA model for all features prediction. ARIMA model is trained just to predict 1 feauture from all features and LSTM predict all features at once."

    def __init__(self):

        name = self.NAME

        features = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
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

        hyperparameters = get_core_parameters(input_size=len(features), num_layers=2)

        model = BaseLSTM(hyperparameters=hyperparameters, features=features)

        model = self.get_finetunable_model(
            hyperparameters=hyperparameters, features=features
        )

        experiment = CompareLSTMARIMAExperiment(
            model=model,
            name=name,
            description=self.DESCRIPTION,
            features=features,
        )

        super().__init__(name, model, features, hyperparameters, experiment)

    def get_finetunable_model(
        self, hyperparameters: LSTMHyperparameters, features: List[str]
    ):

        # Create finetunable model
        finetunable_hyperparameters = LSTMHyperparameters(
            input_size=len(features),
            hidden_size=256,
            sequence_length=hyperparameters.sequence_length,
            learning_rate=0.0001,
            epochs=30,
            batch_size=1,
            num_layers=2,
        )

        model = FineTunableLSTM(
            base_model=BaseLSTM(hyperparameters=hyperparameters, features=features),
            hyperparameters=finetunable_hyperparameters,
        )

        return model

    def run(self, state: str = "Czechia", split_rate: float = 0.8) -> None:

        self.model = pre_train_model_if_needed(
            model=self.model, state=state, save_model=True
        )
        return self.exp.run(state=state, split_rate=split_rate)


class FineTuneLSTMExp(Experiment):

    NAME = "FineTuneLSTMExp"
    DESCRIPTION = "Finetunes the base LSTM model."

    def __init__(self):

        name = self.NAME

        features = [
            col.lower()
            for col in [
                "year",
                "Fertility rate, total",
                "Population, total",
                "Net migration",
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

        # Train the base model
        base_hyperparameters = get_core_parameters(input_size=len(features))

        # Create finetunable model
        finetunable_hyperparameters = LSTMHyperparameters(
            input_size=len(features),
            hidden_size=256,
            sequence_length=base_hyperparameters.sequence_length,
            learning_rate=0.0001,
            epochs=30,
            batch_size=1,
            num_layers=1,
        )

        model = FineTunableLSTM(
            base_model=BaseLSTM(
                hyperparameters=base_hyperparameters, features=features
            ),
            hyperparameters=finetunable_hyperparameters,
        )

        exp = FineTuneExperiment(
            model=model,
            name=name,
            description=self.DESCRIPTION,
            features=features,
        )

        super().__init__(
            name=name,
            model=model,
            features=features,
            hyperparameters=finetunable_hyperparameters,
            experiment=exp,
        )

    def run(self, state="Czechia", split_rate=0.8):

        self.model = pre_train_model_if_needed(
            model=self.model, state=state, save_model=True
        )

        return self.exp.run(state=state, split_rate=split_rate)


def run_experiments():
    exp1 = LSTMOptimalParameters()
    exp1.run()

    exp2 = RNNvsStatisticalMethodsSingleFeature()
    exp2.run()

    exp3 = FineTuneLSTMExp()
    exp3.run()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run experiments
    run_experiments()

    # Run
    # exp.run(state="Czechia", split_rate=0.8)
