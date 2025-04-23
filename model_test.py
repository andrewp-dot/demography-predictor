# Standard libreary imports
import pandas as pd

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_model

from src.evaluation import EvaluateModel

from src.local_model.model import BaseRNN


from src.pipeline import LocalModelPipeline


from feature_explainer.explainers import LSTMExplainer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def main():
    # Setup logging
    setup_logging()

    # Get data
    loader = StatesDataLoader()

    # all_states_dict = loader.load_all_states()

    all_states_dict = loader.load_states(
        states=["Czechia", "United States", "New Zealand"]
    )

    # Get model
    transformer = get_model("put your model transformer here")
    model: BaseRNN = get_model("put your model here")

    # Split data
    X_states_dict, y_states_dict = loader.split_data(
        states_dict=all_states_dict, sequence_len=model.hyperparameters.sequence_length
    )

    pipeline = LocalModelPipeline(model=model, transformer=transformer)

    # Explain
    explainer = LSTMExplainer(pipeline=pipeline)

    input_sequences = explainer.create_sequences(state="Czechia")
    shap_values = explainer.get_shap_values(input_sequences)
    feature_importance = explainer.get_feature_importance(shap_values, save_plot=True)

    print(feature_importance)

    model_evaluation = EvaluateModel(transformer=transformer, model=pipeline)

    all_states_evaluation_df: pd.DataFrame = model_evaluation.eval_for_every_state(
        X_test_states=X_states_dict, y_test_states=y_states_dict
    )

    # Sort this and save this to json
    print(all_states_evaluation_df)

    all_states_evaluation_df = all_states_evaluation_df.sort_values(
        by=["mae", "mse", "rmse", "r2"],
        ascending=[True, True, True, False],
    )

    with open("base_lstm_evaluation_pop_total.json", "w") as f:
        all_states_evaluation_df.to_json(f, indent=4, orient="records")

    # Get the overall performace (weighted single metric)
    overall_df = model_evaluation.eval_for_every_state_overall(
        X_test_states=X_states_dict, y_test_states=y_states_dict
    )

    print(overall_df)

    import matplotlib.pyplot as plt

    model_evaluation.plot_predictions()

    plt.savefig("neco_debilniho.png")


if __name__ == "__main__":
    main()
