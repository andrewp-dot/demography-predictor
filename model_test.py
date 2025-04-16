# Standard libreary imports
import pandas as pd

# Custom imports
from src.utils.log import setup_logging
from src.utils.save_model import get_model

from src.evaluation import EvaluateModel

from src.local_model.model import BaseLSTM
from src.local_model.experimental import ExpLSTM

from src.pipeline import LocalModelPipeline

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader


def main():
    # Setup logging
    setup_logging()

    # Get data
    loader = StatesDataLoader()

    all_states_dict = loader.load_all_states()

    # all_states_dict = loader.load_states(
    #     states=["Czechia", "United States", "New Zealand"]
    # )

    # Get model
    transformer = get_model("ExpLSTM_transformer.pkl")
    model: ExpLSTM = get_model("ExpLSTM.pkl")

    # Split data
    X_states_dict, y_states_dict = loader.split_data(
        states_dict=all_states_dict, sequence_len=model.hyperparameters.sequence_length
    )

    pipeline = LocalModelPipeline(model=model, transformer=transformer)

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

    with open("base_lstm_evaluation.json", "w") as f:
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
