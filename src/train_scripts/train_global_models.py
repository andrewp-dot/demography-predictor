# Standard library imports
import pandas as pd
from typing import List, Dict
from xgboost import XGBRegressor

# from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Custom imports
from src.utils.log import setup_logging
from src.preprocessors.data_transformer import DataTransformer
from src.preprocessors.multiple_states_preprocessing import StatesDataLoader

from src.pipeline import GlobalModelPipeline
from src.global_model.model import GlobalModel, XGBoostTuneParams


def train_global_model(
    name: str,
    states_data: Dict[str, pd.DataFrame],
    features: List[str],
    targets: List[str],
    sequence_len: int,
    tune_parameters: XGBoostTuneParams,
    transfomer: DataTransformer | None = None,
    split_size: float = 0.8,
) -> GlobalModelPipeline:

    global_model = GlobalModel(
        model=XGBRegressor(objective="reg:squarederror", random_state=42),
        features=features,
        targets=targets,
        sequence_len=sequence_len,
        params=tune_parameters,
    )

    states_loader = StatesDataLoader()

    # Create train and test for XGB (Global Model)
    X_train, X_test, y_train, _ = global_model.create_train_test_timeseries(
        states_dfs=states_data,
        states_loader=states_loader,
        split_size=split_size,
    )

    # Scale the training data
    if transfomer is None:
        transfomer = DataTransformer()
        scaled_training_data, _ = transfomer.scale_and_fit(
            training_data=X_train,
            validation_data=X_test,
            features=features,
            # targets=targets, # Here are targets useless for tree based model due to transformation of history columns
        )
    else:
        scaled_training_data = transfomer.scale_data(data=X_train, features=features)
        # scaled_validation_data = transfomer.scale_data(data=X_test)

    # Train XGB
    global_model.train(
        X_train=scaled_training_data,
        y_train=y_train,
        # tune_hyperparams=True
    )

    # Create Pipeline
    pipeline = GlobalModelPipeline(
        name=name, model=global_model, transformer=transfomer
    )
    return pipeline


def main():
    # Load data
    states_loader = StatesDataLoader()
    state_dfs = states_loader.load_all_states()
    state_df_merged = states_loader.merge_states(state_dfs=state_dfs)

    FEATURES: List[str] = [col.lower() for col in [""]]
    TARGETS: List[str] = [""]

    global_model_pipeline = train_global_model(
        states_data=state_df_merged, features=FEATURES, targets=TARGETS
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Run main
    main()
