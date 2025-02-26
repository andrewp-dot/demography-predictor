import pandas as pd
import logging
from pydantic import BaseModel
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import Config, setup_logging
from src.preprocessors.state_preprocessing import StateDataLoader


settings = Config()

logger = logging.getLogger("data_preprocessing")


# TODO: maybe you can use generators from load functions
# TODO: maybe use this instead of Dict
# class SingleStateData(BaseModel):
#     name: str
#     data_df: pd.DataFrame
#     train_data_df: Optional[pd.DataFrame] = None  # Set the default to None
#     test_data_df: Optional[pd.DataFrame] = None  # Set the default to None


class StatesDataLoader:

    def __init__(self) -> None:
        """
        Creates and initializes instace of StatesDataLoader.
        """

        # Read whole dataset
        self.data = pd.read_csv(settings.dataset_path)

        # TODO: remove this after fixing the dataset preprocessors
        # Temporarely rename columns in dataset to lower columns
        mapper = {col: col.lower() for col in self.data.columns}
        self.data.rename(columns=mapper, inplace=True)

        self.__state_loaders: Dict[str, StateDataLoader] = {}

        # Get state loaders from all states
        all_states = self.data["country name"].unique()
        for state in all_states:

            # Create and save loader
            self.__state_loaders[state] = StateDataLoader(state)

    def load_states(self, states: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Loads the states from the given list

        Args:
            states (List[str]): List of state names which should be loaded to pandas dataframe.

        Returns:
            out: (Dict[str, pd.DataFrame]): The key is the state name and the value is state dataframe.
        """

        # Get loader for each state
        state_dfs: Dict[str, pd.DataFrame] = {}
        for state in states:
            try:
                state_dfs[state] = self.__state_loaders[state].load_data()
            except KeyError:
                logger.error(f"State '{state}' not found in the dataset.")

        # Return loaded states
        return state_dfs

    def load_all_states(self) -> Dict[str, pd.DataFrame]:
        """
        Loads data from all states.

        Returns:
            out (Dict[str, pd.DataFrame]): The key is the state name and the value is state dataframe.
        """

        # Get loader for each state
        state_dfs: Dict[str, pd.DataFrame] = {}
        for state, loader in self.__state_loaders.items():
            state_dfs[state] = loader.load_data()

        # Return loaded states
        return state_dfs

    # TODO:
    # TEST THIS
    def split_data(
        self,
        states_dict: Dict[str, pd.DataFrame],
        sequence_len: int,
        split_rate: float = 0.8,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits each dataframe to train and and test set. States with records * (1 - test_size) is lower then sequence len + 1 (length of the target sequence) are excluded.

        Train data has to have length minimum of sequence len + 1 (to create target prediction sequence).

        Args:
            states_dict (Dict[str, pd.DataFrame]): Dictionary of loaded state dataframes. The dictionary key should be the state name.
            sequence_len (int): The sequence length of the model. Need in order to check whether train and corresping target sequence can be created.
            split_rate (float, optional): Percentage value of the training data.. Defaults to 0.8.

        Returns:
            out (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): The 'state_name' is the key. Dict[state_name] = (train_data, test_data)
        """

        # Initialize state split dict
        state_split_dict: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

        # Split each state
        for state_name, loader in self.__state_loaders.items():

            # Get train data and test data
            state_train_df, state_test_df = loader.split_data(
                states_dict[state_name], split_rate=split_rate
            )

            # Check if training data have enough records
            if len(state_train_df) <= sequence_len:
                logger.warning(
                    f"The training data of the state '{state_name}' does not have enough records to create training and target sequence. Therefore this state will be excluded from the training."
                )
                continue

            # Save the state train and test data
            state_split_dict[state_name] = tuple([state_train_df, state_test_df])

        return state_split_dict

    # TODO:
    # Test scaling function
    def scale_data(
        self,
        states_data: Dict[str, pd.DataFrame],
        scaler: MinMaxScaler | RobustScaler | StandardScaler,
    ) -> Tuple[Dict[str, pd.DataFrame], MinMaxScaler | RobustScaler | StandardScaler]:
        """
        Scales data for in the dafarame. Scales only numerical data.

        Args:
            states_data (Dict[str, pd.DataFrame]): States data to scale. The dictionary key should be the name of the state
            scaler (MinMaxScaler | RobustScaler | StandardScaler): Scaler which is used for data scaling.

        Returns:
            out (Tuple[Dict[str, pd.DataFrame], MinMaxScaler | RobustScaler | StandardScaler]): Each state data scaled dict, fitted scaler.
        """

        # Merge data by rows
        states_merged_df = pd.concat(list(states_data.values()), axis=0)

        # Get only numerical columns
        numerical_features_df = states_merged_df.select_dtypes(include=["number"])

        logger.debug(f"Numerical: {numerical_features_df.columns}")

        # Categorical columns
        categorical_features_df = states_merged_df.select_dtypes(exclude=["number"])
        logger.debug(f"Categorical: {categorical_features_df.columns}")

        # Scale data
        merged_scaled_data = scaler.fit_transform(numerical_features_df)

        # Create pandas dataframe from ndarray
        merged_scaled_data_df = pd.DataFrame(
            merged_scaled_data,
            columns=numerical_features_df.columns,
            index=numerical_features_df.index,
        )

        logger.debug(
            f"Merged scaled dataframe: {merged_scaled_data_df.columns} {merged_scaled_data_df.shape}"
        )

        # Get the datframe with numerical data scaled
        merged_scaled_data_df = pd.concat(
            [categorical_features_df, merged_scaled_data_df], axis=1
        )

        logger.critical(
            f"Concatenated merged scaled dataframe: {merged_scaled_data_df.columns} {merged_scaled_data_df.shape}"
        )

        # Split to separate states
        scaled_states: Dict[str, pd.DataFrame] = {}

        for state_name in states_data.keys():

            # Save scaled datframe by state
            scaled_states[state_name] = merged_scaled_data_df.loc[
                merged_scaled_data_df["country name"] == state_name
            ]

        return scaled_states, scaler


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Try the state loader
    states_loader = StatesDataLoader()

    # Try load some of the states
    test_states = ["Afghanistan", "Czechia", "United States", "United Arab Emirates"]
    try:
        test_states_dict = states_loader.load_states(states=test_states)

        got_states = list(test_states_dict.keys())
        logger.info(f"{test_states} == {got_states} (Desired states == Got states)")

        # Check it does equal
        assert got_states == test_states
    except Exception as e:
        logger.error(
            f"Exception occured while loading states: {test_states}. Exeption: {e}"
        )

    # Try load all states
    try:
        all_states_dict = states_loader.load_all_states()

        # Compare the number of all states in the dataset and loaded
        ref_number_all_states = states_loader.data["country name"].nunique()

        got_number_of_states = len(all_states_dict.keys())

        logger.info(
            f"{ref_number_all_states} == {got_number_of_states}, (Reference number of states == Got number of states)"
        )

        # Check it does equal
        assert ref_number_all_states == got_number_of_states
    except Exception as e:
        logger.error(f"Exception occured while loading ALL states. Exeption: {e}")
        exit(1)

    # Split data function
    states_train_test_data_dict = states_loader.split_data(
        all_states_dict, sequence_len=5
    )

    train_data: Dict[str, pd.DataFrame] = {
        state: data[0] for state, data in states_train_test_data_dict.items()
    }

    logger.info(f"Train data: \n {train_data['Czechia'].head()}")

    # Scale states data
    scaled_train_data, scaler = states_loader.scale_data(
        train_data, scaler=MinMaxScaler()
    )

    logger.info(f"Train data: \n {scaled_train_data['Czechia'].head()}")
