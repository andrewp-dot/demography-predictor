import pandas as pd
import logging
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config import Config, setup_logging
from src.preprocessors.state_preprocessing import StateDataLoader


settings = Config()

logger = logging.getLogger("data_preprocessing")

# TODO: maybe you can use generators from load functions


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
            Dict[str, pd.DataFrame]: The key is the state name and the value is state dataframe.
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
            Dict[str, pd.DataFrame]: The key is the state name and the value is state dataframe.
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
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]: The 'state_name' is the key. Dict[state_name] = (train_data, test_data)
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
            state_split_dict[state_name] = tuple(state_train_df, state_test_df)

        return state_split_dict

    # TODO:
    # Test scaling function
    def scale_data(
        self,
        states_data: pd.DataFrame,
        scaler: MinMaxScaler | RobustScaler | StandardScaler,
    ) -> Tuple[pd.DataFrame, MinMaxScaler | RobustScaler | StandardScaler]:
        """
        Scales data for in the dafarame.

        Args:
            states_data (pd.DataFrame): Data to scale.
            scaler (MinMaxScaler | RobustScaler | StandardScaler): Scaler which is used for data scaling.

        Returns:
            Tuple[pd.DataFrame, MinMaxScaler | RobustScaler | StandardScaler]: Scaled data, fitted scaler.
        """
        # Scale data
        scaled_data = scaler.fit_transform(states_data)

        # Create pandas dataframe from ndarray
        scaled_data_df = pd.DataFrame(scaled_data, columns=states_data.columns)

        return scaled_data_df, scaler


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
        assert False
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
