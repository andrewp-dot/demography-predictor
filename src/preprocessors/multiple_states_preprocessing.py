import pandas as pd
import logging
from typing import Dict, List

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


        :param states: List[str]: List of state names which should be loaded to pandas dataframe.
        :return: Dict[str, pd.DataFrame]: The key is the state name and the value is state dataframe.
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

        :return: Dict[str, pd.DataFrame]: The key is the state name and the value is state dataframe.
        """

        # Get loader for each state
        state_dfs: Dict[str, pd.DataFrame] = {}
        for state, loader in self.__state_loaders.items():
            state_dfs[state] = loader.load_data()

        # Return loaded states
        return state_dfs

    # TODO:
    # implement method for splitting data to training and target sequences
    # implement method for scaling data


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
