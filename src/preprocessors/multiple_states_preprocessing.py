import pandas as pd
import logging
import torch

# from pydantic import BaseModel
from typing import Dict, List, Tuple

from config import Config
from src.utils.log import setup_logging
from src.preprocessors.state_preprocessing import StateDataLoader
from src.base import RNNHyperparameters


settings = Config()

logger = logging.getLogger("data_preprocessing")


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
        all_states = self.data["country_name"].unique()
        for state in all_states:

            # Create and save loader
            self.__state_loaders[state] = StateDataLoader(state)

    def load_states(
        self, states: List[str], exclude_covid_years: bool = False
    ) -> Dict[str, pd.DataFrame]:
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
                state_dfs[state] = self.__state_loaders[state].load_data(
                    exclude_covid_years=exclude_covid_years
                )
            except KeyError:
                logger.error(f"State '{state}' not found in the dataset.")

        # Return loaded states
        return state_dfs

    def parse_states(self, states_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create dictionary dataframe with multiple values of the 'country name' column. Inverse function to 'merge_states'.

        Args:
            states_df (pd.DataFrame): Input dataframe with the 'country name' column.

        Returns:
            out: Dict[str, pd.DataFrame]: Dictionary, where the key is the state name and the value is the state dataframe.
        """
        states_data_dict: Dict[str, pd.DataFrame] = {}
        for country_name, group_df in states_df.groupby("country_name"):
            states_data_dict[country_name] = group_df

        return states_data_dict

    def merge_states(self, state_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges given state dataframes from the dict. Useful when creating the single pandas dataframe from multiple loaded states.

        Args:
            state_dfs (Dict[str, pd.DataFrame]): Dictionary where state name is the key.

        Returns:
            out: d.DataFrame: Single dataframe with specified states data.
        """
        return pd.concat(state_dfs.values(), axis=0)

    def load_all_states(
        self, exclude_covid_years: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Loads data from all states.

        Returns:
            out (Dict[str, pd.DataFrame]): The key is the state name and the value is state dataframe.
        """

        # Get loader for each state
        state_dfs: Dict[str, pd.DataFrame] = {}
        for state, loader in self.__state_loaders.items():
            state_dfs[state] = loader.load_data(exclude_covid_years=exclude_covid_years)

        # Return loaded states
        return state_dfs

    def split_data(
        self,
        states_dict: Dict[str, pd.DataFrame],
        sequence_len: int,
        future_steps: int = 1,
        split_rate: float = 0.8,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Splits each dataframe to train and and test set. States with records * (1 - test_size) is lower then sequence len + 1 (length of the target sequence) are excluded.

        Train data has to have length minimum of sequence len + 1 (to create target prediction sequence).

        Args:
            states_dict (Dict[str, pd.DataFrame]): Dictionary of loaded state dataframes. The dictionary key should be the state name.
            sequence_len (int): The sequence length of the model. Need in order to check whether train and corresping target sequence can be created.
            split_rate (float, optional): Percentage value of the training data.. Defaults to 0.8.

        Returns:
            out (Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]): States train data dict, states test data dict. The 'state_name' is the key for both dicts.
        """

        # Initialize state split dict
        state_split_train_dict: Dict[str, pd.DataFrame] = {}
        state_split_test_dict: Dict[str, pd.DataFrame] = {}

        # Split each state
        excluded_states: List[str] = []
        for state_name in states_dict.keys():

            # Get train data and test data
            state_train_df, state_test_df = self.__state_loaders[state_name].split_data(
                states_dict[state_name], split_rate=split_rate
            )

            # Check if training data have enough records
            if len(state_train_df) < sequence_len + future_steps:
                excluded_states.append(f'"{state_name}"')
                continue

            # Save the state train and test data
            state_split_train_dict[state_name] = state_train_df
            state_split_test_dict[state_name] = state_test_df

        # Display warning of state exclusion from training if any
        if excluded_states:
            logger.warning(
                f"States excluded from training due to not having enough records to create training and target sequence. States excluded from the training: {', '.join(excluded_states)}"
            )

        return state_split_train_dict, state_split_test_dict


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
        ref_number_all_states = states_loader.data["country_name"].nunique()

        got_number_of_states = len(all_states_dict.keys())

        logger.info(
            f"{ref_number_all_states} == {got_number_of_states}, (Reference number of states == Got number of states)"
        )

        # Check it does equal
        assert ref_number_all_states == got_number_of_states
    except Exception as e:
        logger.error(f"Exception occured while loading ALL states. Exeption: {e}")
        exit(1)

    # Get features
    FEATURES = list(all_states_dict["Czechia"].columns)
    FEATURES.remove("country_name")

    hyperparameters = RNNHyperparameters(
        input_size=len(FEATURES),
        hidden_size=128,
        future_step_predict=2,
        sequence_length=5,
        num_layers=2,
        learning_rate=0.0001,
        epochs=10,
        batch_size=32,
    )

    # Split data function
    train_data_dict, test_data_dict = states_loader.split_data(
        all_states_dict,
        sequence_len=hyperparameters.sequence_length,
        future_steps=hyperparameters.future_step_predict,
    )

    logger.info(f"Train data: \n {train_data_dict['Czechia'].head()}")
