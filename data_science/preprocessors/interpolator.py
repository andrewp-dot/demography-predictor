import os
import pandas as pd
from dataset_preprocessor import country_sequences_info

from config import Settings

settings = Settings()


class Interpolator:

    def __init__(self, dataset_name: str, dataset_dir: str):
        self.dataset_name: str = dataset_name
        self.dataset_dir: str = dataset_dir
        self.dataset_states_dir: str = os.path.join(dataset_dir, f"states")

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name))

    def get_states_with_incomplete_sequence(
        self,
        df: pd.DataFrame,
    ) -> dict[str, tuple[int, int, list]]:

        country_info_dict = country_sequences_info(df)

        states_with_missing_years = {
            key: value for key, value in country_info_dict.items() if value[2]
        }

        return states_with_missing_years

    def interpolate_missing(
        self,
        state: pd.DataFrame,
        country_sequence_info: tuple[int, int, list],
        max_missing: float = 0.3,
    ) -> pd.DataFrame:
        """Generate missing values using interpolation methods.

        Args:
            state (pd.DataFrame): dataframe to predict missing values
            max_missing (float, optional): percent of maximal part missing data. Defaults to 0.3, means max 30% of missing data.

        Raises:
            NotImplementedError: _description_

        Returns:
            pd.DataFrame: _description_
        """

        min_year, max_year, missing_years = country_sequence_info

        # check if you can interoplate missing values
        if (max_year - min_year) * max_missing < len(missing_years):
            print("Cannot interpolate data")
            return pd.DataFrame({})

        # if missing less than 30 percent of data, use ARIMA

        raise NotImplementedError()

    # TODO:
    # 1. extract states
    # 2a. check if the column is predictable (they are more then 70% of all values)
    # 2b. for every generatable column in every state interpolate missing values from last known min value to last known max value
    # 3. concat states
    # 4. return interpolated


if __name__ == "__main__":

    dataset_saved_dir = os.path.join(
        settings.save_dataset_path, "datasets", f"dataset_{settings.dataset_version}"
    )

    dataset_path = settings.save_dataset_path

    dataset_name = f"dataset_{settings.dataset_version}.csv"
    interpolator = Interpolator(
        dataset_name=dataset_name,
        dataset_dir=dataset_saved_dir,
    )

    data = interpolator.load_data()

    # devide dataset to state data
    state_data = []

    incomplete_states_dict = interpolator.get_states_with_incomplete_sequence(data)
    for state in list(incomplete_states_dict.keys()):
        state_df = data[data["Country Name"] == state]

        # generate missing values in columns to max of X % of missing values in column
        interpolated_state = interpolator.interpolate_missing(
            state_df, country_sequence_info=incomplete_states_dict[state]
        )

        # append interpolated state to state_data
        state_data.append(interpolated_state)

    # concat dataset back together
    interpolated_df = pd.concat(state_data, axis=0)

    print(f"Shape before interpolation (with missing values): {data.shape}")
    print(
        f"Shape with dropped all missing values and not completed sequences: {data.shape}"
    )
    print(f"Shape after interpolation: {interpolated_df.shape}")
