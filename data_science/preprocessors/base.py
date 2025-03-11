import os
import pandas as pd
from abc import abstractmethod

from sklearn.base import BaseEstimator


class BasePreprocessor:

    def __init__(self, to_save_data_dir: str):
        self._to_save_data_dir = to_save_data_dir

    def save_data(self, df: pd.DataFrame, name: str):
        df.to_csv(os.path.join(self._to_save_data_dir, name), index=False)

    @abstractmethod
    def plot_dataset(self) -> None:
        pass

    @abstractmethod
    def process(self, estimator: BaseEstimator) -> pd.DataFrame:
        pass
