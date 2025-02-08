from config import Config

import pandas as pd
import torch
from torch import nn

# Set the seed for reproducibility
torch.manual_seed(42)

# TODO:
# 0. Implement the dataset class -> scale data
# 1. Implement the model
# 2. Implement the training loop
# 3. Implement the evaluation loop
# 4. Implement the prediction loop
# 5. Implement the save and load functions


class LoadStateData:

    def __init__(self, state: str):
        self.state = state

    def load_data(self) -> pd.DataFrame:
        """
        Load the state data
        :return: pd.DataFrame
        """
        data = pd.read_csv(f"{Config.states_data_dir}/{self.state}.csv")
        return data


class LocalModel(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim: int):
        super(LocalModel, self).__init__()
        self.hidden_dim = hidden_dim

    def train(self):
        raise NotImplementedError("train method is not implemented yet.")


if __name__ == "__main__":
    pass
