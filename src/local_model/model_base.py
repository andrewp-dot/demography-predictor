# Standard library imports
import torch
from torch import nn
from abc import abstractmethod

# Custom imports
from src.local_model.base import CustomModelBase, LSTMHyperparameters


class BaseLSTM(CustomModelBase):

    def __init__(self, hyperparameters: LSTMHyperparameters) -> None:
        """
        Initializes Base LSTM model. This model is used as a base model for finetuning.

        The model architecture contains:
        - N (hyperparameters.num_layers) hidden LSTM layers
        - 1 linear output layer

        Args:
            hyperparameters (LSTMHyperparameters): Parameters used for definition of LSTM net.
        """
        super(BaseLSTM, self).__init__()

        # Define architecture
        self.lstm = nn.LSTM(
            input_size=hyperparameters.input_size,
            # Define the number of hidden units in the LSTM layer
            hidden_size=hyperparameters.hidden_size,
            # Define number of layers of the neural LSTM network
            num_layers=hyperparameters.num_layers,
            # 'batch_first' indicates that the input is in the format (batch_size, sequence_length, input_features)
            # not in the format (sequence_length, batch_size, input_features)
            batch_first=True,
        )

        # Output size is equal to hidden size
        output_size = hyperparameters.input_size
        self.linear = nn.Linear(hyperparameters.hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        return super().forward(x)

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ):
        return super().train_model(batch_inputs, batch_targets, display_nth_epoch)
