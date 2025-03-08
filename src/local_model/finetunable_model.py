# Standard library imports
import pandas as pd
import torch
from torch import nn

# Custom imports
from src.local_model.base import LSTMHyperparameters
from src.local_model.model_base import BaseLSTM


class FineTunableLSTM(BaseLSTM):

    def __init__(self, base_model: BaseLSTM, hyperparameters: LSTMHyperparameters):
        super(FineTunableLSTM, self).__init__(hyperparameters)

        # Load pretrained LSTM layers
        self.lstm = base_model.lstm

        # Freeze pretrained LSTM layers
        for param in self.lstm.parameters():
            param.requires_grad = False

        # Add new LSTM layers for fine-tuning
        fine_tune_hidden_size = hyperparameters.hidden_size

        self.new_lstm = nn.LSTM(
            input_size=base_model.lstm.hidden_size,
            hidden_size=fine_tune_hidden_size,
            num_layers=hyperparameters.num_layers,
            batch_first=True,
        )

        # New fine-tunable output layer
        self.fc = nn.Linear(fine_tune_hidden_size, hyperparameters.input_size)

    def forward(self, x: torch.Tensor):
        return super().forward(x)

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ):
        return super().train_model(batch_inputs, batch_targets, display_nth_epoch)
