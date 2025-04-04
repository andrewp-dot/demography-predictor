"""
In this script you can find the basic implementation for the local models. The classes are used for model definition, training and evaluation.
"""

# Standard library imports
import pandas as pd
import torch
from torch import nn

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
from abc import abstractmethod
from typing import List

import logging

logger = logging.getLogger("local_model")


# Set the seed for reproducibility
torch.manual_seed(42)


class LSTMHyperparameters:

    def __init__(
        self,
        input_size: int,
        # output_size: int,
        hidden_size: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        num_layers: int,
        bidirectional: bool = False,
    ):
        """
        Define parameters for LSTM networks

        Args:
            input_size (int): Defines the number of input features.
            output_size (int): Defines the number of output features.
            hidden_size (int): Defines the number of neurons in a layer.
            sequence_length (int): Length of the processing sequnece (number of past samples using for predicition).
            learning_rate (float): Defines how much does the model learn (step in gradient descend).
            epochs (int): Number of epochs to train the nerual network.
            batch_size (int): Number of samples used to update the weights in the neural network. Bigger batch size for faster training and better generalization.
            num_layers (int): Number of LSTM layers (or LSTM combined layers in neura networks). In case of FineTunable networks, defines the number of finetunable layers.
            bidirectional (bool, optional): If you can go also forward and backward (for gatharing context from the past and also from future). Defaults to False.
        """
        self.input_size = input_size  # Number of input features
        # self.output_size = output_size  # Number of output features
        self.hidden_size = hidden_size  # Number of hidden units in the LSTM layer
        self.sequence_length = sequence_length  # Length of the input sequence, i.e., how many time steps the model will look back
        self.learning_rate = (
            learning_rate  # How much the model is learning from the data
        )
        self.epochs = epochs  # Number of times the entire dataset is passed forward and backward through the neural network
        self.batch_size = (
            batch_size  # Number of samples processed in one forward/backward pass
        )
        self.num_layers = num_layers  # Number of LSTM layers
        self.bidirectional = bidirectional

    def __repr__(self) -> str:

        repr_string = f"""
Input size:         {self.input_size}
Batch size:         {self.batch_size}

Hidden size:        {self.hidden_size}
Sequence length:    {self.sequence_length}
Layers:             {self.num_layers}

Learning rate:      {self.learning_rate}
Epochs:             {self.epochs}

Bidirectional:      {self.bidirectional}
"""
        return repr_string


class TrainingStats:

    def __init__(self):
        self.losses = []
        self.epochs = []

    def create_plot(self) -> Figure:
        """
        Creates a figure of training statistics.

        Returns:
            out: Figure: Figure of training statistics.
        """

        # Define figure parameters
        fig = plt.figure(figsize=(10, 5))

        # Name the axis
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Plot the graph(s)
        plt.plot(self.epochs, self.losses)

        # Show the plot
        # plt.legend()
        plt.grid()

        return fig

    def plot(self) -> None:
        fig: Figure = self.create_plot()
        fig.show()


class CustomModelBase(nn.Module):
    """
    Defines the interface for the recurrent neural networks models in this project. From this

    Raises:
        NotImplementedError: If forward method is not implemented yet.
        NotImplementedError: If train_model method is not implemented yet.
        NotImplementedError: If predict method is not implemented yet.
    """

    def __init__(
        self,
        features: List[str],
        targets: List[str],
        hyperparameters: LSTMHyperparameters,
        scaler: MinMaxScaler,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.hyperparameters: LSTMHyperparameters = hyperparameters

        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

        self.SCALER: MinMaxScaler = scaler

    def set_scaler(self, scaler: MinMaxScaler) -> None:
        if self.SCALER is None:
            self.SCALER = scaler

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method for your model is not implemented!")

    @abstractmethod
    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        display_nth_epoch: int = 10,
    ) -> None:
        raise NotImplementedError("Train function for your model is not implemented!")

    @abstractmethod
    def predict(
        self,
        input_data: pd.DataFrame,
        last_year: int,
        target_year: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Predict function for your model is not implemented!")
