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


# TODO: if you want to make the RNNHyperparameters more portable, just support serializing it to dict or loading it from dict.
class RNNHyperparameters:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        future_step_predict: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        num_layers: int,
        output_size: int | None = None,
        bidirectional: bool = False,
    ):
        """
        Define parameters for reccurent neural networks

        Args:
            input_size (int): Defines the number of input features.
            output_size (int): Defines the number of output features. If None the output_size is equal to the input size. Defaults to None.
            hidden_size (int): Defines the number of neurons in a layer.
            future_step_predict (int): Defines how many timestep forward it should the LSTM network predict using 1 sequence.
            sequence_length (int): Length of the processing sequnece (number of past samples using for predicition).
            learning_rate (float): Defines how much does the model learn (step in gradient descend).
            epochs (int): Number of epochs to train the nerual network.
            batch_size (int): Number of samples used to update the weights in the neural network. Bigger batch size for faster training and better generalization.
            num_layers (int): Number of LSTM layers (or LSTM combined layers in neura networks). In case of FineTunable networks, defines the number of finetunable layers.
            bidirectional (bool, optional): If you can go also forward and backward (for gatharing context from the past and also from future). Defaults to False.
        """
        self.input_size = input_size  # Number of input features
        self.output_size = (
            output_size if output_size else input_size
        )  # Number of output features
        self.hidden_size = hidden_size  # Number of hidden units in the LSTM layer
        self.future_step_predict = future_step_predict

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
Output size:        {self.output_size}
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
        self.training_loss = []
        self.validation_loss = []
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
        plt.plot(self.epochs, self.training_loss, label="Training loss")
        plt.plot(self.epochs, self.validation_loss, label="Validation loss")

        # Show the plot
        plt.legend()
        plt.grid()

        return fig

    def plot(self) -> None:
        fig: Figure = self.create_plot()
        fig.show()

    @classmethod
    def from_dict(cls, stats_dict: dict) -> "TrainingStats":
        """
        Creates a TrainingStats instance from a dictionary.

        Args:
            stats_dict (dict): Dictionary containing 'training_loss', 'validation_loss', and 'epochs'.

        Returns:
            out: TrainingStats: An instance populated with the given data.
        """
        instance = cls()
        instance.training_loss = stats_dict.get("training_loss", [])
        instance.validation_loss = stats_dict.get("validation_loss", [])
        instance.epochs = stats_dict.get("epochs", [])
        return instance


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
        hyperparameters: RNNHyperparameters,
        device: torch.device,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # TODO: change this to separate parts for portability
        self.hyperparameters: RNNHyperparameters = hyperparameters

        self.FEATURES: List[str] = features
        self.TARGETS: List[str] = targets

        self.device = device

    def set_device(self, device: torch.device) -> None:
        """
        Set the BaseRNN device property.

        Args:
            device (torch.device): Device for the BaseRNN object.
        """
        self.device = device

    def redetect_device(self) -> None:
        """
        Set the device to cuda if available. Useful if you are loading pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method for your model is not implemented!")

    @abstractmethod
    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_validation_inputs: torch.Tensor | None = None,
        batch_validation_targets: torch.Tensor | None = None,
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
