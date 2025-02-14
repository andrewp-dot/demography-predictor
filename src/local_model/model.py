# Standard library imports
from typing import Tuple
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import logging

# Custom imports
from src.local_model.preprocessing import StateDataLoader
from config import setup_logging


logger = logging.getLogger("local_model")


# Set the seed for reproducibility
torch.manual_seed(42)

# TODO:
# Implement training stats (plotting etc.)


class LSTMHyperparameters:

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        num_layers: int,
        bidirectional: bool = False,
    ):
        self.input_size = input_size  # Number of input features
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


class TrainingStats:

    def __init__(self):
        self.losses = []
        # self.mse = []
        self.epochs = []

    def plot(self):
        # Define figure parameters
        plt.figure(figsize=(10, 5))

        # Name the axis
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        # Plot the graph(s)
        plt.plot(self.epochs, self.losses)

        # Show the plot
        # plt.legend()
        plt.grid()
        plt.show()


class LocalModel(nn.Module):

    def __init__(self, hyperparameters: LSTMHyperparameters):
        super(LocalModel, self).__init__()

        self.hyperparameters: LSTMHyperparameters = hyperparameters

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3 layer model:
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

        # Out layer: Linear layer
        self.linear = nn.Linear(
            in_features=hyperparameters.hidden_size,
            out_features=hyperparameters.input_size,
        )

        # Get stats
        self.training_stats = TrainingStats()

    def __initialize_hidden_states(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden states for the LSTM layers.

        Args:
            n_samples (int): number of samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden states: h_t, c_t
        """
        h_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        )

        c_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        )

        return h_t, c_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the number batch size
        bath_size = x.size(0)

        # initialize the hidden states
        h_t, c_t = self.__initialize_hidden_states(bath_size)

        # if x.dim() == 2:
        #     x = x.unsqueeze(0)  # Add batch dimension: (1, seq_len, input_size)

        # Forward propagate through LSTM
        # print(f"X SHAPE: {x.shape}")
        out, (h_n, c_n) = self.lstm(x, (h_t, c_t))  # In here 'out' is a tensor

        out = self.linear(out[:, -1, :])  # Using the last time step's output

        return out

    def train(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor):
        # Put the model to the device
        self.to(device=self.device)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyperparameters.learning_rate
        )

        # Define the training loop
        num_epochs = self.hyperparameters.epochs

        # Setup epochs for stats
        self.training_stats.epochs = list(range(num_epochs))

        # Training loop
        for epoch in range(num_epochs):

            # Init epoch loss
            epoch_loss = 0

            for batch_input, batch_target in zip(batch_inputs, batch_targets):
                logger.debug(f"[Training loop] input: {batch_input.shape}")
                logger.debug(f"[Training loop] target: {batch_target.shape}")

                # Put the targets to the device
                batch_input, batch_target = batch_input.to(
                    self.device
                ), batch_target.to(self.device)

                # Forward pass
                outputs = self(batch_input)

                # Compute loss
                loss = criterion(outputs, batch_target)
                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss /= len(batch_inputs)
            self.training_stats.losses.append(epoch_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    # Setup logging
    setup_logging()

    # TODO: transform this based on the data to be in the right format
    FEATURES = ["population, total", "net migration"]

    # FEATURES = [
    #     "Fertility rate, total",
    #     "Population, total",
    #     "Net migration",
    #     "Arable land",
    #     "Birth rate, crude",
    #     "GDP growth",
    #     "Death rate, crude",
    #     "Agricultural land",
    #     "Rural population",
    #     "Rural population growth",
    #     "Age dependency ratio",
    #     "Urban population",
    #     "Population growth",
    #     "Adolescent fertility rate",
    #     "Life expectancy at birth, total",
    # ]

    # FEATURES = [col.lower() for col in FEATURES]

    hyperparameters = LSTMHyperparameters(
        input_size=len(FEATURES),
        hidden_size=64,
        sequence_length=5,
        learning_rate=0.001,
        epochs=30,
        batch_size=1,
        num_layers=3,
    )
    rnn = LocalModel(hyperparameters)

    # Load data
    czech_loader = StateDataLoader("Czechia")

    czech_data = czech_loader.load_data()

    # Exclude country name
    czech_data = czech_data.drop(columns=["country name"])

    # Scale data
    scaled_cz_data, cz_scaler = czech_loader.scale_data(
        czech_data, scaler=RobustScaler()
    )

    print(scaled_cz_data.head())

    train, test = czech_loader.split_data(scaled_cz_data)

    # Get input/output sequences
    input_sequences, target_sequences = czech_loader.preprocess_data(
        train, hyperparameters.sequence_length, features=FEATURES
    )

    # Get input/output batches of sequences
    batch_inputs, batch_targets = czech_loader.create_batches(
        batch_size=hyperparameters.batch_size,
        input_sequences=input_sequences,
        target_sequences=target_sequences,
    )

    print("-" * 100)
    logger.info(f"Batch inputs shape: {batch_inputs.shape}")
    logger.info(f"Batch targets shape: {batch_targets.shape}")

    # Train model
    rnn.train(batch_inputs=batch_inputs, batch_targets=batch_targets)

    # Get train stats
    stats = rnn.training_stats

    print("-" * 100)
    print("Loses:")
    print(stats.losses)

    stats.plot()
