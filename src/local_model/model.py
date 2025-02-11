# Standart library imports
from typing import Tuple
import torch
from torch import nn


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
    ):
        self.input_size = input_size  # number of input features
        self.hidden_size = hidden_size  # number of hidden units in the LSTM layer
        self.sequence_length = sequence_length  # length of the input sequence, i.e., how many time steps the model will look back
        self.learning_rate = (
            learning_rate  # how much the model is learning from the data
        )
        self.epochs = epochs  # number of times the entire dataset is passed forward and backward through the neural network
        self.batch_size = (
            batch_size  # number of samples processed in one forward/backward pass
        )


class TrainingStats:

    def __init__(self):
        self.losses = []
        # self.mse = []
        self.epochs = []

    def plot(self):
        raise NotImplementedError("Plot method is not implemented yet.")


class LocalModel(nn.Module):

    def __init__(self, hyperparameters: LSTMHyperparameters):
        super(LocalModel, self).__init__()

        self.hyperparameters: LSTMHyperparameters = hyperparameters

        # 3 layer model:
        self.lstm = nn.LSTM(
            input_size=hyperparameters.input_size,
            # Define the number of hidden units in the LSTM layer
            hidden_size=hyperparameters.hidden_size,
            # Define number of layers of the neural LSTM network
            num_layers=3,
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
        self, n_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden states for the LSTM layers.

        Args:
            n_samples (int): number of samples

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: hidden states: h_t, c_t
        """
        h_t = torch.zeros(
            n_samples, self.hyperparameters.hidden_size, dtype=torch.float32
        )
        c_t = torch.zeros(
            n_samples, self.hyperparameters.hidden_size, dtype=torch.float32
        )

        return h_t, c_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get the number batch size
        n_samples = x.size(0)

        # initialize the hidden states
        h_t, c_t = self.__initialize_hidden_states(n_samples)

        # Forward propagate through LSTM
        out, (h_n, c_n) = self.lstm(x, (h_t, c_t))  # In here 'out' is a tensor

        out = self.linear(out[:, -1, :])  # Using the last time step's output

        return out

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor):
        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyperparameters.learning_rate
        )

        # Define the training loop
        num_epochs = self.hyperparameters.epochs

        self.training_stats.epochs = list(range(num_epochs))
        for epoch in range(num_epochs):
            # Forward pass
            self.train()

            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)

            # get loss
            self.training_stats.losses.append(loss.item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    # TODO: transform this based on the data to be in the right format
    hyperparameters = LSTMHyperparameters(
        input_size=1,
        hidden_size=100,
        sequence_length=1,
        learning_rate=0.001,
        epochs=100,
        batch_size=1,
    )
    rnn = LocalModel()
