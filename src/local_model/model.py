# Standard library imports
from typing import Tuple
import torch
from torch import nn

# Custom imports
from src.local_model.preprocessing import StateDataLoader


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
        raise NotImplementedError("Plot method is not implemented yet.")


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
        self, n_samples: int
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
            self.hyperparameters.batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
        )

        c_t = torch.zeros(
            self.hyperparameters.num_layers
            * (2 if self.hyperparameters.bidirectional else 1),  # for bidirectional
            self.hyperparameters.batch_size,
            self.hyperparameters.hidden_size,
            dtype=torch.float32,
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

        self.training_stats.epochs = list(range(num_epochs))
        for epoch in range(num_epochs):
            for batch_input, batch_target in zip(batch_inputs, batch_targets):
                print(f"[Training loop] input: {batch_input.shape}")
                print(f"[Training loop] target: {batch_target.shape}")
                batch_inputs, batch_targets = batch_inputs.to(
                    self.device
                ), batch_targets.to(self.device)

                # Forward pass
                outputs = self(batch_input)

                # Compute loss
                loss = criterion(outputs, batch_target)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            # # Forward pass
            # self.train()

            # # Forward pass
            # outputs = self(X_train)
            # loss = criterion(outputs, y_train)

            # # get loss
            # self.training_stats.losses.append(loss.item())

            # # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # LSTM explanation - https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2

    # Notes:
    # statefull vs stateless LSTM - can I pass data with different batch sizes?

    # TODO: transform this based on the data to be in the right format
    hyperparameters = LSTMHyperparameters(
        input_size=2,
        hidden_size=100,
        sequence_length=5,
        learning_rate=0.001,
        epochs=100,
        batch_size=1,
        num_layers=3,
    )
    rnn = LocalModel(hyperparameters)

    # Load data
    czech_loader = StateDataLoader("Czechia")

    czech_data = czech_loader.load_data()

    train, test = czech_loader.split_data(czech_data)

    # Get input/output sequences
    input_sequences, target_sequences = czech_loader.preprocess_data(
        train, hyperparameters.sequence_length, features=["population, total"]
    )

    # Get input/output batches of sequences
    batch_inputs, batch_targets = czech_loader.create_batches(
        batch_size=hyperparameters.batch_size,
        input_sequences=input_sequences,
        target_sequences=target_sequences,
    )

    print("-" * 100)
    print("Batch inputs shape: ", end="")
    print(batch_inputs.shape)
    print("Batch targets shape: ", end="")
    print(batch_targets.shape)

    # Train model
    rnn.train(batch_inputs=batch_inputs, batch_targets=batch_targets)

    # Get train stats
    stats = rnn.training_stats

    print("-" * 100)
    print("Loses:")
    print(stats.losses)
