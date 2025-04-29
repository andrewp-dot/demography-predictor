# Standard library imports
import torch
import torch.nn as nn
import logging
import random

from typing import List, Dict, Union

# Custom imports
from src.utils.log import setup_logging
from src.base import CustomModelBase, RNNHyperparameters

from src.pipeline import FeatureModelPipeline
from src.evaluation import EvaluateModel

from src.utils.constants import get_core_hyperparameters

from train_scripts.train_feature_models import preprocess_data

from src.preprocessors.multiple_states_preprocessing import StatesDataLoader
from src.preprocessors.data_transformer import DataTransformer


logger = logging.getLogger("local_model")


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell


# Encoder-Decoder
class Seq2Seq(CustomModelBase):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device,
        features: List[str],
        hyperparameters: RNNHyperparameters,
        targets: List[str] | None = None,
    ):
        super(Seq2Seq, self).__init__(
            features=features,
            targets=targets,
            hyperparameters=hyperparameters,
            scaler=None,
        )
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # self.hyperparameters

    def forward(
        self,
        src: torch.Tensor,
        target_len: int,
        target: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ):
        batch_size = src.size(0)
        output_dim = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, target_len, output_dim).to(self.device)

        hidden, cell = self.encoder(src)

        decoder_input = torch.zeros(batch_size, 1, output_dim).to(self.device)

        for t in range(target_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t : t + 1, :] = decoder_output

            if target is not None and random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t : t + 1, :]  # Use ground-truth
            else:
                decoder_input = decoder_output  # Use prediction

        return outputs

    def train_model(
        self,
        batch_inputs: torch.Tensor,
        batch_targets: torch.Tensor,
        batch_validation_inputs: torch.Tensor | None = None,
        batch_validation_targets: torch.Tensor | None = None,
        display_nth_epoch: int = 10,
        loss_function: Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss] = None,
    ) -> Dict[str, List[int | float]]:
        """
        Trains model using batched input sequences and batched target sequences.

        Args:
            batch_inputs (torch.Tensor): Batches of input sequences.
            batch_targets (torch.Tensor): Batches of target sequences
            display_nth_epoch (int, optional): Display every nth epochs. Always displays first and last epoch. Defaults to 10.
            loss_function (Union[nn.MSELoss, nn.L1Loss, nn.HuberLoss], optional): _description_. Defaults to None.
            training_stats (TrainingStats | None, optional): Training stats object to track the training and validation loss, If None, no records are tracked. Defaults to None.

        Raises:
            ValueError: If the loss is None or Infinity.

        Returns:
            out: Dict[str, List[int | float]]: Training statistics. contains list of epochs, training and validation loss curves.
        """

        torch.autograd.set_detect_anomaly(True)

        # Get loss function
        criterion = loss_function
        if loss_function is None:
            criterion = nn.MSELoss()

        # Put the model to the device
        self.to(device=self.device)

        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        # optimizer = torch.optim.Adam(
        #     self.parameters(), lr=self.hyperparameters.learning_rate
        # )
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        # Define the training stats
        training_stats: Dict[str, List[int | float]] = {
            "epochs": [],
            "training_loss": [],
            "validation_loss": [],
        }

        # Define the training loop
        # num_epochs = self.hyperparameters.epochs
        num_epochs = 10
        training_stats["epochs"] = list(range(num_epochs))

        # Set the flag for gettting the validation loss
        GET_VALIDATION_CURVE: bool = (
            not batch_validation_inputs is None and not batch_validation_targets is None
        )

        def get_teacher_forcing_ratio(epoch: int, total_epochs: int):
            return max(0, 1.0 - epoch / total_epochs)  # Linear decay

        # Training loop
        for epoch in range(num_epochs):

            # Init epoch loss
            epoch_loss = 0.0

            teacher_forcing_ratio = get_teacher_forcing_ratio(epoch, num_epochs)

            for batch_input, batch_target in zip(batch_inputs, batch_targets):

                # Put the targets to the device
                batch_input, batch_target = batch_input.to(
                    device=self.device
                ), batch_target.to(device=self.device)

                # Forward pass
                outputs = self(
                    batch_input,
                    target=batch_target,
                    target_len=batch_target.size(1),
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

                # Compute loss
                loss = criterion(outputs, batch_target)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Loss is NaN/Inf at epoch {epoch}")
                    raise ValueError("Loss became NaN or Inf, stopping training!")

                loss.to(
                    device=self.device
                )  # Use this to prevent errors, see if this will work on azure

                epoch_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()  # Reset gradients
                loss.backward()  # Computes gradients
                optimizer.step()  # Update weights and biases

            epoch_loss /= len(batch_inputs)
            training_stats["training_loss"].append(epoch_loss)

            # Get validation loss if available
            if GET_VALIDATION_CURVE:
                validation_epoch_loss = 0.0

                with torch.no_grad():

                    for batch_input, batch_target in zip(
                        batch_validation_inputs, batch_validation_targets
                    ):

                        # Put the targets to the device
                        batch_input, batch_target = batch_input.to(
                            device=self.device
                        ), batch_target.to(device=self.device)

                        # Forward pass
                        outputs = self(
                            batch_input,
                            target=batch_target,
                            target_len=batch_target.size(1),
                            teacher_forcing_ratio=0.0,  # during validation, no teacher forcing usually!
                        )

                        # Compute loss
                        loss = criterion(outputs, batch_target)

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.error(f"Loss is NaN/Inf at epoch {epoch}")
                            raise ValueError(
                                "Loss became NaN or Inf, stopping training!"
                            )

                        loss.to(
                            device=self.device
                        )  # Use this to prevent errors, see if this will work on azure

                        validation_epoch_loss += loss.item()

                # Save the stats
                validation_epoch_loss /= len(batch_validation_inputs)
                training_stats["validation_loss"].append(validation_epoch_loss)

            # Display loss
            if not epoch % display_nth_epoch or epoch == (
                num_epochs - 1
            ):  # Display first, nth epoch and last
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation loss: {validation_epoch_loss:.4f}"
                )

        return training_stats


def create_seq_to_seq():

    FEATURES = [
        col.lower()
        for col in [
            # "year",
            # "Fertility rate, total",
            "population, total",
            # "Net migration",
            "Arable land",
            # "Birth rate, crude",
            "GDP growth",
            "Death rate, crude",
            "Agricultural land",
            # "Rural population",
            "Rural population growth",
            # "Age dependency ratio",
            "Urban population",
            "Population growth",
            # "Adolescent fertility rate",
            # "Life expectancy at birth, total",
        ]
    ]

    INPUT_DIM = len(FEATURES)  # number of features in input
    OUTPUT_DIM = len(FEATURES)  # number of features in output
    HIDDEN_DIM = 64
    SEQ_LEN = 10
    TARGET_LEN = 5

    # Get data
    loader = StatesDataLoader()

    all_states_dict = loader.load_all_states()

    transformer = DataTransformer()

    hyperparams = (
        get_core_hyperparameters(
            input_size=INPUT_DIM,
            output_size=OUTPUT_DIM,
            hidden_size=HIDDEN_DIM,
            sequence_length=SEQ_LEN,
            future_step_predict=TARGET_LEN,
        ),
    )

    train_input_batch, train_target_batch, val_input_batch, val_target_batch = (
        preprocess_data(
            data=all_states_dict,
            features=FEATURES,
            hyperparameters=hyperparams,
            transformer=transformer,
            is_fitted=False,
        )
    )

    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(INPUT_DIM, HIDDEN_DIM).to(device)
    decoder = Decoder(OUTPUT_DIM, HIDDEN_DIM).to(device)
    model = Seq2Seq(
        encoder, decoder, device, features=FEATURES, hyperparameters=hyperparams
    ).to(device)

    stats = model.train_model(
        batch_inputs=train_input_batch,
        batch_targets=train_target_batch,
        batch_validation_inputs=val_input_batch,
        batch_validation_targets=val_target_batch,
        display_nth_epoch=1,
    )

    pipeline = FeatureModelPipeline(
        name="seq2seq", transformer=transformer, model=Seq2Seq
    )

    pipeline.save_pipeline()


def eval():

    pipeline = FeatureModelPipeline.get_pipeline(name="seq2seq")
    evaluation = EvaluateModel(pipeline=pipeline)

    loader = StatesDataLoader()
    all_states_dict = loader.load_states(
        states=["Czechia", "United States", "Honduras"]
    )

    X_test_states, y_test_states = loader.split_data(
        states_dict=all_states_dict, sequence_len=10, future_steps=5
    )
    df = evaluation.eval_for_every_state(X_test_states, y_test_states)

    print(df)


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Try the encoder decoder
    create_seq_to_seq()

    eval()
