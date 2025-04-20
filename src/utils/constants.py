# Custom library imports
from src.base import LSTMHyperparameters


# Create or load base model
def get_core_hyperparameters(
    input_size: int,
    hidden_size: int = 512,
    future_step_predict: int = 1,
    sequence_length: int = 10,
    learning_rate: float = 0.0001,
    epochs: int = 10,
    batch_size: int = 1,
    num_layers: int = 3,
    output_size: int | None = None,
    bidirectional: bool = False,
) -> LSTMHyperparameters:
    BASE_HYPERPARAMETERS: LSTMHyperparameters = LSTMHyperparameters(
        input_size=input_size,
        hidden_size=hidden_size,
        future_step_predict=future_step_predict,
        output_size=output_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )

    return BASE_HYPERPARAMETERS
