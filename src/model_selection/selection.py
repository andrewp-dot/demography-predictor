import pycaret
import logging
import pandas as pd
from config import setup_logging, Config

# Get config
config = Config()

# Get logger
logger = logging.getLogger("method_selection")


# TODO:
# 1. Load data
def load_data() -> pd.DataFrame:
    return pd.read_csv(config.dataset_path)


# 2. Create regression experiment
def regression_experiment() -> None:
    raise NotImplementedError()


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # 1. Load data
    data_df = load_data()

    # 2. Run regression experiment
