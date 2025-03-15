import logging
import logging.config
import yaml
from colorama import Fore


def setup_logging() -> None:
    # Load YAML config
    try:
        with open("loggers_config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Apply logging configuration
        logging.config.dictConfig(config)
    except Exception as e:
        print(f"Failed to setup logging: {e}!")


class ColorfulFormatter(logging.Formatter):

    LEVEL_COLOR: dict = {
        logging.DEBUG: Fore.CYAN,
        logging.WARNING: Fore.LIGHTYELLOW_EX,
        logging.INFO: Fore.GREEN,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record: logging.LogRecord) -> str:

        # Get items from record
        msg = record.msg
        level_no = record.levelno
        level_name = record.levelname
        asctime = self.formatTime(record, self.datefmt)

        # File info
        file_name = record.filename
        line_num = record.lineno

        # Format message
        formatted_msg = f"{asctime} - {self.LEVEL_COLOR[level_no]}[{level_name} - {file_name}:{line_num}]{Fore.RESET}: {msg} "

        return formatted_msg
