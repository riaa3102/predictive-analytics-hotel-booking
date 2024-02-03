import logging
from typing import Optional

from .dirs import DIRS


def configure_logger(name: Optional[str] = 'main',
                     log_level: Optional[str] = 'DEBUG',
                     log_to_file: Optional[bool] = True,
                     ) -> logging.Logger:

    # Create logger if it doesn't exist
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Set the log level dynamically
        numeric_level = getattr(logging, log_level.upper(), logging.DEBUG)
        logger.setLevel(numeric_level)

        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # Add console handler to the logger
        logger.addHandler(console_handler)

        # Optionally, add a file handler
        if log_to_file:
            file_handler = logging.FileHandler(DIRS["logs_file_path"])
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            # Add file handler to the logger
            logger.addHandler(file_handler)

    return logger
