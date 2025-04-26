"""Create logger based on defined criteria"""

import logging


def create_logger():
    """Logger that outputs INFO to stdout; both DEBUG and INFO to file.

    Returns:
        logging.logger: Configured logger
    """

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("debug.log")

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters
    formatter_console = logging.Formatter("%(levelname)s - %(message)s")
    formatter_file = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Add formatters to handlers
    console_handler.setFormatter(formatter_console)
    file_handler.setFormatter(formatter_file)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
