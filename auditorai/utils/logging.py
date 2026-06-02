"""
Logging utilities for AuditorAI.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create and return a configured logger.

    Sets up a StreamHandler at INFO level with a standard format.

    Args:
        name: The logger name (typically __name__).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
