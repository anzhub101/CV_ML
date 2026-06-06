"""
Unified logger for the CEN454 framework.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Processing started")
    log.warning("Low confidence detected")
    log.error("File not found")

All modules use this so every log line shows the same timestamp format,
the module name, and the level. Logs also write to outputs/run.log.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured with console + file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid adding handlers twice
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    os.makedirs("outputs", exist_ok=True)
    log_path = os.path.join("outputs", "run.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
