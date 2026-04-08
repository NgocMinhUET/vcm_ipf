"""Structured logging setup for the IPF pipeline.

Uses Python stdlib logging with Rich handler for pretty console output.
Each run creates both a console stream and a file handler.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    name: str = "ipf",
) -> logging.Logger:
    """Configure and return the root IPF logger.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: If provided, also log to this file.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _CONFIGURED:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(module_name: str = "ipf") -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"ipf.{module_name}")
