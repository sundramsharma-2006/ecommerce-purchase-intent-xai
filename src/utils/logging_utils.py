"""Logging helper: one consistent format, no print() in library code."""
from __future__ import annotations

import logging

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_FMT)
    logger.setLevel(level)
    return logger
