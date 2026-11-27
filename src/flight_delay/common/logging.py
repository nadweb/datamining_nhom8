# -*- coding: utf-8 -*-
"""Structured pipeline and serving logging configuration."""
import logging
import sys
from pathlib import Path
from typing import Optional

from flight_delay.common.constants import LOG_DIR


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger writing to logs/pipeline.log and stdout once."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    target = log_file or (LOG_DIR / "pipeline.log")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(target, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        root.setLevel(level)
    return logging.getLogger("flight_delay")


configure_logging = setup_logging
