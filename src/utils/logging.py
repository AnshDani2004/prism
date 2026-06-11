"""Logging configuration for PRISM."""

import logging
import sys

from config.settings import Settings


def setup_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure root logger with consistent formatting."""
    settings = settings or Settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("prism")
