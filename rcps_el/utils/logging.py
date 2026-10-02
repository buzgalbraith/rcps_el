"""
Logging that plays well with tqdm progress bars.

A plain StreamHandler writes straight to the terminal, which tears through any
bar that is currently drawn and leaves broken copies of it behind. Routing
records through tqdm.write instead clears the bars, prints the message above
them, then redraws the bars underneath.
"""

import logging

from tqdm import tqdm

PACKAGE_LOGGER = "rcps_el"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


class TqdmLoggingHandler(logging.Handler):
    """Emit log records with tqdm.write so they print above active bars"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    """
    Send rcps_el log records through a tqdm safe handler at the given level.

    Safe to call more than once: the handler is only added the first time and
    later calls just change the level. Records do not propagate to the root
    logger, so a root handler set up by basicConfig will not print them twice.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(level)
    if not any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers):
        handler = TqdmLoggingHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_tqdm_logging() -> None:
    """
    Fall back to a tqdm safe handler when nothing has configured rcps_el logging.

    Without any handler, warnings go to logging.lastResort, which writes
    directly to stderr and breaks the bars. A configuration the caller set up
    (setup_logging or basicConfig) is left alone.
    """
    if not logging.getLogger(PACKAGE_LOGGER).hasHandlers():
        setup_logging(logging.WARNING)
