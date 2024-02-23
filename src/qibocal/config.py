"""Custom logger implemenation."""

import logging
import os

# Logging levels available here https://docs.python.org/3/library/logging.html#logging-levels
QIBOCAL_LOG_LEVEL = 10
if "QIBOCAL_LOG_LEVEL" in os.environ:  # pragma: no cover
    QIBOCAL_LOG_LEVEL = 10 * int(os.environ.get("QIBOCAL_LOG_LEVEL"))


def raise_error(exception, message=None, args=None):
    """Raise exception with logging error.

    Args:
        exception (Exception): python exception.
        message (str): the error message.
    """
    log.error(message)
    if args:
        raise exception(message, args)

    raise exception(message)


# Configuration for logging mechanism
class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm."""

    def __init__(self):
        super().__init__()
        self.FORMATS = None

    def format(self, record):
        """Format the record with specific format."""
        from qibocal import __version__

        fmt = f"[Qibocal {__version__}|%(levelname)s|%(asctime)s]: %(message)s"

        grey = "\x1b[38;20m"
        green = "\x1b[92m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: green + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset,
        }
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(QIBOCAL_LOG_LEVEL)
log.addHandler(CustomHandler())
