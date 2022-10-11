"""Custom logger implemenation."""
import logging
import os

# Logging level from 0 (all) to 4 (errors) (see https://docs.python.org/3/library/logging.html#logging-levels)
QCVV_LOG_LEVEL = 1
if "QCVV_LOG_LEVEL" in os.environ:  # pragma: no cover
    QCVV_LOG_LEVEL = 10 * int(os.environ.get("QCVV_LOG_LEVEL"))


def raise_error(exception, message=None, args=None):
    """Raise exception with logging error.

    Args:
        exception (Exception): python exception.
        message (str): the error message.
    """
    log.error(message)
    if args:
        raise exception(message, args)
    else:
        raise exception(message)


# Configuration for logging mechanism
class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm."""

    def format(self, record):
        """Format the record with specific format."""
        from qcvv import __version__

        fmt = f"[Qcvv {__version__}|%(levelname)s|%(asctime)s]: %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(QCVV_LOG_LEVEL)
log.addHandler(CustomHandler())
