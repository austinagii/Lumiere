"""Signal handling utilities for graceful shutdown."""

import logging
import signal
import sys


logger = logging.getLogger(__name__)


def _exit_handler(sig, frame):
    """Handle process interruption gracefully.

    Args:
        sig: The signal number received.
        frame: The current stack frame (unused).
    """
    print()  # Print newline after ^C
    logger.info("Process halted by user")
    sys.exit(0)


def register_signal_handlers():
    """Register signal handlers for gracefully stopping the process.

    Registers handlers for:
    - SIGINT (Ctrl+C): Graceful shutdown
    - SIGTERM (kill command): Graceful shutdown
    """
    signal.signal(signal.SIGINT, _exit_handler)
    signal.signal(signal.SIGTERM, _exit_handler)
