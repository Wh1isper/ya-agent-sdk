"""Bounded logging configuration for the yaacli CLI and TUI.

The TUI must never write logging output to stderr because that corrupts the
prompt_toolkit display. Non-verbose logging is therefore silent. Verbose
logging is written through one rotating ``yaacli.log`` handler so long-running
sessions cannot grow logs without bound.
"""

from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Logger names to configure
TUI_LOGGER_NAME = "yaacli"
SDK_LOGGER_NAME = "ya_agent_sdk"
PY_WARNINGS_LOGGER_NAME = "py.warnings"

# Verbose log retention defaults: up to 20 MiB across the active file and
# three backups. They are module constants so the retention policy is clear.
LOG_FILE_NAME = "yaacli.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

# Cache for initialization state
_initialized = False
_verbose_mode = False
_configured_root_handlers: list[logging.Handler] = []


def _clear_handlers(logger: logging.Logger) -> None:
    """Remove and close every handler directly attached to ``logger``."""
    handlers = logger.handlers[:]
    logger.handlers.clear()
    for handler in handlers:
        handler.close()


def _make_rotating_file_handler() -> RotatingFileHandler:
    """Create the single bounded verbose log handler with standard formatting."""
    handler = RotatingFileHandler(
        Path.cwd() / LOG_FILE_NAME,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def _configure_logger(name: str, level: int) -> None:
    """Route a yaacli logger through the root's stderr-safe handler."""
    logger = logging.getLogger(name)
    _clear_handlers(logger)
    logger.setLevel(level)
    logger.propagate = True


def _configure_warning_logger() -> None:
    """Route Python warnings through the root handler instead of stderr."""
    logging.captureWarnings(True)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="builtin type swigvarlink has no __module__ attribute",
    )

    logger = logging.getLogger(PY_WARNINGS_LOGGER_NAME)
    _clear_handlers(logger)
    logger.setLevel(logging.WARNING)
    logger.propagate = True


def _redirect_root_logger(verbose: bool = False) -> None:
    """Install the only handler used for yaacli CLI/TUI logging.

    A single rotating handler avoids the unsafe situation where several file
    handlers independently rotate the same file. Third-party logs still reach
    this handler, but the root level excludes their DEBUG and INFO noise.
    """
    global _configured_root_handlers

    root = logging.getLogger()
    _clear_handlers(root)
    handler: logging.Handler = _make_rotating_file_handler() if verbose else logging.NullHandler()
    root.addHandler(handler)
    _configured_root_handlers = [handler]
    root.setLevel(logging.WARNING)


def configure_tui_logging(
    level: int = logging.INFO,
    verbose: bool = False,
) -> None:
    """Configure bounded, stderr-safe logging for TUI mode.

    Non-verbose mode discards logs through a root ``NullHandler``. Verbose mode
    writes DEBUG-and-above yaacli and SDK logs to ``yaacli.log`` using one
    :class:`RotatingFileHandler` (5 MiB per file, three backups by default).
    The old unconsumed asyncio log queue is deliberately not retained.

    Args:
        level: Minimum yaacli/SDK level when verbose logging is disabled.
        verbose: Write DEBUG-and-above yaacli/SDK logs to the rotating file.
    """
    global _initialized

    if _initialized:
        return

    effective_level = logging.DEBUG if verbose else level
    _redirect_root_logger(verbose=verbose)
    _configure_logger(TUI_LOGGER_NAME, effective_level)
    _configure_logger(SDK_LOGGER_NAME, effective_level)
    _configure_warning_logger()
    _initialized = True


def reset_logging() -> None:
    """Reset yaacli logging configuration and close its file handler.

    Useful for tests or when reconfiguring from TUI to another logging setup.
    """
    global _initialized, _verbose_mode, _configured_root_handlers

    logging.captureWarnings(False)

    for name in [TUI_LOGGER_NAME, SDK_LOGGER_NAME, PY_WARNINGS_LOGGER_NAME]:
        _clear_handlers(logging.getLogger(name))

    root = logging.getLogger()
    for handler in _configured_root_handlers:
        if handler in root.handlers:
            root.removeHandler(handler)
        handler.close()
    _configured_root_handlers = []

    _initialized = False
    _verbose_mode = False


def configure_logging(verbose: bool = False) -> None:
    """Configure bounded startup logging before the TUI is initialized.

    Args:
        verbose: If True, write DEBUG-and-above yaacli/SDK logs to the rotating
            file; otherwise discard logging output.
    """
    global _verbose_mode
    _verbose_mode = verbose

    _redirect_root_logger(verbose=verbose)
    level = logging.DEBUG if verbose else logging.WARNING
    _configure_logger(TUI_LOGGER_NAME, level)
    _configure_logger(SDK_LOGGER_NAME, level)
    _configure_warning_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger below the yaacli namespace."""
    if not name.startswith(TUI_LOGGER_NAME):
        name = f"{TUI_LOGGER_NAME}.{name}"

    return logging.getLogger(name)
