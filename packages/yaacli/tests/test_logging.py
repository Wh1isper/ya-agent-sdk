"""Tests for bounded yaacli logging configuration."""

from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler

import yaacli.logging as yaacli_logging
from yaacli.logging import (
    LOG_BACKUP_COUNT,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
    PY_WARNINGS_LOGGER_NAME,
    SDK_LOGGER_NAME,
    TUI_LOGGER_NAME,
    configure_logging,
    configure_tui_logging,
    get_logger,
    reset_logging,
)


class TestConfigureTuiLogging:
    """Tests for stderr-safe bounded TUI logging."""

    def teardown_method(self) -> None:
        reset_logging()

    def test_non_verbose_tui_logging_is_silent_without_a_queue(self) -> None:
        """TUI logging must not retain events in an unconsumed asyncio queue."""
        configure_tui_logging()

        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)
        for name in [TUI_LOGGER_NAME, SDK_LOGGER_NAME]:
            logger = logging.getLogger(name)
            assert logger.handlers == []
            assert logger.propagate is True

    def test_verbose_adds_one_rotating_file_handler(self, tmp_path, monkeypatch) -> None:
        """Verbose TUI logs use one bounded rotating handler, not per-logger files."""
        monkeypatch.chdir(tmp_path)
        configure_tui_logging(verbose=True)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert logger.handlers == []
        handler = logging.getLogger().handlers[0]
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == LOG_MAX_BYTES
        assert handler.backupCount == LOG_BACKUP_COUNT

        logger.info("Test verbose log")
        handler.flush()
        assert "Test verbose log" in (tmp_path / LOG_FILE_NAME).read_text()

    def test_verbose_rotation_bounds_log_files(self, tmp_path, monkeypatch) -> None:
        """The configured rotation policy creates bounded backups on overflow."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(yaacli_logging, "LOG_MAX_BYTES", 160)
        monkeypatch.setattr(yaacli_logging, "LOG_BACKUP_COUNT", 2)
        configure_tui_logging(verbose=True)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        for index in range(8):
            logger.debug("%s %s", index, "x" * 80)
        for handler in logging.getLogger().handlers:
            handler.flush()

        log_files = sorted(tmp_path.glob(f"{LOG_FILE_NAME}*"))
        assert [path.name for path in log_files] == ["yaacli.log", "yaacli.log.1", "yaacli.log.2"]
        assert all(path.stat().st_size <= 160 for path in log_files)

    def test_warnings_are_routed_away_from_stderr(self) -> None:
        """Python warnings remain silent in default TUI mode."""
        configure_tui_logging()

        logger = logging.getLogger(PY_WARNINGS_LOGGER_NAME)
        assert logger.handlers == []
        assert logger.propagate is True
        assert isinstance(logging.getLogger().handlers[0], logging.NullHandler)

    def test_verbose_warnings_use_rotating_file(self, tmp_path, monkeypatch) -> None:
        """Warnings share the bounded verbose logging policy."""
        monkeypatch.chdir(tmp_path)
        configure_tui_logging(verbose=True)

        logger = logging.getLogger(PY_WARNINGS_LOGGER_NAME)
        assert logger.handlers == []
        assert logger.propagate is True
        assert isinstance(logging.getLogger().handlers[0], RotatingFileHandler)

        logger.warning("warning routed")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "warning routed" in (tmp_path / LOG_FILE_NAME).read_text()

    def test_swig_shutdown_warning_is_filtered(self) -> None:
        """Known SWIG shutdown noise stays filtered in TUI mode."""
        configure_tui_logging()

        with warnings.catch_warnings(record=True) as records:
            warnings.warn(
                "builtin type swigvarlink has no __module__ attribute",
                DeprecationWarning,
                stacklevel=1,
            )

        assert records == []

    def test_idempotent(self) -> None:
        """Repeated setup does not duplicate root handlers."""
        configure_tui_logging()
        configure_tui_logging()

        assert len(logging.getLogger().handlers) == 1


class TestGetLogger:
    """Tests for yaacli logger names."""

    def teardown_method(self) -> None:
        reset_logging()

    def test_prefixes_name(self) -> None:
        assert get_logger("mymodule").name == f"{TUI_LOGGER_NAME}.mymodule"

    def test_already_prefixed(self) -> None:
        assert get_logger(f"{TUI_LOGGER_NAME}.other").name == f"{TUI_LOGGER_NAME}.other"


class TestResetLogging:
    """Tests for handler cleanup."""

    def test_closes_verbose_handler(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        configure_tui_logging(verbose=True)
        handler = logging.getLogger().handlers[0]
        logging.getLogger(TUI_LOGGER_NAME).info("open the file")

        reset_logging()

        assert logging.getLogger(TUI_LOGGER_NAME).handlers == []
        assert logging.getLogger(SDK_LOGGER_NAME).handlers == []
        assert logging.getLogger(PY_WARNINGS_LOGGER_NAME).handlers == []
        assert handler not in logging.getLogger().handlers
        assert isinstance(handler, RotatingFileHandler)
        assert handler.stream is None


class TestConfigureLogging:
    """Tests for CLI startup logging."""

    def teardown_method(self) -> None:
        reset_logging()

    def test_silent_mode(self) -> None:
        configure_logging(verbose=False)
        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert logger.handlers == []
        assert logger.propagate is True
        assert isinstance(logging.getLogger().handlers[0], logging.NullHandler)

    def test_verbose_mode_creates_rotating_file(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        configure_logging(verbose=True)

        logger = logging.getLogger(TUI_LOGGER_NAME)
        assert logger.handlers == []
        assert logger.propagate is True
        assert isinstance(logging.getLogger().handlers[0], RotatingFileHandler)

        logger.debug("Test log message")
        for handler in logging.getLogger().handlers:
            handler.flush()
        assert "Test log message" in (tmp_path / LOG_FILE_NAME).read_text()

    def test_startup_warnings_are_routed_away_from_stderr(self) -> None:
        configure_logging(verbose=False)

        logger = logging.getLogger(PY_WARNINGS_LOGGER_NAME)
        assert logger.handlers == []
        assert logger.propagate is True
        assert isinstance(logging.getLogger().handlers[0], logging.NullHandler)
