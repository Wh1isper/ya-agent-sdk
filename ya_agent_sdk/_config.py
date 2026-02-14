"""Configuration management using pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Configuration for agents with environment variable support.

    All settings can be overridden via environment variables with the prefix YA_AGENT_.
    For example, to set working_dir, use YA_AGENT_WORKING_DIR=/path/to/dir.
    """

    model_config = SettingsConfigDict(
        env_prefix="YA_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    image_understanding_model: str | None = None
    """Model to use for image understanding when native vision is unavailable."""

    video_understanding_model: str | None = None
    """Model to use for video understanding when native capability is unavailable."""

    compact_model: str | None = None
    """Model to use for compact when native capability is unavailable."""
