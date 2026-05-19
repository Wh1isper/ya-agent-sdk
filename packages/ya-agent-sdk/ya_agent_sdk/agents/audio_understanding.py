"""Audio understanding agent for analyzing audio content.

This module provides an audio understanding agent that can analyze audio files
and return structured descriptions including speech transcription, music analysis,
sound effects, and environmental audio identification.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.messages import AudioUrl
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from ya_agent_sdk._config import AgentSettings
from ya_agent_sdk._logger import logger
from ya_agent_sdk.agents.models import infer_model
from ya_agent_sdk.presets import resolve_model_settings
from ya_agent_sdk.usage import coerce_run_usage

# =============================================================================
# Exceptions
# =============================================================================


class AudioError(Exception):
    """Base exception for audio processing errors."""

    pass


class AudioSizeError(AudioError):
    """Raised when audio content exceeds the maximum allowed size."""

    def __init__(self, size: int, max_size: int):
        self.size = size
        self.max_size = max_size
        size_mb = size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        super().__init__(f"Audio size ({size_mb:.2f}MB) exceeds maximum allowed size ({max_mb:.0f}MB)")


class AudioInputError(AudioError):
    """Raised when audio input is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class AudioAnalysisError(AudioError):
    """Raised when audio analysis fails."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.cause = cause
        super().__init__(message)


# =============================================================================
# Constants
# =============================================================================

# Default maximum audio content size for base64 encoding (20MB)
DEFAULT_MAX_AUDIO_SIZE = 20 * 1024 * 1024

# Mapping of file extensions to audio media types
AUDIO_MEDIA_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".opus": "audio/opus",
    ".webm": "audio/webm",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}

AGENT_NAME = "audio-understanding"

DEFAULT_AUDIO_ANALYSIS_INSTRUCTION = """Listen to this audio carefully and describe everything you hear in as much detail as possible.

Include:
- All spoken content: transcribe speech verbatim with speaker attribution
- Music: genre, instruments, tempo, mood, lyrics if present
- Sound effects: type, timing, purpose
- Environmental sounds: background noise, ambient audio
- The context, purpose, or intent behind the audio
- Any notable details, transitions, or key moments

Be thorough and comprehensive. The more detail, the better.
"""


# =============================================================================
# Utilities
# =============================================================================


def guess_media_type(source: str | Path) -> str:
    """Guess audio media type from URL or file path.

    Args:
        source: URL string or Path object

    Returns:
        Media type string, defaults to 'audio/mpeg' if unknown
    """
    if isinstance(source, Path):
        ext = source.suffix.lower()
    else:
        parsed = urlparse(source)
        path = parsed.path
        ext = Path(path).suffix.lower() if path else ""

    return AUDIO_MEDIA_TYPES.get(ext, "audio/mpeg")


def build_audio_content(
    audio_url: str | None = None,
    audio_data: bytes | None = None,
    media_type: str | None = None,
    max_size: int = DEFAULT_MAX_AUDIO_SIZE,
) -> AudioUrl | BinaryContent:
    """Build audio input content for AI model consumption.

    Args:
        audio_url: URL of the audio (mutually exclusive with audio_data)
        audio_data: Raw audio bytes (mutually exclusive with audio_url)
        media_type: Optional media type override
        max_size: Maximum allowed size for audio_data in bytes

    Returns:
        AudioUrl for remote audio or BinaryContent for binary data

    Raises:
        AudioInputError: If neither or both audio_url and audio_data are provided
        AudioSizeError: If audio_data exceeds max_size
    """
    if not audio_url and not audio_data:
        raise AudioInputError("Either audio_url or audio_data must be provided")

    if audio_url and audio_data:
        raise AudioInputError("Both audio_url and audio_data cannot be provided")

    if audio_url:
        logger.debug(f"Building audio content from URL: {audio_url}")
        return AudioUrl(url=audio_url, media_type=media_type or guess_media_type(audio_url))

    # audio_data is guaranteed to be not None here due to the checks above
    audio_bytes = audio_data  # type narrowing for pyright
    if audio_bytes is None:
        raise AudioInputError("audio_data is required when audio_url is not provided")

    if len(audio_bytes) > max_size:
        raise AudioSizeError(len(audio_bytes), max_size)

    logger.debug(f"Building audio content from binary data: {len(audio_bytes)} bytes")
    return BinaryContent(
        data=audio_bytes,
        media_type=media_type or "audio/mpeg",
    )


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "audio_understanding.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


# =============================================================================
# Models
# =============================================================================


class AudioDescription(BaseModel):
    """Minimal constraint - let the model freely describe the audio."""

    description: str = Field(description="Detailed, comprehensive description of everything in the audio")


# =============================================================================
# Agent Factory and API
# =============================================================================


def get_audio_understanding_agent(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent[None, AudioDescription]:
    """Create an audio understanding agent.

    Args:
        model: Model string or Model instance. If None, uses config setting.
        model_settings: Optional model settings dict.

    Returns:
        Agent configured for audio understanding.

    Raises:
        ValueError: If no model is specified and config has no default.
    """
    settings = AgentSettings()

    if model is None:
        if settings.audio_understanding_model:
            model = settings.audio_understanding_model
        else:
            raise ValueError("No model specified. Provide model parameter or set YA_AGENT_AUDIO_UNDERSTANDING_MODEL.")

    if model_settings is None and settings.audio_understanding_model_settings:
        model_settings = cast(ModelSettings, resolve_model_settings(settings.audio_understanding_model_settings))

    model_instance = infer_model(model) if isinstance(model, str) else model

    system_prompt = _load_system_prompt()

    return Agent[None, AudioDescription](
        model_instance,
        output_type=AudioDescription,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries={"tools": 3, "output": 3},
    )


async def get_audio_description(
    audio_url: str | None = None,
    audio_data: bytes | None = None,
    media_type: str | None = None,
    instruction: str | None = None,
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    max_audio_size: int = DEFAULT_MAX_AUDIO_SIZE,
    model_wrapper: Callable[[Model, str, dict[str, Any]], Model | Awaitable[Model]] | None = None,
    wrapper_metadata: dict[str, Any] | None = None,
) -> tuple[str, str, RunUsage]:
    """Analyze audio and get a structured description.

    Args:
        audio_url: URL of the audio to analyze.
        audio_data: Raw audio bytes to analyze.
        media_type: Optional media type override.
        instruction: Custom instruction for analysis. If None, uses default.
        model: Model string or Model instance.
        model_settings: Optional model settings dict.
        max_audio_size: Maximum allowed size for audio_data in bytes.
        model_wrapper: Optional wrapper for model instrumentation.
        wrapper_metadata: Context dict passed to model_wrapper (e.g., from ctx.get_wrapper_metadata()).

    Returns:
        Tuple of (description string, model_id, usage).

    Raises:
        AudioInputError: If audio input is invalid.
        AudioSizeError: If audio exceeds size limit.
        AudioAnalysisError: If analysis fails.
    """

    audio_content = build_audio_content(
        audio_url=audio_url,
        audio_data=audio_data,
        media_type=media_type,
        max_size=max_audio_size,
    )

    agent = get_audio_understanding_agent(model=model, model_settings=model_settings)

    # Apply model wrapper if configured
    if model_wrapper is not None:
        effective_context = wrapper_metadata or {}
        wrapped = model_wrapper(cast(Model, agent.model), AGENT_NAME, effective_context)
        agent.model = await wrapped if isawaitable(wrapped) else wrapped

    try:
        result = await agent.run(
            [
                instruction or DEFAULT_AUDIO_ANALYSIS_INSTRUCTION,
                audio_content,
            ],
        )
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise AudioAnalysisError(f"Failed to analyze audio: {e}", cause=e) from e

    # Get model_id from agent's model
    model_id = cast(Model, agent.model).model_name

    return result.output.description, model_id, coerce_run_usage(result.usage)
