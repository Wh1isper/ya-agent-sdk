"""Model profile resolution and persistence for YAACLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from ya_agent_sdk.context import ModelCapability, ModelConfig
from ya_agent_sdk.presets import resolve_model_cfg

from yaacli.config import YaacliConfig
from yaacli.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PROFILE_ID = "default"
STATE_FILE_NAME = "state.json"


class ResolvedModelProfile(BaseModel):
    """Runtime-ready model profile."""

    id: str
    label: str
    model: str
    model_settings: str | dict[str, Any] | None = None
    model_cfg: str | dict[str, Any] | None = None
    is_default: bool = False


class ModelProfileState(BaseModel):
    """Persisted model profile UI state."""

    selected_profile_id: str | None = None


class YaacliState(BaseModel):
    """YAACLI local state stored outside config.toml."""

    model_profile: ModelProfileState = Field(default_factory=ModelProfileState)


def get_state_file(config_dir: Path) -> Path:
    """Return the YAACLI state file path."""
    return config_dir / STATE_FILE_NAME


def load_state(config_dir: Path) -> YaacliState:
    """Load local state from the global config directory."""
    state_file = get_state_file(config_dir)
    if not state_file.exists():
        return YaacliState()

    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return YaacliState.model_validate(data)
    except Exception:
        logger.debug("Failed to load YAACLI state from %s", state_file, exc_info=True)

    return YaacliState()


def save_state(config_dir: Path, state: YaacliState) -> None:
    """Persist local state to the global config directory."""
    config_dir.mkdir(parents=True, exist_ok=True)
    state_file = get_state_file(config_dir)
    state_file.write_text(
        json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_selected_model_profile_id(config_dir: Path, profile_id: str) -> None:
    """Persist the last selected model profile id."""
    state = load_state(config_dir)
    state.model_profile.selected_profile_id = profile_id
    save_state(config_dir, state)


def build_model_profiles(config: YaacliConfig) -> list[ResolvedModelProfile]:
    """Build selectable model profiles from config.

    The first profile is always the startup default from [general] when configured.
    Additional profiles come from [model_profiles.*].
    """
    profiles: list[ResolvedModelProfile] = []

    if config.general.model:
        profiles.append(
            ResolvedModelProfile(
                id=DEFAULT_MODEL_PROFILE_ID,
                label="Default",
                model=config.general.model,
                model_settings=config.general.model_settings,
                model_cfg=config.general.model_cfg,
                is_default=True,
            )
        )

    for profile_id, profile in config.model_profiles.items():
        label = profile.label or profile_id
        profiles.append(
            ResolvedModelProfile(
                id=profile_id,
                label=label,
                model=profile.model,
                model_settings=profile.model_settings,
                model_cfg=profile.model_cfg,
            )
        )

    return profiles


def get_model_profile(config: YaacliConfig, profile_id: str) -> ResolvedModelProfile | None:
    """Find a resolved model profile by id."""
    for profile in build_model_profiles(config):
        if profile.id == profile_id:
            return profile
    return None


def get_startup_model_profile(config: YaacliConfig, config_dir: Path) -> ResolvedModelProfile | None:
    """Return the model profile to use at startup.

    The persisted selection wins when it still exists in config. The [general]
    profile remains the fallback startup default.
    """
    profiles = build_model_profiles(config)
    if not profiles:
        return None

    state = load_state(config_dir)
    selected_id = state.model_profile.selected_profile_id
    if selected_id:
        for profile in profiles:
            if profile.id == selected_id:
                return profile

    return profiles[0]


def format_model_profile_label(profile: ResolvedModelProfile) -> str:
    """Format a compact profile label for status/output."""
    return f"{profile.label} ({profile.model})"


def resolve_profile_model_cfg(model_cfg_input: str | dict[str, Any] | None) -> ModelConfig:
    """Resolve a profile model_cfg into ModelConfig."""
    if model_cfg_input is None:
        return ModelConfig()

    cfg_dict = resolve_model_cfg(model_cfg_input)
    if cfg_dict is None:
        return ModelConfig()

    if "capabilities" in cfg_dict:
        caps = cfg_dict["capabilities"]
        if isinstance(caps, (list, set)):
            cfg_dict["capabilities"] = {ModelCapability(c) if isinstance(c, str) else c for c in caps}

    return ModelConfig(**cfg_dict)


def format_model_profile_choice(profile: ResolvedModelProfile) -> str:
    """Format a profile choice for prompt_toolkit dialogs."""
    suffix = ""
    details: list[str] = []
    if profile.model_settings:
        details.append(f"settings={profile.model_settings}")
    if profile.model_cfg:
        details.append(f"cfg={profile.model_cfg}")
    if details:
        suffix = f"  [{', '.join(details)}]"
    return f"{profile.label}: {profile.model}{suffix}"
