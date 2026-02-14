"""Type definitions for tool return values."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class NavigationResult(BaseModel):
    """Navigation operation result."""

    status: Literal["success", "error", "timeout"]
    url: str
    title: str = ""
    error_message: str | None = None


# State inspection results
class PageInfo(BaseModel):
    """Current page information."""

    url: str
    title: str
    ready_state: str
    viewport: dict[str, int]


class ScreenshotResult(BaseModel):
    """Screenshot operation result."""

    status: Literal["success", "error"]
    url: str
    segments_count: int
    truncated: bool = False
    error_message: str | None = None
    format: str = "png"
    full_page: bool = False


class ElementScreenshotResult(BaseModel):
    """Element screenshot result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    segments_count: int
    element_info: dict[str, Any] | None = None
    error_message: str | None = None


# Interaction results
class ClickResult(BaseModel):
    """Click operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    element_info: dict[str, Any] | None = None
    error_message: str | None = None


class TypeTextResult(BaseModel):
    """Type text operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    text: str
    error_message: str | None = None


class ExecuteScriptResult(BaseModel):
    """Execute JavaScript result."""

    status: Literal["success", "error"]
    result: Any = None
    error_message: str | None = None


# Query results
class ElementInfo(BaseModel):
    """Element information."""

    selector: str
    tag_name: str
    text: str
    attributes: dict[str, str]
    bounding_box: dict[str, float] | None = None


class FindElementsResult(BaseModel):
    """Find elements result."""

    status: Literal["success", "error"]
    selector: str
    count: int
    elements: list[ElementInfo] = []
    error_message: str | None = None


class WaitResult(BaseModel):
    """Wait operation result."""

    status: Literal["success", "timeout", "error"]
    wait_type: str
    selector: str | None = None
    error_message: str | None = None
    elapsed_time: float | None = None


class SelectOptionResult(BaseModel):
    """Select option operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    value: str | None = None
    index: int | None = None
    label: str | None = None
    error_message: str | None = None


class CheckboxResult(BaseModel):
    """Checkbox operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    checked: bool
    error_message: str | None = None


class FileUploadResult(BaseModel):
    """File upload operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    files: list[str]
    error_message: str | None = None


class DialogResult(BaseModel):
    """Dialog handling result."""

    status: Literal["success", "error", "no_dialog"]
    dialog_type: str | None = None
    message: str | None = None
    accepted: bool | None = None
    prompt_text: str | None = None
    error_message: str | None = None


class HoverResult(BaseModel):
    """Hover operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    element_info: dict[str, Any] | None = None
    error_message: str | None = None


class KeyPressResult(BaseModel):
    """Key press operation result."""

    status: Literal["success", "error"]
    key: str
    error_message: str | None = None


class FocusResult(BaseModel):
    """Focus operation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    error_message: str | None = None


class ValidationResult(BaseModel):
    """Element validation result."""

    status: Literal["success", "error", "not_found"]
    selector: str
    result: bool | None = None
    error_message: str | None = None
