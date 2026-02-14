"""Browser automation tools."""

from ya_agent_sdk.toolsets.browser_use.tools.dialog import (
    accept_dialog,
    dismiss_dialog,
    handle_dialog,
)
from ya_agent_sdk.toolsets.browser_use.tools.form import (
    check,
    select_option,
    uncheck,
    upload_file,
)
from ya_agent_sdk.toolsets.browser_use.tools.interaction import (
    click_element,
    execute_javascript,
    focus,
    hover,
    press_key,
    scroll_to,
    type_text,
)
from ya_agent_sdk.toolsets.browser_use.tools.navigation import (
    go_back,
    go_forward,
    navigate_to_url,
    reload_page,
)
from ya_agent_sdk.toolsets.browser_use.tools.query import (
    find_elements,
    get_element_attributes,
    get_element_text,
)
from ya_agent_sdk.toolsets.browser_use.tools.state import (
    get_page_content,
    get_page_info,
    get_viewport_info,
    take_element_screenshot,
    take_screenshot,
)
from ya_agent_sdk.toolsets.browser_use.tools.validation import (
    is_checked,
    is_enabled,
    is_visible,
)
from ya_agent_sdk.toolsets.browser_use.tools.wait import (
    wait_for_load_state,
    wait_for_navigation,
    wait_for_selector,
)

# Export all tools for registration
ALL_TOOLS = [
    # Navigation
    navigate_to_url,
    go_back,
    go_forward,
    reload_page,
    # State inspection
    get_page_info,
    get_page_content,
    take_screenshot,
    take_element_screenshot,
    get_viewport_info,
    # Interaction
    click_element,
    type_text,
    execute_javascript,
    scroll_to,
    hover,
    press_key,
    focus,
    # Query
    find_elements,
    get_element_text,
    get_element_attributes,
    # Wait
    wait_for_selector,
    wait_for_navigation,
    wait_for_load_state,
    # Form
    select_option,
    check,
    uncheck,
    upload_file,
    # Dialog
    handle_dialog,
    accept_dialog,
    dismiss_dialog,
    # Validation
    is_visible,
    is_enabled,
    is_checked,
]

__all__ = [
    "ALL_TOOLS",
    "accept_dialog",
    "check",
    "click_element",
    "dismiss_dialog",
    "execute_javascript",
    "find_elements",
    "focus",
    "get_element_attributes",
    "get_element_text",
    "get_page_content",
    "get_page_info",
    "get_viewport_info",
    "go_back",
    "go_forward",
    "handle_dialog",
    "hover",
    "is_checked",
    "is_enabled",
    "is_visible",
    "navigate_to_url",
    "press_key",
    "reload_page",
    "scroll_to",
    "select_option",
    "take_element_screenshot",
    "take_screenshot",
    "type_text",
    "uncheck",
    "upload_file",
    "wait_for_load_state",
    "wait_for_navigation",
    "wait_for_selector",
]
