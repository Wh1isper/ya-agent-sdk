"""Reusable prompt-toolkit shell for YAACLI's startup and runtime phases."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from prompt_toolkit import Application
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition, FilterOrBool
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.output.base import Output
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.selection import SelectionState, SelectionType
from prompt_toolkit.styles import BaseStyle
from prompt_toolkit.widgets import Box, Frame, TextArea


@dataclass(frozen=True, slots=True)
class ComposeSnapshot:
    """Serializable compose state transferred across the startup handoff."""

    text: str = ""
    cursor_position: int = 0
    selection_original_cursor_position: int | None = None
    selection_type: str | None = None
    submit_when_ready: bool = False
    input_mode: str = "send"
    mouse_enabled: bool = True

    @classmethod
    def capture(
        cls,
        input_area: TextArea,
        *,
        submit_when_ready: bool = False,
        input_mode: str = "send",
        mouse_enabled: bool = True,
    ) -> ComposeSnapshot:
        """Capture the draft without claiming that it has been submitted."""
        if input_mode not in {"send", "edit"}:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        selection = input_area.buffer.selection_state
        return cls(
            text=input_area.buffer.text,
            cursor_position=input_area.buffer.cursor_position,
            selection_original_cursor_position=(selection.original_cursor_position if selection is not None else None),
            selection_type=(selection.type.value if selection is not None else None),
            submit_when_ready=submit_when_ready,
            input_mode=input_mode,
            mouse_enabled=mouse_enabled,
        )

    @classmethod
    def from_payload(cls, payload: object) -> ComposeSnapshot:
        """Validate one untrusted JSON payload at the process boundary."""
        if not isinstance(payload, dict):
            raise TypeError("Compose snapshot must be a JSON object")
        allowed_fields = {
            "text",
            "cursor_position",
            "selection_original_cursor_position",
            "selection_type",
            "submit_when_ready",
            "input_mode",
            "mouse_enabled",
        }
        unknown_fields = set(payload) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown compose snapshot fields: {sorted(unknown_fields)}")

        text = payload.get("text", "")
        cursor_position = payload.get("cursor_position", 0)
        selection_position = payload.get("selection_original_cursor_position")
        selection_type = payload.get("selection_type")
        submit_when_ready = payload.get("submit_when_ready", False)
        input_mode = payload.get("input_mode", "send")
        mouse_enabled = payload.get("mouse_enabled", True)
        if not isinstance(text, str):
            raise TypeError("Compose snapshot text must be a string")
        if not isinstance(cursor_position, int) or isinstance(cursor_position, bool):
            raise TypeError("Compose snapshot cursor position must be an integer")
        if selection_position is not None and (
            not isinstance(selection_position, int) or isinstance(selection_position, bool)
        ):
            raise TypeError("Compose snapshot selection position must be an integer or null")
        if selection_type is not None and not isinstance(selection_type, str):
            raise TypeError("Compose snapshot selection type must be a string or null")
        if selection_type is not None and selection_type not in {item.value for item in SelectionType}:
            raise ValueError(f"Unsupported selection type: {selection_type}")
        if not isinstance(submit_when_ready, bool):
            raise TypeError("Compose snapshot submit state must be a boolean")
        if input_mode not in {"send", "edit"}:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        if not isinstance(mouse_enabled, bool):
            raise TypeError("Compose snapshot mouse state must be a boolean")
        return cls(
            text=text,
            cursor_position=cursor_position,
            selection_original_cursor_position=selection_position,
            selection_type=selection_type,
            submit_when_ready=submit_when_ready,
            input_mode=input_mode,
            mouse_enabled=mouse_enabled,
        )

    def restore(self, input_area: TextArea) -> None:
        """Restore this snapshot into an empty runtime compose buffer."""
        input_area.buffer.text = self.text
        input_area.buffer.cursor_position = min(max(0, self.cursor_position), len(self.text))
        if self.selection_original_cursor_position is None or self.selection_type is None:
            input_area.buffer.selection_state = None
            return
        try:
            selection_type = SelectionType(self.selection_type)
        except ValueError:
            input_area.buffer.selection_state = None
            return
        input_area.buffer.selection_state = SelectionState(
            original_cursor_position=min(max(0, self.selection_original_cursor_position), len(self.text)),
            type=selection_type,
        )


@dataclass(frozen=True, slots=True)
class TUIShellCallbacks:
    """Rendering and interaction callbacks shared by both shell controllers."""

    output_text: Callable[[], Any]
    scroll_output: Callable[[int], None]
    task_text: Callable[[], Any]
    task_height: Callable[[], int]
    has_tasks: Callable[[], bool]
    model_selector_text: Callable[[], Any]
    model_selector_height: Callable[[], int]
    model_selector_open: Callable[[], bool]
    session_selector_text: Callable[[], Any]
    session_selector_height: Callable[[], int]
    session_selector_width: Callable[[], int]
    session_selector_title: Callable[[], Any]
    session_selector_open: Callable[[], bool]
    status_text: Callable[[], Any]
    status_height: Callable[[], int]
    prompt: Callable[[], Any]
    terminal_height: Callable[[], int]
    input_key_bindings: Callable[[TextArea], KeyBindings]
    application_key_bindings: Callable[[TextArea], KeyBindings]
    command_words: Callable[[], Iterable[str]] = lambda: ()
    session_words: Callable[[], Iterable[str]] = lambda: ()
    skill_words: Callable[[], Iterable[str]] = lambda: ()


@dataclass(slots=True)
class TUIShell:
    """One concrete prompt-toolkit application and its transferable compose state."""

    application: Application[None]
    input_area: TextArea
    output_window: Window

    def capture_compose(
        self,
        *,
        submit_when_ready: bool = False,
        input_mode: str = "send",
        mouse_enabled: bool = True,
    ) -> ComposeSnapshot:
        return ComposeSnapshot.capture(
            self.input_area,
            submit_when_ready=submit_when_ready,
            input_mode=input_mode,
            mouse_enabled=mouse_enabled,
        )


class LeasedVt100Output(Vt100_Output):
    """VT100 output whose alternate-screen ownership can cross one process handoff."""

    suppress_enter_alternate_screen: bool = False
    suppress_quit_alternate_screen: bool = False
    alternate_screen_active: bool = False

    def enter_alternate_screen(self) -> None:
        if self.suppress_enter_alternate_screen or self.alternate_screen_active:
            return
        super().enter_alternate_screen()
        self.alternate_screen_active = True

    def quit_alternate_screen(self) -> None:
        if self.suppress_quit_alternate_screen or not self.alternate_screen_active:
            return
        super().quit_alternate_screen()
        self.alternate_screen_active = False


def create_leased_output(*, enter: bool, leave: bool) -> LeasedVt100Output:
    """Create terminal output with explicit alternate-screen lease behavior."""
    if os.name != "posix" or not sys.stdout.isatty():
        raise RuntimeError("Alternate-screen handoff requires a POSIX TTY")
    output = cast(LeasedVt100Output, LeasedVt100Output.from_pty(sys.stdout, term=os.environ.get("TERM")))
    output.suppress_enter_alternate_screen = not enter
    output.suppress_quit_alternate_screen = not leave
    output.alternate_screen_active = not enter
    return output


def build_tui_shell(
    callbacks: TUIShellCallbacks,
    *,
    style: BaseStyle,
    completer: Completer | None = None,
    output: Output | None = None,
    input_source: Input | None = None,
    input_read_only: FilterOrBool = False,
    mouse_support: FilterOrBool = True,
) -> TUIShell:
    """Build the canonical YAACLI layout without importing the agent runtime."""

    class ScrollableFormattedTextControl(FormattedTextControl):
        def mouse_handler(self, mouse_event: MouseEvent) -> object:
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                callbacks.scroll_output(-3)
                application.invalidate()
                return None
            if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                callbacks.scroll_output(3)
                application.invalidate()
                return None
            return super().mouse_handler(mouse_event)

    class ScrollableBufferControl(BufferControl):
        def mouse_handler(self, mouse_event: MouseEvent) -> object:
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                if self.buffer.document.cursor_position_row > 0:
                    self.buffer.cursor_up()
                return None
            if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                document = self.buffer.document
                if document.cursor_position_row < document.line_count - 1:
                    self.buffer.cursor_down()
                return None
            return super().mouse_handler(mouse_event)

    output_window = Window(
        content=ScrollableFormattedTextControl(callbacks.output_text),
        wrap_lines=False,
    )
    task_window = ConditionalContainer(
        Window(
            content=FormattedTextControl(callbacks.task_text),
            height=callbacks.task_height,
            style="class:task-pane",
            wrap_lines=False,
        ),
        filter=Condition(callbacks.has_tasks),
    )
    model_selector_window = ConditionalContainer(
        Window(
            content=FormattedTextControl(callbacks.model_selector_text),
            height=callbacks.model_selector_height,
            style="class:model-selector",
            wrap_lines=False,
        ),
        filter=Condition(callbacks.model_selector_open),
    )
    session_selector_body = Box(
        Window(
            content=FormattedTextControl(callbacks.session_selector_text),
            height=callbacks.session_selector_height,
            style="class:session-selector",
            wrap_lines=False,
        ),
        padding_left=1,
        padding_right=1,
        style="class:session-selector",
    )
    session_selector_window = ConditionalContainer(
        Frame(
            session_selector_body,
            title=callbacks.session_selector_title,
            style="class:session-selector.frame",
        ),
        filter=Condition(callbacks.session_selector_open),
    )
    status_bar = Window(
        content=FormattedTextControl(callbacks.status_text),
        height=callbacks.status_height,
        style="class:status-bar",
        wrap_lines=True,
    )
    input_area = TextArea(
        multiline=True,
        prompt=callbacks.prompt,
        style="class:input-area",
        focusable=True,
        height=lambda: 3 if callbacks.terminal_height() < 28 else 5,
        scrollbar=True,
        completer=completer,
        complete_while_typing=completer is not None,
        read_only=input_read_only,
    )
    original_control = input_area.control
    scrollable_control = ScrollableBufferControl(
        buffer=original_control.buffer,
        input_processors=original_control.input_processors,
        include_default_input_processors=False,
        lexer=original_control.lexer,
        focus_on_click=original_control.focus_on_click,
        key_bindings=callbacks.input_key_bindings(input_area),
    )
    input_area.window.content = scrollable_control
    input_area.control = scrollable_control

    body = HSplit([output_window, task_window, status_bar, input_area])
    root = FloatContainer(
        content=body,
        floats=[
            Float(top=1, left=2, right=2, content=model_selector_window),
            Float(top=1, width=callbacks.session_selector_width, content=session_selector_window),
            Float(
                xcursor=True,
                ycursor=True,
                content=CompletionsMenu(max_height=8, scroll_offset=1, display_arrows=True),
            ),
        ],
    )
    application = Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=callbacks.application_key_bindings(input_area),
        style=style,
        full_screen=True,
        mouse_support=mouse_support,
        refresh_interval=1.0,
        output=output,
        input=input_source,
    )
    return TUIShell(application=application, input_area=input_area, output_window=output_window)
