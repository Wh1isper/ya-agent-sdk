"""Pydantic AI toolset wrapper implementing CodeAct."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import keyword
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Annotated, Any, Literal, NotRequired, Self, cast

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_ai import (
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    FunctionToolset,
    ModelRetry,
    RunContext,
    Tool,
    ToolDefinition,
    ToolFailed,
    ToolReturn,
    UserError,
    WrapperToolset,
)
from pydantic_ai.exceptions import ToolRetryError
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import (
    BinaryContent,
    InstructionPart,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnContent,
    UserContent,
    is_multi_modal_content,
)
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolApproved, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_monty import (
    MontyConversionError,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
    MontyTypingError,
)
from typing_extensions import TypedDict
from ya_agent_environment.exceptions import EnvironmentError as AgentEnvironmentError
from ya_agent_environment.output import truncate_utf8_head_tail

from ya_agent_sdk.codeact.config import CodeActConfig
from ya_agent_sdk.codeact.executor import is_sandbox_panic
from ya_agent_sdk.codeact.programs import (
    extract_persistent_bound_names,
    load_program_source,
    program_inputs,
    validate_static_tool_references,
)
from ya_agent_sdk.codeact.runtime import CodeActExecution, CodeActRunState
from ya_agent_sdk.context import AgentContext

_RUN_CODE = "run_code"
_RUN_PROGRAM = "run_program"
_RESERVED_TOOL_NAMES = frozenset({_RUN_CODE, _RUN_PROGRAM})
_INVALID_IDENT_CHARS = re.compile(r"[^a-zA-Z0-9_]")
_SENSITIVE_KEY = re.compile(r"(?:password|passwd|secret|token|api[_-]?key|authorization|cookie)", re.I)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:password|passwd|secret|token|api[_-]?key|authorization|cookie)\b[\"']?\s*[:=]\s*)"
    r"(?:\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*'|[^\s,;}]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIALS = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@")
_SECRET_HEADER = re.compile(r"(?im)(\b(?:(?:proxy-)?authorization|(?:set-)?cookie)\s*:\s*)[^\r\n]+")
_TOOL_RETURN_CONTENT = TypeAdapter(ToolReturnContent)
_CODEACT_CONTRACT_VERSION = 1

_DESCRIPTION = """Write and run Python in a restricted Monty sandbox.

The sandbox provides a Python subset, has no ambient filesystem, network, process,
environment, credential, or clock access, and cannot install third-party packages.
Use the functions listed below to interact with the host. Function arguments are
keyword-only. Async functions must be awaited; independent async calls may be
combined with `await asyncio.gather(...)`.

The final expression is returned. State is retained between run_code calls in this
agent run. Set restart=true to discard that state. CodeAct does not make tool calls
transactional, retry-safe, or reversible."""

_PROGRAM_DESCRIPTION = """Execute a reviewed CodeAct Python program from the current Environment workspace.

The file must be strict UTF-8, end in .codeact.py, and define exactly
`async def main(inputs)`. Each invocation uses a fresh Monty session. Source is read
through the current Environment FileOperator; host effects remain available only
through injected CodeAct-eligible tools."""


class _TraceValueDescriptor(TypedDict):
    bytes: int
    sha256: str
    preview: str
    truncated: bool


class _ToolCallRecord(TypedDict):
    call_id: str
    sandbox_name: str
    tool_name: str
    outcome: Literal["running", "completed", "denied", "deferred", "cancelled", "failed"]
    args_bytes: int
    args_sha256: str
    args_preview: str
    args_truncated: bool
    result_bytes: NotRequired[int]
    result_sha256: NotRequired[str]
    result_preview: NotRequired[str]
    result_truncated: NotRequired[bool]
    error: NotRequired[str]
    duration_ms: NotRequired[int]
    tool_return_metadata_omitted: NotRequired[bool]


@dataclass
class _ExecutionBudget:
    config: CodeActConfig
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    calls_reserved: int = 0
    calls_started: int = 0
    nested_result_bytes: int = 0
    supplemental_content: list[tuple[int, list[UserContent]]] = field(default_factory=list)
    records: list[_ToolCallRecord] = field(default_factory=list)

    async def reserve(self) -> int:
        async with self.lock:
            if self.calls_reserved >= self.config.max_tool_calls:
                raise RuntimeError(f"CodeAct nested tool-call limit ({self.config.max_tool_calls}) exceeded")
            self.calls_reserved += 1
            return self.calls_reserved

    async def mark_started(self, record: _ToolCallRecord) -> None:
        async with self.lock:
            self.calls_started += 1
            self.records.append(record)

    def _reserve_nested_result_locked(self, size: int) -> None:
        if size > self.config.max_output_bytes:
            raise RuntimeError(f"CodeAct nested tool result exceeds max_output_bytes={self.config.max_output_bytes}")
        if self.nested_result_bytes + size > self.config.max_output_bytes:
            raise RuntimeError(
                f"CodeAct cumulative nested tool results exceed max_output_bytes={self.config.max_output_bytes}"
            )
        self.nested_result_bytes += size

    async def reserve_nested_result(self, size: int) -> None:
        async with self.lock:
            self._reserve_nested_result_locked(size)

    async def reserve_supplemental_content(
        self,
        ordinal: int,
        content: list[UserContent],
        size: int,
    ) -> None:
        async with self.lock:
            self._reserve_nested_result_locked(size)
            self.supplemental_content.append((ordinal, content))

    def ordered_supplemental_content(self) -> list[UserContent]:
        return [
            item for _, content in sorted(self.supplemental_content, key=lambda entry: entry[0]) for item in content
        ]


@dataclass
class _Catalog:
    definitions: dict[str, ToolDefinition]
    sanitized_to_original: dict[str, str]
    wrapped_tools: dict[str, ToolsetTool[AgentContext]]
    fingerprint: str


@dataclass
class CodeActToolset(WrapperToolset[AgentContext]):
    """Augment a final agent toolset with ``run_code`` and ``run_program``."""

    config: CodeActConfig
    _run_state: CodeActRunState | None = field(default=None, init=False, repr=False, compare=False)
    _last_catalog: str = field(default="", init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentContext]) -> AbstractToolset[AgentContext]:
        wrapped = await self.wrapped.for_run(ctx)
        return replace(self, wrapped=wrapped)

    async def for_run_step(self, ctx: RunContext[AgentContext]) -> AbstractToolset[AgentContext]:
        wrapped = await self.wrapped.for_run_step(ctx)
        if wrapped is self.wrapped:
            return self
        result = replace(self, wrapped=wrapped)
        result._run_state = self._run_state
        result._last_catalog = self._last_catalog
        return result

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        self._run_state = CodeActRunState(self.config)
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        state = self._run_state
        self._run_state = None
        try:
            if state is not None:
                await state.close()
        finally:
            wrapped_result = await self.wrapped.__aexit__(*args)
        return wrapped_result

    async def get_instructions(
        self, ctx: RunContext[AgentContext]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        upstream = await self.wrapped.get_instructions(ctx)
        if not self._last_catalog:
            return upstream
        instruction = InstructionPart(content=self._last_catalog, dynamic=True)
        if upstream is None:
            return instruction
        if isinstance(upstream, (str, InstructionPart)):
            return [upstream, instruction]
        return [*upstream, instruction]

    async def get_tools(self, ctx: RunContext[AgentContext]) -> dict[str, ToolsetTool[AgentContext]]:
        wrapped_tools = cast(dict[str, ToolsetTool[AgentContext]], await self.wrapped.get_tools(ctx))
        conflicts = _RESERVED_TOOL_NAMES.intersection(wrapped_tools)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise UserError(f"CodeAct reserved tool name conflict: {names}")

        catalog = _build_catalog(wrapped_tools)
        self._last_catalog = _render_catalog(catalog.definitions)
        tools: list[Tool[AgentContext]] = []
        if self.config.inline:
            tools.append(
                Tool(
                    self._run_code,
                    name=_RUN_CODE,
                    description=_DESCRIPTION + ("\n\n" + self._last_catalog if self._last_catalog else ""),
                    sequential=True,
                    metadata={"code_arg_name": "code", "code_arg_language": "python"},
                )
            )
        if self.config.programs and ctx.deps.file_operator is not None:
            tools.append(
                Tool(
                    self._run_program,
                    name=_RUN_PROGRAM,
                    description=_PROGRAM_DESCRIPTION,
                    sequential=True,
                    metadata={"codeact_program_runner": True},
                )
            )
        own = await FunctionToolset(tools, id="codeact").get_tools(ctx)
        return {**wrapped_tools, **cast(dict[str, ToolsetTool[AgentContext]], own)}

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentContext],
        tool: ToolsetTool[AgentContext],
    ) -> Any:
        if name == _RUN_CODE:
            return await self._run_code(ctx, **tool_args)
        if name == _RUN_PROGRAM:
            return await self._run_program(ctx, **tool_args)
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)

    async def _run_code(
        self,
        ctx: RunContext[AgentContext],
        code: Annotated[str, Field(description="Python source to execute in the restricted sandbox.")],
        restart: Annotated[bool, Field(description="Reset run-local REPL state before executing.")] = False,
    ) -> Any:
        raw = code.encode("utf-8")
        if len(raw) > self.config.max_source_bytes:
            raise ModelRetry(f"Code exceeds max_source_bytes={self.config.max_source_bytes}")
        return await self._execute_code(
            ctx,
            code,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            source_path=None,
            restart=restart,
        )

    async def _run_program(
        self,
        ctx: RunContext[AgentContext],
        path: Annotated[str, Field(description="Environment FileOperator path to a .codeact.py program.")],
        inputs: Annotated[
            dict[str, Any] | None,
            Field(description="JSON-compatible values passed to main(inputs)."),
        ] = None,
    ) -> Any:
        try:
            program = await load_program_source(
                ctx,
                path,
                max_source_bytes=self.config.max_source_bytes,
            )
        except (AgentEnvironmentError, ValueError, OSError, RuntimeError) as exc:
            message = _preview_text(_redact_text(str(exc)), self.config.trace_preview_bytes)
            raise ModelRetry(message) from exc
        return await self._execute_code(
            ctx,
            program.executable_source,
            source_sha256=program.source_sha256,
            source_path=program.path,
            restart=False,
            inputs=program_inputs(inputs),
        )

    async def _execute_code(  # noqa: C901
        self,
        ctx: RunContext[AgentContext],
        code: str,
        *,
        source_sha256: str,
        source_path: str | None,
        restart: bool,
        inputs: dict[str, Any] | None = None,
    ) -> Any:
        state = self._run_state
        if state is None:
            raise RuntimeError("CodeAct toolset must be entered before execution")
        parent_manager = ctx.tool_manager
        if parent_manager is None or parent_manager.tools is None:
            raise RuntimeError("CodeAct requires an active Pydantic AI ToolManager")

        current_tools = {
            name: cast(ToolsetTool[AgentContext], tool)
            for name, tool in parent_manager.tools.items()
            if name not in _RESERVED_TOOL_NAMES
        }
        catalog = _build_catalog(current_tools)
        if source_path is not None:
            try:
                validate_static_tool_references(
                    code,
                    valid_tool_names=set(catalog.definitions),
                    functions_see_complete_module=True,
                )
            except ValueError as exc:
                message = _preview_text(_redact_text(str(exc)), self.config.trace_preview_bytes)
                raise ModelRetry(message) from exc

        def preflight_inline(known_names: set[str]) -> set[str]:
            try:
                validate_static_tool_references(
                    code,
                    valid_tool_names=set(catalog.definitions),
                    known_names=known_names,
                )
            except ValueError as exc:
                message = _preview_text(_redact_text(str(exc)), self.config.trace_preview_bytes)
                raise ModelRetry(message) from exc
            return extract_persistent_bound_names(code)

        nested_manager = ToolManager(
            toolset=parent_manager.toolset,
            root_capability=parent_manager.root_capability,
            ctx=ctx,
            tools=catalog.wrapped_tools,
            default_max_retries=parent_manager.default_max_retries,
        )
        budget = _ExecutionBudget(self.config)

        async def dispatch(sandbox_name: str, kwargs: dict[str, Any]) -> Any:  # noqa: C901
            ordinal = await budget.reserve()
            original_name = catalog.sanitized_to_original.get(sandbox_name, sandbox_name)
            try:
                _bounded_json_size(kwargs, self.config.max_output_bytes)
            except _JsonSizeLimitExceeded as exc:
                raise RuntimeError(
                    f"CodeAct nested tool arguments exceed max_output_bytes={self.config.max_output_bytes}"
                ) from exc
            args_bytes = _redacted_json_bytes(kwargs)
            call_id = f"{ctx.tool_call_id or 'codeact'}__{ordinal}"
            call = ToolCallPart(tool_name=original_name, args=kwargs, tool_call_id=call_id)
            try:
                validated = await nested_manager.validate_tool_call(call, wrap_validation_errors=False)
            except ValidationError as exc:
                raise ModelRetry(_format_validation_error(exc)) from exc
            args_descriptor = _trace_value_descriptor(args_bytes, self.config.trace_preview_bytes)
            record = _ToolCallRecord(
                call_id=call_id,
                sandbox_name=sandbox_name,
                tool_name=original_name,
                outcome="running",
                args_bytes=args_descriptor["bytes"],
                args_sha256=args_descriptor["sha256"],
                args_preview=args_descriptor["preview"],
                args_truncated=args_descriptor["truncated"],
            )
            await budget.mark_started(record)
            started = time.monotonic()
            try:
                result = await _execute_validated_call(nested_manager, validated, call)
                if isinstance(result, ToolDenied):
                    record["outcome"] = "denied"
                    try:
                        _bounded_json_size(result.message, self.config.max_output_bytes)
                    except _JsonSizeLimitExceeded as exc:
                        raise RuntimeError(
                            f"CodeAct nested tool denial exceeds max_output_bytes={self.config.max_output_bytes}"
                        ) from exc
                    denied = _redacted_json_bytes(result.message)
                    _set_result_descriptor(record, denied, self.config.trace_preview_bytes)
                    denied_message = _preview_text(
                        _redact_text(str(result.message)),
                        self.config.trace_preview_bytes,
                    )
                    raise PermissionError(f"Tool {original_name!r} call denied: {denied_message}")
                result = await _unwrap_tool_return(
                    result,
                    ordinal=ordinal,
                    budget=budget,
                    record=record,
                )
                try:
                    _bounded_json_size(result, self.config.max_output_bytes)
                except _JsonSizeLimitExceeded as exc:
                    raise RuntimeError(
                        f"CodeAct nested tool result exceeds max_output_bytes={self.config.max_output_bytes}"
                    ) from exc
                plain = _TOOL_RETURN_CONTENT.dump_python(result, mode="json")
                try:
                    result_size = _bounded_json_size(plain, self.config.max_output_bytes)
                except _JsonSizeLimitExceeded as exc:
                    raise RuntimeError(
                        f"CodeAct nested tool result exceeds max_output_bytes={self.config.max_output_bytes}"
                    ) from exc
                await budget.reserve_nested_result(result_size)
                result_bytes = _redacted_json_bytes(plain)
                record["outcome"] = "completed"
                _set_result_descriptor(record, result_bytes, self.config.trace_preview_bytes)
                record["result_bytes"] = result_size
                return plain
            except (ApprovalRequired, CallDeferred) as exc:
                record["outcome"] = "deferred"
                record["error"] = _preview_text(type(exc).__name__, self.config.trace_preview_bytes)
                raise RuntimeError(
                    f"Tool {original_name!r} requires deferred host interaction that was not resolved inline"
                ) from exc
            except BaseException as exc:
                if record["outcome"] == "running":
                    record["outcome"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                    if isinstance(exc, ValidationError):
                        error = _format_validation_error(exc)
                    else:
                        error = _redact_text(f"{type(exc).__name__}: {exc}")
                    record["error"] = _preview_text(error, self.config.trace_preview_bytes)
                if isinstance(exc, ValidationError):
                    # Monty renders external exception strings into its runtime
                    # traceback, so sanitize before crossing that boundary.
                    raise TypeError(_format_validation_error(exc)) from None
                raise
            finally:
                record["duration_ms"] = max(0, round((time.monotonic() - started) * 1000))

        sequential_names = {name for name, definition in catalog.definitions.items() if definition.sequential}
        global_sequential = parent_manager.get_parallel_execution_mode() == "sequential"
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                if source_path is None:
                    execution = await state.execute_inline(
                        code,
                        dispatch=dispatch,
                        valid_names=set(catalog.definitions),
                        sequential_names=sequential_names,
                        global_sequential=global_sequential,
                        restart=restart,
                        preflight=preflight_inline,
                    )
                else:
                    execution = await state.execute_program(
                        code,
                        script_name=source_path,
                        inputs=inputs or {},
                        dispatch=dispatch,
                        valid_names=set(catalog.definitions),
                        sequential_names=sequential_names,
                        global_sequential=global_sequential,
                    )
            return self._success_result(
                execution,
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except asyncio.CancelledError:
            if source_path is None:
                await state.reset_inline()
            raise
        except ModelRetry:
            raise
        except ValidationError as exc:
            return self._raise_or_fail(
                exc,
                message=_format_validation_error(exc),
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except (MontyCrashedError, MontyConversionError) as exc:
            if source_path is None:
                await state.reset_inline()
            return self._raise_or_fail(
                exc,
                message="Sandbox worker failed and the inline session was reset",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except MontySyntaxError as exc:
            return self._raise_or_fail(
                exc,
                message=f"Syntax error in code:\n{exc.display()}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except MontyTypingError as exc:
            return self._raise_or_fail(
                exc,
                message=f"Type error in code:\n{exc.display()}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except MontyRuntimeError as exc:
            return self._raise_or_fail(
                exc,
                message=f"Runtime error:\n{exc.display()}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except TimeoutError as exc:
            if source_path is None:
                await state.reset_inline()
            return self._raise_or_fail(
                exc,
                message=f"CodeAct execution exceeded timeout_seconds={self.config.timeout_seconds}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except Exception as exc:
            return self._raise_or_fail(
                exc,
                message=f"CodeAct execution failed: {type(exc).__name__}: {exc}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        except BaseException as exc:
            if not is_sandbox_panic(exc):
                if source_path is None:
                    await state.reset_inline()
                raise
            if source_path is None:
                await state.reset_inline()
            return self._raise_or_fail(
                exc,
                message="Code aborted inside the sandbox and the inline session was reset",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )

    def _success_result(
        self,
        execution: CodeActExecution,
        *,
        source_sha256: str,
        source_path: str | None,
        catalog: _Catalog,
        budget: _ExecutionBudget,
    ) -> ToolReturn:
        result = execution.completed.output
        if result is not None:
            # Reconstruct Pydantic AI multimodal values serialized for Monty's
            # plain-data boundary (for example BinaryContent screenshots).
            result = _TOOL_RETURN_CONTENT.validate_python(result)
        printed = execution.printed
        if not printed:
            value: Any = result if result is not None else {}
        elif result is None:
            value = {"output": printed}
        elif _contains_multimodal(result):
            value = [printed, *result] if isinstance(result, list) else [printed, result]
        else:
            value = {"output": printed, "result": result}

        try:
            _bounded_json_size(value, self.config.max_output_bytes)
        except _JsonSizeLimitExceeded:
            self._raise_failure(
                f"CodeAct returned output exceeds max_output_bytes={self.config.max_output_bytes}",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            )
        supplemental_content = budget.ordered_supplemental_content()
        return ToolReturn(
            return_value=value,
            content=supplemental_content or None,
            metadata=_metadata(
                status="completed",
                source_sha256=source_sha256,
                source_path=source_path,
                catalog=catalog,
                budget=budget,
            ),
        )

    def _raise_or_fail(
        self,
        exc: BaseException,
        *,
        message: str,
        source_sha256: str,
        source_path: str | None,
        catalog: _Catalog,
        budget: _ExecutionBudget,
    ) -> Any:
        message = _preview_text(_redact_text(message), budget.config.trace_preview_bytes)
        if budget.calls_started == 0:
            raise ModelRetry(message) from exc
        self._raise_failure(
            message,
            source_sha256=source_sha256,
            source_path=source_path,
            catalog=catalog,
            budget=budget,
        )

    @staticmethod
    def _raise_failure(
        message: str,
        *,
        source_sha256: str,
        source_path: str | None,
        catalog: _Catalog,
        budget: _ExecutionBudget,
    ) -> None:
        message = _redact_text(message)
        payload = _metadata(
            status="failed",
            source_sha256=source_sha256,
            source_path=source_path,
            catalog=catalog,
            budget=budget,
        )
        payload["error"] = _preview_text(message, budget.config.trace_preview_bytes)
        raise ToolFailed(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


async def _unwrap_tool_return(
    result: Any,
    *,
    ordinal: int,
    budget: _ExecutionBudget,
    record: _ToolCallRecord,
) -> Any:
    if not isinstance(result, ToolReturn):
        return result
    if result.content is not None:
        if isinstance(result.content, str):
            original_content: list[UserContent] | tuple[UserContent, ...] = [result.content]
        elif isinstance(result.content, (list, tuple)):
            original_content = result.content
        else:
            raise TypeError(f"Unsupported CodeAct ToolReturn.content sequence: {type(result.content).__name__}")
        try:
            content_size = _bounded_json_size(
                original_content,
                budget.config.max_output_bytes,
            )
        except _JsonSizeLimitExceeded as exc:
            raise RuntimeError(
                f"CodeAct nested tool supplemental content exceeds max_output_bytes={budget.config.max_output_bytes}"
            ) from exc
        supplemental = list(original_content)
        await budget.reserve_supplemental_content(
            ordinal,
            supplemental,
            content_size,
        )
    if result.metadata is not None:
        record["tool_return_metadata_omitted"] = True
    return result.return_value


async def _execute_validated_call(
    manager: ToolManager[AgentContext],
    validated: Any,
    call: ToolCallPart,
) -> Any:
    """Execute a validated nested call and resolve public deferred-result variants."""

    try:
        return await manager.execute_tool_call(validated, wrap_validation_errors=False)
    except (CallDeferred, ApprovalRequired) as exc:
        requests = DeferredToolRequests(
            approvals=[call] if isinstance(exc, ApprovalRequired) else [],
            calls=[call] if isinstance(exc, CallDeferred) else [],
            metadata={call.tool_call_id: exc.metadata} if exc.metadata else {},
        )
        deferred = await manager.resolve_deferred_tool_calls(requests)
        if deferred is None:
            raise
        resolved = deferred.to_tool_call_results().get(call.tool_call_id)
        if resolved is None:
            raise
        if isinstance(resolved, ToolDenied):
            return resolved
        if isinstance(resolved, ToolFailed):
            raise resolved from exc
        if isinstance(resolved, ToolApproved):
            approved_call = call
            if resolved.override_args is not None:
                approved_call = replace(call, args=resolved.override_args)
            approved = await manager.validate_tool_call(
                approved_call,
                approved=True,
                metadata=deferred.metadata.get(call.tool_call_id),
                wrap_validation_errors=False,
            )
            return await manager.execute_tool_call(approved, wrap_validation_errors=False)
        if isinstance(resolved, ModelRetry):
            retry = RetryPromptPart(
                content=resolved.message,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
            )
            raise ToolRetryError(retry) from exc
        if isinstance(resolved, RetryPromptPart):
            retry = replace(
                resolved,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
            )
            raise ToolRetryError(retry) from exc
        return deferred.calls[call.tool_call_id]


def _build_catalog(  # noqa: C901
    wrapped_tools: dict[str, ToolsetTool[AgentContext]],
) -> _Catalog:
    definitions: dict[str, ToolDefinition] = {}
    sanitized_to_original: dict[str, str] = {}
    eligible_tools: dict[str, ToolsetTool[AgentContext]] = {}
    for original_name, tool in wrapped_tools.items():
        definition = tool.tool_def
        metadata = definition.metadata or {}
        if metadata.get("codeact") is not True:
            continue
        if definition.tool_kind is not None or definition.defer_loading or definition.unless_native:
            continue
        if "code_arg_name" in metadata or metadata.get("codeact_program_runner") is True:
            continue
        _validate_local_schema_refs(
            definition.parameters_json_schema,
            tool_name=original_name,
            schema_name="parameter",
        )
        if definition.return_schema is not None:
            _validate_local_schema_refs(
                definition.return_schema,
                tool_name=original_name,
                schema_name="return",
            )
        properties = definition.parameters_json_schema.get("properties", {})
        if not isinstance(properties, dict):
            raise UserError(f"CodeAct tool {original_name!r} has an invalid object parameter schema")
        invalid_parameters = [
            str(name)
            for name in properties
            if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name)
        ]
        if invalid_parameters:
            rendered = ", ".join(repr(name) for name in invalid_parameters)
            raise UserError(
                f"CodeAct tool {original_name!r} has parameter names that are not valid Python identifiers: {rendered}"
            )
        safe_name = _sanitize_tool_name(original_name)
        if safe_name in _RESERVED_TOOL_NAMES:
            raise UserError(f"Tool {original_name!r} sanitizes to reserved CodeAct name {safe_name!r}")
        if safe_name in definitions:
            existing = sanitized_to_original.get(safe_name, safe_name)
            raise UserError(
                f"CodeAct tool-name collision: {existing!r} and {original_name!r} both map to {safe_name!r}"
            )
        if safe_name != original_name:
            sanitized_to_original[safe_name] = original_name
            definition = replace(definition, name=safe_name)
        definitions[safe_name] = definition
        eligible_tools[original_name] = tool

    fingerprint_payload = [
        {
            "contract_version": _CODEACT_CONTRACT_VERSION,
            "sandbox_name": name,
            "canonical_name": sanitized_to_original.get(name, name),
            "parameters_schema": definition.parameters_json_schema,
            "return_schema": definition.return_schema,
            "sequential": definition.sequential,
        }
        for name, definition in sorted(definitions.items())
    ]
    fingerprint = hashlib.sha256(_json_bytes(fingerprint_payload)).hexdigest()
    return _Catalog(
        definitions=definitions,
        sanitized_to_original=sanitized_to_original,
        wrapped_tools=eligible_tools,
        fingerprint=fingerprint,
    )


def _validate_local_schema_refs(schema: Any, *, tool_name: str, schema_name: str) -> None:
    """Fail closed when a tool schema contains an invalid or dangling JSON Pointer."""

    if not isinstance(schema, (dict, list)):
        raise UserError(f"CodeAct tool {tool_name!r} has an invalid {schema_name} schema")
    pending = [schema]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
            continue
        if not isinstance(item, dict):
            continue
        reference = item.get("$ref")
        if reference is not None and (
            not isinstance(reference, str) or not _local_json_pointer_exists(schema, reference)
        ):
            raise UserError(
                f"CodeAct tool {tool_name!r} has an invalid or dangling local $ref "
                f"in its {schema_name} schema: {reference!r}"
            )
        pending.extend(item.values())


def _local_json_pointer_exists(root: dict[Any, Any] | list[Any], reference: str) -> bool:
    if reference == "#":
        return True
    if not reference.startswith("#/"):
        return False
    current: Any = root
    for encoded_token in reference[2:].split("/"):
        if re.search(r"~(?:[^01]|$)", encoded_token):
            return False
        token = encoded_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                return False
            current = current[token]
        elif isinstance(current, list):
            if re.fullmatch(r"0|[1-9][0-9]*", token) is None:
                return False
            index = int(token)
            if index >= len(current):
                return False
            current = current[index]
        else:
            return False
    return True


def _sanitize_tool_name(name: str) -> str:
    safe = _INVALID_IDENT_CHARS.sub("_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    if keyword.iskeyword(safe):
        safe += "_"
    return safe or "_"


def _render_catalog(definitions: dict[str, ToolDefinition]) -> str:
    if not definitions:
        return ""
    signatures = [definition.function_signature for definition in definitions.values()]
    if any(signature is None for signature in signatures):
        missing = [name for name, definition in definitions.items() if definition.function_signature is None]
        raise UserError(f"CodeAct tools lack function signatures: {', '.join(missing)}")
    typed_signatures = cast(list[FunctionSignature], signatures)
    conflicts = FunctionSignature.get_conflicting_type_names(typed_signatures)
    types = FunctionSignature.render_type_definitions(typed_signatures, conflicts)
    functions = [
        definition.render_signature(
            "...",
            is_async=not definition.sequential,
            conflicting_type_names=conflicts,
        )
        for definition in definitions.values()
    ]
    rendered_python = "\n\n".join([*types, *functions])
    try:
        ast.parse(rendered_python, mode="exec")
    except SyntaxError as exc:
        raise UserError(
            f"CodeAct tool schemas cannot be rendered as valid Python: {exc.msg} at line {exc.lineno}"
        ) from exc

    blocks = ["The following host functions are available inside CodeAct. All parameters are keyword-only."]
    if types:
        blocks.append("```python\n" + "\n\n".join(types) + "\n```")
    blocks.append("```python\n" + "\n\n".join(functions) + "\n```")
    return "\n\n".join(blocks)


def _metadata(
    *,
    status: str,
    source_sha256: str,
    source_path: str | None,
    catalog: _Catalog,
    budget: _ExecutionBudget,
) -> dict[str, Any]:
    return {
        "codeact": {
            "contract_version": _CODEACT_CONTRACT_VERSION,
            "status": status,
            "source_sha256": source_sha256,
            "source_path": source_path,
            "catalog_fingerprint": catalog.fingerprint,
            "tool_calls": budget.records,
            "tool_call_count": budget.calls_started,
            "nested_result_bytes": budget.nested_result_bytes,
        }
    }


def _trace_value_descriptor(value: bytes, preview_bytes: int) -> _TraceValueDescriptor:
    preview, truncated = truncate_utf8_head_tail(value.decode("utf-8", errors="replace"), preview_bytes)
    return {
        "bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
        "preview": preview,
        "truncated": truncated,
    }


def _set_result_descriptor(record: _ToolCallRecord, value: bytes, preview_bytes: int) -> None:
    descriptor = _trace_value_descriptor(value, preview_bytes)
    record["result_bytes"] = descriptor["bytes"]
    record["result_sha256"] = descriptor["sha256"]
    record["result_preview"] = descriptor["preview"]
    record["result_truncated"] = descriptor["truncated"]


def _preview_text(value: str, max_bytes: int) -> str:
    return truncate_utf8_head_tail(value, max_bytes)[0]


class _JsonSizeLimitExceeded(ValueError):
    pass


@dataclass(frozen=True)
class _KnownJsonStringSize:
    encoded_bytes: int


class _BoundedJsonSizer:
    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.seen: set[int] = set()

    def add(self, total: int, increment: int) -> int:
        total += increment
        if total > self.max_bytes:
            raise _JsonSizeLimitExceeded
        return total

    def string_size(self, text: str) -> int:
        total = self.add(0, 2)  # Quotes.
        for char in text:
            codepoint = ord(char)
            if char in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
                total = self.add(total, 2)
            elif codepoint < 0x20 or 0xD800 <= codepoint <= 0xDFFF:
                total = self.add(total, 6)
            else:
                total = self.add(total, len(char.encode("utf-8")))
        return total

    @staticmethod
    def key_text(key: Any) -> str:
        if isinstance(key, str):
            return key
        if key is None:
            return "null"
        if key is True:
            return "true"
        if key is False:
            return "false"
        if isinstance(key, (int, float)):
            return json.dumps(key, ensure_ascii=False)
        raise TypeError(f"keys must be str, int, float, bool or None, not {type(key).__name__}")

    def measure_mapping(self, item: dict[Any, Any]) -> int:
        identity = id(item)
        if identity in self.seen:
            raise ValueError("Circular reference detected")
        self.seen.add(identity)
        try:
            total = self.add(0, 2)
            for index, (key, nested) in enumerate(item.items()):
                if index:
                    total = self.add(total, 1)
                total = self.add(total, self.string_size(self.key_text(key)))
                total = self.add(total, 1)
                total = self.add(total, self.measure(nested))
            return total
        finally:
            self.seen.remove(identity)

    def measure_sequence(self, item: list[Any] | tuple[Any, ...]) -> int:
        identity = id(item)
        if identity in self.seen:
            raise ValueError("Circular reference detected")
        self.seen.add(identity)
        try:
            total = self.add(0, 2)
            for index, nested in enumerate(item):
                if index:
                    total = self.add(total, 1)
                total = self.add(total, self.measure(nested))
            return total
        finally:
            self.seen.remove(identity)

    def measure_binary_content(self, item: BinaryContent) -> int:
        # Pydantic serializes BinaryContent.data as base64 in JSON mode. This
        # computes that expansion without allocating the encoded payload.
        data_size = 4 * ((len(item.data) + 2) // 3)
        return self.measure_mapping({
            "data": _KnownJsonStringSize(data_size),
            "media_type": item.media_type,
            "vendor_metadata": item.vendor_metadata,
            "kind": item.kind,
            "identifier": item.identifier,
        })

    def measure_dataclass(self, item: Any) -> int:
        identity = id(item)
        if identity in self.seen:
            raise ValueError("Circular reference detected")
        self.seen.add(identity)
        try:
            dataclass_fields = fields(item)
            total = self.add(0, 2)
            for index, dataclass_field in enumerate(dataclass_fields):
                if index:
                    total = self.add(total, 1)
                total = self.add(total, self.string_size(dataclass_field.name))
                total = self.add(total, 1)
                total = self.add(total, self.measure(getattr(item, dataclass_field.name)))
            return total
        finally:
            self.seen.remove(identity)

    def measure(self, item: Any) -> int:
        if isinstance(item, _KnownJsonStringSize):
            return self.add(self.add(0, 2), item.encoded_bytes)
        if isinstance(item, str):
            return self.string_size(item)
        if isinstance(item, bytes):
            raise TypeError("Raw bytes are not supported CodeAct JSON values; use BinaryContent")
        if isinstance(item, BinaryContent):
            return self.measure_binary_content(item)
        if isinstance(item, dict):
            return self.measure_mapping(item)
        if isinstance(item, (list, tuple)):
            return self.measure_sequence(item)
        if item is None or isinstance(item, (bool, int, float)):
            encoded = json.dumps(item, ensure_ascii=False).encode("utf-8")
            return self.add(0, len(encoded))
        if is_dataclass(item) and not isinstance(item, type):
            return self.measure_dataclass(item)
        raise TypeError(f"Unsupported CodeAct JSON value: {type(item).__name__}")


def _bounded_json_size(value: Any, max_bytes: int) -> int:
    """Measure compact UTF-8 JSON without allocating an unbounded encoded copy."""

    return _BoundedJsonSizer(max_bytes).measure(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Nested tool argument validation failed:"]
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "arguments"
        message = _redact_text(str(error.get("msg", "Invalid value")))
        error_type = str(error.get("type", "validation_error"))
        lines.append(f"- {location}: {message} [{error_type}]")
    return "\n".join(lines)


def _redacted_json_bytes(value: Any) -> bytes:
    text = _json_bytes(_redact(value)).decode("utf-8")
    return _redact_text(text).encode("utf-8")


def _redact_text(value: str) -> str:
    value = _SECRET_HEADER.sub(r"\1[REDACTED]", value)
    value = _SENSITIVE_ASSIGNMENT.sub(r'\1"[REDACTED]"', value)
    value = _BEARER_TOKEN.sub("Bearer [REDACTED]", value)
    return _URL_CREDENTIALS.sub(r"\1[REDACTED]@", value)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


def _contains_multimodal(value: Any) -> bool:
    if is_multi_modal_content(value):
        return True
    return isinstance(value, list) and any(is_multi_modal_content(item) for item in value)
