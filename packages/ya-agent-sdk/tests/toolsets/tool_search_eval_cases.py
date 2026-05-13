"""Deterministic tool-search retrieval evaluation cases."""

from __future__ import annotations

from dataclasses import dataclass, field

from ya_agent_sdk.toolsets.tool_search.metadata import ToolMetadata


@dataclass(frozen=True)
class ToolSearchEvalCase:
    """A fixed query and expected search result set for tool retrieval evaluation."""

    id: str
    query: str
    expected: frozenset[str]
    category: str
    max_results: int = 3
    acceptable: frozenset[str] = field(default_factory=frozenset)
    forbidden: frozenset[str] = field(default_factory=frozenset)
    notes: str = ""


def build_eval_catalog() -> list[ToolMetadata]:
    """Build a representative tool metadata catalog for retrieval evaluation."""
    return [
        ToolMetadata(
            name="view",
            description="Read contents of a file from the filesystem",
            parameter_names=["file_path", "limit", "offset"],
            parameter_descriptions={
                "file_path": "Path to the file to read",
                "limit": "Maximum number of lines to read",
                "offset": "Line number to start reading from",
            },
            namespace="filesystem",
        ),
        ToolMetadata(
            name="edit",
            description="Edit a file by replacing exact text",
            parameter_names=["file_path", "old_string", "new_string", "replace_all"],
            parameter_descriptions={
                "file_path": "Path to the file",
                "old_string": "Text to replace",
                "new_string": "Replacement text",
                "replace_all": "Replace all occurrences",
            },
            namespace="filesystem",
        ),
        ToolMetadata(
            name="write",
            description="Write or overwrite file content",
            parameter_names=["file_path", "content", "mode"],
            parameter_descriptions={"content": "Content to write", "mode": "Write or append mode"},
            namespace="filesystem",
        ),
        ToolMetadata(
            name="grep",
            description="Search file contents using regex patterns with context lines",
            parameter_names=["pattern", "include", "context_lines"],
            parameter_descriptions={
                "pattern": "Regular expression pattern to search for",
                "include": "Glob pattern to filter files",
                "context_lines": "Context lines before and after matches",
            },
            namespace="filesystem",
        ),
        ToolMetadata(
            name="glob",
            description="Find files by glob pattern",
            parameter_names=["pattern", "max_results", "include_ignored"],
            parameter_descriptions={"pattern": "Glob pattern to match files"},
            namespace="filesystem",
        ),
        ToolMetadata(
            name="shell_exec",
            description="Execute a shell command and return output",
            parameter_names=["command", "cwd", "timeout_seconds", "background"],
            parameter_descriptions={
                "command": "The shell command to execute",
                "cwd": "Working directory",
                "background": "Run command in background",
            },
            namespace="shell",
        ),
        ToolMetadata(
            name="shell_monitor",
            description="Start a background shell process with output monitoring",
            parameter_names=["command", "cwd", "environment"],
            parameter_descriptions={"command": "The shell command to execute"},
            namespace="shell",
        ),
        ToolMetadata(
            name="shell_kill",
            description="Terminate a running background shell process",
            parameter_names=["process_id"],
            parameter_descriptions={"process_id": "Process ID of the background process"},
            namespace="shell",
        ),
        ToolMetadata(
            name="shell_input",
            description="Write text to a background process standard input",
            parameter_names=["process_id", "text", "close_stdin"],
            parameter_descriptions={"text": "Text to write to stdin"},
            namespace="shell",
        ),
        ToolMetadata(
            name="search",
            description="Search the web for information using search APIs",
            parameter_names=["query", "num"],
            parameter_descriptions={"query": "The web search query"},
            namespace="web",
        ),
        ToolMetadata(
            name="fetch",
            description="Read web files or check HTTP resource availability",
            parameter_names=["url", "head_only"],
            parameter_descriptions={"head_only": "Only check existence without downloading content"},
            namespace="web",
        ),
        ToolMetadata(
            name="scrape",
            description="Convert websites to Markdown format for content analysis",
            parameter_names=["url"],
            parameter_descriptions={"url": "The web page URL to scrape"},
            namespace="web",
        ),
        ToolMetadata(
            name="download",
            description="Download files from URLs and save to local filesystem",
            parameter_names=["urls", "save_dir"],
            parameter_descriptions={"urls": "List of URLs to download"},
            namespace="web",
        ),
        ToolMetadata(
            name="send_message",
            description="Send an instant message to a chat or user",
            parameter_names=["chat_id", "content"],
            parameter_descriptions={"chat_id": "Target chat identifier", "content": "Message content"},
            namespace="im",
        ),
        ToolMetadata(
            name="search_messages",
            description="Search chat messages and conversation history",
            parameter_names=["query", "chat_id"],
            parameter_descriptions={"query": "Message search query"},
            namespace="im",
        ),
        ToolMetadata(
            name="create_calendar_event",
            description="Create a calendar event and invite attendees",
            parameter_names=["title", "start_time", "end_time", "attendees"],
            parameter_descriptions={
                "title": "Event title",
                "start_time": "Event start time",
                "attendees": "Email addresses of event attendees",
            },
            namespace="calendar",
        ),
        ToolMetadata(
            name="query_freebusy",
            description="Query calendar free busy status and availability",
            parameter_names=["user_ids", "time_min", "time_max"],
            parameter_descriptions={"user_ids": "Users to query for busy time"},
            namespace="calendar",
        ),
        ToolMetadata(
            name="send_email",
            description="Send an email message to a recipient",
            parameter_names=["recipient", "subject", "body"],
            parameter_descriptions={"recipient": "Email address", "subject": "Email subject line"},
            namespace="mail",
        ),
        ToolMetadata(
            name="reply_email",
            description="Reply to an existing email thread",
            parameter_names=["thread_id", "body"],
            parameter_descriptions={"thread_id": "Email conversation thread identifier"},
            namespace="mail",
        ),
        ToolMetadata(
            name="get_weather",
            description="Get the current weather in a given location",
            parameter_names=["location", "unit"],
            parameter_descriptions={"location": "The city and state", "unit": "Temperature unit"},
            namespace="weather",
        ),
        ToolMetadata(
            name="get_forecast",
            description="Get the weather forecast for multiple days ahead",
            parameter_names=["location", "days"],
            parameter_descriptions={"days": "Number of days to forecast"},
            namespace="weather",
        ),
        ToolMetadata(
            name="get_stock_price",
            description="Get the current stock price for a ticker symbol",
            parameter_names=["ticker"],
            parameter_descriptions={"ticker": "Stock ticker symbol like AAPL"},
            namespace="finance",
        ),
        ToolMetadata(
            name="convert_currency",
            description="Convert an amount from one currency to another using exchange rates",
            parameter_names=["amount", "from_currency", "to_currency"],
            parameter_descriptions={"from_currency": "Source currency code", "to_currency": "Target currency code"},
            namespace="finance",
        ),
        ToolMetadata(
            name="filesystem",
            description="Filesystem tools for reading, writing, editing, and finding local files",
            namespace="filesystem",
            is_namespace_entry=True,
            namespace_tool_names=["view", "edit", "write", "grep", "glob"],
        ),
        ToolMetadata(
            name="shell",
            description="Shell command execution and background process management tools",
            namespace="shell",
            is_namespace_entry=True,
            namespace_tool_names=["shell_exec", "shell_monitor", "shell_kill", "shell_input"],
        ),
        ToolMetadata(
            name="web",
            description="Web search, HTTP fetch, website scraping, and download tools",
            namespace="web",
            is_namespace_entry=True,
            namespace_tool_names=["search", "fetch", "scrape", "download"],
        ),
        ToolMetadata(
            name="im",
            description="Instant messaging tools for sending and searching chat messages",
            namespace="im",
            is_namespace_entry=True,
            namespace_tool_names=["send_message", "search_messages"],
        ),
        ToolMetadata(
            name="calendar",
            description="Calendar scheduling, meeting, and availability tools",
            namespace="calendar",
            is_namespace_entry=True,
            namespace_tool_names=["create_calendar_event", "query_freebusy"],
        ),
        ToolMetadata(
            name="mail",
            description="Email tools for sending messages and replying to threads",
            namespace="mail",
            is_namespace_entry=True,
            namespace_tool_names=["send_email", "reply_email"],
        ),
        ToolMetadata(
            name="weather",
            description="Weather tools for current conditions and forecasts",
            namespace="weather",
            is_namespace_entry=True,
            namespace_tool_names=["get_weather", "get_forecast"],
        ),
        ToolMetadata(
            name="finance",
            description="Finance tools for stock prices and currency conversion",
            namespace="finance",
            is_namespace_entry=True,
            namespace_tool_names=["get_stock_price", "convert_currency"],
        ),
    ]


EVAL_CASES: tuple[ToolSearchEvalCase, ...] = (
    ToolSearchEvalCase(
        id="exact_shell_exec", query="shell_exec", expected=frozenset({"shell_exec"}), category="exact_name"
    ),
    ToolSearchEvalCase(id="exact_grep", query="grep", expected=frozenset({"grep"}), category="exact_name"),
    ToolSearchEvalCase(
        id="snake_case_stock_price",
        query="stock price",
        expected=frozenset({"get_stock_price"}),
        acceptable=frozenset({"finance"}),
        category="snake_case",
    ),
    ToolSearchEvalCase(
        id="snake_case_old_new_string",
        query="old string new string",
        expected=frozenset({"edit"}),
        category="parameter",
    ),
    ToolSearchEvalCase(
        id="file_regex_search",
        query="search file contents regex",
        expected=frozenset({"grep"}),
        forbidden=frozenset({"search"}),
        category="multi_term",
    ),
    ToolSearchEvalCase(
        id="background_process_monitor",
        query="start background process monitor output",
        expected=frozenset({"shell_monitor"}),
        acceptable=frozenset({"shell"}),
        category="multi_term",
    ),
    ToolSearchEvalCase(
        id="download_file_url",
        query="download file from url",
        expected=frozenset({"download"}),
        category="multi_term",
    ),
    ToolSearchEvalCase(
        id="calendar_attendees",
        query="attendees start time title",
        expected=frozenset({"create_calendar_event"}),
        acceptable=frozenset({"calendar"}),
        category="parameter",
    ),
    ToolSearchEvalCase(
        id="email_subject_body",
        query="recipient subject body",
        expected=frozenset({"send_email"}),
        acceptable=frozenset({"mail"}),
        category="parameter",
    ),
    ToolSearchEvalCase(
        id="filesystem_namespace",
        query="read write edit files",
        expected=frozenset({"filesystem", "view", "write", "edit"}),
        category="namespace",
    ),
    ToolSearchEvalCase(
        id="calendar_namespace",
        query="calendar schedule meeting",
        expected=frozenset({"calendar", "create_calendar_event"}),
        category="namespace",
    ),
    ToolSearchEvalCase(
        id="mail_namespace",
        query="email inbox reply forward",
        expected=frozenset({"mail", "reply_email", "send_email"}),
        category="namespace",
    ),
    ToolSearchEvalCase(
        id="website_to_markdown",
        query="turn website into markdown",
        expected=frozenset({"scrape"}),
        acceptable=frozenset({"web"}),
        category="intent",
    ),
    ToolSearchEvalCase(
        id="check_url_exists",
        query="check http resource exists",
        expected=frozenset({"fetch"}),
        acceptable=frozenset({"web"}),
        category="intent",
    ),
    ToolSearchEvalCase(
        id="send_chat_message",
        query="send chat message to someone",
        expected=frozenset({"send_message"}),
        acceptable=frozenset({"im"}),
        category="intent",
    ),
    ToolSearchEvalCase(
        id="search_chat_history",
        query="search chat history",
        expected=frozenset({"search_messages"}),
        acceptable=frozenset({"im"}),
        forbidden=frozenset({"search", "grep"}),
        category="ambiguous",
    ),
    ToolSearchEvalCase(
        id="web_search_docs",
        query="search the web for docs",
        expected=frozenset({"search"}),
        acceptable=frozenset({"web"}),
        forbidden=frozenset({"grep", "search_messages"}),
        category="ambiguous",
    ),
    ToolSearchEvalCase(
        id="local_file_search",
        query="search local file contents",
        expected=frozenset({"grep"}),
        acceptable=frozenset({"filesystem"}),
        forbidden=frozenset({"search"}),
        category="ambiguous",
    ),
    ToolSearchEvalCase(id="empty_query", query="", expected=frozenset(), category="negative"),
    ToolSearchEvalCase(
        id="unknown_capability",
        query="quantum banana allocator",
        expected=frozenset(),
        category="negative",
    ),
)
