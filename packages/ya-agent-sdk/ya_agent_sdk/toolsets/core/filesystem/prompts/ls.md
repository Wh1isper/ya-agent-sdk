<ls-tool>
List directory contents with file info (name, type, size, modified time).
Results are limited to 500 entries by default.

<parameters>
- `path`: directory path to list.
- `ignore`: glob patterns to exclude by entry name before type/stat work.
- `max_results`: maximum entry count; use `-1` for unlimited when the directory is known to be small enough.
</parameters>

<best-practices>
- Use ignore parameter to filter out unwanted entries (logs, cache, node_modules)
- For recursive file search: use glob instead
- For content search: use grep instead
- Very large responses are saved to a temp file with `output_file_path`; use view to read it.
</best-practices>
</ls-tool>
